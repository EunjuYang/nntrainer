// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd.
 *
 * @file   unittest_mha_core.cpp
 * @date   09 March 2026
 * @brief  Unit tests for MHACore layer with FP32-only mode
 * @see    https://github.com/nntrainer/nntrainer
 * @bug    No known bugs except for NYI items
 */
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include <layer_context.h>
#include <layer_devel.h>
#include <mha_core.h>
#include <tensor.h>
#include <tensor_dim.h>
#include <var_grad.h>
#include <weight.h>

namespace {

/**
 * @brief Naive FP32 reference implementation for Q*K^T computation
 * Computes attention scores: output[row][head] = dot(Q[head], K[row][head]) /
 * sqrt(head_dim)
 *
 * @param query (num_heads_Q * head_dim) flat array
 * @param key_cache (num_rows, num_heads_KV * head_dim) flat array
 * @param output (num_rows, num_heads_Q) flat array
 * @param num_rows number of cached rows to attend to
 * @param num_heads_Q number of query heads
 * @param num_heads_KV number of key/value heads
 * @param head_dim dimension per head
 */
void reference_qk_dot(const float *query, const float *key_cache,
                       float *output, int num_rows, int num_heads_Q,
                       int num_heads_KV, int head_dim) {
  int gqa_size = num_heads_Q / num_heads_KV;
  for (int row = 0; row < num_rows; ++row) {
    for (int n = 0; n < num_heads_KV; ++n) {
      for (int g = 0; g < gqa_size; ++g) {
        int q_head = n * gqa_size + g;
        const float *q_ptr = query + q_head * head_dim;
        const float *k_ptr = key_cache + (row * num_heads_KV + n) * head_dim;
        float sum = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
          sum += q_ptr[d] * k_ptr[d];
        }
        output[row * num_heads_Q + q_head] = sum / std::sqrt((float)head_dim);
      }
    }
  }
}

/**
 * @brief Naive softmax over each row of attention scores
 *
 * @param data (num_rows, num_heads) flat array - modified in-place
 * @param num_rows number of rows
 * @param num_heads number of heads (width of each row)
 */
void reference_softmax_rows(float *data, int num_rows, int num_heads) {
  // Each "head" has its own softmax across the rows
  // But actually in attention, softmax is applied per head across the key
  // dimension In MHA: for each head, softmax is over the key/time dimension
  // Here: data[row][head] represents score for (head, key_position=row)
  // Softmax should be over key positions (rows) for each head
  for (int h = 0; h < num_heads; ++h) {
    float max_val = -1e30f;
    for (int r = 0; r < num_rows; ++r) {
      max_val = std::max(max_val, data[r * num_heads + h]);
    }
    float sum = 0.0f;
    for (int r = 0; r < num_rows; ++r) {
      data[r * num_heads + h] = std::exp(data[r * num_heads + h] - max_val);
      sum += data[r * num_heads + h];
    }
    for (int r = 0; r < num_rows; ++r) {
      data[r * num_heads + h] /= sum;
    }
  }
}

/**
 * @brief Naive FP32 reference for attention_weights * V_cache
 *
 * @param attn_weights (num_rows, num_heads_Q) flat array
 * @param vcache (num_rows, num_heads_KV * head_dim) flat array
 * @param output (num_heads_Q * head_dim) flat array
 * @param num_rows number of cached rows
 * @param num_heads_Q number of query heads
 * @param num_heads_KV number of value heads
 * @param head_dim dimension per head
 */
void reference_attn_v(const float *attn_weights, const float *vcache,
                       float *output, int num_rows, int num_heads_Q,
                       int num_heads_KV, int head_dim) {
  int gqa_size = num_heads_Q / num_heads_KV;
  // Zero output
  std::memset(output, 0, num_heads_Q * head_dim * sizeof(float));

  for (int n = 0; n < num_heads_KV; ++n) {
    for (int g = 0; g < gqa_size; ++g) {
      int q_head = n * gqa_size + g;
      float *out_ptr = output + q_head * head_dim;
      for (int row = 0; row < num_rows; ++row) {
        float w = attn_weights[row * num_heads_Q + q_head];
        const float *v_ptr = vcache + (row * num_heads_KV + n) * head_dim;
        for (int d = 0; d < head_dim; ++d) {
          out_ptr[d] += w * v_ptr[d];
        }
      }
    }
  }
}

} // anonymous namespace

/**
 * @brief Test MHACoreLayer creation and property setting
 */
TEST(MHACoreLayer, create_and_set_properties_p) {
  auto layer = std::make_unique<causallm::MHACoreLayer>();
  EXPECT_EQ(layer->getType(), "mha_core");

  EXPECT_NO_THROW(layer->setProperty(
    {"num_heads=4", "num_heads_KV=2", "projected_key_dim=8",
     "projected_value_dim=8", "max_timestep=32", "force_fp32=true"}));
}

/**
 * @brief Test that force_fp32 property can be set to false (default)
 */
TEST(MHACoreLayer, force_fp32_default_false_p) {
  auto layer = std::make_unique<causallm::MHACoreLayer>();

  EXPECT_NO_THROW(layer->setProperty(
    {"num_heads=2", "num_heads_KV=2", "projected_key_dim=4",
     "projected_value_dim=4", "max_timestep=16", "force_fp32=false"}));
}

/**
 * @brief Test MHACoreLayer finalize with force_fp32=true
 *        Verifies that the layer can be finalized with FP32 KV cache
 */
TEST(MHACoreLayer, finalize_with_force_fp32_p) {
  auto layer = std::make_unique<causallm::MHACoreLayer>();

  unsigned int num_heads_Q = 4;
  unsigned int num_heads_KV = 2;
  unsigned int head_dim = 8;
  unsigned int max_timestep = 32;
  unsigned int batch_size = 1;
  unsigned int seq_len = 1;

  layer->setProperty(
    {"num_heads=" + std::to_string(num_heads_Q),
     "num_heads_KV=" + std::to_string(num_heads_KV),
     "projected_key_dim=" + std::to_string(head_dim),
     "projected_value_dim=" + std::to_string(head_dim),
     "max_timestep=" + std::to_string(max_timestep), "force_fp32=true",
     "is_causal=true"});

  ml::train::TensorDim query_dim(
    {batch_size, 1, seq_len, num_heads_Q * head_dim});
  ml::train::TensorDim key_dim(
    {batch_size, 1, seq_len, num_heads_KV * head_dim});
  ml::train::TensorDim value_dim(
    {batch_size, 1, seq_len, num_heads_KV * head_dim});

  std::vector<ml::train::TensorDim> input_dims = {query_dim, key_dim,
                                                   value_dim};

  nntrainer::InitLayerContext init_context(input_dims, {true}, false,
                                           "mha_core_test");
  EXPECT_NO_THROW(layer->finalize(init_context));

  // Verify output dimensions
  auto out_specs = init_context.getOutSpecs();
  ASSERT_EQ(out_specs.size(), 1u);
  EXPECT_EQ(out_specs[0].variable_spec.dim.width(), num_heads_Q * head_dim);

  // Verify tensor specs (should have cache_key and cache_value as FP32)
  auto tensor_specs = init_context.getTensorsSpec();
  ASSERT_GE(tensor_specs.size(), 2u);

  // cache_key and cache_value should be FP32 when force_fp32 is true
  for (size_t i = 0; i < 2; ++i) {
    auto &dim = std::get<0>(tensor_specs[i]);
    EXPECT_EQ(dim.getDataType(), ml::train::TensorDim::DataType::FP32)
      << "KV cache tensor " << i << " should be FP32 when force_fp32=true";
  }
}

/**
 * @brief Test MHACoreLayer finalize with force_fp32=false (default)
 *        Verifies that the layer uses UINT16 (non-ENABLE_FP16) for KV cache
 */
TEST(MHACoreLayer, finalize_without_force_fp32_p) {
  auto layer = std::make_unique<causallm::MHACoreLayer>();

  unsigned int num_heads_Q = 4;
  unsigned int num_heads_KV = 2;
  unsigned int head_dim = 8;
  unsigned int max_timestep = 32;
  unsigned int batch_size = 1;
  unsigned int seq_len = 1;

  layer->setProperty(
    {"num_heads=" + std::to_string(num_heads_Q),
     "num_heads_KV=" + std::to_string(num_heads_KV),
     "projected_key_dim=" + std::to_string(head_dim),
     "projected_value_dim=" + std::to_string(head_dim),
     "max_timestep=" + std::to_string(max_timestep), "force_fp32=false",
     "is_causal=true"});

  ml::train::TensorDim query_dim(
    {batch_size, 1, seq_len, num_heads_Q * head_dim});
  ml::train::TensorDim key_dim(
    {batch_size, 1, seq_len, num_heads_KV * head_dim});
  ml::train::TensorDim value_dim(
    {batch_size, 1, seq_len, num_heads_KV * head_dim});

  std::vector<ml::train::TensorDim> input_dims = {query_dim, key_dim,
                                                   value_dim};

  nntrainer::InitLayerContext init_context(input_dims, {true}, false,
                                           "mha_core_test");
  EXPECT_NO_THROW(layer->finalize(init_context));

  // Verify tensor specs - KV cache should NOT be FP32
  auto tensor_specs = init_context.getTensorsSpec();
  ASSERT_GE(tensor_specs.size(), 2u);

  for (size_t i = 0; i < 2; ++i) {
    auto &dim = std::get<0>(tensor_specs[i]);
    EXPECT_NE(dim.getDataType(), ml::train::TensorDim::DataType::FP32)
      << "KV cache tensor " << i
      << " should NOT be FP32 when force_fp32=false";
  }
}

/**
 * @brief Test compute_kcaches<float> correctness
 *        Compares SIMD-optimized FP32 key cache computation against naive
 *        reference
 */
TEST(MHACoreCompute, compute_kcaches_fp32_p) {
  const int num_heads_Q = 4;
  const int num_heads_KV = 2;
  const int head_dim = 8;
  const int num_rows = 5;
  const int gqa_size = num_heads_Q / num_heads_KV;

  // Prepare query: (num_heads_Q * head_dim)
  std::vector<float> query(num_heads_Q * head_dim);
  for (size_t i = 0; i < query.size(); ++i) {
    query[i] = 0.1f * (i % 7) - 0.3f;
  }

  // Prepare key cache: (num_rows, num_heads_KV * head_dim)
  std::vector<float> kcache(num_rows * num_heads_KV * head_dim);
  for (size_t i = 0; i < kcache.size(); ++i) {
    kcache[i] = 0.05f * (i % 11) - 0.25f;
  }

  // Reference output
  std::vector<float> ref_output(num_rows * num_heads_Q, 0.0f);
  reference_qk_dot(query.data(), kcache.data(), ref_output.data(), num_rows,
                    num_heads_Q, num_heads_KV, head_dim);

  // Backend output
  std::vector<float> test_output(num_rows * num_heads_Q, 0.0f);
  nntrainer::compute_kcaches<float>(query.data(), kcache.data(),
                                    test_output.data(), num_rows, num_heads_KV,
                                    head_dim, gqa_size, 4);

  // Compare
  for (int i = 0; i < num_rows * num_heads_Q; ++i) {
    EXPECT_NEAR(test_output[i], ref_output[i], 1e-5f)
      << "Mismatch at index " << i;
  }
}

/**
 * @brief Test compute_fp32vcache_fp32_transposed correctness
 *        Compares optimized FP32 value cache computation against naive
 *        reference
 */
TEST(MHACoreCompute, compute_fp32vcache_transposed_p) {
  const int num_heads_Q = 4;
  const int num_heads_KV = 2;
  const int head_dim = 8;
  const int num_rows = 5;
  const int gqa_size = num_heads_Q / num_heads_KV;

  // Prepare attention weights: (num_rows, num_heads_Q)
  // Simulate softmax output (values sum to 1 per head across rows)
  std::vector<float> attn_weights(num_rows * num_heads_Q);
  for (int h = 0; h < num_heads_Q; ++h) {
    float sum = 0.0f;
    for (int r = 0; r < num_rows; ++r) {
      attn_weights[r * num_heads_Q + h] = 1.0f / num_rows;
    }
  }

  // Prepare value cache: (num_rows, num_heads_KV * head_dim)
  std::vector<float> vcache(num_rows * num_heads_KV * head_dim);
  for (size_t i = 0; i < vcache.size(); ++i) {
    vcache[i] = 0.1f * (i % 13) - 0.6f;
  }

  // Reference output
  std::vector<float> ref_output(num_heads_Q * head_dim, 0.0f);
  reference_attn_v(attn_weights.data(), vcache.data(), ref_output.data(),
                    num_rows, num_heads_Q, num_heads_KV, head_dim);

  // Backend output
  std::vector<float> test_output(num_heads_Q * head_dim, 0.0f);
  // row_num is the last valid row index (0-based), so num_rows - 1
  nntrainer::compute_fp32vcache_fp32_transposed(
    num_rows - 1, attn_weights.data(), vcache.data(), test_output.data(),
    num_heads_KV, gqa_size, head_dim);

  // Compare
  for (int i = 0; i < num_heads_Q * head_dim; ++i) {
    EXPECT_NEAR(test_output[i], ref_output[i], 1e-5f)
      << "Mismatch at index " << i;
  }
}

/**
 * @brief Test full FP32 attention pipeline (Q*K^T → softmax → attn*V)
 *        This verifies the end-to-end correctness of FP32 compute functions
 *        used in the force_fp32 mode of MHACore
 */
TEST(MHACoreCompute, full_fp32_attention_pipeline_p) {
  const int num_heads_Q = 4;
  const int num_heads_KV = 2;
  const int head_dim = 16;
  const int num_rows = 8;
  const int gqa_size = num_heads_Q / num_heads_KV;

  // Prepare query
  std::vector<float> query(num_heads_Q * head_dim);
  for (size_t i = 0; i < query.size(); ++i) {
    query[i] = std::sin(0.1f * i) * 0.5f;
  }

  // Prepare key cache
  std::vector<float> kcache(num_rows * num_heads_KV * head_dim);
  for (size_t i = 0; i < kcache.size(); ++i) {
    kcache[i] = std::cos(0.07f * i) * 0.3f;
  }

  // Prepare value cache
  std::vector<float> vcache(num_rows * num_heads_KV * head_dim);
  for (size_t i = 0; i < vcache.size(); ++i) {
    vcache[i] = std::sin(0.13f * i + 1.0f) * 0.4f;
  }

  // Step 1: Q*K^T
  std::vector<float> attn_scores(num_rows * num_heads_Q, 0.0f);
  nntrainer::compute_kcaches<float>(query.data(), kcache.data(),
                                    attn_scores.data(), num_rows, num_heads_KV,
                                    head_dim, gqa_size, 4);

  // Step 1 reference
  std::vector<float> ref_scores(num_rows * num_heads_Q, 0.0f);
  reference_qk_dot(query.data(), kcache.data(), ref_scores.data(), num_rows,
                    num_heads_Q, num_heads_KV, head_dim);

  for (int i = 0; i < num_rows * num_heads_Q; ++i) {
    EXPECT_NEAR(attn_scores[i], ref_scores[i], 1e-5f)
      << "QK score mismatch at index " << i;
  }

  // Step 2: Softmax
  reference_softmax_rows(attn_scores.data(), num_rows, num_heads_Q);

  // Verify softmax properties
  for (int h = 0; h < num_heads_Q; ++h) {
    float sum = 0.0f;
    for (int r = 0; r < num_rows; ++r) {
      float val = attn_scores[r * num_heads_Q + h];
      EXPECT_GE(val, 0.0f) << "Softmax output should be non-negative";
      sum += val;
    }
    EXPECT_NEAR(sum, 1.0f, 1e-5f)
      << "Softmax outputs should sum to 1 for head " << h;
  }

  // Step 3: attn_weights * V
  std::vector<float> output(num_heads_Q * head_dim, 0.0f);
  nntrainer::compute_fp32vcache_fp32_transposed(
    num_rows - 1, attn_scores.data(), vcache.data(), output.data(),
    num_heads_KV, gqa_size, head_dim);

  std::vector<float> ref_output(num_heads_Q * head_dim, 0.0f);
  reference_attn_v(attn_scores.data(), vcache.data(), ref_output.data(),
                    num_rows, num_heads_Q, num_heads_KV, head_dim);

  for (int i = 0; i < num_heads_Q * head_dim; ++i) {
    EXPECT_NEAR(output[i], ref_output[i], 1e-5f)
      << "Attention output mismatch at index " << i;
  }
}

/**
 * @brief Test MHACoreLayer incremental forwarding with force_fp32=true
 *        Creates a proper RunLayerContext and runs incremental forwarding
 */
TEST(MHACoreLayer, incremental_forwarding_fp32_p) {
  auto layer = std::make_unique<causallm::MHACoreLayer>();

  const unsigned int num_heads_Q = 4;
  const unsigned int num_heads_KV = 2;
  const unsigned int head_dim = 8;
  const unsigned int max_timestep = 16;
  const unsigned int batch_size = 1;
  const unsigned int seq_len = 3;

  layer->setProperty(
    {"num_heads=" + std::to_string(num_heads_Q),
     "num_heads_KV=" + std::to_string(num_heads_KV),
     "projected_key_dim=" + std::to_string(head_dim),
     "projected_value_dim=" + std::to_string(head_dim),
     "max_timestep=" + std::to_string(max_timestep), "force_fp32=true",
     "is_causal=true"});

  ml::train::TensorDim query_dim(
    {batch_size, 1, seq_len, num_heads_Q * head_dim});
  ml::train::TensorDim key_dim(
    {batch_size, 1, seq_len, num_heads_KV * head_dim});
  ml::train::TensorDim value_dim(
    {batch_size, 1, seq_len, num_heads_KV * head_dim});

  std::vector<ml::train::TensorDim> input_dims = {query_dim, key_dim,
                                                   value_dim};

  nntrainer::InitLayerContext init_context(input_dims, {true}, false,
                                           "mha_core_test");
  ASSERT_NO_THROW(layer->finalize(init_context));

  // Create tensors for RunLayerContext
  // Inputs
  std::vector<nntrainer::Var_Grad> inputs;
  for (auto &dim : init_context.getInputDimensions()) {
    inputs.emplace_back(dim, nntrainer::Initializer::NONE, true, true,
                        "input");
  }

  // Initialize inputs with deterministic values
  {
    auto &q = inputs[0].getVariableRef();
    float *q_data = q.getData<float>();
    for (unsigned int i = 0; i < q.size(); ++i) {
      q_data[i] = std::sin(0.1f * i) * 0.5f;
    }

    auto &k = inputs[1].getVariableRef();
    float *k_data = k.getData<float>();
    for (unsigned int i = 0; i < k.size(); ++i) {
      k_data[i] = std::cos(0.07f * i) * 0.3f;
    }

    auto &v = inputs[2].getVariableRef();
    float *v_data = v.getData<float>();
    for (unsigned int i = 0; i < v.size(); ++i) {
      v_data[i] = std::sin(0.13f * i + 1.0f) * 0.4f;
    }
  }

  // Outputs
  std::vector<nntrainer::Var_Grad> outputs;
  for (auto &spec : init_context.getOutSpecs()) {
    outputs.emplace_back(spec.variable_spec.dim, nntrainer::Initializer::NONE,
                         true, true, "output");
    outputs.back().getVariableRef().setZero();
  }

  // Weights (only if use_sink=true, which we don't use)
  std::vector<nntrainer::Weight> weights;

  // Internal tensors (KV cache etc.)
  std::vector<nntrainer::Var_Grad> tensors;
  for (auto &spec : init_context.getTensorsSpec()) {
    tensors.emplace_back(spec, true);
    tensors.back().getVariableRef().setZero();
  }

  // Create RunLayerContext
  std::vector<nntrainer::Weight *> w_ptrs;
  for (auto &w : weights)
    w_ptrs.push_back(&w);

  std::vector<nntrainer::Var_Grad *> in_ptrs;
  for (auto &in : inputs)
    in_ptrs.push_back(&in);

  std::vector<nntrainer::Var_Grad *> out_ptrs;
  for (auto &out : outputs)
    out_ptrs.push_back(&out);

  std::vector<nntrainer::Var_Grad *> t_ptrs;
  for (auto &t : tensors)
    t_ptrs.push_back(&t);

  nntrainer::RunLayerContext run_context("mha_core_test", true, 0.0f, false,
                                         1.0, nullptr, false, w_ptrs, in_ptrs,
                                         out_ptrs, t_ptrs);

  // Run incremental forwarding (prefill: from=0, to=seq_len)
  EXPECT_NO_THROW(
    layer->incremental_forwarding(run_context, 0, seq_len, false));

  // Verify output is not all zeros (computation happened)
  auto &output_tensor = outputs[0].getVariableRef();
  float *out_data = output_tensor.getData<float>();
  bool all_zero = true;
  for (unsigned int i = 0; i < output_tensor.size(); ++i) {
    if (std::abs(out_data[i]) > 1e-10f) {
      all_zero = false;
      break;
    }
  }
  EXPECT_FALSE(all_zero)
    << "Output should not be all zeros after incremental forwarding";

  // Verify output contains finite values
  for (unsigned int i = 0; i < output_tensor.size(); ++i) {
    EXPECT_TRUE(std::isfinite(out_data[i]))
      << "Output at index " << i << " is not finite: " << out_data[i];
  }

  // Run single-token decoding step (from=seq_len, to=seq_len+1)
  // Prepare single-token inputs
  ml::train::TensorDim q_step_dim({batch_size, 1, 1, num_heads_Q * head_dim});
  ml::train::TensorDim k_step_dim(
    {batch_size, 1, 1, num_heads_KV * head_dim});
  ml::train::TensorDim v_step_dim(
    {batch_size, 1, 1, num_heads_KV * head_dim});
  ml::train::TensorDim o_step_dim({batch_size, 1, 1, num_heads_Q * head_dim});

  std::vector<nntrainer::Var_Grad> step_inputs;
  step_inputs.emplace_back(q_step_dim, nntrainer::Initializer::NONE, true, true,
                           "input");
  step_inputs.emplace_back(k_step_dim, nntrainer::Initializer::NONE, true, true,
                           "input");
  step_inputs.emplace_back(v_step_dim, nntrainer::Initializer::NONE, true, true,
                           "input");

  // Fill step inputs
  {
    float *q_data = step_inputs[0].getVariableRef().getData<float>();
    for (unsigned int i = 0; i < step_inputs[0].getVariableRef().size(); ++i)
      q_data[i] = 0.2f * std::sin(0.3f * i);

    float *k_data = step_inputs[1].getVariableRef().getData<float>();
    for (unsigned int i = 0; i < step_inputs[1].getVariableRef().size(); ++i)
      k_data[i] = 0.15f * std::cos(0.2f * i);

    float *v_data = step_inputs[2].getVariableRef().getData<float>();
    for (unsigned int i = 0; i < step_inputs[2].getVariableRef().size(); ++i)
      v_data[i] = 0.1f * std::sin(0.25f * i + 0.5f);
  }

  std::vector<nntrainer::Var_Grad> step_outputs;
  step_outputs.emplace_back(o_step_dim, nntrainer::Initializer::NONE, true,
                            true, "output");
  step_outputs.back().getVariableRef().setZero();

  std::vector<nntrainer::Var_Grad *> step_in_ptrs;
  for (auto &in : step_inputs)
    step_in_ptrs.push_back(&in);

  std::vector<nntrainer::Var_Grad *> step_out_ptrs;
  for (auto &out : step_outputs)
    step_out_ptrs.push_back(&out);

  nntrainer::RunLayerContext step_context("mha_core_test", true, 0.0f, false,
                                          1.0, nullptr, false, w_ptrs,
                                          step_in_ptrs, step_out_ptrs, t_ptrs);

  // Decoding step
  EXPECT_NO_THROW(
    layer->incremental_forwarding(step_context, seq_len, seq_len + 1, false));

  // Verify decoding output
  auto &step_output_tensor = step_outputs[0].getVariableRef();
  float *step_out_data = step_output_tensor.getData<float>();
  all_zero = true;
  for (unsigned int i = 0; i < step_output_tensor.size(); ++i) {
    if (std::abs(step_out_data[i]) > 1e-10f) {
      all_zero = false;
      break;
    }
  }
  EXPECT_FALSE(all_zero)
    << "Decoding output should not be all zeros";

  for (unsigned int i = 0; i < step_output_tensor.size(); ++i) {
    EXPECT_TRUE(std::isfinite(step_out_data[i]))
      << "Decoding output at index " << i
      << " is not finite: " << step_out_data[i];
  }
}

/**
 * @brief Test MHACoreLayer with GQA (Grouped Query Attention) configuration
 *        num_heads_Q != num_heads_KV
 */
TEST(MHACoreLayer, finalize_gqa_configuration_p) {
  auto layer = std::make_unique<causallm::MHACoreLayer>();

  unsigned int num_heads_Q = 8;
  unsigned int num_heads_KV = 2;
  unsigned int head_dim = 16;

  layer->setProperty(
    {"num_heads=" + std::to_string(num_heads_Q),
     "num_heads_KV=" + std::to_string(num_heads_KV),
     "projected_key_dim=" + std::to_string(head_dim),
     "projected_value_dim=" + std::to_string(head_dim), "max_timestep=64",
     "force_fp32=true", "is_causal=true"});

  ml::train::TensorDim query_dim({1, 1, 1, num_heads_Q * head_dim});
  ml::train::TensorDim key_dim({1, 1, 1, num_heads_KV * head_dim});
  ml::train::TensorDim value_dim({1, 1, 1, num_heads_KV * head_dim});

  std::vector<ml::train::TensorDim> input_dims = {query_dim, key_dim,
                                                   value_dim};

  nntrainer::InitLayerContext init_context(input_dims, {true}, false,
                                           "mha_core_gqa");
  EXPECT_NO_THROW(layer->finalize(init_context));

  auto out_specs = init_context.getOutSpecs();
  ASSERT_EQ(out_specs.size(), 1u);
  EXPECT_EQ(out_specs[0].variable_spec.dim.width(), num_heads_Q * head_dim);
}

/**
 * @brief Test compute_kcaches<float> with GQA (group_size > 1)
 */
TEST(MHACoreCompute, compute_kcaches_fp32_gqa_p) {
  const int num_heads_Q = 8;
  const int num_heads_KV = 2;
  const int head_dim = 16;
  const int num_rows = 4;
  const int gqa_size = num_heads_Q / num_heads_KV;

  std::vector<float> query(num_heads_Q * head_dim);
  for (size_t i = 0; i < query.size(); ++i)
    query[i] = std::sin(0.05f * i) * 0.3f;

  std::vector<float> kcache(num_rows * num_heads_KV * head_dim);
  for (size_t i = 0; i < kcache.size(); ++i)
    kcache[i] = std::cos(0.03f * i + 0.5f) * 0.2f;

  std::vector<float> ref_output(num_rows * num_heads_Q, 0.0f);
  reference_qk_dot(query.data(), kcache.data(), ref_output.data(), num_rows,
                    num_heads_Q, num_heads_KV, head_dim);

  std::vector<float> test_output(num_rows * num_heads_Q, 0.0f);
  nntrainer::compute_kcaches<float>(query.data(), kcache.data(),
                                    test_output.data(), num_rows, num_heads_KV,
                                    head_dim, gqa_size, 4);

  for (int i = 0; i < num_rows * num_heads_Q; ++i) {
    EXPECT_NEAR(test_output[i], ref_output[i], 1e-4f)
      << "GQA QK mismatch at index " << i;
  }
}

/**
 * @brief Test MHACoreLayer with non-causal attention and force_fp32
 */
TEST(MHACoreLayer, finalize_non_causal_fp32_p) {
  auto layer = std::make_unique<causallm::MHACoreLayer>();

  layer->setProperty({"num_heads=2", "num_heads_KV=2", "projected_key_dim=4",
                       "projected_value_dim=4", "max_timestep=16",
                       "force_fp32=true", "is_causal=false"});

  ml::train::TensorDim query_dim({1, 1, 1, 8});
  ml::train::TensorDim key_dim({1, 1, 1, 8});
  ml::train::TensorDim value_dim({1, 1, 1, 8});

  std::vector<ml::train::TensorDim> input_dims = {query_dim, key_dim,
                                                   value_dim};

  nntrainer::InitLayerContext init_context(input_dims, {true}, false,
                                           "mha_core_noncausal");
  EXPECT_NO_THROW(layer->finalize(init_context));
}

/**
 * @brief Test compute_kcaches<float> with local window (sliding window
 * attention)
 */
TEST(MHACoreCompute, compute_kcaches_fp32_local_window_p) {
  const int num_heads_Q = 2;
  const int num_heads_KV = 2;
  const int head_dim = 8;
  const int num_rows = 10;
  const int gqa_size = 1;
  const size_t local_window_size = 4;

  std::vector<float> query(num_heads_Q * head_dim);
  for (size_t i = 0; i < query.size(); ++i)
    query[i] = 0.1f * ((i + 3) % 5) - 0.2f;

  std::vector<float> kcache(num_rows * num_heads_KV * head_dim);
  for (size_t i = 0; i < kcache.size(); ++i)
    kcache[i] = 0.08f * ((i + 7) % 9) - 0.35f;

  // With local_window_size=4, only the last 4 rows are used
  std::vector<float> test_output(local_window_size * num_heads_Q, 0.0f);
  nntrainer::compute_kcaches<float>(query.data(), kcache.data(),
                                    test_output.data(), num_rows, num_heads_KV,
                                    head_dim, gqa_size, 4, local_window_size);

  // Reference: compute only for the windowed rows
  int start_row = num_rows - local_window_size;
  std::vector<float> ref_output(local_window_size * num_heads_Q, 0.0f);
  for (int n = 0; n < num_heads_KV; ++n) {
    for (int g = 0; g < gqa_size; ++g) {
      int q_head = n * gqa_size + g;
      const float *q_ptr = query.data() + q_head * head_dim;
      for (int w = 0; w < (int)local_window_size; ++w) {
        int row = start_row + w;
        const float *k_ptr =
          kcache.data() + (row * num_heads_KV + n) * head_dim;
        float sum = 0.0f;
        for (int d = 0; d < head_dim; ++d)
          sum += q_ptr[d] * k_ptr[d];
        ref_output[w * num_heads_Q + q_head] = sum / std::sqrt((float)head_dim);
      }
    }
  }

  for (int i = 0; i < (int)local_window_size * num_heads_Q; ++i) {
    EXPECT_NEAR(test_output[i], ref_output[i], 1e-5f)
      << "Windowed QK mismatch at index " << i;
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
