/**
 * Copyright (C) 2020 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *
 * @file	moe_layer.cpp
 * @date	09 June 2025
 * @brief	This is a Mixture of Expert Layer Class for Neural Network
 * @see		https://github.com/nnstreamer/
 * @author	Eunju Yang <ej.yang@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <acti_func.h>
#include <cmath>
#include <node_exporter.h>
#include <omp.h>
#include <qwen_moe_layer.h>
#include <stdexcept>
#include <immintrin.h>  // For SIMD optimizations
#include <algorithm>

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;
// Cache line size for optimal memory access
static constexpr size_t CACHE_LINE_SIZE = 64;

MoELayer::MoELayer() :
  LayerImpl(),
  num_experts(0),
  topk(0),
  moe_props(props::NumExperts(), props::NumExpertsPerToken(),
            nntrainer::props::Unit(), props::MoEActivation()),
  expert_gate_proj_indices({}),
  expert_up_proj_indices({}),
  expert_down_proj_indices({}),
  gate_idx(std::numeric_limits<unsigned>::max()),
  router_logits_idx(std::numeric_limits<unsigned>::max()),
  expert_mask_idx(std::numeric_limits<unsigned>::max()) {}

void MoELayer::finalize(nntrainer::InitLayerContext &context) {

  // 1. Validate input/output dimensions
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "MoE layer only supports single input";

  auto &weight_regularizer =
    std::get<nntrainer::props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<nntrainer::props::WeightRegularizerConstant>(*layer_impl_props);
  auto &weight_initializer =
    std::get<nntrainer::props::WeightInitializer>(*layer_impl_props);
  auto &weight_decay =
    std::get<nntrainer::props::WeightDecay>(*layer_impl_props);

  // 2. Set output dimensions (same as input)
  const auto &in_dim = context.getInputDimensions()[SINGLE_INOUT_IDX];
  const bool is_nchw = context.getFormat() == nntrainer::Tformat::NCHW;
  std::vector<nntrainer::TensorDim> output_dims(1);
  output_dims[SINGLE_INOUT_IDX] = in_dim;
  context.setOutputDimensions(output_dims);

  // 3. Get MoE properties
  num_experts = std::get<props::NumExperts>(moe_props).get();
  topk = std::get<props::NumExpertsPerToken>(moe_props).get();
  const unsigned int intermediate_size =
    std::get<nntrainer::props::Unit>(moe_props).get();
  const unsigned int hidden_size = in_dim.width(); // Feature dimension

  // activation function
  if (std::get<props::MoEActivation>(moe_props).empty()) {
    throw std::runtime_error("Activation type is not set for MoE layer");
  }
  switch (context.getActivationDataType()) {
  case ml::train::TensorDim::DataType::FP32:
    acti_func.setActiFunc<float>(
      std::get<props::MoEActivation>(moe_props).get());
    break;
  default:
    throw std::runtime_error("Unsupported activation data type for MoE layer");
  }

  // 4. Initialie gate layer (router)
  nntrainer::TensorDim gate_dim(
    1, is_nchw ? 1 : num_experts, is_nchw ? hidden_size : 1,
    is_nchw ? num_experts : hidden_size,
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()),
    is_nchw ? 0b0011 : 0b0101);

  gate_idx = context.requestWeight(
    gate_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "gate", true);

  // 5. Initializer expert weights
  expert_gate_proj_indices.reserve(num_experts);
  expert_up_proj_indices.reserve(num_experts);
  expert_down_proj_indices.reserve(num_experts);

  nntrainer::TensorDim expert_gate_dim(
    1, is_nchw ? 1 : intermediate_size, is_nchw ? hidden_size : 1,
    is_nchw ? intermediate_size : hidden_size,
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()),
    is_nchw ? 0b0011 : 0b0101);

  nntrainer::TensorDim expert_down_dim(
    1, is_nchw ? 1 : hidden_size, is_nchw ? intermediate_size : 1,
    is_nchw ? hidden_size : intermediate_size,
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()),
    is_nchw ? 0b0011 : 0b0101);

  for (unsigned int i = 0; i < num_experts; ++i) {
    // Up projection
    expert_up_proj_indices.push_back(context.requestWeight(
      expert_gate_dim, // Same dimensions as gate projection
      weight_initializer, weight_regularizer, weight_regularizer_constant,
      weight_decay, "expert_up_" + std::to_string(i), false));

    // Gate projection
    expert_gate_proj_indices.push_back(context.requestWeight(
      expert_gate_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay,
      "expert_gate_" + std::to_string(i), true));

    // Down projection
    expert_down_proj_indices.push_back(context.requestWeight(
      expert_down_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay,
      "expert_down_" + std::to_string(i), false));
  }

  // 6. Request intermediate tensors with optimized sizes
  const unsigned batch_size = in_dim.batch();
  const unsigned seq_len = in_dim.height();
  const unsigned total_tokens = batch_size * seq_len;

  // Router logits :  [batch * seq, num_experts]
  router_logits_idx =
    context.requestTensor({total_tokens, 1, 1, num_experts}, "router_logits",
                          nntrainer::Initializer::NONE, false,
                          nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);

  // Expert mask: [num_experts, batch*seq] - optimized layout
  expert_mask_idx =
    context.requestTensor({num_experts, 1, topk, total_tokens}, "expert_mask",
                          nntrainer::Initializer::ZEROS, false,
                          nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
  
  // Pre-allocate thread-local intermediate buffers for better performance
  unsigned int max_threads = omp_get_max_threads();
  thread_local_buffers.resize(max_threads);
  for (unsigned int i = 0; i < max_threads; ++i) {
    thread_local_buffers[i].gate_out.resize(intermediate_size);
    thread_local_buffers[i].up_out.resize(intermediate_size);
    thread_local_buffers[i].intermediate.resize(intermediate_size);
  }
}

void MoELayer::forwarding(nntrainer::RunLayerContext &context, bool training) {
  nntrainer::Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &output = context.getOutput(SINGLE_INOUT_IDX);

  nntrainer::Tensor &router_logits = context.getTensor(router_logits_idx);
  nntrainer::Tensor &expert_mask = context.getTensor(expert_mask_idx);

  const unsigned batch_size = input.batch();
  const unsigned seq_len = input.height();
  const unsigned hidden_size = input.width();
  const unsigned total_tokens = batch_size * seq_len;

  // reshape input: [B,1,S,H] -> [B*S,1,1,H]
  input.reshape({total_tokens, 1, 1, hidden_size});

  // reshape output: [B,1,S,H] -> [B*S,1,1,H]
  output.reshape({total_tokens, 1, 1, hidden_size});
  output.setZero();

  // routing
  nntrainer::Tensor &gate_weights = context.getWeight(gate_idx);
  input.dot(gate_weights, router_logits);
  router_logits.apply(nntrainer::ActiFunc::softmax<float>, router_logits);
  auto topk_result = router_logits.topK(topk);
  auto topk_values = std::get<0>(topk_result);
  auto topk_indices = std::get<1>(topk_result);

  const uint32_t *indices_data = topk_indices.getData<uint32_t>();
  const float *values_data = topk_values.getData<float>();
  
  // Pre-compute expert token assignments for better cache locality
  std::vector<std::vector<std::pair<unsigned, float>>> expert_assignments(
    num_experts);
  
  // Reserve space to avoid reallocations
  for (unsigned int i = 0; i < num_experts; ++i) {
    expert_assignments[i].reserve(total_tokens * topk / num_experts + 1);
  }
  
  // Optimized assignment collection with better memory access pattern
  for (unsigned int i = 0; i < total_tokens; ++i) {
    const unsigned int base_idx = i * topk;
    for (unsigned int k = 0; k < topk; ++k) {
      unsigned expert_idx = indices_data[base_idx + k];
      float weight = values_data[base_idx + k];
      expert_assignments[expert_idx].emplace_back(i, weight);
      expert_mask.setValue(expert_idx, 0, k, i, 1.0f);
    }
  }

// expert forwarding with optimized memory access and SIMD
#pragma omp parallel
  {
    int thread_id = omp_get_thread_num();
    auto &buffer = thread_local_buffers[thread_id];
    
#pragma omp for schedule(dynamic, 1)
    for (int expert_idx = 0; expert_idx < static_cast<int>(num_experts);
         ++expert_idx) {
      const auto &assignments = expert_assignments[expert_idx];
      if (assignments.empty())
        continue;

      // Use optimized expert forward computation with thread-local buffers
      compute_expert_forward_optimized(
        input, output, assignments,
        context.getWeight(expert_gate_proj_indices[expert_idx]),
        context.getWeight(expert_up_proj_indices[expert_idx]),
        context.getWeight(expert_down_proj_indices[expert_idx]), 
        hidden_size, buffer);
    }
  }

  // reshape output: [B*S,1,1,H] -> [B,1,S,H]
  output.reshape({batch_size, 1, seq_len, hidden_size});
}

// Optimized compute function using SIMD and better memory patterns
inline void MoELayer::compute_expert_forward_optimized(
  const nntrainer::Tensor &input, nntrainer::Tensor &output,
  const std::vector<std::pair<unsigned, float>> &token_assignments,
  const nntrainer::Tensor &gate_proj, const nntrainer::Tensor &up_proj,
  const nntrainer::Tensor &down_proj, unsigned int hidden_size,
  ThreadLocalBuffer &buffer) {

  const unsigned intermediate_size = gate_proj.width();
  const unsigned num_tokens = token_assignments.size();

  if (num_tokens == 0)
    return;

  const float *gate_data = gate_proj.getData<float>();
  const float *up_data = up_proj.getData<float>();
  const float *down_data = down_proj.getData<float>();

  // Process tokens in batches for better cache utilization
  constexpr size_t BATCH_SIZE = 4;
  
  for (size_t batch_start = 0; batch_start < num_tokens; batch_start += BATCH_SIZE) {
    size_t batch_end = std::min(batch_start + BATCH_SIZE, num_tokens);
    
    for (size_t i = batch_start; i < batch_end; ++i) {
      const unsigned token_idx = token_assignments[i].first;
      const float weight = token_assignments[i].second;
      
      const float *input_ptr = input.getData<float>() + token_idx * hidden_size;
      float *output_ptr = output.getData<float>() + token_idx * hidden_size;
      
      // Reset buffers
      std::fill(buffer.gate_out.begin(), buffer.gate_out.end(), 0.0f);
      std::fill(buffer.up_out.begin(), buffer.up_out.end(), 0.0f);
      
      // Optimized matrix multiplication using SIMD when possible
      // Gate projection: input @ gate_proj^T
      optimized_gemv(input_ptr, gate_data, buffer.gate_out.data(), 
                     hidden_size, intermediate_size);
      
      // Up projection: input @ up_proj^T
      optimized_gemv(input_ptr, up_data, buffer.up_out.data(), 
                     hidden_size, intermediate_size);
      
      // Apply activation (silu) and multiply with up_out
      apply_silu_and_multiply(buffer.gate_out.data(), buffer.up_out.data(), 
                              buffer.intermediate.data(), intermediate_size);
      
      // Down projection and accumulate: intermediate @ down_proj^T
      optimized_gemv_accumulate(buffer.intermediate.data(), down_data, 
                                output_ptr, intermediate_size, hidden_size, weight);
    }
  }
}

// Optimized matrix-vector multiplication
inline void MoELayer::optimized_gemv(const float *matrix_row, const float *matrix,
                                     float *result, size_t in_dim, size_t out_dim) {
  // Use SIMD instructions for better performance
  #pragma omp simd aligned(matrix_row, matrix, result : CACHE_LINE_SIZE)
  for (size_t i = 0; i < out_dim; ++i) {
    float sum = 0.0f;
    const float *mat_ptr = matrix + i * in_dim;
    
    // Unroll loop for better performance
    size_t j = 0;
    for (; j + 7 < in_dim; j += 8) {
      sum += matrix_row[j] * mat_ptr[j];
      sum += matrix_row[j+1] * mat_ptr[j+1];
      sum += matrix_row[j+2] * mat_ptr[j+2];
      sum += matrix_row[j+3] * mat_ptr[j+3];
      sum += matrix_row[j+4] * mat_ptr[j+4];
      sum += matrix_row[j+5] * mat_ptr[j+5];
      sum += matrix_row[j+6] * mat_ptr[j+6];
      sum += matrix_row[j+7] * mat_ptr[j+7];
    }
    
    // Handle remaining elements
    for (; j < in_dim; ++j) {
      sum += matrix_row[j] * mat_ptr[j];
    }
    
    result[i] = sum;
  }
}

// Optimized matrix-vector multiplication with accumulation
inline void MoELayer::optimized_gemv_accumulate(const float *matrix_row, const float *matrix,
                                                float *result, size_t in_dim, size_t out_dim,
                                                float scale) {
  #pragma omp simd aligned(matrix_row, matrix, result : CACHE_LINE_SIZE)
  for (size_t i = 0; i < out_dim; ++i) {
    float sum = 0.0f;
    const float *mat_ptr = matrix + i * in_dim;
    
    // Unroll loop for better performance
    size_t j = 0;
    for (; j + 7 < in_dim; j += 8) {
      sum += matrix_row[j] * mat_ptr[j];
      sum += matrix_row[j+1] * mat_ptr[j+1];
      sum += matrix_row[j+2] * mat_ptr[j+2];
      sum += matrix_row[j+3] * mat_ptr[j+3];
      sum += matrix_row[j+4] * mat_ptr[j+4];
      sum += matrix_row[j+5] * mat_ptr[j+5];
      sum += matrix_row[j+6] * mat_ptr[j+6];
      sum += matrix_row[j+7] * mat_ptr[j+7];
    }
    
    // Handle remaining elements
    for (; j < in_dim; ++j) {
      sum += matrix_row[j] * mat_ptr[j];
    }
    
    result[i] += sum * scale;
  }
}

// Apply SiLU activation and element-wise multiplication
inline void MoELayer::apply_silu_and_multiply(const float *gate_out, const float *up_out,
                                              float *result, size_t size) {
  #pragma omp simd aligned(gate_out, up_out, result : CACHE_LINE_SIZE)
  for (size_t i = 0; i < size; ++i) {
    // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    float x = gate_out[i];
    float sigmoid = 1.0f / (1.0f + expf(-x));
    result[i] = x * sigmoid * up_out[i];
  }
}

void MoELayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                      unsigned int from, unsigned int to,
                                      bool training) {
  if (from) {
    NNTR_THROW_IF(to - from != 1, std::invalid_argument)
      << "incremental step size is not 1";
    from = 0;
    to = 1;
  }

  nntrainer::Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &output_ = context.getOutput(SINGLE_INOUT_IDX);

  nntrainer::Tensor &router_logits_ = context.getTensor(router_logits_idx);
  nntrainer::Tensor &expert_mask = context.getTensor(expert_mask_idx);

  nntrainer::TensorDim input_step_dim = input_.getDim();
  nntrainer::TensorDim output_step_dim = output_.getDim();
  nntrainer::TensorDim router_logits_step_dim = router_logits_.getDim();

  input_step_dim.batch(1);
  output_step_dim.batch(1);
  router_logits_step_dim.batch(to - from);

  input_step_dim.height(to - from);
  output_step_dim.height(to - from);

  // Get thread-local buffer for single-threaded incremental processing
  auto &buffer = thread_local_buffers[0];

  for (unsigned int b = 0; b < input_.batch(); ++b) {

    auto input = input_.getSharedDataTensor(
      input_step_dim, b * input_step_dim.getFeatureLen(), true);
    auto output = output_.getSharedDataTensor(
      output_step_dim, b * output_step_dim.getFeatureLen(), true);
    auto router_logits =
      router_logits_.getSharedDataTensor(router_logits_step_dim, 0, true);

    const unsigned batch_size = input.batch();
    const unsigned seq_len = input.height();
    const unsigned hidden_size = input.width();
    const unsigned total_tokens = batch_size * seq_len;

    // For incremental forwarding, we typically process one token
    // Optimize for this case
    if (total_tokens == 1) {
      // Direct processing without reshaping for single token
      output.setZero();
      expert_mask.setZero();

      // Routing - optimized for single token
      nntrainer::Tensor &gate_weights = context.getWeight(gate_idx);
      const float *input_data = input.getData<float>();
      float *logits_data = router_logits.getData<float>();
      
      // Direct matrix-vector multiplication for routing
      optimized_gemv(input_data, gate_weights.getData<float>(), 
                     logits_data, hidden_size, num_experts);
      
      // Apply softmax in-place
      float max_val = *std::max_element(logits_data, logits_data + num_experts);
      float sum = 0.0f;
      for (unsigned i = 0; i < num_experts; ++i) {
        logits_data[i] = expf(logits_data[i] - max_val);
        sum += logits_data[i];
      }
      float inv_sum = 1.0f / sum;
      for (unsigned i = 0; i < num_experts; ++i) {
        logits_data[i] *= inv_sum;
      }
      
      // Find top-k experts efficiently for single token
      std::vector<std::pair<float, unsigned>> expert_scores;
      expert_scores.reserve(num_experts);
      for (unsigned i = 0; i < num_experts; ++i) {
        expert_scores.emplace_back(logits_data[i], i);
      }
      
      // Partial sort to get top-k
      std::partial_sort(expert_scores.begin(), expert_scores.begin() + topk,
                        expert_scores.end(), std::greater<std::pair<float, unsigned>>());
      
      // Normalize top-k weights
      float topk_sum = 0.0f;
      for (unsigned k = 0; k < topk; ++k) {
        topk_sum += expert_scores[k].first;
      }
      float norm_factor = 1.0f / topk_sum;
      
      // Process selected experts directly
      float *output_data = output.getData<float>();
      for (unsigned k = 0; k < topk; ++k) {
        unsigned expert_idx = expert_scores[k].second;
        float weight = expert_scores[k].first * norm_factor;
        
        // Set expert mask
        expert_mask.setValue(expert_idx, 0, k, 0, 1.0f);
        
        // Process this expert
        const float *gate_data = context.getWeight(expert_gate_proj_indices[expert_idx]).getData<float>();
        const float *up_data = context.getWeight(expert_up_proj_indices[expert_idx]).getData<float>();
        const float *down_data = context.getWeight(expert_down_proj_indices[expert_idx]).getData<float>();
        
        const unsigned intermediate_size = context.getWeight(expert_gate_proj_indices[expert_idx]).width();
        
        // Gate projection
        optimized_gemv(input_data, gate_data, buffer.gate_out.data(), 
                       hidden_size, intermediate_size);
        
        // Up projection
        optimized_gemv(input_data, up_data, buffer.up_out.data(), 
                       hidden_size, intermediate_size);
        
        // Apply activation and multiply
        apply_silu_and_multiply(buffer.gate_out.data(), buffer.up_out.data(),
                                buffer.intermediate.data(), intermediate_size);
        
        // Down projection with accumulation
        optimized_gemv_accumulate(buffer.intermediate.data(), down_data,
                                  output_data, intermediate_size, hidden_size, weight);
      }
      
      continue; // Skip the general path
    }

    // General path for multiple tokens (fallback)
    // reshape input: [B,1,S,H] -> [B*S,1,1,H]
    input.reshape({total_tokens, 1, 1, hidden_size});

    // reshape output: [B,1,S,H] -> [B*S,1,1,H]
    output.reshape({total_tokens, 1, 1, hidden_size});
    output.setZero();
    expert_mask.setZero();

    // routing
    nntrainer::Tensor &gate_weights = context.getWeight(gate_idx);
    input.dot(gate_weights, router_logits);
    router_logits.apply(nntrainer::ActiFunc::softmax<float>, router_logits);
    auto topk_result = router_logits.topK(topk);
    auto topk_values = std::get<0>(topk_result);
    auto topk_indices = std::get<1>(topk_result);

    // norm_topk_prob
    topk_values.divide_i(topk_values.sum(3));

    const uint32_t *indices_data = topk_indices.getData<uint32_t>();
    const float *values_data = topk_values.getData<float>();
    
    // Set expert mask and prepare assignments
    std::vector<std::vector<std::pair<unsigned, float>>> expert_assignments(
      num_experts);
    for (int i = 0; i < static_cast<int>(total_tokens); ++i) {
      for (int k = 0; k < static_cast<int>(topk); ++k) {
        unsigned expert_idx = indices_data[i * topk + k];
        float weight = values_data[i * topk + k];
        expert_mask.setValue(expert_idx, 0, k, i, 1.0f);
        expert_assignments[expert_idx].emplace_back(i, weight);
      }
    }

    // expert forwarding with optimized memory access
    for (int expert_idx = 0; expert_idx < static_cast<int>(num_experts);
         ++expert_idx) {
      const auto &assignments = expert_assignments[expert_idx];
      if (assignments.empty())
        continue;

      // Use optimized computation
      compute_expert_forward_optimized(
        input, output, assignments,
        context.getWeight(expert_gate_proj_indices[expert_idx]),
        context.getWeight(expert_up_proj_indices[expert_idx]),
        context.getWeight(expert_down_proj_indices[expert_idx]), 
        hidden_size, buffer);
    }

    // reshape output: [B*S,1,1,H] -> [B,1,S,H]
    output.reshape({batch_size, 1, seq_len, hidden_size});
  }
}

// Keep the original compute_expert_forward for compatibility
inline void MoELayer::compute_expert_forward(
  const nntrainer::Tensor &input, nntrainer::Tensor &output,
  const std::vector<std::pair<unsigned, float>> &token_assignments,
  const nntrainer::Tensor &gate_proj, const nntrainer::Tensor &up_proj,
  const nntrainer::Tensor &down_proj, unsigned int hidden_size) {

  // Use thread-local buffer 0 for backward compatibility
  compute_expert_forward_optimized(input, output, token_assignments,
                                   gate_proj, up_proj, down_proj,
                                   hidden_size, thread_local_buffers[0]);
}

void MoELayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, moe_props);
  nntrainer::LayerImpl::setProperty(remain_props);
}

void MoELayer::calcDerivative(nntrainer::RunLayerContext &context) {
  // MoE layer does not support derivative calculation
  throw std::runtime_error("MoE layer does not support derivative calculation");
}

void MoELayer::calcGradient(nntrainer::RunLayerContext &context) {
  // MoE layer does not support gradient calculation
  throw std::runtime_error("MoE layer does not support gradient calculation");
}

void MoELayer::exportTo(nntrainer::Exporter &exporter,
                        const ml::train::ExportMethods &method) const {
  nntrainer::LayerImpl::exportTo(exporter, method);
  exporter.saveResult(moe_props, method, this); // Save MoE specific properties
}

} // namespace causallm
