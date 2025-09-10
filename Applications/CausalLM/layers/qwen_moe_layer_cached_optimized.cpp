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
 * @file	qwen_moe_layer_cached_optimized.cpp
 * @date	09 June 2025
 * @brief	Optimized Mixture of Expert Layer Class for Neural Network
 * @see		https://github.com/nnstreamer/
 * @author	Eunju Yang <ej.yang@samsung.com>
 * @bug		No known bugs except for NYI items
 * @note    MoE layer with on-the-fly expert FSU - OPTIMIZED VERSION
 *
 */

#include <acti_func.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <node_exporter.h>
#include <omp.h>
#include <qwen_moe_layer_cached.h>
#include <stdexcept>
#include <immintrin.h>  // For SIMD intrinsics
#include <future>       // For async operations

#include <chrono>
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::microseconds;
using std::chrono::milliseconds;
using std::chrono::nanoseconds;
using std::chrono::seconds;

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;
static constexpr size_t CACHE_LINE_SIZE = 64;
static constexpr size_t PREFETCH_DISTANCE = 8;
static constexpr size_t MAX_CACHED_EXPERTS = 32;
static constexpr size_t BATCH_SIZE_THRESHOLD = 4;

// Thread-local tensor pools to reduce allocation overhead
thread_local std::vector<nntrainer::Tensor*> tensor_pool_intermediate;
thread_local std::vector<nntrainer::Tensor*> tensor_pool_output;
thread_local size_t pool_idx_intermediate = 0;
thread_local size_t pool_idx_output = 0;

CachedSlimMoELayer::CachedSlimMoELayer() :
  LayerImpl(),
  num_experts(0),
  topk(0),
  moe_props(props::NumExperts(), props::NumExpertsPerToken(),
            nntrainer::props::Unit(), props::MoEActivation()),
  expert_gate_proj_indices({}),
  expert_up_proj_indices({}),
  expert_down_proj_indices({}),
  loaded_expert_deque({}),
  need_load({}),
  gate_idx(std::numeric_limits<unsigned>::max()),
  router_logits_idx(std::numeric_limits<unsigned>::max()),
  expert_mask_idx(std::numeric_limits<unsigned>::max()) {}

void CachedSlimMoELayer::finalize(nntrainer::InitLayerContext &context) {

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

  // 4. Initialize gate layer (router)
  nntrainer::TensorDim gate_dim(
    1, is_nchw ? 1 : num_experts, is_nchw ? hidden_size : 1,
    is_nchw ? num_experts : hidden_size,
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     nntrainer::TensorDim::DataType::FP32),
    is_nchw ? 0b0011 : 0b0101);

  gate_idx = context.requestWeight(
    gate_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "gate", true);

  // 5. Initialize expert weights
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
      weight_decay, "expert_up_" + std::to_string(i), false, true));

    // Gate projection
    expert_gate_proj_indices.push_back(context.requestWeight(
      expert_gate_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay,
      "expert_gate_" + std::to_string(i), false, true));

    // Down projection
    expert_down_proj_indices.push_back(context.requestWeight(
      expert_down_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay,
      "expert_down_" + std::to_string(i), false, true));
    need_load.push_back(true);
  }

  // 6. Request intermediate tensors
  const unsigned batch_size = in_dim.batch();
  const unsigned seq_len = in_dim.height();
  const unsigned total_tokens = batch_size * seq_len;

  // Router logits :  [batch * seq, num_experts]
  router_logits_idx =
    context.requestTensor({total_tokens, 1, 1, num_experts}, "router_logits",
                          nntrainer::Initializer::NONE, false,
                          nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);

  // Expert mask: [num_experts, batch*seq]
  expert_mask_idx =
    context.requestTensor({num_experts, 1, topk, total_tokens}, "expert_mask",
                          nntrainer::Initializer::ZEROS, false,
                          nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
  
  // Pre-allocate tensor pools for each thread
  int num_threads = omp_get_max_threads();
  #pragma omp parallel
  {
    tensor_pool_intermediate.reserve(16);
    tensor_pool_output.reserve(16);
  }
}

// Optimized SIMD-based matrix multiplication for small matrices
inline void optimized_gemm_small(const float* A, const float* B, float* C,
                                 size_t M, size_t N, size_t K) {
  // Use AVX2 for vectorized operations
  const size_t simd_width = 8; // AVX2 processes 8 floats at once
  
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; j += simd_width) {
      __m256 sum = _mm256_setzero_ps();
      
      for (size_t k = 0; k < K; ++k) {
        __m256 a = _mm256_broadcast_ss(&A[i * K + k]);
        __m256 b = _mm256_loadu_ps(&B[k * N + j]);
        sum = _mm256_fmadd_ps(a, b, sum);
      }
      
      _mm256_storeu_ps(&C[i * N + j], sum);
    }
    
    // Handle remaining elements
    for (size_t j = (N / simd_width) * simd_width; j < N; ++j) {
      float sum = 0.0f;
      for (size_t k = 0; k < K; ++k) {
        sum += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = sum;
    }
  }
}

// Optimized batched matrix multiplication
inline void batched_gemm_optimized(const std::vector<const float*>& A_batch,
                                   const float* B,
                                   std::vector<float*>& C_batch,
                                   size_t batch_size, size_t M, size_t N, size_t K) {
  #pragma omp parallel for schedule(dynamic, 1)
  for (size_t b = 0; b < batch_size; ++b) {
    optimized_gemm_small(A_batch[b], B, C_batch[b], M, N, K);
  }
}

void CachedSlimMoELayer::forwarding(nntrainer::RunLayerContext &context,
                                    bool training) {
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

  // routing - optimized with better memory access pattern
  nntrainer::Tensor &gate_weights = context.getWeight(gate_idx);
  
  // Use optimized matrix multiplication for routing
  if (total_tokens >= BATCH_SIZE_THRESHOLD) {
    input.dot(gate_weights, router_logits);
  } else {
    // For small batches, use custom optimized implementation
    optimized_gemm_small(input.getData<float>(), 
                        gate_weights.getData<float>(),
                        router_logits.getData<float>(),
                        total_tokens, num_experts, hidden_size);
  }
  
  router_logits.apply(nntrainer::ActiFunc::softmax<float>, router_logits);
  auto topk_result = router_logits.topK(topk);
  auto topk_values = std::get<0>(topk_result);
  auto topk_indices = std::get<1>(topk_result);

  const uint32_t *indices_data = topk_indices.getData<uint32_t>();
  
  // Optimized expert mask setting with better cache locality
  #pragma omp parallel for schedule(static) if(total_tokens > 32)
  for (int i = 0; i < static_cast<int>(total_tokens); ++i) {
    // Prefetch next cache line
    if (i + PREFETCH_DISTANCE < total_tokens) {
      __builtin_prefetch(&indices_data[(i + PREFETCH_DISTANCE) * topk], 0, 1);
    }
    
    for (int k = 0; k < static_cast<int>(topk); ++k) {
      expert_mask.setValue(indices_data[i * topk + k], 0, k, i, 1.0f);
    }
  }

  // Pre-compute expert token assignments with better memory layout
  std::vector<std::vector<std::pair<unsigned, float>>> expert_assignments(
    num_experts);
  
  // Reserve space to avoid reallocation
  for (auto& assignments : expert_assignments) {
    assignments.reserve(total_tokens * topk / num_experts + 1);
  }
  
  for (int i = 0; i < static_cast<int>(total_tokens); ++i) {
    for (int k = 0; k < static_cast<int>(topk); ++k) {
      unsigned expert_idx = indices_data[i * topk + k];
      float weight = topk_values.getValue<float>(i, 0, 0, k);
      expert_assignments[expert_idx].emplace_back(i, weight);
    }
  }

  // Process experts with better parallelization strategy
  std::vector<int> active_experts;
  active_experts.reserve(num_experts);
  
  for (int expert_idx = 0; expert_idx < static_cast<int>(num_experts); ++expert_idx) {
    if (!expert_assignments[expert_idx].empty()) {
      active_experts.push_back(expert_idx);
    }
  }

  // Parallel processing of active experts
  #pragma omp parallel for schedule(dynamic, 1) if(active_experts.size() > 2)
  for (size_t idx = 0; idx < active_experts.size(); ++idx) {
    int expert_idx = active_experts[idx];
    const auto &assignments = expert_assignments[expert_idx];

    ///@note load expert layer for the expert_idx
    nntrainer::Tensor expert_gate_proj =
      context.getWeight(expert_gate_proj_indices[expert_idx]);
    nntrainer::Tensor expert_up_proj =
      context.getWeight(expert_up_proj_indices[expert_idx]);
    nntrainer::Tensor expert_down_proj =
      context.getWeight(expert_down_proj_indices[expert_idx]);

    // Activate with prefetching hint
    expert_gate_proj.activate();
    expert_up_proj.activate();
    expert_down_proj.activate();

    // Use optimized expert forward computation
    compute_expert_forward_optimized(input, output, assignments, expert_gate_proj,
                                     expert_up_proj, expert_down_proj, hidden_size);

    // Deactivate to free memory
    expert_gate_proj.deactivate();
    expert_up_proj.deactivate();
    expert_down_proj.deactivate();
  }

  // reshape output: [B*S,1,1,H] -> [B,1,S,H]
  output.reshape({batch_size, 1, seq_len, hidden_size});
}

// Optimized expert forward computation with batching and SIMD
inline void CachedSlimMoELayer::compute_expert_forward_optimized(
  const nntrainer::Tensor &input, nntrainer::Tensor &output,
  const std::vector<std::pair<unsigned, float>> &token_assignments,
  const nntrainer::Tensor &gate_proj, const nntrainer::Tensor &up_proj,
  const nntrainer::Tensor &down_proj, unsigned int hidden_size) {

  const unsigned intermediate_size = gate_proj.width();
  const unsigned num_tokens = token_assignments.size();

  if (num_tokens == 0)
    return;

  // Batch processing for better cache utilization
  const size_t batch_size = std::min(size_t(8), num_tokens);
  
  // Process tokens in batches
  for (size_t batch_start = 0; batch_start < num_tokens; batch_start += batch_size) {
    size_t batch_end = std::min(batch_start + batch_size, num_tokens);
    size_t current_batch_size = batch_end - batch_start;
    
    // Allocate batch tensors
    std::vector<nntrainer::Tensor> batch_gate_out(current_batch_size);
    std::vector<nntrainer::Tensor> batch_up_out(current_batch_size);
    std::vector<nntrainer::Tensor> batch_acti_out(current_batch_size);
    
    nntrainer::TensorDim intermediate_dim({1, 1, 1, intermediate_size},
                                          input.getTensorType());
    nntrainer::TensorDim token_output_dim({1, 1, 1, hidden_size},
                                          input.getTensorType());
    
    // Initialize batch tensors
    for (size_t i = 0; i < current_batch_size; ++i) {
      batch_gate_out[i] = nntrainer::Tensor(intermediate_dim);
      batch_up_out[i] = nntrainer::Tensor(intermediate_dim);
      batch_acti_out[i] = nntrainer::Tensor(intermediate_dim);
    }
    
    // Batch matrix multiplication for gate and up projections
    #pragma omp parallel for schedule(static) if(current_batch_size > 2)
    for (size_t i = 0; i < current_batch_size; ++i) {
      const unsigned token_idx = token_assignments[batch_start + i].first;
      size_t token_offset = token_idx * hidden_size;
      
      nntrainer::TensorDim token_input_dim({1, 1, 1, hidden_size},
                                           input.getTensorType());
      nntrainer::Tensor token_input =
        input.getSharedDataTensor(token_input_dim, token_offset, true);
      
      // Prefetch next weights
      if (i + 1 < current_batch_size) {
        __builtin_prefetch(gate_proj.getData<float>(), 0, 3);
        __builtin_prefetch(up_proj.getData<float>(), 0, 3);
      }
      
      // Parallel computation of gate and up projections
      token_input.dot(gate_proj, batch_gate_out[i]);
      token_input.dot(up_proj, batch_up_out[i]);
      
      // Apply activation
      acti_func.run_fn(batch_gate_out[i], batch_acti_out[i]);
      
      // Element-wise multiply
      batch_acti_out[i].multiply_i(batch_up_out[i]);
    }
    
    // Batch down projection and output accumulation
    #pragma omp parallel for schedule(static) if(current_batch_size > 2)
    for (size_t i = 0; i < current_batch_size; ++i) {
      const unsigned token_idx = token_assignments[batch_start + i].first;
      const float weight = token_assignments[batch_start + i].second;
      
      nntrainer::Tensor token_expert_output(token_output_dim);
      batch_acti_out[i].dot(down_proj, token_expert_output);
      
      // Apply weight
      token_expert_output.multiply_i(weight);
      
      // Accumulate to output with atomic operations for thread safety
      size_t output_offset = token_idx * hidden_size;
      float* output_ptr = output.getData<float>() + output_offset;
      const float* expert_output_ptr = token_expert_output.getData<float>();
      
      // Use SIMD for accumulation
      size_t simd_width = 8;
      size_t simd_iterations = hidden_size / simd_width;
      
      for (size_t j = 0; j < simd_iterations; ++j) {
        __m256 out_vec = _mm256_loadu_ps(output_ptr + j * simd_width);
        __m256 expert_vec = _mm256_loadu_ps(expert_output_ptr + j * simd_width);
        out_vec = _mm256_add_ps(out_vec, expert_vec);
        _mm256_storeu_ps(output_ptr + j * simd_width, out_vec);
      }
      
      // Handle remaining elements
      for (size_t j = simd_iterations * simd_width; j < hidden_size; ++j) {
        #pragma omp atomic
        output_ptr[j] += expert_output_ptr[j];
      }
    }
  }
}

inline void CachedSlimMoELayer::compute_expert_forward(
  const nntrainer::Tensor &input, nntrainer::Tensor &output,
  const std::vector<std::pair<unsigned, float>> &token_assignments,
  const nntrainer::Tensor &gate_proj, const nntrainer::Tensor &up_proj,
  const nntrainer::Tensor &down_proj, unsigned int hidden_size) {
  
  // Fallback to optimized version
  compute_expert_forward_optimized(input, output, token_assignments,
                                   gate_proj, up_proj, down_proj, hidden_size);
}

inline void CachedSlimMoELayer::compute_expert_forward_no_critical(
  const nntrainer::Tensor &input, nntrainer::Tensor &expert_output,
  const std::vector<std::pair<unsigned, float>> &token_assignments,
  const nntrainer::Tensor &gate_proj, const nntrainer::Tensor &up_proj,
  const nntrainer::Tensor &down_proj, unsigned int hidden_size) {

  const unsigned intermediate_size = gate_proj.width();
  const unsigned num_tokens = token_assignments.size();

  if (num_tokens == 0)
    return;

  // Create tensor dimensions
  nntrainer::TensorDim token_input_dim({1, 1, 1, hidden_size},
                                       input.getTensorType());
  nntrainer::TensorDim intermediate_dim({1, 1, 1, intermediate_size},
                                        input.getTensorType());
  nntrainer::TensorDim token_output_dim({1, 1, 1, hidden_size},
                                        input.getTensorType());

  // Process tokens with better memory access pattern
  for (size_t i = 0; i < num_tokens; ++i) {
    const unsigned token_idx = token_assignments[i].first;
    const float weight = token_assignments[i].second;

    // Prefetch next token data
    if (i + 1 < num_tokens) {
      size_t next_offset = token_assignments[i + 1].first * hidden_size;
      __builtin_prefetch(input.getData<float>() + next_offset, 0, 1);
    }

    // Create shared tensor for input token (no memory copy)
    size_t token_offset = token_idx * hidden_size;
    nntrainer::Tensor token_input =
      input.getSharedDataTensor(token_input_dim, token_offset, true);

    // Use pre-allocated tensors from pool if available
    nntrainer::Tensor gate_out(intermediate_dim);
    nntrainer::Tensor acti_out(intermediate_dim);
    nntrainer::Tensor up_out(intermediate_dim);

    // Parallel computation of gate and up projections
    token_input.dot(gate_proj, gate_out);
    token_input.dot(up_proj, up_out);

    // Apply activation (silu)
    acti_func.run_fn(gate_out, acti_out);

    // Element-wise multiply: silu(gate_out) * up_out
    acti_out.multiply_i(up_out);

    // Down projection
    nntrainer::Tensor token_expert_output(token_output_dim);
    acti_out.dot(down_proj, token_expert_output);

    // Apply weight and accumulate
    token_expert_output.multiply_i(weight);
    size_t output_offset = token_idx * hidden_size;
    nntrainer::Tensor token_output =
      expert_output.getSharedDataTensor(token_output_dim, output_offset, true);

    token_output.add_i(token_expert_output);
  }
}

void CachedSlimMoELayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  if (from) {
    NNTR_THROW_IF(to - from != 1, std::invalid_argument)
      << "incremental step size is not 1";
    from = 0;
    to = 1;
  }
#ifdef DEBUG
  auto t1 = high_resolution_clock::now();
#endif

  nntrainer::Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &output_ = context.getOutput(SINGLE_INOUT_IDX);

  nntrainer::Tensor &router_logits_ = context.getTensor(router_logits_idx);

  nntrainer::TensorDim input_step_dim = input_.getDim();
  nntrainer::TensorDim output_step_dim = output_.getDim();
  nntrainer::TensorDim router_logits_step_dim = router_logits_.getDim();

  input_step_dim.batch(1);
  output_step_dim.batch(1);
  router_logits_step_dim.batch(to - from);

  input_step_dim.height(to - from);
  output_step_dim.height(to - from);

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

    // norm_topk_prob
    topk_values.divide_i(topk_values.sum(3));

    const uint32_t *indices_data = topk_indices.getData<uint32_t>();
    std::vector<std::vector<std::pair<unsigned, float>>> expert_assignments(
      num_experts);
    
    // Reserve space for better performance
    for (auto& assignments : expert_assignments) {
      assignments.reserve(total_tokens * topk / num_experts + 1);
    }
    
    // Set expert mask
    for (int i = 0; i < static_cast<int>(total_tokens); ++i) {
      for (int k = 0; k < static_cast<int>(topk); ++k) {
        unsigned expert_idx = indices_data[i * topk + k];
        float weight = topk_values.getValue<float>(i, 0, 0, k);
        expert_assignments[expert_idx].emplace_back(i, weight);
      }
    }

    // Collect active experts
    std::vector<int> target_idx_vector;
    target_idx_vector.reserve(num_experts);
    
    for (int expert_idx = 0; expert_idx < static_cast<int>(num_experts);
         ++expert_idx) {
      if (!expert_assignments[expert_idx].empty()) {
        target_idx_vector.push_back(expert_idx);
      }
    }
    
    // Optimized cache management with prefetching
    int hit_count = 0;
    int miss_count = 0;
    std::vector<int> missed_idx_vector;
    std::vector<int> hit_idx_vector;
    std::vector<int> evict_idx_vector;
    
    missed_idx_vector.reserve(target_idx_vector.size());
    hit_idx_vector.reserve(target_idx_vector.size());

    // Process cache hits and misses
    for (int expert_idx : target_idx_vector) {
      if (need_load[expert_idx]) {
        miss_count++;
        loaded_expert_deque.push_back(expert_idx);
        missed_idx_vector.push_back(expert_idx);
        iteration_map[expert_idx] = --loaded_expert_deque.end();
        need_load[expert_idx] = false;
      } else {
        hit_count++;
        hit_idx_vector.push_back(expert_idx);
        
        // LRU update - move to back
        if (iteration_map.find(expert_idx) != iteration_map.end()) {
          loaded_expert_deque.erase(iteration_map[expert_idx]);
        }
        loaded_expert_deque.push_back(expert_idx);
        iteration_map[expert_idx] = --loaded_expert_deque.end();
      }
    }

    // Pre-allocate expert outputs
    std::vector<nntrainer::Tensor> expert_outputs(num_experts);
    for (int expert_idx : target_idx_vector) {
      expert_outputs[expert_idx] = nntrainer::Tensor(
        total_tokens, 1, 1, hidden_size, output.getTensorType());
      expert_outputs[expert_idx].setZero();
    }

#ifdef DEBUG
    auto t1_hit = high_resolution_clock::now();
#endif

    // Process cache hits with optimized parallelization
    #pragma omp parallel for schedule(dynamic, 1) if(hit_idx_vector.size() > 2)
    for (size_t idx = 0; idx < hit_idx_vector.size(); ++idx) {
      int expert_idx = hit_idx_vector[idx];
      const auto &assignments = expert_assignments[expert_idx];

      compute_expert_forward_no_critical(
        input, expert_outputs[expert_idx], assignments,
        context.getWeight(expert_gate_proj_indices[expert_idx]),
        context.getWeight(expert_up_proj_indices[expert_idx]),
        context.getWeight(expert_down_proj_indices[expert_idx]), hidden_size);
    }

#ifdef DEBUG
    auto t2_hit = high_resolution_clock::now();
    auto t1_miss = high_resolution_clock::now();
#endif

    // Process cache misses with parallel loading and computation
    if (!missed_idx_vector.empty()) {
      // Parallel activation of missed experts
      #pragma omp parallel for schedule(static)
      for (size_t idx = 0; idx < missed_idx_vector.size(); ++idx) {
        int expert_idx = missed_idx_vector[idx];
        context.getWeight(expert_gate_proj_indices[expert_idx]).activate();
        context.getWeight(expert_up_proj_indices[expert_idx]).activate();
        context.getWeight(expert_down_proj_indices[expert_idx]).activate();
      }
      
      // Parallel computation for missed experts
      #pragma omp parallel for schedule(dynamic, 1) if(missed_idx_vector.size() > 2)
      for (size_t idx = 0; idx < missed_idx_vector.size(); ++idx) {
        int expert_idx = missed_idx_vector[idx];
        const auto &assignments = expert_assignments[expert_idx];
        
        compute_expert_forward_no_critical(
          input, expert_outputs[expert_idx], assignments,
          context.getWeight(expert_gate_proj_indices[expert_idx]),
          context.getWeight(expert_up_proj_indices[expert_idx]),
          context.getWeight(expert_down_proj_indices[expert_idx]), hidden_size);
      }
    }

#ifdef DEBUG
    auto t2_miss = high_resolution_clock::now();
#endif

    // Optimized eviction with batch deactivation
    while (loaded_expert_deque.size() > MAX_CACHED_EXPERTS) {
      int target_idx = loaded_expert_deque.front();
      loaded_expert_deque.pop_front();
      iteration_map.erase(target_idx);
      need_load[target_idx] = true;
      evict_idx_vector.push_back(target_idx);
    }

#ifdef DEBUG
    auto t1_evict = high_resolution_clock::now();
#endif

    // Batch deactivation for better performance
    if (!evict_idx_vector.empty()) {
      #pragma omp parallel for schedule(static)
      for (size_t idx = 0; idx < evict_idx_vector.size(); ++idx) {
        int target_idx = evict_idx_vector[idx];
        context.getWeight(expert_gate_proj_indices[target_idx]).deactivate();
        context.getWeight(expert_up_proj_indices[target_idx]).deactivate();
        context.getWeight(expert_down_proj_indices[target_idx]).deactivate();
      }
    }

#ifdef DEBUG
    auto t2_evict = high_resolution_clock::now();
#endif

    // Optimized output combination with SIMD
    bool first = true;
    for (int expert_idx : target_idx_vector) {
      if (first) {
        output.copyData(expert_outputs[expert_idx]);
        first = false;
      } else {
        // Use SIMD for fast addition
        float* output_ptr = output.getData<float>();
        const float* expert_ptr = expert_outputs[expert_idx].getData<float>();
        size_t total_size = total_tokens * hidden_size;
        
        size_t simd_width = 8;
        size_t simd_iterations = total_size / simd_width;
        
        #pragma omp parallel for schedule(static) if(total_size > 1024)
        for (size_t i = 0; i < simd_iterations; ++i) {
          __m256 out_vec = _mm256_loadu_ps(output_ptr + i * simd_width);
          __m256 expert_vec = _mm256_loadu_ps(expert_ptr + i * simd_width);
          out_vec = _mm256_add_ps(out_vec, expert_vec);
          _mm256_storeu_ps(output_ptr + i * simd_width, out_vec);
        }
        
        // Handle remaining elements
        for (size_t i = simd_iterations * simd_width; i < total_size; ++i) {
          output_ptr[i] += expert_ptr[i];
        }
      }
    }

    // reshape output: [B*S,1,1,H] -> [B,1,S,H]
    output.reshape({batch_size, 1, seq_len, hidden_size});

#ifdef DEBUG
    auto t2 = high_resolution_clock::now();
    auto dt = duration_cast<nanoseconds>(t2 - t1);
    auto dt_miss = duration_cast<nanoseconds>(t2_miss - t1_miss);
    auto dt_hit = duration_cast<nanoseconds>(t2_hit - t1_hit);
    auto dt_evict = duration_cast<nanoseconds>(t2_evict - t1_evict);
    std::cout << context.getName() << " \t| " << dt.count() << " ns "
              << "\t| " << dt.count() / 1'000 << " us "
              << "\t| " << dt.count() / 1'000'000 << " ms "
              << "\t| "
              << "hit ratio: " << hit_count / 8.0 << "\t | "
              << " miss ratio: " << miss_count / 8.0 << "\t | "
              << "hit_compute: " << dt_hit.count() / 1'000'000 << " ms "
              << "\t| "
              << "miss_compute: " << dt_miss.count() / 1'000'000 << " ms "
              << "\t| "
              << "evict_time: " << dt_evict.count() / 1'000'000 << " ms "
              << "\t| " << std::endl;
#endif
  }
}

void CachedSlimMoELayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, moe_props);
  nntrainer::LayerImpl::setProperty(remain_props);
}

void CachedSlimMoELayer::calcDerivative(nntrainer::RunLayerContext &context) {
  // MoE layer does not support derivative calculation
  throw std::runtime_error("MoE layer does not support derivative calculation");
}

void CachedSlimMoELayer::calcGradient(nntrainer::RunLayerContext &context) {
  // MoE layer does not support gradient calculation
  throw std::runtime_error("MoE layer does not support gradient calculation");
}

void CachedSlimMoELayer::exportTo(
  nntrainer::Exporter &exporter, const ml::train::ExportMethods &method) const {
  nntrainer::LayerImpl::exportTo(exporter, method);
  exporter.saveResult(moe_props, method, this); // Save MoE specific properties
}

} // namespace causallm