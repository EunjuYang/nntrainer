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
#include <algorithm>
#include <cmath>
#include <cstring>
#include <node_exporter.h>
#include <omp.h>
#include <qwen_moe_layer.h>
#include <stdexcept>

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

MoELayer::MoELayer() :
  LayerImpl(),
  num_experts(0),
  topk(0),
  moe_props(props::NumExperts(), props::NumExpertsPerToken(),
            nntrainer::props::Unit(), props::MoEActivation()),
  expert_gate_proj_indices({}

inline void MoELayer::optimize_expert_mask_and_assignments(
    nntrainer::Tensor &expert_mask,
    std::vector<std::vector<std::pair<unsigned, float>>> &expert_assignments,
    const uint32_t *indices_data, const float *values_data,
    unsigned int total_tokens, unsigned int topk, unsigned int num_experts) {
  
  float *mask_data = expert_mask.getData<float>();
  const auto mask_strides = expert_mask.getStrides();
  
  // Strategy 1: For very small token counts (common in incremental inference)
  if (total_tokens == 1) {
    // Optimized path for single token - parallelize topk loop if beneficial (typical: topk=8)
#pragma omp parallel for schedule(static) if(topk >= 8)
    for (int k = 0; k < static_cast<int>(topk); ++k) {
      const unsigned int expert_idx = indices_data[k];
      const float weight = values_data[k];
      
      // Direct offset calculation for single token
      const size_t mask_offset = expert_idx * mask_strides[0] + k * mask_strides[2];
      mask_data[mask_offset] = 1.0f;
      
      // Thread-safe assignment
#pragma omp critical
      {
        expert_assignments[expert_idx].emplace_back(0, weight);
      }
    }
    return;
  }
  
  // Strategy 2: For small token counts, process with limited parallelization
  if (total_tokens <= 4) {
    // Parallelize the nested loops if the total work is substantial
#pragma omp parallel for collapse(2) schedule(static) if(total_tokens * topk > 16)
    for (int i = 0; i < static_cast<int>(total_tokens); ++i) {
      for (int k = 0; k < static_cast<int>(topk); ++k) {
        const unsigned int idx = i * topk + k;
        const unsigned int expert_idx = indices_data[idx];
        const float weight = values_data[idx];
        
        const size_t mask_offset = expert_idx * mask_strides[0] + k * mask_strides[2] + i;
        mask_data[mask_offset] = 1.0f;
        
#pragma omp critical
        {
          expert_assignments[expert_idx].emplace_back(i, weight);
        }
      }
    }
    return;
  }
  
  // Strategy 3: For larger token counts, use optimized batching with parallelization
  // Group by expert to improve cache locality and enable vectorization
  std::vector<std::vector<std::pair<unsigned int, unsigned int>>> expert_positions(num_experts);
  
  // First pass: collect positions for each expert - parallelize this
#pragma omp parallel for collapse(2) schedule(static)
  for (int i = 0; i < static_cast<int>(total_tokens); ++i) {
    for (int k = 0; k < static_cast<int>(topk); ++k) {
      const unsigned int idx = i * topk + k;
      const unsigned int expert_idx = indices_data[idx];
      
      // Thread-safe position collection
#pragma omp critical
      {
        expert_positions[expert_idx].emplace_back(i, k);
      }
    }
  }
  
  // Second pass: process each expert's positions in batch - parallelize by expert
#pragma omp parallel for schedule(dynamic) if(num_experts > 4)
  for (int expert_idx = 0; expert_idx < static_cast<int>(num_experts); ++expert_idx) {
    const auto &positions = expert_positions[expert_idx];
    if (positions.empty()) continue;
    
    // Calculate base offset for this expert
    float *expert_mask_base = mask_data + expert_idx * mask_strides[0];
    
    // Process positions for this expert
    for (const auto &pos : positions) {
      const unsigned int i = pos.first;  // token index
      const unsigned int k = pos.second; // topk index
      const unsigned int idx = i * topk + k;
      const float weight = values_data[idx];
      
      // Set mask with optimized offset calculation
      const size_t local_offset = k * mask_strides[2] + i;
      expert_mask_base[local_offset] = 1.0f;
      
      // This is already expert-specific, so no critical section needed
      expert_assignments[expert_idx].emplace_back(i, weight);
    }
  }
}),
  expert_up_proj_indices({}),
  expert_down_proj_indices({}),
  gate_idx(std::numeric_limits<unsigned>::max()),
  router_logits_idx(std::numeric_limits<unsigned>::max()),
  expert_mask_idx(std::numeric_limits<unsigned>::max()),
  // Initialize optimization members
  cached_expert_assignments({}),
  active_experts({}),
  expert_workload({}) {}

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

  // 6. Request intermediate tensors
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

  // Pre-allocate expert assignments cache
  cached_expert_assignments.resize(num_experts);
  for (auto &assignments : cached_expert_assignments) {
    assignments.reserve(total_tokens); // Pre-allocate maximum possible size
  }

  // Pre-allocate optimized structures for common 8-expert case
  active_experts.reserve(16); // Reserve more than typical 8 for safety
  expert_workload.resize(num_experts);
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
  
  // Fast zero initialization
  std::memset(output.getData<float>(), 0, output.size() * sizeof(float));
  
  // Optimized expert mask initialization
  optimize_expert_mask_clear(expert_mask, total_tokens, topk);

  // routing
  nntrainer::Tensor &gate_weights = context.getWeight(gate_idx);
  input.dot(gate_weights, router_logits);
  router_logits.apply(nntrainer::ActiFunc::softmax<float>, router_logits);
  auto topk_result = router_logits.topK(topk);
  auto topk_values = std::get<0>(topk_result);
  auto topk_indices = std::get<1>(topk_result);

  // Cache data pointers for better performance
  const uint32_t *indices_data = topk_indices.getData<uint32_t>();
  const float *values_data = topk_values.getData<float>();

  // Clear and rebuild expert assignments with optimized access
  // Parallelize the clearing if we have many experts
#pragma omp parallel for schedule(static) if(num_experts > 8)
  for (int expert_idx = 0; expert_idx < static_cast<int>(num_experts); ++expert_idx) {
    cached_expert_assignments[expert_idx].clear();
  }
  active_experts.clear();

  // Optimized expert mask setting and assignment building with 8-expert optimization
  optimize_expert_mask_and_assignments_8expert(
    expert_mask, cached_expert_assignments, active_experts,
    indices_data, values_data, total_tokens, topk, num_experts);

  // Choose processing strategy based on active expert count
  const unsigned int active_count = active_experts.size();
  
  // Always use optimized active-expert-only processing
  // The distinction is now mainly about scheduling strategy
  process_experts_optimized(input, output, context, hidden_size, active_count);

  // reshape output: [B*S,1,1,H] -> [B,1,S,H]
  output.reshape({batch_size, 1, seq_len, hidden_size});
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

  // Pre-clear cached expert assignments for reuse (fast clear without deallocation)
  for (auto &assignments : cached_expert_assignments) {
    assignments.clear();
  }

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
    
    // Fast zero initialization using memset for better performance
    std::memset(output.getData<float>(), 0, output.size() * sizeof(float));
    std::memset(expert_mask.getData<float>(), 0, expert_mask.size() * sizeof(float));

    // routing
    nntrainer::Tensor &gate_weights = context.getWeight(gate_idx);
    input.dot(gate_weights, router_logits);
    router_logits.apply(nntrainer::ActiFunc::softmax<float>, router_logits);
    auto topk_result = router_logits.topK(topk);
    auto topk_values = std::get<0>(topk_result);
    auto topk_indices = std::get<1>(topk_result);

    // norm_topk_prob - keep original normalization method
    topk_values.divide_i(topk_values.sum(3));

    // Cache frequently accessed data pointers for better performance
    const uint32_t *indices_data = topk_indices.getData<uint32_t>();
    const float *values_data = topk_values.getData<float>();
    float *mask_data = expert_mask.getData<float>();
    
    // Get expert mask dimensions for direct indexing
    const auto mask_strides = expert_mask.getStrides();

    // Optimized single loop for both mask setting and assignment building
    // This reduces memory access and improves cache locality
    for (unsigned int i = 0; i < total_tokens; ++i) {
      for (unsigned int k = 0; k < topk; ++k) {
        const unsigned int idx = i * topk + k;
        const unsigned int expert_idx = indices_data[idx];
        const float weight = values_data[idx];
        
        // Direct memory access instead of setValue for better performance
        const size_t mask_offset = expert_idx * mask_strides[0] + k * mask_strides[2] + i;
        mask_data[mask_offset] = 1.0f;
        
        // Build expert assignments
        cached_expert_assignments[expert_idx].emplace_back(i, weight);
      }
    }

    // Prefetch weight data for better cache performance (compiler-specific optimization)
    // This helps when we have many experts
#ifdef __GNUC__
    if (num_experts > 4) {
      for (unsigned int expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
        if (!cached_expert_assignments[expert_idx].empty()) {
          __builtin_prefetch(context.getWeight(expert_gate_proj_indices[expert_idx]).getData<float>(), 0, 3);
          __builtin_prefetch(context.getWeight(expert_up_proj_indices[expert_idx]).getData<float>(), 0, 3);
          __builtin_prefetch(context.getWeight(expert_down_proj_indices[expert_idx]).getData<float>(), 0, 3);
        }
      }
    }
#endif

    // expert forwarding with optimized memory access
#pragma omp parallel for schedule(dynamic)
    for (int expert_idx = 0; expert_idx < static_cast<int>(num_experts); ++expert_idx) {
      const auto &assignments = cached_expert_assignments[expert_idx];
      if (assignments.empty())
        continue;

      // Use same computation as forwarding - maintain consistency
      compute_expert_forward_fast(
        input, output, assignments,
        context.getWeight(expert_gate_proj_indices[expert_idx]),
        context.getWeight(expert_up_proj_indices[expert_idx]),
        context.getWeight(expert_down_proj_indices[expert_idx]), hidden_size);
    }

    // reshape output: [B*S,1,1,H] -> [B,1,S,H]
    output.reshape({batch_size, 1, seq_len, hidden_size});
    
    // Fast clear for next iteration (no deallocation)
    for (auto &assignments : cached_expert_assignments) {
      assignments.clear();
    }
  }
}

inline void MoELayer::compute_expert_forward(
  const nntrainer::Tensor &input, nntrainer::Tensor &output,
  const std::vector<std::pair<unsigned, float>> &token_assignments,
  const nntrainer::Tensor &gate_proj, const nntrainer::Tensor &up_proj,
  const nntrainer::Tensor &down_proj, unsigned int hidden_size) {

  const unsigned intermediate_size = gate_proj.width();
  const unsigned num_tokens = token_assignments.size();

  if (num_tokens == 0)
    return;

  // Create tensor dimensions for single token processing
  nntrainer::TensorDim token_input_dim({1, 1, 1, hidden_size},
                                       input.getTensorType());
  nntrainer::TensorDim intermediate_dim({1, 1, 1, intermediate_size},
                                        input.getTensorType());
  nntrainer::TensorDim token_output_dim({1, 1, 1, hidden_size},
                                        input.getTensorType());

  // Process each token individually to avoid memory copies
  for (size_t i = 0; i < num_tokens; ++i) {
    const unsigned token_idx = token_assignments[i].first;
    const float weight = token_assignments[i].second;

    // Create shared tensor for input token (no memory copy)
    size_t token_offset = token_idx * hidden_size;
    nntrainer::Tensor token_input =
      input.getSharedDataTensor(token_input_dim, token_offset, true);

    // Create intermediate tensors for this token
    nntrainer::Tensor gate_out(intermediate_dim);
    nntrainer::Tensor acti_out(intermediate_dim);
    nntrainer::Tensor up_out(intermediate_dim);

    // Gate projection using optimized dot operation
    token_input.dot(gate_proj, gate_out);

    // Apply activation (silu)
    acti_func.run_fn(gate_out, acti_out);

    // Up projection using optimized dot operation
    token_input.dot(up_proj, up_out);

    // Element-wise multiply: silu(gate_out) * up_out
    acti_out.multiply_i(up_out);

    // Down projection using optimized dot operation
    nntrainer::Tensor token_expert_output(token_output_dim);
    acti_out.dot(down_proj, token_expert_output);

    // Apply weight and accumulate to final output using shared tensor
    size_t output_offset = token_idx * hidden_size;
    nntrainer::Tensor token_output =
      output.getSharedDataTensor(token_output_dim, output_offset, true);

    // Scale by weight and accumulate
    token_expert_output.multiply_i(weight);
    token_output.add_i(token_expert_output);
  }
}

inline void MoELayer::compute_expert_forward_fast(
  const nntrainer::Tensor &input, nntrainer::Tensor &output,
  const std::vector<std::pair<unsigned, float>> &token_assignments,
  const nntrainer::Tensor &gate_proj, const nntrainer::Tensor &up_proj,
  const nntrainer::Tensor &down_proj, unsigned int hidden_size) {

  const unsigned intermediate_size = gate_proj.width();
  const unsigned num_tokens = token_assignments.size();

  if (num_tokens == 0)
    return;

  // Cache frequently accessed pointers
  const float *input_data = input.getData<float>();
  float *output_data = output.getData<float>();

  // Fast path for single token (common in incremental inference)
  if (num_tokens == 1) {
    const unsigned token_idx = token_assignments[0].first;
    const float weight = token_assignments[0].second;
    
    const size_t input_offset = token_idx * hidden_size;
    const size_t output_offset = token_idx * hidden_size;

    // Create tensor views for single token (lightweight, no memory copy)
    nntrainer::TensorDim token_input_dim({1, 1, 1, hidden_size}, input.getTensorType());
    nntrainer::TensorDim intermediate_dim({1, 1, 1, intermediate_size}, input.getTensorType());
    nntrainer::TensorDim token_output_dim({1, 1, 1, hidden_size}, input.getTensorType());

    nntrainer::Tensor token_input = input.getSharedDataTensor(token_input_dim, input_offset, true);
    nntrainer::Tensor gate_out(intermediate_dim);
    nntrainer::Tensor up_out(intermediate_dim);
    nntrainer::Tensor token_expert_output(token_output_dim);

    // Parallel execution of independent gate and up projections
    // This is the key optimization: gate and up projections are independent and can run in parallel
    // Only parallelize if the computation is substantial enough to justify OpenMP overhead
#pragma omp parallel sections if(intermediate_size * hidden_size > 50000)
    {
#pragma omp section
      {
        // Gate projection
        token_input.dot(gate_proj, gate_out);
        // Apply activation (silu) in-place
        acti_func.run_fn(gate_out, gate_out);
      }
#pragma omp section
      {
        // Up projection (independent of gate)
        token_input.dot(up_proj, up_out);
      }
    }
    
    // Element-wise multiply: silu(gate_out) * up_out
    gate_out.multiply_i(up_out);
    
    // Down projection
    gate_out.dot(down_proj, token_expert_output);
    
    // Fast accumulation with weight
    const float *expert_output_data = token_expert_output.getData<float>();
    float * const output_ptr = output_data + output_offset;
    
    // Vectorized accumulation
    for (unsigned int i = 0; i < hidden_size; ++i) {
      output_ptr[i] += expert_output_data[i] * weight;
    }
  } else {
    // Batch processing for multiple tokens with parallel gate/up projections
    // Pre-allocate intermediate tensors to avoid repeated allocation
    thread_local std::vector<nntrainer::Tensor> gate_outs, up_outs, token_expert_outputs;
    thread_local std::vector<nntrainer::Tensor> token_inputs;
    
    // Resize thread-local vectors if needed
    if (gate_outs.size() < num_tokens) {
      gate_outs.clear();
      up_outs.clear();
      token_expert_outputs.clear();
      token_inputs.clear();
      
      nntrainer::TensorDim token_input_dim({1, 1, 1, hidden_size}, input.getTensorType());
      nntrainer::TensorDim intermediate_dim({1, 1, 1, intermediate_size}, input.getTensorType());
      nntrainer::TensorDim token_output_dim({1, 1, 1, hidden_size}, input.getTensorType());
      
      for (size_t i = 0; i < num_tokens; ++i) {
        gate_outs.emplace_back(intermediate_dim);
        up_outs.emplace_back(intermediate_dim);
        token_expert_outputs.emplace_back(token_output_dim);
      }
    }

    // Process all tokens with parallel gate/up projections
    for (size_t i = 0; i < num_tokens; ++i) {
      const unsigned token_idx = token_assignments[i].first;
      const float weight = token_assignments[i].second;
      
      size_t token_offset = token_idx * hidden_size;
      nntrainer::TensorDim token_input_dim({1, 1, 1, hidden_size}, input.getTensorType());
      nntrainer::Tensor token_input = input.getSharedDataTensor(token_input_dim, token_offset, true);

      // Parallel gate and up projections for this token
#pragma omp parallel sections if(intermediate_size * hidden_size > 50000)
      {
#pragma omp section
        {
          // Gate projection and activation
          token_input.dot(gate_proj, gate_outs[i]);
          acti_func.run_fn(gate_outs[i], gate_outs[i]);
        }
#pragma omp section
        {
          // Up projection (independent)
          token_input.dot(up_proj, up_outs[i]);
        }
      }
      
      // Element-wise multiply and down projection
      gate_outs[i].multiply_i(up_outs[i]);
      gate_outs[i].dot(down_proj, token_expert_outputs[i]);
      
      // Accumulate to output with weight
      size_t output_offset = token_idx * hidden_size;
      const float *expert_output_data = token_expert_outputs[i].getData<float>();
      float * const output_ptr = output_data + output_offset;
      
      for (unsigned int j = 0; j < hidden_size; ++j) {
        output_ptr[j] += expert_output_data[j] * weight;
      }
    }
  }
}

inline void MoELayer::optimize_expert_mask_clear(nntrainer::Tensor &expert_mask, 
                                                 unsigned int total_tokens, 
                                                 unsigned int topk) {
  // Instead of clearing entire mask, only clear the portion that will be used
  // This reduces memory bandwidth significantly for small token counts
  
  const size_t used_elements = static_cast<size_t>(num_experts) * topk * total_tokens;
  const size_t total_elements = expert_mask.size();
  
  float *mask_data = expert_mask.getData<float>();
  
  if (used_elements < total_elements / 4) {
    // For small usage, selectively clear only used regions
    // Clear by expert to maintain cache locality - parallelize expert loop
    const auto mask_strides = expert_mask.getStrides();
    
#pragma omp parallel for schedule(static) if(num_experts > 4)
    for (int expert_idx = 0; expert_idx < static_cast<int>(num_experts); ++expert_idx) {
      float *expert_start = mask_data + expert_idx * mask_strides[0];
      const size_t expert_used_size = topk * total_tokens * sizeof(float);
      std::memset(expert_start, 0, expert_used_size);
    }
  } else {
    // For large usage, clear entire mask
    std::memset(mask_data, 0, total_elements * sizeof(float));
  }
}

inline void MoELayer::optimize_expert_mask_and_assignments(
    nntrainer::Tensor &expert_mask,
    std::vector<std::vector<std::pair<unsigned, float>>> &expert_assignments,
    const uint32_t *indices_data, const float *values_data,
    unsigned int total_tokens, unsigned int topk, unsigned int num_experts) {
  
  float *mask_data = expert_mask.getData<float>();
  const auto mask_strides = expert_mask.getStrides();
  
  // Strategy 1: For very small token counts (common in incremental inference)
  if (total_tokens == 1) {
    // Optimized path for single token
    for (unsigned int k = 0; k < topk; ++k) {
      const unsigned int expert_idx = indices_data[k];
      const float weight = values_data[k];
      
      // Direct offset calculation for single token
      const size_t mask_offset = expert_idx * mask_strides[0] + k * mask_strides[2];
      mask_data[mask_offset] = 1.0f;
      
      expert_assignments[expert_idx].emplace_back(0, weight);
    }
    return;
  }
  
  // Strategy 2: For small token counts, process sequentially
  if (total_tokens <= 4) {
    for (unsigned int i = 0; i < total_tokens; ++i) {
      for (unsigned int k = 0; k < topk; ++k) {
        const unsigned int idx = i * topk + k;
        const unsigned int expert_idx = indices_data[idx];
        const float weight = values_data[idx];
        
        const size_t mask_offset = expert_idx * mask_strides[0] + k * mask_strides[2] + i;
        mask_data[mask_offset] = 1.0f;
        
        expert_assignments[expert_idx].emplace_back(i, weight);
      }
    }
    return;
  }
  
  // Strategy 3: For larger token counts, use optimized batching
  // Group by expert to improve cache locality and enable vectorization
  std::vector<std::vector<std::pair<unsigned int, unsigned int>>> expert_positions(num_experts);
  
  // First pass: collect positions for each expert
  for (unsigned int i = 0; i < total_tokens; ++i) {
    for (unsigned int k = 0; k < topk; ++k) {
      const unsigned int idx = i * topk + k;
      const unsigned int expert_idx = indices_data[idx];
      expert_positions[expert_idx].emplace_back(i, k);
    }
  }
  
  // Second pass: process each expert's positions in batch
  for (unsigned int expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
    const auto &positions = expert_positions[expert_idx];
    if (positions.empty()) continue;
    
    // Calculate base offset for this expert
    float *expert_mask_base = mask_data + expert_idx * mask_strides[0];
    
    // Process positions for this expert
    for (const auto &pos : positions) {
      const unsigned int i = pos.first;  // token index
      const unsigned int k = pos.second; // topk index
      const unsigned int idx = i * topk + k;
      const float weight = values_data[idx];
      
      // Set mask with optimized offset calculation
      const size_t local_offset = k * mask_strides[2] + i;
      expert_mask_base[local_offset] = 1.0f;
      
      expert_assignments[expert_idx].emplace_back(i, weight);
    }
  }
}

inline void MoELayer::optimize_expert_mask_and_assignments_8expert(
    nntrainer::Tensor &expert_mask,
    std::vector<std::vector<std::pair<unsigned, float>>> &expert_assignments,
    std::vector<unsigned int> &active_experts,
    const uint32_t *indices_data, const float *values_data,
    unsigned int total_tokens, unsigned int topk, unsigned int num_experts) {
  
  float *mask_data = expert_mask.getData<float>();
  const auto mask_strides = expert_mask.getStrides();
  
  // Clear workload counters
  std::fill(expert_workload.begin(), expert_workload.end(), 0);
  
  // Strategy optimized for typical 8-expert case in incremental inference
  if (total_tokens == 1) {
    // Single token optimization - most common case
    // Parallelize topk loop if topk is substantial (typical case: topk=8)
#pragma omp parallel for schedule(static) if(topk >= 8)
    for (int k = 0; k < static_cast<int>(topk); ++k) {
      const unsigned int expert_idx = indices_data[k];
      const float weight = values_data[k];
      
      // Use atomic operations for thread safety when parallel
      bool was_zero = false;
#pragma omp critical
      {
        if (expert_workload[expert_idx] == 0) {
          active_experts.push_back(expert_idx);
          was_zero = true;
        }
        expert_workload[expert_idx]++;
      }
      
      // Direct offset calculation for single token
      const size_t mask_offset = expert_idx * mask_strides[0] + k * mask_strides[2];
      mask_data[mask_offset] = 1.0f;
      
#pragma omp critical
      {
        expert_assignments[expert_idx].emplace_back(0, weight);
      }
    }
    return;
  }
  
  // Multi-token case - build active expert list first with parallelization
  // Parallelize the outer token loop for better scalability
#pragma omp parallel for collapse(2) schedule(static) if(total_tokens * topk > 64)
  for (int i = 0; i < static_cast<int>(total_tokens); ++i) {
    for (int k = 0; k < static_cast<int>(topk); ++k) {
      const unsigned int idx = i * topk + k;
      const unsigned int expert_idx = indices_data[idx];
      
#pragma omp critical
      {
        if (expert_workload[expert_idx] == 0) {
          active_experts.push_back(expert_idx);
        }
        expert_workload[expert_idx]++;
      }
    }
  }
  
  // Sort active experts for better cache locality if we have many
  if (active_experts.size() > 8) {
    std::sort(active_experts.begin(), active_experts.end());
  }
  
  // Process assignments and mask setting with parallelization
  // Use collapse(2) to parallelize both token and topk dimensions
#pragma omp parallel for collapse(2) schedule(static) if(total_tokens * topk > 32)
  for (int i = 0; i < static_cast<int>(total_tokens); ++i) {
    for (int k = 0; k < static_cast<int>(topk); ++k) {
      const unsigned int idx = i * topk + k;
      const unsigned int expert_idx = indices_data[idx];
      const float weight = values_data[idx];
      
      const size_t mask_offset = expert_idx * mask_strides[0] + k * mask_strides[2] + i;
      mask_data[mask_offset] = 1.0f;
      
      // Thread-safe assignment building
#pragma omp critical
      {
        expert_assignments[expert_idx].emplace_back(i, weight);
      }
    }
  }
}

inline void MoELayer::process_experts_optimized(
    const nntrainer::Tensor &input, nntrainer::Tensor &output,
    nntrainer::RunLayerContext &context, unsigned int hidden_size, 
    unsigned int active_count) {
  
  // Prefetch weights for active experts only (much more efficient than prefetching all)
#ifdef __GNUC__
  for (unsigned int i = 0; i < active_count; ++i) {
    const unsigned int expert_idx = active_experts[i];
    __builtin_prefetch(context.getWeight(expert_gate_proj_indices[expert_idx]).getData<float>(), 0, 3);
    __builtin_prefetch(context.getWeight(expert_up_proj_indices[expert_idx]).getData<float>(), 0, 3);
    __builtin_prefetch(context.getWeight(expert_down_proj_indices[expert_idx]).getData<float>(), 0, 3);
  }
#endif

  // Expert-level parallelism is generally more efficient than intra-expert parallelism
  // because experts are completely independent and each expert has substantial computational cost
  
  if (active_count == 1) {
    // Single expert: Focus on intra-expert optimization since no expert-level parallelism possible
    // The single expert still has substantial computation (multiple matrix multiplications)
    const unsigned int expert_idx = active_experts[0];
    const auto &assignments = cached_expert_assignments[expert_idx];
    
    if (!assignments.empty()) {
      // Use the intra-expert parallelized version for single expert
      compute_expert_forward_fast(
        input, output, assignments,
        context.getWeight(expert_gate_proj_indices[expert_idx]),
        context.getWeight(expert_up_proj_indices[expert_idx]),
        context.getWeight(expert_down_proj_indices[expert_idx]), hidden_size);
    }
  } else {
    // Multiple experts: Expert-level parallelism is the priority
    // Each expert's work is substantial enough to justify parallel processing
    // Use dynamic scheduling for better load balancing across experts
    
#pragma omp parallel for schedule(dynamic, 1) 
    for (int i = 0; i < static_cast<int>(active_count); ++i) {
      const unsigned int expert_idx = active_experts[i];
      const auto &assignments = cached_expert_assignments[expert_idx];
      
      if (!assignments.empty()) {
        // Note: In expert-level parallelism, we might want to disable intra-expert parallelism
        // to avoid nested parallelism overhead, but compute_expert_forward_fast
        // uses conditional parallelism based on problem size
        compute_expert_forward_fast(
          input, output, assignments,
          context.getWeight(expert_gate_proj_indices[expert_idx]),
          context.getWeight(expert_up_proj_indices[expert_idx]),
          context.getWeight(expert_down_proj_indices[expert_idx]), hidden_size);
      }
    }
  }
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
