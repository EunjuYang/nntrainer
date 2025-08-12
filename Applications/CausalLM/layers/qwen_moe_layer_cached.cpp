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
 * @file	qwen_moe_layer_fsu.cpp
 * @date	09 June 2025
 * @brief	This is a Mixture of Expert Layer Class for Neural Network
 * @see		https://github.com/nnstreamer/
 * @author	Eunju Yang <ej.yang@samsung.com>
 * @bug		No known bugs except for NYI items
 * @note    MoE layer with on-the-fly expert FSU
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
#include <thread>
#include <future>
#include <chrono>

#ifdef __ANDROID__
#include <sys/mman.h>
#include <unistd.h>
#include <cstdlib>
#endif

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

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
  expert_mask_idx(std::numeric_limits<unsigned>::max()) {
#ifdef __ANDROID__
  // Android에서 캐시 크기를 환경 변수로 설정 가능하게 함
  const char* cache_size_env = std::getenv("NNTRAINER_MOE_CACHE_SIZE");
  if (cache_size_env) {
    max_cached_experts = std::stoi(cache_size_env);
  } else {
    max_cached_experts = 16; // 기본값
  }
  
  // Prefetch 활성화 여부
  const char* prefetch_env = std::getenv("NNTRAINER_MOE_PREFETCH");
  enable_prefetch = prefetch_env && std::string(prefetch_env) == "1";
  
  // Async deactivation 활성화 여부
  const char* async_deact_env = std::getenv("NNTRAINER_ASYNC_DEACTIVATE");
  enable_async_deactivation = async_deact_env && std::string(async_deact_env) == "1";
#else
  max_cached_experts = 16;
  enable_prefetch = false;
  enable_async_deactivation = false;
#endif

  // Start background deactivation thread only if enabled
  if (enable_async_deactivation) {
    deactivation_thread = std::thread([this]() {
      while (!deactivation_thread_stop.load()) {
        std::unique_lock<std::mutex> lock(deactivation_mutex);
        deactivation_cv.wait(lock, [this] { 
          return !deactivation_queue.empty() || deactivation_thread_stop.load(); 
        });
        
        while (!deactivation_queue.empty()) {
          auto [expert_idx, context_ptr] = deactivation_queue.front();
          deactivation_queue.pop();
          lock.unlock();
          
          // Perform deactivation outside of lock
          if (context_ptr) {
            context_ptr->getWeight(expert_gate_proj_indices[expert_idx]).deactivate();
            context_ptr->getWeight(expert_up_proj_indices[expert_idx]).deactivate();
            context_ptr->getWeight(expert_down_proj_indices[expert_idx]).deactivate();
          }
          
          lock.lock();
        }
      }
    });
  }
}

CachedSlimMoELayer::~CachedSlimMoELayer() {
  // Stop and join the deactivation thread
  deactivation_thread_stop.store(true);
  deactivation_cv.notify_all();
  if (deactivation_thread.joinable()) {
    deactivation_thread.join();
  }
}

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

  // 4. Initialie gate layer (router)
  nntrainer::TensorDim gate_dim(
    1, is_nchw ? 1 : num_experts, is_nchw ? hidden_size : 1,
    is_nchw ? num_experts : hidden_size,
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     nntrainer::TensorDim::DataType::FP32),
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
  
  // Initialize expert history tracking for prefetching
  expert_history.resize(num_experts);

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
                          
#ifdef __ANDROID__
  // Android-specific: Pre-allocate and pin memory for frequently used experts
  const char* pin_memory_env = std::getenv("NNTRAINER_PIN_EXPERT_MEMORY");
  if (pin_memory_env && std::string(pin_memory_env) == "1") {
    // Pin the first few experts in memory to reduce page faults
    int num_to_pin = std::min(4u, num_experts);
    for (int i = 0; i < num_to_pin; ++i) {
      // These will be pinned when first activated
      // Mark them for special handling
      // This is a hint to the system that these are high-priority
    }
  }
#endif
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
  output.setZero();

  // routing
  nntrainer::Tensor &gate_weights = context.getWeight(gate_idx);
  input.dot(gate_weights, router_logits);
  router_logits.apply(nntrainer::ActiFunc::softmax<float>, router_logits);
  auto topk_result = router_logits.topK(topk);
  auto topk_values = std::get<0>(topk_result);
  auto topk_indices = std::get<1>(topk_result);

  const uint32_t *indices_data = topk_indices.getData<uint32_t>();
#pragma omp parallel for collapse(2)
  for (int i = 0; i < static_cast<int>(total_tokens); ++i) {
    for (int k = 0; k < static_cast<int>(topk); ++k) {
      expert_mask.setValue(indices_data[i * topk + k], 0, k, i, 1.0f);
    }
  }

  // Pre-compute expert token assignments for better cache locality
  std::vector<std::vector<std::pair<unsigned, float>>> expert_assignments(
    num_experts);
  for (int i = 0; i < static_cast<int>(total_tokens); ++i) {
    for (int k = 0; k < static_cast<int>(topk); ++k) {
      unsigned expert_idx = indices_data[i * topk + k];
      float weight = topk_values.getValue<float>(i, 0, 0, k);
      expert_assignments[expert_idx].emplace_back(i, weight);
    }
  }

  for (int expert_idx = 0; expert_idx < static_cast<int>(num_experts);
       ++expert_idx) {
    const auto &assignments = expert_assignments[expert_idx];
    if (assignments.empty())
      continue;

    ///@note load expert layer for the expert_idx
    nntrainer::Tensor expert_gate_proj =
      context.getWeight(expert_gate_proj_indices[expert_idx]);
    nntrainer::Tensor expert_up_proj =
      context.getWeight(expert_up_proj_indices[expert_idx]);
    nntrainer::Tensor expert_down_proj =
      context.getWeight(expert_down_proj_indices[expert_idx]);

    ///@note Please note that expert_gate_proj is virtual tensor,
    ///      which is not allocated so far. It will be allocated when it is
    ///      used. `activate(read=true)` will allocate its memory and will read
    ///      from the original weight. activate is true by default. i.e., mmap
    expert_gate_proj.activate();
    expert_up_proj.activate();
    expert_down_proj.activate();

    // Use optimized expert forward computation without memory copies
    compute_expert_forward(input, output, assignments, expert_gate_proj,
                           expert_up_proj, expert_down_proj, hidden_size);

    ////@note Please note that the virtual tensor is deactivated after usage
    ////      This will allocate and load data from the storage on-the-fly
    ////      i.e., unmap
    expert_gate_proj.deactivate();
    expert_up_proj.deactivate();
    expert_down_proj.deactivate();
  }

  // reshape output: [B*S,1,1,H] -> [B,1,S,H]
  output.reshape({batch_size, 1, seq_len, hidden_size});
}

inline void CachedSlimMoELayer::compute_expert_forward(
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

  // Create a temporary output tensor for this expert to avoid critical section
  nntrainer::Tensor expert_output(output.batch(), output.channel(),
                                  output.height(), output.width(),
                                  output.getTensorType());
  expert_output.setZero();

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

    // Apply weight and accumulate to expert's temporary output
    token_expert_output.multiply_i(weight);
    size_t output_offset = token_idx * hidden_size;
    nntrainer::Tensor token_output =
      expert_output.getSharedDataTensor(token_output_dim, output_offset, true);

    token_output.add_i(token_expert_output);
  }

  // Add expert's result to final output (no critical section in sequential
  // mode)
  output.add_i(expert_output);
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

    // Apply weight and accumulate to expert's output (no critical section
    // needed)
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
    // Set expert mask
    for (int i = 0; i < static_cast<int>(total_tokens); ++i) {
      for (int k = 0; k < static_cast<int>(topk); ++k) {
        unsigned expert_idx = indices_data[i * topk + k];
        float weight = topk_values.getValue<float>(i, 0, 0, k);
        expert_assignments[expert_idx].emplace_back(i, weight);
      }
    }

    // Prepare expert outputs
    std::vector<nntrainer::Tensor> expert_outputs(num_experts);
    for (int expert_idx = 0; expert_idx < static_cast<int>(num_experts);
         ++expert_idx) {
      if (!expert_assignments[expert_idx].empty()) {
        expert_outputs[expert_idx] = nntrainer::Tensor(
          total_tokens, 1, 1, hidden_size, output.getTensorType());
        expert_outputs[expert_idx].setZero();
      }
    }
    
    // Collect active experts and categorize them
    std::vector<int> active_experts;
    std::vector<int> loaded_experts;
    std::vector<int> experts_to_load;
    
    for (int expert_idx = 0; expert_idx < static_cast<int>(num_experts);
         ++expert_idx) {
      if (!expert_assignments[expert_idx].empty()) {
        active_experts.push_back(expert_idx);
        if (!need_load[expert_idx]) {
          loaded_experts.push_back(expert_idx);
        } else {
          experts_to_load.push_back(expert_idx);
        }
      }
    }
    
    // OPTIMIZATION: True overlap of loading and computation
    // Strategy: Load expert N+1 while computing expert N
    
    // Step 1: Process already loaded experts in parallel (no I/O needed)
    #pragma omp parallel for schedule(dynamic) if(loaded_experts.size() > 1)
    for (size_t i = 0; i < loaded_experts.size(); ++i) {
      int expert_idx = loaded_experts[i];
      const auto &assignments = expert_assignments[expert_idx];
      
      compute_expert_forward_no_critical(
        input, expert_outputs[expert_idx], assignments,
        context.getWeight(expert_gate_proj_indices[expert_idx]),
        context.getWeight(expert_up_proj_indices[expert_idx]),
        context.getWeight(expert_down_proj_indices[expert_idx]), hidden_size);
    }
    
    // Step 2: Pipeline loading and computation for new experts
    if (!experts_to_load.empty()) {
      // Load first expert synchronously (can't avoid this)
      int first_expert = experts_to_load[0];
      context.getWeight(expert_gate_proj_indices[first_expert]).activate();
      context.getWeight(expert_up_proj_indices[first_expert]).activate();
      context.getWeight(expert_down_proj_indices[first_expert]).activate();
      
      loaded_expert_deque.push_back(first_expert);
      iteration_map[first_expert] = --loaded_expert_deque.end();
      need_load[first_expert] = false;
      
      // Process remaining experts with pipelining
      for (size_t i = 0; i < experts_to_load.size(); ++i) {
        int current_expert = experts_to_load[i];
        
        // Start async loading of next expert (if exists)
        std::future<void> next_load_future;
        int next_expert = -1;
        if (i + 1 < experts_to_load.size()) {
          next_expert = experts_to_load[i + 1];
          // Launch async loading of next expert
          next_load_future = std::async(std::launch::async, [&context, this, next_expert]() {
            context.getWeight(expert_gate_proj_indices[next_expert]).activate();
            context.getWeight(expert_up_proj_indices[next_expert]).activate();
            context.getWeight(expert_down_proj_indices[next_expert]).activate();
          });
        }
        
        // AGGRESSIVE PREFETCH: Also start loading expert i+2 if available
        std::future<void> next_next_load_future;
        int next_next_expert = -1;
        if (enable_prefetch && i + 2 < experts_to_load.size()) {
          next_next_expert = experts_to_load[i + 2];
          next_next_load_future = std::async(std::launch::async, [&context, this, next_next_expert]() {
            // Lower priority prefetch - just trigger mmap, don't wait
            context.getWeight(expert_gate_proj_indices[next_next_expert]).activate();
            context.getWeight(expert_up_proj_indices[next_next_expert]).activate();
            context.getWeight(expert_down_proj_indices[next_next_expert]).activate();
          });
        }
        
        // Compute current expert while next is loading
        const auto &assignments = expert_assignments[current_expert];
        compute_expert_forward_no_critical(
          input, expert_outputs[current_expert], assignments,
          context.getWeight(expert_gate_proj_indices[current_expert]),
          context.getWeight(expert_up_proj_indices[current_expert]),
          context.getWeight(expert_down_proj_indices[current_expert]), hidden_size);
        
        // Wait for next expert loading to complete (if it was started)
        if (next_expert != -1) {
          next_load_future.wait();
          // Update cache tracking for the loaded expert
          loaded_expert_deque.push_back(next_expert);
          iteration_map[next_expert] = --loaded_expert_deque.end();
          need_load[next_expert] = false;
        }
        
        // Check if i+2 prefetch completed (don't wait, just check)
        if (next_next_expert != -1 && next_next_load_future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
          // Update cache if prefetch completed
          loaded_expert_deque.push_back(next_next_expert);
          iteration_map[next_next_expert] = --loaded_expert_deque.end();
          need_load[next_next_expert] = false;
          // Remove from experts_to_load to avoid double processing
          experts_to_load.erase(experts_to_load.begin() + i + 2);
        }
      }
    }
    
    // Update LRU for already loaded experts
    for (int expert_idx : loaded_experts) {
      if (iteration_map.find(expert_idx) != iteration_map.end()) {
        loaded_expert_deque.erase(iteration_map[expert_idx]);
      }
      loaded_expert_deque.push_back(expert_idx);
      iteration_map[expert_idx] = --loaded_expert_deque.end();
    }
    
    // Cache eviction - use async or sync based on configuration
    std::vector<int> evicted_experts;
    while (loaded_expert_deque.size() > max_cached_experts) {
      int target_idx = loaded_expert_deque.front();
      evicted_experts.push_back(target_idx);
      
      if (enable_async_deactivation) {
        // Queue for background deactivation
        {
          std::lock_guard<std::mutex> lock(deactivation_mutex);
          deactivation_queue.push({target_idx, &context});
        }
        deactivation_cv.notify_one();
      } else {
        // Synchronous deactivation (original behavior)
        context.getWeight(expert_gate_proj_indices[target_idx]).deactivate();
        context.getWeight(expert_up_proj_indices[target_idx]).deactivate();
        context.getWeight(expert_down_proj_indices[target_idx]).deactivate();
      }
      
      // Update tracking immediately
      loaded_expert_deque.pop_front();
      iteration_map.erase(target_idx);
      need_load[target_idx] = true;
    }
    
    // Optional: Wait for deactivations if we evicted many experts (memory pressure)
    // This prevents OOM when many experts are being swapped
    if (enable_async_deactivation && evicted_experts.size() > 4) {
      // Give deactivation thread time to free memory
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // Combine expert outputs
    for (int expert_idx : active_experts) {
      output.add_i(expert_outputs[expert_idx]);
    }

    // reshape output: [B*S,1,1,H] -> [B,1,S,H]
    output.reshape({batch_size, 1, seq_len, hidden_size});
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
