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
 */

#include <qwen_moe_layer_cached.h>
#include <acti_func.h>
#include <algorithm>
#include <cmath>
#include <node_exporter.h>
#include <omp.h>
#include <stdexcept>
#include <iostream>

#ifdef __ANDROID__
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
  gate_idx(std::numeric_limits<unsigned>::max()),
  router_logits_idx(std::numeric_limits<unsigned>::max()),
  expert_mask_idx(std::numeric_limits<unsigned>::max()) {
#ifdef __ANDROID__
  const char *cache_size_env = std::getenv("NNTRAINER_MOE_CACHE_SIZE");
  if (cache_size_env) {
    base_cache_size = std::stoi(cache_size_env);
    current_cache_size = base_cache_size;
  }
#endif
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
  const unsigned int hidden_size = in_dim.width();

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
    expert_up_proj_indices.push_back(context.requestWeight(
      expert_gate_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay,
      "expert_up_" + std::to_string(i), false, true));

    expert_gate_proj_indices.push_back(context.requestWeight(
      expert_gate_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay,
      "expert_gate_" + std::to_string(i), false, true));

    expert_down_proj_indices.push_back(context.requestWeight(
      expert_down_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay,
      "expert_down_" + std::to_string(i), false, true));
  }

  // Initialize cache management
  is_cached.resize(num_experts, false);
  last_token_position.resize(num_experts, -1);
  cache_order.reserve(base_cache_size * 2); // Reserve extra space for dynamic sizing

  // 6. Request intermediate tensors
  const unsigned batch_size = in_dim.batch();
  const unsigned seq_len = in_dim.height();
  const unsigned total_tokens = batch_size * seq_len;

  router_logits_idx =
    context.requestTensor({total_tokens, 1, 1, num_experts}, "router_logits",
                          nntrainer::Initializer::NONE, false,
                          nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);

  expert_mask_idx =
    context.requestTensor({num_experts, 1, topk, total_tokens}, "expert_mask",
                          nntrainer::Initializer::ZEROS, false,
                          nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
}

void CachedSlimMoELayer::forwarding(nntrainer::RunLayerContext &context,
                                    bool training) {
  // Use incremental_forwarding for consistency
  incremental_forwarding(context, 0, 0, training);
}

void CachedSlimMoELayer::updateCacheSize(int unique_experts, int total_expert_requests) {
  // Calculate diversity ratio
  expert_diversity_ratio = static_cast<float>(unique_experts) / total_expert_requests;
  
  // Adjust cache size based on diversity
  // High diversity (>0.7) -> increase cache size
  // Low diversity (<0.3) -> decrease cache size
  if (expert_diversity_ratio > 0.7f) {
    current_cache_size = std::min(base_cache_size * 2, num_experts);
  } else if (expert_diversity_ratio < 0.3f) {
    current_cache_size = std::max(base_cache_size / 2, topk * 2u);
  } else {
    current_cache_size = base_cache_size;
  }
  
  // Print stats periodically for debugging
#ifdef DEBUG
  static int call_count = 0;
  if (++call_count % 100 == 0) {
    std::cerr << "[MoE Cache] Hit rate: " << getCacheHitRate() 
              << ", Diversity: " << expert_diversity_ratio
              << ", Cache size: " << current_cache_size << std::endl;
  }
#endif
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

    input.reshape({total_tokens, 1, 1, hidden_size});
    output.reshape({total_tokens, 1, 1, hidden_size});
    output.setZero();

    // Routing
    nntrainer::Tensor &gate_weights = context.getWeight(gate_idx);
    input.dot(gate_weights, router_logits);
    router_logits.apply(nntrainer::ActiFunc::softmax<float>, router_logits);
    auto topk_result = router_logits.topK(topk);
    auto topk_values = std::get<0>(topk_result);
    auto topk_indices = std::get<1>(topk_result);

    topk_values.divide_i(topk_values.sum(3));

    const uint32_t *indices_data = topk_indices.getData<uint32_t>();
    std::vector<std::vector<std::pair<unsigned, float>>> expert_assignments(
      num_experts);

    // Track max token position and unique experts
    std::vector<int> expert_max_token_pos(num_experts, -1);
    std::vector<bool> expert_used(num_experts, false);
    int unique_expert_count = 0;
    int total_expert_requests = 0;
    
    for (int i = 0; i < static_cast<int>(total_tokens); ++i) {
      for (int k = 0; k < static_cast<int>(topk); ++k) {
        unsigned expert_idx = indices_data[i * topk + k];
        float weight = topk_values.getValue<float>(i, 0, 0, k);
        expert_assignments[expert_idx].emplace_back(i, weight);
        expert_max_token_pos[expert_idx] = i;
        
        if (!expert_used[expert_idx]) {
          expert_used[expert_idx] = true;
          unique_expert_count++;
        }
        total_expert_requests++;
      }
    }

    // Update cache size based on diversity
    updateCacheSize(unique_expert_count, total_expert_requests);

    // Determine if this is batch processing or incremental
    bool is_batch_processing = (total_tokens > 1);
    
    // Collect and sort active experts
    std::vector<std::pair<int, int>> expert_priority;
    for (int expert_idx = 0; expert_idx < static_cast<int>(num_experts);
         ++expert_idx) {
      if (!expert_assignments[expert_idx].empty()) {
        expert_priority.emplace_back(expert_idx, expert_max_token_pos[expert_idx]);
      }
    }
    
    // Sort experts by token position for batch processing
    if (is_batch_processing) {
      std::sort(expert_priority.begin(), expert_priority.end(),
                [](const auto &a, const auto &b) {
                  return a.second > b.second; // Higher position first
                });
    }

    // Wait for any pending prefetch
    if (prefetch_future.valid()) {
      prefetch_future.wait();
      if (prefetching_expert >= 0 && !is_cached[prefetching_expert]) {
        is_cached[prefetching_expert] = true;
        cache_order.push_back(prefetching_expert);
      }
      prefetching_expert = -1;
    }

    // Process experts with prefetching
    for (size_t idx = 0; idx < expert_priority.size(); ++idx) {
      const auto &[expert_idx, max_pos] = expert_priority[idx];
      const auto &assignments = expert_assignments[expert_idx];

      total_requests++;
      
      // Check cache status
      if (!is_cached[expert_idx]) {
        cache_misses++;
        
        // Load current expert
        context.getWeight(expert_gate_proj_indices[expert_idx]).activate();
        context.getWeight(expert_up_proj_indices[expert_idx]).activate();
        context.getWeight(expert_down_proj_indices[expert_idx]).activate();
        
        is_cached[expert_idx] = true;
        last_token_position[expert_idx] = max_pos;
        cache_order.push_back(expert_idx);
        
        // Start prefetching next expert if available
        if (idx + 1 < expert_priority.size()) {
          int next_expert = expert_priority[idx + 1].first;
          if (!is_cached[next_expert] && !prefetch_future.valid()) {
            prefetching_expert = next_expert;
            prefetch_future = std::async(std::launch::async, 
              [&context, this, next_expert]() {
                context.getWeight(expert_gate_proj_indices[next_expert]).activate();
                context.getWeight(expert_up_proj_indices[next_expert]).activate();
                context.getWeight(expert_down_proj_indices[next_expert]).activate();
              });
          }
        }
        
        // Cache eviction with adaptive size
        while (cache_order.size() > current_cache_size) {
          // Find expert to evict
          int evict_candidate_idx = 0;
          
          if (is_batch_processing) {
            // Evict expert with lowest token position
            int min_token_pos = last_token_position[cache_order[0]];
            for (size_t i = 1; i < cache_order.size() - 1; ++i) {
              if (last_token_position[cache_order[i]] < min_token_pos) {
                min_token_pos = last_token_position[cache_order[i]];
                evict_candidate_idx = i;
              }
            }
          }
          
          int evict_idx = cache_order[evict_candidate_idx];
          
          // Don't evict if it's being prefetched
          if (evict_idx == prefetching_expert) {
            if (evict_candidate_idx < cache_order.size() - 1) {
              evict_candidate_idx++;
              evict_idx = cache_order[evict_candidate_idx];
            }
          }
          
          cache_order.erase(cache_order.begin() + evict_candidate_idx);
          
          if (is_cached[evict_idx]) {
            context.getWeight(expert_gate_proj_indices[evict_idx]).deactivate();
            context.getWeight(expert_up_proj_indices[evict_idx]).deactivate();
            context.getWeight(expert_down_proj_indices[evict_idx]).deactivate();
            is_cached[evict_idx] = false;
            last_token_position[evict_idx] = -1;
          }
        }
      } else {
        cache_hits++;
        
        // Update LRU and token position
        last_token_position[expert_idx] = max_pos;
        auto it = std::find(cache_order.begin(), cache_order.end(), expert_idx);
        if (it != cache_order.end()) {
          cache_order.erase(it);
          cache_order.push_back(expert_idx);
        }
      }

      // Compute expert forward
      nntrainer::Tensor expert_output(total_tokens, 1, 1, hidden_size,
                                      output.getTensorType());
      expert_output.setZero();
      
      compute_expert_forward_no_critical(
        input, expert_output, assignments,
        context.getWeight(expert_gate_proj_indices[expert_idx]),
        context.getWeight(expert_up_proj_indices[expert_idx]),
        context.getWeight(expert_down_proj_indices[expert_idx]), hidden_size);
      
      output.add_i(expert_output);
    }

    // Clean up any remaining prefetch
    if (prefetch_future.valid()) {
      prefetch_future.wait();
      if (prefetching_expert >= 0 && !is_cached[prefetching_expert]) {
        is_cached[prefetching_expert] = true;
        last_token_position[prefetching_expert] = -1;
        cache_order.push_back(prefetching_expert);
      }
      prefetching_expert = -1;
    }

    output.reshape({batch_size, 1, seq_len, hidden_size});
  }
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

  nntrainer::TensorDim token_input_dim({1, 1, 1, hidden_size},
                                       input.getTensorType());
  nntrainer::TensorDim intermediate_dim({1, 1, 1, intermediate_size},
                                        input.getTensorType());
  nntrainer::TensorDim token_output_dim({1, 1, 1, hidden_size},
                                        input.getTensorType());

  // Process tokens in reverse order for batch mode (better cache locality)
  for (int i = num_tokens - 1; i >= 0; --i) {
    const unsigned token_idx = token_assignments[i].first;
    const float weight = token_assignments[i].second;

    size_t token_offset = token_idx * hidden_size;
    nntrainer::Tensor token_input =
      input.getSharedDataTensor(token_input_dim, token_offset, true);

    nntrainer::Tensor gate_out(intermediate_dim);
    nntrainer::Tensor acti_out(intermediate_dim);
    nntrainer::Tensor up_out(intermediate_dim);

    token_input.dot(gate_proj, gate_out);
    acti_func.run_fn(gate_out, acti_out);
    token_input.dot(up_proj, up_out);
    acti_out.multiply_i(up_out);

    nntrainer::Tensor token_expert_output(token_output_dim);
    acti_out.dot(down_proj, token_expert_output);
    token_expert_output.multiply_i(weight);

    size_t output_offset = token_idx * hidden_size;
    nntrainer::Tensor token_output =
      expert_output.getSharedDataTensor(token_output_dim, output_offset, true);
    token_output.add_i(token_expert_output);
  }
}

void CachedSlimMoELayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, moe_props);
  nntrainer::LayerImpl::setProperty(remain_props);
}

void CachedSlimMoELayer::calcDerivative(nntrainer::RunLayerContext &context) {
  throw std::runtime_error("MoE layer does not support derivative calculation");
}

void CachedSlimMoELayer::calcGradient(nntrainer::RunLayerContext &context) {
  throw std::runtime_error("MoE layer does not support gradient calculation");
}

void CachedSlimMoELayer::exportTo(
  nntrainer::Exporter &exporter,
  const ml::train::ExportMethods &method) const {
  nntrainer::LayerImpl::exportTo(exporter, method);
  exporter.saveResult(moe_props, method, this);
}

} // namespace causallm
