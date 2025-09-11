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

#include <chrono>
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::microseconds; // or microseconds
using std::chrono::milliseconds; // or microseconds
using std::chrono::nanoseconds;  // or microseconds
using std::chrono::seconds;      // or microseconds

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
}

void CachedSlimMoELayer::forwarding(nntrainer::RunLayerContext &context,
                                    bool training) {}

inline void CachedSlimMoELayer::compute_expert_forward(
  const nntrainer::Tensor &input, nntrainer::Tensor &output,
  const std::vector<std::pair<unsigned, float>> &token_assignments,
  const nntrainer::Tensor &gate_proj, const nntrainer::Tensor &up_proj,
  const nntrainer::Tensor &down_proj, unsigned int hidden_size) {

  const unsigned intermediate_size = gate_proj.width();
  const unsigned num_tokens = token_assignments.size();

  if (num_tokens == 0)
    return;

  // Create tensor dimensions for batch processing
  nntrainer::TensorDim token_input_dim({1, 1, num_tokens, hidden_size},
                                       input.getTensorType());
  nntrainer::TensorDim intermediate_dim({1, 1, num_tokens, intermediate_size},
                                        input.getTensorType());
  nntrainer::TensorDim token_output_dim({1, 1, num_tokens, hidden_size},
                                        input.getTensorType());
  nntrainer::TensorDim out_step_dim({1, 1, 1, hidden_size},
                                    input.getTensorType());
  nntrainer::TensorDim step_dim({1, 1, 1, intermediate_size},
                                input.getTensorType());
  
  // Use thread-local memory pool for better performance
  static thread_local nntrainer::Tensor gate_out;
  static thread_local nntrainer::Tensor acti_out;
  static thread_local nntrainer::Tensor up_out;
  static thread_local nntrainer::Tensor token_input;
  static thread_local nntrainer::Tensor token_expert_output;
  
  // Resize tensors only if dimensions changed
  if (gate_out.empty() || gate_out.getDim().getDataLen() < intermediate_dim.getDataLen()) {
    gate_out = nntrainer::Tensor(intermediate_dim);
    acti_out = nntrainer::Tensor(intermediate_dim);
    up_out = nntrainer::Tensor(intermediate_dim);
  }
  if (token_input.empty() || token_input.getDim().getDataLen() < token_input_dim.getDataLen()) {
    token_input = nntrainer::Tensor(token_input_dim);
    token_expert_output = nntrainer::Tensor(token_output_dim);
  }
  
  // Reshape tensors to current batch size
  gate_out.reshape(intermediate_dim);
  acti_out.reshape(intermediate_dim);
  up_out.reshape(intermediate_dim);
  token_input.reshape(token_input_dim);
  token_expert_output.reshape(token_output_dim)

  if (num_tokens > 1) {
    /** Batch processing for prefill - use optimized tensor operations */
    // Create a temporary tensor for batch gathering
    std::vector<unsigned int> indices;
    indices.reserve(num_tokens);
    for (const auto& assignment : token_assignments) {
      indices.push_back(assignment.first);
    }
    
    // Use optimized batch copy if contiguous
    if (input.getContiguous() && token_input.getContiguous()) {
      #pragma omp parallel for schedule(static) if(num_tokens > 8)
      for (size_t i = 0; i < num_tokens; ++i) {
        const unsigned token_idx = indices[i];
        // Use tensor's optimized copy operation
        nntrainer::Tensor src_view = input.getSharedDataTensor(
          {1, 1, 1, hidden_size}, token_idx * hidden_size, true);
        nntrainer::Tensor dst_view = token_input.getSharedDataTensor(
          {1, 1, 1, hidden_size}, i * hidden_size, true);
        dst_view.copyData(src_view);
      }
    } else {
      // Fallback to element-wise copy for non-contiguous tensors
      for (size_t i = 0; i < num_tokens; ++i) {
        const unsigned token_idx = indices[i];
        for (unsigned j = 0; j < hidden_size; ++j) {
          token_input.setValue(0, 0, i, j, 
                              input.getValue<float>(0, 0, token_idx, j));
        }
      }
    }
  } else {
    /** Single token generation - use zero-copy shared tensor */
    unsigned token_idx = token_assignments[0].first;
    size_t token_offset = token_idx * hidden_size;
    token_input = input.getSharedDataTensor(token_input_dim, token_offset, true);
  }

  // Use tensor's optimized dot operations for projections
  token_input.dot(gate_proj, gate_out);
  token_input.dot(up_proj, up_out);

  if (num_tokens == 1) {
    // Single token: apply activation and multiply
    acti_func.run_fn(gate_out, acti_out);
    acti_out.multiply_i(up_out);
  } else {
    // Batch processing: use backend-optimized swiglu
    // The swiglu function is already optimized in the backend
    for (size_t i = 0; i < num_tokens; ++i) {
      nntrainer::Tensor gate_slice = gate_out.getSharedDataTensor(
        {1, 1, 1, intermediate_size}, i * intermediate_size, true);
      nntrainer::Tensor up_slice = up_out.getSharedDataTensor(
        {1, 1, 1, intermediate_size}, i * intermediate_size, true);
      nntrainer::Tensor acti_slice = acti_out.getSharedDataTensor(
        {1, 1, 1, intermediate_size}, i * intermediate_size, true);
      
      // Use the backend-optimized swiglu implementation
      nntrainer::swiglu(
        intermediate_size,
        acti_slice.getData<float>(),
        gate_slice.getData<float>(),
        up_slice.getData<float>());
    }
  }

  // Down projection using tensor operation
  acti_out.dot(down_proj, token_expert_output);

  // Accumulate weighted results to output using tensor operations
  for (size_t i = 0; i < num_tokens; ++i) {
    const unsigned token_idx = token_assignments[i].first;
    const float weight = token_assignments[i].second;
    
    // Get shared tensors for source and destination
    nntrainer::Tensor expert_slice = token_expert_output.getSharedDataTensor(
      out_step_dim, i * hidden_size, true);
    nntrainer::Tensor output_slice = output.getSharedDataTensor(
      out_step_dim, token_idx * hidden_size, true);
    
    // Use tensor's optimized multiply and add operations
    if (weight != 1.0f) {
      // Create temporary for weighted expert output
      nntrainer::Tensor weighted = expert_slice.multiply(weight);
      output_slice.add_i(weighted);
    } else {
      // Direct add for weight = 1.0
      output_slice.add_i(expert_slice);
    }
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
    // Set expert mask
    for (int i = 0; i < static_cast<int>(total_tokens); ++i) {
      for (int k = 0; k < static_cast<int>(topk); ++k) {
        unsigned expert_idx = indices_data[i * topk + k];
        float weight = topk_values.getValue<float>(i, 0, 0, k);
        expert_assignments[expert_idx].emplace_back(i, weight);
      }
    }

    // Use thread-local memory pool for expert outputs
    static thread_local std::vector<nntrainer::Tensor> expert_outputs;
    if (expert_outputs.size() < num_experts) {
      expert_outputs.resize(num_experts);
    }
    
    // Allocate/reuse expert output tensors
    for (int expert_idx = 0; expert_idx < static_cast<int>(num_experts);
         ++expert_idx) {
      if (!expert_assignments[expert_idx].empty()) {
        nntrainer::TensorDim expert_dim(total_tokens, 1, 1, hidden_size, 
                                        output.getTensorType());
        if (expert_outputs[expert_idx].empty() || 
            expert_outputs[expert_idx].getDim() != expert_dim) {
          expert_outputs[expert_idx] = nntrainer::Tensor(expert_dim);
        }
        // Use tensor's setZero for initialization
        expert_outputs[expert_idx].setZero();
      }
    }
    std::vector<int> target_idx_vector;

    for (int expert_idx = 0; expert_idx < static_cast<int>(num_experts);
         ++expert_idx) {
      const auto &assignments = expert_assignments[expert_idx];
      if (assignments.empty())
        continue;

      target_idx_vector.push_back(expert_idx);
    }
    int hit_count = 0;
    int miss_count = 0;
    std::vector<int> missed_idx_vector;
    std::vector<int> hit_idx_vector;
    std::vector<int> evict_idx_vector;

    for (int expert_idx : target_idx_vector) {
      if (need_load[expert_idx]) {
        miss_count += 1;
        loaded_expert_deque.push_back(expert_idx);
        missed_idx_vector.push_back(expert_idx);
        iteration_map[expert_idx] = --loaded_expert_deque.end();
        need_load[expert_idx] = false;
      } else {
        hit_count += 1;
        hit_idx_vector.push_back(expert_idx);
        // move recently used index to back;
        // ___________________________________________
        // |old element <================ new elemnt |
        // -------------------------------------------

        // LRU Algorithm
        if (iteration_map.find(expert_idx) != iteration_map.end()) {
          loaded_expert_deque.erase(iteration_map[expert_idx]);
        }
        loaded_expert_deque.push_back(expert_idx);
        iteration_map[expert_idx] = --loaded_expert_deque.end();
      }
    }

#ifdef DEBUG
    auto t1_hit = high_resolution_clock::now();
#endif
    // Process cached experts with optimal scheduling
    #pragma omp parallel for schedule(guided) if(hit_idx_vector.size() > 2)
    for (size_t idx = 0; idx < hit_idx_vector.size(); ++idx) {
      int expert_idx = hit_idx_vector[idx];
      const auto &assignments = expert_assignments[expert_idx];

      compute_expert_forward(
        input, expert_outputs[expert_idx], assignments,
        context.getWeight(expert_gate_proj_indices[expert_idx]),
        context.getWeight(expert_up_proj_indices[expert_idx]),
        context.getWeight(expert_down_proj_indices[expert_idx]), hidden_size);
    }
#ifdef DEBUG
    auto t2_hit = high_resolution_clock::now();

    auto t1_miss = high_resolution_clock::now();
#endif
    // Process missed experts with weight activation
    #pragma omp parallel for schedule(guided) if(missed_idx_vector.size() > 2)
    for (size_t idx = 0; idx < missed_idx_vector.size(); ++idx) {
      int expert_idx = missed_idx_vector[idx];
      
      // Activate weights using tensor's virtual memory mechanism
      context.getWeight(expert_gate_proj_indices[expert_idx]).activate();
      context.getWeight(expert_up_proj_indices[expert_idx]).activate();
      context.getWeight(expert_down_proj_indices[expert_idx]).activate();
      
      const auto &assignments = expert_assignments[expert_idx];
      compute_expert_forward(
        input, expert_outputs[expert_idx], assignments,
        context.getWeight(expert_gate_proj_indices[expert_idx]),
        context.getWeight(expert_up_proj_indices[expert_idx]),
        context.getWeight(expert_down_proj_indices[expert_idx]), hidden_size);
    }
#ifdef DEBUG
    auto t2_miss = high_resolution_clock::now();
#endif

    while (loaded_expert_deque.size() > 32) {
      int target_idx = loaded_expert_deque.front();
      loaded_expert_deque.pop_front();
      iteration_map.erase(target_idx);
      need_load[target_idx] = true;
      evict_idx_vector.push_back(target_idx);
    }

#ifdef DEBUG
    auto t1_evict = high_resolution_clock::now();
#endif
    // Batch eviction of unused experts
    if (!evict_idx_vector.empty()) {
      #pragma omp parallel for schedule(static) if(evict_idx_vector.size() > 4)
      for (size_t idx = 0; idx < evict_idx_vector.size(); ++idx) {
        int target_idx = evict_idx_vector[idx];
        // Use tensor's deactivate for virtual memory management
        context.getWeight(expert_gate_proj_indices[target_idx]).deactivate();
        context.getWeight(expert_up_proj_indices[target_idx]).deactivate();
        context.getWeight(expert_down_proj_indices[target_idx]).deactivate();
      }
    }
#ifdef DEBUG
    auto t2_evict = high_resolution_clock::now();
#endif

    // Combine expert outputs using tensor operations
    bool first = true;
    for (int expert_idx : target_idx_vector) {
      if (expert_assignments[expert_idx].empty()) continue;
      
      if (first) {
        // First expert: copy directly
        output.copyData(expert_outputs[expert_idx]);
        first = false;
      } else {
        // Subsequent experts: use tensor's optimized add operation
        // The add_i operation is already optimized in the backend
        output.add_i(expert_outputs[expert_idx]);
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

void CachedSlimMoELayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  ml::train::TensorDim input_dim = context.getInput(SINGLE_INOUT_IDX).getDim();
  ml::train::TensorDim output_dim =
    context.getOutput(SINGLE_INOUT_IDX).getDim();

  input_dim.height(input_dimensions[0].height());
  output_dim.height(input_dimensions[0].height());

  context.updateInput(SINGLE_INOUT_IDX, input_dim);
  context.updateOutput(SINGLE_INOUT_IDX, output_dim);
}

} // namespace causallm
