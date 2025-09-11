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
 * @file	gpt_oss_moe_layer_cached.cpp
 * @date	05 Sep 2025
 * @brief	This is a Mixture of Expert Layer Class for Gpt-Oss model
 * @see		https://github.com/nnstreamer/
 * @author	Eunju Yang <ej.yang@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <acti_func.h>
#include <algorithm>
#include <cmath>
#include <gpt_oss_moe_layer_cached.h>
#include <node_exporter.h>
#include <omp.h>
#include <stdexcept>

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

CachedSlimGptOssMoELayer::CachedSlimGptOssMoELayer() :
  LayerImpl(),
  num_experts(0),
  topk(0),
  moe_props(props::NumExperts(), props::NumExpertsPerToken(),
            nntrainer::props::Unit()),
  expert_gate_proj_indices({}),
  expert_gate_bias_indices({}),
  expert_up_proj_indices({}),
  expert_up_bias_indices({}),
  expert_down_proj_indices({}),
  expert_down_bias_indices({}),
  gate_idx(std::numeric_limits<unsigned>::max()),
  gate_bias_idx(std::numeric_limits<unsigned>::max()),
  loaded_expert_deque({}),
  need_load({}),
  router_logits_idx(std::numeric_limits<unsigned>::max()),
  expert_mask_idx(std::numeric_limits<unsigned>::max()) {}

void CachedSlimGptOssMoELayer::finalize(nntrainer::InitLayerContext &context) {

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

  // pure tensor
  nntrainer::TensorDim gate_bias_dim(
    1, 1, 1, num_experts,
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getActivationDataType()));
  // pure tensor
  gate_bias_idx =
    context.requestWeight(gate_bias_dim, weight_initializer, weight_regularizer,
                          1.0f, weight_decay, "gate_bias", false);

  // 5. Initializer expert weights (virtual tensor)
  expert_gate_proj_indices.reserve(num_experts);
  expert_up_proj_indices.reserve(num_experts);
  expert_down_proj_indices.reserve(num_experts);
  expert_gate_bias_indices.reserve(num_experts);
  expert_up_bias_indices.reserve(num_experts);
  expert_down_bias_indices.reserve(num_experts);

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

  nntrainer::TensorDim expert_gate_bias_dim(
    1, 1, 1, intermediate_size,
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getActivationDataType()),
    is_nchw ? 0b0011 : 0b0101);

  nntrainer::TensorDim expert_down_bias_dim(
    1, 1, 1, hidden_size,
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getActivationDataType()),
    is_nchw ? 0b0011 : 0b0101);

  for (unsigned int i = 0; i < num_experts; ++i) {
    // Up projection
    expert_up_proj_indices.push_back(context.requestWeight(
      expert_gate_dim, // Same dimensions as gate projection
      weight_initializer, weight_regularizer, weight_regularizer_constant,
      weight_decay, "expert_up_" + std::to_string(i), false, true));

    expert_up_bias_indices.push_back(context.requestWeight(
      expert_gate_bias_dim, // Same dimensions as gate projection
      weight_initializer, weight_regularizer, weight_regularizer_constant,
      weight_decay, "expert_up_bias_" + std::to_string(i), false, true));

    // Gate projection
    expert_gate_proj_indices.push_back(context.requestWeight(
      expert_gate_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay,
      "expert_gate_" + std::to_string(i), false, true));

    expert_gate_bias_indices.push_back(context.requestWeight(
      expert_gate_bias_dim, // Same dimensions as gate projection
      weight_initializer, weight_regularizer, weight_regularizer_constant,
      weight_decay, "expert_gate_bias_" + std::to_string(i), false, true));

    // Down projection
    expert_down_proj_indices.push_back(context.requestWeight(
      expert_down_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay,
      "expert_down_" + std::to_string(i), false, true));

    expert_down_bias_indices.push_back(context.requestWeight(
      expert_down_bias_dim, // Same dimensions as gate projection
      weight_initializer, weight_regularizer, weight_regularizer_constant,
      weight_decay, "expert_down_bias_" + std::to_string(i), false, true));

    // need_load this expert
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

void CachedSlimGptOssMoELayer::forwarding(nntrainer::RunLayerContext &context,
                                          bool training) {}

void CachedSlimGptOssMoELayer::incremental_forwarding(
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

    // Allocate expert outputs only for active experts to save memory
    std::vector<nntrainer::Tensor> expert_outputs(num_experts);
    std::vector<bool> expert_active(num_experts, false);
    int active_expert_count = 0;
    for (int expert_idx = 0; expert_idx < static_cast<int>(num_experts);
         ++expert_idx) {
      if (!expert_assignments[expert_idx].empty()) {
        expert_outputs[expert_idx] = nntrainer::Tensor(
          total_tokens, 1, 1, hidden_size, output.getTensorType());
        expert_active[expert_idx] = true;
        active_expert_count++;
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
// run hit experts with better scheduling
#pragma omp parallel for schedule(static) if(hit_idx_vector.size() > 2)
    for (size_t i = 0; i < hit_idx_vector.size(); ++i) {
      int expert_idx = hit_idx_vector[i];
      const auto &assignments = expert_assignments[expert_idx];

      compute_expert_forward(
        input, expert_outputs[expert_idx], assignments,
        context.getWeight(expert_gate_proj_indices[expert_idx]),
        context.getWeight(expert_up_proj_indices[expert_idx]),
        context.getWeight(expert_down_proj_indices[expert_idx]),
        context.getWeight(expert_gate_bias_indices[expert_idx]),
        context.getWeight(expert_up_bias_indices[expert_idx]),
        context.getWeight(expert_down_bias_indices[expert_idx]), hidden_size);
    }
#ifdef DEBUG
    auto t2_hit = high_resolution_clock::now();

    auto t1_miss = high_resolution_clock::now();
#endif
#pragma omp parallel for schedule(static) if(missed_idx_vector.size() > 2)
    for (size_t i = 0; i < missed_idx_vector.size(); ++i) {
      int expert_idx = missed_idx_vector[i];
      context.getWeight(expert_gate_proj_indices[expert_idx]).activate();
      context.getWeight(expert_up_proj_indices[expert_idx]).activate();
      context.getWeight(expert_down_proj_indices[expert_idx]).activate();

      context.getWeight(expert_gate_bias_indices[expert_idx]).activate();
      context.getWeight(expert_up_bias_indices[expert_idx]).activate();
      context.getWeight(expert_down_bias_indices[expert_idx]).activate();

      const auto &assignments = expert_assignments[expert_idx];

      compute_expert_forward(
        input, expert_outputs[expert_idx], assignments,
        context.getWeight(expert_gate_proj_indices[expert_idx]),
        context.getWeight(expert_up_proj_indices[expert_idx]),
        context.getWeight(expert_down_proj_indices[expert_idx]),
        context.getWeight(expert_gate_bias_indices[expert_idx]),
        context.getWeight(expert_up_bias_indices[expert_idx]),
        context.getWeight(expert_down_bias_indices[expert_idx]), hidden_size);
    }
#ifdef DEBUG
    auto t2_miss = high_resolution_clock::now();
#endif

    while (loaded_expert_deque.size() > 8) {
      int target_idx = loaded_expert_deque.front();
      loaded_expert_deque.pop_front();
      iteration_map.erase(target_idx);
      need_load[target_idx] = true;
      evict_idx_vector.push_back(target_idx);
    }

#ifdef DEBUG
    auto t1_evict = high_resolution_clock::now();
#endif
#pragma omp parallel for schedule(static) if(evict_idx_vector.size() > 2)
    for (size_t i = 0; i < evict_idx_vector.size(); ++i) {
      int target_idx = evict_idx_vector[i];
      context.getWeight(expert_gate_proj_indices[target_idx]).deactivate();
      context.getWeight(expert_up_proj_indices[target_idx]).deactivate();
      context.getWeight(expert_down_proj_indices[target_idx]).deactivate();
      context.getWeight(expert_gate_bias_indices[target_idx]).deactivate();
      context.getWeight(expert_up_bias_indices[target_idx]).deactivate();
      context.getWeight(expert_down_bias_indices[target_idx]).deactivate();
    }
#ifdef DEBUG
    auto t2_evict = high_resolution_clock::now();
#endif

    // Combine expert outputs more efficiently
    output.setValue(0.0f); // Initialize to zero
    for (int expert_idx : target_idx_vector) {
      if (expert_active[expert_idx]) {
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

// Optimized batch processing version of expert forward
inline void CachedSlimGptOssMoELayer::compute_expert_forward(
  const nntrainer::Tensor &input, nntrainer::Tensor &expert_output,
  const std::vector<std::pair<unsigned, float>> &token_assignments,
  const nntrainer::Tensor &gate_proj, const nntrainer::Tensor &up_proj,
  const nntrainer::Tensor &down_proj, const nntrainer::Tensor &gate_bias,
  const nntrainer::Tensor &up_bias, const nntrainer::Tensor &down_bias,
  unsigned int hidden_size) {

  const unsigned intermediate_size = gate_proj.width();
  const unsigned num_tokens = token_assignments.size();

  if (num_tokens == 0)
    return;

  // Batch processing for better cache utilization and SIMD optimization
  if (num_tokens > 1) {
    // Pre-allocate batch tensors
    nntrainer::TensorDim batch_input_dim({num_tokens, 1, 1, hidden_size},
                                         input.getTensorType());
    nntrainer::TensorDim batch_intermediate_dim({num_tokens, 1, 1, intermediate_size},
                                                input.getTensorType());
    nntrainer::TensorDim batch_output_dim({num_tokens, 1, 1, hidden_size},
                                          input.getTensorType());
    
    nntrainer::Tensor batch_input(batch_input_dim);
    nntrainer::Tensor batch_gate_out(batch_intermediate_dim);
    nntrainer::Tensor batch_up_out(batch_intermediate_dim);
    nntrainer::Tensor batch_down_out(batch_output_dim);
    
    // Gather input tokens
    #pragma omp parallel for if(num_tokens > 4)
    for (size_t i = 0; i < num_tokens; ++i) {
      const unsigned token_idx = token_assignments[i].first;
      std::memcpy(batch_input.getData<float>() + i * hidden_size,
                 input.getData<float>() + token_idx * hidden_size,
                 hidden_size * sizeof(float));
    }
    
    // Batch matrix multiplications
    batch_input.dot(gate_proj, batch_gate_out);
    batch_input.dot(up_proj, batch_up_out);
    
    // Apply biases and clamp in batch
    float *gate_data = batch_gate_out.getData<float>();
    float *up_data = batch_up_out.getData<float>();
    
    #pragma omp parallel for if(num_tokens > 2)
    for (size_t i = 0; i < num_tokens; ++i) {
      size_t offset = i * intermediate_size;
      // Gate: add bias and clamp
      for (size_t j = 0; j < intermediate_size; ++j) {
        gate_data[offset + j] += gate_bias.getValue<float>(0, 0, 0, j);
        gate_data[offset + j] = std::min(gate_data[offset + j], limit);
      }
      // Up: add bias and clamp
      for (size_t j = 0; j < intermediate_size; ++j) {
        up_data[offset + j] += up_bias.getValue<float>(0, 0, 0, j);
        up_data[offset + j] = std::max(-limit, std::min(limit, up_data[offset + j]));
      }
    }
    
    // Apply swiglu activation in batch
    #pragma omp parallel for if(num_tokens > 2)
    for (size_t i = 0; i < num_tokens; ++i) {
      size_t offset = i * intermediate_size;
      float *acti_ptr = gate_data + offset; // Reuse gate buffer
      float *gate_ptr = gate_data + offset;
      float *up_ptr = up_data + offset;
      
      // Optimized swiglu: (up + 1) * gate * sigmoid(gate * alpha)
      for (size_t j = 0; j < intermediate_size; ++j) {
        float gate_val = gate_ptr[j];
        float up_val = up_ptr[j] + 1.0f;
        float sigmoid = 1.0f / (1.0f + std::exp(-alpha * gate_val));
        acti_ptr[j] = gate_val * sigmoid * up_val;
      }
    }
    
    // Down projection
    batch_gate_out.dot(down_proj, batch_down_out);
    
    // Scatter results with weight and bias
    #pragma omp parallel for if(num_tokens > 4)
    for (size_t i = 0; i < num_tokens; ++i) {
      const unsigned token_idx = token_assignments[i].first;
      const float weight = token_assignments[i].second;
      
      float *out_ptr = expert_output.getData<float>() + token_idx * hidden_size;
      const float *down_ptr = batch_down_out.getData<float>() + i * hidden_size;
      
      for (size_t j = 0; j < hidden_size; ++j) {
        out_ptr[j] += (down_ptr[j] + down_bias.getValue<float>(0, 0, 0, j)) * weight;
      }
    }
  } else {
    // Single token processing (original code for edge case)
    const unsigned token_idx = token_assignments[0].first;
    const float weight = token_assignments[0].second;
    
    nntrainer::TensorDim token_dim({1, 1, 1, hidden_size}, input.getTensorType());
    nntrainer::TensorDim inter_dim({1, 1, 1, intermediate_size}, input.getTensorType());
    
    size_t token_offset = token_idx * hidden_size;
    nntrainer::Tensor token_input = input.getSharedDataTensor(token_dim, token_offset, true);
    
    nntrainer::Tensor gate_out(inter_dim);
    nntrainer::Tensor up_out(inter_dim);
    
    token_input.dot(gate_proj, gate_out);
    token_input.dot(up_proj, up_out);
    
    gate_out.add_i(gate_bias);
    up_out.add_i(up_bias);
    
    // Clamp values
    float *gate_ptr = gate_out.getData<float>();
    float *up_ptr = up_out.getData<float>();
    for (size_t j = 0; j < intermediate_size; ++j) {
      gate_ptr[j] = std::min(gate_ptr[j], limit);
      up_ptr[j] = std::max(-limit, std::min(limit, up_ptr[j]));
    }
    
    // Apply swiglu
    for (size_t j = 0; j < intermediate_size; ++j) {
      float sigmoid = 1.0f / (1.0f + std::exp(-alpha * gate_ptr[j]));
      gate_ptr[j] = gate_ptr[j] * sigmoid * (up_ptr[j] + 1.0f);
    }
    
    nntrainer::Tensor output(token_dim);
    gate_out.dot(down_proj, output);
    output.add_i(down_bias);
    output.multiply_i(weight);
    
    nntrainer::Tensor token_output = expert_output.getSharedDataTensor(token_dim, token_offset, true);
    token_output.add_i(output);
  }
}

void CachedSlimGptOssMoELayer::setProperty(
  const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, moe_props);
  nntrainer::LayerImpl::setProperty(remain_props);
}

void CachedSlimGptOssMoELayer::calcDerivative(
  nntrainer::RunLayerContext &context) {
  // MoE layer does not support derivative calculation
  throw std::runtime_error("MoE layer does not support derivative calculation");
}

void CachedSlimGptOssMoELayer::calcGradient(
  nntrainer::RunLayerContext &context) {
  // MoE layer does not support gradient calculation
  throw std::runtime_error("MoE layer does not support gradient calculation");
}

void CachedSlimGptOssMoELayer::exportTo(
  nntrainer::Exporter &exporter, const ml::train::ExportMethods &method) const {
  nntrainer::LayerImpl::exportTo(exporter, method);
  exporter.saveResult(moe_props, method, this); // Save MoE specific properties
}

} // namespace causallm