/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
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
 * @file	qwen3_omni_moe_causallm.cpp
 * @date	25 March 2026
 * @brief	This defines a qwen3_omni_moe causal language model.
 * @see		https://github.com/nnstreamer/
 * @author	Eunju Yang <ej.yang@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */
#include <algorithm>
#include <llm_util.hpp>
#include <model.h>
#include <qwen3_omni_moe_causallm.h>

namespace causallm {

void Qwen3OmniMoECausalLM::setupParameters(json &cfg, json &generation_cfg,
                                            json &nntr_cfg) {
  // Save dense intermediate_size before parent overwrites INTERMEDIATE_SIZE
  DENSE_INTERMEDIATE_SIZE =
    cfg.value("intermediate_size", cfg.value("moe_intermediate_size", 768));

  // Call parent which sets NUM_EXPERTS, NUM_EXPERTS_PER_TOK,
  // INTERMEDIATE_SIZE (= moe_intermediate_size)
  Qwen3MoECausalLM::setupParameters(cfg, generation_cfg, nntr_cfg);

  // Omni-MoE specific parameters
  DECODER_SPARSE_STEP = cfg.value("decoder_sparse_step", 1);

  MLP_ONLY_LAYERS.clear();
  if (cfg.contains("mlp_only_layers") && !cfg["mlp_only_layers"].is_null()) {
    MLP_ONLY_LAYERS = cfg["mlp_only_layers"].get<std::vector<int>>();
  }
}

bool Qwen3OmniMoECausalLM::isMoELayer(int layer_id) const {
  // Check if layer is in mlp_only_layers
  if (std::find(MLP_ONLY_LAYERS.begin(), MLP_ONLY_LAYERS.end(), layer_id) !=
      MLP_ONLY_LAYERS.end()) {
    return false;
  }

  // Check decoder_sparse_step: MoE if (layer_id + 1) % step == 0
  return (layer_id + 1) % DECODER_SPARSE_STEP == 0;
}

std::vector<LayerHandle>
Qwen3OmniMoECausalLM::createMlp(const int layer_id, int dim, int hidden_dim,
                                 std::string input_name) {

  if (isMoELayer(layer_id)) {
    // Use MoE layer (from Qwen3MoECausalLM)
    return Qwen3MoECausalLM::createMlp(layer_id, dim, hidden_dim, input_name);
  } else {
    // Use dense MLP with DENSE_INTERMEDIATE_SIZE
    return Transformer::createMlp(layer_id, dim, DENSE_INTERMEDIATE_SIZE,
                                  input_name);
  }
}

} // namespace causallm
