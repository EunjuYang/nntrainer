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
 * @file	qwen3_moe_causallm.cpp
 * @date	23 July 2025
 * @brief	This defines a qwen3_moe causal language model.
 * @see		https://github.com/nnstreamer/
 * @author	Eunju Yang <ej.yang@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */
#include <llm_util.hpp>
#include <model.h>
#include <qwen3_apex.h>

#include <app_context.h>
#include <engine.h>
#include <qwen_moe_layer_apex.h>

namespace causallm {

void Qwen3ApexMoECausalLM::setupParameters(json &cfg, json &generation_cfg,
                                           json &nntr_cfg) {
  Qwen3CausalLM(cfg, generation_cfg, nntr_cfg);

  // parameters for Qwen3MoE model
  try {
    NUM_EXPERTS = cfg["num_experts"];
    NUM_EXPERTS_PER_TOK = cfg["num_experts_per_tok"];
    INTERMEDIATE_SIZE = cfg["moe_intermediate_size"];
    USE_K = nntr_cfg["use_k"].get<std::vector<unsigned int>>();
    CACHE_SIZE = nntr_cfg["cache_size"].get<std::vector<unsigned int>>();
  } catch (const std::exception &e) {
    throw std::runtime_error("Qwen3MoE: num_experts and num_experts_per_tok "
                             "are not specified in the config file");
  }
}

std::vector<LayerHandle>
Qwen3ApexMoECausalLM::createMlp(const int layer_id, int dim, int hidden_dim,
                                std::string input_name) {

  std::vector<LayerHandle> layers;
  layers.push_back(createLayer(
    "moe_apex",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_down"),
     withKey("input_layers", input_name), withKey("unit", hidden_dim),
     withKey("num_experts", NUM_EXPERTS),
     withKey("num_experts_per_token", NUM_EXPERTS_PER_TOK),
     withKey("moe_activation", "swish"), withKey("use_k", USE_K[layer_id]),
     withKey("cache_size", CACHE_SIZE[layer_id])}));

  return layers;
}

void Qwen3ApexMoECausalLM::registerCustomLayers() {

  Qwen3CausalLM::registerCustomLayers();
  auto &ct_engine = nntrainer::Engine::Global();
  auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));

  try {
    app_context->registerFactory(
      nntrainer::createLayer<causallm::ApexMoeLayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
  }
}

} // namespace causallm
