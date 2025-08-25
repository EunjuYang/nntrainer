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
 * @file   gpt_oss_causallm.cpp
 * @date   January 2025
 * @brief  This defines a GptOss causal language model.
 * @see    https://github.com/nnstreamer/nntrainer
 * @author AI Assistant
 * @bug    No known bugs except for NYI items
 */

#include <gpt_oss_causallm.h>
#include <llm_util.hpp>
#include <model.h>
#include <app_context.h>
#include <engine.h>
#include <layer.h>
#include <iostream>
#include <stdexcept>

namespace causallm {

void GptOssCausalLM::setupParameters(json &cfg, json &generation_cfg,
                                      json &nntr_cfg) {
  // Call parent class setup first
  CausalLM::setupParameters(cfg, generation_cfg, nntr_cfg);

  // Setup GptOss specific parameters
  try {
    // Model architecture parameters
    if (cfg.contains("hidden_act")) {
      ACTIVATION_FUNCTION = cfg["hidden_act"].get<std::string>();
    } else if (cfg.contains("activation_function")) {
      ACTIVATION_FUNCTION = cfg["activation_function"].get<std::string>();
    }

    if (cfg.contains("add_bias")) {
      USE_BIAS = cfg["add_bias"].get<bool>();
    }

    if (cfg.contains("layer_norm_epsilon")) {
      LAYER_NORM_EPS = cfg["layer_norm_epsilon"].get<float>();
    } else if (cfg.contains("layer_norm_eps")) {
      LAYER_NORM_EPS = cfg["layer_norm_eps"].get<float>();
    }

    if (cfg.contains("resid_pdrop")) {
      DROPOUT_RATE = cfg["resid_pdrop"].get<float>();
    } else if (cfg.contains("hidden_dropout_prob")) {
      DROPOUT_RATE = cfg["hidden_dropout_prob"].get<float>();
    }

    if (cfg.contains("attn_pdrop")) {
      ATTENTION_DROPOUT = cfg["attn_pdrop"].get<float>();
    } else if (cfg.contains("attention_probs_dropout_prob")) {
      ATTENTION_DROPOUT = cfg["attention_probs_dropout_prob"].get<float>();
    }

    if (cfg.contains("use_cache")) {
      USE_CACHE = cfg["use_cache"].get<bool>();
    }

    if (cfg.contains("scale_attn_weights")) {
      SCALE_EMBEDDINGS = cfg["scale_attn_weights"].get<bool>();
    }

    if (cfg.contains("n_ctx")) {
      MAX_CONTEXT_LENGTH = cfg["n_ctx"].get<unsigned int>();
    } else if (cfg.contains("max_position_embeddings")) {
      MAX_CONTEXT_LENGTH = cfg["max_position_embeddings"].get<unsigned int>();
    }

    if (cfg.contains("position_embedding_type")) {
      POSITION_EMBEDDING_TYPE = cfg["position_embedding_type"].get<std::string>();
    }

    // Override some parameters for GptOss
    if (cfg.contains("n_embd")) {
      DIM = cfg["n_embd"].get<int>();
    }
    
    if (cfg.contains("n_layer")) {
      NUM_LAYERS = cfg["n_layer"].get<int>();
    }
    
    if (cfg.contains("n_head")) {
      NUM_HEADS = cfg["n_head"].get<int>();
      HEAD_DIM = DIM / NUM_HEADS;
    }

    // Set intermediate size for MLP
    if (cfg.contains("n_inner")) {
      INTERMEDIATE_SIZE = cfg["n_inner"].get<int>();
    } else if (!cfg.contains("intermediate_size")) {
      // Default to 4 * hidden_size if not specified
      INTERMEDIATE_SIZE = 4 * DIM;
    }

    // GptOss typically doesn't use separate key-value heads
    NUM_KEY_VALUE_HEADS = NUM_HEADS;
    GQA_SIZE = 1;

    // No RoPE in standard GptOss, uses learned position embeddings
    ROPE_THETA = 0;

  } catch (const std::exception &e) {
    std::cerr << "Error setting up GptOss parameters: " << e.what() << std::endl;
    throw std::runtime_error("Failed to setup GptOss parameters: " + 
                             std::string(e.what()));
  }

  std::cout << "GptOss Model Configuration:" << std::endl;
  std::cout << "  - Hidden size: " << DIM << std::endl;
  std::cout << "  - Number of layers: " << NUM_LAYERS << std::endl;
  std::cout << "  - Number of heads: " << NUM_HEADS << std::endl;
  std::cout << "  - Head dimension: " << HEAD_DIM << std::endl;
  std::cout << "  - Intermediate size: " << INTERMEDIATE_SIZE << std::endl;
  std::cout << "  - Activation function: " << ACTIVATION_FUNCTION << std::endl;
  std::cout << "  - Max context length: " << MAX_CONTEXT_LENGTH << std::endl;
  std::cout << "  - Position embedding type: " << POSITION_EMBEDDING_TYPE << std::endl;
}

std::vector<LayerHandle>
GptOssCausalLM::createAttention(const int layer_id, int seq_len, int n_heads,
                                int head_dim, std::string query_name,
                                std::string key_name, std::string value_name) {
  std::vector<LayerHandle> layers;

  // Create QKV projection layer
  layers.push_back(createLayer(
    "qkv",
    {withKey("name", "layer" + std::to_string(layer_id) + "_qkv"),
     withKey("input_layers", query_name),
     withKey("num_heads", n_heads),
     withKey("head_dim", head_dim),
     withKey("add_bias", USE_BIAS)}));

  // Create multi-head attention core
  layers.push_back(createLayer(
    "mha_core",
    {withKey("name", "layer" + std::to_string(layer_id) + "_mha"),
     withKey("input_layers", "layer" + std::to_string(layer_id) + "_qkv"),
     withKey("num_heads", n_heads),
     withKey("head_dim", head_dim),
     withKey("seq_len", seq_len),
     withKey("dropout", ATTENTION_DROPOUT),
     withKey("scale", true)}));

  // Create output projection
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attn_out"),
     withKey("input_layers", "layer" + std::to_string(layer_id) + "_mha"),
     withKey("unit", DIM),
     withKey("bias_initializer", USE_BIAS ? "zeros" : "none"),
     withKey("activation", "none")}));

  // Add dropout if specified
  if (DROPOUT_RATE > 0) {
    layers.push_back(createLayer(
      "dropout",
      {withKey("name", "layer" + std::to_string(layer_id) + "_attn_dropout"),
       withKey("input_layers", "layer" + std::to_string(layer_id) + "_attn_out"),
       withKey("rate", DROPOUT_RATE)}));
  }

  return layers;
}

std::vector<LayerHandle>
GptOssCausalLM::createMlp(const int layer_id, int dim, int hidden_dim,
                          std::string input_name) {
  std::vector<LayerHandle> layers;

  // First linear layer (up projection)
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_mlp_fc1"),
     withKey("input_layers", input_name),
     withKey("unit", hidden_dim),
     withKey("bias_initializer", USE_BIAS ? "zeros" : "none"),
     withKey("activation", ACTIVATION_FUNCTION)}));

  // Second linear layer (down projection)
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_mlp_fc2"),
     withKey("input_layers", "layer" + std::to_string(layer_id) + "_mlp_fc1"),
     withKey("unit", dim),
     withKey("bias_initializer", USE_BIAS ? "zeros" : "none"),
     withKey("activation", "none")}));

  // Add dropout if specified
  if (DROPOUT_RATE > 0) {
    layers.push_back(createLayer(
      "dropout",
      {withKey("name", "layer" + std::to_string(layer_id) + "_mlp_dropout"),
       withKey("input_layers", "layer" + std::to_string(layer_id) + "_mlp_fc2"),
       withKey("rate", DROPOUT_RATE)}));
  }

  return layers;
}

std::vector<LayerHandle>
GptOssCausalLM::createTransformerDecoderBlock(const int layer_id,
                                              std::string input_name) {
  std::vector<LayerHandle> layers;
  std::string current_output = input_name;

  // Layer Normalization before attention
  layers.push_back(createLayer(
    "layer_normalization",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ln1"),
     withKey("input_layers", current_output),
     withKey("epsilon", LAYER_NORM_EPS),
     withKey("axis", -1)}));
  
  std::string ln1_output = "layer" + std::to_string(layer_id) + "_ln1";

  // Self-attention
  auto attention_layers = createAttention(
    layer_id, MAX_SEQ_LEN, NUM_HEADS, HEAD_DIM,
    ln1_output, ln1_output, ln1_output);
  layers.insert(layers.end(), attention_layers.begin(), attention_layers.end());

  // Get the last attention layer name
  std::string attn_output = attention_layers.back()->getName();

  // Residual connection for attention
  layers.push_back(createLayer(
    "addition",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attn_residual"),
     withKey("input_layers", current_output + "," + attn_output)}));
  
  current_output = "layer" + std::to_string(layer_id) + "_attn_residual";

  // Layer Normalization before MLP
  layers.push_back(createLayer(
    "layer_normalization",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ln2"),
     withKey("input_layers", current_output),
     withKey("epsilon", LAYER_NORM_EPS),
     withKey("axis", -1)}));
  
  std::string ln2_output = "layer" + std::to_string(layer_id) + "_ln2";

  // MLP
  auto mlp_layers = createMlp(layer_id, DIM, INTERMEDIATE_SIZE, ln2_output);
  layers.insert(layers.end(), mlp_layers.begin(), mlp_layers.end());

  // Get the last MLP layer name
  std::string mlp_output = mlp_layers.back()->getName();

  // Residual connection for MLP
  layers.push_back(createLayer(
    "addition",
    {withKey("name", "layer" + std::to_string(layer_id) + "_mlp_residual"),
     withKey("input_layers", current_output + "," + mlp_output)}));

  return layers;
}

void GptOssCausalLM::constructModel() {
  std::vector<LayerHandle> layers;
  
  // Token embedding layer
  layers.push_back(createLayer(
    "embedding",
    {withKey("name", "token_embedding"),
     withKey("in_dim", NUM_VOCAB),
     withKey("out_dim", DIM),
     withKey("input_layers", "input")}));
  
  std::string current_output = "token_embedding";

  // Position embedding (if using learned position embeddings)
  if (POSITION_EMBEDDING_TYPE == "learned") {
    layers.push_back(createLayer(
      "embedding",
      {withKey("name", "position_embedding"),
       withKey("in_dim", MAX_CONTEXT_LENGTH),
       withKey("out_dim", DIM),
       withKey("input_layers", "position_ids")}));
    
    // Add token and position embeddings
    layers.push_back(createLayer(
      "addition",
      {withKey("name", "embeddings_sum"),
       withKey("input_layers", "token_embedding,position_embedding")}));
    
    current_output = "embeddings_sum";
  }

  // Embedding dropout
  if (DROPOUT_RATE > 0) {
    layers.push_back(createLayer(
      "dropout",
      {withKey("name", "embedding_dropout"),
       withKey("input_layers", current_output),
       withKey("rate", DROPOUT_RATE)}));
    current_output = "embedding_dropout";
  }

  // Transformer decoder blocks
  for (int i = 0; i < NUM_LAYERS; i++) {
    auto decoder_layers = createTransformerDecoderBlock(i, current_output);
    layers.insert(layers.end(), decoder_layers.begin(), decoder_layers.end());
    current_output = "layer" + std::to_string(i) + "_mlp_residual";
  }

  // Final layer normalization
  layers.push_back(createLayer(
    "layer_normalization",
    {withKey("name", "final_ln"),
     withKey("input_layers", current_output),
     withKey("epsilon", LAYER_NORM_EPS),
     withKey("axis", -1)}));

  // LM head (output projection)
  if (TIE_WORD_EMBEDDINGS) {
    layers.push_back(createLayer(
      "tie_word_embedding",
      {withKey("name", "lm_head"),
       withKey("input_layers", "final_ln"),
       withKey("unit", NUM_VOCAB),
       withKey("reference_layer", "token_embedding"),
       withKey("activation", "none")}));
  } else {
    layers.push_back(createLayer(
      "fully_connected",
      {withKey("name", "lm_head"),
       withKey("input_layers", "final_ln"),
       withKey("unit", NUM_VOCAB),
       withKey("bias_initializer", USE_BIAS ? "zeros" : "none"),
       withKey("activation", "none")}));
  }

  // Set model layers
  model->setProperty({withKey("input_layers", "input")});
  
  for (auto &layer : layers) {
    model->addLayer(layer);
  }

  // Add output layer
  output_list.push_back("lm_head");
}

void GptOssCausalLM::registerCustomLayers() {
  // Register any custom layers needed for GptOss
  // Most layers should already be registered by the parent class
  CausalLM::registerCustomLayers();
  
  // If we need any GptOss-specific custom layers, register them here
  auto &ct_engine = nntrainer::Engine::Global();
  auto app_context = static_cast<nntrainer::AppContext *>(
    ct_engine.getRegisteredContext("cpu"));
  
  // Currently using standard layers, but we can add custom ones if needed
  try {
    // Example: Register a custom GptOss layer if needed
    // app_context->registerFactory(
    //   nntrainer::createLayer<causallm::GptOssCustomLayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "Failed to register GptOss custom layers: " << e.what() 
              << std::endl;
  }
}

} // namespace causallm