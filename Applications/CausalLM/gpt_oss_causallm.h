// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   gpt_oss_causallm.h
 * @date   January 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author AI Assistant
 * @bug    No known bugs except for NYI items
 * @note   This file implements GptOss model for CausalLM
 *         Based on: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_oss/
 */

#ifndef __GPT_OSS_CAUSAL_LM_H__
#define __GPT_OSS_CAUSAL_LM_H__

#include <causal_lm.h>

namespace causallm {

/**
 * @brief GptOssCausalLM class
 * @note  This class implements the GptOss architecture for causal language modeling
 *        GptOss is a GPT-style model with optimizations for efficiency
 */
class GptOssCausalLM : public CausalLM {

public:
  static constexpr const char *architectures = "GptOssForCausalLM";

  /**
   * @brief Construct a new GptOssCausalLM object
   * @param cfg Configuration for the model (config.json)
   * @param generation_cfg Configuration for generation (generation_config.json)
   * @param nntr_cfg Configuration for nntrainer (nntrainer_config.json)
   */
  GptOssCausalLM(json &cfg, json &generation_cfg, json &nntr_cfg) :
    CausalLM(cfg, generation_cfg, nntr_cfg) {
    setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  /**
   * @brief Destroy the GptOssCausalLM object
   */
  virtual ~GptOssCausalLM() = default;

  /**
   * @brief Setup parameters specific to GptOss model
   * @param cfg Model configuration
   * @param generation_cfg Generation configuration
   * @param nntr_cfg NNTrainer configuration
   */
  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;

  /**
   * @brief Create attention layers for GptOss
   * @param layer_id Layer index
   * @param seq_len Sequence length
   * @param n_heads Number of attention heads
   * @param head_dim Dimension per head
   * @param query_name Query layer name
   * @param key_name Key layer name
   * @param value_name Value layer name
   * @return Vector of layer handles
   */
  std::vector<LayerHandle> createAttention(const int layer_id, int seq_len,
                                           int n_heads, int head_dim,
                                           std::string query_name,
                                           std::string key_name,
                                           std::string value_name) override;

  /**
   * @brief Create MLP (feed-forward) layers for GptOss
   * @param layer_id Layer index
   * @param dim Model dimension
   * @param hidden_dim Hidden dimension
   * @param input_name Input layer name
   * @return Vector of layer handles
   */
  std::vector<LayerHandle> createMlp(const int layer_id, int dim,
                                     int hidden_dim,
                                     std::string input_name) override;

  /**
   * @brief Create Transformer decoder block for GptOss
   * @param layer_id Layer index
   * @param input_name Input layer name
   * @return Vector of layer handles
   */
  std::vector<LayerHandle>
  createTransformerDecoderBlock(const int layer_id, 
                                std::string input_name) override;

  /**
   * @brief Register custom layers specific to GptOss
   */
  void registerCustomLayers() override;

  /**
   * @brief Construct the complete GptOss model
   */
  void constructModel() override;

private:
  // GptOss specific parameters
  std::string ACTIVATION_FUNCTION = "gelu";  /**< Activation function type */
  bool USE_BIAS = true;                      /**< Whether to use bias in linear layers */
  bool USE_LAYER_NORM = true;                /**< Whether to use layer normalization */
  float LAYER_NORM_EPS = 1e-5;               /**< Layer normalization epsilon */
  float DROPOUT_RATE = 0.1;                  /**< Dropout rate */
  float ATTENTION_DROPOUT = 0.1;             /**< Attention dropout rate */
  bool USE_CACHE = false;                    /**< Whether to use KV cache */
  bool SCALE_EMBEDDINGS = false;             /**< Whether to scale embeddings */
  unsigned int MAX_CONTEXT_LENGTH = 2048;    /**< Maximum context length */
  std::string POSITION_EMBEDDING_TYPE = "learned"; /**< Position embedding type */
};

} // namespace causallm

#endif /* __GPT_OSS_CAUSAL_LM_H__ */