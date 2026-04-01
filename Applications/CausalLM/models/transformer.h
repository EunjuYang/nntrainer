// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   transformer.h
 * @date   31 Dec 2025
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   This transformer.h constructs a class for Transformer model which can
 * be a parent of CausalLM and Encoder models with transformer structure.
 * @note   This transformer assumes the following structure :
 *
 *           [Input]
 *              |
 *         [Embedding]
 *              |
 *        [Decoder Block] (repeated N times)
 *              |
 *          [RMSNorm]
 *
 */
#ifndef __TRANSFORMER_H__
#define __TRANSFORMER_H__

#pragma once

#include <limits.h>
#include <random>

#include "performance_metrics.h"
#include <transformer_base.h>

namespace causallm {

/**
 * @brief Transformer Class
 */
WIN_EXPORT class Transformer : virtual public TransformerBase {

public:
  /**
   * @brief Construct a new Transformer object
   * @param cfg Configuration for the model (config.json)
   * @param generation_cfg Configuration for the generation (generation.json)
   * @param nntr_cfg Configuration for nntrainer (nntrainer_config.json)
   * @param model_type Type of the model (default: ModelType::MODEL)
   */
  Transformer(json &cfg, json &generation_cfg, json &nntr_cfg,
              ModelType model_type = ModelType::MODEL);

  /**
   * @brief Destroy the Transformer object
   */
  virtual ~Transformer() {}

  /**
   * @brief Initialize and Construct the Transformer model
   */
  void initialize() override;

  /**
   * @brief Load the model weights from a file
   */
  void load_weight(const std::string &weight_path) override;

  /**
   * @brief Save the weight to a file
   */
  void save_weight(const std::string &weight_path) override;

  /**
   * @brief run the Transformer model (simple)
   */
  void run(const WSTR prompt, void *output_buf = nullptr,
           bool log_output = true) override;

  /**
   * @brief run the Transformer model (full)
   */
  void run(const WSTR prompt, const WSTR system_prompt,
           const WSTR tail_prompt, void *output_buf = nullptr,
           bool log_output = true) override;

  /**
   * @brief Get PerformanceMetrics
   */
  PerformanceMetrics getPerformanceMetrics() const {
    return performance_metrics;
  }

protected:
  /**
   * @brief Setup the parameters for the Transformer model
   */
  virtual void setupParameters(json &cfg, json &generation_cfg, json &nntr_cfg);

  /**
   * @brief Construct Model
   */
  virtual void constructModel();

  /**
   * @brief create Attention Layer
   */
  virtual std::vector<LayerHandle>
  createTransformerDecoderBlock(const int layer_id, std::string input_name);

  /**
   * @brief create Attention Layer
   */
  virtual std::vector<LayerHandle>
  createAttention(const int layer_id, int seq_len, int n_heads, int head_dim,
                  std::string query_name, std::string key_name,
                  std::string value_name);

  /**
   * @brief create Feed Forward Layer
   */
  virtual std::vector<LayerHandle> createMlp(const int layer_id, int dim,
                                             int hidden_dim,
                                             std::string input_name);

  /**
   * @brief register CustomLayers
   */
  virtual void registerCustomLayers();

  int HEAD_DIM;
  int INTERMEDIATE_SIZE;
  bool USE_VOCAB_SELECTION;
  bool TIE_WORD_EMBEDDINGS;
  int NUM_HEADS;
  int NUM_KEY_VALUE_HEADS;
  std::string MODEL_TENSOR_TYPE;
  std::string EMBEDDING_DTYPE; /** embedding dtype */
  std::string FC_LAYER_DTYPE;  /** custom_fc_lora */

  unsigned int SLIDING_WINDOW = UINT_MAX;
  unsigned int SLIDING_WINDOW_PATTERN = 5;
  unsigned int ROPE_THETA = 10000; /**< RoPE theta value */
  float NORM_EPS = 1e-5;          /**< RMSNorm epsilon value */
  float EMBEDDING_SCALE = 1.0f;
  int GQA_SIZE;

  unsigned int MAX_POSITION_EMBEDDINGS; /**< max_position embeddings */
  bool MEMORY_SWAP;                     /**< memory swap option */
  unsigned int FSU_LOOKAHEAD;
  float ATTN_LOGIT_SOFTCAPPING = 0.0f; /**< attention logit softcapping */
  bool IS_CAUSAL = true;

  // Performance metrics
  PerformanceMetrics performance_metrics;
};

} // namespace causallm

#endif
