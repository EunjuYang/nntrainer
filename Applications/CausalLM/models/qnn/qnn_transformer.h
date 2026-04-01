// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   qnn_transformer.h
 * @date   31 Mar 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   This qnn_transformer.h constructs a class for QNN-based Transformer
 * model. Unlike the NNTrainer Transformer, this class loads a pre-compiled QNN
 * graph binary and uses the QNN runtime for inference.
 */

#ifndef __QNN_TRANSFORMER_H__
#define __QNN_TRANSFORMER_H__

#pragma once

#include <transformer_base.h>

namespace causallm {

/**
 * @brief QNNTransformer Class
 * @note  This class handles QNN-specific model initialization, weight loading,
 *        and inference. It reads a QNN-specific configuration format instead of
 *        the HuggingFace config.json format.
 */
WIN_EXPORT class QNNTransformer : virtual public TransformerBase {

public:
  /**
   * @brief Construct a new QNNTransformer object
   * @param cfg QNN model configuration
   * @param generation_cfg Configuration for generation
   * @param nntr_cfg Configuration for nntrainer runtime
   */
  QNNTransformer(json &cfg, json &generation_cfg, json &nntr_cfg,
                 ModelType model_type = ModelType::MODEL);

  /**
   * @brief Destroy the QNNTransformer object
   */
  virtual ~QNNTransformer() = default;

  /**
   * @brief Initialize the QNN model
   * @note  This loads the QNN context and prepares the graph for inference.
   */
  void initialize() override;

  /**
   * @brief Load model weights
   * @param weight_path Path to the QNN binary (graph:weights format)
   */
  void load_weight(const std::string &weight_path) override;

  /**
   * @brief Save model weights
   * @param weight_path Path to save
   */
  void save_weight(const std::string &weight_path) override;

  /**
   * @brief Run inference (simple)
   */
  void run(const WSTR prompt, void *output_buf = nullptr,
           bool log_output = true) override;

  /**
   * @brief Run inference (full)
   */
  void run(const WSTR prompt, const WSTR system_prompt,
           const WSTR tail_prompt, void *output_buf = nullptr,
           bool log_output = true) override;

protected:
  /**
   * @brief Setup QNN-specific parameters from configuration
   */
  virtual void setupParameters(json &cfg, json &generation_cfg,
                                json &nntr_cfg);

  /**
   * @brief Construct the QNN model graph
   */
  virtual void constructModel();

  std::string QNN_CONTEXT_BIN; /**< Path to QNN context binary */

  std::string MODEL_TENSOR_TYPE;
  std::string EMBEDDING_DTYPE;
  std::string FC_LAYER_DTYPE;

  bool MEMORY_SWAP = false;
  unsigned int FSU_LOOKAHEAD = 1;
};

} // namespace causallm

#endif /* __QNN_TRANSFORMER_H__ */
