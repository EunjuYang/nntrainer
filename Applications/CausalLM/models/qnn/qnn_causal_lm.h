// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   qnn_causal_lm.h
 * @date   31 Mar 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   This qnn_causal_lm.h constructs a class for QNN-based Causal
 * Language Model. It combines QNNTransformer's model loading with
 * generation logic (tokenization, sampling, EOS detection).
 *
 * @note   Structure:
 *
 *        [QNNTransformer]
 *              |
 *           [LMHead]   (part of QNN graph)
 */

#ifndef __QNN_CAUSAL_LM_H__
#define __QNN_CAUSAL_LM_H__

#pragma once

#include <random>

#include <qnn_transformer.h>

namespace causallm {

/**
 * @brief QNNCausalLM Class
 */
WIN_EXPORT class QNNCausalLM : virtual public QNNTransformer {

public:
  /**
   * @brief Construct a new QNNCausalLM object
   * @param cfg QNN model configuration
   * @param generation_cfg Configuration for generation
   * @param nntr_cfg Configuration for nntrainer runtime
   */
  QNNCausalLM(json &cfg, json &generation_cfg, json &nntr_cfg);

  /**
   * @brief Destroy the QNNCausalLM object
   */
  virtual ~QNNCausalLM() {
    if (ids_history)
      free(ids_history);
  }

  /**
   * @brief run the QNNCausalLM model
   */
  void run(const WSTR prompt, bool do_sample = false,
           const WSTR system_prompt = "",
           const WSTR tail_prompt = "") override;

protected:
  /**
   * @brief Setup the parameters for the QNNCausalLM model
   */
  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;

  /**
   * @brief generate next token from logits
   */
  std::vector<unsigned int> generate(float *logits, bool do_sample);

  /**
   * @brief register Outputs
   */
  virtual void
  registerOutputs(std::unique_ptr<tokenizers::Tokenizer> &tokenizer,
                  std::vector<unsigned int> ids, unsigned int pos,
                  const std::vector<bool> &eos_list);

  /** internal buffer */
  std::vector<std::string>
    output_list;             /**< List of output names for the model */
  unsigned int *ids_history; /**< History of input IDs for the model */

  std::string LMHEAD_DTYPE; /** lmhead dtype */
  std::vector<unsigned int> EOS_TOKEN_ID;
  unsigned int BOS_TOKEN_ID;
  float TEMPERATURE;
  unsigned int TOP_K;
  float TOP_P;

  std::vector<unsigned int> BAD_WORD_IDS; /**< List of bad word IDs */
  unsigned int NUM_BADWORDS;              /**< Number of bad words */

  std::mt19937 rng; /**< Random Number Gen */
};

} // namespace causallm

#endif /* __QNN_CAUSAL_LM_H__ */
