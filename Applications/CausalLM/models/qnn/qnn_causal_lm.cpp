// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   qnn_causal_lm.cpp
 * @date   31 Mar 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This file defines QNNCausalLM's basic actions
 */

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include <llm_util.hpp>
#include <qnn_causal_lm.h>

namespace causallm {

QNNCausalLM::QNNCausalLM(json &cfg, json &generation_cfg, json &nntr_cfg) :
  QNNTransformer(cfg, generation_cfg, nntr_cfg, ModelType::CAUSALLM) {
  setupParameters(cfg, generation_cfg, nntr_cfg);
}

void QNNCausalLM::setupParameters(json &cfg, json &generation_cfg,
                                  json &nntr_cfg) {
  // Initialize output list
  for (unsigned int i = 0; i < BATCH_SIZE; ++i)
    output_list.push_back("");

  // allocate memory for the internal buffer
  ids_history = (unsigned int *)malloc(static_cast<size_t>(BATCH_SIZE) *
                                       MAX_SEQ_LEN * sizeof(unsigned int));

  BAD_WORD_IDS = nntr_cfg["bad_word_ids"].get<std::vector<unsigned int>>();
  NUM_BADWORDS = BAD_WORD_IDS.size();

  LMHEAD_DTYPE = nntr_cfg.contains("lmhead_dtype")
                   ? nntr_cfg["lmhead_dtype"]
                   : nntr_cfg["embedding_dtype"];

  if (generation_cfg["eos_token_id"].is_array()) {
    EOS_TOKEN_ID =
      generation_cfg["eos_token_id"].empty()
        ? cfg["eos_token_id"].get<std::vector<unsigned int>>()
        : generation_cfg["eos_token_id"].get<std::vector<unsigned int>>();
  } else {
    EOS_TOKEN_ID.clear();
    EOS_TOKEN_ID.push_back(generation_cfg["eos_token_id"].get<unsigned int>());
  }
  BOS_TOKEN_ID = generation_cfg["bos_token_id"].empty()
                   ? cfg["bos_token_id"].get<unsigned int>()
                   : generation_cfg["bos_token_id"].get<unsigned int>();
  TOP_K = generation_cfg.contains("top_k")
            ? generation_cfg["top_k"].get<unsigned int>()
            : 20;
  TOP_P = generation_cfg.contains("top_p")
            ? generation_cfg["top_p"].get<float>()
            : 0.95;
  TEMPERATURE = generation_cfg.contains("temperature")
                  ? generation_cfg["temperature"].get<float>()
                  : 0.7;
}

void QNNCausalLM::registerOutputs(
  std::unique_ptr<tokenizers::Tokenizer> &tokenizer,
  std::vector<unsigned int> ids, unsigned int pos,
  const std::vector<bool> &eos_list) {

  static const std::vector<char> puncts{',', '!', ':', ';', '?'};
  for (size_t b = 0; b < ids.size(); ++b) {
    if (!eos_list[b]) {
      ids_history[b * MAX_SEQ_LEN + pos] = ids[b];
      std::vector<int> decode_ids = {static_cast<int>(ids[b])};
      std::string decoded_str = tokenizer->Decode(decode_ids);

      if (std::find(puncts.begin(), puncts.end(), decoded_str.back()) !=
          puncts.end()) {
        // last symbol is a punctuation, hold on
      } else {
#if defined(_WIN32)
        std::wcout << L"" << utf8_to_wstring(decoded_str);
        std::wcout.flush();
#else
        std::cout << decoded_str;
        std::cout.flush();
#endif
        output_list[b].append(decoded_str);
      }
    }
  }
}

std::vector<unsigned int> QNNCausalLM::generate(float *logits, bool do_sample) {

  std::vector<unsigned int> outputs;
  for (unsigned int iteration = 0; iteration < BATCH_SIZE; ++iteration) {

    // apply bad words penalty
    if (BAD_WORD_IDS.size() != 0 && NUM_BADWORDS != 0) {
      applyBadWordsPenalty(logits, BAD_WORD_IDS.data(), NUM_BADWORDS);
    }

    if (do_sample == false) {
      unsigned int argmax_idx =
        std::distance(logits, std::max_element(logits, logits + NUM_VOCAB));
      outputs.push_back(argmax_idx);
    } else {
      float max_logits = applyTKP(logits, NUM_VOCAB, TEMPERATURE, TOP_K, TOP_P);
      float sum_exp_logits = 0;
      for (unsigned int i = 0; i < NUM_VOCAB; i++) {
        float exp_x = exp(logits[i] - max_logits);
        sum_exp_logits += exp_x;
        logits[i] = exp_x;
      }

      for (unsigned int i = 0; i < NUM_VOCAB; ++i) {
        logits[i] /= sum_exp_logits;
      }

      std::discrete_distribution<int> dist(logits, logits + NUM_VOCAB);
      unsigned int sampled_idx = dist(rng);
      outputs.push_back(sampled_idx);
    }

    logits = logits + NUM_VOCAB;
  }

  return outputs;
}

void QNNCausalLM::run(const WSTR prompt, bool do_sample,
                      const WSTR system_prompt, const WSTR tail_prompt) {
  if (!is_initialized) {
    throw std::runtime_error("QNNCausalLM model is not initialized. Please "
                             "call initialize() before run().");
  }

  output_list.clear();
  for (unsigned int b = 0; b < BATCH_SIZE; ++b) {
    output_list.push_back("");
  }

  /// @note The QNN inference loop should be implemented here.
  /// Unlike NNTrainer CausalLM which uses model->incremental_inference(),
  /// QNN CausalLM will use the QNN runtime API for graph execution.
  /// The generation loop (tokenize -> infer -> sample -> decode) follows
  /// the same pattern but uses QNN-specific APIs for the inference step.
}

} // namespace causallm
