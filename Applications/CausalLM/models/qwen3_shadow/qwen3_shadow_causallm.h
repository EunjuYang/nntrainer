// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   qwen3_causallm.h
 * @date   10 July 2025
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   Please refer to the following code :
 *  https://github.com/huggingface/transformers/blob/v4.52.3/src/transformers/models/qwen3/modeling_qwen3.py
 */

#ifndef __QWEN3_SHADOW_CAUSAL_LM_H__
#define __QWEN3_SHADOW_CAUSAL_LM_H__

#include <causal_lm.h>

namespace causallm {

/**
 * @brief Qwen3ShadowTransformer class
 */
class Qwen3ShadowTransformer : virtual public Transformer {
public:
  static constexpr const char *architectures = "Qwen3ShadowTransformer";

  Qwen3ShadowTransformer(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg) {
    if (nntr_cfg.contains("sparse_ratios")) {
      sparse_ratios_str = nntr_cfg["sparse_ratios"];
    }
  }

  virtual ~Qwen3ShadowTransformer() = default;

  std::vector<LayerHandle> createAttention(const int layer_id, int seq_len,
                                           int n_heads, int head_dim,
                                           std::string query_name,
                                           std::string key_name,
                                           std::string value_name) override;

  void registerCustomLayers() override;

private:
  std::string sparse_ratios_str;
};

/**
 * @brief Qwen3ShadowCausalLM class
 */
class Qwen3ShadowCausalLM : public CausalLM, public Qwen3ShadowTransformer {

public:
  static constexpr const char *architectures = "Qwen3ShadowForCausalLM";

  Qwen3ShadowCausalLM(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg, ModelType::CAUSALLM),
    CausalLM(cfg, generation_cfg, nntr_cfg),
    Qwen3ShadowTransformer(cfg, generation_cfg, nntr_cfg) {}

  virtual ~Qwen3ShadowCausalLM() = default;

  void registerCustomLayers() override;

private:
};
} // namespace causallm

#endif /* __QWEN3_SHADOW_CAUSAL_LM_H__ */
