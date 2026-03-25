// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   qwen3_omni_moe_causallm.h
 * @date   25 March 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   Qwen3OmniMoECausalLM supports Qwen3-Omni-MoE thinker text model.
 *         Key differences from Qwen3MoECausalLM:
 *         - Supports mixed dense/MoE layers via decoder_sparse_step and
 *           mlp_only_layers config parameters.
 *         - Fused expert weights (gate_up_proj) are handled in the weight
 *           converter (split into separate gate_proj/up_proj per expert).
 * @see
 * https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Thinking
 */

#ifndef __QWEN3_OMNI_MOE_CAUSAL_LM_H__
#define __QWEN3_OMNI_MOE_CAUSAL_LM_H__

#include <qwen3_moe_causallm.h>

namespace causallm {

/**
 * @brief Qwen3OmniMoECausalLM class
 * @note  This class inherits Qwen3MoECausalLM and adds support for
 *        mixed dense/MoE layers (decoder_sparse_step, mlp_only_layers).
 */
class Qwen3OmniMoECausalLM : public Qwen3MoECausalLM {

public:
  static constexpr const char *architectures = "Qwen3OmniMoeForCausalLM";

  Qwen3OmniMoECausalLM(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg, ModelType::CAUSALLM),
    Qwen3MoECausalLM(cfg, generation_cfg, nntr_cfg) {
    setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  virtual ~Qwen3OmniMoECausalLM() = default;

  std::vector<LayerHandle> createMlp(const int layer_id, int dim,
                                     int hidden_dim,
                                     std::string input_name) override;

  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;

private:
  unsigned int DECODER_SPARSE_STEP;
  int DENSE_INTERMEDIATE_SIZE;
  std::vector<int> MLP_ONLY_LAYERS;

  /**
   * @brief Check if a given layer should use MoE or dense MLP
   * @param layer_id The layer index
   * @return true if the layer should use MoE
   */
  bool isMoELayer(int layer_id) const;
};
}; // namespace causallm

#endif /* __QWEN3_OMNI_MOE_CAUSAL_LM_H__ */
