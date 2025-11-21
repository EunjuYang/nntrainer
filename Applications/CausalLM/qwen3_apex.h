// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   qwen3_moe_causallm.h
 * @date   15 July 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __QWEN3_APEX_H__
#define __QWEN3_APEX_H__

#include <causal_lm.h>
#include <qwen3_causallm.h>

namespace causallm {

/**
 * @brief Qwen3ApexMoECausalLM class
 * @note  This class inherits Qwewn3CaUSALlm
 */
class Qwen3ApexMoECausalLM : public Qwen3CausalLM {

public:
  static constexpr const char *architectures = "Qwen3ApexMoECausalLM";

  Qwen3ApexMoECausalLM(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Qwen3CausalLM(cfg, generation_cfg, nntr_cfg) {
    setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  virtual ~Qwen3ApexMoECausalLM() = default;

  std::vector<LayerHandle> createMlp(const int layer_id, int dim,
                                     int hidden_dim,
                                     std::string input_name) override;

  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;

  void registerCustomLayers() override;

private:
  unsigned int NUM_EXPERTS;
  unsigned int NUM_EXPERTS_PER_TOK;
  std::vector<unsigned int> USE_K;
  std::vector<unsigned int> CACHE_SIZE;
};
}; // namespace causallm

#endif /* __QWEN3_APEX_H__ */
