// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   qwen_moe_layer_cached.h
 * @date   09 June 2025
 * @brief  This is Mixture of Expert Layer Class of Neural Network
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __MOE_LAYER_H__
#define __MOE_LAYER_H__
#ifdef __cplusplus

#include <acti_func.h>
#include <causallm_common_properties.h>
#include <common_properties.h>
#include <layer_impl.h>
#include <deque>

namespace causallm {

/**
 * @class   CachedSlimMoELayer
 * @brief   Mixture of Expert Layer with adaptive caching
 */
class CachedSlimMoELayer : public nntrainer::LayerImpl {
public:
  /**
   * @brief     Constructor of Mixture of Expert Layer
   */
  CachedSlimMoELayer();

  /**
   * @brief     Destructor of Mixture of Expert Layer
   */
  ~CachedSlimMoELayer() = default;

  /**
   * @brief  Move constructor.
   * @param[in] CachedSlimMoELayer &&
   */
  CachedSlimMoELayer(CachedSlimMoELayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @param[in] rhs CachedSlimMoELayer to be moved.
   */
  CachedSlimMoELayer &operator=(CachedSlimMoELayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(nntrainer::InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(nntrainer::RunLayerContext &context,
                  bool training) override;

  /**
   * @copydoc Layer::incremental_forwarding(RunLayerContext &context, unsigned)
   */
  void incremental_forwarding(nntrainer::RunLayerContext &context,
                              unsigned int from, unsigned int to,
                              bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc Layer::calcGradient(RunLayerContext &context)
   */
  void calcGradient(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, const ml::train::ExportMethods
   * &methods)
   */
  void exportTo(nntrainer::Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override {
    return CachedSlimMoELayer::type;
  };

  /**
   * @brief Layer::supportBackwarding()
   */
  bool supportBackwarding() const override { return false; }

  static constexpr const char *type = "moe_cached_slim";

private:
  unsigned int num_experts;      /**< number of experts */
  unsigned int topk;             /**< number of experts per token */
  nntrainer::ActiFunc acti_func; /**< activation function */
  std::tuple<props::NumExperts, props::NumExpertsPerToken,
             nntrainer::props::Unit, props::MoEActivation>
    moe_props;

  // Weight indices
  std::vector<unsigned int> expert_gate_proj_indices;
  std::vector<unsigned int> expert_up_proj_indices;
  std::vector<unsigned int> expert_down_proj_indices;

  // Adaptive cache management
  std::vector<bool> is_cached;   /**< O(1) lookup for cached status */
  std::vector<int> cache_order;  /**< LRU tracking */
  unsigned int max_cached_experts = 16;
  unsigned int batch_cache_size = 24;  /**< Larger cache for batch processing */
  
  // Pattern tracking for incremental mode
  std::deque<std::vector<int>> recent_expert_patterns; /**< Recent expert usage */
  static constexpr size_t PATTERN_HISTORY_SIZE = 3;
  bool is_batch_mode = false;  /**< Current processing mode */
  
  // Expert usage frequency for adaptive caching
  std::vector<unsigned int> expert_usage_count; /**< Usage frequency tracking */

  unsigned int gate_idx;
  unsigned int router_logits_idx;
  unsigned int expert_mask_idx;

  /**
   * @brief Expert forward computation
   */
  inline void compute_expert_forward_no_critical(
    const nntrainer::Tensor &input, nntrainer::Tensor &expert_output,
    const std::vector<std::pair<unsigned, float>> &token_assignments,
    const nntrainer::Tensor &gate_proj, const nntrainer::Tensor &up_proj,
    const nntrainer::Tensor &down_proj, unsigned int hidden_size);
    
  /**
   * @brief Predict next experts based on recent patterns
   */
  std::vector<int> predictNextExperts(const std::vector<int> &current_experts);
};
} // namespace causallm

#endif /* __cplusplus */
#endif /* __MOE_LAYER_H__ */
