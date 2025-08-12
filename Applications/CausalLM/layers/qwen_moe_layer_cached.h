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

namespace causallm {

/**
 * @class   CachedSlimMoELayer
 * @brief   Mixture of Expert Layer with efficient caching
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
   * @brief  Move constructor - deleted due to atomic members
   */
  CachedSlimMoELayer(CachedSlimMoELayer &&rhs) noexcept = delete;

  /**
   * @brief  Move assignment operator - deleted due to atomic members
   */
  CachedSlimMoELayer &operator=(CachedSlimMoELayer &&rhs) = delete;

  /**
   * @brief  Copy constructor - deleted
   */
  CachedSlimMoELayer(const CachedSlimMoELayer &) = delete;

  /**
   * @brief  Copy assignment operator - deleted
   */
  CachedSlimMoELayer &operator=(const CachedSlimMoELayer &) = delete;

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

  // Simplified cache management - minimal overhead
  std::vector<bool> is_cached;   /**< O(1) lookup for cached status */
  unsigned int cache_head = 0;   /**< Circular buffer head for LRU */
  unsigned int cache_count = 0;  /**< Current number of cached experts */
  std::vector<int> cache_ring;   /**< Circular buffer for cached experts */
  std::vector<int> cache_position; /**< Position in cache ring (-1 if not cached) */
  
  // Dynamic cache sizing
  unsigned int base_cache_size = 16;
  unsigned int current_cache_size = 16;
  
  // Simple statistics (no atomic needed in single-threaded context)
  unsigned int cache_hits = 0;
  unsigned int cache_misses = 0;

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
   * @brief Update cache size based on expert diversity
   */
  void updateCacheSize(int unique_experts, int total_requests);
  
  /**
   * @brief Add expert to cache (handles eviction if needed)
   */
  void addToCache(int expert_idx, nntrainer::RunLayerContext &context);
  
  /**
   * @brief Update LRU position for cached expert
   */
  void updateCacheLRU(int expert_idx);
};
} // namespace causallm

#endif /* __cplusplus */
#endif /* __MOE_LAYER_H__ */
