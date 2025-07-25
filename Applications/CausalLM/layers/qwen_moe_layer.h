// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   moe_layer.h
 * @date   09 June 2025
 * @brief  This is Mixture of Expert Layer Class of Neural Network
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   This file is part of the Mixture of Expert Layer implementation.
 *         It does not support shared experts.
 *         This layer is implemented based on the LLama-MoE.
 *         For more information, please refer to the following link:
 *         https://arxiv.org/pdf/2406.16554
 * @todo   This layer does not support backwarding yet.
 */

#ifndef __MOE_LAYER_H__
#define __MOE_LAYER_H__
#ifdef __cplusplus

#include <acti_func.h>
#include <common_properties.h>
#include <layer_impl.h>

namespace causallm {

namespace props {
/**
 * @brief MoE activation type
 */
class MoEActivation final
  : public nntrainer::EnumProperty<nntrainer::props::ActivationTypeInfo> {
public:
  using prop_tag = nntrainer::enum_class_prop_tag;
  static constexpr const char *key = "moe_activation";
};
/**
 * @brief NumExperts,  Number of experts property
 */
class NumExperts : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key = "num_experts"; /**< unique key to access */
  using prop_tag = nntrainer::uint_prop_tag;        /**< property type */
};

/**
 * @brief NumExpertsPerToken,  Number of experts per token property
 */
class NumExpertsPerToken : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key =
    "num_experts_per_token";                 /**< unique key to access */
  using prop_tag = nntrainer::uint_prop_tag; /**< property type */
};
} // namespace props
/**
 * @class   MoELayer
 * @brief   Mixture of Expert Layer
 */
class MoELayer : public nntrainer::LayerImpl {
public:
  /**
   * @brief     Constructor of Mixture of Expert Layer
   */
  MoELayer();

  /**
   * @brief     Destructor of Mixture of Expert Layer
   */
  ~MoELayer() = default;

  /**
   * @brief  Move constructor.
   *  @param[in] MoELayer &&
   */
  MoELayer(MoELayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @param[in] rhs MoELayer to be moved.
   */
  MoELayer &operator=(MoELayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(nntrainer::InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(nntrainer::RunLayerContext &context, bool training) override;

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
  const std::string getType() const override { return MoELayer::type; };

  /**
   * @brief Layer::supportBackwarding()
   */
  bool supportBackwarding() const override { return false; }

  static constexpr const char *type = "qwen_moe"; /**< type of the layer */

private:
  unsigned int num_experts;      /**< number of experts */
  unsigned int topk;             /**< number of experts per token, i.e., topk */
  nntrainer::ActiFunc acti_func; /**< activation function for the expert */
  std::tuple<props::NumExperts, props::NumExpertsPerToken,
             nntrainer::props::Unit, props::MoEActivation>
    moe_props;

  // weight indeices
  std::vector<unsigned int> expert_gate_proj_indices;
  std::vector<unsigned int> expert_up_proj_indices;
  std::vector<unsigned int> expert_down_proj_indices;
  unsigned int gate_idx;

  // Intermediate tensor indices
  unsigned int router_logits_idx;
  unsigned int expert_mask_idx;

  // Optimization-related members
  std::vector<std::vector<std::pair<unsigned, float>>> cached_expert_assignments; /**< cached expert assignments to avoid repeated allocation */
  std::vector<unsigned int> active_experts; /**< list of active expert indices for current batch */
  std::vector<unsigned int> expert_workload; /**< workload count per expert for load balancing */

  /**
   * @brief expert forward computation without memory copies
   * @param input Input tensor (reshaped to [total_tokens, 1, 1, hidden_size])
   * @param output Output tensor to accumulate results
   * @param token_assignments Vector of (token_index, weight) pairs for this
   * expert
   * @param gate_proj Gate projection weight tensor
   * @param up_proj Up projection weight tensor
   * @param down_proj Down projection weight tensor
   * @param hidden_size Hidden dimension size
   */
  inline void compute_expert_forward(
    const nntrainer::Tensor &input, nntrainer::Tensor &output,
    const std::vector<std::pair<unsigned, float>> &token_assignments,
    const nntrainer::Tensor &gate_proj, const nntrainer::Tensor &up_proj,
    const nntrainer::Tensor &down_proj, unsigned int hidden_size);

  /**
   * @brief optimized expert forward computation with better cache locality
   * @param input Input tensor (reshaped to [total_tokens, 1, 1, hidden_size])
   * @param output Output tensor to accumulate results
   * @param token_assignments Vector of (token_index, weight) pairs for this expert
   * @param gate_proj Gate projection weight tensor
   * @param up_proj Up projection weight tensor
   * @param down_proj Down projection weight tensor
   * @param hidden_size Hidden dimension size
   * @note Maintains exact same behavior as compute_expert_forward but with performance optimizations
   */
  inline void compute_expert_forward_fast(
    const nntrainer::Tensor &input, nntrainer::Tensor &output,
    const std::vector<std::pair<unsigned, float>> &token_assignments,
    const nntrainer::Tensor &gate_proj, const nntrainer::Tensor &up_proj,
    const nntrainer::Tensor &down_proj, unsigned int hidden_size);

  /**
   * @brief optimized expert mask clearing with minimal memory bandwidth
   * @param expert_mask Expert mask tensor to clear
   * @param total_tokens Total number of tokens
   * @param topk Number of experts per token
   * @note Only clears the portion of mask that will be used
   */
  inline void optimize_expert_mask_clear(nntrainer::Tensor &expert_mask,
                                        unsigned int total_tokens, 
                                        unsigned int topk);

  /**
   * @brief optimized expert mask setting and assignment building
   * @param expert_mask Expert mask tensor to set
   * @param expert_assignments Expert assignment vectors to build
   * @param indices_data TopK indices data
   * @param values_data TopK values data  
   * @param total_tokens Total number of tokens
   * @param topk Number of experts per token
   * @param num_experts Total number of experts
   * @note Uses cache-friendly access patterns and specialized paths for different token counts
   */
  /**
   * @brief optimized expert mask setting and assignment building
   * @param expert_mask Expert mask tensor to set
   * @param expert_assignments Expert assignment vectors to build
   * @param indices_data TopK indices data
   * @param values_data TopK values data  
   * @param total_tokens Total number of tokens
   * @param topk Number of experts per token
   * @param num_experts Total number of experts
   * @note Uses cache-friendly access patterns and specialized paths for different token counts
   */
  inline void optimize_expert_mask_and_assignments(
    nntrainer::Tensor &expert_mask,
    std::vector<std::vector<std::pair<unsigned, float>>> &expert_assignments,
    const uint32_t *indices_data, const float *values_data,
    unsigned int total_tokens, unsigned int topk, unsigned int num_experts);

  /**
   * @brief optimized expert mask setting and assignment building for 8-expert case
   * @param expert_mask Expert mask tensor to set
   * @param expert_assignments Expert assignment vectors to build
   * @param active_experts List to store active expert indices
   * @param indices_data TopK indices data
   * @param values_data TopK values data  
   * @param total_tokens Total number of tokens
   * @param topk Number of experts per token
   * @param num_experts Total number of experts
   * @note Specialized for typical 8-expert incremental inference pattern
   */
  inline void optimize_expert_mask_and_assignments_8expert(
    nntrainer::Tensor &expert_mask,
    std::vector<std::vector<std::pair<unsigned, float>>> &expert_assignments,
    std::vector<unsigned int> &active_experts,
    const uint32_t *indices_data, const float *values_data,
    unsigned int total_tokens, unsigned int topk, unsigned int num_experts);

  /**
   * @brief process experts with adaptive parallel execution strategy
   * @param input Input tensor
   * @param output Output tensor  
   * @param context Run layer context
   * @param hidden_size Hidden dimension size
   * @param active_count Number of active experts
   * @note Uses different scheduling strategies based on expert count:
   *       - 1 expert: sequential execution
   *       - 2-4 experts: static scheduling  
   *       - 5-12 experts: dynamic scheduling (includes typical 8-expert case)
   *       - >12 experts: dynamic scheduling with larger chunks
   */
  inline void process_experts_optimized(
    const nntrainer::Tensor &input, nntrainer::Tensor &output,
    nntrainer::RunLayerContext &context, unsigned int hidden_size,
    unsigned int active_count);
};
} // namespace causallm

#endif /* __cplusplus */
#endif /* __MOE_LAYER_H__ */
