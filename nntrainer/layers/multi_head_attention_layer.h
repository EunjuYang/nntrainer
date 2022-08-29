// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 hyeonseok Lee <hs89.lee@samsung.com>
 *
 * @file   multi_head_attention_layer.h
 * @date   08 July 2022
 * @see    https://github.com/nnstreamer/nntrainer
 *         https://arxiv.org/abs/1706.03762
 * @author hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is MultiHeadAttention Layer Class for Neural Network
 *
 */

#ifndef __MULTI_HEAD_ATTENTION_LAYER_H__
#define __MULTI_HEAD_ATTENTION_LAYER_H__
#ifdef __cplusplus

#include <acti_func.h>
#include <layer_impl.h>

namespace nntrainer {

/**
 * @class   Multi Head Attention Layer
 * @brief   Implementation of multi head attention which is described in paper
 * "Attention is all you need"
 */
class MultiHeadAttentionLayer : public LayerImpl {
public:
  /**
   * @brief     Constructor of MultiHeadAttention Layer
   */
  MultiHeadAttentionLayer();

  /**
   * @brief     Destructor of MultiHeadAttention Layer
   */
  ~MultiHeadAttentionLayer();

  /**
   *  @brief  Move constructor of MultiHeadAttentionLayer.
   *  @param[in] MultiHeadAttentionLayer &&
   */
  MultiHeadAttentionLayer(MultiHeadAttentionLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs MultiHeadAttentionLayer to be moved.
   */
  MultiHeadAttentionLayer &operator=(MultiHeadAttentionLayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(RunLayerContext &context) override;

  /**
   * @copydoc Layer::calcGradient(RunLayerContext &context)
   */
  void calcGradient(RunLayerContext &context) override;

  /**
   * @copydoc bool supportBackwarding() const
   */
  bool supportBackwarding() const override { return true; };

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override {
    return MultiHeadAttentionLayer::type;
  };

  /**
   * @copydoc Layer::setBatch(RunLayerContext &context, unsigned int batch)
   */
  void setBatch(RunLayerContext &context, unsigned int batch) override;

  inline static const std::string type = "multi_head_attention";

protected:
  /**
   * @brief     Finalize the multi head attention layer with the given context
   * @param[in] context InitLayerContext
   *
   * @note This function provides the basic finalize details which can be shared
   * with derived classes as well
   */
  void finalizeCommon(InitLayerContext &context);

private:
  std::tuple<props::NumHeads, props::ProjectedKeyDim, props::ProjectedValueDim,
             props::OutputShape, props::DropOutRate,
             props::ProvideAttentionMask, props::ReturnAttentionWeight,
             props::AverageAttentionWeight>
    multi_head_attention_props; /**< multi_head_attention layer properties */

  ActiFunc sm; /** softmax activation operation */
  std::array<unsigned int, 6>
    inout_idx; /**< indices of the inputs and outputs */
  std::array<unsigned int, 25>
    weight_idx; /**< indices of the weights and tensors */

  /**
   * @brief     to protect overflow
   */
  float epsilon;

  /**
   * @brief calculate common derivative
   * @param context Context of the layer
   */
  void calcCommonDerivative(RunLayerContext &context);
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __MULTI_HEAD_ATTENTION_LAYER_H__ */