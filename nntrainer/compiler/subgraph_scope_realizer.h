// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file subgraph_scope_realizer.h
 * @date 10 Jan 2025
 * @brief NNTrainer subgraph scope realizer
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug No known bugs except for NYI items
 */
#ifndef __SUBGRAPH_SCOPE_REALIZER_H__
#define __SUBGRAPH_SCOPE_REALIZER_H__
#include <common_properties.h>
#include <realizer.h>

namespace nntrainer {

/**
 * @brief SubGraph Realizer which adding some properties for subgraph
 * construction.
 */
class SubgraphScopeRealizer final : public GraphRealizer {
public:
  /**
   * @brief Destroy the subgraph realizer object
   */
  ~SubgraphScopeRealizer();
  /**
   * @brief realized graph
   */
  GraphRepresentation realize(const GraphRepresentation &reference) override;
};
} /* namespace nntrainer */
#endif /* __SUBGRAPH_REALIZER_H__ */
