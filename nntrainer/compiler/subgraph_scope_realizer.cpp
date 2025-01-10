// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file subgraph_scope_realizer.cpp
 * @date 10 Jan 2025
 * @brief NNTrainer subgraph scope realizer
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <base_properties.h>
#include <common_properties.h>
#include <layer_node.h>
#include <nntrainer_error.h>
#include <node_exporter.h>
#include <remap_realizer.h>
#include <subgraph_scope_realizer.h>
#include <tuple>

#include <iostream>

namespace nntrainer {
SubgraphScopeRealizer::~SubgraphScopeRealizer() {}
/**
 * @note
 * subgraphrealize conducts the following two steps:
 *  1. rename all the nodes in the subgraph
 *     to be unique by adding scope as its prefix
 *  2. check input_layers property and if there are some
 *     layers that are not assigned graph_scope in name,
 *     add graph scoped input names
 * It only works only when name / input_layers do not contain "/"
 */
GraphRepresentation
SubgraphScopeRealizer::realize(const GraphRepresentation &reference) {
  GraphRepresentation processed;
  for (auto &node : reference) {
    processed.push_back(node);
    auto node_name = node->getGraphName();
    if (node_name.find("/") >= node_name.length())
      node->setName(node->getGraphName() + "/" + node->getName());
    for (unsigned int i = 0; i < node->getNumInputConnections(); ++i) {
      auto in_conn = node->getInputConnectionName(i);
      if (in_conn.find("/") >= in_conn.length())
        node->setInputConnectionName(i, node->getGraphName() + "/" + in_conn);
    }
  }
  return processed;
}
} // namespace nntrainer
