// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file    network_graph.h
 * @date    12 May 2020
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @author  Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   This is Graph Core Class for Neural Network
 *
 */

#include <algorithm>
#include <sstream>

#include <graph_core.h>

#include <activation_layer.h>
#include <addition_layer.h>
#include <bn_layer.h>
#include <concat_layer.h>
#include <connection.h>
#include <flatten_layer.h>
#include <grucell.h>
#include <identity_layer.h>
#include <input_layer.h>
#include <iostream>
#include <layer_node.h>
#include <layer_normalization_layer.h>
#include <lstmcell.h>
#include <multiout_layer.h>
#include <network_graph.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <profiler.h>
#include <rnn.h>
#include <rnncell.h>
#include <split_layer.h>
#include <time_dist.h>
#include <tracer.h>
#include <util_func.h>

namespace nntrainer {

std::vector<TensorDim> SubGraphBase::getOutputDimension() const {
  NNTR_THROW_IF(label_dims.empty(), std::invalid_argument)
    << "[NetworkGraph] the graph has no node identified as output!";
  /// for now, outputting label_dims works, later label dim will be different
  /// from output dimension
  return label_dims;
}

std::vector<TensorDim> SubGraphBase::getInputDimension() const {
  NNTR_THROW_IF(input_dims.empty(), std::invalid_argument)
    << "[NetworkGraph] the graph has no node identified as input!";
  return input_dims;
}

std::vector<std::shared_ptr<LayerNode>>
SubGraphBase::getUnsortedLayers(const std::string &input_layer,
                                const std::string &output_layer) const {

  /// @fixme: this won't work if input, output layers are not in order
  /// Further, this function must be removed. There should be rather
  /// getAllNames and getLayerByName instead of getUnsortedLayers.

  /** count layers after output layer */
  unsigned int num_layers_remove_end = 0;
  if (!output_layer.empty()) {
    for (auto iter = crbegin(); iter != crend(); iter++) {
      if ((*iter)->getName() != output_layer)
        num_layers_remove_end++;
      else
        break;
    }
  }

  if (num_layers_remove_end == size())
    return {};

  /** count layers before input layer */
  unsigned int num_layers_remove_start = 0;
  if (!input_layer.empty()) {
    for (auto iter = cbegin(); iter != cend() - num_layers_remove_end; iter++) {
      if ((*iter)->getName() != input_layer)
        num_layers_remove_start++;
      else
        break;
    }
  }

  /** copy the graph and return */
  std::vector<std::shared_ptr<LayerNode>> ret;
  std::transform(cbegin() + num_layers_remove_start,
                 cend() - num_layers_remove_end, std::back_inserter(ret),
                 [](auto const &elem) { return LNODE(elem); });

  return ret;
}

unsigned int SubGraphBase::setExecutionOrder() {

  auto backward_order = size();
  for (auto iter = getBackwardingBeginIter(); iter != getBackwardingEndIter();
       iter++) {
    auto &node = *iter;
    auto order_idx = getBackwardingEndIter() - iter - 1;
    auto forward_order = order_idx;
    auto calc_gradient_order = backward_order;
    if (node->getTrainable())
      backward_order++;
    auto calc_derivative_order = backward_order;
    if (node->getTrainable())
      backward_order++;
    auto apply_gradient_order = backward_order++;

    node->setExecutionOrder({forward_order, calc_gradient_order,
                             calc_derivative_order, apply_gradient_order});
  }

  /**
   * This sets max execution order temporarily till model is initialized.
   * This set max execution order is used to extend gradient exec orders for
   * clipping.
   */
  graph_exec_end = std::get<3>((*(cbegin()))->getExecutionOrder());
  return graph_exec_end;
}

void SubGraphBase::addGraphNode(std::shared_ptr<GraphNode> node) {
  node_list.push_back(node);
  node_map[node->getName()] = node_list.size() - 1;
}

const std::shared_ptr<GraphNode> &
SubGraphBase::getNode(unsigned int ith) const {
  return node_list.at(ith);
}

const std::shared_ptr<GraphNode> &
SubGraphBase::getSortedNode(unsigned int ith) const {
  return Sorted.at(ith);
}

const unsigned int
SubGraphBase::getSortedNodeIdx(const std::string &name) const {
  return sorted_node_map.at(name);
}

void SubGraphBase::makeAdjacencyList(
  std::vector<std::list<std::shared_ptr<GraphNode>>> &adj) {
  /** initialize the adj list */
  for (auto &node : node_list) {
    adj.push_back(std::list<std::shared_ptr<GraphNode>>({node}));
  }

  /** make the connections */
  for (auto &node : node_list) {
    for (auto const &in_conn : node->getInputConnections()) {
      unsigned int to_node_id = getNodeIdx(in_conn);
      adj[to_node_id].push_back(node);
    }
  }
}

void SubGraphBase::topologicalSortUtil(
  std::vector<std::list<std::shared_ptr<GraphNode>>> &adj, unsigned int ith,
  std::vector<bool> &visited,
  std::stack<std::shared_ptr<GraphNode>> &dfs_stack) {
  visited[ith] = true;

  std::list<std::shared_ptr<GraphNode>>::iterator i;
  for (i = adj[ith].begin(); i != adj[ith].end(); ++i) {
    auto index = getNodeIdx((*i)->getName());
    if (!visited[index])
      topologicalSortUtil(adj, index, visited, dfs_stack);
  }

  dfs_stack.push(getNode(ith));
}

void SubGraphBase::topologicalSort() {
  std::vector<std::list<std::shared_ptr<GraphNode>>> adj;
  std::stack<std::shared_ptr<GraphNode>> dfs_stack;
  std::vector<bool> visited(node_list.size(), false);

  makeAdjacencyList(adj);
  Sorted.clear();

  // Quite likely this is not needed - verify this
  // TODO : After make node list of graph, we have to find root. (That means it
  // should be the only one input for now.). Need to support multiple input and
  // support search.

  for (unsigned int i = 0; i < adj.size(); ++i) {
    if (visited[i] == false) {
      topologicalSortUtil(adj, i, visited, dfs_stack);
    }
  }

  while (dfs_stack.empty() == false) {
    Sorted.push_back(dfs_stack.top());
    dfs_stack.pop();
  }

  if (Sorted.size() != node_list.size())
    throw std::runtime_error("Internal error in topologicalSort");
  unsigned int idx = 0;
  for (auto &n : Sorted) {
    sorted_node_map[n->getName()] = idx;
    idx++;
  }
}

const std::shared_ptr<GraphNode> &
SubGraphBase::getNode(const std::string &name) const {
  return node_list.at(node_map.at(name));
}

void SubGraphBase::addNode(std::shared_ptr<GraphNode> node, bool ensure_name) {
  /** Ensure that the node has a name and is unique */
  if (ensure_name)
    ensureName(*node);

  /** Insert the node to the graph */
  addGraphNode(node);
}

void SubGraphBase::ensureName(GraphNode &node, const std::string &prefix_,
                              const std::string &postfix_, bool force_rename) {
  auto to_lower = [](const std::string &str) -> std::string {
    std::string ret = str;
    std::transform(ret.begin(), ret.end(), ret.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return ret;
  };

  std::string orig_name = to_lower(node.getName());
  std::string prefix = to_lower(prefix_);
  std::string postfix = to_lower(postfix_);

  bool orig_name_empty = orig_name.empty();
  /** If node already has name which is unique and valid, and force is
   * disabled, then nothing to do.
   */
  if (!orig_name_empty && !force_rename && !verifyNode(orig_name)) {
    node.setName(orig_name);
    node_names.emplace(orig_name);
    return;
  }

  /** If just prefix with node name makes it unique - directly set the name */
  if (!orig_name_empty) {
    std::string direct_name = prefix + orig_name + postfix;
    if (!verifyNode(direct_name)) {
      node.setName(direct_name);
      node_names.emplace(direct_name);
      return;
    }
  }

  std::unordered_set<std::string>::iterator iter;
  std::string name;
  if (orig_name_empty) {
    orig_name = node.getType();
  }

  std::string direct_name = prefix + orig_name + postfix;

  do {
    name = direct_name + std::to_string(def_name_count++);
    iter = node_names.find(name);
  } while (iter != node_names.end());

  node.setName(name);
  node_names.emplace(name);
}

void SubGraphBase::replaceNode(std::shared_ptr<GraphNode> from,
                               std::shared_ptr<GraphNode> to) {
  if (node_map.find(from->getName()) == node_map.end())
    throw std::invalid_argument("Graph node to be replaced is missing");
  if (node_map.find(to->getName()) != node_map.end())
    throw std::invalid_argument("Nodes in the graph must be unique");

  unsigned int from_idx = getNodeIdx(from->getName());
  node_list[from_idx] = to;
  node_map.erase(from->getName());
  node_map[to->getName()] = from_idx;
}

int SubGraphBase::isCompilable() {
  if (compiled) {
    ml_loge("Graph is already compiled");
    return ML_ERROR_NOT_SUPPORTED;
  }

  if (empty()) {
    ml_loge("Graph is empty");
    return ML_ERROR_INVALID_PARAMETER;
  }

  return ML_ERROR_NONE;
}

int SubGraphBase::checkCompiledGraph() {
  /** Dimension of input layers must be known */
  for (auto iter = cbegin(); iter != cend(); iter++) {
    auto lnode = (*iter);
    if (lnode->getNumInputConnections() == 0) {
      if (!lnode->hasInputShapeProperty()) {
        ml_loge("Layer with no inbound connection need input_shape property");
        return ML_ERROR_INVALID_PARAMETER;
      }
    }
  }

  return ML_ERROR_NONE;
}

void SubGraphBase::markNodesForBackwarding() {
  /** accumulate all the nodes which must support backwarding */
  std::unordered_set<std::string> must_support_backwarding;
  if (exec_mode == ExecutionMode::INFERENCE) {
    for (auto iter = cbegin(); iter != cend(); iter++) {
      auto lnode = (*iter);
      lnode->needsCalcGradient(false);
      lnode->needsCalcDerivative(false);
    }
    return;
  }

  /**
   * if a node is trainable, then all the nodes ahead of it must support
   * backwarding operation
   */
  for (auto iter = cbegin(); iter != cend(); iter++) {
    auto lnode = (*iter);
    if (lnode->getTrainable() ||
        must_support_backwarding.find(lnode->getName()) !=
          must_support_backwarding.end()) {
      if (lnode->getTrainable()) {
        lnode->needsCalcGradient(true);
      }
#ifdef ENABLE_TEST
      if (lnode->supportBackwarding() && !optimize_memory) {
        lnode->needsCalcDerivative(true);
      }
#endif

      for (auto i = 0u, num_node = lnode->getNumOutputConnections();
           i < num_node; ++i) {
        auto conn = lnode->getOutputConnection(i);
        if (!conn) {
          continue;
        }

        must_support_backwarding.insert(conn->getName());
      }
    }
  }

  /** mark all the required nodes support backwarding */
  for (auto const &node_name : must_support_backwarding) {
    auto ln = LNODE(getNode(node_name)).get();
    ln->needsCalcDerivative(true);
  }
}

void SubGraphBase::setOutputConnections() {
  for (auto layer_iter = cbegin(); layer_iter != cend(); layer_iter++) {
    const auto &node = *layer_iter;
    for (auto i = 0u, num_inode = node->getNumInputConnections(); i < num_inode;
         ++i) {
      const auto &name = node->getInputConnectionName(i);
      const auto &idx = node->getInputConnectionIndex(i);

      auto node_setting_output = getLayerNode(name);
      node_setting_output->setOutputConnection(idx, node->getName(), i);
    }
  }
}

void SubGraphBase::realizeInputOutputNode() {
  for (auto iter = cbegin(); iter != cend(); ++iter) {
    if (iter->getInputConnections().size() == 0) {
      input_list.push_back(*iter);
    }
    if (iter->getOutputConnections().size() == 0) {
      output_list.push_back(*iter);
    }
  }
}

int SubGraphBase::addLossLayer(const std::string &loss_type_) {
  for (unsigned int i = 0; i < getNumOutputNodes(); ++i) {
    auto output_layer_node = LNODE(getOutputNode(i));
    std::string loss_type = loss_type_;

    if (output_layer_node->requireLabel())
      continue;

    if (loss_type.empty())
      continue;

    auto second_to_last_layer_node = output_layer_node;
    bool is_cross_entropy_loss =
      istrequal(loss_type, CrossEntropyLossLayer::type);
    if (is_cross_entropy_loss) {
      auto type = output_layer_node->getType();

      if (type != ActivationLayer::type) {
        throw exception::not_supported(
          "Error: Cross Entropy need last layer to have softmax or sigmoid"
          "activation.");
      }

      switch (output_layer_node->getActivationType()) {
      case ActivationType::ACT_SIGMOID:
        loss_type = CrossEntropySigmoidLossLayer::type;
        break;
      case ActivationType::ACT_SOFTMAX:
        loss_type = CrossEntropySoftmaxLossLayer::type;
        break;
      default:
        throw exception::not_supported(
          "Error: Cross Entropy not supported without softmax or sigmoid.");
      }

      second_to_last_layer_node =
        LNODE(getNode(output_layer_node->getInputConnectionName(0)));
    }

    std::shared_ptr<LayerNode> lnode = createLayerNode(loss_type);
    ensureName(*lnode);

    if (second_to_last_layer_node->getDistribute()) {
      lnode->setProperty({"distribute=true"});
    }

    /// @todo remove this by add loss at realization
    second_to_last_layer_node->setOutputLayers({lnode->getName()});
    lnode->setProperty(
      {"input_layers=" + second_to_last_layer_node->getName()});

    if (is_cross_entropy_loss) {
      replaceNode(output_layer_node, lnode);
    } else {
      addNode(lnode, false);
    }
    replaceOutputNode(i, lnode);
  }

  return ML_ERROR_NONE;
}

unsigned int SubGraphBase::getNodeIdx(const std::string &name) {
  return node_map.at(name);
}

void SubGraphBase::inPlaceOptimize() {
  if (optimize_memory) {
    for (unsigned int idx = 0; idx < size(); ++idx) {
      auto const &lnode = getSortedLayerNode(idx);
      lnode->setInPlaceType(canExecuteInPlace(lnode));
    }
  }
}

InPlaceType
SubGraphBase::canExecuteInPlace(const std::shared_ptr<LayerNode> &lnode) {
  InPlaceType inplace_type = lnode->initializeInPlace();

  if (inplace_type == InPlaceType::NONE) {
    return inplace_type;
  }

  if (lnode->getType() == InputLayer::type &&
      !istrequal(getTensorType()[2], "FP32")) {
    return InPlaceType::NONE;
  }

  if (lnode->getType() == MultiOutLayer::type) {
    return InPlaceType::RESTRICTING;
  }

  /** A case where it can operate in-place even if there is a multi-out type
   * input connection. */
  if (inplace_type == InPlaceType::RESTRICTING) {
    for (size_t i = 0, num_node = lnode->getNumInputConnections(); i < num_node;
         ++i) {
      const std::string &input_name = lnode->getInputConnectionName(i);
      if (getLayerNode(input_name)->getInPlaceType() ==
          InPlaceType::RESTRICTING)
        return inplace_type;
    }
    return InPlaceType::NON_RESTRICTING;
  }
  /** A case where it cannot operate in-place if there is a multi-out type
   * input connection. */
  else { /** condition: NON_RESTRICTING */
    for (size_t i = 0, num_node = lnode->getNumInputConnections(); i < num_node;
         ++i) {
      const std::string &input_name = lnode->getInputConnectionName(i);
      if (getLayerNode(input_name)->getInPlaceType() ==
          InPlaceType::RESTRICTING)
        return InPlaceType::NONE;
    }
    return inplace_type;
  }
}

void SubGraphBase::resetLossScale(float scale) {
  loss_scale = scale;
  for (auto iter = cbegin(); iter != cend(); iter++) {
    auto &ln = *iter;
    ln->getRunContext().setLossScale(scale);
  }
}

LayerNode *SubGraphBase::computeBackwardEnd() {
  int max_exec_order = -1;
  LayerNode *node = nullptr;

  if (!optimize_memory) {
    return (*cbegin()).get();
  }

  for (auto iter = getBackwardingBeginIter(); iter != getBackwardingEndIter();
       iter++) {
    auto &ln = *iter;
    const auto &exec_order = ln->getExecutionOrder();
    int cur_order = std::get<0>(exec_order);
    if (ln->needsCalcDerivative() || ln->needsCalcGradient()) {
#ifdef ENABLE_TEST
      cur_order = std::get<2>(exec_order);
#else
      cur_order = std::get<1>(exec_order);
#endif
    }

    NNTR_THROW_IF(max_exec_order == cur_order, std::invalid_argument)
      << "layer node: " << ln->getName()
      << " has duplicated max_exec_order, this should not happen, current "
         "execution order: "
      << max_exec_order;

    if (max_exec_order < cur_order) {
      max_exec_order = cur_order;
      node = ln.get();
    }
  }

  return node;
}

} /* namespace nntrainer */
