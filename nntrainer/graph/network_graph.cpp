// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    network_graph.h
 * @date    19 Oct 2020
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   This is Network Graph Class for Neural Network
 *
 * @todo    Support multi-input graph.
 */

#include "graph_node.h"
#include "tensor.h"
#include <cmath>
#include <stdexcept>
#include <string>

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

#define LNODE(x) std::static_pointer_cast<LayerNode>(x)

namespace nntrainer {

int NetworkGraph::compile(const std::string &loss_type) {

  /// @todo needs to be updated
  //  1. Current compile code assumes a graph consists of one subgraph.
  //     This code call graph.compile() and update the network info based on its
  //     compilation result.
  //  2. graph.compile conducts addLossLayer. It should be updated to apply it
  //     only for the last subgraph
  int status = ML_ERROR_NONE;

  status = graph.compile(loss_type);
  NN_RETURN_STATUS();

  forward_iter_end = graph.getForwardIterEnd();
  compiled = true;

  return status;
}

void NetworkGraph::setExecutionOrder() {
  /**
   * This sets max execution order temporarily till model is initialized.
   * This set max execution order is used to extend gradient exec orders for
   * clipping.
   */
  graph_exec_end = graph.setExecutionOrder();
}

void NetworkGraph::addLayerNode(std::unique_ptr<Layer> layer) {
  graph.addNode(std::make_unique<LayerNode>(std::move(layer)));
}

int NetworkGraph::isCompilable() {
  // @todo need to be updated
  return graph.isCompilable();
}

int NetworkGraph::checkCompiledGraph() {
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

void NetworkGraph::markNodesForBackwarding() {

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
    auto ln = LNODE(graph.getNode(node_name)).get();
    ln->needsCalcDerivative(true);
  }
}

void NetworkGraph::setBatchSize(unsigned int batch_size) {
  if (batch_size == this->batch_size)
    return;

  this->batch_size = batch_size;
  if (!input_list.empty() && getInputDimension()[0].batch() == batch_size)
    return;

  auto allocated = tensor_manager->isAllocated();

  if (allocated)
    deallocateTensors();

  /// @todo it needs to be updated later
  //        The current cbegin() returns graph.cbegin();
  //        It should be updated when the graph is generalied to the vector of
  //        subgraphs
  for (auto iter = cbegin(); iter != cend(); iter++) {
    if ((*iter)->isFinalized()) {
      /// resize tensors spec
      /// @todo remove below, if custom tensor needs to change dimension
      /// according to the tensor, it must be done explicitly, or at least have
      /// a property to control the behavior
      const RunLayerContext &context = (*iter)->getRunContext();
      for (unsigned int idx = 0; idx < context.getNumTensors(); idx++) {
        auto const &ts = context.getTensor(idx);
        tensor_manager->setBatchSize(ts.getName(), ts.getDim().batch());
        if (context.tensorHasGradient(idx)) {
          auto const &ts_grad = context.getTensorGrad(idx);
          tensor_manager->setBatchSize(ts_grad.getName(),
                                       ts_grad.getDim().batch());
        }
      }
      /// override setting batch as per request
      (*iter)->setBatch(batch_size);
    }
  }
  /// resize input and output spec
  tensor_manager->setBatchSize(batch_size);

  if (allocated)
    allocateTensors(exec_mode);

  /** update input and label dimensions */
  for (unsigned int idx = 0; idx < input_list.size(); idx++)
    input_dims[idx] = tensor_manager->getTensor(input_list[idx])->getDim();
  for (unsigned int idx = 0; idx < label_list.size(); idx++)
    label_dims[idx] = tensor_manager->getTensor(label_list[idx])->getDim();
}

sharedConstTensors
NetworkGraph::forwarding(bool training,
                         std::function<bool(void *userdata)> stop_cb,
                         void *userdata, bool swap_mode) {
  /// @todo needs to be updated
  // Current forwarding code assumes a graph consists of one subgraph.
  // This code call graph.forwarding().
  return graph.forwarding(training, stop_cb, userdata, swap_mode);
}

sharedConstTensors NetworkGraph::incremental_forwarding(
  unsigned int from, unsigned int to, bool training,
  std::function<bool(void *userdata)> stop_cb, void *userdata) {
  /// @todo needs to be updated
  // Current forwarding code assumes a graph consists of one subgraph.
  // This code call graph.forwarding().
  return graph.incremental_forwarding(from, to, training, stop_cb, userdata);
}

bool NetworkGraph::backwarding(
  int iteration, std::function<void(Weight &, int)> &lazy_apply_grad_op,
  std::function<bool(void *userdata)> stop_cb, void *userdata,
  bool is_grad_opt_mode) {
  /// @todo needs to be updated
  // Current backwarding code assumes a graph consists of one subgraph.
  // This code call graph.backwarding().
  return graph.backwarding(iteration, lazy_apply_grad_op, stop_cb, userdata,
                           is_grad_opt_mode);
}

LayerNode *NetworkGraph::computeBackwardEnd() {
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

/**
 * @brief Allocate memory for all the managed tensors
 */
void NetworkGraph::allocateTensors(ExecutionMode exec_mode_) {
  exec_mode = exec_mode_;
  if (exec_mode == ExecutionMode::INFERENCE)
    /**
     * get the order of execution/usage order for the forwarding of the last
     * layer and pass that as the max_exec_order ensuring that all tensors
     * with usage less than the max_exec_order are allocated.
     */
    tensor_manager->allocateTensors(
      std::get<0>((*(cend() - 1))->getExecutionOrder()));
  else {
    /**
     * get the order of execution/usage order for the backwarding of the first
     * layer (as that will be the last layer to executed in the backwarding)
     * and pass that as the max_exec_order ensuring that all tensors with
     * usage less than the max_exec_order are allocated.
     * @todo if model is gradient clipping, we have to add last execution order
     * + 1
     */
    tensor_manager->allocateTensors(
      std::get<3>(backward_iter_end->getExecutionOrder()));
  }
}

std::vector<TensorDim> NetworkGraph::getInputDimension() const {
  /// @todo needs to be updated
  // Current getInputDimension code assumes a graph consists of one subgraph.
  // This code call graph.getInputDimension().
  return graph.getInputDimension();
}

unsigned int NetworkGraph::getBatchSize() const { return batch_size; }

std::vector<TensorDim> NetworkGraph::getOutputDimension() const {
  /// @todo needs to be updated
  // Current getOutputDimension code assumes a graph consists of one subgraph.
  // This code call graph.getOutputDimension().
  return graph.getOutputDimension();
}

std::vector<std::shared_ptr<LayerNode>>
NetworkGraph::getUnsortedLayers(const std::string &input_layer,
                                const std::string &output_layer) const {
  // @todo needs to be updated
  // Current getUnsortedLayers code assumes a graph consists of one subgraph.
  // This code call graph.getUnsortedLayers().
  return graph.getUnsortedLayers(input_layer, output_layer);
}

std::vector<std::shared_ptr<LayerNode>> NetworkGraph::getLayerNodes() const {
  /// @todo needs to be updated
  // Current getLayerNodes code assumes a graph consists of one subgraph.
  return std::vector<std::shared_ptr<LayerNode>>(graph.cbegin(), graph.cend());
}

void NetworkGraph::addLayer(std::shared_ptr<LayerNode> layer) {
  if (compiled)
    throw std::runtime_error("Cannot modify graph after compile");

  /** Insert the layer to the graph */
  /// @todo needs to be updated
  // Current addLayer code assumes a graph consists of one subgraph.
  graph.addNode(layer);
}

InPlaceType
NetworkGraph::canExecuteInPlace(const std::shared_ptr<LayerNode> &lnode) {
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
      if (graph.getLayerNode(input_name)->getInPlaceType() ==
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
      if (graph.getLayerNode(input_name)->getInPlaceType() ==
          InPlaceType::RESTRICTING)
        return InPlaceType::NONE;
    }
    return inplace_type;
  }
}

void NetworkGraph::inPlaceOptimize() {
  if (optimize_memory) {
    for (unsigned int idx = 0; idx < graph.size(); ++idx) {
      auto const &lnode = graph.getSortedLayerNode(idx);
      lnode->setInPlaceType(canExecuteInPlace(lnode));
    }
  }
}

#ifdef ENABLE_TEST

std::map<std::string, std::vector<unsigned int>>
NetworkGraph::getLayerExecutionOrders(const std::shared_ptr<LayerNode> &lnode) {
  const auto &init_context = lnode->getInitContext();
  auto out_specs = init_context.getOutSpecs();
  auto weight_specs = init_context.getWeightsSpec();
  auto tensor_specs = init_context.getTensorsSpec();

  std::map<std::string, std::vector<unsigned int>> exec_orders;

  for (auto &spec : out_specs) {
    const auto &name = lnode->getName() + ":" + spec.variable_spec.name;
    auto orders = tensor_manager->getTensorExecutionOrders(name, false);
    exec_orders.insert({name, orders});
    try {
      auto orders_grad =
        tensor_manager->getTensorExecutionOrders(name + ":grad", false);
      exec_orders.insert({name + ":grad", orders_grad});
    } catch (const std::exception &e) {
      ml_logi("Cannot find grad tensor for %s:grad", name.c_str());
      continue;
    }
  }

  for (auto &spec : weight_specs) {
    const auto &name = std::get<const std::string>(spec);
    auto orders = tensor_manager->getTensorExecutionOrders(name, true);
    exec_orders.insert({name, orders});
    try {
      auto orders_grad =
        tensor_manager->getTensorExecutionOrders(name + ":grad", false);
      exec_orders.insert({name + ":grad", orders_grad});
    } catch (const std::exception &e) {
      ml_logi("Cannot find grad tensor for %s:grad", name.c_str());
      continue;
    }
  }

  for (auto &spec : tensor_specs) {
    const auto &name = std::get<const std::string>(spec);
    auto orders = tensor_manager->getTensorExecutionOrders(name, false);
    exec_orders.insert({name, orders});
    try {
      auto orders_grad =
        tensor_manager->getTensorExecutionOrders(name + ":grad", false);
      exec_orders.insert({name + ":grad", orders_grad});
    } catch (const std::exception &e) {
      ml_logi("Cannot find grad tensor for %s:grad", name.c_str());
      continue;
    }
  }

  return exec_orders;
}

#endif // ENABLE_TEST

int NetworkGraph::initialize(ExecutionMode mode,
                             const std::vector<Connection> &model_input_names,
                             const std::vector<Connection> &model_label_names) {

  exec_mode = mode;
  tensor_manager->setExecutionMode(mode);
  /// @todo needs to be updated
  // Current initialize code assumes a graph consists of one subgraph.
  // This code call graph.initialize().
  // In this version, backward_iter_end is passed to graph.initialize();
  // Later, it should be updated.
  auto status = graph.initialize(tensor_manager, mode, model_input_names,
                                 model_label_names);

  // update the graph info.
  backward_iter_end = graph.getBackwardIterEnd();
  const auto &label_list_ = graph.getLabelList();
  const auto &input_list_ = graph.getInputList();
  const auto &output_list_ = graph.getOutputList();
  const auto &label_dims_ = graph.getLabelDims();
  const auto &input_dims_ = graph.getInputDims();

  std::copy(label_list_.begin(), label_list_.end(), label_list.begin());
  std::copy(input_list_.begin(), input_list_.end(), input_list.begin());
  std::copy(output_list_.begin(), output_list_.end(), output_list.begin());
  std::copy(label_dims_.begin(), label_dims_.end(), label_dims.begin());
  std::copy(input_dims_.begin(), input_dims_.end(), input_dims.begin());

  return status;
}

int NetworkGraph::reinitialize(
  const std::vector<Connection> &model_input_names,
  const std::vector<Connection> &model_label_names) {
  /// @todo needs to be updated
  // Current reinitialize code assumes a graph consists of one subgraph.
  // This code call graph.reinitialize().
  return graph.reinitialize(model_input_names, model_label_names);
}

void NetworkGraph::setExternalTensors(const std::vector<Tensor> &data,
                                      const std::vector<std::string> names) {

  /// feed or clear label
  for (unsigned int idx = 0; idx < names.size(); idx++) {
    if (data.empty())
      tensor_manager->fillPlaceholder(names[idx], Tensor());
    else if (data.size() == 1)
      tensor_manager->fillPlaceholder(names[idx], data[0]);
    else
      tensor_manager->fillPlaceholder(names[idx], data[idx]);
  }
}

void NetworkGraph::setInputsLabels(const std::vector<Tensor> &inputs,
                                   const std::vector<Tensor> &labels) {

  NNTR_THROW_IF(labels.size() > 1 && labels.size() != label_list.size(),
                std::invalid_argument)
    << "label size does not match with the network requirements"
    << " label size: " << labels.size()
    << " requirements size: " << label_list.size();

  NNTR_THROW_IF(inputs.size() > 1 && inputs.size() != input_list.size(),
                std::invalid_argument)
    << "input size does not match with the network requirements"
    << " input size: " << inputs.size()
    << " requirements size: " << input_list.size();

  setExternalTensors(inputs, input_list);
  setExternalTensors(labels, label_list);
}

void NetworkGraph::setInputsLabels(sharedConstTensors &inputs,
                                   sharedConstTensors &labels) {

  std::vector<Tensor> ins;
  std::transform(
    inputs.begin(), inputs.end(), std::back_inserter(ins),
    [](auto const &val) -> const auto & { return *val.get(); });

  std::vector<Tensor> labs;
  std::transform(
    labels.begin(), labels.end(), std::back_inserter(labs),
    [](auto const &val) -> const auto & { return *val.get(); });

  setInputsLabels(ins, labs);
}

std::vector<Tensor> NetworkGraph::getOutputTensors() const {
  std::vector<Tensor> output_tensors;
  output_tensors.reserve(output_list.size());

  for (auto const &name : output_list)
    output_tensors.push_back(*tensor_manager->getTensor(name));

  return output_tensors;
}

void NetworkGraph::flushCache() { tensor_manager->flushCache(); }

void NetworkGraph::flushCacheExcept(unsigned int order) {
  tensor_manager->flushCacheExcept(order);
}

void NetworkGraph::LoadTensors(unsigned int order) {
  tensor_manager->LoadTensors(order);
}

bool NetworkGraph::checkLoadComplete(unsigned int order) {
  return tensor_manager->checkLoadComplete(order);
}

bool NetworkGraph::checkUnloadComplete(unsigned int order) {
  return tensor_manager->checkUnloadComplete(order);
}

void NetworkGraph::UnloadTensors(unsigned int order) {
  tensor_manager->UnloadTensors(order);
}

void NetworkGraph::requestOptimizerVariable(
  std::function<std::vector<TensorDim>(const TensorDim &)> cb,
  bool request_only_trainable) {
  for (auto const &w : tensor_manager->getWeights()) {
    if (w->isGradientLastAccess() && w->hasGradient()) {
      const TensorDim &dim = w->getDim();
      std::vector<TensorDim> dims = cb(dim);
      w->setOptimizerVariables(tensor_manager->requestWeightOptimizerVariables(
        dims, w->getName(), ":opt", TensorLifespan::MAX_LIFESPAN,
        w->isGradientClipByGlobalNorm(), w->isMixedPrecision(),
        Initializer::ZEROS));
    }
  }
}

void NetworkGraph::resetLossScale(float scale) {
  loss_scale = scale;

  /// @todo resetLossScale for all subgraphs
  for (auto iter = cbegin(); iter != cend(); iter++) {
    auto &ln = *iter;
    ln->getRunContext().setLossScale(scale);
  }
}

void NetworkGraph::setOptimizer(std::shared_ptr<OptimizerWrapped> opt_) {
  opt = opt_;
  /// @todo set optimizer for all subgraphs
  graph.setOptimizer(opt);
  graph.setIsMixedPrecision(isMixedPrecision());
};

} /* namespace nntrainer */
