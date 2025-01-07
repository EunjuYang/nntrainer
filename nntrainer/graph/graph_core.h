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

#ifndef __GRAPH_CORE_H__
#define __GRAPH_CORE_H__
#ifdef __cplusplus

#include <list>
#include <map>
#include <memory>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <activation_layer.h>
#include <cross_entropy_loss_layer.h>
#include <cross_entropy_sigmoid_loss_layer.h>
#include <cross_entropy_softmax_loss_layer.h>
#include <graph_node.h>
#include <layer_node.h>
#include <manager.h>
#include <optimizer_wrapped.h>

namespace nntrainer {

#define LNODE(x) std::static_pointer_cast<LayerNode>(x)

/**
 * @class   Graph Core Class
 * @brief   Graph Core Class which provides core graph functionalities
 */
class SubGraphBase {

public:
  /**
   * @brief     Constructor of SubGraphBase Class
   */
  SubGraphBase() :
    sorted(false),
    optimize_memory(true),
    exec_mode(ExecutionMode::TRAIN),
    node_names(),
    def_name_count(0),
    compiled(false),
    graph_exec_end(0),
    backward_iter_end(nullptr),
    forward_iter_end(nullptr),
    tensor_format("NCHW"),
    tensor_dtype(split("FP32-FP32", getRegex("\\-"))) {
    nan_count = 0;
  }

  /**
   * @brief     Destructor of SubGraphBase Class
   */
  virtual ~SubGraphBase() {}

  /**
   * @brief Compile the subgraph (virtual)
   * @param[in] loss_type loss for the graph
   * @retval ML_ERROR_NONE on success, error on failure
   */
  virtual int compile(const std::string &loss_type) = 0;

  /**
   * @brief get current flat graph from the model before sorting
   * @note graph contains pointer to the actual nodes, which is not deeply
   * copied.
   * @retval current flat graph
   *
   * @todo remove getting unsorted layers from model loader, compile model
   * loader
   */
  std::vector<std::shared_ptr<LayerNode>>
  getUnsortedLayers(const std::string &input_layer,
                    const std::string &output_layer) const;

  /**
   * @brief initialize the subgraph (virtual)
   * @param model_input_names model input connection if empty list given, all of
   * node that can be inputs will be identified in the sort order
   * @param model_label_names model label names if empty list given, all of node
   * that can be labels will be identified in the sort order
   * @param tensor_manager tensor manager
   * @return int ML_ERROR_NONE if successful
   */
  virtual int
  initialize(std::shared_ptr<Manager> tensor_manager_,
             ExecutionMode mode = ExecutionMode::TRAIN,
             const std::vector<Connection> &model_input_names = {},
             const std::vector<Connection> &model_label_names = {}) = 0;

  /**
   * @brief reinitialize the subgraph (virtual)
   * @param model_input_names model input connection if empty list given, all of
   * node that can be inputs will be identified in the sort order
   * @param model_label_names model label names if empty list given, all of node
   * that can be labels will be identified in the sort order
   * @return int ML_ERROR_NONE if successful
   */
  virtual int
  reinitialize(const std::vector<Connection> &model_input_names = {},
               const std::vector<Connection> &model_label_names = {}) = 0;

  /**
   * @brief Create run layer context from the given init layer context
   *
   * @param lnode layer node to finalize and set run context
   * @param prev_inputs previous input information
   * @todo needs to be updated. finalize context should be called at the level
   * of subgraph (same subgraph might need to share the layer context)
   */
  virtual std::vector<Var_Grad *>
  finalizeContext(const std::shared_ptr<LayerNode> &lnode,
                  const std::vector<Var_Grad *> &prev_inputs) = 0;

  /**
   * @brief Recreate run layer context from the given init layer context
   *
   * @param lnode layer node to finalize and set run context
   * @param prev_inputs previous input information
   * @todo needs to be updated. finalize context should be called at the level
   * of subgraph (same subgraph might need to share the layer context)
   */
  virtual std::vector<Var_Grad *>
  refinalizeContext(const std::shared_ptr<LayerNode> &lnode,
                    const std::vector<Var_Grad *> &prev_inputs) = 0;

  /**
   * @brief return model tensor type
   *
   * @return TensorDim::Format NCHW or NHWC
   */
  std::array<std::string, 3> getTensorType() {
    return {tensor_format, tensor_dtype[0], tensor_dtype[1]};
  };

  /**
   * @brief Set the order of execution for all the nodes in the graph
   * @details This sets the order of execution using the order from the
   * topological sort. The order of forwarding matches the topological sort. The
   * order for backwarding is in the exact reverse order. The calcDerivative()
   * is expected to be called right after calcGradient().
   */
  unsigned int setExecutionOrder();

  /**
   * @brief     reset the loss scale
   * @param[in] scale
   */
  void resetLossScale(float scale);

  /**
   * @brief setter of optimizer
   */
  void setOptimizer(std::shared_ptr<OptimizerWrapped> opt_) { opt = opt_; }

  /**
   * @brief     forwarding the subgraph
   * @param[in] training true if forwarding is on training
   * @retval output tensors
   */
  virtual sharedConstTensors forwarding(
    bool training = false,
    std::function<bool(void *userdata)> stop_cb =
      [](void *user_data) { return false; },
    void *user_data = nullptr, bool swap_mode = false) = 0;

  /**
   * @brief     forwarding the subgraph
   * @param[in] from start step
   * @param[in] to end step
   * @param[in] training true if forwarding is on training
   * @retval output tensors
   */
  virtual sharedConstTensors incremental_forwarding(
    unsigned int from, unsigned int to, bool training = false,
    std::function<bool(void *userdata)> stop_cb =
      [](void *user_data) { return false; },
    void *user_data = nullptr) = 0;

  /**
   * @brief     backwarding the subgraph
   * @param[in] iteration current iteration number
   * @param[in] lazy_apply_grad_op operation for applying the lazy gradients
   * @retval ret it is false then the gradient has NaN valude in mixed precision
   * training. If it is, then we need to control the loss scale factor and
   * compute again the derivatives.
   */
  virtual bool backwarding(
    int iteration, std::function<void(Weight &, int)> &lazy_apply_grad_op,
    std::function<bool(void *userdata)> stop_cb =
      [](void *user_data) { return false; },
    void *user_data = nullptr, bool is_grad_opt_mode = false) = 0;

  /**
   * @brief applyGradients on the subgraph
   */
  virtual void applyGradients(LayerNode *node, int iteration) = 0;

  /**
   * @brief Add the given node into Graph
   * @param[in] node shared_ptr of node
   */
  void addNode(std::shared_ptr<GraphNode> node, bool ensure_name = true);

  /**
   * @brief getter of number of nodes
   * @param[out] number of nodes
   */
  unsigned int size() const { return node_list.size(); }

  /**
   * @brief get if the graph is empty
   * @param[out] true if empty, else false
   */
  bool empty() const { return node_list.empty(); }

  /**
   * @brief     Swap function for the class
   */
  friend void swap(SubGraphBase &lhs, SubGraphBase &rhs) {
    using std::swap;

    swap(lhs.node_list, rhs.node_list);
    swap(lhs.node_map, rhs.node_map);
    swap(lhs.Sorted, rhs.Sorted);
    swap(lhs.node_names, rhs.node_names);
    swap(lhs.def_name_count, rhs.def_name_count);
  }

  /**
   * @brief getter of GraphNode with index number
   * @param[in] index
   * @ret GraphNode
   */
  const std::shared_ptr<GraphNode> &getNode(unsigned int ith) const;

  /**
   * @brief getter of Sorted GraphNode with index number
   * @param[in] index
   * @ret GraphNode
   */
  const std::shared_ptr<GraphNode> &getSortedNode(unsigned int ith) const;

  /**
   * @brief getter of Sorted GraphNode index with name
   * @param[in] layer name
   * @ret index
   */
  const unsigned int getSortedNodeIdx(const std::string &name) const;

  /**
   * @brief getter of GraphNode with node name
   * @param[in] node name
   * @retval GraphNode
   */
  const std::shared_ptr<GraphNode> &getNode(const std::string &name) const;

  /**
   * @brief     get begin iterator for the forwarding
   * @retval    const iterator marking the begin of forwarding
   * @note      this function should not be used when node_list is empty.
   * if node_list is empty, end iterator is dereferenced,
   * This action is undefined behavior.
   */
  template <
    typename T = GraphNode,
    std::enable_if_t<std::is_base_of<GraphNode, T>::value, T> * = nullptr>
  inline graph_const_iterator<T> cbegin() const {
    if (Sorted.empty())
      return graph_const_iterator<T>(&(*node_list.cbegin()));
    else
      return graph_const_iterator<T>(&(*Sorted.cbegin()));
  }

  /**
   * @brief     get begin iterator for the forwarding
   * @retval    const iterator marking the begin of forwarding
   * @note      this function should not be used when node_list is empty.
   * if node_list is empty, end iterator is dereferenced,
   * This action is undefined behavior.
   */
  graph_const_iterator<LayerNode> cbegin() const { return cbegin<LayerNode>(); }

  /**
   * @brief     get end iterator for the forwarding
   * @retval    const iterator marking the end of forwarding
   * @note      this function should not be used when node_list is empty.
   * if node_list is empty, end iterator is dereferenced,
   * This action is undefined behavior.
   */
  template <
    typename T = GraphNode,
    std::enable_if_t<std::is_base_of<GraphNode, T>::value, T> * = nullptr>
  inline graph_const_iterator<T> cend() const {
    if (Sorted.empty())
      return graph_const_iterator<T>(&(*node_list.cbegin())) + node_list.size();
    else
      return graph_const_iterator<T>(&(*Sorted.cbegin())) + Sorted.size();
  }

  /**
   * @brief     get end iterator for the forwarding
   * @retval    const iterator marking the end of forwarding
   * @note      this function should not be used when node_list is empty.
   * if node_list is empty, end iterator is dereferenced,
   * This action is undefined behavior.
   */
  graph_const_iterator<LayerNode> cend() const { return cend<LayerNode>(); }

  /**
   * @brief     get begin iterator for the backwarding
   * @retval    const reverse iterator marking the begin of backwarding
   */
  template <
    typename T = GraphNode,
    std::enable_if_t<std::is_base_of<GraphNode, T>::value, T> * = nullptr>
  inline graph_const_reverse_iterator<T> crbegin() const {
    return graph_const_reverse_iterator<T>(cend<T>());
  }

  /**
   * @brief     get begin iterator for the backwarding
   * @retval    const reverse iterator marking the begin of backwarding
   */
  graph_const_reverse_iterator<LayerNode> crbegin() const {
    return crbegin<LayerNode>();
  }

  /**
   * @brief     get end iterator for the backwarding
   * @retval    const reverse iterator marking the end of backwarding
   */
  template <
    typename T = GraphNode,
    std::enable_if_t<std::is_base_of<GraphNode, T>::value, T> * = nullptr>
  inline graph_const_reverse_iterator<T> crend() const {
    return graph_const_reverse_iterator<T>(cbegin<T>());
  }

  /**
   * @brief     get end iterator for the backwarding
   * @retval    const reverse iterator marking the end of backwarding
   */
  graph_const_reverse_iterator<LayerNode> crend() const {
    return crend<LayerNode>();
  }

  /**
   * @brief     get begin iterator for the backwarding
   * @retval    const reverse iterator marking the begin of backwarding
   * @todo  needs to update
   */
  graph_const_reverse_iterator<LayerNode> getBackwardingBeginIter() const {
    return crbegin();
  }

  /**
   * @brief     get end iterator for the backwarding
   * @retval    const reverse iterator marking the end of backwarding
   * @todo  need to update
   */
  graph_const_reverse_iterator<LayerNode> getBackwardingEndIter() {
    return crend();
  }

  /**
   * @brief     getter of output dimension of graph
   * @retval    output tensor dim list
   * @todo  needs to update
   */
  std::vector<TensorDim> getOutputDimension() const;

  /**
   * @brief     getter of input dimension of graph
   * @retval    input tensor dim list
   * @todo  needs to update
   */
  std::vector<TensorDim> getInputDimension() const;

  /**
   * @brief Sorting and Define order to calculate : Depth First Search
   */
  void topologicalSort();

  /**
   * @brief     Copy the graph
   * @param[in] from Graph Object to copy
   * @retval    Graph Object copyed
   */
  SubGraphBase &copy(SubGraphBase &from) {
    node_list.resize(from.node_list.size());
    if (this != &from) {
      //      for (unsigned int i = 0; i < node_list.size(); ++i)
      //        node_list[i]->copy(from.node_list[i]);
    }
    return *this;
  }

  /**
   * @brief     Ensure that node has a name.
   * @param[in] node GraphNode whose name is to be ensured to be valid
   * @param[in] prefix Prefix to be attached to the node name
   * @param[in] postfix Postfix to be attached to the node name
   * @param[in] force_rename If the node must be forcefully rename
   * @details   Ensures that the node has a unique and a valid name. A valid
   * name pre-assigned to the node can be changed if force_rename is enabled.
   */
  void ensureName(GraphNode &node, const std::string &prefix = "",
                  const std::string &postfix = "", bool force_rename = false);

  /**
   * @brief   Replace graph node in node_list
   * @param   from Graph node to be replaced
   * @param   to Graph node to replace
   */
  void replaceNode(std::shared_ptr<GraphNode> from,
                   std::shared_ptr<GraphNode> to);

  /**
   * @brief   getter of graph input nodes with index number
   * @param   idx
   * @return  graph node of input node
   */
  const std::shared_ptr<GraphNode> &getInputNode(unsigned int idx) {
    return input_list[idx];
  }

  /**
   * @brief   getter of number of input nodes
   * @return  number of input nodes
   */
  unsigned int getNumInputNodes() { return input_list.size(); }

  /**
   * @brief   getter of graph output nodes with index number
   * @param   idx
   * @return  graph node of output node
   */
  const std::shared_ptr<GraphNode> &getOutputNode(unsigned int idx) {
    return output_list[idx];
  }

  /**
   * @brief   getter of number of output nodes
   * @return  number of output nodes
   */
  unsigned int getNumOutputNodes() { return output_list.size(); }

  /**
   * @brief       replace output node
   * @param idx   output node index to be replaced
   * @param node  graph node shared pointer to replace
   */
  void replaceOutputNode(unsigned int idx, std::shared_ptr<GraphNode> node) {
    output_list[idx] = node;
  }

  /**
   * @brief     Verify if the node exists
   */
  inline bool verifyNode(const std::string &name) {
    if (node_names.find(name) == node_names.end())
      return false;
    return true;
  }

  /**
   * @brief getter of Sorted LayerNode with index number
   * @param[in] index
   * @ret LayerNode
   */
  std::shared_ptr<LayerNode> getSortedLayerNode(unsigned int ith) const {
    return std::static_pointer_cast<LayerNode>(getSortedNode(ith));
  }

  /**
   * @brief getter of LayerNode with layer name
   * @param[in] layer name
   * @retval LayerNode
   */
  std::shared_ptr<LayerNode> getLayerNode(const std::string &layer_name) const {
    return std::static_pointer_cast<LayerNode>(getNode(layer_name));
  }

  /**
   * @brief     check if the graph is ready to compile.
   * @retval #ML_ERROR_NONE graph is ready to compile
   * @retval #ML_ERROR_INVALID_PARAMETER not ready to compile.
   */
  int isCompilable();

  /**
   * @brief     adding loss layer at last position
   * @param[in] loss_type loss type
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int addLossLayer(const std::string &loss_type);

  /**
   * @brief setIsMixedPrecision
   */
  void setIsMixedPrecision(bool is_mixed_precision_) {
    is_mixed_precision = is_mixed_precision_;
  }

  /**
   * @brief     check if the compiled graph is of correct form.
   * @retval #ML_ERROR_NONE graph is compiled correctly
   * @retval #ML_ERROR_INVALID_PARAMETER did not compile correctly
   */
  int checkCompiledGraph();

  /**
   * @brief     mark nodes required for backwarding.
   */
  void markNodesForBackwarding();

  /**
   * @brief set lookahead
   */
  void setLookAhead(unsigned int lookahead_) { lookahead = lookahead_; }

  /**
   * @brief
   */
  LayerNode *getForwardIterEnd() const { return forward_iter_end; };

  /**
   * @brief
   */
  LayerNode *getBackwardIterEnd() const { return backward_iter_end; };

  /**
   * @brief
   */
  std::vector<std::string> &getInputList() { return input_name_list; }

  /**
   * @brief
   */
  std::vector<std::string> &getLabelList() { return label_name_list; }

  /**
   * @brief
   */
  std::vector<std::string> &getOutputList() { return output_name_list; }

  /**
   * @brief
   */
  std::vector<TensorDim> &getLabelDims() { return label_dims; }

  /**
   * @brief
   */
  std::vector<TensorDim> &getInputDims() { return input_dims; }

protected:
  std::vector<std::shared_ptr<GraphNode>> input_list;
  std::vector<std::shared_ptr<GraphNode>> output_list;
  std::vector<std::shared_ptr<GraphNode>> node_list; /**< Unordered Node List */
  std::vector<std::string>
    label_name_list; /**< identifier for the model labels */
  std::vector<std::string>
    input_name_list; /**< identifier for the model inputs */
  std::vector<std::string>
    output_name_list;                /**< identifier for the model outputs */
  std::vector<TensorDim> label_dims; /**< graph label dimensions */
  std::vector<TensorDim> input_dims; /**< graph input dimensions */
  std::unordered_map<std::string, int> node_map; /**< Unordered Node map  */
  std::unordered_map<std::string, int>
    sorted_node_map;                              /**< Unordered Node map  */
  std::vector<std::shared_ptr<GraphNode>> Sorted; /**< Ordered Node List  */
  bool sorted;             /** if the node_list is sorted */
  bool optimize_memory;    /**< optimize memory */
  ExecutionMode exec_mode; /**< execution mode with which the graph has been */

  std::unordered_set<std::string>
    node_names;       /**< Set containing all the names of nodes in the model */
  int def_name_count; /**< Count assigned to node names declared by default */
  bool compiled;      /**< if the sub graph is compiled */
  unsigned int graph_exec_end; /**< Inclusive, last execution order of the */
  std::shared_ptr<Manager> tensor_manager; /**< tensors manager */
  unsigned int lookahead;

  /** Props used in subgraph op. */
  LayerNode
    *backward_iter_end; /**< inclusive end node of the valid backward
                           execution when initialized, nodes after this node
                           does not required backwarding thus making it noop */
  LayerNode
    *forward_iter_end; /**< inclusive end node of the forward execution */
  std::unordered_map<std::string, int>
    profile_keys; /**< profile keys based on the layer type */
  std::vector<Weight *>
    lazy_weights; /**< weights with delayed grad update, e.g., gradient
                     clipping, loss scaling */

  std::string tensor_format; /**< Model Tensor Format: NCHW or NHWC */
  std::vector<std::string> tensor_dtype; /**< Model Tensor Type: FP32, FP16 */

  std::shared_ptr<OptimizerWrapped> opt;

  bool is_mixed_precision;
  bool is_clip_grad;
  float loss_scale;
  unsigned int nan_count;

  /**
   * @brief     set output connections for all the layers
   */
  void setOutputConnections();

  /**
   * @brief find which node is a input or output node in the subgraph
   */
  void realizeInputOutputNode();

  /**
   * @brief     Optimize the graph memory utilization for in-place operations
   */
  void inPlaceOptimize();

  /**
   * @brief     topological sort
   * @param[in] ith index of GraphNode
   * @param[in] visited temp list
   * @param[in] stack for Node list to visit.
   */
  void
  topologicalSortUtil(std::vector<std::list<std::shared_ptr<GraphNode>>> &adj,
                      unsigned int ith, std::vector<bool> &visited,
                      std::stack<std::shared_ptr<GraphNode>> &Stack);

  /**
   * @brief Add given GraphNode to the Graph
   * @param[in] node shared_ptr of GraphNode
   */
  void addGraphNode(std::shared_ptr<GraphNode> node);

  /**
   * @brief     make adjancency list for the current graph
   */
  void
  makeAdjacencyList(std::vector<std::list<std::shared_ptr<GraphNode>>> &adj);

  /**
   * @brief     Get index of the node with given name
   *
   * @param     name Name of the node
   * @return    internal index of the node
   */
  unsigned int getNodeIdx(const std::string &name);

  /**
   * @brief     Check if the given node can execute in-place
   *
   * @param lnode node to check for in-place execution
   *
   * @return the mode of inplace for the layer
   */
  InPlaceType canExecuteInPlace(const std::shared_ptr<LayerNode> &lnode);

  /**
   * @brief compute optimized backward end. This function calculated the valid
   * end of the graph backward, if memory_optimize is unset, this returns
   * beginning of the graph node.
   *
   * @return end of the backward iter;
   */
  LayerNode *computeBackwardEnd();
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __NETWORK_GRAPH_H__ */
