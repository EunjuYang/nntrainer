// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file    subgraph_cpu.h
 * @date    06 Jan 2025
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Eunju Yang <ej.yang@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   This is a Subgraph Class for CPU engine
 *
 */

#ifndef __SUBGRAPH_CPU_H__
#define __SUBGRAPH_CPU_H__
#ifdef __cplusplus

#include <iostream>

#include <graph_core.h>

namespace nntrainer {

class SubGraphCpu : public SubGraphBase {

public:
  /**
   * @brief     Constructor of SubGraphCpu Class
   */
  SubGraphCpu() : SubGraphBase() {}

  /**
   * @brief     Destructor of SubGraphCpu Class
   */
  ~SubGraphCpu() {}

  /**
   * @brief Compile the subgraph running with a cpu engine
   * @param[in] loss_type loss for the graph
   * @retval ML_ERROR_NONE on success, error on failure
   */
  int compile(const std::string &loss_type) override;

  /**
   * @brief initialize the subgraph running with a cpu engine
   * @param model_input_names model input connection if empty list given, all of
   * node that can be inputs will be identified in the sort order
   * @param model_label_names model label names if empty list given, all of node
   * that can be labels will be identified in the sort order
   * @return int ML_ERROR_NONE if successful
   */
  int initialize(
    std::shared_ptr<Manager> tensor_manager_,
    ExecutionMode mode = ExecutionMode::TRAIN,
    const std::vector<Connection> &model_input_names = {},
    const std::vector<Connection> &model_label_names = {}) override;

  /**
   * @brief reinitialize the subgraph running with a cpu engine
   * @param model_input_names model input connection if empty list given, all of
   * node that can be inputs will be identified in the sort order
   * @param model_label_names model label names if empty list given, all of node
   * that can be labels will be identified in the sort order
   * @return int ML_ERROR_NONE if successful
   */
  int reinitialize(
    const std::vector<Connection> &model_input_names = {},
    const std::vector<Connection> &model_label_names = {}) override;

  /**
   * @brief Create run layer context from the given init layer context
   *
   * @param lnode layer node to finalize and set run context
   * @param prev_inputs previous input information
   * @todo needs to be updated. finalize context should be called at the level
   * of subgraph (same subgraph might need to share the layer context)
   */
  std::vector<Var_Grad *>
  finalizeContext(const std::shared_ptr<LayerNode> &lnode,
                  const std::vector<Var_Grad *> &prev_inputs) override;

  /**
   * @brief Recreate run layer context from the given init layer context
   *
   * @param lnode layer node to finalize and set run context
   * @param prev_inputs previous input information
   * @todo needs to be updated. finalize context should be called at the level
   * of subgraph (same subgraph might need to share the layer context)
   */
  std::vector<Var_Grad *>
  refinalizeContext(const std::shared_ptr<LayerNode> &lnode,
                    const std::vector<Var_Grad *> &prev_inputs) override;

  /**
   * @brief     forwarding the subgraph running with a cpu engine
   * @param[in] training true if forwarding is on training
   * @retval output tensors
   */
  sharedConstTensors forwarding(
    bool training = false,
    std::function<bool(void *userdata)> stop_cb =
      [](void *user_data) { return false; },
    void *user_data = nullptr, bool swap_mode = false) override;

  /**
   * @brief     forwarding the subgraph running with a cpu engine
   * @param[in] from start step
   * @param[in] to end step
   * @param[in] training true if forwarding is on training
   * @retval output tensors
   */
  sharedConstTensors incremental_forwarding(
    unsigned int from, unsigned int to, bool training = false,
    std::function<bool(void *userdata)> stop_cb =
      [](void *user_data) { return false; },
    void *user_data = nullptr) override;

  /**
   * @brief     backwarding the subgraph running with a cpu engine
   * @param[in] iteration current iteration number
   * @param[in] lazy_apply_grad_op operation for applying the lazy gradients
   * @retval ret it is false then the gradient has NaN valude in mixed precision
   * training. If it is, then we need to control the loss scale factor and
   * compute again the derivatives.
   */
  bool backwarding(
    int iteration, std::function<void(Weight &, int)> &lazy_apply_grad_op,
    std::function<bool(void *userdata)> stop_cb =
      [](void *user_data) { return false; },
    void *user_data = nullptr, bool is_grad_opt_mode = false) override;

  void applyGradients(LayerNode *node, int iteration) override;

  /**
   * @brief backwarding operation
   */
  bool backwarding_op(std::shared_ptr<LayerNode> node, int iteration,
                      std::function<bool(void *userdata)> stop_cb,
                      void *userdata, bool is_grad_opt_mode);
  /*
   * @brief forwarding_op function
   */
  void incremental_forwarding_op(std::shared_ptr<LayerNode> node,
                                 unsigned int from, unsigned int to,
                                 bool training);

  /**
   * @brief forwarding_op function
   */
  void forwarding_op(std::shared_ptr<LayerNode> node, bool training,
                     bool swap_mode = false);

  /**
   * @brief apply_op function
   */
  void apply_grad_op(Weight &w, int iteration);
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __SUBGRAPH_CPU_H__ */