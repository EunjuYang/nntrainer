// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   transformer_base.h
 * @date   31 Mar 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   This transformer_base.h defines an abstract base class for
 * Transformer-based models. It provides the common interface and shared state
 * that both NNTrainer-based Transformer and QNN-based Transformer can inherit.
 */

#ifndef __TRANSFORMER_BASE_H__
#define __TRANSFORMER_BASE_H__

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#define WSTR std::string
#define WCHAR_P std::string &
#else
#define WIN_EXPORT
#define WSTR std::string
#define WCHAR_P std::string &
#endif

#include <cerrno>
#include <cstring>
#include <map>
#include <memory>
#include <string>

#include <layer.h>
#include <model.h>
#include <tensor_api.h>

#include <fstream>
#include <tokenizers_c.h>
#include <tokenizers_cpp.h>

#include "json.hpp"
#include "performance_metrics.h"

namespace causallm {

/*** ALIAS ****/
using LayerHandle = ml::train::LayerHandle;
using Tensor = ml::train::Tensor;
using ModelHandle = std::unique_ptr<ml::train::Model>;

using json = nlohmann::json;

/**
 * @brief Model Type Enum
 */
enum class ModelType { MODEL, CAUSALLM, EMBEDDING, UNKNOWN };

/**
 * @brief TransformerBase Abstract Class
 * @note  This is the common interface for all Transformer-based models.
 *        Both NNTrainer Transformer and QNN Transformer inherit from this
 */
WIN_EXPORT class TransformerBase {

public:
  /**
   * @brief Default constructor
   */
  TransformerBase() = default;

  /**
   * @brief Destroy the TransformerBase object
   */
  virtual ~TransformerBase() = default;

  /**
   * @brief Initialize and Construct the Transformer model
   */
  virtual void initialize() = 0;

  /**
   * @brief Load the model weights from a file
   */
  virtual void load_weight(const std::string &weight_path) = 0;

  /**
   * @brief Save the weight to a file
   */
  virtual void save_weight(const std::string &weight_path) = 0;

  /**
   * @brief Save the weight to a file with type conversion
   * @param weight_path Path to save the weight file
   * @param dtype Global target data type for all layers (NONE = keep original)
   * @param layer_dtype_map Per-layer data type overrides (layer name -> dtype)
   * @param target_isa Target ISA for quantization (default: DEFAULT)
   */
  virtual void
  save_weight(const std::string &weight_path,
              ml::train::TensorDim::DataType dtype,
              const std::map<std::string, ml::train::TensorDim::DataType>
                &layer_dtype_map = {},
              ml::train::ISA target_isa = ml::train::ISA::DEFAULT) {
    (void)weight_path;
    (void)dtype;
    (void)layer_dtype_map;
    (void)target_isa;
    throw std::runtime_error(
      "This TransformerBase implementation does not support dtype conversion "
      "while saving weights.");
  }

  /**
   * @brief run the Transformer model (simple)
   * @param prompt User prompt
   * @param output_buf Optional output pointer. For CausalLM, pass
   *                   std::vector<std::string>*. For Sentence Transformer, pass
   *                   std::vector<float *>*. nullptr to skip output collection.
   * @param log_output Whether to log output to stdout
   */
  virtual void run(const WSTR prompt, void *output_buf = nullptr,
                   bool log_output = true) = 0;

  /**
   * @brief run the Transformer model (full)
   * @param prompt User prompt
   * @param system_prompt System prompt prepended to user prompt
   * @param tail_prompt Tail prompt appended to user prompt
   * @param output_buf Optional output pointer (see simple overload for types)
   * @param log_output Whether to log output to stdout
   */
  virtual void run(const WSTR prompt, const WSTR system_prompt = WSTR(),
                   const WSTR tail_prompt = WSTR(),
                   void *output_buf = nullptr, bool log_output = true) = 0;

protected:
  bool is_initialized = false; /**< Flag to check if the model is initialized */
  ModelHandle model;

  /** tokenizer */
  std::unique_ptr<tokenizers::Tokenizer> tokenizer;

  unsigned int NUM_VOCAB;
  int DIM;
  int NUM_LAYERS;

  unsigned int MAX_SEQ_LEN;
  unsigned int BATCH_SIZE;
  unsigned int INIT_SEQ_LEN;
  unsigned int NUM_TO_GENERATE;
};

/**
 * Loads JSON data from a file with detailed error handling
 * @param file_path Path to JSON file
 * @return JSON object
 * @throws std::runtime_error on file open or parse failure
 */
inline json LoadJsonFile(const std::string &file_path) {
  std::ifstream file(file_path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + file_path +
                             " | Reason: " + std::strerror(errno));
  }

  try {
    json data;
    file >> data;
    return data;
  } catch (const json::parse_error &e) {
    throw std::runtime_error("JSON parse error in " + file_path +
                             " | Details: " + e.what());
  }
}

} // namespace causallm

#endif
