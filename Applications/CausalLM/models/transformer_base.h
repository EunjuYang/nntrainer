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
#define WSTR std::wstring
#define WCHAR_P wchar_t *
#else
#define WIN_EXPORT
#define WSTR std::string
#define WCHAR_P std::string &
#endif

#include <layer.h>
#include <model.h>

#include <fstream>
#include <tokenizers_c.h>
#include <tokenizers_cpp.h>

#include "json.hpp"

namespace causallm {

/*** ALIAS ****/
using LayerHandle = std::shared_ptr<ml::train::Layer>;
using ModelHandle = std::unique_ptr<ml::train::Model>;

using json = nlohmann::json;

/**
 * @brief Model Type Enum
 */
enum class ModelType { MODEL, CAUSALLM, EMBEDDING, UNKNOWN };

/**
 * @brief TransformerBase Abstract Class
 * @note  This is the common interface for all Transformer-based models.
 *        Both NNTrainer Transformer and QNN Transformer inherit from this.
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
   * @brief run the Transformer model
   */
  virtual void run(const WSTR prompt, bool do_sample = false,
                   const WSTR system_prompt = "",
                   const WSTR tail_prompt = "",
                   bool log_output = true) = 0;

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
