// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   qnn_transformer.cpp
 * @date   31 Mar 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This file defines QNNTransformer's basic actions
 */

#include <fstream>

#include <engine.h>
#include <model.h>

#include <tokenizers_cpp.h>

#include <qnn_transformer.h>

namespace causallm {

static std::string LoadBytesFromFile(const std::string &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + path);
  }
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::string buffer(size, ' ');
  if (!file.read(&buffer[0], size)) {
    throw std::runtime_error("Failed to read file: " + path);
  }
  return buffer;
}

static ModelType strToModelType(std::string model_type) {

  std::string model_type_lower = model_type;
  std::transform(model_type_lower.begin(), model_type_lower.end(),
                 model_type_lower.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  static const std::unordered_map<std::string, ModelType> model_type_map = {
    {"model", ModelType::MODEL},
    {"causallm", ModelType::CAUSALLM},
    {"embedding", ModelType::EMBEDDING}};

  if (model_type_map.find(model_type_lower) == model_type_map.end()) {
    return ModelType::UNKNOWN;
  }

  return model_type_map.at(model_type_lower);
}

QNNTransformer::QNNTransformer(json &cfg, json &generation_cfg, json &nntr_cfg,
                               ModelType model_type) {

  std::string config_model_type_str = "Model";
  if (nntr_cfg.contains("model_type")) {
    config_model_type_str = nntr_cfg["model_type"].get<std::string>();
  }

  ModelType config_model_type = strToModelType(config_model_type_str);

  if (model_type != config_model_type) {
    throw std::runtime_error("model_type mismatch. Class Type: " +
                             std::to_string(static_cast<int>(model_type)) +
                             ", Config Type: " + config_model_type_str);
  }

  setupParameters(cfg, generation_cfg, nntr_cfg);

  // prep tokenizer
  tokenizer = tokenizers::Tokenizer::FromBlobJSON(
    LoadBytesFromFile(nntr_cfg["tokenizer_file"]));
}

void QNNTransformer::setupParameters(json &cfg, json &generation_cfg,
                                     json &nntr_cfg) {
  /** Initialize nntr parameters */
  BATCH_SIZE = nntr_cfg["batch_size"].get<unsigned int>();
  MODEL_TENSOR_TYPE = nntr_cfg["model_tensor_type"].get<std::string>();
  INIT_SEQ_LEN = nntr_cfg["init_seq_len"];
  MAX_SEQ_LEN = nntr_cfg["max_seq_len"];
  NUM_TO_GENERATE = nntr_cfg["num_to_generate"];
  MEMORY_SWAP = nntr_cfg.contains("fsu") ? nntr_cfg["fsu"].get<bool>() : false;
  FSU_LOOKAHEAD = nntr_cfg.contains("fsu_lookahead")
                    ? nntr_cfg["fsu_lookahead"].get<unsigned int>()
                    : 1;
  EMBEDDING_DTYPE = nntr_cfg["embedding_dtype"];
  FC_LAYER_DTYPE = nntr_cfg["fc_layer_dtype"];

  /** Initialize model parameters from QNN config */
  if (cfg.contains("vocab_size"))
    NUM_VOCAB = cfg["vocab_size"];
  if (cfg.contains("hidden_size"))
    DIM = cfg["hidden_size"];
  if (cfg.contains("num_hidden_layers"))
    NUM_LAYERS = cfg["num_hidden_layers"];

  /** QNN-specific config */
  if (nntr_cfg.contains("qnn_context_bin"))
    QNN_CONTEXT_BIN = nntr_cfg["qnn_context_bin"].get<std::string>();
}

void QNNTransformer::initialize() {
  constructModel();

  std::vector<std::string> model_props = {
    withKey("batch_size", BATCH_SIZE), withKey("epochs", "1"),
    withKey("model_tensor_type", MODEL_TENSOR_TYPE)};
  if (MEMORY_SWAP) {
    model_props.emplace_back(withKey("fsu", "true"));
    model_props.emplace_back(withKey("fsu_lookahead", FSU_LOOKAHEAD));
  }

  model->setProperty(model_props);

  if (model->compile(ml::train::ExecutionMode::INFERENCE)) {
    throw std::invalid_argument("QNN Model compilation failed.");
  }

  if (model->initialize(ml::train::ExecutionMode::INFERENCE)) {
    throw std::invalid_argument("QNN Model initialization failed.");
  }

  is_initialized = true;
}

void QNNTransformer::constructModel() {
  model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);

  /// @note QNN model construction is done via loading a pre-compiled
  /// QNN context binary, not by constructing layers programmatically.
  /// Subclasses should override this if they need custom QNN graph setup.
}

void QNNTransformer::load_weight(const std::string &weight_path) {
  if (!is_initialized) {
    throw std::runtime_error(
      "QNNTransformer model is not initialized. Please call "
      "initialize() before load_weight().");
  }

  try {
    model->load(weight_path, ml::train::ModelFormat::MODEL_FORMAT_QNN);
  } catch (const std::exception &e) {
    throw std::runtime_error("Failed to load QNN model weights: " +
                             std::string(e.what()));
  }
}

void QNNTransformer::save_weight(const std::string &weight_path) {
  if (!is_initialized) {
    throw std::runtime_error(
      "QNNTransformer model is not initialized. Please call "
      "initialize() before save_weight().");
  }

  try {
    model->save(weight_path, ml::train::ModelFormat::MODEL_FORMAT_BIN);
  } catch (const std::exception &e) {
    throw std::runtime_error("Failed to save QNN model weights: " +
                             std::string(e.what()));
  }
}

void QNNTransformer::run(const WSTR prompt, void *output_buf,
                         bool log_output) {
  run(prompt, "", "", output_buf, log_output);
}

void QNNTransformer::run(const WSTR prompt, const WSTR system_prompt,
                         const WSTR tail_prompt, void *output_buf,
                         bool log_output) {
  if (!is_initialized) {
    throw std::runtime_error(
      "QNNTransformer model is not initialized. Please call "
      "initialize() before run().");
  }
  /// @note This part should be filled in by derived classes (e.g. QNNCausalLM).
}

} // namespace causallm
