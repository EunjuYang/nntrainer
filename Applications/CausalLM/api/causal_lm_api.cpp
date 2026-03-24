// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    causal_lm_api.cpp
 * @date    21 Jan 2026
 * @brief   This is a C API for CausalLM application
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Eunju Yang <ej.yang@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#include "causal_lm_api.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "causal_lm.h"
#include "embedding_gemma.h"
#include "gemma3_causallm.h"
#include "gptoss_cached_slim_causallm.h"
#include "gptoss_causallm.h"
#include "json.hpp"
#include "model_config_internal.h"
#include "qwen2_causallm.h"
#include "qwen2_embedding.h"
#include "qwen3_cached_slim_moe_causallm.h"
#include "qwen3_causallm.h"
#include "qwen3_embedding.h"
#include "qwen3_moe_causallm.h"
#include "qwen3_slim_moe_causallm.h"
#include "sentence_transformer.h"
#include <factory.h>
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>

using json = nlohmann::json;

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------
static std::unique_ptr<causallm::Transformer> g_model;
static std::mutex g_mutex;
static bool g_initialized = false;
static std::string g_architecture = "";
static bool g_use_chat_template = false;
static bool g_verbose = false;
static std::string g_last_output = "";
static double g_initialization_duration_ms = 0.0;
static std::vector<float *> g_last_embedding_output;

static std::map<std::string, std::string> g_model_path_map = {
  {"QWEN3-0.6B", "qwen3-0.6b"},
  {"EMBEDDING-QWEN3", "embedding-qwen3"},
  {"EMBEDDING-QWEN2", "embedding-qwen2"},
  {"EMBEDDING-GEMMA3", "embedding-gemma3"},
};

struct RegisteredModel {
  std::string arch_name;
  ModelRuntimeConfig config;
};
static std::map<std::string, RegisteredModel> g_model_registry;
static std::map<std::string, ModelArchConfig> g_arch_config_map;

// ---------------------------------------------------------------------------
// Runtime type checks (using dynamic_cast on the loaded model)
// ---------------------------------------------------------------------------
static bool is_embedding_model() {
  return dynamic_cast<causallm::SentenceTransformer *>(g_model.get()) !=
         nullptr;
}

// ---------------------------------------------------------------------------
// Factory registration
// ---------------------------------------------------------------------------
static void register_models() {
  static std::once_flag flag;
  std::call_once(flag, []() {
    auto &F = causallm::Factory::Instance();

    // CausalLM models
    F.registerModel("LlamaForCausalLM", [](json c, json g, json n) {
      return std::make_unique<causallm::CausalLM>(c, g, n);
    });
    F.registerModel("Qwen2ForCausalLM", [](json c, json g, json n) {
      return std::make_unique<causallm::Qwen2CausalLM>(c, g, n);
    });
    F.registerModel("Qwen3ForCausalLM", [](json c, json g, json n) {
      return std::make_unique<causallm::Qwen3CausalLM>(c, g, n);
    });
    F.registerModel("Qwen3MoeForCausalLM", [](json c, json g, json n) {
      return std::make_unique<causallm::Qwen3MoECausalLM>(c, g, n);
    });
    F.registerModel("Qwen3SlimMoeForCausalLM", [](json c, json g, json n) {
      return std::make_unique<causallm::Qwen3SlimMoECausalLM>(c, g, n);
    });
    F.registerModel("Qwen3CachedSlimMoeForCausalLM",
                    [](json c, json g, json n) {
                      return std::make_unique<
                        causallm::Qwen3CachedSlimMoECausalLM>(c, g, n);
                    });
    F.registerModel("GptOssForCausalLM", [](json c, json g, json n) {
      return std::make_unique<causallm::GptOssForCausalLM>(c, g, n);
    });
    F.registerModel("GptOssCachedSlimCausalLM", [](json c, json g, json n) {
      return std::make_unique<causallm::GptOssCachedSlimCausalLM>(c, g, n);
    });
    F.registerModel("Gemma3ForCausalLM", [](json c, json g, json n) {
      return std::make_unique<causallm::Gemma3CausalLM>(c, g, n);
    });

    // Embedding models
    F.registerModel("Qwen3Embedding", [](json c, json g, json n) {
      return std::make_unique<causallm::Qwen3Embedding>(c, g, n);
    });
    F.registerModel("Qwen2Embedding", [](json c, json g, json n) {
      return std::make_unique<causallm::Qwen2Embedding>(c, g, n);
    });
    F.registerModel("EmbeddingGemma", [](json c, json g, json n) {
      return std::make_unique<causallm::EmbeddingGemma>(c, g, n);
    });

    register_builtin_model_configs();
  });
}

// ---------------------------------------------------------------------------
// ModelType → model name / architecture helpers
// ---------------------------------------------------------------------------
static const char *get_model_name_from_type(ModelType type) {
  switch (type) {
  case CAUSAL_LM_MODEL_QWEN3_0_6B:
    return "QWEN3-0.6B";
  case CAUSAL_LM_MODEL_EMBEDDING_QWEN3:
    return "EMBEDDING-QWEN3";
  case CAUSAL_LM_MODEL_EMBEDDING_QWEN2:
    return "EMBEDDING-QWEN2";
  case CAUSAL_LM_MODEL_EMBEDDING_GEMMA3:
    return "EMBEDDING-GEMMA3";
  default:
    return nullptr;
  }
}

/**
 * @brief Resolve the factory architecture key from backbone + model_type.
 * @details For embedding models, config.json contains the backbone architecture
 *          (e.g., "Qwen3ForCausalLM") while the actual factory key is the
 *          embedding variant (e.g., "Qwen3Embedding"). This mirrors the
 *          resolve_architecture() logic in main.cpp.
 * @param model_type From nntr_cfg["model_type"] (e.g., "CausalLM", "Embedding")
 * @param backbone   From cfg["architectures"][0] (e.g., "Qwen3ForCausalLM")
 * @return Resolved factory key
 */
static std::string resolve_architecture(const std::string &model_type,
                                        const std::string &backbone) {
  std::string mt = model_type;
  std::transform(mt.begin(), mt.end(), mt.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  if (mt == "embedding") {
    if (backbone == "Qwen3ForCausalLM")
      return "Qwen3Embedding";
    if (backbone == "Gemma3ForCausalLM" || backbone == "Gemma3TextModel")
      return "EmbeddingGemma";
    if (backbone == "Qwen2Model")
      return "Qwen2Embedding";
    throw std::invalid_argument(
      "Unsupported backbone for embedding model: " + backbone);
  }

  return backbone;
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------
static std::string apply_chat_template(const std::string &architecture,
                                       const std::string &input) {
  if (architecture == "LlamaForCausalLM") {
    return "[INST] " + input + " [/INST]";
  } else if (architecture == "Qwen2ForCausalLM" ||
             architecture == "Qwen3ForCausalLM" ||
             architecture == "Qwen3MoeForCausalLM" ||
             architecture == "Qwen3SlimMoeForCausalLM" ||
             architecture == "Qwen3CachedSlimMoeForCausalLM") {
    return "<|im_start|>user\n" + input + "<|im_end|>\n<|im_start|>assistant\n";
  } else if (architecture == "Gemma3ForCausalLM") {
    return "<start_of_turn>user\n" + input +
           "<end_of_turn>\n<start_of_turn>model\n";
  }
  return input;
}

static std::string get_quantization_suffix(ModelQuantizationType type) {
  switch (type) {
  case CAUSAL_LM_QUANTIZATION_W4A32:
    return "-w4a32";
  case CAUSAL_LM_QUANTIZATION_W16A16:
    return "-w16a16";
  case CAUSAL_LM_QUANTIZATION_W8A16:
    return "-w8a16";
  case CAUSAL_LM_QUANTIZATION_W32A32:
    return "-w32a32";
  default:
    return "-w4a32";
  }
}

static std::string get_quantization_suffix_upper(ModelQuantizationType type) {
  switch (type) {
  case CAUSAL_LM_QUANTIZATION_W4A32:
    return "-W4A32";
  case CAUSAL_LM_QUANTIZATION_W16A16:
    return "-W16A16";
  case CAUSAL_LM_QUANTIZATION_W8A16:
    return "-W8A16";
  case CAUSAL_LM_QUANTIZATION_W32A32:
    return "-W32A32";
  default:
    return "";
  }
}

static std::string resolve_model_path(const std::string &model_key,
                                      ModelQuantizationType quant_type) {
  std::string path_upper = model_key;
  std::transform(path_upper.begin(), path_upper.end(), path_upper.begin(),
                 ::toupper);

  std::string base_dir_name;
  if (g_model_path_map.count(path_upper)) {
    base_dir_name = g_model_path_map[path_upper];
  } else {
    base_dir_name = path_upper;
    std::transform(base_dir_name.begin(), base_dir_name.end(),
                   base_dir_name.begin(), ::tolower);
  }

  return "./models/" + base_dir_name + get_quantization_suffix(quant_type);
}

static bool check_file_exists(const std::string &path) {
  struct stat buffer;
  return (stat(path.c_str(), &buffer) == 0);
}

// ---------------------------------------------------------------------------
// Internal config → JSON conversion helpers
// ---------------------------------------------------------------------------

/** Populate cfg (config.json equivalent) from ModelArchConfig */
static void populate_arch_config(json &cfg, json &generation_cfg,
                                 const ModelArchConfig &ac) {
  cfg["vocab_size"] = ac.vocab_size;
  cfg["hidden_size"] = ac.hidden_size;
  cfg["intermediate_size"] = ac.intermediate_size;
  cfg["num_hidden_layers"] = ac.num_hidden_layers;
  cfg["num_attention_heads"] = ac.num_attention_heads;
  cfg["head_dim"] = ac.head_dim;
  cfg["num_key_value_heads"] =
    ac.num_key_value_heads > 0 ? ac.num_key_value_heads : ac.num_attention_heads;
  cfg["max_position_embeddings"] = ac.max_position_embeddings;
  cfg["rope_theta"] = ac.rope_theta;
  cfg["rms_norm_eps"] = ac.rms_norm_eps;
  cfg["tie_word_embeddings"] = ac.tie_word_embeddings;
  cfg["sliding_window"] =
    (ac.sliding_window != UINT_MAX) ? json(ac.sliding_window) : json(nullptr);
  cfg["sliding_window_pattern"] = ac.sliding_window_pattern;
  cfg["architectures"] = {std::string(ac.architecture)};

  if (ac.num_eos_token_ids > 0) {
    std::vector<unsigned int> eos_ids(ac.eos_token_ids,
                                      ac.eos_token_ids + ac.num_eos_token_ids);
    generation_cfg["eos_token_id"] = eos_ids;
  }
  generation_cfg["bos_token_id"] = ac.bos_token_id;
}

/** Populate nntr_cfg (nntr_config.json equivalent) from ModelRuntimeConfig */
static void populate_runtime_config(json &nntr_cfg, json &generation_cfg,
                                    const ModelRuntimeConfig &rc,
                                    const std::string &model_dir_path) {
  nntr_cfg["batch_size"] = rc.batch_size;
  nntr_cfg["model_type"] = std::string(rc.model_type);
  nntr_cfg["model_tensor_type"] = std::string(rc.model_tensor_type);
  nntr_cfg["init_seq_len"] = rc.init_seq_len;
  nntr_cfg["max_seq_len"] = rc.max_seq_len;
  nntr_cfg["num_to_generate"] = rc.num_to_generate;
  nntr_cfg["fsu"] = rc.fsu;
  nntr_cfg["fsu_lookahead"] = rc.fsu_lookahead;
  nntr_cfg["embedding_dtype"] = std::string(rc.embedding_dtype);
  nntr_cfg["fc_layer_dtype"] = std::string(rc.fc_layer_dtype);
  nntr_cfg["model_file_name"] = std::string(rc.model_file_name);
  nntr_cfg["tokenizer_file"] =
    model_dir_path + "/" + std::string(rc.tokenizer_file);

  if (strlen(rc.lmhead_dtype) > 0)
    nntr_cfg["lmhead_dtype"] = std::string(rc.lmhead_dtype);

  if (rc.num_bad_word_ids > 0) {
    std::vector<unsigned int> bad_ids(rc.bad_word_ids,
                                      rc.bad_word_ids + rc.num_bad_word_ids);
    nntr_cfg["bad_word_ids"] = bad_ids;
  }

  generation_cfg["top_k"] = rc.top_k;
  generation_cfg["top_p"] = rc.top_p;
  generation_cfg["temperature"] = rc.temperature;
  generation_cfg["do_sample"] = false;

  // Embedding-specific: inline modules take priority over file path
  if (strlen(rc.modules_config) > 0) {
    nntr_cfg["modules"] = json::parse(rc.modules_config);
  } else if (strlen(rc.module_config_path) > 0) {
    nntr_cfg["module_config_path"] =
      model_dir_path + "/" + std::string(rc.module_config_path);
  }
}

// ---------------------------------------------------------------------------
// Debug validation
// ---------------------------------------------------------------------------
static void validate_models() {
  std::cout << "[DEBUG] Validating model files..." << std::endl;
  for (auto const &[key, val] : g_model_path_map) {
    std::vector<ModelQuantizationType> quant_types = {
      CAUSAL_LM_QUANTIZATION_UNKNOWN, CAUSAL_LM_QUANTIZATION_W4A32,
      CAUSAL_LM_QUANTIZATION_W16A16, CAUSAL_LM_QUANTIZATION_W32A32};

    for (auto qt : quant_types) {
      std::string quant_suffix = get_quantization_suffix(qt);

      std::string lookup_key = key;
      if (qt != CAUSAL_LM_QUANTIZATION_UNKNOWN) {
        std::transform(quant_suffix.begin(), quant_suffix.end(),
                       quant_suffix.begin(), ::toupper);
        lookup_key += quant_suffix;
      }

      std::string resolved_path = resolve_model_path(key, qt);

      if (g_model_registry.count(lookup_key)) {
        RegisteredModel &rm = g_model_registry[lookup_key];
        std::string full_path =
          resolved_path + "/" + rm.config.model_file_name;
        std::cout << (check_file_exists(full_path) ? "  [OK] " : "  [FAIL] ")
                  << "Reg Config: " << lookup_key << " -> " << full_path
                  << std::endl;
      } else if (check_file_exists(resolved_path)) {
        bool has_config = check_file_exists(resolved_path + "/config.json");
        bool has_nntr =
          check_file_exists(resolved_path + "/nntr_config.json");
        std::cout
          << ((has_config && has_nntr) ? "  [OK] " : "  [FAIL] ")
          << "External Config: " << lookup_key << " -> " << resolved_path
          << std::endl;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Public API: setOptions / registerModelArchitecture / registerModel
// ---------------------------------------------------------------------------
ErrorCode setOptions(Config config) {
  g_use_chat_template = config.use_chat_template;
  g_verbose = config.verbose;
  if (config.debug_mode) {
    register_models();
    validate_models();
  }
  return CAUSAL_LM_ERROR_NONE;
}

ErrorCode registerModelArchitecture(const char *arch_name,
                                    ModelArchConfig config) {
  if (arch_name == nullptr)
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  std::lock_guard<std::mutex> lock(g_mutex);
  std::string name(arch_name);
  std::transform(name.begin(), name.end(), name.begin(), ::toupper);
  g_arch_config_map[name] = config;
  return CAUSAL_LM_ERROR_NONE;
}

ErrorCode registerModel(const char *model_name, const char *arch_name,
                        ModelRuntimeConfig config) {
  if (model_name == nullptr || arch_name == nullptr)
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  std::lock_guard<std::mutex> lock(g_mutex);
  std::string name(model_name);
  std::transform(name.begin(), name.end(), name.begin(), ::toupper);
  std::string aname(arch_name);
  std::transform(aname.begin(), aname.end(), aname.begin(), ::toupper);
  g_model_registry[name] = {aname, config};
  return CAUSAL_LM_ERROR_NONE;
}

// ---------------------------------------------------------------------------
// Public API: loadModel (unified for CausalLM and Embedding)
// ---------------------------------------------------------------------------
ErrorCode loadModel(BackendType compute, ModelType modeltype,
                    ModelQuantizationType quant_type) {

  auto start_init = std::chrono::high_resolution_clock::now();

  register_models();

  const char *model_name = get_model_name_from_type(modeltype);
  if (model_name == nullptr)
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;

  std::lock_guard<std::mutex> lock(g_mutex);
  try {
    std::string model_name_upper(model_name);
    std::transform(model_name_upper.begin(), model_name_upper.end(),
                   model_name_upper.begin(), ::toupper);
    std::string lookup_name = model_name_upper +
                              get_quantization_suffix_upper(quant_type);
    std::string model_dir_path = resolve_model_path(model_name, quant_type);

    json cfg, generation_cfg, nntr_cfg;
    std::string architecture;

    if (g_model_registry.count(lookup_name)) {
      // ---- Internal configuration (registered in model_config.cpp) ----
      RegisteredModel &rm = g_model_registry[lookup_name];
      if (!g_arch_config_map.count(rm.arch_name)) {
        std::cerr << "Architecture '" << rm.arch_name
                  << "' not found for model '" << lookup_name << "'"
                  << std::endl;
        return CAUSAL_LM_ERROR_MODEL_LOAD_FAILED;
      }
      populate_arch_config(cfg, generation_cfg, g_arch_config_map[rm.arch_name]);
      populate_runtime_config(nntr_cfg, generation_cfg, rm.config,
                              model_dir_path);
    } else {
      // ---- External configuration (file-based) ----
      cfg = causallm::LoadJsonFile(model_dir_path + "/config.json");
      try {
        generation_cfg =
          causallm::LoadJsonFile(model_dir_path + "/generation_config.json");
      } catch (...) {
        generation_cfg = json::object();
      }
      nntr_cfg =
        causallm::LoadJsonFile(model_dir_path + "/nntr_config.json");

      if (nntr_cfg.contains("tokenizer_file")) {
        nntr_cfg["tokenizer_file"] =
          model_dir_path + "/" +
          nntr_cfg["tokenizer_file"].get<std::string>();
      }
      if (nntr_cfg.contains("module_config_path")) {
        std::string mp = nntr_cfg["module_config_path"].get<std::string>();
        if (mp.find('/') == std::string::npos)
          nntr_cfg["module_config_path"] = model_dir_path + "/" + mp;
      }
    }

    // Resolve architecture: extract backbone from cfg["architectures"],
    // then resolve to embedding variant if model_type is "Embedding".
    if (!cfg.contains("architectures") || !cfg["architectures"].is_array() ||
        cfg["architectures"].empty()) {
      return CAUSAL_LM_ERROR_INVALID_PARAMETER;
    }
    std::string backbone =
      cfg["architectures"].get<std::vector<std::string>>()[0];
    std::string model_type_str =
      nntr_cfg.value("model_type", "CausalLM");
    architecture = resolve_architecture(model_type_str, backbone);

    // Create, initialize, load
    g_model = causallm::Factory::Instance().create(architecture, cfg,
                                                   generation_cfg, nntr_cfg);
    if (!g_model)
      return CAUSAL_LM_ERROR_MODEL_LOAD_FAILED;

    std::string weight_file_name =
      nntr_cfg.value("model_file_name", "pytorch_model.bin");
    g_model->initialize();
    g_model->load_weight(model_dir_path + "/" + weight_file_name);

    g_initialized = true;
    g_architecture = architecture;
    g_initialization_duration_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start_init)
        .count();

  } catch (const std::exception &e) {
    std::cerr << "Exception in loadModel: " << e.what() << std::endl;
    return CAUSAL_LM_ERROR_MODEL_LOAD_FAILED;
  } catch (...) {
    std::cerr << "Unknown exception in loadModel" << std::endl;
    return CAUSAL_LM_ERROR_MODEL_LOAD_FAILED;
  }

  return CAUSAL_LM_ERROR_NONE;
}

// ---------------------------------------------------------------------------
// Public API: runModel (text output, CausalLM only)
// ---------------------------------------------------------------------------
ErrorCode runModel(const char *inputTextPrompt, const char **outputText) {
  if (!g_initialized || !g_model)
    return CAUSAL_LM_ERROR_NOT_INITIALIZED;
  if (inputTextPrompt == nullptr || outputText == nullptr)
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  if (is_embedding_model()) {
    std::cerr << "runModel: text output is not supported for embedding models. "
                 "Use runModelFloat() instead."
              << std::endl;
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }

  try {
    std::lock_guard<std::mutex> lock(g_mutex);
    std::string input(inputTextPrompt);

    if (g_use_chat_template)
      input = apply_chat_template(g_architecture, input);

#if defined(_WIN32)
    g_model->run(std::wstring(input.begin(), input.end()), false, L"", L"",
                 g_verbose);
#else
    g_model->run(input, false, "", "", g_verbose);
#endif

    g_last_output = "";
    auto *causal = dynamic_cast<causallm::CausalLM *>(g_model.get());
    if (causal)
      g_last_output = causal->getOutput(0);

    *outputText = g_last_output.c_str();

  } catch (const std::exception &e) {
    std::cerr << "Exception in runModel: " << e.what() << std::endl;
    return CAUSAL_LM_ERROR_INFERENCE_FAILED;
  }

  return CAUSAL_LM_ERROR_NONE;
}

// ---------------------------------------------------------------------------
// Public API: runModelFloat (float vector output, Embedding only)
// ---------------------------------------------------------------------------
ErrorCode runModelFloat(const char *inputTextPrompt, float **outputData,
                         unsigned int *outputDim, unsigned int *outputLength) {
  if (!g_initialized || !g_model)
    return CAUSAL_LM_ERROR_NOT_INITIALIZED;
  if (inputTextPrompt == nullptr || outputData == nullptr ||
      outputDim == nullptr || outputLength == nullptr)
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;

  auto *encoder =
    dynamic_cast<causallm::SentenceTransformer *>(g_model.get());
  if (!encoder) {
    std::cerr << "runModelFloat: float output is not supported for CausalLM "
                 "models. Use runModel() instead."
              << std::endl;
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }

  try {
    std::lock_guard<std::mutex> lock(g_mutex);
    std::string input(inputTextPrompt);

#if defined(_WIN32)
    g_last_embedding_output =
      encoder->encode(std::wstring(input.begin(), input.end()), L"", L"");
#else
    g_last_embedding_output = encoder->encode(input, "", "");
#endif

    if (g_last_embedding_output.empty())
      return CAUSAL_LM_ERROR_INFERENCE_FAILED;

    *outputData = g_last_embedding_output[0];
    *outputDim = g_model->getDim();
    *outputLength = g_model->getBatchSize();

  } catch (const std::exception &e) {
    std::cerr << "Exception in runModelFloat: " << e.what() << std::endl;
    return CAUSAL_LM_ERROR_INFERENCE_FAILED;
  }

  return CAUSAL_LM_ERROR_NONE;
}

// ---------------------------------------------------------------------------
// Public API: getPerformanceMetrics
// ---------------------------------------------------------------------------
ErrorCode getPerformanceMetrics(PerformanceMetrics *metrics) {
  if (!g_initialized || !g_model)
    return CAUSAL_LM_ERROR_NOT_INITIALIZED;
  if (metrics == nullptr)
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;

  try {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_model->hasRun())
      return CAUSAL_LM_ERROR_INFERENCE_NOT_RUN;

    *metrics = g_model->getPerformanceMetrics();
    metrics->initialization_duration_ms = g_initialization_duration_ms;

  } catch (const std::exception &e) {
    std::cerr << "Exception in getPerformanceMetrics: " << e.what()
              << std::endl;
    return CAUSAL_LM_ERROR_UNKNOWN;
  }

  return CAUSAL_LM_ERROR_NONE;
}
