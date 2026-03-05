// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @file   quantize.cpp
 * @date   04 March 2026
 * @brief  Quantization utility for CausalLM models.
 *         Reads a FP32 model and converts weights to a target data type,
 *         saving both the quantized .bin file and a new nntr_config.json.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * @usage
 *   nntr_quantize <model_path> [options]
 *
 *   Required:
 *     <model_path>        Path to the model directory containing:
 *                           config.json, generation_config.json,
 *                           nntr_config.json, and the .bin weight file.
 *
 *   Options:
 *     --model_tensor_type <W-A>  Target model tensor type (default: Q4_0-FP32)
 *                                This determines the default dtype for all
 *                                FC (weight-bearing) layers.
 *                                Format: <weight_dtype>-<activation_dtype>
 *     --embedding_dtype <type>   Override dtype for embedding (default: FP32)
 *     --lmhead_dtype <type>      Override dtype for LM head (default: FP32)
 *     --output, -o <path>        Output directory (default: <model_path>)
 *     --output_bin <name>        Output bin filename (auto-generated if omitted)
 *     --config <path>            Use a target nntr_config.json directly
 *
 *   Supported model_tensor_type values:
 *     FP32-FP32, FP16-FP32, FP16-FP16, Q4_0-FP32, Q4_0-FP16, Q4_K-FP32
 *
 *   Supported per-layer dtype values: FP32, FP16, Q4_0, Q6_K, Q4_K
 *
 *   Examples:
 *     # Quantize to Q4_0 weights with FP32 activations (default):
 *     nntr_quantize /path/to/qwen3-4b
 *
 *     # Quantize with specific model_tensor_type:
 *     nntr_quantize /path/to/qwen3-4b --model_tensor_type Q4_0-FP16
 *
 *     # Override embedding to stay FP32 while FC uses Q4_0:
 *     nntr_quantize /path/to/qwen3-4b --model_tensor_type Q4_0-FP32 \
 *                                      --embedding_dtype FP32
 *
 *     # Use a target nntr_config.json:
 *     nntr_quantize /path/to/qwen3-4b --config /path/to/target_nntr_config.json
 */

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "json.hpp"
#include <app_context.h>
#include <factory.h>
#include <tensor_dim.h>

#include "causal_lm.h"
#include "embedding_gemma.h"
#include "gemma3_causallm.h"
#include "gptoss_cached_slim_causallm.h"
#include "gptoss_causallm.h"
#include "qwen2_causallm.h"
#include "qwen2_embedding.h"
#include "qwen3_cached_slim_moe_causallm.h"
#include "qwen3_causallm.h"
#include "qwen3_embedding.h"
#include "qwen3_moe_causallm.h"
#include "qwen3_slim_moe_causallm.h"

using json = nlohmann::json;
using DataType = ml::train::TensorDim::DataType;

namespace {

/**
 * @brief Map of string data type names to DataType enum values
 */
const std::map<std::string, DataType> dtype_str_map = {
  {"FP32", DataType::FP32}, {"FP16", DataType::FP16},
  {"Q4_0", DataType::Q4_0}, {"Q6_K", DataType::Q6_K},
  {"Q4_K", DataType::Q4_K}, {"NONE", DataType::NONE},
};

/**
 * @brief Convert string to DataType enum
 */
DataType strToDataType(const std::string &s) {
  std::string upper = s;
  std::transform(upper.begin(), upper.end(), upper.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  auto it = dtype_str_map.find(upper);
  if (it == dtype_str_map.end()) {
    throw std::invalid_argument("Unsupported data type: " + s +
                                ". Supported: FP32, FP16, Q4_0, Q6_K, Q4_K");
  }
  return it->second;
}

/**
 * @brief Convert DataType enum to string
 */
std::string dataTypeToStr(DataType dt) {
  for (const auto &[key, val] : dtype_str_map) {
    if (val == dt)
      return key;
  }
  return "NONE";
}

/**
 * @brief Parse model_tensor_type string "W-A" into weight and activation dtypes
 *        e.g. "Q4_0-FP32" -> (Q4_0, FP32)
 */
std::pair<DataType, DataType>
parseModelTensorType(const std::string &tensor_type) {
  auto dash_pos = tensor_type.find('-');
  if (dash_pos == std::string::npos) {
    throw std::invalid_argument(
      "Invalid model_tensor_type format: '" + tensor_type +
      "'. Expected format: <weight_dtype>-<activation_dtype> (e.g. Q4_0-FP32)");
  }
  std::string w_str = tensor_type.substr(0, dash_pos);
  std::string a_str = tensor_type.substr(dash_pos + 1);
  return {strToDataType(w_str), strToDataType(a_str)};
}

/**
 * @brief Generate a descriptive output bin filename from model_tensor_type
 */
std::string generateOutputBinName(const std::string &original_bin,
                                  const std::string &model_tensor_type) {
  std::string base = original_bin;
  // Remove .bin extension
  auto dot_pos = base.rfind(".bin");
  if (dot_pos != std::string::npos)
    base = base.substr(0, dot_pos);

  // Remove old dtype suffix patterns
  std::vector<std::string> dtype_suffixes = {
    "_fp32",      "_fp16",    "_q40",      "_q4_0",
    "_q6k",       "_q6_k",   "_q4k",      "_q4_k",
    "_q40-fp32",  "_q40-fp16", "_fp32-fp32"};
  for (const auto &suffix : dtype_suffixes) {
    auto pos = base.rfind(suffix);
    if (pos != std::string::npos && pos + suffix.size() == base.size()) {
      base = base.substr(0, pos);
      break;
    }
  }

  // Build new suffix from model_tensor_type (e.g. "Q4_0-FP32" -> "q40-fp32")
  std::string suffix = model_tensor_type;
  std::transform(suffix.begin(), suffix.end(), suffix.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  // Remove underscores for cleaner naming
  suffix.erase(std::remove(suffix.begin(), suffix.end(), '_'), suffix.end());

  return base + "_" + suffix + ".bin";
}

/**
 * @brief Resolve architecture name from config
 */
std::string resolve_architecture(std::string model_type,
                                 const std::string &architecture) {
  std::transform(model_type.begin(), model_type.end(), model_type.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  if (model_type == "embedding") {
    if (architecture == "Qwen3ForCausalLM")
      return "Qwen3Embedding";
    else if (architecture == "Gemma3ForCausalLM" ||
             architecture == "Gemma3TextModel")
      return "EmbeddingGemma";
    else if (architecture == "Qwen2Model")
      return "Qwen2Embedding";
    else
      throw std::invalid_argument(
        "Unsupported architecture for embedding model: " + architecture);
  }
  return architecture;
}

/**
 * @brief Register all CausalLM model factories
 */
void registerAllModels() {
  auto &factory = causallm::Factory::Instance();

  factory.registerModel(
    "LlamaForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::CausalLM>(cfg, generation_cfg,
                                                   nntr_cfg);
    });
  factory.registerModel(
    "Qwen2ForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Qwen2CausalLM>(cfg, generation_cfg,
                                                        nntr_cfg);
    });
  factory.registerModel(
    "Qwen2Embedding", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Qwen2Embedding>(cfg, generation_cfg,
                                                         nntr_cfg);
    });
  factory.registerModel(
    "Qwen3ForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Qwen3CausalLM>(cfg, generation_cfg,
                                                        nntr_cfg);
    });
  factory.registerModel(
    "Qwen3MoeForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Qwen3MoECausalLM>(cfg, generation_cfg,
                                                           nntr_cfg);
    });
  factory.registerModel(
    "Qwen3SlimMoeForCausalLM",
    [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Qwen3SlimMoECausalLM>(
        cfg, generation_cfg, nntr_cfg);
    });
  factory.registerModel(
    "Qwen3CachedSlimMoeForCausalLM",
    [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Qwen3CachedSlimMoECausalLM>(
        cfg, generation_cfg, nntr_cfg);
    });
  factory.registerModel(
    "Qwen3Embedding", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Qwen3Embedding>(cfg, generation_cfg,
                                                         nntr_cfg);
    });
  factory.registerModel(
    "GptOssForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::GptOssForCausalLM>(cfg, generation_cfg,
                                                            nntr_cfg);
    });
  factory.registerModel(
    "GptOssCachedSlimCausalLM",
    [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::GptOssCachedSlimCausalLM>(
        cfg, generation_cfg, nntr_cfg);
    });
  factory.registerModel(
    "Gemma3ForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Gemma3CausalLM>(cfg, generation_cfg,
                                                         nntr_cfg);
    });
  factory.registerModel(
    "EmbeddingGemma", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::EmbeddingGemma>(cfg, generation_cfg,
                                                         nntr_cfg);
    });
}

/**
 * @brief Print usage information
 */
void printUsage(const char *prog) {
  std::cout
    << "Usage: " << prog << " <model_path> [options]\n"
    << "\n"
    << "Quantize a CausalLM model from FP32 to a target data type.\n"
    << "\n"
    << "Required:\n"
    << "  <model_path>                 Path to model directory containing:\n"
    << "                                 config.json, generation_config.json,\n"
    << "                                 nntr_config.json, and .bin weight file\n"
    << "\n"
    << "Options:\n"
    << "  --model_tensor_type <W-A>    Target tensor type (default: Q4_0-FP32)\n"
    << "                               Determines default dtype for all FC layers.\n"
    << "                               Format: <weight_dtype>-<activation_dtype>\n"
    << "  --embedding_dtype <type>     Override dtype for embedding (default: FP32)\n"
    << "  --lmhead_dtype <type>        Override dtype for LM head (default: FP32)\n"
    << "  --output, -o <path>          Output directory (default: <model_path>)\n"
    << "  --output_bin <name>          Output .bin filename (auto-generated if omitted)\n"
    << "  --config <path>              Use a target nntr_config.json directly.\n"
    << "                               Reads model_tensor_type, embedding_dtype,\n"
    << "                               lmhead_dtype from the target config.\n"
    << "  --help, -h                   Show this help message\n"
    << "\n"
    << "Supported model_tensor_type values:\n"
    << "  FP32-FP32, FP16-FP32, FP16-FP16, Q4_0-FP32, Q4_0-FP16, Q4_K-FP32\n"
    << "\n"
    << "Supported per-layer dtype values: FP32, FP16, Q4_0, Q6_K, Q4_K\n"
    << "\n"
    << "Examples:\n"
    << "  # Quantize to Q4_0 weights / FP32 activations (default):\n"
    << "  " << prog << " /path/to/qwen3-4b\n"
    << "\n"
    << "  # Quantize with Q4_0-FP16:\n"
    << "  " << prog << " /path/to/qwen3-4b --model_tensor_type Q4_0-FP16\n"
    << "\n"
    << "  # Q4_0-FP32 but keep embedding as FP32:\n"
    << "  " << prog << " /path/to/qwen3-4b --embedding_dtype FP32\n"
    << "\n"
    << "  # Use a target nntr_config.json:\n"
    << "  " << prog
    << " /path/to/qwen3-4b --config /path/to/target_nntr_config.json\n";
}

/**
 * @brief Build the layer_dtype_map for the model based on target dtypes.
 *
 * The model_tensor_type (e.g. "Q4_0-FP32") determines the default weight dtype
 * for all FC layers via nntrainer's model property. However, for save-with-dtype
 * we need to explicitly map each layer.
 *
 * Layer naming convention in Transformer:
 *   - embedding0              : embedding layer
 *   - layer{i}_wq/wk/wv      : attention Q/K/V projections (FC layers)
 *   - layer{i}_attention_out  : attention output projection (FC layer)
 *   - layer{i}_ffn_up/gate/down : FFN layers (FC layers)
 *   - output_of_causallm      : LM head (FC layer)
 *   - RMSNorm / other layers  : not quantized (no weights or small)
 *
 * The dtype map assigns:
 *   - FC layers               -> weight_dtype (from model_tensor_type)
 *   - embedding0              -> embedding_dtype (override)
 *   - output_of_causallm      -> lmhead_dtype (override)
 */
std::map<std::string, DataType>
buildLayerDtypeMap(int num_layers, DataType weight_dtype,
                   DataType embd_dtype, DataType lmhead_dtype) {

  std::map<std::string, DataType> dtype_map;

  // Embedding layer - override from default weight_dtype
  dtype_map["embedding0"] = embd_dtype;

  // Transformer decoder layers - all FC layers get weight_dtype
  for (int i = 0; i < num_layers; ++i) {
    std::string prefix = "layer" + std::to_string(i);

    // Attention FC layers
    dtype_map[prefix + "_wq"] = weight_dtype;
    dtype_map[prefix + "_wk"] = weight_dtype;
    dtype_map[prefix + "_wv"] = weight_dtype;
    dtype_map[prefix + "_attention_out"] = weight_dtype;

    // FFN FC layers
    dtype_map[prefix + "_ffn_up"] = weight_dtype;
    dtype_map[prefix + "_ffn_gate"] = weight_dtype;
    dtype_map[prefix + "_ffn_down"] = weight_dtype;
  }

  // LM Head layer - override from default weight_dtype
  dtype_map["output_of_causallm"] = lmhead_dtype;

  return dtype_map;
}

} // anonymous namespace

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printUsage(argv[0]);
    return EXIT_FAILURE;
  }

  std::string first_arg = argv[1];
  if (first_arg == "--help" || first_arg == "-h") {
    printUsage(argv[0]);
    return EXIT_SUCCESS;
  }

  // Parse arguments
  std::string model_path = argv[1];
  std::string output_dir = "";
  std::string model_tensor_type_str = "Q4_0-FP32";
  std::string embd_dtype_str = "FP32";
  std::string lmhead_dtype_str = "FP32";
  std::string output_bin_name = "";
  std::string target_config_path = "";

  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    if ((arg == "--output" || arg == "-o") && i + 1 < argc) {
      output_dir = argv[++i];
    } else if (arg == "--model_tensor_type" && i + 1 < argc) {
      model_tensor_type_str = argv[++i];
    } else if (arg == "--embedding_dtype" && i + 1 < argc) {
      embd_dtype_str = argv[++i];
    } else if (arg == "--lmhead_dtype" && i + 1 < argc) {
      lmhead_dtype_str = argv[++i];
    } else if (arg == "--output_bin" && i + 1 < argc) {
      output_bin_name = argv[++i];
    } else if (arg == "--config" && i + 1 < argc) {
      target_config_path = argv[++i];
    } else if (arg == "--help" || arg == "-h") {
      printUsage(argv[0]);
      return EXIT_SUCCESS;
    } else {
      std::cerr << "Unknown option: " << arg << "\n";
      printUsage(argv[0]);
      return EXIT_FAILURE;
    }
  }

  try {
    // =========================================================================
    // Step 1: Load source configurations
    // =========================================================================
    std::cout << "==========================================================\n";
    std::cout << "  NNTrainer CausalLM Quantization Utility\n";
    std::cout << "==========================================================\n";
    std::cout << "[1/5] Loading configurations from: " << model_path << "\n";

    json cfg = causallm::LoadJsonFile(model_path + "/config.json");
    json generation_cfg =
      causallm::LoadJsonFile(model_path + "/generation_config.json");
    json nntr_cfg = causallm::LoadJsonFile(model_path + "/nntr_config.json");

    // If a target config is specified, read dtypes from it
    if (!target_config_path.empty()) {
      std::cout << "  Using target config: " << target_config_path << "\n";
      json target_cfg = causallm::LoadJsonFile(target_config_path);
      if (target_cfg.contains("model_tensor_type") &&
          !target_cfg["model_tensor_type"].is_null())
        model_tensor_type_str =
          target_cfg["model_tensor_type"].get<std::string>();
      if (target_cfg.contains("embedding_dtype") &&
          !target_cfg["embedding_dtype"].is_null())
        embd_dtype_str = target_cfg["embedding_dtype"].get<std::string>();
      if (target_cfg.contains("lmhead_dtype") &&
          !target_cfg["lmhead_dtype"].is_null())
        lmhead_dtype_str = target_cfg["lmhead_dtype"].get<std::string>();
      if (target_cfg.contains("model_file_name") &&
          !target_cfg["model_file_name"].is_null() && output_bin_name.empty())
        output_bin_name = target_cfg["model_file_name"].get<std::string>();
    }

    // Parse model_tensor_type -> (weight_dtype, activation_dtype)
    auto [weight_dtype, activation_dtype] =
      parseModelTensorType(model_tensor_type_str);
    DataType embd_dtype = strToDataType(embd_dtype_str);
    DataType lmhead_dtype = strToDataType(lmhead_dtype_str);

    // Validate source model is FP32
    std::string src_tensor_type = "FP32-FP32";
    if (nntr_cfg.contains("model_tensor_type") &&
        !nntr_cfg["model_tensor_type"].is_null()) {
      src_tensor_type = nntr_cfg["model_tensor_type"].get<std::string>();
    }
    if (src_tensor_type != "FP32-FP32") {
      std::cerr << "[WARNING] Source model_tensor_type is '" << src_tensor_type
                << "', not 'FP32-FP32'.\n"
                << "  Quantization from non-FP32 models may produce unexpected "
                   "results.\n";
    }

    // Setup output directory
    if (output_dir.empty())
      output_dir = model_path;
    std::filesystem::create_directories(output_dir);

    // Determine output bin filename
    std::string original_bin =
      nntr_cfg["model_file_name"].get<std::string>();
    if (output_bin_name.empty()) {
      output_bin_name =
        generateOutputBinName(original_bin, model_tensor_type_str);
    }

    std::string src_weight_path = model_path + "/" + original_bin;
    std::string dst_weight_path = output_dir + "/" + output_bin_name;

    int num_layers = cfg["num_hidden_layers"].get<int>();

    std::cout << "  Architecture:      "
              << cfg["architectures"].get<std::vector<std::string>>()[0]
              << "\n";
    std::cout << "  Num layers:        " << num_layers << "\n";
    std::cout << "  Source:            " << src_weight_path << "\n";
    std::cout << "  Target:            " << dst_weight_path << "\n";
    std::cout << "  model_tensor_type: " << model_tensor_type_str << "\n";
    std::cout << "    weight dtype:    " << dataTypeToStr(weight_dtype) << "\n";
    std::cout << "    activation dtype:" << dataTypeToStr(activation_dtype)
              << "\n";
    std::cout << "  embedding_dtype:   " << dataTypeToStr(embd_dtype) << "\n";
    std::cout << "  lmhead_dtype:      " << dataTypeToStr(lmhead_dtype) << "\n";
    std::cout << "\n";

    // =========================================================================
    // Step 2: Register models & create model instance
    // =========================================================================
    std::cout << "[2/5] Creating and initializing model...\n";

    registerAllModels();

    std::string architecture =
      cfg["architectures"].get<std::vector<std::string>>()[0];
    if (nntr_cfg.contains("model_type") &&
        !nntr_cfg["model_type"].is_null()) {
      std::string model_type = nntr_cfg["model_type"].get<std::string>();
      architecture = resolve_architecture(model_type, architecture);
    }

    auto model = causallm::Factory::Instance().create(architecture, cfg,
                                                       generation_cfg, nntr_cfg);
    if (!model) {
      throw std::runtime_error("Failed to create model for architecture: " +
                               architecture);
    }

    model->initialize();
    std::cout << "  Model initialized successfully.\n";

    // =========================================================================
    // Step 3: Load FP32 weights
    // =========================================================================
    std::cout << "[3/5] Loading FP32 weights from: " << src_weight_path << "\n";
    model->load_weight(src_weight_path);
    std::cout << "  Weights loaded successfully.\n";

    // =========================================================================
    // Step 4: Build layer dtype map and save quantized weights
    // =========================================================================
    std::cout << "[4/5] Quantizing and saving weights to: " << dst_weight_path
              << "\n";

    auto layer_dtype_map =
      buildLayerDtypeMap(num_layers, weight_dtype, embd_dtype, lmhead_dtype);

    std::cout << "  Layer dtype mapping (" << layer_dtype_map.size()
              << " layers):\n";
    for (const auto &[name, dt] : layer_dtype_map) {
      std::cout << "    " << name << " -> " << dataTypeToStr(dt) << "\n";
    }

    model->save_weight(dst_weight_path, DataType::NONE, layer_dtype_map);

    // Report file size
    auto src_size = std::filesystem::file_size(src_weight_path);
    auto dst_size = std::filesystem::file_size(dst_weight_path);
    double ratio = static_cast<double>(dst_size) / src_size * 100.0;

    std::cout << "  Source size:  " << (src_size / (1024 * 1024)) << " MB\n";
    std::cout << "  Output size:  " << (dst_size / (1024 * 1024)) << " MB\n";
    std::cout << "  Compression:  " << std::fixed << std::setprecision(1)
              << ratio << "%\n";

    // =========================================================================
    // Step 5: Generate new nntr_config.json
    // =========================================================================
    std::cout << "[5/5] Generating nntr_config.json...\n";

    json new_nntr_cfg = nntr_cfg;
    new_nntr_cfg["model_file_name"] = output_bin_name;
    new_nntr_cfg["model_tensor_type"] = model_tensor_type_str;
    new_nntr_cfg["embedding_dtype"] = dataTypeToStr(embd_dtype);
    new_nntr_cfg["lmhead_dtype"] = dataTypeToStr(lmhead_dtype);

    std::string output_config_path = output_dir + "/nntr_config.json";

    // If output is same dir and we'd overwrite, save as
    // nntr_config_quantized.json
    if (output_dir == model_path) {
      output_config_path = output_dir + "/nntr_config_quantized.json";
    }

    std::ofstream config_out(output_config_path);
    if (!config_out.is_open()) {
      throw std::runtime_error("Failed to open output config: " +
                                output_config_path);
    }
    config_out << new_nntr_cfg.dump(4) << std::endl;
    config_out.close();

    std::cout << "  Config saved to: " << output_config_path << "\n";

    // =========================================================================
    // Done
    // =========================================================================
    std::cout << "\n";
    std::cout << "==========================================================\n";
    std::cout << "  Quantization complete!\n";
    std::cout << "==========================================================\n";
    std::cout << "\n";
    std::cout << "To run the quantized model:\n";
    if (output_dir == model_path) {
      std::cout << "  1. Rename nntr_config_quantized.json to "
                   "nntr_config.json\n";
      std::cout << "  2. nntr_causallm " << model_path << "\n";
    } else {
      std::cout << "  1. Copy config.json and generation_config.json to "
                << output_dir << "\n";
      std::cout << "  2. nntr_causallm " << output_dir << "\n";
    }

  } catch (const std::exception &e) {
    std::cerr << "\n[!] FATAL ERROR: " << e.what() << "\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
