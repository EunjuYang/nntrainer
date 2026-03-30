/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
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
 *
 * @file	main.cpp
 * @date	23 July 2025
 * @brief	This is a main file for CausalLM application
 * @see		https://github.com/nnstreamer/
 * @author	Eunju Yang <ej.yang@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "json.hpp"
#include <app_context.h>
#include <factory.h>

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
#include <models/gemma3/function.h>
#include <cstdio>

#include <atomic>
#include <chrono>
#include <iomanip>
#include <thread>

using json = nlohmann::json;

/**
 * @brief Per-phase peak RSS tracker.
 *
 * Reads VmRSS from /proc/self/status (actual physical memory used).
 * Tracks peak independently for each phase so that weight-loading peak
 * does not mask inference-phase memory changes.
 */
struct PeakRSSTracker {
  std::atomic<size_t> peak_kb{0};
  std::atomic<bool> running{false};
  std::thread worker;

  static size_t current_rss_kb() {
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
      if (line.rfind("VmRSS:", 0) == 0) {
        size_t kb = 0;
        std::sscanf(line.c_str(), "VmRSS: %zu kB", &kb);
        return kb;
      }
    }
    return 0;
  }

  void start() {
    peak_kb.store(current_rss_kb());
    running.store(true);
    worker = std::thread([this] {
      while (running.load()) {
        size_t cur = current_rss_kb();
        size_t prev = peak_kb.load();
        if (cur > prev)
          peak_kb.store(cur);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      }
    });
  }

  size_t stop() {
    running.store(false);
    if (worker.joinable())
      worker.join();
    // final sample
    size_t cur = current_rss_kb();
    size_t prev = peak_kb.load();
    if (cur > prev)
      peak_kb.store(cur);
    return peak_kb.load();
  }

  void reset() { peak_kb.store(current_rss_kb()); }
};

std::string resolve_architecture(std::string model_type,
                                 const std::string &architecture) {
  std::transform(model_type.begin(), model_type.end(), model_type.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  if (model_type == "embedding") {
    if (architecture == "Qwen3ForCausalLM") {
      return "Qwen3Embedding";
    } else if (architecture == "Gemma3ForCausalLM" ||
               architecture == "Gemma3TextModel") {
      return "EmbeddingGemma";
    } else if (architecture == "Qwen2Model") {
      return "Qwen2Embedding";
    } else {
      throw std::invalid_argument(
        "Unsupported architecture for embedding model: " + architecture);
    }
  }

  return architecture;
}

int main(int argc, char *argv[]) {

  auto start_time = std::chrono::high_resolution_clock::now();

  /** Register all runnable causallm models to factory */
  causallm::Factory::Instance().registerModel(
    "LlamaForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::CausalLM>(cfg, generation_cfg,
                                                  nntr_cfg);
    });
  causallm::Factory::Instance().registerModel(
    "Qwen2ForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Qwen2CausalLM>(cfg, generation_cfg,
                                                       nntr_cfg);
    });
  causallm::Factory::Instance().registerModel(
    "Qwen2Embedding", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Qwen2Embedding>(cfg, generation_cfg,
                                                        nntr_cfg);
    });
  causallm::Factory::Instance().registerModel(
    "Qwen3ForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Qwen3CausalLM>(cfg, generation_cfg,
                                                       nntr_cfg);
    });
  causallm::Factory::Instance().registerModel(
    "Qwen3MoeForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Qwen3MoECausalLM>(cfg, generation_cfg,
                                                          nntr_cfg);
    });
  causallm::Factory::Instance().registerModel(
    "Qwen3SlimMoeForCausalLM",
    [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Qwen3SlimMoECausalLM>(
        cfg, generation_cfg, nntr_cfg);
    });
  causallm::Factory::Instance().registerModel(
    "Qwen3CachedSlimMoeForCausalLM",
    [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Qwen3CachedSlimMoECausalLM>(
        cfg, generation_cfg, nntr_cfg);
    });
  causallm::Factory::Instance().registerModel(
    "Qwen3Embedding", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Qwen3Embedding>(cfg, generation_cfg,
                                                        nntr_cfg);
    });
  causallm::Factory::Instance().registerModel(
    "GptOssForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::GptOssForCausalLM>(cfg, generation_cfg,
                                                           nntr_cfg);
    });
  causallm::Factory::Instance().registerModel(
    "GptOssCachedSlimCausalLM",
    [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::GptOssCachedSlimCausalLM>(
        cfg, generation_cfg, nntr_cfg);
    });
  causallm::Factory::Instance().registerModel(
    "Gemma3ForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Gemma3CausalLM>(cfg, generation_cfg,
                                                        nntr_cfg);
    });
  causallm::Factory::Instance().registerModel(
    "EmbeddingGemma", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::EmbeddingGemma>(cfg, generation_cfg,
                                                        nntr_cfg);
    });

  // Validate arguments
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <model_path> [input_prompt]\n"
              << "  <model_path>   : Path to model directory\n"
              << "  [input_prompt] : Optional input text (uses sample_input or "
                 "chat_input if omitted)\n";
    return EXIT_FAILURE;
  }

  const std::string model_path = argv[1];
  std::string input_text;
  std::string system_head_prompt = "";
  std::string system_tail_prompt = "";

  std::cout << model_path << std::endl;

  try {
    // Load configuration files
    json cfg = causallm::LoadJsonFile(model_path + "/config.json");
    json generation_cfg =
      causallm::LoadJsonFile(model_path + "/generation_config.json");
    json nntr_cfg = causallm::LoadJsonFile(model_path + "/nntr_config.json");

    if (nntr_cfg.contains("system_prompt")) {
      system_head_prompt =
        nntr_cfg["system_prompt"]["head_prompt"].get<std::string>();
      system_tail_prompt =
        nntr_cfg["system_prompt"]["tail_prompt"].get<std::string>();
    }

    // Construct weight file path
    const std::string weight_file =
      model_path + "/" + nntr_cfg["model_file_name"].get<std::string>();

    std::cout << weight_file << std::endl;

    // Initialize and run model
    std::string architecture =
      cfg["architectures"].get<std::vector<std::string>>()[0];

    if (nntr_cfg.contains("model_type")) {
      std::string model_type = nntr_cfg["model_type"].get<std::string>();
      architecture = resolve_architecture(model_type, architecture);
    }

    // Determine input text
    if (argc >= 3) {
      input_text = argv[2];
    } else {
      if (nntr_cfg.contains("chat_input")) {
        if (architecture == "Gemma3ForCausalLM") {
          input_text = causallm::gemma3::apply_function_gemma_template(
            nntr_cfg["chat_input"]);
        } else {
          std::cerr << "[Warning] 'chat_input' is set but support for model "
                       "architecture '"
                    << architecture
                    << "' is not implemented. Falling back to 'sample_input'."
                    << std::endl;
          input_text = nntr_cfg["sample_input"].get<std::string>();
        }
      } else {
        input_text = nntr_cfg["sample_input"].get<std::string>();
      }
    }

    PeakRSSTracker tracker;

    auto model = causallm::Factory::Instance().create(architecture, cfg,
                                                      generation_cfg, nntr_cfg);

    // --- Phase 1: initialize (pool calloc, lazy pages) ---
    tracker.start();
    model->initialize();
    size_t init_peak = tracker.stop();
    size_t init_rss = PeakRSSTracker::current_rss_kb();

    // --- Phase 2: load_weight (mmap + copy into weight pool) ---
    tracker.reset();
    tracker.start();
    model->load_weight(weight_file);
    size_t load_peak = tracker.stop();
    size_t load_rss = PeakRSSTracker::current_rss_kb();

    bool do_sample = generation_cfg.value("do_sample", false);

    // --- Phase 3: inference ---
    tracker.reset();
    tracker.start();
#if defined(_WIN32)
    model->run(input_text.c_str(), do_sample, system_head_prompt.c_str(),
               system_tail_prompt.c_str());
#else
    model->run(input_text, do_sample, system_head_prompt, system_tail_prompt);
#endif
    size_t run_peak = tracker.stop();
    size_t run_rss = PeakRSSTracker::current_rss_kb();

    auto finish_time = std::chrono::high_resolution_clock::now();
    auto e2e_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      finish_time - start_time);

    // --- Memory Report ---
    std::cout << "\n===== Memory Report (VmRSS) =====\n";
    std::cout << "Phase          | Peak (KB) | Peak (MB) | End (KB)  | End (MB)\n";
    std::cout << "---------------|-----------|-----------|-----------|--------\n";
    std::cout << "initialize     | " << std::setw(9) << init_peak
              << " | " << std::setw(9) << std::fixed << std::setprecision(1)
              << init_peak / 1024.0 << " | " << std::setw(9) << init_rss
              << " | " << std::setw(6) << init_rss / 1024.0 << "\n";
    std::cout << "load_weight    | " << std::setw(9) << load_peak
              << " | " << std::setw(9) << load_peak / 1024.0
              << " | " << std::setw(9) << load_rss
              << " | " << std::setw(6) << load_rss / 1024.0 << "\n";
    std::cout << "inference      | " << std::setw(9) << run_peak
              << " | " << std::setw(9) << run_peak / 1024.0
              << " | " << std::setw(9) << run_rss
              << " | " << std::setw(6) << run_rss / 1024.0 << "\n";
    std::cout << "=================================\n";
    std::cout << "[e2e time]: " << e2e_duration.count() << " ms \n";

  } catch (const std::exception &e) {
    std::cerr << "\n[!] FATAL ERROR: " << e.what() << "\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
