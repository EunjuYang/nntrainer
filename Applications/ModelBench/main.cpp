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
 * @file	main.cpp
 * @date	09 March 2026
 * @brief	Model benchmark application comparable to llama-bench.
 *          Measures pure inference throughput (pp and tg) using random
 *          token IDs, excluding tokenization and sampling overhead.
 *          Like llama-bench, this uses random token IDs directly -
 *          no chat template, no tokenization, no sampling overhead.
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Auto-generated
 * @bug		No known bugs except for NYI items
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "json.hpp"
#include <app_context.h>
#include <factory.h>

#include "causal_lm.h"
#include "gemma3_causallm.h"
#include "gptoss_cached_slim_causallm.h"
#include "gptoss_causallm.h"
#include "qwen2_causallm.h"
#include "qwen3_cached_slim_moe_causallm.h"
#include "qwen3_causallm.h"
#include "qwen3_moe_causallm.h"
#include "qwen3_slim_moe_causallm.h"

using json = nlohmann::json;

struct BenchParams {
  std::string model_path;
  std::vector<unsigned int> n_prompt = {512};
  std::vector<unsigned int> n_gen = {128};
  unsigned int repetitions = 5;
  bool run_pp = true;
  bool run_tg = true;
};

struct BenchResult {
  std::string test_type; // "pp" or "tg"
  unsigned int n_tokens;
  unsigned int n_prompt; // context: how many prompt tokens for this tg test
  double avg_tokens_per_sec;
  double stddev_tokens_per_sec;
  std::vector<double> samples; // tokens/sec per repetition
};

static void print_usage(const char *prog) {
  std::cout
    << "Usage: " << prog << " <model_path> [options]\n"
    << "\n"
    << "Model benchmark tool (comparable to llama-bench)\n"
    << "Measures pure model inference throughput using random token IDs.\n"
    << "No tokenization, no sampling, no chat template overhead.\n"
    << "\n"
    << "Options:\n"
    << "  -p, --n-prompt N[,N,...] Number of prompt tokens for pp test "
       "(default: 512)\n"
    << "  -n, --n-gen N[,N,...]    Number of tokens for tg test (default: "
       "128)\n"
    << "  -r, --repetitions N      Number of repetitions (default: 5)\n"
    << "  --no-pp                  Disable prompt processing test\n"
    << "  --no-tg                  Disable token generation test\n"
    << "  -h, --help               Show this help message\n"
    << "\n"
    << "Examples:\n"
    << "  " << prog << " /path/to/model\n"
    << "  " << prog << " /path/to/model -p 128,256,512 -n 64,128\n"
    << "  " << prog << " /path/to/model -p 1024 -n 256 -r 3\n";
}

static std::vector<unsigned int> parse_comma_list(const std::string &s) {
  std::vector<unsigned int> result;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    result.push_back(static_cast<unsigned int>(std::stoul(item)));
  }
  return result;
}

static BenchParams parse_args(int argc, char *argv[]) {
  BenchParams params;

  if (argc < 2) {
    print_usage(argv[0]);
    exit(EXIT_FAILURE);
  }

  params.model_path = argv[1];

  if (params.model_path == "-h" || params.model_path == "--help") {
    print_usage(argv[0]);
    exit(EXIT_SUCCESS);
  }

  for (int i = 2; i < argc; i++) {
    std::string arg = argv[i];

    if ((arg == "-p" || arg == "--n-prompt") && i + 1 < argc) {
      params.n_prompt = parse_comma_list(argv[++i]);
    } else if ((arg == "-n" || arg == "--n-gen") && i + 1 < argc) {
      params.n_gen = parse_comma_list(argv[++i]);
    } else if ((arg == "-r" || arg == "--repetitions") && i + 1 < argc) {
      params.repetitions = static_cast<unsigned int>(std::stoul(argv[++i]));
    } else if (arg == "--no-pp") {
      params.run_pp = false;
    } else if (arg == "--no-tg") {
      params.run_tg = false;
    } else if (arg == "-h" || arg == "--help") {
      print_usage(argv[0]);
      exit(EXIT_SUCCESS);
    } else {
      std::cerr << "Unknown option: " << arg << "\n";
      print_usage(argv[0]);
      exit(EXIT_FAILURE);
    }
  }

  return params;
}

static void print_separator() {
  std::cout << std::string(90, '-') << "\n";
}

static void print_results(const std::vector<BenchResult> &results,
                           const std::string &model_path) {
  std::cout << "\n";
  print_separator();
  std::cout << "model: " << model_path << "\n";
  print_separator();
  std::cout << std::left << std::setw(8) << "test" << std::setw(10)
            << "tokens" << std::setw(12) << "pp_ctx" << std::setw(15)
            << "avg t/s" << std::setw(15) << "stddev t/s" << "samples (t/s)\n";
  print_separator();

  for (const auto &r : results) {
    std::cout << std::left << std::setw(8) << r.test_type << std::setw(10)
              << r.n_tokens;

    if (r.test_type == "tg") {
      std::cout << std::setw(12) << r.n_prompt;
    } else {
      std::cout << std::setw(12) << "-";
    }

    std::cout << std::fixed << std::setprecision(2) << std::setw(15)
              << r.avg_tokens_per_sec << std::setw(15)
              << r.stddev_tokens_per_sec;

    std::cout << "[";
    for (size_t i = 0; i < r.samples.size(); i++) {
      if (i > 0)
        std::cout << ", ";
      std::cout << std::fixed << std::setprecision(2) << r.samples[i];
    }
    std::cout << "]\n";
  }
  print_separator();
}

/**
 * @brief Register all CausalLM model types with the factory.
 */
static void register_models() {
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
}

int main(int argc, char *argv[]) {

  BenchParams params = parse_args(argc, argv);
  register_models();

  std::cout << "nntrainer model benchmark tool\n";
  std::cout << "Comparable to llama-bench: uses random token IDs, "
            << "measures pure inference throughput.\n";
  std::cout << "No tokenization, no sampling, no chat template overhead.\n\n";

  try {
    // Load configuration
    json cfg = causallm::LoadJsonFile(params.model_path + "/config.json");
    json generation_cfg =
      causallm::LoadJsonFile(params.model_path + "/generation_config.json");
    json nntr_cfg =
      causallm::LoadJsonFile(params.model_path + "/nntr_config.json");

    const unsigned int num_vocab = cfg["vocab_size"].get<unsigned int>();
    const std::string architecture =
      cfg["architectures"].get<std::vector<std::string>>()[0];

    std::cout << "model: " << params.model_path << "\n";
    std::cout << "architecture: " << architecture << "\n";
    std::cout << "vocab_size: " << num_vocab << "\n";
    std::cout << "repetitions: " << params.repetitions << "\n";

    if (params.run_pp) {
      std::cout << "pp tokens: ";
      for (size_t i = 0; i < params.n_prompt.size(); i++) {
        if (i > 0)
          std::cout << ",";
        std::cout << params.n_prompt[i];
      }
      std::cout << "\n";
    }
    if (params.run_tg) {
      std::cout << "tg tokens: ";
      for (size_t i = 0; i < params.n_gen.size(); i++) {
        if (i > 0)
          std::cout << ",";
        std::cout << params.n_gen[i];
      }
      std::cout << "\n";
    }

    std::vector<BenchResult> all_results;

    // Fixed seed random number generator for reproducibility (like llama-bench)
    std::mt19937 rng(42);

    // Run all test combinations
    for (unsigned int pp_tokens : params.n_prompt) {
      for (unsigned int tg_tokens : params.n_gen) {

        unsigned int total_tokens = pp_tokens + tg_tokens;
        unsigned int orig_max_seq_len =
          nntr_cfg["max_seq_len"].get<unsigned int>();

        if (total_tokens > orig_max_seq_len) {
          std::cerr << "Warning: pp(" << pp_tokens << ") + tg(" << tg_tokens
                    << ") = " << total_tokens << " exceeds max_seq_len("
                    << orig_max_seq_len
                    << "). Adjusting max_seq_len for this test.\n";
        }

        // Override nntr_config for benchmark purposes
        json bench_nntr_cfg = nntr_cfg;
        bench_nntr_cfg["init_seq_len"] = pp_tokens;
        bench_nntr_cfg["max_seq_len"] =
          std::max(total_tokens, orig_max_seq_len);
        bench_nntr_cfg["num_to_generate"] = tg_tokens;
        bench_nntr_cfg["batch_size"] = 1;

        const std::string weight_file =
          params.model_path + "/" +
          bench_nntr_cfg["model_file_name"].get<std::string>();

        std::cout << "\n=== Initializing model for pp=" << pp_tokens
                  << " tg=" << tg_tokens << " ===\n";

        // Create and initialize model
        // Note: init_seq_len affects model graph structure (input shape),
        // so we need to re-create the model for different pp_tokens values
        auto model_ptr = causallm::Factory::Instance().create(
          architecture, cfg, generation_cfg, bench_nntr_cfg);
        model_ptr->initialize();
        model_ptr->load_weight(weight_file);

        // Generate random token IDs (like llama-bench: rand() % n_vocab)
        std::vector<float> input_tokens(pp_tokens);
        std::uniform_int_distribution<unsigned int> dist(0, num_vocab - 1);
        for (unsigned int i = 0; i < pp_tokens; i++) {
          input_tokens[i] = static_cast<float>(dist(rng));
        }

        // Prepare input buffer (padded to max_seq_len)
        unsigned int buf_max_seq =
          bench_nntr_cfg["max_seq_len"].get<unsigned int>();
        std::vector<float> input_buf(buf_max_seq, 0.0f);
        std::vector<float *> input = {input_buf.data()};
        std::vector<float *> label;

        // --- Prompt Processing (pp) benchmark ---
        if (params.run_pp) {
          BenchResult pp_result;
          pp_result.test_type = "pp";
          pp_result.n_tokens = pp_tokens;
          pp_result.n_prompt = pp_tokens;

          std::cout << "Running pp benchmark (n=" << pp_tokens << ")...\n";

          for (unsigned int rep = 0; rep < params.repetitions; rep++) {
            for (unsigned int i = 0; i < pp_tokens; i++) {
              input_buf[i] = input_tokens[i];
            }

            auto start = std::chrono::high_resolution_clock::now();

            // Prefill: process all prompt tokens at once (batch decode)
            model_ptr->run_prefill(input, label, pp_tokens, 0, pp_tokens);

            auto end = std::chrono::high_resolution_clock::now();
            double elapsed_ms =
              std::chrono::duration<double, std::milli>(end - start).count();
            double tps = (pp_tokens / elapsed_ms) * 1000.0;
            pp_result.samples.push_back(tps);

            std::cout << "  rep " << (rep + 1) << "/" << params.repetitions
                      << ": " << std::fixed << std::setprecision(2) << tps
                      << " t/s (" << elapsed_ms << " ms)\n";
          }

          // Compute avg and stddev
          double sum = std::accumulate(pp_result.samples.begin(),
                                       pp_result.samples.end(), 0.0);
          pp_result.avg_tokens_per_sec = sum / pp_result.samples.size();

          double sq_sum = 0.0;
          for (double s : pp_result.samples) {
            sq_sum += (s - pp_result.avg_tokens_per_sec) *
                      (s - pp_result.avg_tokens_per_sec);
          }
          pp_result.stddev_tokens_per_sec =
            std::sqrt(sq_sum / pp_result.samples.size());

          all_results.push_back(pp_result);
        }

        // --- Token Generation (tg) benchmark ---
        if (params.run_tg) {
          BenchResult tg_result;
          tg_result.test_type = "tg";
          tg_result.n_tokens = tg_tokens;
          tg_result.n_prompt = pp_tokens;

          std::cout << "Running tg benchmark (n=" << tg_tokens
                    << ", pp_ctx=" << pp_tokens << ")...\n";

          for (unsigned int rep = 0; rep < params.repetitions; rep++) {

            // First, do prefill to fill KV cache (not timed)
            for (unsigned int i = 0; i < pp_tokens; i++) {
              input_buf[i] = input_tokens[i];
            }
            auto prefill_output =
              model_ptr->run_prefill(input, label, pp_tokens, 0, pp_tokens);

            // Argmax on prefill output to get first generation token
            float *logits = prefill_output[0];
            unsigned int next_token = 0;
            float max_logit = logits[0];
            for (unsigned int v = 1; v < num_vocab; v++) {
              if (logits[v] > max_logit) {
                max_logit = logits[v];
                next_token = v;
              }
            }

            // Measure token generation only (single token steps)
            auto start = std::chrono::high_resolution_clock::now();

            for (unsigned int t = 0; t < tg_tokens; t++) {
              input_buf[0] = static_cast<float>(next_token);

              unsigned int pos = pp_tokens + t;
              auto gen_output = model_ptr->run_generation(
                input, label, pp_tokens, pos, pos + 1);

              // Argmax only (no sampling - same as llama-bench)
              logits = gen_output[0];
              next_token = 0;
              max_logit = logits[0];
              for (unsigned int v = 1; v < num_vocab; v++) {
                if (logits[v] > max_logit) {
                  max_logit = logits[v];
                  next_token = v;
                }
              }
            }

            auto end = std::chrono::high_resolution_clock::now();
            double elapsed_ms =
              std::chrono::duration<double, std::milli>(end - start).count();
            double tps = (tg_tokens / elapsed_ms) * 1000.0;
            tg_result.samples.push_back(tps);

            std::cout << "  rep " << (rep + 1) << "/" << params.repetitions
                      << ": " << std::fixed << std::setprecision(2) << tps
                      << " t/s (" << elapsed_ms << " ms)\n";
          }

          // Compute avg and stddev
          double sum = std::accumulate(tg_result.samples.begin(),
                                       tg_result.samples.end(), 0.0);
          tg_result.avg_tokens_per_sec = sum / tg_result.samples.size();

          double sq_sum = 0.0;
          for (double s : tg_result.samples) {
            sq_sum += (s - tg_result.avg_tokens_per_sec) *
                      (s - tg_result.avg_tokens_per_sec);
          }
          tg_result.stddev_tokens_per_sec =
            std::sqrt(sq_sum / tg_result.samples.size());

          all_results.push_back(tg_result);
        }
      }
    }

    // Print final summary
    print_results(all_results, params.model_path);

  } catch (const std::exception &e) {
    std::cerr << "\n[!] FATAL ERROR: " << e.what() << "\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
