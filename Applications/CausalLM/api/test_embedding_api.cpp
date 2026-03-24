// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   test_embedding_api.cpp
 * @date   23 Mar 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @brief  Test application for CausalLM Embedding API (runModelFloat)
 * @bug    No known bugs except for NYI items
 */

#include "causal_lm_api.h"
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {
constexpr const char *RESET = "\033[0m";
constexpr const char *BOLD = "\033[1m";
constexpr const char *CYAN = "\033[36m";
constexpr const char *GREEN = "\033[32m";
constexpr const char *YELLOW = "\033[33m";
constexpr const char *RED = "\033[31m";
constexpr const char *GRAY = "\033[90m";

void section(const char *title) {
  std::cout << "\n" << BOLD << CYAN << "── " << title << " ──" << RESET << "\n";
}

bool check(ErrorCode err, const char *msg) {
  if (err != CAUSAL_LM_ERROR_NONE) {
    std::cerr << RED << "FAIL: " << RESET << msg << " (error=" << err << ")\n";
    return false;
  }
  std::cout << GREEN << "  OK: " << RESET << msg << "\n";
  return true;
}

ModelQuantizationType parse_quant(const char *s) {
  std::string q(s);
  if (q == "W4A32")
    return CAUSAL_LM_QUANTIZATION_W4A32;
  if (q == "W16A16")
    return CAUSAL_LM_QUANTIZATION_W16A16;
  if (q == "W8A16")
    return CAUSAL_LM_QUANTIZATION_W8A16;
  if (q == "W32A32")
    return CAUSAL_LM_QUANTIZATION_W32A32;
  return CAUSAL_LM_QUANTIZATION_UNKNOWN;
}

ModelType parse_model_type(const char *s) {
  std::string m(s);
  if (m == "EMBEDDING-QWEN3")
    return CAUSAL_LM_MODEL_EMBEDDING_QWEN3;
  if (m == "EMBEDDING-QWEN2")
    return CAUSAL_LM_MODEL_EMBEDDING_QWEN2;
  if (m == "EMBEDDING-GEMMA3")
    return CAUSAL_LM_MODEL_EMBEDDING_GEMMA3;
  return CAUSAL_LM_MODEL_EMBEDDING_QWEN3; // default
}

float cosine_similarity(const float *a, const float *b, unsigned int dim) {
  double dot = 0, norm_a = 0, norm_b = 0;
  for (unsigned int i = 0; i < dim; ++i) {
    dot += a[i] * b[i];
    norm_a += a[i] * a[i];
    norm_b += b[i] * b[i];
  }
  if (norm_a == 0 || norm_b == 0)
    return 0;
  return static_cast<float>(dot / (std::sqrt(norm_a) * std::sqrt(norm_b)));
}

void print_vector(const float *data, unsigned int dim, unsigned int max_show) {
  unsigned int n = std::min(dim, max_show);
  std::cout << GRAY << "  [";
  for (unsigned int i = 0; i < n; ++i) {
    std::cout << std::fixed << std::setprecision(6) << data[i];
    if (i < n - 1)
      std::cout << ", ";
  }
  if (dim > max_show)
    std::cout << ", ... (" << dim << " dims)";
  std::cout << "]" << RESET << "\n";
}

void print_usage(const char *prog) {
  std::cout << YELLOW << "Usage: " << RESET << prog
            << " <model_type> [quantization] [prompt1] [prompt2]\n\n";
  std::cout << "  model_type    EMBEDDING-QWEN3 | EMBEDDING-QWEN2 | "
               "EMBEDDING-GEMMA3\n";
  std::cout << "  quantization  W4A32 | W16A16 | W8A16 | W32A32 | UNKNOWN "
               "(default: UNKNOWN)\n";
  std::cout << "  prompt1       First text to encode (default: 'Hello, how are "
               "you?')\n";
  std::cout << "  prompt2       Second text for similarity comparison "
               "(optional)\n\n";
  std::cout << "Examples:\n";
  std::cout << "  " << prog << " EMBEDDING-QWEN3 W16A16\n";
  std::cout << "  " << prog
            << " EMBEDDING-QWEN3 W16A16 \"cat\" \"kitten\"\n";
}
} // namespace

int main(int argc, char *argv[]) {
  std::cout << BOLD << CYAN << "\n  CausalLM Embedding API Test\n" << RESET;

  if (argc < 2) {
    print_usage(argv[0]);
    return 1;
  }

  ModelType model_type = parse_model_type(argv[1]);
  ModelQuantizationType quant_type =
    (argc >= 3) ? parse_quant(argv[2]) : CAUSAL_LM_QUANTIZATION_UNKNOWN;
  const char *prompt1 = (argc >= 4) ? argv[3] : "Hello, how are you?";
  const char *prompt2 = (argc >= 5) ? argv[4] : nullptr;

  // ── 1. Set Options ──
  section("Set Options");
  Config config = {};
  config.use_chat_template = false;
  config.debug_mode = false;
  config.verbose = false;
  if (!check(setOptions(config), "setOptions"))
    return 1;

  // ── 2. Load Model ──
  section("Load Model");
  std::cout << CYAN << "  Model: " << RESET << argv[1] << "\n";
  std::cout << CYAN << "  Quant: " << RESET
            << ((argc >= 3) ? argv[2] : "UNKNOWN") << "\n";

  ErrorCode err = loadModel(CAUSAL_LM_BACKEND_CPU, model_type, quant_type);
  if (!check(err, "loadModel"))
    return 1;

  // ── 3. Verify: runModel should fail for embedding models ──
  section("API Validation");
  {
    const char *dummy_output = nullptr;
    ErrorCode should_fail = runModel("test", &dummy_output);
    if (should_fail == CAUSAL_LM_ERROR_INVALID_PARAMETER) {
      std::cout << GREEN << "  OK: " << RESET
                << "runModel correctly rejected embedding model\n";
    } else {
      std::cerr << RED << "FAIL: " << RESET
                << "runModel should return INVALID_PARAMETER for embedding "
                   "model, got "
                << should_fail << "\n";
    }
  }

  // ── 4. Run Embedding (prompt 1) ──
  section("Run Embedding");
  std::cout << CYAN << "  Prompt: " << RESET << "\"" << prompt1 << "\"\n";

  float *emb_data = nullptr;
  unsigned int emb_dim = 0, emb_len = 0;
  err = runModelFloat(prompt1, &emb_data, &emb_dim, &emb_len);
  if (!check(err, "runModelFloat"))
    return 1;

  std::cout << CYAN << "  Dim:    " << RESET << emb_dim << "\n";
  std::cout << CYAN << "  Batch:  " << RESET << emb_len << "\n";
  print_vector(emb_data, emb_dim, 8);

  // Sanity check: vector should not be all zeros
  float norm = 0;
  for (unsigned int i = 0; i < emb_dim; ++i)
    norm += emb_data[i] * emb_data[i];
  norm = std::sqrt(norm);
  std::cout << CYAN << "  L2 Norm: " << RESET << std::fixed
            << std::setprecision(6) << norm << "\n";

  if (norm < 1e-6f) {
    std::cerr << RED << "FAIL: " << RESET
              << "Embedding vector is all zeros!\n";
    return 1;
  }

  // Save prompt1 embedding for comparison
  std::vector<float> emb1(emb_data, emb_data + emb_dim);

  // ── 5. Cosine similarity (if prompt2 given) ──
  if (prompt2) {
    section("Similarity Comparison");
    std::cout << CYAN << "  Prompt A: " << RESET << "\"" << prompt1 << "\"\n";
    std::cout << CYAN << "  Prompt B: " << RESET << "\"" << prompt2 << "\"\n";

    float *emb_data2 = nullptr;
    unsigned int emb_dim2 = 0, emb_len2 = 0;
    err = runModelFloat(prompt2, &emb_data2, &emb_dim2, &emb_len2);
    if (!check(err, "runModelFloat (prompt2)"))
      return 1;

    print_vector(emb_data2, emb_dim2, 8);

    float sim = cosine_similarity(emb1.data(), emb_data2, emb_dim);
    std::cout << "\n"
              << BOLD << CYAN << "  Cosine Similarity: " << RESET << BOLD
              << std::fixed << std::setprecision(4) << sim << RESET << "\n";
  }

  // ── Done ──
  section("Result");
  std::cout << GREEN << BOLD << "  All tests passed!" << RESET << "\n\n";

  return 0;
}
