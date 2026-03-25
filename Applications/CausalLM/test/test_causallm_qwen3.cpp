// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   test_causallm_qwen3.cpp
 * @date   25 March 2026
 * @brief  Verification test for Qwen3 CausalLM model.
 *         Compares C++ inference results against Python (HuggingFace)
 * reference.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "json.hpp"
#include <app_context.h>
#include <factory.h>

#include <causal_lm.h>
#include <llm_util.hpp>
#include <qwen3_causallm.h>

using json = nlohmann::json;

/**
 * @brief Helper to load numpy .npy files (int32 or float32, 1D)
 */
template <typename T> static std::vector<T> load_npy(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f.is_open())
    throw std::runtime_error("Cannot open npy file: " + path);

  // Parse numpy .npy header
  char magic[6];
  f.read(magic, 6);

  uint8_t major_ver, minor_ver;
  f.read(reinterpret_cast<char *>(&major_ver), 1);
  f.read(reinterpret_cast<char *>(&minor_ver), 1);

  uint16_t header_len;
  f.read(reinterpret_cast<char *>(&header_len), 2);

  std::string header(header_len, ' ');
  f.read(&header[0], header_len);

  // Read all remaining data
  std::vector<T> data;
  T val;
  while (f.read(reinterpret_cast<char *>(&val), sizeof(T))) {
    data.push_back(val);
  }

  return data;
}

/**
 * @brief Testable subclass of Qwen3CausalLM that exposes inference internals
 */
class TestableQwen3CausalLM : public causallm::Qwen3CausalLM {
public:
  TestableQwen3CausalLM(json &cfg, json &gen_cfg, json &nntr_cfg) :
    causallm::Transformer(cfg, gen_cfg, nntr_cfg,
                          causallm::ModelType::CAUSALLM),
    causallm::Qwen3CausalLM(cfg, gen_cfg, nntr_cfg) {}

  struct TestResult {
    std::vector<float> prefill_logits; /**< Full logits from prefill step */
    std::vector<unsigned int>
      generated_ids; /**< Greedy-decoded token IDs per step */
  };

  /**
   * @brief Run inference and return prefill logits and generated token IDs.
   *        Uses pre-computed input token IDs (bypasses tokenizer encoding).
   */
  TestResult testInference(const std::vector<int32_t> &input_token_ids,
                           int num_generate) {
    TestResult result;

    if (!is_initialized) {
      throw std::runtime_error("Model not initialized");
    }

    unsigned int input_len = input_token_ids.size();
    float *input_sample =
      (float *)malloc(sizeof(float) * BATCH_SIZE * MAX_SEQ_LEN);
    memset(input_sample, 0, sizeof(float) * BATCH_SIZE * MAX_SEQ_LEN);

    for (unsigned int b = 0; b < BATCH_SIZE; ++b) {
      for (unsigned int i = 0; i < input_len; ++i) {
        input_sample[static_cast<size_t>(b) * MAX_SEQ_LEN + i] =
          static_cast<float>(input_token_ids[i]);
        ids_history[static_cast<size_t>(b) * MAX_SEQ_LEN + i] =
          input_token_ids[i];
      }
    }

    std::vector<float *> input;
    std::vector<float *> label;
    input.push_back(input_sample);

    // Prefill
    auto output = model->incremental_inference(BATCH_SIZE, input, label,
                                               input_len, 0, input_len, false);

    // Capture prefill logits (full vocab)
    result.prefill_logits.assign(output[0], output[0] + NUM_VOCAB);

    // Get first generated token (greedy argmax)
    auto first_token_id = static_cast<unsigned int>(std::distance(
      output[0], std::max_element(output[0], output[0] + NUM_VOCAB)));
    result.generated_ids.push_back(first_token_id);

    // Update input for next step
    for (unsigned int b = 0; b < BATCH_SIZE; ++b) {
      input_sample[static_cast<size_t>(b) * MAX_SEQ_LEN] =
        static_cast<float>(first_token_id);
      ids_history[static_cast<size_t>(b) * MAX_SEQ_LEN + input_len] =
        first_token_id;
    }

    // Generate subsequent tokens
    for (int step = 1; step < num_generate; ++step) {
      unsigned int pos = input_len + step;
      auto step_output = model->incremental_inference(BATCH_SIZE, input, label,
                                                      input_len, pos - 1, pos);

      auto next_token = static_cast<unsigned int>(std::distance(
        step_output[0],
        std::max_element(step_output[0], step_output[0] + NUM_VOCAB)));
      result.generated_ids.push_back(next_token);

      for (unsigned int b = 0; b < BATCH_SIZE; ++b) {
        input_sample[static_cast<size_t>(b) * MAX_SEQ_LEN] =
          static_cast<float>(next_token);
        ids_history[static_cast<size_t>(b) * MAX_SEQ_LEN + pos] = next_token;
      }
    }

    free(input_sample);
    return result;
  }
};

/**
 * @brief Test fixture that loads model once and runs inference in SetUp
 */
class CausalLMQwen3Test : public ::testing::Test {
protected:
  static std::string model_path;
  static TestableQwen3CausalLM::TestResult test_result;
  static bool inference_done;
  static std::vector<int32_t> ref_generated;
  static std::vector<float> ref_logits;
  static std::vector<int> ref_top_indices;
  static std::vector<float> ref_top_values;

  static void SetUpTestSuite() {
    const char *env_path = std::getenv("CAUSALLM_TEST_MODEL_PATH");
    if (env_path == nullptr) {
      return;
    }
    model_path = std::string(env_path);

    // Check required files exist
    std::vector<std::string> required = {"config.json",
                                         "generation_config.json",
                                         "nntr_config.json",
                                         "reference_input_ids.npy",
                                         "reference_generated_ids.npy",
                                         "reference_prefill_logits.npy",
                                         "reference_topk.json"};

    for (const auto &fname : required) {
      std::ifstream check(model_path + "/" + fname);
      if (!check.good()) {
        std::cerr << "Missing required file: " << model_path << "/" << fname
                  << std::endl;
        return;
      }
    }

    // Load configs
    json cfg = causallm::LoadJsonFile(model_path + "/config.json");
    json gen_cfg =
      causallm::LoadJsonFile(model_path + "/generation_config.json");
    json nntr_cfg = causallm::LoadJsonFile(model_path + "/nntr_config.json");

    int num_generate = nntr_cfg["num_to_generate"].get<int>();

    // Load reference data
    auto input_ids = load_npy<int32_t>(model_path + "/reference_input_ids.npy");
    ref_generated =
      load_npy<int32_t>(model_path + "/reference_generated_ids.npy");
    ref_logits = load_npy<float>(model_path + "/reference_prefill_logits.npy");

    json topk_ref = causallm::LoadJsonFile(model_path + "/reference_topk.json");
    ref_top_indices = topk_ref["top_indices"].get<std::vector<int>>();
    ref_top_values = topk_ref["top_values"].get<std::vector<float>>();

    // Create and run model
    std::cout << "[Setup] Initializing Qwen3CausalLM model..." << std::endl;

    TestableQwen3CausalLM model(cfg, gen_cfg, nntr_cfg);
    model.initialize();

    std::string weight_file =
      model_path + "/" + nntr_cfg["model_file_name"].get<std::string>();
    std::cout << "[Setup] Loading weights from " << weight_file << std::endl;
    model.load_weight(weight_file);

    std::cout << "[Setup] Running inference with " << input_ids.size()
              << " input tokens, generating " << num_generate << " tokens..."
              << std::endl;
    test_result = model.testInference(input_ids, num_generate);
    inference_done = true;
    std::cout << "[Setup] Inference complete." << std::endl;
  }

  void SetUp() override {
    if (model_path.empty()) {
      GTEST_SKIP() << "CAUSALLM_TEST_MODEL_PATH not set. "
                   << "Run generate_reference.py first and set the env var.";
    }
    if (!inference_done) {
      GTEST_SKIP() << "Model inference did not complete successfully.";
    }
  }
};

// Static member definitions
std::string CausalLMQwen3Test::model_path;
TestableQwen3CausalLM::TestResult CausalLMQwen3Test::test_result;
bool CausalLMQwen3Test::inference_done = false;
std::vector<int32_t> CausalLMQwen3Test::ref_generated;
std::vector<float> CausalLMQwen3Test::ref_logits;
std::vector<int> CausalLMQwen3Test::ref_top_indices;
std::vector<float> CausalLMQwen3Test::ref_top_values;

/**
 * @brief Verify that C++ prefill logits match Python reference (top-K)
 */
TEST_F(CausalLMQwen3Test, verify_prefill_logits) {
  // Verify prefill logits size
  ASSERT_EQ(test_result.prefill_logits.size(), ref_logits.size())
    << "Vocab size mismatch between C++ and Python";

  // Find top-K from C++ logits
  std::vector<std::pair<int, float>> cpp_indexed_logits;
  for (size_t i = 0; i < test_result.prefill_logits.size(); ++i) {
    cpp_indexed_logits.push_back(
      {static_cast<int>(i), test_result.prefill_logits[i]});
  }
  std::partial_sort(
    cpp_indexed_logits.begin(),
    cpp_indexed_logits.begin() + static_cast<long>(ref_top_indices.size()),
    cpp_indexed_logits.end(),
    [](const auto &a, const auto &b) { return a.second > b.second; });

  std::cout << "[Prefill Logit Comparison]" << std::endl;
  for (size_t k = 0; k < ref_top_indices.size(); ++k) {
    std::cout << "  Top-" << k << ": C++=[" << cpp_indexed_logits[k].first
              << ", " << cpp_indexed_logits[k].second << "] Python=["
              << ref_top_indices[k] << ", " << ref_top_values[k] << "]"
              << std::endl;
  }

  // Verify top-1 index matches (critical for greedy decoding correctness)
  EXPECT_EQ(cpp_indexed_logits[0].first, ref_top_indices[0])
    << "Top-1 token mismatch: C++ predicted token "
    << cpp_indexed_logits[0].first << " but Python predicted "
    << ref_top_indices[0];

  // Verify top-K logit values with tolerance
  float logit_tolerance = 1e-3;
  for (size_t k = 0; k < ref_top_indices.size(); ++k) {
    int ref_idx = ref_top_indices[k];
    float cpp_val = test_result.prefill_logits[ref_idx];
    float ref_val = ref_top_values[k];
    float diff = std::abs(cpp_val - ref_val);

    EXPECT_LT(diff, logit_tolerance)
      << "Logit mismatch at rank " << k << " (token " << ref_idx
      << "): C++=" << cpp_val << " Python=" << ref_val << " diff=" << diff;
  }
}

/**
 * @brief Verify that C++ generated token IDs match Python reference
 */
TEST_F(CausalLMQwen3Test, verify_generated_tokens) {
  ASSERT_EQ(test_result.generated_ids.size(),
            static_cast<size_t>(ref_generated.size()))
    << "Number of generated tokens mismatch";

  std::cout << "[Token Generation Comparison]" << std::endl;
  int mismatch_count = 0;
  for (size_t i = 0; i < test_result.generated_ids.size(); ++i) {
    bool match = (test_result.generated_ids[i] ==
                  static_cast<unsigned int>(ref_generated[i]));
    std::cout << "  Step " << i << ": C++=" << test_result.generated_ids[i]
              << " Python=" << ref_generated[i]
              << (match ? " [OK]" : " [MISMATCH]") << std::endl;
    if (!match)
      ++mismatch_count;
  }

  for (size_t i = 0; i < test_result.generated_ids.size(); ++i) {
    EXPECT_EQ(test_result.generated_ids[i],
              static_cast<unsigned int>(ref_generated[i]))
      << "Token mismatch at generation step " << i;
  }

  std::cout << "Result: " << test_result.generated_ids.size() - mismatch_count
            << "/" << test_result.generated_ids.size() << " tokens matched"
            << std::endl;
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
