// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    unittest_causallm_api.cpp
 * @date    31 Mar 2026
 * @brief   Unit tests for CausalLM C API (causal_lm_api.h)
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Eunju Yang <ej.yang@samsung.com>
 * @bug     No known bugs except for NYI items
 *
 * @note    Tests are categorized by suffix:
 *          _p : positive test (expected to succeed)
 *          _n : negative test (expected to fail gracefully)
 */

#include <gtest/gtest.h>

#include "causal_lm_api.h"

#include <cstring>

/**
 * @brief Test suite for setOptions API
 */
class SetOptionsTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}
};

/**
 * @brief setOptions with valid config
 */
TEST_F(SetOptionsTest, set_options_p) {
  Config config;
  config.use_chat_template = true;
  config.debug_mode = false;
  config.verbose = false;

  ErrorCode err = setOptions(config);
  EXPECT_EQ(err, CAUSAL_LM_ERROR_NONE);
}

/**
 * @brief setOptions with all false
 */
TEST_F(SetOptionsTest, set_options_all_false_p) {
  Config config;
  config.use_chat_template = false;
  config.debug_mode = false;
  config.verbose = false;

  ErrorCode err = setOptions(config);
  EXPECT_EQ(err, CAUSAL_LM_ERROR_NONE);
}

/**
 * @brief setOptions with all true
 */
TEST_F(SetOptionsTest, set_options_all_true_p) {
  Config config;
  config.use_chat_template = true;
  config.debug_mode = true;
  config.verbose = true;

  ErrorCode err = setOptions(config);
  EXPECT_EQ(err, CAUSAL_LM_ERROR_NONE);
}

/**
 * @brief Test suite for runModel API (before model load)
 */
class RunModelUninitializedTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}
};

/**
 * @brief runModel before loadModel should return NOT_INITIALIZED
 */
TEST_F(RunModelUninitializedTest, run_model_not_initialized_n) {
  const char *output = nullptr;
  ErrorCode err = runModel("Hello", &output);
  EXPECT_EQ(err, CAUSAL_LM_ERROR_NOT_INITIALIZED);
}

/**
 * @brief runModel with null prompt before init
 */
TEST_F(RunModelUninitializedTest, run_model_null_prompt_not_initialized_n) {
  const char *output = nullptr;
  ErrorCode err = runModel(nullptr, &output);
  EXPECT_EQ(err, CAUSAL_LM_ERROR_NOT_INITIALIZED);
}

/**
 * @brief runModel with null output buffer before init
 */
TEST_F(RunModelUninitializedTest, run_model_null_output_not_initialized_n) {
  ErrorCode err = runModel("Hello", nullptr);
  EXPECT_EQ(err, CAUSAL_LM_ERROR_NOT_INITIALIZED);
}

/**
 * @brief Test suite for runEmbeddingModel API (before model load)
 */
class RunEmbeddingUninitializedTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}
};

/**
 * @brief runEmbeddingModel before loadModel should return NOT_INITIALIZED
 */
TEST_F(RunEmbeddingUninitializedTest,
       run_embedding_model_not_initialized_n) {
  EmbeddingResult result;
  memset(&result, 0, sizeof(EmbeddingResult));
  ErrorCode err = runEmbeddingModel("Hello", &result);
  EXPECT_EQ(err, CAUSAL_LM_ERROR_NOT_INITIALIZED);
}

/**
 * @brief runEmbeddingModel with null prompt before init
 */
TEST_F(RunEmbeddingUninitializedTest,
       run_embedding_null_prompt_not_initialized_n) {
  EmbeddingResult result;
  memset(&result, 0, sizeof(EmbeddingResult));
  ErrorCode err = runEmbeddingModel(nullptr, &result);
  EXPECT_EQ(err, CAUSAL_LM_ERROR_NOT_INITIALIZED);
}

/**
 * @brief runEmbeddingModel with null result before init
 */
TEST_F(RunEmbeddingUninitializedTest,
       run_embedding_null_result_not_initialized_n) {
  ErrorCode err = runEmbeddingModel("Hello", nullptr);
  EXPECT_EQ(err, CAUSAL_LM_ERROR_NOT_INITIALIZED);
}

/**
 * @brief Test suite for getPerformanceMetrics API (before model load)
 */
class MetricsUninitializedTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}
};

/**
 * @brief getPerformanceMetrics before loadModel should return NOT_INITIALIZED
 */
TEST_F(MetricsUninitializedTest, get_metrics_not_initialized_n) {
  PerformanceMetrics metrics;
  ErrorCode err = getPerformanceMetrics(&metrics);
  EXPECT_EQ(err, CAUSAL_LM_ERROR_NOT_INITIALIZED);
}

/**
 * @brief getPerformanceMetrics with null pointer before init
 */
TEST_F(MetricsUninitializedTest, get_metrics_null_not_initialized_n) {
  ErrorCode err = getPerformanceMetrics(nullptr);
  EXPECT_EQ(err, CAUSAL_LM_ERROR_NOT_INITIALIZED);
}

/**
 * @brief Test suite for freeEmbeddingResult API
 */
class FreeEmbeddingResultTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}
};

/**
 * @brief freeEmbeddingResult with null pointer should not crash
 */
TEST_F(FreeEmbeddingResultTest, free_null_p) {
  freeEmbeddingResult(nullptr);
  SUCCEED();
}

/**
 * @brief freeEmbeddingResult with zero-initialized struct should not crash
 */
TEST_F(FreeEmbeddingResultTest, free_zero_initialized_p) {
  EmbeddingResult result;
  memset(&result, 0, sizeof(EmbeddingResult));
  freeEmbeddingResult(&result);
  EXPECT_EQ(result.embeddings, nullptr);
  EXPECT_EQ(result.embedding_dim, 0u);
  EXPECT_EQ(result.batch_size, 0u);
}

/**
 * @brief freeEmbeddingResult with allocated memory should free correctly
 */
TEST_F(FreeEmbeddingResultTest, free_allocated_p) {
  EmbeddingResult result;
  result.embedding_dim = 128;
  result.batch_size = 1;
  result.embeddings = (float *)malloc(sizeof(float) * 128);
  ASSERT_NE(result.embeddings, nullptr);

  freeEmbeddingResult(&result);
  EXPECT_EQ(result.embeddings, nullptr);
  EXPECT_EQ(result.embedding_dim, 0u);
  EXPECT_EQ(result.batch_size, 0u);
}

/**
 * @brief Double free should not crash (embeddings set to null after first free)
 */
TEST_F(FreeEmbeddingResultTest, double_free_p) {
  EmbeddingResult result;
  result.embedding_dim = 64;
  result.batch_size = 2;
  result.embeddings = (float *)malloc(sizeof(float) * 128);
  ASSERT_NE(result.embeddings, nullptr);

  freeEmbeddingResult(&result);
  EXPECT_EQ(result.embeddings, nullptr);

  // Second free should be safe
  freeEmbeddingResult(&result);
  EXPECT_EQ(result.embeddings, nullptr);
}

/**
 * @brief Test suite for loadModel API (without model files)
 */
class LoadModelTest : public ::testing::Test {
protected:
  void SetUp() override {
    Config config;
    config.use_chat_template = false;
    config.debug_mode = false;
    config.verbose = false;
    setOptions(config);
  }
  void TearDown() override {}
};

/**
 * @brief loadModel with non-existent model files should fail gracefully
 * @note  This test expects MODEL_LOAD_FAILED since no weight files exist in the
 *        test environment
 */
TEST_F(LoadModelTest, load_model_no_files_n) {
  ErrorCode err = loadModel(CAUSAL_LM_BACKEND_CPU, CAUSAL_LM_MODEL_QWEN3_0_6B,
                            CAUSAL_LM_QUANTIZATION_W4A32);
  EXPECT_NE(err, CAUSAL_LM_ERROR_NONE);
}

/**
 * @brief loadModel with embedding model type (no files)
 */
TEST_F(LoadModelTest, load_embedding_model_no_files_n) {
  ErrorCode err =
    loadModel(CAUSAL_LM_BACKEND_CPU, CAUSAL_LM_MODEL_QWEN3_EMBEDDING,
              CAUSAL_LM_QUANTIZATION_UNKNOWN);
  EXPECT_NE(err, CAUSAL_LM_ERROR_NONE);
}

/**
 * @brief Test suite for EmbeddingResult struct
 */
class EmbeddingResultStructTest : public ::testing::Test {};

/**
 * @brief EmbeddingResult struct has expected layout
 */
TEST_F(EmbeddingResultStructTest, struct_layout_p) {
  EmbeddingResult result;
  memset(&result, 0, sizeof(EmbeddingResult));

  EXPECT_EQ(result.embeddings, nullptr);
  EXPECT_EQ(result.embedding_dim, 0u);
  EXPECT_EQ(result.batch_size, 0u);
}

/**
 * @brief EmbeddingResult can hold large dimensions
 */
TEST_F(EmbeddingResultStructTest, large_dimensions_p) {
  EmbeddingResult result;
  result.embedding_dim = 4096;
  result.batch_size = 32;
  result.embeddings = nullptr;

  EXPECT_EQ(result.embedding_dim, 4096u);
  EXPECT_EQ(result.batch_size, 32u);
}

/**
 * @brief Test suite for ErrorCode enum values
 */
class ErrorCodeTest : public ::testing::Test {};

/**
 * @brief ErrorCode values should be unique and match expected values
 */
TEST_F(ErrorCodeTest, error_code_values_p) {
  EXPECT_EQ(CAUSAL_LM_ERROR_NONE, 0);
  EXPECT_EQ(CAUSAL_LM_ERROR_INVALID_PARAMETER, 1);
  EXPECT_EQ(CAUSAL_LM_ERROR_MODEL_LOAD_FAILED, 2);
  EXPECT_EQ(CAUSAL_LM_ERROR_INFERENCE_FAILED, 3);
  EXPECT_EQ(CAUSAL_LM_ERROR_NOT_INITIALIZED, 4);
  EXPECT_EQ(CAUSAL_LM_ERROR_INFERENCE_NOT_RUN, 5);
  EXPECT_EQ(CAUSAL_LM_ERROR_UNKNOWN, 99);
}

/**
 * @brief Test suite for ModelType enum values
 */
class ModelTypeTest : public ::testing::Test {};

/**
 * @brief ModelType enum includes all expected values
 */
TEST_F(ModelTypeTest, model_type_values_p) {
  EXPECT_EQ(CAUSAL_LM_MODEL_QWEN3_0_6B, 0);
  EXPECT_EQ(CAUSAL_LM_MODEL_QWEN3_EMBEDDING, 1);
  EXPECT_EQ(CAUSAL_LM_MODEL_QWEN2_EMBEDDING, 2);
  EXPECT_EQ(CAUSAL_LM_MODEL_GEMMA_EMBEDDING, 3);
}

/**
 * @brief Test suite for ModelQuantizationType enum values
 */
class QuantizationTypeTest : public ::testing::Test {};

/**
 * @brief ModelQuantizationType enum includes all expected values
 */
TEST_F(QuantizationTypeTest, quantization_type_values_p) {
  EXPECT_EQ(CAUSAL_LM_QUANTIZATION_UNKNOWN, 0);
  EXPECT_EQ(CAUSAL_LM_QUANTIZATION_W4A32, 1);
  EXPECT_EQ(CAUSAL_LM_QUANTIZATION_W16A16, 2);
  EXPECT_EQ(CAUSAL_LM_QUANTIZATION_W8A16, 3);
  EXPECT_EQ(CAUSAL_LM_QUANTIZATION_W32A32, 4);
}

/**
 * @brief Test suite for BackendType enum values
 */
class BackendTypeTest : public ::testing::Test {};

/**
 * @brief BackendType enum includes all expected values
 */
TEST_F(BackendTypeTest, backend_type_values_p) {
  EXPECT_EQ(CAUSAL_LM_BACKEND_CPU, 0);
  EXPECT_EQ(CAUSAL_LM_BACKEND_GPU, 1);
  EXPECT_EQ(CAUSAL_LM_BACKEND_NPU, 2);
}

/**
 * @brief Test suite for Config struct
 */
class ConfigStructTest : public ::testing::Test {};

/**
 * @brief Config struct fields can be set independently
 */
TEST_F(ConfigStructTest, config_fields_p) {
  Config config;
  config.use_chat_template = true;
  config.debug_mode = false;
  config.verbose = true;

  EXPECT_TRUE(config.use_chat_template);
  EXPECT_FALSE(config.debug_mode);
  EXPECT_TRUE(config.verbose);
}

/**
 * @brief Main function for unit test execution
 */
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
