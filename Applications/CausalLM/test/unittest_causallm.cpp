// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    unittest_causallm.cpp
 * @date    31 Mar 2026
 * @brief   Unit tests for CausalLM models (Qwen3, etc.)
 *          Tests model construction, weight save/load, and inference.
 *          Extensible for adding new model architectures.
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Eunju Yang <ej.yang@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <factory.h>
#include <qwen3_causallm.h>

namespace fs = std::filesystem;
using json = nlohmann::json;

// ============================================================================
// Test Configuration
// ============================================================================

/**
 * @brief Model test configuration.
 *        Add new model configs here to extend test coverage.
 */
struct ModelTestConfig {
  std::string name;          /**< Human-readable model name */
  std::string architecture;  /**< Factory architecture key */
  json cfg;                  /**< Model architecture config (config.json) */
  json generation_cfg;       /**< Generation config (generation_config.json) */
  json nntr_cfg;             /**< NNTrainer runtime config (nntr_config.json) */
  bool has_qk_norm;          /**< Whether model has Q/K norm (Qwen3-specific) */
  bool tie_word_embeddings;  /**< Whether embedding and lm_head share weights */
};

/**
 * @brief Compute total weight file size for a Qwen3-compatible model.
 */
static size_t computeWeightFileSize(const ModelTestConfig &config) {
  int vocab_size = config.cfg["vocab_size"].get<int>();
  int hidden_size = config.cfg["hidden_size"].get<int>();
  int intermediate_size = config.cfg["intermediate_size"].get<int>();
  int num_layers = config.cfg["num_hidden_layers"].get<int>();
  int num_heads = config.cfg["num_attention_heads"].get<int>();
  int head_dim = config.cfg.contains("head_dim")
                   ? config.cfg["head_dim"].get<int>()
                   : hidden_size / num_heads;
  int num_kv_heads = config.cfg.contains("num_key_value_heads")
                       ? config.cfg["num_key_value_heads"].get<int>()
                       : num_heads;

  size_t total = 0;

  // Embedding: [vocab_size, hidden_size]
  total += static_cast<size_t>(vocab_size) * hidden_size;

  // Per-layer weights
  for (int i = 0; i < num_layers; ++i) {
    total += hidden_size; // attention_norm

    total += static_cast<size_t>(hidden_size) * num_heads * head_dim; // Q
    if (config.has_qk_norm)
      total += head_dim; // Q norm

    total += static_cast<size_t>(hidden_size) * num_kv_heads * head_dim; // K
    if (config.has_qk_norm)
      total += head_dim; // K norm

    total += static_cast<size_t>(hidden_size) * num_kv_heads * head_dim; // V

    total += static_cast<size_t>(num_heads) * head_dim * hidden_size; // O

    total += hidden_size; // ffn_norm

    total += static_cast<size_t>(hidden_size) * intermediate_size; // up
    total += static_cast<size_t>(hidden_size) * intermediate_size; // gate
    total += static_cast<size_t>(intermediate_size) * hidden_size; // down
  }

  // output_norm
  total += hidden_size;

  // lm_head (skip if tied)
  if (!config.tie_word_embeddings) {
    total += static_cast<size_t>(hidden_size) * vocab_size;
  }

  return total * sizeof(float);
}

/**
 * @brief Generate a dummy weight file with small random values.
 */
static void generateDummyWeights(const std::string &path, size_t size_bytes,
                                 unsigned int seed = 42) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> dist(0.0f, 0.02f);

  std::ofstream ofs(path, std::ios::binary);
  ASSERT_TRUE(ofs.is_open()) << "Failed to create weight file: " << path;

  size_t num_floats = size_bytes / sizeof(float);
  constexpr size_t CHUNK = 4096;
  std::vector<float> buffer(CHUNK);

  for (size_t written = 0; written < num_floats; written += CHUNK) {
    size_t count = std::min(CHUNK, num_floats - written);
    for (size_t j = 0; j < count; ++j) {
      buffer[j] = dist(rng);
    }
    ofs.write(reinterpret_cast<const char *>(buffer.data()),
              static_cast<std::streamsize>(count * sizeof(float)));
  }
  ofs.close();
}

/**
 * @brief Create a minimal BPE tokenizer JSON for testing.
 */
static std::string createTestTokenizer(const std::string &dir) {
  std::string path = dir + "/test_tokenizer.json";
  std::ofstream ofs(path);
  EXPECT_TRUE(ofs.is_open()) << "Failed to create tokenizer: " << path;
  ofs << R"({
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {"id": 0, "content": "<unk>", "single_word": false, "lstrip": false,
     "rstrip": false, "normalized": false, "special": true},
    {"id": 1, "content": "<s>", "single_word": false, "lstrip": false,
     "rstrip": false, "normalized": false, "special": true},
    {"id": 2, "content": "</s>", "single_word": false, "lstrip": false,
     "rstrip": false, "normalized": false, "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false,
                     "trim_offsets": true, "use_regex": true},
  "post_processor": null,
  "decoder": {"type": "ByteLevel", "add_prefix_space": true,
               "trim_offsets": true, "use_regex": true},
  "model": {
    "type": "BPE",
    "dropout": null,
    "unk_token": "<unk>",
    "continuing_subword_prefix": null,
    "end_of_word_suffix": null,
    "fuse_unk": false,
    "byte_fallback": false,
    "vocab": {"<unk>": 0, "<s>": 1, "</s>": 2},
    "merges": []
  }
})";
  ofs.close();
  return path;
}

// ============================================================================
// Model Configuration Builders
// ============================================================================

/**
 * @brief Create a tiny Qwen3 config for fast unit testing.
 *        Uses small dimensions (2 layers, hidden_size=64) to keep
 *        memory usage and runtime minimal.
 */
static ModelTestConfig createQwen3TinyConfig(const std::string &tokenizer_path) {
  ModelTestConfig config;
  config.name = "Qwen3-Tiny";
  config.architecture = "Qwen3ForCausalLM";
  config.has_qk_norm = true;
  config.tie_word_embeddings = false;

  config.cfg = {{"architectures", {"Qwen3ForCausalLM"}},
                {"vocab_size", 32},
                {"hidden_size", 64},
                {"intermediate_size", 128},
                {"num_hidden_layers", 2},
                {"num_attention_heads", 4},
                {"head_dim", 16},
                {"num_key_value_heads", 2},
                {"max_position_embeddings", 512},
                {"rope_theta", 10000},
                {"rms_norm_eps", 1e-6},
                {"tie_word_embeddings", false},
                {"sliding_window", nullptr},
                {"sliding_window_pattern", 0}};

  config.generation_cfg = {{"bos_token_id", 1},
                           {"eos_token_id", {2}},
                           {"top_k", 1},
                           {"top_p", 1.0},
                           {"temperature", 1.0}};

  config.nntr_cfg = {{"model_type", "CausalLM"},
                     {"model_tensor_type", "FP32-FP32"},
                     {"model_file_name", "test_weights.bin"},
                     {"fc_layer_dtype", "FP32"},
                     {"embedding_dtype", "FP32"},
                     {"batch_size", 1},
                     {"init_seq_len", 4},
                     {"max_seq_len", 8},
                     {"num_to_generate", 2},
                     {"fsu", false},
                     {"fsu_lookahead", 1},
                     {"bad_word_ids", json::array()},
                     {"tokenizer_file", tokenizer_path}};

  return config;
}

/**
 * @brief Create the real Qwen3 0.6B config.
 *        Suitable for full-scale integration testing (requires ~2.4GB memory).
 */
static ModelTestConfig
createQwen3_0_6BConfig(const std::string &tokenizer_path) {
  ModelTestConfig config;
  config.name = "Qwen3-0.6B";
  config.architecture = "Qwen3ForCausalLM";
  config.has_qk_norm = true;
  config.tie_word_embeddings = true;

  config.cfg = {{"architectures", {"Qwen3ForCausalLM"}},
                {"vocab_size", 151936},
                {"hidden_size", 1024},
                {"intermediate_size", 3072},
                {"num_hidden_layers", 28},
                {"num_attention_heads", 16},
                {"head_dim", 128},
                {"num_key_value_heads", 8},
                {"max_position_embeddings", 40960},
                {"rope_theta", 1000000},
                {"rms_norm_eps", 1e-6},
                {"tie_word_embeddings", true},
                {"sliding_window", nullptr},
                {"sliding_window_pattern", 0}};

  config.generation_cfg = {{"bos_token_id", 151643},
                           {"eos_token_id", {151645, 151643}},
                           {"top_k", 20},
                           {"top_p", 0.95},
                           {"temperature", 0.6}};

  config.nntr_cfg = {{"model_type", "CausalLM"},
                     {"model_tensor_type", "FP32-FP32"},
                     {"model_file_name", "nntr_qwen3_0.6b_fp32.bin"},
                     {"fc_layer_dtype", "FP32"},
                     {"embedding_dtype", "FP32"},
                     {"batch_size", 1},
                     {"init_seq_len", 1024},
                     {"max_seq_len", 2048},
                     {"num_to_generate", 512},
                     {"fsu", false},
                     {"fsu_lookahead", 2},
                     {"bad_word_ids", json::array()},
                     {"tokenizer_file", tokenizer_path}};

  return config;
}

// ============================================================================
// Add new model configs below this line.
// To add a new model:
//   1. Create a "createXxxConfig()" function following the pattern above.
//   2. Add it to the INSTANTIATE_TEST_SUITE_P at the bottom of this file.
// ============================================================================

// ============================================================================
// Test Fixture
// ============================================================================

/**
 * @brief Parameterized test fixture for CausalLM models.
 *        Each test case receives a ModelTestConfig.
 */
class CausalLMModelTest : public ::testing::TestWithParam<ModelTestConfig> {
protected:
  void SetUp() override {
    config_ = GetParam();

    // Create temp directory for test artifacts
    test_dir_ = fs::temp_directory_path() / ("causallm_test_" + config_.name);
    fs::create_directories(test_dir_);

    weight_path_ = (test_dir_ / "test_weights.bin").string();

    // Register model factory (idempotent)
    registerFactories();
  }

  void TearDown() override {
    // Clean up test artifacts
    std::error_code ec;
    fs::remove_all(test_dir_, ec);
  }

  static void registerFactories() {
    static bool registered = false;
    if (registered)
      return;
    registered = true;

    causallm::Factory::Instance().registerModel(
      "Qwen3ForCausalLM", [](json cfg, json gen_cfg, json nntr_cfg) {
        return std::make_unique<causallm::Qwen3CausalLM>(cfg, gen_cfg,
                                                          nntr_cfg);
      });
  }

  ModelTestConfig config_;
  fs::path test_dir_;
  std::string weight_path_;
};

// ============================================================================
// Test Cases
// ============================================================================

/**
 * @brief Test that model config is valid and weight size can be computed.
 */
TEST_P(CausalLMModelTest, ConfigValidation) {
  auto &cfg = config_.cfg;

  EXPECT_TRUE(cfg.contains("vocab_size"));
  EXPECT_TRUE(cfg.contains("hidden_size"));
  EXPECT_TRUE(cfg.contains("num_hidden_layers"));
  EXPECT_TRUE(cfg.contains("num_attention_heads"));
  EXPECT_TRUE(cfg.contains("intermediate_size"));
  EXPECT_TRUE(cfg.contains("rms_norm_eps"));
  EXPECT_TRUE(cfg.contains("rope_theta"));
  EXPECT_TRUE(cfg.contains("tie_word_embeddings"));
  EXPECT_TRUE(cfg.contains("max_position_embeddings"));

  EXPECT_GT(cfg["vocab_size"].get<int>(), 0);
  EXPECT_GT(cfg["hidden_size"].get<int>(), 0);
  EXPECT_GT(cfg["num_hidden_layers"].get<int>(), 0);
  EXPECT_GT(cfg["num_attention_heads"].get<int>(), 0);

  size_t weight_size = computeWeightFileSize(config_);
  EXPECT_GT(weight_size, 0u);

  std::cout << "[INFO] " << config_.name
            << " weight size: " << weight_size / 1024 / 1024 << " MB"
            << std::endl;
}

/**
 * @brief Test model construction via factory and initialization.
 */
TEST_P(CausalLMModelTest, ModelConstruction) {
  auto model = causallm::Factory::Instance().create(
    config_.architecture, config_.cfg, config_.generation_cfg,
    config_.nntr_cfg);

  ASSERT_NE(model, nullptr)
    << "Factory failed to create model: " << config_.architecture;

  // initialize() will register custom layers, construct model graph,
  // compile and initialize
  ASSERT_NO_THROW(model->initialize())
    << "Model initialization failed for: " << config_.name;
}

/**
 * @brief Test weight file generation and loading.
 */
TEST_P(CausalLMModelTest, WeightSaveLoad) {
  // Create model and initialize
  auto model = causallm::Factory::Instance().create(
    config_.architecture, config_.cfg, config_.generation_cfg,
    config_.nntr_cfg);
  ASSERT_NE(model, nullptr);
  ASSERT_NO_THROW(model->initialize());

  // Save weights (initialized with default values)
  std::string save_path = (test_dir_ / "saved_weights.bin").string();
  ASSERT_NO_THROW(model->save_weight(save_path));

  // Verify saved file exists and has expected size
  ASSERT_TRUE(fs::exists(save_path));
  auto file_size = fs::file_size(save_path);
  EXPECT_GT(file_size, 0u);

  std::cout << "[INFO] " << config_.name
            << " saved weight file: " << file_size / 1024 << " KB"
            << std::endl;

  // Load weights back into a new model instance
  auto model2 = causallm::Factory::Instance().create(
    config_.architecture, config_.cfg, config_.generation_cfg,
    config_.nntr_cfg);
  ASSERT_NE(model2, nullptr);
  ASSERT_NO_THROW(model2->initialize());
  ASSERT_NO_THROW(model2->load_weight(save_path));
}

/**
 * @brief Test loading a pre-generated dummy weight file.
 */
TEST_P(CausalLMModelTest, DummyWeightLoad) {
  // Generate dummy weights matching the model's expected size
  size_t expected_size = computeWeightFileSize(config_);
  generateDummyWeights(weight_path_, expected_size);

  ASSERT_TRUE(fs::exists(weight_path_));
  EXPECT_EQ(fs::file_size(weight_path_), expected_size);

  // Create model and load the dummy weights
  auto model = causallm::Factory::Instance().create(
    config_.architecture, config_.cfg, config_.generation_cfg,
    config_.nntr_cfg);
  ASSERT_NE(model, nullptr);
  ASSERT_NO_THROW(model->initialize());
  ASSERT_NO_THROW(model->load_weight(weight_path_));
}

/**
 * @brief Test forward pass (inference) with dummy input.
 */
TEST_P(CausalLMModelTest, ForwardPass) {
  // Create model, initialize, and save/load weights
  auto model = causallm::Factory::Instance().create(
    config_.architecture, config_.cfg, config_.generation_cfg,
    config_.nntr_cfg);
  ASSERT_NE(model, nullptr);
  ASSERT_NO_THROW(model->initialize());

  // Save initialized weights and load them to ensure proper weight state
  std::string save_path = (test_dir_ / "forward_weights.bin").string();
  ASSERT_NO_THROW(model->save_weight(save_path));
  ASSERT_NO_THROW(model->load_weight(save_path));

  // Run inference with a dummy prompt
  // Note: tokenizer is minimal, so output may be empty or garbled.
  // We are testing that the forward pass completes without error.
  ASSERT_NO_THROW(model->run("test", false, "", "", false));
}

/**
 * @brief Test that weight file size matches between save and compute.
 */
TEST_P(CausalLMModelTest, WeightSizeConsistency) {
  auto model = causallm::Factory::Instance().create(
    config_.architecture, config_.cfg, config_.generation_cfg,
    config_.nntr_cfg);
  ASSERT_NE(model, nullptr);
  ASSERT_NO_THROW(model->initialize());

  // Save weights and check file size
  std::string save_path = (test_dir_ / "size_check.bin").string();
  ASSERT_NO_THROW(model->save_weight(save_path));

  auto saved_size = fs::file_size(save_path);
  size_t computed_size = computeWeightFileSize(config_);

  // The saved file may include additional metadata or tensor headers,
  // so we check that saved size >= computed parameter size.
  EXPECT_GE(saved_size, computed_size)
    << "Saved weight file (" << saved_size
    << " bytes) is smaller than expected parameter data (" << computed_size
    << " bytes) for " << config_.name;

  std::cout << "[INFO] " << config_.name
            << " saved: " << saved_size / 1024 << " KB"
            << ", computed: " << computed_size / 1024 << " KB" << std::endl;
}

// ============================================================================
// Test Suite Instantiation
// ============================================================================

// Shared test resource directory
static std::string g_test_dir;
static std::string g_tokenizer_path;

/**
 * @brief Initialize shared test resources.
 */
static void initTestResources() {
  static bool initialized = false;
  if (initialized)
    return;
  initialized = true;

  g_test_dir = (fs::temp_directory_path() / "causallm_test_shared").string();
  fs::create_directories(g_test_dir);
  g_tokenizer_path = createTestTokenizer(g_test_dir);
}

/**
 * @brief Generate test configurations.
 *        Add new models here to include them in the test suite.
 */
static std::vector<ModelTestConfig> generateTestConfigs() {
  initTestResources();
  return {
    createQwen3TinyConfig(g_tokenizer_path),
    // Uncomment to test with real 0.6B config (requires ~2.4GB memory):
    // createQwen3_0_6BConfig(g_tokenizer_path),
  };
}

/**
 * @brief Custom test name generator for readable output.
 */
static std::string
testNameGenerator(const ::testing::TestParamInfo<ModelTestConfig> &info) {
  std::string name = info.param.name;
  // Replace non-alphanumeric characters for GTest compatibility
  std::replace_if(
    name.begin(), name.end(), [](char c) { return !std::isalnum(c); }, '_');
  return name;
}

INSTANTIATE_TEST_SUITE_P(CausalLM, CausalLMModelTest,
                         ::testing::ValuesIn(generateTestConfigs()),
                         testNameGenerator);

// ============================================================================
// Standalone Tests (non-parameterized)
// ============================================================================

/**
 * @brief Test that factory rejects unknown architecture.
 */
TEST(CausalLMFactory, UnknownArchitectureReturnsNull) {
  json cfg = {{"architectures", {"UnknownModel"}}};
  json gen_cfg = {};
  json nntr_cfg = {};

  auto model = causallm::Factory::Instance().create("UnknownModel", cfg,
                                                     gen_cfg, nntr_cfg);
  EXPECT_EQ(model, nullptr);
}

/**
 * @brief Test Qwen3 0.6B config weight size calculation.
 */
TEST(Qwen3WeightSize, Qwen3_0_6B_WeightSize) {
  initTestResources();
  auto config = createQwen3_0_6BConfig(g_tokenizer_path);
  size_t weight_size = computeWeightFileSize(config);

  // Qwen3 0.6B with tie_word_embeddings=true should be roughly ~1.2GB
  // (parameters only, no lm_head since tied)
  // Approximate: embedding(151936*1024) + 28 layers + output_norm
  EXPECT_GT(weight_size, 500u * 1024 * 1024);  // > 500 MB
  EXPECT_LT(weight_size, 3000u * 1024 * 1024); // < 3 GB

  std::cout << "[INFO] Qwen3-0.6B computed weight size: "
            << weight_size / 1024 / 1024 << " MB" << std::endl;
}

/**
 * @brief Test dummy weight generation utility.
 */
TEST(WeightGeneration, GenerateDummyWeights) {
  std::string path =
    (fs::temp_directory_path() / "test_dummy_weights.bin").string();
  size_t target_size = 1024 * sizeof(float); // 1K floats

  generateDummyWeights(path, target_size);
  ASSERT_TRUE(fs::exists(path));
  EXPECT_EQ(fs::file_size(path), target_size);

  // Read back and verify not all zeros
  std::ifstream ifs(path, std::ios::binary);
  std::vector<float> data(1024);
  ifs.read(reinterpret_cast<char *>(data.data()),
           static_cast<std::streamsize>(target_size));
  ifs.close();

  bool has_nonzero = false;
  for (float v : data) {
    EXPECT_FALSE(std::isnan(v)) << "Generated weight contains NaN";
    EXPECT_FALSE(std::isinf(v)) << "Generated weight contains Inf";
    if (v != 0.0f)
      has_nonzero = true;
  }
  EXPECT_TRUE(has_nonzero) << "All generated weights are zero";

  fs::remove(path);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  int result = RUN_ALL_TESTS();

  // Cleanup shared resources
  if (!g_test_dir.empty()) {
    std::error_code ec;
    fs::remove_all(g_test_dir, ec);
  }

  return result;
}
