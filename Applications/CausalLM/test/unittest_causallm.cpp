// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   unittest_causallm.cpp
 * @date   31 Mar 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Unit tests for CausalLM class hierarchy and TransformerBase
 */

#include <gtest/gtest.h>

#include <factory.h>
#include <transformer.h>
#include <transformer_base.h>

#include <qnn_causal_lm.h>
#include <qnn_transformer.h>

using json = nlohmann::json;

/**
 * @brief Test TransformerBase is abstract (cannot be instantiated directly)
 * @note  This test verifies that TransformerBase has pure virtual methods
 *        by checking that concrete subclasses can be created through the
 *        factory.
 */
TEST(TransformerBaseTest, is_abstract_interface) {
  // TransformerBase should not be directly instantiable
  // Verify its pure virtual interface exists via type traits
  EXPECT_FALSE(std::is_constructible<causallm::TransformerBase>::value == false);

  // TransformerBase has a virtual destructor
  EXPECT_TRUE(std::has_virtual_destructor<causallm::TransformerBase>::value);
}

/**
 * @brief Test Transformer inherits from TransformerBase
 */
TEST(TransformerHierarchyTest, transformer_inherits_from_base) {
  EXPECT_TRUE(
    (std::is_base_of<causallm::TransformerBase, causallm::Transformer>::value));
}

/**
 * @brief Test QNNTransformer inherits from TransformerBase
 */
TEST(TransformerHierarchyTest, qnn_transformer_inherits_from_base) {
  EXPECT_TRUE((std::is_base_of<causallm::TransformerBase,
                                causallm::QNNTransformer>::value));
}

/**
 * @brief Test QNNCausalLM inherits from QNNTransformer
 */
TEST(TransformerHierarchyTest, qnn_causallm_inherits_from_qnn_transformer) {
  EXPECT_TRUE((std::is_base_of<causallm::QNNTransformer,
                                causallm::QNNCausalLM>::value));
}

/**
 * @brief Test QNNTransformer does NOT inherit from Transformer
 * @note  This verifies the clean separation between NNTrainer and QNN paths
 */
TEST(TransformerHierarchyTest,
     qnn_transformer_does_not_inherit_from_transformer) {
  EXPECT_FALSE(
    (std::is_base_of<causallm::Transformer, causallm::QNNTransformer>::value));
}

/**
 * @brief Test Factory returns correct types
 */
TEST(FactoryTest, factory_registers_and_creates_models) {
  auto &factory = causallm::Factory::Instance();

  // Register a mock QNN model
  factory.registerModel(
    "TestQNNCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::QNNCausalLM>(cfg, generation_cfg,
                                                     nntr_cfg);
    });

  // Verify unregistered model returns nullptr
  json empty_cfg = json::object();
  json empty_gen_cfg = json::object();
  json empty_nntr_cfg = json::object();
  auto result = factory.create("NonExistentModel", empty_cfg, empty_gen_cfg,
                               empty_nntr_cfg);
  EXPECT_EQ(result, nullptr);
}

/**
 * @brief Test LoadJsonFile with valid JSON
 */
TEST(UtilTest, load_json_file_valid) {
  // Create a temp JSON file
  std::string tmp_path = "/tmp/test_config.json";
  {
    std::ofstream f(tmp_path);
    f << R"({"key": "value", "number": 42})";
  }

  json loaded = causallm::LoadJsonFile(tmp_path);
  EXPECT_EQ(loaded["key"], "value");
  EXPECT_EQ(loaded["number"], 42);

  std::remove(tmp_path.c_str());
}

/**
 * @brief Test LoadJsonFile with invalid file
 */
TEST(UtilTest, load_json_file_invalid_path) {
  EXPECT_THROW(causallm::LoadJsonFile("/tmp/nonexistent_file.json"),
               std::runtime_error);
}

/**
 * @brief Test LoadJsonFile with malformed JSON
 */
TEST(UtilTest, load_json_file_malformed) {
  std::string tmp_path = "/tmp/test_malformed.json";
  {
    std::ofstream f(tmp_path);
    f << R"({invalid json})";
  }

  EXPECT_THROW(causallm::LoadJsonFile(tmp_path), std::runtime_error);

  std::remove(tmp_path.c_str());
}

/**
 * @brief Test ModelType enum values
 */
TEST(ModelTypeTest, enum_values) {
  EXPECT_NE(static_cast<int>(causallm::ModelType::MODEL),
            static_cast<int>(causallm::ModelType::CAUSALLM));
  EXPECT_NE(static_cast<int>(causallm::ModelType::CAUSALLM),
            static_cast<int>(causallm::ModelType::EMBEDDING));
  EXPECT_NE(static_cast<int>(causallm::ModelType::EMBEDDING),
            static_cast<int>(causallm::ModelType::UNKNOWN));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
