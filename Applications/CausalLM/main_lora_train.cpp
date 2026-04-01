// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   main_lora_train.cpp
 * @date   01 Apr 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  LoRA training entry point for CausalLM
 *
 * Usage:
 *   nntr_causallm_lora_train <model_dir> <train_data.txt> [options]
 *
 * Options:
 *   --lr <float>         Learning rate (default: 1e-4)
 *   --epochs <int>       Number of epochs (default: 1)
 *   --output <path>      Output path for LoRA weights (default: lora_weights.bin)
 *
 * The model_dir should contain:
 *   - config.json           (HuggingFace model config)
 *   - generation_config.json
 *   - nntr_config.json      (with lora_rank, lora_alpha, lora_target set)
 */

#include <cstdlib>
#include <iostream>
#include <string>

#include <causal_lm.h>
#include <lora_train.h>
#include <transformer.h>

#include <model.h>

using json = nlohmann::json;

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " <model_dir> <train_data.txt> [--lr <float>] [--epochs <int>]"
                 " [--output <path>]"
              << std::endl;
    return 1;
  }

  std::string model_dir = argv[1];
  std::string train_data_path = argv[2];
  float lr = 1e-4f;
  unsigned int epochs = 1;
  std::string output_path = "lora_weights.bin";

  // Parse optional arguments
  for (int i = 3; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--lr" && i + 1 < argc) {
      lr = std::atof(argv[++i]);
    } else if (arg == "--epochs" && i + 1 < argc) {
      epochs = std::atoi(argv[++i]);
    } else if (arg == "--output" && i + 1 < argc) {
      output_path = argv[++i];
    }
  }

  try {
    // Load configs
    std::string config_path = model_dir + "/config.json";
    std::string gen_config_path = model_dir + "/generation_config.json";
    std::string nntr_config_path = model_dir + "/nntr_config.json";

    auto cfg = causallm::LoadJsonFile(config_path);
    auto gen_cfg = causallm::LoadJsonFile(gen_config_path);
    auto nntr_cfg = causallm::LoadJsonFile(nntr_config_path);

    // Validate LoRA configuration
    unsigned int lora_rank = nntr_cfg.value("lora_rank", 0u);
    if (lora_rank == 0) {
      std::cerr << "Error: lora_rank must be > 0 in nntr_config.json"
                << std::endl;
      return 1;
    }

    std::cout << "=== CausalLM LoRA Training ===" << std::endl;
    std::cout << "Model dir: " << model_dir << std::endl;
    std::cout << "LoRA rank: " << lora_rank << std::endl;
    std::cout << "LoRA alpha: " << nntr_cfg.value("lora_alpha", 0u)
              << std::endl;
    std::cout << "Learning rate: " << lr << std::endl;
    std::cout << "Epochs: " << epochs << std::endl;
    std::cout << "Train data: " << train_data_path << std::endl;
    std::cout << "Output: " << output_path << std::endl;

    // Create CausalLM model
    causallm::CausalLM model(cfg, gen_cfg, nntr_cfg);

    // Initialize for training
    model.initializeForTraining(lr, epochs);

    // Load pre-trained weights
    std::string weight_path =
      model_dir + "/" + nntr_cfg["model_file_name"].get<std::string>();
    std::cout << "Loading weights from: " << weight_path << std::endl;
    model.load_weight(weight_path);

    // Setup training data
    std::string tokenizer_path =
      nntr_cfg["tokenizer_file"].get<std::string>();
    auto tokenizer_blob = causallm::LoadBytesFromFile(tokenizer_path);
    auto tokenizer = tokenizers::Tokenizer::FromBlobJSON(tokenizer_blob);

    unsigned int seq_len = nntr_cfg["init_seq_len"].get<unsigned int>();
    causallm::TrainingDataGenerator data_gen(tokenizer.get(), seq_len);
    data_gen.loadTextFile(train_data_path);

    std::cout << "Training samples: " << data_gen.getNumSamples() << std::endl;

    if (data_gen.getNumSamples() == 0) {
      std::cerr << "Error: Not enough training data (need > seq_len tokens)"
                << std::endl;
      return 1;
    }

    // TODO: Set dataset and run training
    // model->setDataset(...)
    // model->train(...)
    // model->save(output_path)

    std::cout << "LoRA training setup complete." << std::endl;
    std::cout << "Note: Full training loop requires integrating with "
                 "nntrainer's model->train() API."
              << std::endl;

    // Save LoRA weights
    model.save_weight(output_path);
    std::cout << "LoRA weights saved to: " << output_path << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
