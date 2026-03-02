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
 * @file    main.cpp
 * @date    February 2026
 * @brief   MeZO (Memory-Efficient Zero-Order) optimization training example on
 * MNIST
 * @see     https://github.com/nntrainer/nntrainer
 */

#include "mnist_loader.h"
#include <chrono>
#include <dataset.h>
#include <iomanip>
#include <iostream>
#include <layer.h>
#include <memory>
#include <model.h>
#include <neuralnet.h>
#include <nntrainer-api-common.h>
#include <optimizer.h>
#include <string>
#include <tensor.h>
#include <util_func.h>
#include <vector>

using LayerHandle = std::shared_ptr<ml::train::Layer>;
using ModelHandle = std::unique_ptr<ml::train::Model>;

std::vector<LayerHandle> createSimpleGraph() {
  using ml::train::createLayer;

  std::vector<LayerHandle> layers;

  // Input layer
  layers.push_back(
    createLayer("input", {nntrainer::withKey("name", "input0"),
                          nntrainer::withKey("input_shape", "1:1:784")}));

  // Hidden layer
  layers.push_back(
    createLayer("fully_connected",
                {nntrainer::withKey("unit", 256),
                 nntrainer::withKey("weight_initializer", "xavier_uniform"),
                 nntrainer::withKey("activation", "relu")}));

  // Output layer
  layers.push_back(
    createLayer("fully_connected",
                {nntrainer::withKey("unit", 10),
                 nntrainer::withKey("weight_initializer", "xavier_uniform")}));

  return layers;
}

int main(int argc, char **argv) {
  std::cout << "=====================================" << std::endl;
  std::cout << "  MeZO Training on MNIST" << std::endl;
  std::cout << "=====================================" << std::endl;

  // Parse arguments
  std::string images_path = "train-images-idx3-ubyte";
  std::string labels_path = "train-labels-idx1-ubyte";
  unsigned int epochs = 10;
  unsigned int batch_size = 64;
  float learning_rate = 0.1f;
  float mezo_epsilon = 0.01f;

  if (argc >= 3) {
    images_path = argv[1];
    labels_path = argv[2];
  }
  if (argc >= 4) {
    epochs = std::stoul(argv[3]);
  }
  if (argc >= 5) {
    batch_size = std::stoul(argv[4]);
  }
  if (argc >= 6) {
    learning_rate = std::stof(argv[5]);
  }
  if (argc >= 7) {
    mezo_epsilon = std::stof(argv[6]);
  }

  std::cout << "\nConfiguration:" << std::endl;
  std::cout << "  Images: " << images_path << std::endl;
  std::cout << "  Labels: " << labels_path << std::endl;
  std::cout << "  Epochs: " << epochs << std::endl;
  std::cout << "  Batch Size: " << batch_size << std::endl;
  std::cout << "  Learning Rate: " << learning_rate << std::endl;
  std::cout << "  MeZO Epsilon: " << mezo_epsilon << std::endl;

  // Load MNIST dataset
  std::vector<float> images, labels;
  if (!mezo::loadMNIST(images_path, labels_path, images, labels, 10)) {
    std::cerr << "Failed to load MNIST dataset" << std::endl;
    return 1;
  }

  size_t num_samples = images.size() / 784; // 28x28 = 784
  std::cout << "Loaded " << num_samples << " samples (normalized)" << std::endl;

  auto model = ml::train::createModel();

  // Add layers
  auto layers = createSimpleGraph();
  for (auto &layer : layers) {
    model->addLayer(layer);
  }

  try {
    // Compile the model
    model->compile(ml::train::ExecutionMode::INFERENCE);
    std::cout << "Model compiled successfully." << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Error compiling model: " << e.what() << std::endl;
    return 1;
  }

  try {
    model->initialize();
    std::cout << "Model initialized successfully. " << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Error initializing model: " << e.what() << std::endl;
    return 1;
  }

  model->summarize(std::cout, ML_TRAIN_SUMMARY_MODEL);

  size_t feature_len = 784;
  size_t label_len = 1; // Class indices instead of one-hot encoding
  size_t num_batches = (num_samples + batch_size - 1) / batch_size;

  std::vector<float *> input_batches;
  std::vector<float *> label_batches;
  std::vector<unsigned int> batch_sizes;
  input_batches.reserve(num_batches);
  label_batches.reserve(num_batches);
  batch_sizes.reserve(num_batches);

  for (size_t b = 0; b < num_batches; ++b) {
    size_t offset = b * batch_size;
    unsigned int cur_batch_size =
      std::min(batch_size, static_cast<unsigned int>(num_samples - offset));
    float *in_ptr = images.data() + offset * feature_len;
    float *lbl_ptr = labels.data() + offset * label_len;
    input_batches.push_back(in_ptr);
    label_batches.push_back(lbl_ptr);
    batch_sizes.push_back(cur_batch_size);
  }

  // Convert ml::train::Model to NeuralNetwork for MeZO training
  auto neural_net = static_cast<nntrainer::NeuralNetwork *>(model.get());

  // Set up training properties for MeZO
  std::vector<std::string> train_properties = {
    "epochs=" + std::to_string(epochs),
    "learning_rate=" + std::to_string(learning_rate),
    "mezo_epsilon=" + std::to_string(mezo_epsilon)};

  std::cout << "\nStarting MeZO training..." << std::endl;

  try {
    // Train using MeZO algorithm with clipped dataset
    auto train_stats = neural_net->trainMeZO(batch_sizes, input_batches,
                                             label_batches, train_properties);

    std::cout << "\nTraining completed!" << std::endl;
    std::cout << "Final loss: " << train_stats.loss << std::endl;
    std::cout << "Epochs completed: " << train_stats.epoch_idx + 1 << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Error during MeZO training: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}