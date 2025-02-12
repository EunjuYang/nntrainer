// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   main.cpp
 * @date   10 Dec 2024
 * @brief  Test Application for Asynch FSU
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */
#include <array>
#include <chrono>
#include <ctime>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include <layer.h>
#include <model.h>
#include <optimizer.h>
#include <unistd.h>

#ifdef PROFILE
#include <profiler.h>
#endif

using LayerHandle = std::shared_ptr<ml::train::Layer>;
using ModelHandle = std::unique_ptr<ml::train::Model>;

/**
 * @brief make "key=value" from key and value
 *
 * @tparam T type of a value
 * @param key key
 * @param value value
 * @return std::string with "key=value"
 */
template <typename T>
static std::string withKey(const std::string &key, const T &value) {
  std::stringstream ss;
  ss << key << "=" << value;
  return ss.str();
}

template <typename T>
static std::string withKey(const std::string &key,
                           std::initializer_list<T> value) {
  if (std::empty(value)) {
    throw std::invalid_argument("empty data cannot be converted");
  }

  std::stringstream ss;
  ss << key << "=";

  auto iter = value.begin();
  for (; iter != value.end() - 1; ++iter) {
    ss << *iter << ',';
  }
  ss << *iter;

  return ss.str();
}

/**
 * @brief Create network
 *
 * @return vector of layers that contain full graph of asynch
 */
std::vector<LayerHandle> createGraph() {
  using ml::train::createLayer;

  std::vector<LayerHandle> layers;

  layers.push_back(createLayer(
    "input", {withKey("name", "input0"), withKey("input_shape", "1:1:1024")}));

  for (int i = 0; i < 100; i++) {
    layers.push_back(createLayer(
      "fully_connected",
      {withKey("unit", 1024), withKey("weight_initializer", "xavier_uniform"),
       withKey("disable_bias", "true")}));
  }

  return layers;
}

ModelHandle create() {
  ModelHandle model = ml::train::createModel(ml::train::ModelType::NEURAL_NET,
                                             {withKey("loss", "mse")});

  for (auto &layer : createGraph()) {
    model->addLayer(layer);
  }

  return model;
}

double createAndRun(unsigned int batch_size, std::string model_type,
                    std::string swap_on_off, std::string look_ahaed) {

  // setup model
  ModelHandle model = create();
  model->setProperty({withKey("batch_size", batch_size),
                      withKey("model_tensor_type", model_type)});
  model->setProperty({withKey("memory_swap", swap_on_off)});
  if (swap_on_off == "true") {
    model->setProperty({withKey("memory_swap_lookahead", look_ahaed)});
  }

  if (model->compile(ml::train::ExecutionMode::INFERENCE)) {
    throw std::invalid_argument("model compilation failed!");
  }

  if (model->initialize(ml::train::ExecutionMode::INFERENCE)) {
    throw std::invalid_argument("model initialization failed!");
  }

  const unsigned int feature_size = 1 * 1 * 1024;
  float input[feature_size];

  for (unsigned int j = 0; j < feature_size; ++j)
    input[j] = (j / (float)feature_size);

  std::vector<float *> in;
  std::vector<float *> answer;

  in.push_back(input);

  // to test asynch fsu, we do need save the model weight data in file
  const std::string filePath = "./simplefc_weight_" + model_type + ".bin";
  if (access(filePath.c_str(), F_OK) != 0)
    model->save(filePath, ml::train::ModelFormat::MODEL_FORMAT_BIN);

  ///////////////////////////////////////////////////
  auto start = std::chrono::system_clock::now();
  model->load(filePath);
  answer = model->inference(1, in);
  auto end = std::chrono::system_clock::now();
  ///////////////////////////////////////////////////
  in.clear();

  return std::chrono::duration<double>(end - start).count();
}

int main(int argc, char *argv[]) {

#ifdef PROFILE
  auto listener =
    std::make_shared<nntrainer::profile::GenericProfileListener>();
  nntrainer::profile::Profiler::Global().subscribe(listener);
#endif

  std::string swap_on = "true";
  std::string look_ahead = "1";
  std::string weight_dtype = "FP16";
  std::string tensor_dtype = "FP16";
  unsigned int num_tests = 1;

  if (argc > 1)
    num_tests = std::stoi(argv[1]);
  if (argc > 2)
    swap_on = argv[2]; 
  if (argc > 3)
    look_ahead = argv[3]; 
  if (argc > 4)
    weight_dtype = argv[4];
  if (argc > 5)
    tensor_dtype = argv[5];

  std::string model_type = weight_dtype + "-" + tensor_dtype;

  std::cout << "============================" << std::endl;
  std::cout << "num_tests : " << num_tests << std::endl;
  std::cout << "swap_on : " << swap_on << std::endl;
  std::cout << "look_ahead : " << look_ahead << std::endl;
  std::cout << "model_type : " << model_type << std::endl;
  std::cout << "============================" << std::endl;

  unsigned int batch_size = 1;
  unsigned int epoch = 1;
  double time = 0;

  try {
    for (unsigned int i = 0; i < num_tests; ++i)
      time += createAndRun(batch_size, model_type, swap_on, look_ahead);
    std::cout << time / num_tests << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "uncaught error while running! details: " << e.what()
              << std::endl;
    return EXIT_FAILURE;
  }

#ifdef PROFILE
  std::cout << *listener;
#endif

  return EXIT_SUCCESS;
}
