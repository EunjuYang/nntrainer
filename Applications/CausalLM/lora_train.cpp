// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   lora_train.cpp
 * @date   01 Apr 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  LoRA training data pipeline implementation
 */

#include "lora_train.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace causallm {

TrainingDataGenerator::TrainingDataGenerator(tokenizers::Tokenizer *tokenizer,
                                             unsigned int seq_len) :
  tokenizer_(tokenizer),
  seq_len_(seq_len),
  current_idx_(0) {}

void TrainingDataGenerator::loadTextFile(const std::string &path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open training data file: " + path);
  }

  std::stringstream buffer;
  buffer << file.rdbuf();
  std::string text = buffer.str();

  // Tokenize the text
  auto ids = tokenizer_->Encode(text);
  all_token_ids_.insert(all_token_ids_.end(), ids.begin(), ids.end());

  std::cout << "[TrainingData] Loaded " << path << ": " << ids.size()
            << " tokens, total: " << all_token_ids_.size() << std::endl;
}

void TrainingDataGenerator::addTokenIds(const std::vector<int> &ids) {
  all_token_ids_.insert(all_token_ids_.end(), ids.begin(), ids.end());
}

unsigned int TrainingDataGenerator::getNumSamples() const {
  if (all_token_ids_.size() <= seq_len_) {
    return 0;
  }
  return static_cast<unsigned int>(all_token_ids_.size() - seq_len_);
}

void TrainingDataGenerator::reset() { current_idx_ = 0; }

int TrainingDataGenerator::dataCb(float **input, float **label, bool *last,
                                  void *user_data) {
  auto *self = static_cast<TrainingDataGenerator *>(user_data);

  if (self->current_idx_ + self->seq_len_ >= self->all_token_ids_.size()) {
    *last = true;
    self->reset();
    return 0;
  }

  // Input: [t_i, t_{i+1}, ..., t_{i+seq_len-1}]
  // Label: [t_{i+1}, t_{i+2}, ..., t_{i+seq_len}]
  for (unsigned int j = 0; j < self->seq_len_; j++) {
    input[0][j] =
      static_cast<float>(self->all_token_ids_[self->current_idx_ + j]);
    label[0][j] =
      static_cast<float>(self->all_token_ids_[self->current_idx_ + j + 1]);
  }

  self->current_idx_++;
  *last = false;
  return 0;
}

} // namespace causallm
