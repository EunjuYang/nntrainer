// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    model_config.cpp
 * @date    22 Jan 2026
 * @brief   This is a sample code for internal regitration of model_config.
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Eunju Yang <ej.yang@samsung.com>
 * @bug     No known bugs except for NYI items
 */
#include "causal_lm_api.h"
#include "json.hpp"
#include "model_config_internal.h"
#include <climits>
#include <cstring>

using json = nlohmann::json;

static void register_qwen3_0_6b() {
  // 1. Architecture Config
  ModelArchConfig ac;
  memset(&ac, 0, sizeof(ModelArchConfig));

  ac.vocab_size = 151936;
  ac.hidden_size = 1024;
  ac.intermediate_size = 3072;
  ac.num_hidden_layers = 28;
  ac.num_attention_heads = 16;
  ac.head_dim = 128;
  ac.num_key_value_heads = 8;
  ac.max_position_embeddings = 40960;
  ac.rope_theta = 1000000.0f;
  ac.rms_norm_eps = 1e-06f;
  ac.tie_word_embeddings = false;
  ac.sliding_window = UINT_MAX;
  ac.sliding_window_pattern = 0;
  strncpy(ac.architecture, "Qwen3ForCausalLM", sizeof(ac.architecture) - 1);

  ac.bos_token_id = 151643;
  ac.eos_token_ids[0] = 151645;
  ac.num_eos_token_ids = 1;

  registerModelArchitecture("Qwen3-0.6B-Arch", ac);

  // 2. Runtime Config
  ModelRuntimeConfig rc;
  memset(&rc, 0, sizeof(ModelRuntimeConfig));

  rc.batch_size = 1;
  strncpy(rc.model_type, "CausalLM", sizeof(rc.model_type) - 1);
  strncpy(rc.model_tensor_type, "Q4_0-FP32", sizeof(rc.model_tensor_type) - 1);
  rc.init_seq_len = 1024;
  rc.max_seq_len = 2048;
  rc.num_to_generate = 512;
  rc.fsu = false;
  rc.fsu_lookahead = 2;
  strncpy(rc.embedding_dtype, "Q6_K", sizeof(rc.embedding_dtype) - 1);
  strncpy(rc.fc_layer_dtype, "Q4_0", sizeof(rc.fc_layer_dtype) - 1);
  strncpy(rc.model_file_name, "qwen3-0.6b-q6k-q40-q40-fp32-arm.bin",
          sizeof(rc.model_file_name) - 1);
  strncpy(rc.tokenizer_file, "tokenizer.json", sizeof(rc.tokenizer_file) - 1);
  strncpy(rc.lmhead_dtype, "Q4_0", sizeof(rc.lmhead_dtype) - 1);
  rc.num_bad_word_ids = 0;

  rc.top_k = 20;
  rc.top_p = 0.95f;
  rc.temperature = 0.7f;

  registerModel("Qwen3-0.6B-W4A32", "Qwen3-0.6B-Arch", rc);

  // Example for W32A32 (FP32)
  ModelRuntimeConfig rc_fp32 = rc;
  strncpy(rc_fp32.model_tensor_type, "FP32-FP32",
          sizeof(rc_fp32.model_tensor_type) - 1);
  strncpy(rc_fp32.fc_layer_dtype, "FP32", sizeof(rc_fp32.fc_layer_dtype) - 1);
  strncpy(rc_fp32.embedding_dtype, "FP32", sizeof(rc_fp32.embedding_dtype) - 1);
  strncpy(rc_fp32.lmhead_dtype, "FP32", sizeof(rc_fp32.lmhead_dtype) - 1);
  strncpy(rc_fp32.model_file_name, "qwen3-0.6b-fp32.bin",
          sizeof(rc_fp32.model_file_name) - 1);
  registerModel("Qwen3-0.6B-W32A32", "Qwen3-0.6B-Arch", rc_fp32);

  // Register default alias
  registerModel("Qwen3-0.6B", "Qwen3-0.6B-Arch", rc);
}

/**
 * @brief Register Qwen3-based Embedding model (example)
 * @details This example shows how to register an embedding model with all
 *          configuration embedded in code, including the module pipeline.
 *          No config.json, nntr_config.json, or modules.json files are needed.
 *
 *          Required files in ./models/embedding-qwen3-w16a16/:
 *            - tokenizer.json
 *            - embedding-qwen3-fp16.bin  (weight file)
 */
static void register_embedding_qwen3() {
  // 1. Architecture Config (same as base Qwen3 architecture)
  ModelArchConfig ac;
  memset(&ac, 0, sizeof(ModelArchConfig));

  ac.vocab_size = 151936;
  ac.hidden_size = 1024;
  ac.intermediate_size = 3072;
  ac.num_hidden_layers = 28;
  ac.num_attention_heads = 16;
  ac.head_dim = 128;
  ac.num_key_value_heads = 8;
  ac.max_position_embeddings = 32768;
  ac.rope_theta = 1000000.0f;
  ac.rms_norm_eps = 1e-06f;
  ac.tie_word_embeddings = true;
  ac.sliding_window = UINT_MAX;
  ac.sliding_window_pattern = 0;
  strncpy(ac.architecture, "Qwen3ForCausalLM", sizeof(ac.architecture) - 1);

  ac.bos_token_id = 151643;
  ac.eos_token_ids[0] = 151645;
  ac.num_eos_token_ids = 1;

  registerModelArchitecture("Embedding-Qwen3-Arch", ac);

  // 2. Runtime Config for Embedding
  ModelRuntimeConfig rc;
  memset(&rc, 0, sizeof(ModelRuntimeConfig));

  rc.batch_size = 1;
  strncpy(rc.model_type, "Embedding", sizeof(rc.model_type) - 1);
  strncpy(rc.model_tensor_type, "FP16-FP16",
          sizeof(rc.model_tensor_type) - 1);
  rc.init_seq_len = 512;
  rc.max_seq_len = 8192;
  rc.num_to_generate = 0; // Embedding models don't generate tokens
  rc.fsu = false;
  rc.fsu_lookahead = 0;
  strncpy(rc.embedding_dtype, "FP16", sizeof(rc.embedding_dtype) - 1);
  strncpy(rc.fc_layer_dtype, "FP16", sizeof(rc.fc_layer_dtype) - 1);
  strncpy(rc.model_file_name, "embedding-qwen3-fp16.bin",
          sizeof(rc.model_file_name) - 1);
  strncpy(rc.tokenizer_file, "tokenizer.json",
          sizeof(rc.tokenizer_file) - 1);
  rc.num_bad_word_ids = 0;

  rc.top_k = 0;
  rc.top_p = 0;
  rc.temperature = 0;

  // 3. Inline modules pipeline config (replaces modules.json + module configs)
  json modules = json::array({
    {{"idx", 0}, {"type", "sentence_transformers.models.Transformer"}},
    {{"idx", 1},
     {"type", "sentence_transformers.models.Pooling"},
     {"config",
      {{"word_embedding_dimension", 1024},
       {"pooling_mode_cls_token", false},
       {"pooling_mode_mean_tokens", true},
       {"pooling_mode_max_tokens", false},
       {"pooling_mode_mean_sqrt_len_tokens", false},
       {"pooling_mode_weightedmean_tokens", false},
       {"pooling_mode_lasttoken", false}}}},
    {{"idx", 2}, {"type", "sentence_transformers.models.Normalize"}},
  });
  std::string modules_str = modules.dump();
  strncpy(rc.modules_config, modules_str.c_str(),
          sizeof(rc.modules_config) - 1);

  registerModel("EMBEDDING-QWEN3-W16A16", "Embedding-Qwen3-Arch", rc);

  // Register default alias
  registerModel("EMBEDDING-QWEN3", "Embedding-Qwen3-Arch", rc);
}

int register_builtin_model_configs() {
  register_qwen3_0_6b();
  register_embedding_qwen3();
  // Add more models here...
  return 0;
}
