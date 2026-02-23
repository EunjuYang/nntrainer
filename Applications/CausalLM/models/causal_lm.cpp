// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 * Copyright (C) 2025 Seungback Hong <sb92.hong@samsung.com>
 * Copyright (C) 2025 Hyeonseok Lee <hs89.lee@samsung.com>
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   causal_lm.cpp
 * @date   10 July 2025
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This file defines CausalLM's basic actions
 * @note   This causal_lm.h constructs a class for Transformer-based Causal
 * Language Model (CausalLM). It aims to support AutoModelForCausalLM with
 * nntrainer. It supports the following models:
 *          - Llama
 */

#include <algorithm>
#include <app_context.h>
#include <cmath>
#include <engine.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

#include <common.h>
#include <layer_context.h>
#include <lm_head.h>
#include <mha_core.h>
#include <tensor.h>

#include <causal_lm.h>
#include <llm_util.hpp>

namespace causallm {

CausalLM::CausalLM(json &cfg, json &generation_cfg, json &nntr_cfg) :
  Transformer(cfg, generation_cfg, nntr_cfg, ModelType::CAUSALLM) {
  setupParameters(cfg, generation_cfg, nntr_cfg);
}

void CausalLM::setupParameters(json &cfg, json &generation_cfg,
                               json &nntr_cfg) {
  // Initialize output list
  for (unsigned int i = 0; i < BATCH_SIZE; ++i)
    output_list.push_back("");

  // allocate memory for the internal buffer
  ids_history = (unsigned int *)malloc(static_cast<size_t>(BATCH_SIZE) *
                                       MAX_SEQ_LEN * sizeof(unsigned int));

  BAD_WORD_IDS = nntr_cfg["bad_word_ids"].get<std::vector<unsigned int>>();
  NUM_BADWORDS = BAD_WORD_IDS.size();

  LMHEAD_DTYPE = nntr_cfg.contains("lmhead_dtype")
                   ? nntr_cfg["lmhead_dtype"]
                   : nntr_cfg["embedding_dtype"];

  USE_KVCACHE = false;
  PRE_COMPUTED_CACHE_PATH = "";
  SYS_PROMP_LEN = 0;

  if (nntr_cfg.contains("system_prompt") &&
      nntr_cfg["system_prompt"].contains("kvcache")) {
    USE_KVCACHE = true;
    PRE_COMPUTED_CACHE_PATH =
      nntr_cfg["system_prompt"]["kvcache"]["pre_computed_cache_path"];
    if (nntr_cfg["system_prompt"]["kvcache"].contains("sys_prompt_token_size"))
      SYS_PROMP_LEN =
        nntr_cfg["system_prompt"]["kvcache"]["sys_prompt_token_size"]
          .get<unsigned int>();
  }

  if (generation_cfg["eos_token_id"].is_array()) {
    EOS_TOKEN_ID =
      generation_cfg["eos_token_id"].empty()
        ? cfg["eos_token_id"].get<std::vector<unsigned int>>()
        : generation_cfg["eos_token_id"].get<std::vector<unsigned int>>();
  } else {
    EOS_TOKEN_ID.clear();
    EOS_TOKEN_ID.push_back(generation_cfg["eos_token_id"].get<unsigned int>());
  }
  BOS_TOKEN_ID = generation_cfg["bos_token_id"].empty()
                   ? cfg["bos_token_id"].get<unsigned int>()
                   : generation_cfg["bos_token_id"].get<unsigned int>();
  TOP_K = generation_cfg.contains("top_k")
            ? generation_cfg["top_k"].get<unsigned int>()
            : 20;
  TOP_P = generation_cfg.contains("top_p")
            ? generation_cfg["top_p"].get<float>()
            : 0.95;
  TEMPERATURE = generation_cfg.contains("temperature")
                  ? generation_cfg["temperature"].get<float>()
                  : 0.7;
  global_token_len = 0;
}

void CausalLM::constructModel() {

  // It adds all transformer model's block to model
  Transformer::constructModel();

  const std::string lmhead_type =
    TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "lm_head";

  // add lmhead
  std::vector<std::string> lmhead_prop = {
    withKey("name", "output_of_causallm"),
    withKey("unit", NUM_VOCAB),
    withKey("disable_bias", "true"),
    withKey("input_layers", "output_norm"),
    withKey("weight_dtype", LMHEAD_DTYPE),
  };

  if (TIE_WORD_EMBEDDINGS)
    lmhead_prop.emplace_back(withKey("shared_from", "embedding0"));

  model->addLayer(createLayer(lmhead_type, lmhead_prop));
}

void CausalLM::registerOutputs(
  std::unique_ptr<tokenizers::Tokenizer> &tokenizer,
  std::vector<unsigned int> ids, unsigned int pos,
  const std::vector<bool> &eos_list) {

  static const std::vector<char> puncts{',', '!', ':', ';', '?'};
  for (size_t b = 0; b < ids.size(); ++b) {
    if (!eos_list[b]) {
      pending_ids_.push_back(static_cast<int>(ids[b]));
      ids_history[b * MAX_SEQ_LEN + pos] = ids[b];
      std::string decoded_str = tokenizer->Decode(pending_ids_);

      if (std::find(puncts.begin(), puncts.end(), decoded_str.back()) !=
          puncts.end()) {
        // last symbol is a punctuation, hold on
      } else if (decoded_str.size() >= 3 &&
                 decoded_str.compare(decoded_str.size() - 3, 3, "") == 0) {
        // ends with an incomplete token, hold on
      } else {
#if defined(_WIN32)
        std::wcout << L"" << utf8_to_wstring(decoded_str);
        std::wcout.flush();
#else
        std::cout << decoded_str;
        std::cout.flush();
#endif
        output_list[b].append(decoded_str);
        pending_ids_.clear();
      }
    }
  }
}

/**
 * @brief Save the Key-Value cache for every MHACore layer to a binary file.
 *
 * Only the first `to_` sequence positions are serialised so that the file
 * stores exactly the system-prompt slice of each layer's K/V tensors.
 * The on-disk layout is:
 *
 *   [ layer_0 K (to_ positions) ][ layer_0 V (to_ positions) ]
 *   [ layer_1 K (to_ positions) ][ layer_1 V (to_ positions) ]
 *   ...
 *
 * @param path  Destination file path.
 * @param to_   Number of sequence positions (i.e. system-prompt token count)
 *              to save per layer.  Must be <= the allocated height of the
 *              K/V cache tensors.
 */
void CausalLM::save_kvcache(std::string path, int to_) {
  auto f = nntrainer::checkedOpenStream<std::ofstream>(
    path, std::ios::out | std::ios::binary | std::ios::trunc);

  // Callback invoked for every layer in the model.
  // We only act on MHACore layers because those own the K/V cache tensors.
  std::function<void(ml::train::Layer &, nntrainer::RunLayerContext &, void *)>
    fn = [&f](ml::train::Layer &l, nntrainer::RunLayerContext &context,
              void *idx) {
      if (l.getType() == causallm::MHACoreLayer::type) {
        // `idx` carries to_ as an intptr_t to satisfy the void* callback API.
        int to = static_cast<int>(reinterpret_cast<intptr_t>(idx));

        // getTensor(0) → K cache,  getTensor(1) → V cache.
        auto k_cache = context.getTensor(0);
        auto v_cache = context.getTensor(1);

        // Create a view that covers only the first `to` rows (sequence
        // positions) so we do not serialise uninitialised tail memory.
        ml::train::TensorDim k_dim = k_cache.getDim();
        ml::train::TensorDim v_dim = v_cache.getDim();
        k_dim.height(to);
        v_dim.height(to);
        nntrainer::Tensor k_cache_prompt =
          k_cache.getSharedDataTensor(k_dim, 0, true);
        nntrainer::Tensor v_cache_prompt =
          v_cache.getSharedDataTensor(v_dim, 0, true);

        k_cache_prompt.save(f);
        v_cache_prompt.save(f);
      }
    };

  void *arg = reinterpret_cast<void *>(static_cast<intptr_t>(to_));
  model->forEachLayer(fn, arg);
  f.close();
}

/**
 * @brief Load a previously saved Key-Value cache from a binary file.
 *
 * Restores the first `to_` sequence positions of each MHACore layer's K/V
 * tensors from the file produced by save_kvcache().  After loading, the KV
 * cache behaves as if the system prompt has already been processed: the model
 * can continue inference from position `to_` without re-running the system
 * prompt through the network.
 *
 * @param path  Source file path (produced by save_kvcache).
 * @param to_   Number of sequence positions to restore per layer.  Must match
 *              the value used when the cache was saved.
 */
void CausalLM::load_kvcache(std::string path, int to_) {
  auto f = nntrainer::checkedOpenStream<std::ifstream>(
    path, std::ios::in | std::ios::binary);

  // Allocate model memory before writing into the KV cache tensors.
  model->allocate(ml::train::ExecutionMode::INFERENCE);

  // Callback invoked for every layer in the model.
  std::function<void(ml::train::Layer &, nntrainer::RunLayerContext &, void *)>
    fn = [&f](ml::train::Layer &l, nntrainer::RunLayerContext &context,
              void *idx) {
      if (l.getType() == causallm::MHACoreLayer::type) {
        // getTensor(0) → K cache,  getTensor(1) → V cache.
        auto k_cache = context.getTensor(0);
        auto v_cache = context.getTensor(1);

        int to = static_cast<int>(reinterpret_cast<intptr_t>(idx));

        // Build a view covering only the first `to` rows so that we read
        // exactly the bytes written by save_kvcache().
        ml::train::TensorDim k_dim = k_cache.getDim();
        ml::train::TensorDim v_dim = v_cache.getDim();
        k_dim.height(to);
        v_dim.height(to);
        nntrainer::Tensor k_cache_prompt =
          k_cache.getSharedDataTensor(k_dim, 0, true);
        nntrainer::Tensor v_cache_prompt =
          v_cache.getSharedDataTensor(v_dim, 0, true);

        k_cache_prompt.read(f);
        v_cache_prompt.read(f);
      }
    };

  void *arg = reinterpret_cast<void *>(static_cast<intptr_t>(to_));
  model->forEachLayer(fn, arg);
  f.close();
}

std::vector<unsigned int> CausalLM::generate(float *logits, bool do_sample,
                                             float repetition_penalty,
                                             unsigned int *input_ids,
                                             unsigned int NUM_INPUT_IDS) {

  std::vector<unsigned int> outputs;
  for (unsigned int iteration = 0; iteration < BATCH_SIZE; ++iteration) {

    // apply repetition penalty
    if (repetition_penalty != 1 && input_ids != nullptr && NUM_INPUT_IDS != 0) {
      applyRepetitionPenalty(logits, input_ids, NUM_INPUT_IDS,
                             repetition_penalty);
    }

    // apply bad words penalty
    if (BAD_WORD_IDS.size() != 0 && NUM_BADWORDS != 0) {
      applyBadWordsPenalty(logits, BAD_WORD_IDS.data(), NUM_BADWORDS);
    }

    // return argmax if do_sample is false
    if (do_sample == false) {
      unsigned int argmax_idx =
        std::distance(logits, std::max_element(logits, logits + NUM_VOCAB));
      outputs.push_back(argmax_idx);
    } else {
      // apply temperature & top-k & top-p to logits
      float max_logits = applyTKP(logits, NUM_VOCAB, TEMPERATURE, TOP_K, TOP_P);
      // transform logits to softmax
      float sum_exp_logits = 0;
      for (unsigned int i = 0; i < NUM_VOCAB; i++) {
        float exp_x = exp(logits[i] - max_logits);
        sum_exp_logits += exp_x;
        logits[i] = exp_x;
      }

      for (unsigned int i = 0; i < NUM_VOCAB; ++i) {
        logits[i] /= sum_exp_logits;
      }

      // sample from final logits
      std::discrete_distribution<int> dist(logits, logits + NUM_VOCAB);
      unsigned int sampled_idx = dist(rng);

      // add sampled word
      outputs.push_back(sampled_idx);
    }

    // set batch offset
    logits = logits + NUM_VOCAB;
    input_ids = input_ids + MAX_SEQ_LEN;
  }

  return outputs;
};

void CausalLM::registerCustomLayers() {
  Transformer::registerCustomLayers();
  const auto &ct_engine = nntrainer::Engine::Global();
  const auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));
  try {
    app_context->registerFactory(nntrainer::createLayer<causallm::LmHeadLayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
  }
}

void CausalLM::run(const WSTR prompt, bool do_sample, const WSTR system_prompt,
                   const WSTR tail_prompt) {

  if (!is_initialized) {
    throw std::runtime_error("CausalLM model is not initialized. Please call "
                             "initialize() before run().");
  }

  output_list.clear();
  for (unsigned int b = 0; b < BATCH_SIZE; ++b) {
    output_list.push_back("");
  }

  if (MAX_SEQ_LEN < INIT_SEQ_LEN) {
    throw std::invalid_argument(
      "MAX_SEQ_LEN must be greater than or equal to INIT_SEQ_LEN");
  }

  /**
   * Variables for Log
   */
  unsigned int generation_cnt = 0;
  int64_t total_generation_duration = 0;

  // -------------------------------------------------------------------------
  // INPUT PREPARATION
  // -------------------------------------------------------------------------
  //
  // KV-cache token layout across the full MAX_SEQ_LEN context window:
  //
  //  Case A – no KV cache (USE_KVCACHE == false):
  //
  //   KV cache positions:
  //   | 0 ............. init_len-1 | init_len ... init_len+NUM_TO_GENERATE-1 |
  //   |<---- prefill (full prompt) --->|<--------- decoded tokens ---------->|
  //
  //  Case B – KV cache SAVE mode (SAVE_KVCACHE == true):
  //   Only the system-prompt is processed and saved; run() returns early.
  //
  //   KV cache positions:
  //   | 0 ........... SYS_PROMP_LEN-1 |
  //   |<---- system prompt (saved) --->|
  //
  //  Case C – KV cache LOAD mode (USE_KVCACHE == true, SAVE_KVCACHE == false):
  //
  //   KV cache positions:
  //   | 0 ......... SYS_PROMP_LEN-1 | SYS_PROMP_LEN .. SYS_PROMP_LEN+init_len-1 | ... decoded ... |
  //   |<-- system prompt (loaded) -->|<---------- user prompt (prefill) --------->|<--- decoded --->|
  //
  //   For multi-turn conversations global_token_len accumulates the tokens
  //   from all previous turns (user prompts + decoded tokens, excluding the
  //   system prompt).  New turns are appended at position:
  //     SYS_PROMP_LEN + global_token_len
  // -------------------------------------------------------------------------

  std::vector<float *> input;
  std::vector<float *> label;

  // Decide whether this call should save the KV cache rather than run normal
  // inference.  We enter save mode when:
  //   - KV-cache feature is enabled (USE_KVCACHE),
  //   - a system_prompt was provided (non-empty),
  //   - the cache file does not yet exist on disk.
  SAVE_KVCACHE = (USE_KVCACHE && system_prompt != "" &&
                  !std::filesystem::exists(PRE_COMPUTED_CACHE_PATH));

#if defined(_WIN32)
  // Print the full input text: system prompt + user prompt.
  // Mirror the non-Windows branch which also prints tail_prompt.
  std::wcout << system_prompt << prompt << tail_prompt << std::endl;

  // In SAVE_KVCACHE mode tokenise only the system_prompt (the user prompt
  // will be handled in the next call after the cache is available).
  std::wstring prompt_ = prompt;
  if (!SAVE_KVCACHE)
    prompt_ += tail_prompt;
  std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
  auto _input = tokenizer->Encode(converter.to_bytes(prompt_));
#else
  // Print the full text that will be processed in this call.
  std::cout << system_prompt << prompt << tail_prompt << std::endl;

  // Build the string that will actually be tokenised and fed to the model.
  //   SAVE mode : tokenise only the system_prompt so we can compute and
  //               persist the system-prompt KV cache.
  //   LOAD mode : tokenise only the user prompt + tail; the system-prompt
  //               positions in the KV cache are restored from disk.
  //   No cache  : tokenise the full concatenation of all parts.
  std::string prompt_;

  if (USE_KVCACHE) {
    prompt_ = SAVE_KVCACHE ? system_prompt : (prompt + tail_prompt);
  } else {
    prompt_ = system_prompt + prompt + tail_prompt;
  }

  // If SYS_PROMP_LEN was not supplied in the config, derive it by tokenising
  // the system_prompt string (only needed once per session when loading cache).
  if (USE_KVCACHE && !SAVE_KVCACHE && SYS_PROMP_LEN == 0)
    SYS_PROMP_LEN = tokenizer->Encode(system_prompt).size();

  auto _input = tokenizer->Encode(prompt_);
  ///@note BOS token insertion is intentionally disabled.
  // _input.insert(_input.begin(), BOS_TOKEN_ID);
#endif

  // Clamp the tokenised input to the budget available for user text:
  //   budget = MAX_SEQ_LEN - NUM_TO_GENERATE
  // Tokens beyond this limit are silently dropped.
  // Note: this budget does not account for SYS_PROMP_LEN, so callers should
  // ensure SYS_PROMP_LEN + user_tokens + NUM_TO_GENERATE <= MAX_SEQ_LEN.
  ///@todo Enforce the SYS_PROMP_LEN-aware budget here.
  std::vector<int64_t> init_input;
  unsigned int _len = _input.size();
  unsigned int num_allow_str = MAX_SEQ_LEN - NUM_TO_GENERATE;
  unsigned int text_len = (_len > num_allow_str) ? num_allow_str : _len;

  for (unsigned int i = 0; i < text_len; ++i)
    init_input.push_back(_input[i]);

  _input.clear(); // raw token list no longer needed

  // init_len: number of input tokens the model will actually see during
  // prefill (may be less than the original prompt length after clamping).
  unsigned int init_len = init_input.size();

  // input_sample: flat buffer [ batch_0_tokens | batch_1_tokens | ... ]
  // Each batch slot is MAX_SEQ_LEN floats wide.
  //   Prefill  → slots [0, init_len) hold the prompt token IDs.
  //   Decoding → slot [0] is overwritten with the latest generated token;
  //              the model uses start_pos / end_pos to locate it in the
  //              sequence without re-reading the full buffer every step.
  float *input_sample =
    (float *)malloc(sizeof(float) * BATCH_SIZE * MAX_SEQ_LEN);
  std::vector<bool> eos_list(BATCH_SIZE, false);

  // input_len tracks the total number of sequence positions the model has
  // already populated in the KV cache.  It starts equal to init_len and
  // grows by SYS_PROMP_LEN after prefill (see below).
  unsigned int input_len = init_len;

  // Populate input_sample and ids_history with the prompt token IDs.
  // ids_history is a history buffer used for repetition-penalty lookups.
  for (unsigned int b = 0; b < BATCH_SIZE; ++b) {
    for (unsigned int i = 0; i < input_len; ++i) {
      input_sample[static_cast<size_t>(b) * MAX_SEQ_LEN + i] =
        static_cast<float>(init_input[i]);
      ids_history[static_cast<size_t>(b) * MAX_SEQ_LEN + i] = init_input[i];
    }
  }

  // -------------------------------------------------------------------------
  // PREFILL
  // -------------------------------------------------------------------------
  // Feed all prompt tokens into the model in a single forward pass.
  // The model fills the KV cache for every prompt position and returns
  // logits for the last position, which we use to predict the first
  // generated token.

  input.push_back(input_sample);

  auto start_prefill = std::chrono::high_resolution_clock::now();

  std::vector<float *> output;

  if (SAVE_KVCACHE) {
    // -----------------------------------------------------------------------
    // KV-CACHE SAVE MODE
    // -----------------------------------------------------------------------
    // Run the model on the system prompt only.  The resulting KV cache
    // (positions [0, init_len)) is serialised to disk so future calls can
    // skip this step entirely.
    //
    // incremental_inference arguments:
    //   seq_len   = input_len        (number of tokens in the system prompt)
    //   start_pos = global_token_len (always 0 for the first save call)
    //   end_pos   = input_len + global_token_len
    //   last_only = false            (we need all position outputs to fill KV)
    std::cout << "\n==============[KV CACHE SAVE MODE]================\n";
    output = model->incremental_inference(BATCH_SIZE, input, label, input_len,
                                          global_token_len,
                                          input_len + global_token_len, false);

    SYS_PROMP_LEN = input_len;
    save_kvcache(PRE_COMPUTED_CACHE_PATH, SYS_PROMP_LEN);

    std::cout
      << "kv caches are saved in " << PRE_COMPUTED_CACHE_PATH << std::endl
      << "and the size of prompt is " << SYS_PROMP_LEN << ".\n"
      << "You may need this prompt length to set the \"sys_prompt_token_size\""
      << "\n==================================================\n"
      << std::endl;
    return; // Nothing more to do; next call will enter LOAD mode.
  }

  if (USE_KVCACHE) {
    // -----------------------------------------------------------------------
    // KV-CACHE LOAD MODE
    // -----------------------------------------------------------------------
    // Restore the pre-computed system-prompt KV cache from disk.
    // After this call, KV-cache positions [0, SYS_PROMP_LEN) are populated
    // as if the model just processed the system prompt.
    load_kvcache(PRE_COMPUTED_CACHE_PATH, SYS_PROMP_LEN);
  } else {
    // No KV-cache: treat the full concatenated prompt as a single prefill.
    SYS_PROMP_LEN = 0;
  }

  // Run the model over the user prompt (init_len tokens).
  //
  // incremental_inference arguments:
  //   seq_len   = init_len                      (user-prompt token count)
  //   start_pos = SYS_PROMP_LEN + global_token_len
  //               ├─ SYS_PROMP_LEN: skip the pre-filled system-prompt slots
  //               └─ global_token_len: skip tokens from previous turns
  //   end_pos   = SYS_PROMP_LEN + init_len + global_token_len
  //   last_only = false (fill KV for all positions; last logit predicts tok_0)
  output = model->incremental_inference(BATCH_SIZE, input, label, init_len,
                                        SYS_PROMP_LEN + global_token_len,
                                        SYS_PROMP_LEN + input_len +
                                          global_token_len,
                                        false);

  // Extract the first generated token from the prefill logits.
  // Pass init_len (tokens actually written to ids_history) rather than _len
  // (the pre-clamp size) so the repetition-penalty window is correct.
  std::vector<unsigned int> id_list(generate_multi_tokens(
    output[0], NUM_VOCAB, BATCH_SIZE, 1, ids_history, init_len));

  // Emit the token predicted after prefill (position init_len in the sequence)
  // unless the prompt filled the entire prefill budget exactly.
  if (init_len < INIT_SEQ_LEN)
    registerOutputs(tokenizer, id_list, init_len, eos_list);

  auto finish_prefill = std::chrono::high_resolution_clock::now();
  auto prefill_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
    finish_prefill - start_prefill);

  // -------------------------------------------------------------------------
  // AUTO-REGRESSIVE DECODING LOOP
  // -------------------------------------------------------------------------
  // After prefill we enter single-step decoding: one forward pass per token.
  //
  // input_len is extended to cover the system-prompt positions that the KV
  // cache already holds.  This value is passed as seq_len to
  // incremental_inference so the attention layer knows the full KV context
  // size (system prompt + user prompt).
  input_len += SYS_PROMP_LEN;

  // Place the first generated token (from prefill) into slot [0] of every
  // batch entry in input_sample.  During decoding the model always reads
  // the single new token from slot [0] of the buffer; the KV cache provides
  // the context for all earlier positions.
  for (unsigned int b = 0; b < BATCH_SIZE; ++b)
    input_sample[static_cast<size_t>(b) * MAX_SEQ_LEN] =
      static_cast<float>(id_list[b]);

  auto start_generation = std::chrono::high_resolution_clock::now();

  // token_generation_idx is the position in the full KV-cache sequence that
  // the new token occupies (1-based relative to input_len):
  //   iter 1: position = input_len + 1  → decoding step 1
  //   iter 2: position = input_len + 2  → decoding step 2
  //   ...
  // global_token_len shifts all positions to account for tokens already in
  // the KV cache from previous multi-turn calls.
  for (unsigned int token_generation_idx = input_len + 1;
       token_generation_idx < input_len + 1 + NUM_TO_GENERATE;
       ++token_generation_idx) {

    // Single-step inference.
    //
    // incremental_inference arguments:
    //   seq_len   = input_len                           (full KV context size)
    //   start_pos = token_generation_idx - 1 + global_token_len
    //               (position of the token we just placed in input_sample[0])
    //   end_pos   = token_generation_idx + global_token_len
    //               (position after the new token — the model predicts this)
    //   last_only = true (default): return only the logits for the last pos
    auto output_interval =
      model->incremental_inference(BATCH_SIZE, input, label, input_len,
                                   token_generation_idx - 1 + global_token_len,
                                   token_generation_idx + global_token_len);

    // Sample (or greedily pick) the next token from the output logits.
    std::vector<unsigned int> ids_list(generate(output_interval[0], do_sample));

    // Feed the newly sampled token back as the input for the next step.
    for (unsigned int b = 0; b < BATCH_SIZE; ++b) {
      input_sample[static_cast<size_t>(b) * MAX_SEQ_LEN] =
        static_cast<float>(ids_list[b]);
    }
    registerOutputs(tokenizer, ids_list, token_generation_idx, eos_list);
    ++generation_cnt;

    // Mark any batch entry that produced an EOS token as finished.
    for (unsigned int j = 0; j < BATCH_SIZE; ++j) {
      if (!eos_list[j] && (std::find(EOS_TOKEN_ID.begin(), EOS_TOKEN_ID.end(),
                                     ids_list[j]) != EOS_TOKEN_ID.end())) {
        eos_list[j] = true;
      }
    }

    // Stop when every batch entry has emitted an EOS token.
    bool is_finish = true;
    for (unsigned int j = 0; j < BATCH_SIZE; ++j) {
      if (!eos_list[j]) {
        is_finish = false;
        break;
      }
    }

    if (is_finish) {
      break;
    }
  }

  // Always release the input buffer after the generation loop, whether
  // the loop exited early (EOS found) or ran to the maximum token limit.
  free(input_sample);

  // -------------------------------------------------------------------------
  // MULTI-TURN CONTEXT UPDATE
  // -------------------------------------------------------------------------
  // global_token_len accumulates the number of tokens processed in previous
  // turns of a multi-turn conversation, *excluding* the system prompt.
  // It is used as an offset in all subsequent calls to incremental_inference
  // so that new turns are appended at the correct KV-cache position:
  //   next_start_pos = SYS_PROMP_LEN + global_token_len
  //
  // We add init_len (user-prompt tokens) and generation_cnt (decoded tokens)
  // but NOT SYS_PROMP_LEN, because the system-prompt offset is applied
  // separately via the SYS_PROMP_LEN term in each incremental_inference call.
  global_token_len += (generation_cnt + init_len);

  auto finish_generation = std::chrono::high_resolution_clock::now();
  auto generation_duration =
    std::chrono::duration_cast<std::chrono::milliseconds>(finish_generation -
                                                          start_generation);

  std::cout << "\n\n";
  std::cout << "=================[ LLM with NNTrainer ]===================\n";
  std::cout << "prefill: " << init_len << " tokens, "
            << prefill_duration.count() << " ms, "
            << ((double)init_len / prefill_duration.count() * 1000) << " TPS\n";
  std::cout << "generation: " << generation_cnt << " tokens, "
            << generation_duration.count() << " ms, "
            << ((double)generation_cnt / generation_duration.count() * 1000)
            << " TPS\n";
  std::cout << "==========================================================\n";
};

} // namespace causallm
