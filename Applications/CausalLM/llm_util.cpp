// SPDX-License-Identifier: Apache-2.0
/**
 *
 * @file   llm_util.cpp
 * @brief  util functions for llm (refactored from main.cpp)
 * @date   21 August 2024
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <limits>
#include <llm_util.hpp>
#include <unordered_set>
#include <utility>

std::vector<unsigned int> generate_multi_tokens(
  float *logits, unsigned int NUM_VOCAB, unsigned int NUM_TARGET_TOKENS,
  float repetition_penalty, unsigned int *input_ids, unsigned int NUM_INPUT_IDS,
  unsigned int *bad_words_ids, unsigned int NUM_BAD_WORDS_IDS) {

  std::vector<unsigned int> outputs;

  // apply repetition penalty
  if (repetition_penalty != 1 && input_ids != nullptr && NUM_INPUT_IDS != 0) {
    applyRepetitionPenalty(logits, input_ids, NUM_INPUT_IDS,
                           repetition_penalty);
  }

  // apply bad words penalty
  if (bad_words_ids != nullptr && NUM_BAD_WORDS_IDS != 0)
    applyBadWordsPenalty(logits, bad_words_ids, NUM_BAD_WORDS_IDS);

  if (NUM_TARGET_TOKENS == 0 || NUM_VOCAB == 0)
    return outputs;

  // Fast path for NUM_TARGET_TOKENS == 1 (BATCH_SIZE == 1 is the typical
  // deployment): just argmax. Avoids a ~NUM_VOCAB * sizeof(pair) allocation
  // per prefill call (~1.2 MB for vocab_size=151936).
  if (NUM_TARGET_TOKENS == 1) {
    unsigned int argmax_idx = static_cast<unsigned int>(std::distance(
      logits, std::max_element(logits, logits + NUM_VOCAB)));
    outputs.push_back(argmax_idx);
    return outputs;
  }

  // General path: bounded min-heap over (logit, index) of size
  // NUM_TARGET_TOKENS -- O(N log k) with heap memory ~k pairs instead of N.
  const unsigned int k = std::min(NUM_TARGET_TOKENS, NUM_VOCAB);

  std::vector<std::pair<float, unsigned int>> heap;
  heap.reserve(k);

  auto min_heap_cmp = [](const std::pair<float, unsigned int> &a,
                         const std::pair<float, unsigned int> &b) {
    return a.first > b.first; // min on top
  };

  for (unsigned int i = 0; i < k; ++i)
    heap.emplace_back(logits[i], i);
  std::make_heap(heap.begin(), heap.end(), min_heap_cmp);

  for (unsigned int i = k; i < NUM_VOCAB; ++i) {
    if (logits[i] > heap.front().first) {
      std::pop_heap(heap.begin(), heap.end(), min_heap_cmp);
      heap.back() = {logits[i], i};
      std::push_heap(heap.begin(), heap.end(), min_heap_cmp);
    }
  }

  // Sort top-k survivors in descending order.
  std::sort(heap.begin(), heap.end(),
            [](const std::pair<float, unsigned int> &a,
               const std::pair<float, unsigned int> &b) {
              return a.first > b.first;
            });

  outputs.reserve(k);
  for (unsigned int i = 0; i < k; ++i)
    outputs.push_back(heap[i].second);

  return outputs;
}

void applyRepetitionPenalty(float *logits, unsigned int *input_ids,
                            unsigned int NUM_INPUT_IDS,
                            float repetition_penalty) {
  // Apply the penalty *once per unique token id*. The previous implementation
  // walked input_ids directly, so a token that appeared k times got penalised
  // k times (i.e. logit /= repetition_penalty^k for positive logits), which
  // compounds exponentially over long contexts. HuggingFace applies the
  // penalty once per unique token; matching that behaviour here.
  std::unordered_set<unsigned int> seen;
  seen.reserve(NUM_INPUT_IDS);
  for (unsigned int i = 0; i < NUM_INPUT_IDS; ++i) {
    const unsigned int id = input_ids[i];
    if (!seen.insert(id).second)
      continue;
    if (logits[id] < 0) {
      logits[id] *= repetition_penalty;
    } else {
      logits[id] /= repetition_penalty;
    }
  }
}

void applyBadWordsPenalty(float *logits, unsigned int *bad_words_ids,
                          unsigned int NUM_BAD_WORDS_IDS) {
  for (unsigned int i = 0; i < NUM_BAD_WORDS_IDS; ++i) {
    logits[bad_words_ids[i]] = -INFINITY;
  }
}

/**
 * @brief Apply temperature & top-k & top-p to logits.
 *
 *        Entries of `logits` that are filtered out (i.e. not among the top-k
 *        or outside the nucleus defined by top-p) are set to -INFINITY, so a
 *        caller that runs softmax over the full `len` automatically gives them
 *        zero probability (exp(-inf) == 0).
 *
 *        Special cases:
 *          - `temperature <= 1e-5` is treated as "do not scale" (same as
 *            the original behaviour -- it avoids division by zero).
 *          - `top_k == 0` or `top_k >= len` disables top-k filtering.
 *          - `top_p >= 1.0` disables top-p filtering.
 *          - When both filters are disabled the function just applies
 *            temperature and returns the max logit, skipping all sorting.
 *
 * @return Max logit among the survivors (for the follow-up softmax).
 */
float applyTKP(float *logits, int len, float temperature, unsigned int top_k,
               float top_p) {

  // 1. Temperature scaling (in place). Skip when temperature is effectively 1.
  if (temperature > 1e-5f && std::fabs(temperature - 1.0f) > 1e-6f) {
    const float inv_temp = 1.0f / temperature;
    for (int i = 0; i < len; ++i)
      logits[i] *= inv_temp;
  }

  const bool disable_topk = (top_k == 0 || top_k >= static_cast<unsigned int>(len));
  const bool disable_topp = (top_p >= 1.0f);

  // 2. Fast path: no filtering required -> only the max is needed.
  if (disable_topk && disable_topp) {
    return *std::max_element(logits, logits + len);
  }

  // 3. Select the top-k logits with a bounded min-heap. When top-k is
  //    disabled but top-p is active we still have to consider every token,
  //    so fall back to k == len (sorted below).
  const unsigned int k =
    disable_topk ? static_cast<unsigned int>(len) : top_k;

  std::vector<std::pair<float, int>> heap;
  heap.reserve(k);

  // min-heap by logit value -- smallest on top so we can evict efficiently.
  auto min_heap_cmp = [](const std::pair<float, int> &a,
                         const std::pair<float, int> &b) {
    return a.first > b.first;
  };

  for (unsigned int i = 0; i < k; ++i)
    heap.emplace_back(logits[i], static_cast<int>(i));
  std::make_heap(heap.begin(), heap.end(), min_heap_cmp);

  for (int i = static_cast<int>(k); i < len; ++i) {
    if (logits[i] > heap.front().first) {
      std::pop_heap(heap.begin(), heap.end(), min_heap_cmp);
      heap.back() = {logits[i], i};
      std::push_heap(heap.begin(), heap.end(), min_heap_cmp);
    }
  }

  // Sort survivors in descending order of logit.
  std::sort(heap.begin(), heap.end(),
            [](const std::pair<float, int> &a,
               const std::pair<float, int> &b) { return a.first > b.first; });

  // 4. Apply top-p (nucleus) on probabilities -- softmax over survivors and
  //    keep the smallest prefix whose cumulative probability >= top_p.
  unsigned int keep = k;
  if (!disable_topp) {
    const float max_l = heap[0].first;
    // Numerically stable softmax over the k survivors.
    float sum_exp = 0.0f;
    std::vector<float> probs(k);
    for (unsigned int i = 0; i < k; ++i) {
      probs[i] = std::exp(heap[i].first - max_l);
      sum_exp += probs[i];
    }
    float cum = 0.0f;
    keep = 0;
    const float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    for (unsigned int i = 0; i < k; ++i) {
      cum += probs[i] * inv_sum;
      ++keep;
      if (cum >= top_p)
        break;
    }
    if (keep == 0)
      keep = 1; // safety: always keep at least one token.
  }

  // 5. Mask all logits to -INFINITY, then restore the survivors.
  //    This keeps the existing full-length softmax in the caller correct.
  const float neg_inf = -std::numeric_limits<float>::infinity();
  std::fill_n(logits, len, neg_inf);
  for (unsigned int i = 0; i < keep; ++i)
    logits[heap[i].second] = heap[i].first;

  return heap[0].first;
}

/**
 * @brief Compact variant of applyTKP.
 *
 *        See the declaration in llm_util.hpp. This version is used by the
 *        token-generation hot path so that the subsequent softmax and sampler
 *        only touch the `k` (<= top_k) survivors rather than the full
 *        vocabulary.
 */
float applyTKPCompact(const float *logits, int len, float temperature,
                      unsigned int top_k, float top_p,
                      std::vector<std::pair<int, float>> &survivors) {

  // Temperature scaling is monotonic for temperature > 0, so we can do top-k
  // on unscaled logits and only scale the survivors. This avoids a full-vocab
  // pass for temperature.
  const bool apply_temp =
    (temperature > 1e-5f && std::fabs(temperature - 1.0f) > 1e-6f);
  const float inv_temp = apply_temp ? (1.0f / temperature) : 1.0f;

  const bool disable_topk =
    (top_k == 0 || top_k >= static_cast<unsigned int>(len));
  const bool disable_topp = (top_p >= 1.0f);

  // Fast path: no top-k and no top-p. Only the argmax matters in practice;
  // but callers that request this path still need survivors populated so they
  // can sample. Because sampling over the entire vocabulary is what the user
  // asked for in this case, fall through to the generic selection below with
  // k == len.
  const unsigned int k =
    disable_topk ? static_cast<unsigned int>(len) : top_k;

  survivors.clear();
  survivors.reserve(k);

  // min-heap by logit value -- smallest on top so we can evict efficiently.
  auto min_heap_cmp = [](const std::pair<int, float> &a,
                         const std::pair<int, float> &b) {
    return a.second > b.second;
  };

  // Build a size-k min-heap over (index, unscaled-logit) pairs.
  for (unsigned int i = 0; i < k; ++i)
    survivors.emplace_back(static_cast<int>(i), logits[i]);
  std::make_heap(survivors.begin(), survivors.end(), min_heap_cmp);

  for (int i = static_cast<int>(k); i < len; ++i) {
    if (logits[i] > survivors.front().second) {
      std::pop_heap(survivors.begin(), survivors.end(), min_heap_cmp);
      survivors.back() = {i, logits[i]};
      std::push_heap(survivors.begin(), survivors.end(), min_heap_cmp);
    }
  }

  // Sort survivors in descending order of logit.
  std::sort(survivors.begin(), survivors.end(),
            [](const std::pair<int, float> &a,
               const std::pair<int, float> &b) {
              return a.second > b.second;
            });

  // Apply temperature scaling on the (up to) k survivors.
  if (apply_temp) {
    for (auto &p : survivors)
      p.second *= inv_temp;
  }

  // Apply top-p (nucleus) on probabilities. Numerically-stable softmax over
  // the k survivors, then cumulative-sum cutoff.
  if (!disable_topp) {
    const float max_l = survivors[0].second;
    float sum_exp = 0.0f;
    std::vector<float> probs(survivors.size());
    for (size_t i = 0; i < survivors.size(); ++i) {
      probs[i] = std::exp(survivors[i].second - max_l);
      sum_exp += probs[i];
    }
    const float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    float cum = 0.0f;
    size_t keep = 0;
    for (size_t i = 0; i < probs.size(); ++i) {
      cum += probs[i] * inv_sum;
      ++keep;
      if (cum >= top_p)
        break;
    }
    if (keep == 0)
      keep = 1;
    survivors.resize(keep);
  }

  return survivors.empty() ? -std::numeric_limits<float>::infinity()
                           : survivors[0].second;
}
