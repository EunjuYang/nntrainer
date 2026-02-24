// SPDX-License-Identifier: Apache-2.0
/**
 * @file benchmark_attention_kernels.cpp
 * @brief Benchmark and correctness verification for attention kernel
 *        optimizations. Tests softmax_row, compute_kcaches,
 *        compute_fp16vcache_fp32_transposed, and compute_rotary_emb_value
 *        on x86 (AVX2) and ARM (NEON) backends.
 *
 * Build (standalone x86):
 *   g++ -std=c++17 -O2 -mavx2 -mfma -mf16c -fopenmp \
 *       -I../nntrainer/tensor/cpu_backend \
 *       -I../nntrainer/tensor/cpu_backend/x86 \
 *       -I../nntrainer/tensor \
 *       benchmark_attention_kernels.cpp \
 *       ../nntrainer/tensor/cpu_backend/x86/avx2_impl.cpp \
 *       -o benchmark_attention_kernels
 */

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include <cpu_backend.h>
#include <fp16.h>

// ─── helpers ──────────────────────────────────────────────────────────────────
static std::mt19937 rng(42);
static std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

static void fill_random(float *buf, size_t n) {
  for (size_t i = 0; i < n; ++i)
    buf[i] = dist(rng);
}

static void fill_random_fp16(uint16_t *buf, size_t n) {
  for (size_t i = 0; i < n; ++i)
    buf[i] = nntrainer::compute_fp32_to_fp16(dist(rng));
}

static float max_abs_diff(const float *a, const float *b, size_t n) {
  float max_diff = 0.0f;
  for (size_t i = 0; i < n; ++i)
    max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
  return max_diff;
}

static bool all_finite(const float *buf, size_t n) {
  for (size_t i = 0; i < n; ++i)
    if (!std::isfinite(buf[i]))
      return false;
  return true;
}

// Check softmax properties: all values in [0,1] and columns sum to 1.
// The softmax is computed per-column (per attention head) across rows.
static bool check_softmax_valid(const float *buf, size_t start_row,
                                size_t end_row, size_t num_heads,
                                float tol = 1e-4f) {
  for (size_t c = 0; c < num_heads; ++c) {
    float sum = 0.0f;
    for (size_t r = start_row; r < end_row; ++r) {
      float v = buf[r * num_heads + c];
      if (v < -tol || v > 1.0f + tol || !std::isfinite(v))
        return false;
      sum += v;
    }
    if (std::abs(sum - 1.0f) > tol)
      return false;
  }
  return true;
}

using Clock = std::chrono::high_resolution_clock;

struct BenchResult {
  double us; // microseconds
  float max_diff;
  bool valid;
};

// ─── Reference implementations (naive, no SIMD) ──────────────────────────────
// These serve as the "before optimization" baseline for correctness checks.

static void ref_softmax_row_inplace(float *qk_out, size_t start_row,
                                    size_t end_row, size_t num_heads) {
  // Softmax is computed per column (per attention head) across all rows
  // (context positions).  The layout is [row, head] where each column
  // represents one head and rows represent key positions.
  for (size_t c = 0; c < num_heads; ++c) {
    float max_val = -INFINITY;
    for (size_t r = start_row; r < end_row; ++r)
      max_val = std::max(max_val, qk_out[r * num_heads + c]);
    float sum = 0.0f;
    for (size_t r = start_row; r < end_row; ++r) {
      qk_out[r * num_heads + c] =
        std::exp(qk_out[r * num_heads + c] - max_val);
      sum += qk_out[r * num_heads + c];
    }
    for (size_t r = start_row; r < end_row; ++r)
      qk_out[r * num_heads + c] /= sum;
  }
}

static void ref_compute_kcaches(const float *in, const uint16_t *kcache,
                                float *output, int num_rows,
                                int num_cache_head, int head_dim, int gqa_size,
                                size_t local_window_size) {
  float inv_sqrt = 1.0f / std::sqrt((float)head_dim);
  int start_row = num_rows < (int)local_window_size
                    ? 0
                    : num_rows - (int)local_window_size;
  int row_cnt = num_rows < (int)local_window_size ? num_rows
                                                  : (int)local_window_size;

  std::vector<float> tmp(head_dim);
  for (int n = 0; n < num_cache_head; ++n) {
    for (int t_row = 0; t_row < row_cnt; ++t_row) {
      int row = start_row + t_row;
      const uint16_t *kptr = kcache + (row * num_cache_head + n) * head_dim;
      for (int i = 0; i < head_dim; ++i)
        tmp[i] = nntrainer::compute_fp16_to_fp32(kptr[i]);

      for (int g = 0; g < gqa_size; ++g) {
        const float *in_ptr = in + n * gqa_size * head_dim + g * head_dim;
        float sum = 0.0f;
        for (int i = 0; i < head_dim; ++i)
          sum += in_ptr[i] * tmp[i];
        output[t_row * num_cache_head * gqa_size + n * gqa_size + g] =
          sum * inv_sqrt;
      }
    }
  }
}

static void ref_compute_vcache(int row_num, const float *in,
                               const uint16_t *vcache, float *output,
                               int num_cache_head, int gqa_size, int head_dim,
                               size_t local_window_size) {
  int j_start = row_num < (int)local_window_size
                  ? 0
                  : row_num + 1 - (int)local_window_size;

  std::vector<float> tmp(head_dim);
  for (int n = 0; n < num_cache_head; ++n) {
    for (int g = 0; g < gqa_size; ++g) {
      for (int d = 0; d < head_dim; ++d)
        output[(n * gqa_size + g) * head_dim + d] = 0.0f;
    }

    for (int j = j_start; j <= row_num; ++j) {
      const uint16_t *vptr = vcache + (j * num_cache_head + n) * head_dim;
      for (int i = 0; i < head_dim; ++i)
        tmp[i] = nntrainer::compute_fp16_to_fp32(vptr[i]);

      for (int g = 0; g < gqa_size; ++g) {
        float a_val =
          in[(row_num < (int)local_window_size
                ? j
                : (j - j_start)) *
               gqa_size * num_cache_head +
             n * gqa_size + g];
        for (int d = 0; d < head_dim; ++d)
          output[(n * gqa_size + g) * head_dim + d] += a_val * tmp[d];
      }
    }
  }
}

// ─── Benchmark functions ─────────────────────────────────────────────────────

struct ModelConfig {
  const char *name;
  int num_q_heads;
  int num_kv_heads;
  int head_dim;
  int gqa_size;
};

static void benchmark_softmax(const ModelConfig &cfg, int ctx_len,
                              int num_iters) {
  size_t num_heads = cfg.num_q_heads;
  size_t start_row = 0;
  size_t end_row = ctx_len;
  size_t buf_size = end_row * num_heads;

  std::vector<float> input(buf_size);
  fill_random(input.data(), buf_size);

  // --- Reference ---
  std::vector<float> ref_out(input);
  ref_softmax_row_inplace(ref_out.data(), start_row, end_row, num_heads);

  // --- Optimized ---
  std::vector<float> opt_out(input);
  nntrainer::softmax_row_inplace<float>(opt_out.data(), start_row, end_row,
                                        num_heads, (float *)nullptr);

  float diff = max_abs_diff(ref_out.data(), opt_out.data(), buf_size);
  bool valid = check_softmax_valid(opt_out.data(), start_row, end_row,
                                   num_heads);

  // --- Perf ---
  double total_us = 0.0;
  for (int iter = 0; iter < num_iters; ++iter) {
    std::vector<float> tmp(input);
    auto t0 = Clock::now();
    nntrainer::softmax_row_inplace<float>(tmp.data(), start_row, end_row,
                                          num_heads, (float *)nullptr);
    auto t1 = Clock::now();
    total_us +=
      std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
  }

  std::cout << "  softmax_row (" << cfg.name << ", ctx=" << ctx_len
            << "): " << std::fixed << std::setprecision(1)
            << total_us / num_iters << " us/iter"
            << "  max_diff=" << std::scientific << std::setprecision(2) << diff
            << "  valid=" << (valid ? "OK" : "FAIL") << std::endl;
}

static void benchmark_kcaches(const ModelConfig &cfg, int ctx_len,
                              int num_iters) {
  int num_rows = ctx_len;
  int num_cache_head = cfg.num_kv_heads;
  int head_dim = cfg.head_dim;
  int gqa_size = cfg.gqa_size;
  int tile_size = 32;
  size_t local_window = ctx_len;

  size_t q_size = (size_t)num_cache_head * gqa_size * head_dim;
  size_t k_size = (size_t)num_rows * num_cache_head * head_dim;
  int row_cnt = num_rows < (int)local_window ? num_rows : (int)local_window;
  size_t out_size = (size_t)row_cnt * num_cache_head * gqa_size;

  std::vector<float> query(q_size);
  std::vector<uint16_t> kcache(k_size);
  fill_random(query.data(), q_size);
  fill_random_fp16(kcache.data(), k_size);

  // --- Reference ---
  std::vector<float> ref_out(out_size, 0.0f);
  ref_compute_kcaches(query.data(), kcache.data(), ref_out.data(), num_rows,
                      num_cache_head, head_dim, gqa_size, local_window);

  // --- Optimized ---
  std::vector<float> opt_out(out_size, 0.0f);
  nntrainer::compute_kcaches<uint16_t>(
    query.data(), kcache.data(), opt_out.data(), num_rows, num_cache_head,
    head_dim, gqa_size, tile_size, local_window, 0, -1);

  float diff = max_abs_diff(ref_out.data(), opt_out.data(), out_size);
  bool valid = all_finite(opt_out.data(), out_size);

  // --- Perf ---
  double total_us = 0.0;
  for (int iter = 0; iter < num_iters; ++iter) {
    std::vector<float> tmp(out_size, 0.0f);
    auto t0 = Clock::now();
    nntrainer::compute_kcaches<uint16_t>(
      query.data(), kcache.data(), tmp.data(), num_rows, num_cache_head,
      head_dim, gqa_size, tile_size, local_window, 0, -1);
    auto t1 = Clock::now();
    total_us +=
      std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
  }

  std::cout << "  compute_kcaches (" << cfg.name << ", ctx=" << ctx_len
            << "): " << std::fixed << std::setprecision(1)
            << total_us / num_iters << " us/iter"
            << "  max_diff=" << std::scientific << std::setprecision(2) << diff
            << "  valid=" << (valid ? "OK" : "FAIL") << std::endl;
}

static void benchmark_vcache(const ModelConfig &cfg, int ctx_len,
                             int num_iters) {
  int row_num = ctx_len - 1;
  int num_cache_head = cfg.num_kv_heads;
  int head_dim = cfg.head_dim;
  int gqa_size = cfg.gqa_size;
  size_t local_window = ctx_len;

  size_t v_size = (size_t)ctx_len * num_cache_head * head_dim;
  size_t in_size = (size_t)ctx_len * gqa_size * num_cache_head;
  size_t out_size = (size_t)num_cache_head * gqa_size * head_dim;

  std::vector<float> attn_probs(in_size);
  std::vector<uint16_t> vcache(v_size);
  fill_random(attn_probs.data(), in_size);
  fill_random_fp16(vcache.data(), v_size);

  // --- Reference ---
  std::vector<float> ref_out(out_size, 0.0f);
  ref_compute_vcache(row_num, attn_probs.data(), vcache.data(), ref_out.data(),
                     num_cache_head, gqa_size, head_dim, local_window);

  // --- Optimized ---
  std::vector<float> opt_out(out_size, 0.0f);
  nntrainer::compute_fp16vcache_fp32_transposed(
    row_num, attn_probs.data(), vcache.data(), opt_out.data(), num_cache_head,
    gqa_size, head_dim, local_window, 0, -1);

  float diff = max_abs_diff(ref_out.data(), opt_out.data(), out_size);
  bool valid = all_finite(opt_out.data(), out_size);

  // --- Perf ---
  double total_us = 0.0;
  for (int iter = 0; iter < num_iters; ++iter) {
    std::vector<float> tmp(out_size, 0.0f);
    auto t0 = Clock::now();
    nntrainer::compute_fp16vcache_fp32_transposed(
      row_num, attn_probs.data(), vcache.data(), tmp.data(), num_cache_head,
      gqa_size, head_dim, local_window, 0, -1);
    auto t1 = Clock::now();
    total_us +=
      std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
  }

  std::cout << "  compute_vcache (" << cfg.name << ", ctx=" << ctx_len
            << "): " << std::fixed << std::setprecision(1)
            << total_us / num_iters << " us/iter"
            << "  max_diff=" << std::scientific << std::setprecision(2) << diff
            << "  valid=" << (valid ? "OK" : "FAIL") << std::endl;
}

static void benchmark_rope(const ModelConfig &cfg, int num_iters) {
  int width = cfg.head_dim;
  int dim = cfg.head_dim;
  int half_ = dim / 2;

  std::vector<float> input(width);
  std::vector<float> cos_v(half_);
  std::vector<float> sin_v(half_);
  fill_random(input.data(), width);
  for (int i = 0; i < half_; ++i) {
    float angle = dist(rng);
    cos_v[i] = std::cos(angle);
    sin_v[i] = std::sin(angle);
  }

  // --- Reference: simple scalar RoPE ---
  std::vector<float> ref_out(input);
  for (int k = 0; k < half_; ++k) {
    float a = ref_out[k];
    float b = ref_out[k + half_];
    ref_out[k] = a * cos_v[k] - b * sin_v[k];
    ref_out[k + half_] = a * sin_v[k] + b * cos_v[k];
  }

  // --- Optimized ---
  std::vector<float> opt_out(input);
  nntrainer::compute_rotary_emb_value(width, dim, half_, opt_out.data(),
                                      nullptr, cos_v.data(), sin_v.data(),
                                      false);

  float diff = max_abs_diff(ref_out.data(), opt_out.data(), width);
  bool valid = all_finite(opt_out.data(), width);

  // --- Perf ---
  double total_us = 0.0;
  for (int iter = 0; iter < num_iters; ++iter) {
    std::vector<float> tmp(input);
    auto t0 = Clock::now();
    nntrainer::compute_rotary_emb_value(width, dim, half_, tmp.data(), nullptr,
                                        cos_v.data(), sin_v.data(), false);
    auto t1 = Clock::now();
    total_us +=
      std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
  }

  std::cout << "  compute_rotary_emb (" << cfg.name
            << "): " << std::fixed << std::setprecision(1)
            << total_us / num_iters << " us/iter"
            << "  max_diff=" << std::scientific << std::setprecision(2) << diff
            << "  valid=" << (valid ? "OK" : "FAIL") << std::endl;
}

// ─── Main ────────────────────────────────────────────────────────────────────

int main() {
  // Model configurations matching Qwen family
  ModelConfig configs[] = {
    {"Qwen2-0.5B", 14, 2, 64, 7},  // 14 Q heads, 2 KV heads, dim=64, gqa=7
    {"Qwen3-4B", 32, 8, 128, 4},   // 32 Q heads, 8 KV heads, dim=128, gqa=4
  };
  int ctx_lens[] = {1, 64, 256, 512};
  int num_iters = 100;

  std::cout << "========================================" << std::endl;
  std::cout << "Attention Kernel Benchmark + Verification" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << "(Iterations: " << num_iters << ")" << std::endl;
  std::cout << std::endl;

  for (auto &cfg : configs) {
    std::cout << "--- " << cfg.name << " (Q=" << cfg.num_q_heads
              << " KV=" << cfg.num_kv_heads << " dim=" << cfg.head_dim
              << " gqa=" << cfg.gqa_size << ") ---" << std::endl;

    for (int ctx : ctx_lens) {
      benchmark_softmax(cfg, ctx, num_iters);
    }
    std::cout << std::endl;

    for (int ctx : ctx_lens) {
      benchmark_kcaches(cfg, ctx, num_iters);
    }
    std::cout << std::endl;

    for (int ctx : ctx_lens) {
      benchmark_vcache(cfg, ctx, num_iters);
    }
    std::cout << std::endl;

    benchmark_rope(cfg, num_iters);
    std::cout << std::endl;
  }

  return 0;
}
