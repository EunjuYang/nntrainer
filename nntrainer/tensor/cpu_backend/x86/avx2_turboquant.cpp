// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 nntrainer authors
 *
 * @file   avx2_turboquant.cpp
 * @date   16 April 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  AVX2 accelerated TurboQuant KV-cache kernels.
 *
 *         Entry points mirror the public TurboQuant API in cpu_backend.h.
 *         Only head_dim in {64, 128} is accelerated; other widths fall
 *         back to the scalar reference (the dispatcher in
 *         x86_compute_backend.cpp enforces this, but each entry point is
 *         also defensive).
 */

#include "avx2_turboquant.h"

#include <cmath>
#include <cstring>
#include <immintrin.h>

#include <fallback_internal.h>
#include <turboquant_utils.h>

namespace nntrainer::avx2 {

namespace {

/**
 * @brief Horizontal sum of the 8 lanes of a __m256.
 */
static inline float hsum256_ps(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 s = _mm_add_ps(lo, hi);
  __m128 shuf = _mm_movehdup_ps(s);
  __m128 sums = _mm_add_ps(s, shuf);
  shuf = _mm_movehl_ps(shuf, sums);
  sums = _mm_add_ss(sums, shuf);
  return _mm_cvtss_f32(sums);
}

/**
 * @brief In-place Fast Walsh-Hadamard Transform with orthogonal scaling
 *        (1/sqrt(n)). Specialized for n == 64 or n == 128; stages with
 *        len >= 8 use 256-bit butterflies, smaller stages stay scalar.
 */
static inline void fwht_avx2(float *x, int n) {
  for (int len = 1; len < n; len <<= 1) {
    if (len < 8) {
      for (int i = 0; i < n; i += (len << 1)) {
        for (int j = 0; j < len; ++j) {
          float u = x[i + j];
          float v = x[i + j + len];
          x[i + j] = u + v;
          x[i + j + len] = u - v;
        }
      }
    } else {
      for (int i = 0; i < n; i += (len << 1)) {
        for (int j = 0; j < len; j += 8) {
          __m256 u = _mm256_loadu_ps(x + i + j);
          __m256 v = _mm256_loadu_ps(x + i + j + len);
          _mm256_storeu_ps(x + i + j, _mm256_add_ps(u, v));
          _mm256_storeu_ps(x + i + j + len, _mm256_sub_ps(u, v));
        }
      }
    }
  }
  const float inv_sqrt_n = 1.0f / std::sqrt((float)n);
  const __m256 s = _mm256_set1_ps(inv_sqrt_n);
  for (int i = 0; i < n; i += 8) {
    __m256 v = _mm256_loadu_ps(x + i);
    _mm256_storeu_ps(x + i, _mm256_mul_ps(v, s));
  }
}

/**
 * @brief Load 4 packed nibble bytes and return 8 int32 lanes, each holding
 *        one 4-bit centroid index for the corresponding float position.
 *
 *        Layout (matches turboquant_utils.h packing):
 *          byte k  low nibble  -> element 2k
 *          byte k  high nibble -> element 2k+1
 */
static inline __m256i unpack_4bytes_to_8_indices(const uint8_t *packed) {
  uint32_t bits;
  std::memcpy(&bits, packed, sizeof(bits));
  const __m256i shifts = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
  const __m256i mask = _mm256_set1_epi32(0x0F);
  __m256i idx = _mm256_srlv_epi32(_mm256_set1_epi32((int)bits), shifts);
  return _mm256_and_si256(idx, mask);
}

/**
 * @brief Dequantize one packed head and compute the dot product with a
 *        pre-rotated query, for head_dim in {64, 128}. The caller owns
 *        horizontal summation and final scaling.
 */
static inline float dot_rq_packed4(const float *rq, const uint8_t *packed,
                                   const float *centroids, int head_dim) {
  __m256 acc = _mm256_setzero_ps();
  const int chunks = head_dim / 8; // 8 or 16
  for (int k = 0; k < chunks; ++k) {
    __m256i idx = unpack_4bytes_to_8_indices(packed + 4 * k);
    __m256 centroid = _mm256_i32gather_ps(centroids, idx, 4);
    __m256 rq_vec = _mm256_loadu_ps(rq + 8 * k);
    acc = _mm256_fmadd_ps(rq_vec, centroid, acc);
  }
  return hsum256_ps(acc);
}

/**
 * @brief Rotate a query head in place: rq[i] = q[i] * signs[i], then FWHT.
 */
static inline void rotate_query(const float *q, const float *signs, float *rq,
                                int head_dim) {
  for (int i = 0; i < head_dim; i += 8) {
    __m256 qv = _mm256_loadu_ps(q + i);
    __m256 sv = _mm256_loadu_ps(signs + i);
    _mm256_storeu_ps(rq + i, _mm256_mul_ps(qv, sv));
  }
  fwht_avx2(rq, head_dim);
}

} // namespace

void quantize_kv_turboquant(const float *input, uint8_t *out_packed,
                            float *out_norms, const float *rot_signs,
                            int head_dim, int num_heads) {
  nntrainer::__fallback_quantize_kv_turboquant(
    input, out_packed, out_norms, rot_signs, head_dim, num_heads);
}

void compute_kcaches_packed4(const float *query, const uint8_t *kcache_packed,
                             const float *kcache_norms, float *output,
                             int num_rows, int num_cache_head, int head_dim,
                             int gqa_size, int tile_size,
                             const float *rot_signs, size_t local_window_size,
                             int head_start, int head_end) {
  // Defensive fallback: only head_dim 64/128 are supported by the codebook
  // and by the packed layout used below.
  if (head_dim != 64 && head_dim != 128) {
    nntrainer::__fallback_compute_kcaches_packed4(
      query, kcache_packed, kcache_norms, output, num_rows, num_cache_head,
      head_dim, gqa_size, tile_size, rot_signs, local_window_size, head_start,
      head_end);
    return;
  }

  const nntrainer::LloydMaxCodebook &cb = nntrainer::get_codebook(head_dim);
  const int actual_head_end = (head_end < 0) ? num_cache_head : head_end;
  const int start_row =
    (size_t)num_rows < local_window_size ? 0 : num_rows - (int)local_window_size;
  const int row_cnt =
    (size_t)num_rows < local_window_size ? num_rows : (int)local_window_size;
  const int packed_row_bytes = num_cache_head * head_dim / 2;
  const int packed_head_bytes = head_dim / 2;
  const float inv_sqrt_d = 1.0f / std::sqrt((float)head_dim);

  // Stack-allocated rotated query buffer (gqa_size * head_dim floats).
  // head_dim <= 128 and gqa_size is typically small, so stack is fine.
  alignas(32) float rotated_queries[/*gqa_size*/ 32 * /*head_dim*/ 128];
  // Guard against unusually large gqa_size; fall back if we'd overflow.
  if (gqa_size * head_dim > (int)(sizeof(rotated_queries) / sizeof(float))) {
    nntrainer::__fallback_compute_kcaches_packed4(
      query, kcache_packed, kcache_norms, output, num_rows, num_cache_head,
      head_dim, gqa_size, tile_size, rot_signs, local_window_size, head_start,
      head_end);
    return;
  }

  for (int n = head_start; n < actual_head_end; ++n) {
    for (int g = 0; g < gqa_size; ++g) {
      const float *q_ptr = query + n * gqa_size * head_dim + g * head_dim;
      rotate_query(q_ptr, rot_signs, rotated_queries + g * head_dim, head_dim);
    }

    for (int t_row = 0; t_row < row_cnt; ++t_row) {
      const int row = start_row + t_row;
      const uint8_t *packed_ptr =
        kcache_packed + row * packed_row_bytes + n * packed_head_bytes;
      const float norm = kcache_norms[row * num_cache_head + n];
      const float post_scale = norm * inv_sqrt_d;

      for (int g = 0; g < gqa_size; ++g) {
        const float *rq = rotated_queries + g * head_dim;
        const float sum = dot_rq_packed4(rq, packed_ptr, cb.centroids, head_dim);
        output[t_row * num_cache_head * gqa_size + n * gqa_size + g] =
          sum * post_scale;
      }
    }
  }
}

void compute_vcache_packed4(int row_num, const float *attn_weights,
                            const uint8_t *vcache_packed,
                            const float *vcache_norms, float *output,
                            int num_cache_head, int gqa_size, int head_dim,
                            const float *rot_signs, size_t local_window_size,
                            int head_start, int head_end) {
  if (head_dim != 64 && head_dim != 128) {
    nntrainer::__fallback_compute_vcache_packed4(
      row_num, attn_weights, vcache_packed, vcache_norms, output,
      num_cache_head, gqa_size, head_dim, rot_signs, local_window_size,
      head_start, head_end);
    return;
  }

  const nntrainer::LloydMaxCodebook &cb = nntrainer::get_codebook(head_dim);
  const int actual_head_end = (head_end < 0) ? num_cache_head : head_end;
  const int packed_row_bytes = num_cache_head * head_dim / 2;
  const int packed_head_bytes = head_dim / 2;
  const int j_start = (size_t)row_num < local_window_size
                        ? 0
                        : row_num + 1 - (int)local_window_size;
  const int chunks = head_dim / 8; // 8 or 16

  alignas(32) float acc[128]; // head_dim <= 128

  for (int n = head_start; n < actual_head_end; ++n) {
    for (int h = 0; h < gqa_size; ++h) {
      // Zero the accumulator (vectorized).
      const __m256 zero = _mm256_setzero_ps();
      for (int k = 0; k < chunks; ++k)
        _mm256_storeu_ps(acc + 8 * k, zero);

      for (int j = j_start; j <= row_num; ++j) {
        const float a_val =
          attn_weights[((j - j_start) * num_cache_head + n) * gqa_size + h];
        const uint8_t *packed_ptr =
          vcache_packed + j * packed_row_bytes + n * packed_head_bytes;
        const float norm = vcache_norms[j * num_cache_head + n];
        const __m256 scale_v = _mm256_set1_ps(a_val * norm);

        for (int k = 0; k < chunks; ++k) {
          __m256i idx = unpack_4bytes_to_8_indices(packed_ptr + 4 * k);
          __m256 centroid = _mm256_i32gather_ps(cb.centroids, idx, 4);
          __m256 a = _mm256_loadu_ps(acc + 8 * k);
          a = _mm256_fmadd_ps(scale_v, centroid, a);
          _mm256_storeu_ps(acc + 8 * k, a);
        }
      }

      fwht_avx2(acc, head_dim);

      float *out_head = output + (n * gqa_size + h) * head_dim;
      for (int k = 0; k < chunks; ++k) {
        __m256 a = _mm256_loadu_ps(acc + 8 * k);
        __m256 s = _mm256_loadu_ps(rot_signs + 8 * k);
        _mm256_storeu_ps(out_head + 8 * k, _mm256_mul_ps(a, s));
      }
    }
  }
}

} // namespace nntrainer::avx2
