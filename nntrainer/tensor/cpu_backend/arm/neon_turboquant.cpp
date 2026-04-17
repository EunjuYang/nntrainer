// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 nntrainer authors
 *
 * @file   neon_turboquant.cpp
 * @date   17 April 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  NEON accelerated TurboQuant KV-cache kernels.
 *
 *         Entry points mirror the public TurboQuant API in cpu_backend.h.
 *         Only head_dim in {64, 128} is accelerated; other widths fall
 *         back to the scalar reference (the dispatcher in
 *         arm_compute_backend.cpp enforces this, but each entry point is
 *         also defensive).
 */

#include "neon_turboquant.h"

#include <arm_neon.h>
#include <cmath>
#include <cstring>

#include <fallback_internal.h>
#include <turboquant_utils.h>

namespace nntrainer::neon {

namespace {

#if defined(__aarch64__) || defined(_M_ARM64)
#define NEON_FMA_F32(_X, _Y, _Z) vfmaq_f32(_X, _Y, _Z)
#else
#define NEON_FMA_F32(_X, _Y, _Z) vaddq_f32(_X, vmulq_f32(_Y, _Z))
#endif

/**
 * @brief Horizontal sum of the 4 lanes of a float32x4_t.
 */
static inline float hsum128_ps(float32x4_t v) {
#if defined(__aarch64__) || defined(_M_ARM64)
  return vaddvq_f32(v);
#else
  float32x2_t lo = vget_low_f32(v);
  float32x2_t hi = vget_high_f32(v);
  float32x2_t s = vadd_f32(lo, hi);
  s = vpadd_f32(s, s);
  return vget_lane_f32(s, 0);
#endif
}

/**
 * @brief In-place Fast Walsh-Hadamard Transform with orthogonal scaling
 *        (1/sqrt(n)). Specialized for n == 64 or n == 128; stages with
 *        len >= 4 use 128-bit butterflies, smaller stages stay scalar.
 */
static inline void fwht_neon(float *x, int n) {
  for (int len = 1; len < n; len <<= 1) {
    if (len < 4) {
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
        for (int j = 0; j < len; j += 4) {
          float32x4_t u = vld1q_f32(x + i + j);
          float32x4_t v = vld1q_f32(x + i + j + len);
          vst1q_f32(x + i + j, vaddq_f32(u, v));
          vst1q_f32(x + i + j + len, vsubq_f32(u, v));
        }
      }
    }
  }
  const float inv_sqrt_n = 1.0f / std::sqrt((float)n);
  const float32x4_t s = vdupq_n_f32(inv_sqrt_n);
  for (int i = 0; i < n; i += 4) {
    float32x4_t v = vld1q_f32(x + i);
    vst1q_f32(x + i, vmulq_f32(v, s));
  }
}

/**
 * @brief Load 2 packed nibble bytes and return 4 float centroids indexed
 *        from the codebook.
 *
 *        Layout (matches turboquant_utils.h packing):
 *          byte k  low nibble  -> element 2k
 *          byte k  high nibble -> element 2k+1
 */
static inline float32x4_t gather4_centroids(const uint8_t *packed,
                                            const float *centroids) {
  const uint8_t b0 = packed[0];
  const uint8_t b1 = packed[1];
  float32x4_t r = vdupq_n_f32(0.0f);
  r = vsetq_lane_f32(centroids[b0 & 0x0F], r, 0);
  r = vsetq_lane_f32(centroids[(b0 >> 4) & 0x0F], r, 1);
  r = vsetq_lane_f32(centroids[b1 & 0x0F], r, 2);
  r = vsetq_lane_f32(centroids[(b1 >> 4) & 0x0F], r, 3);
  return r;
}

/**
 * @brief Dequantize one packed head and compute the dot product with a
 *        pre-rotated query, for head_dim in {64, 128}. The caller owns
 *        horizontal summation and final scaling.
 */
static inline float dot_rq_packed4(const float *rq, const uint8_t *packed,
                                   const float *centroids, int head_dim) {
  float32x4_t acc = vdupq_n_f32(0.0f);
  const int chunks = head_dim / 4; // 16 or 32
  for (int k = 0; k < chunks; ++k) {
    float32x4_t centroid = gather4_centroids(packed + 2 * k, centroids);
    float32x4_t rq_vec = vld1q_f32(rq + 4 * k);
    acc = NEON_FMA_F32(acc, rq_vec, centroid);
  }
  return hsum128_ps(acc);
}

/**
 * @brief Rotate a query head in place: rq[i] = q[i] * signs[i], then FWHT.
 */
static inline void rotate_query(const float *q, const float *signs, float *rq,
                                int head_dim) {
  for (int i = 0; i < head_dim; i += 4) {
    float32x4_t qv = vld1q_f32(q + i);
    float32x4_t sv = vld1q_f32(signs + i);
    vst1q_f32(rq + i, vmulq_f32(qv, sv));
  }
  fwht_neon(rq, head_dim);
}

/**
 * @brief Per-head TurboQuant quantize (paper Algorithm 1), NEON path.
 *        head_dim must be 64 or 128; caller is responsible for gating.
 *
 *        Pipeline:
 *          1) norm^2 via FMA accumulator, then sqrt.
 *          2) rotated[i] = input[i] * (1/norm) * rot_signs[i], then FWHT.
 *          3) Lloyd-Max quantize: idx = count of boundaries that val > b_k,
 *             computed across 4 lanes at a time as 15 compare+sub passes.
 *          4) Pack two nibbles per byte into out_packed.
 */
static inline void quantize_head_neon(const float *input, int head_dim,
                                      uint8_t *out_packed, float *out_norm,
                                      const float *rot_signs,
                                      const nntrainer::LloydMaxCodebook &cb) {
  const int chunks = head_dim / 4; // 16 or 32

  // Step 1: norm^2 → norm.
  float32x4_t sum_sq = vdupq_n_f32(0.0f);
  for (int k = 0; k < chunks; ++k) {
    float32x4_t x = vld1q_f32(input + 4 * k);
    sum_sq = NEON_FMA_F32(sum_sq, x, x);
  }
  const float norm = std::sqrt(hsum128_ps(sum_sq));
  *out_norm = norm;
  const float inv_norm = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;
  const float32x4_t inv_norm_v = vdupq_n_f32(inv_norm);

  // Step 2: rotated = input * inv_norm * rot_signs, then FWHT.
  alignas(16) float rotated[128];
  for (int k = 0; k < chunks; ++k) {
    float32x4_t x = vld1q_f32(input + 4 * k);
    float32x4_t s = vld1q_f32(rot_signs + 4 * k);
    vst1q_f32(rotated + 4 * k, vmulq_f32(vmulq_f32(x, inv_norm_v), s));
  }
  fwht_neon(rotated, head_dim);

  // Steps 3+4: 4-bit Lloyd-Max quantize + pack 2 nibbles per byte.
  for (int k = 0; k < chunks; ++k) {
    float32x4_t val = vld1q_f32(rotated + 4 * k);
    uint32x4_t idx = vdupq_n_u32(0);
    // idx lane = #(boundaries that val > b_i). Since boundaries are sorted
    // ascending, this yields the target bin index in [0, 15].
    for (int b = 0; b < 15; ++b) {
      float32x4_t bnd = vdupq_n_f32(cb.boundaries[b]);
      uint32x4_t cmp = vcgtq_f32(val, bnd);
      // cmp lane is either all-ones (uint32 = 0xFFFFFFFF) or 0;
      // subtracting 0xFFFFFFFF is equivalent to adding 1.
      idx = vsubq_u32(idx, cmp);
    }

    alignas(16) uint32_t q[4];
    vst1q_u32(q, idx);

    uint8_t *out = out_packed + 2 * k;
    out[0] = (uint8_t)((q[1] << 4) | q[0]);
    out[1] = (uint8_t)((q[3] << 4) | q[2]);
  }
}

} // namespace

void quantize_kv_turboquant(const float *input, uint8_t *out_packed,
                            float *out_norms, const float *rot_signs,
                            int head_dim, int num_heads) {
  if (head_dim != 64 && head_dim != 128) {
    nntrainer::__fallback_quantize_kv_turboquant(
      input, out_packed, out_norms, rot_signs, head_dim, num_heads);
    return;
  }

  const nntrainer::LloydMaxCodebook &cb = nntrainer::get_codebook(head_dim);
  const int packed_head_bytes = head_dim / 2;
  for (int h = 0; h < num_heads; ++h) {
    quantize_head_neon(input + h * head_dim, head_dim,
                       out_packed + h * packed_head_bytes, out_norms + h,
                       rot_signs, cb);
  }
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
  const int start_row = (size_t)num_rows < local_window_size
                          ? 0
                          : num_rows - (int)local_window_size;
  const int row_cnt =
    (size_t)num_rows < local_window_size ? num_rows : (int)local_window_size;
  const int packed_row_bytes = num_cache_head * head_dim / 2;
  const int packed_head_bytes = head_dim / 2;
  const float inv_sqrt_d = 1.0f / std::sqrt((float)head_dim);

  // Stack-allocated rotated query buffer (gqa_size * head_dim floats).
  // head_dim <= 128 and gqa_size is typically small, so stack is fine.
  alignas(16) float rotated_queries[/*gqa_size*/ 32 * /*head_dim*/ 128];
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
        const float sum =
          dot_rq_packed4(rq, packed_ptr, cb.centroids, head_dim);
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
  const int chunks = head_dim / 4; // 16 or 32

  alignas(16) float acc[128]; // head_dim <= 128

  for (int n = head_start; n < actual_head_end; ++n) {
    for (int h = 0; h < gqa_size; ++h) {
      // Zero the accumulator (vectorized).
      const float32x4_t zero = vdupq_n_f32(0.0f);
      for (int k = 0; k < chunks; ++k)
        vst1q_f32(acc + 4 * k, zero);

      for (int j = j_start; j <= row_num; ++j) {
        const float a_val =
          attn_weights[((j - j_start) * num_cache_head + n) * gqa_size + h];
        const uint8_t *packed_ptr =
          vcache_packed + j * packed_row_bytes + n * packed_head_bytes;
        const float norm = vcache_norms[j * num_cache_head + n];
        const float32x4_t scale_v = vdupq_n_f32(a_val * norm);

        for (int k = 0; k < chunks; ++k) {
          float32x4_t centroid =
            gather4_centroids(packed_ptr + 2 * k, cb.centroids);
          float32x4_t a = vld1q_f32(acc + 4 * k);
          a = NEON_FMA_F32(a, scale_v, centroid);
          vst1q_f32(acc + 4 * k, a);
        }
      }

      fwht_neon(acc, head_dim);

      float *out_head = output + (n * gqa_size + h) * head_dim;
      for (int k = 0; k < chunks; ++k) {
        float32x4_t a = vld1q_f32(acc + 4 * k);
        float32x4_t s = vld1q_f32(rot_signs + 4 * k);
        vst1q_f32(out_head + 4 * k, vmulq_f32(a, s));
      }
    }
  }
}

} // namespace nntrainer::neon
