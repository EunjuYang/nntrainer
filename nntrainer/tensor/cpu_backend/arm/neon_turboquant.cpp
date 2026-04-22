// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 nntrainer authors
 *
 * @file   neon_turboquant.cpp
 * @date   22 April 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
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

/**
 * @brief Horizontal sum of the 4 lanes of a float32x4_t. Uses the aarch64
 *        vaddvq_f32 intrinsic where available, falling back to a paired
 *        reduction on armv7.
 */
static inline float hsum_f32x4(float32x4_t v) {
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
 * @brief Load 2 packed nibble bytes and return 4 int32 lanes, each holding
 *        one 4-bit centroid index for the corresponding float position.
 *
 *        Layout (matches turboquant_utils.h packing):
 *          byte k  low nibble  -> element 2k
 *          byte k  high nibble -> element 2k+1
 *
 *        Lanes:
 *          0: bits[ 0.. 3]  (byte0 low nibble)
 *          1: bits[ 4.. 7]  (byte0 high nibble)
 *          2: bits[ 8..11]  (byte1 low nibble)
 *          3: bits[12..15]  (byte1 high nibble)
 */
static inline int32x4_t unpack_2bytes_to_4_indices(const uint8_t *packed) {
  uint32_t bits = 0;
  std::memcpy(&bits, packed, 2);
  // Variable right-shift via vshlq_u32 with negative counts.
  const int32x4_t shifts = {0, -4, -8, -12};
  uint32x4_t v = vdupq_n_u32(bits);
  v = vshlq_u32(v, shifts);
  v = vandq_u32(v, vdupq_n_u32(0x0Fu));
  return vreinterpretq_s32_u32(v);
}

/**
 * @brief Gather 4 float centroid values from a 16-entry LUT using NEON
 *        index lanes. ARM has no native float gather instruction, so we
 *        extract lanes and perform scalar loads into a vector register.
 *        With a 64-byte codebook this stays in L1 and the cost is 4
 *        scalar loads per vector.
 */
static inline float32x4_t gather_centroids(const float *centroids,
                                           int32x4_t idx) {
  const int i0 = vgetq_lane_s32(idx, 0);
  const int i1 = vgetq_lane_s32(idx, 1);
  const int i2 = vgetq_lane_s32(idx, 2);
  const int i3 = vgetq_lane_s32(idx, 3);
  float32x4_t r = vdupq_n_f32(centroids[i0]);
  r = vsetq_lane_f32(centroids[i1], r, 1);
  r = vsetq_lane_f32(centroids[i2], r, 2);
  r = vsetq_lane_f32(centroids[i3], r, 3);
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
    int32x4_t idx = unpack_2bytes_to_4_indices(packed + 2 * k);
    float32x4_t centroid = gather_centroids(centroids, idx);
    float32x4_t rq_v = vld1q_f32(rq + 4 * k);
    acc = vfmaq_f32(acc, rq_v, centroid);
  }
  return hsum_f32x4(acc);
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
  alignas(16) float rotated_queries[/*gqa_size*/ 32 * /*head_dim*/ 128];
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
          int32x4_t idx = unpack_2bytes_to_4_indices(packed_ptr + 2 * k);
          float32x4_t centroid = gather_centroids(cb.centroids, idx);
          float32x4_t a = vld1q_f32(acc + 4 * k);
          a = vfmaq_f32(a, scale_v, centroid);
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
