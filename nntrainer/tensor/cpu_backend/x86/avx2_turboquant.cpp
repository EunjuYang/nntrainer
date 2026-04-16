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
 *         Per-kernel AVX2 paths are introduced in follow-up commits; this
 *         skeleton simply delegates to the scalar fallback so the new
 *         translation unit can be wired into the build first.
 */

#include "avx2_turboquant.h"

#include <fallback_internal.h>

namespace nntrainer::avx2 {

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
  nntrainer::__fallback_compute_kcaches_packed4(
    query, kcache_packed, kcache_norms, output, num_rows, num_cache_head,
    head_dim, gqa_size, tile_size, rot_signs, local_window_size, head_start,
    head_end);
}

void compute_vcache_packed4(int row_num, const float *attn_weights,
                            const uint8_t *vcache_packed,
                            const float *vcache_norms, float *output,
                            int num_cache_head, int gqa_size, int head_dim,
                            const float *rot_signs, size_t local_window_size,
                            int head_start, int head_end) {
  nntrainer::__fallback_compute_vcache_packed4(
    row_num, attn_weights, vcache_packed, vcache_norms, output, num_cache_head,
    gqa_size, head_dim, rot_signs, local_window_size, head_start, head_end);
}

} // namespace nntrainer::avx2
