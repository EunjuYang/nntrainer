// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 nntrainer authors
 *
 * @file   avx2_turboquant.h
 * @date   16 April 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  AVX2 accelerated TurboQuant KV-cache kernels.
 *
 *         These entry points mirror the public TurboQuant API exposed in
 *         cpu_backend.h. Only head_dim in {64, 128} is accelerated; callers
 *         are expected to fall back to the scalar reference for any other
 *         head_dim (the x86 dispatcher in x86_compute_backend.cpp handles
 *         this).
 */
#ifndef __AVX2_TURBOQUANT_H__
#define __AVX2_TURBOQUANT_H__
#ifdef __cplusplus

#include <cstddef>
#include <cstdint>

namespace nntrainer::avx2 {

/**
 * @copydoc quantize_kv_turboquant in cpu_backend.h
 */
void quantize_kv_turboquant(const float *input, uint8_t *out_packed,
                            float *out_norms, const float *rot_signs,
                            int head_dim, int num_heads);

/**
 * @copydoc compute_kcaches_packed4 in cpu_backend.h
 */
void compute_kcaches_packed4(const float *query, const uint8_t *kcache_packed,
                             const float *kcache_norms, float *output,
                             int num_rows, int num_cache_head, int head_dim,
                             int gqa_size, int tile_size,
                             const float *rot_signs, size_t local_window_size,
                             int head_start, int head_end);

/**
 * @copydoc compute_vcache_packed4 in cpu_backend.h
 */
void compute_vcache_packed4(int row_num, const float *attn_weights,
                            const uint8_t *vcache_packed,
                            const float *vcache_norms, float *output,
                            int num_cache_head, int gqa_size, int head_dim,
                            const float *rot_signs, size_t local_window_size,
                            int head_start, int head_end);

} // namespace nntrainer::avx2

#endif /* __cplusplus */
#endif /* __AVX2_TURBOQUANT_H__ */
