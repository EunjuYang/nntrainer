// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   x86_compute_backend.cpp
 * @date   23 April 2024
 * @see    https://github.com/nntrainer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Compute backend for x86
 *
 */

#include <assert.h>

#include <avx2_impl.h>
#ifdef USE_BLAS
#include <cblas_interface.h>
#endif
#include <fallback_internal.h>
#include <ggml_interface.h>
#include <immintrin.h>
#include <nntrainer_error.h>
#include <q4_0_utils.h>
#include <turboquant_utils.h>
#include <x86_compute_backend.h>

#include <vector>

#define ROW_MAJOR 0
#define COL_MAJOR 1

namespace nntrainer {

void init_backend() {
  __ggml_init();
  // Do not repeatedly call set_num_threads. It's a global config.
  __openblas_set_num_threads(-1); // -1 = BLAS_NUM_THREADS if defined.
}

void scopy_int4_to_float32(const unsigned int N, const uint8_t *X,
                           const unsigned int incX, float *Y,
                           const unsigned int incY) {
  __fallback_scopy_int4_to_float32(N, X, incX, Y, incY);
}

void copy_s16(const unsigned int N, const int16_t *X, int16_t *Y) {
  __fallback_copy_s16(N, X, Y);
}

void copy_u16(const unsigned int N, const uint16_t *X, uint16_t *Y) {
  __fallback_copy_u16(N, X, Y);
}

void copy_s16_fp32(const unsigned int N, const int16_t *X, float *Y) {
  __fallback_copy_s16_fp32(N, X, Y);
}

void copy_u16_fp32(const unsigned int N, const uint16_t *X, float *Y) {
  nntrainer::avx2::copy_f16_f32(N, X, Y);
}

void copy_fp32_u32(const unsigned int N, const float *X, uint32_t *Y) {
  __fallback_copy_fp32_u32(N, X, Y);
}

void copy_fp32_u16(const unsigned int N, const float *X, uint16_t *Y) {
  nntrainer::avx2::copy_f32_f16(N, X, Y);
}

void copy_fp32_u8(const unsigned int N, const float *X, uint8_t *Y) {
  __fallback_copy_fp32_u8(N, X, Y);
}

void copy_fp32_s16(const unsigned int N, const float *X, int16_t *Y) {
  __fallback_copy_fp32_s16(N, X, Y);
}

void copy_fp32_s8(const unsigned int N, const float *X, int8_t *Y) {
  __fallback_copy_fp32_s8(N, X, Y);
}

/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 * @param[in] Y uint32_t * for Vector Y
 */
template <> void copy_fp32(const unsigned int N, const float *X, uint32_t *Y) {
  copy_fp32_u32(N, X, Y);
}

/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 * @param[in] Y uint16_t * for Vector Y
 */
template <> void copy_fp32(const unsigned int N, const float *X, uint16_t *Y) {
  copy_fp32_u16(N, X, Y);
}

/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 * @param[in] Y uint16_t * for Vector Y
 */
template <> void copy_fp32(const unsigned int N, const float *X, uint8_t *Y) {
  copy_fp32_u8(N, X, Y);
}

/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 * @param[in] Y int16_t * for Vector Y
 */
template <> void copy_fp32(const unsigned int N, const float *X, int16_t *Y) {
  copy_fp32_s16(N, X, Y);
}

/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 * @param[in] Y int8_t * for Vector Y
 */
template <> void copy_fp32(const unsigned int N, const float *X, int8_t *Y) {
  copy_fp32_s8(N, X, Y);
}

void scopy_int8_to_float32(const unsigned int N, const uint8_t *X,
                           const unsigned int incX, float *Y,
                           const unsigned int incY) {
  __fallback_scopy_uint8_to_float32(N, X, incX, Y, incY);
}

void scopy_int8_to_float32(const unsigned int N, const int8_t *X,
                           const unsigned int incX, float *Y,
                           const unsigned int incY) {
  __fallback_scopy_int8_to_float32(N, X, incX, Y, incY);
}

template <>
void sine(const unsigned int N, float *X, float *Y, float alpha, float beta) {
  __fallback_sine(N, X, Y, alpha, beta);
}

template <>
void cosine(const unsigned int N, float *X, float *Y, float alpha, float beta) {
  __fallback_cosine(N, X, Y, alpha, beta);
}

void inv_sqrt_inplace(const unsigned int N, float *X) {
  __fallback_inv_sqrt_inplace(N, X);
}

void ele_mul(const unsigned int N, const float *X, const float *Y, float *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  nntrainer::avx2::ele_mul(N, X, Y, Z, alpha, beta, i_stride, o_stride);
}

void ele_add(const unsigned int N, const float *X, const float *Y, float *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  nntrainer::avx2::ele_add(N, X, Y, Z, alpha, beta, i_stride, o_stride);
}

void ele_sub(const unsigned N, const float *X, const float *Y, float *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  __fallback_ele_sub(N, X, Y, Z, alpha, beta, i_stride, o_stride);
}

void ele_div(const unsigned N, const float *X, const float *Y, float *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  __fallback_ele_div(N, X, Y, Z, alpha, beta, i_stride, o_stride);
}

void saxpy(const unsigned int N, const float alpha, const float *X,
           const unsigned int incX, float *Y, const unsigned int incY) {
#ifdef USE_BLAS
  __cblas_saxpy(N, alpha, X, incX, Y, incY);
#else
  __fallback_saxpy(N, alpha, X, incX, Y, incY);
#endif
}

void sgemv(const unsigned int TStorageOrder, bool TransA, const unsigned int M,
           const unsigned int N, const float alpha, const float *A,
           const unsigned int lda, const float *X, const unsigned int incX,
           const float beta, float *Y, const unsigned int incY) {
#ifdef USE_BLAS
  __cblas_sgemv(TStorageOrder, TransA, M, N, alpha, A, lda, X, incX, beta, Y,
                incY);
#else
  __fallback_sgemv(TStorageOrder, TransA, M, N, alpha, A, lda, X, incX, beta, Y,
                   incY);
#endif
}

float sdot(const unsigned int N, const float *X, const unsigned int incX,
           const float *Y, const unsigned int incY) {
#ifdef USE_BLAS
  return __cblas_sdot(N, X, incX, Y, incY);
#else
  return __fallback_sdot(N, X, incX, Y, incY);
#endif
}

void scopy(const unsigned int N, const uint8_t *X, const unsigned int incX,
           uint8_t *Y, const unsigned int incY) {
  __fallback_scopy(N, X, incX, Y, incY);
}

void scopy(const unsigned int N, const int8_t *X, const unsigned int incX,
           int8_t *Y, const unsigned int incY) {
  __fallback_scopy(N, X, incX, Y, incY);
}

void scopy(const unsigned int N, const float *X, const unsigned int incX,
           float *Y, const unsigned int incY) {
  /// @note cblas_scopy is evoking SIGSEGV for some reason. Use custom
  /// implementation instead.
  // __cblas_scopy(N, X, incX, Y, incY);
  nntrainer::avx2::custom_scopy(N, X, incX, Y, incY);
}

void sscal(const unsigned int N, const float alpha, float *X,
           const unsigned int incX) {
#ifdef USE_BLAS
  __cblas_sscal(N, alpha, X, incX);
#else
  __fallback_sscal(N, alpha, X, incX);
#endif
}

float snrm2(const unsigned int N, const float *X, const unsigned int incX) {
#ifdef USE_BLAS
  return __cblas_snrm2(N, X, incX);
#else
  return __fallback_snrm2(N, X, incX);
#endif
}

void sgemm(const unsigned int TStorageOrder, bool TransA, bool TransB,
           const unsigned int M, const unsigned int N, const unsigned int K,
           const float alpha, const float *A, const unsigned int lda,
           const float *B, const unsigned int ldb, const float beta, float *C,
           const unsigned int ldc) {
#ifdef USE_BLAS
  __cblas_sgemm(TStorageOrder, TransA, TransB, M, N, K, alpha, A, lda, B, ldb,
                beta, C, ldc);
#else
  __fallback_sgemm(TStorageOrder, TransA, TransB, M, N, K, alpha, A, lda, B,
                   ldb, beta, C, ldc);
#endif
}

unsigned int isamax(const unsigned int N, const float *X,
                    const unsigned int incX) {
#ifdef USE_BLAS
  return __cblas_isamax(N, X, incX);
#else
  return __fallback_isamax(N, X, incX);
#endif
}
void transpose_matrix(const unsigned int M, const unsigned int N,
                      const float *src, unsigned int ld_src, float *dst,
                      unsigned int ld_dst) {
  nntrainer::avx2::transpose_matrix(M, N, src, ld_src, dst, ld_dst);
}

bool is_valid(const unsigned int N, const float *input) {
  return nntrainer::avx2::is_valid(N, input);
}

void unpack_q4_0x8_transpose16(const void *src, uint16_t *d_out,
                               uint16_t *qs_out, int N, int K) {
  return nntrainer::avx2::unpack_q4_0x8_transpose16(src, d_out, qs_out, N, K);
}

template <>
void calc_trigonometric_vals_dup(unsigned int N_half, float *angle, float *cos_,
                                 float *sin_, unsigned int from,
                                 float attention_scaling) {
  __fallback_calc_trigonometric_vals_dup(N_half, angle, cos_, sin_, from,
                                         attention_scaling);
}

void swiglu(const unsigned int N, float *X, float *Y, float *Z) {
  nntrainer::avx2::swiglu(N, X, Y, Z);
}

void swiglu(const unsigned int N, float *X, float *Y, float *Z, float alpha) {
  nntrainer::avx2::swiglu(N, X, Y, Z, alpha);
}

void tanh_gelu(const unsigned int N, const float *X, float *Y) {
  // AVX implmenetation will be implemented, now fallback instead
  __fallback_tanh_gelu(N, X, Y);
}

void tanh_gelu_v2(const unsigned int N, const float *X, float *Y) {
  __fallback_tanh_gelu(N, X, Y);
}

void gelu_v2(const unsigned int N, const float *X, float *Y) {
  __fallback_gelu_v2(N, X, Y);
}

void tanh_gelu_mul(const unsigned int N, float *X, float *Y, float *Z) {
  __fallback_tanh_gelu_mul(N, X, Y, Z);
}

void tanh_gelu_v2_mul(const unsigned int N, float *X, float *Y, float *Z) {
  __fallback_tanh_gelu_mul(N, X, Y, Z);
}

float max_val(const unsigned int N, float *X) { return __fallback_max(N, X); }

void softmax(const unsigned int N, float *X, float *Y) {
  __fallback_softmax(N, X, Y);
}

template <>
void gemm_q4_0(const unsigned int M, const unsigned int N, const unsigned int K,
               const float *A, const unsigned int lda, const void *B,
               const unsigned int ldb, float *C, const unsigned int ldc) {
  return __ggml_q4_0_8x8_q8_0_GEMM(M, N, K, A, lda, B, ldb, C, ldc);
}

void gemm_q4_0(const unsigned int M, std::vector<unsigned int> Ns,
               const unsigned int K, const float *A, const unsigned int lda,
               std::vector<void *> Bs, std::vector<unsigned int> ldbs,
               std::vector<float *> Cs, std::vector<unsigned int> ldcs) {
  throw std::runtime_error("Error: NYI for gemm_q4_0 with vectored weights");
}

void gemm_q4_K(const unsigned int M, const unsigned int N, const unsigned int K,
               const float *A, const unsigned int lda, const void *B,
               const unsigned int ldb, float *C, const unsigned int ldc) {
  return __ggml_q4_K_8x8_q8_K_GEMM(M, N, K, A, lda, B, ldb, C, ldc);
}

void gemm_q4_K(const unsigned int M, std::vector<unsigned int> Ns,
               const unsigned int K, const float *A, const unsigned int lda,
               std::vector<void *> Bs, std::vector<unsigned int> ldbs,
               std::vector<float *> Cs, std::vector<unsigned int> ldcs) {
  return __ggml_q4_K_8x8_q8_K_GEMM(M, Ns, K, A, lda, Bs, ldbs, Cs, ldcs);
}

float dot_q6_K_q8_K(const unsigned int K, const void *v_q6_K,
                    const void *v_q8_K) {
  return __ggml_vec_dot_q6_K_q8_K(K, v_q6_K, v_q8_K);
}

float dot_q6_K_f32(const unsigned int K, const void *v_q6_K, const float *f) {
  return __ggml_vec_dot_q6_K_f32(K, v_q6_K, f);
}

template <>
void gemm_q6_K(const unsigned int M, const unsigned int N, const unsigned int K,
               const float *A, const unsigned int lda, const void *B,
               const unsigned int ldb, float *C, const unsigned int ldc) {
  return __ggml_gemm_q6_K(M, N, K, A, lda, B, ldb, C, ldc);
}

size_t quantize_q4_0(const float *src, void *dst, int64_t nrow,
                     int64_t n_per_row, const float *quant_weights) {
  return __ggml_quantize_q4_0(src, dst, nrow, n_per_row, quant_weights);
}

size_t quantize_q4_K(const float *src, void *dst, int64_t nrow,
                     int64_t n_per_row, const float *quant_weights) {
  return __ggml_quantize_q4_K(src, dst, nrow, n_per_row, quant_weights);
}

size_t quantize_q6_K(const float *src, void *dst, int64_t nrow,
                     int64_t n_per_row, const float *quant_weights) {
  return __ggml_quantize_q6_K(src, dst, nrow, n_per_row, quant_weights);
}

void quantize_row_q6_K(const float *src, void *dst, int64_t k) {
  __ggml_quantize_row_q6_K(src, dst, k);
}

template <> void quantize_row_q8_K(const float *src, void *dst, int64_t k) {
  __ggml_quantize_row_q8_K(src, dst, k);
}

void dequantize_row_q4_K(const void *x_raw, float *y, int64_t k) {
  __ggml_dequantize_row_q4_K(x_raw, y, k);
}

void dequantize_row_q4_0(const void *x_raw, float *y, int64_t k) {
  __ggml_dequantize_row_q4_0(x_raw, y, k);
}

void dequantize_row_q6_K(const void *x, float *y, int64_t k) {
  __ggml_dequantize_row_q6_K(x, y, k);
}

template <> void dequantize_row_q8_K(const void *x, float *y, int64_t k) {
  __ggml_dequantize_row_q8_K(x, y, k);
}

void repack_q4_0(void *W, void *repacked_W, size_t data_size,
                 const unsigned int M, const unsigned int N) {
  __ggml_repack_q4_0_to_q4_0_8(W, repacked_W, data_size, M, N);
}

void repack_q4_0_to_q4_0_8(void *W, void *repacked_W, size_t data_size,
                           const unsigned int M, const unsigned int N) {
  __ggml_repack_q4_0_to_q4_0_8(W, repacked_W, data_size, M, N);
}

void repack_q4_K(void *W, void *repacked_W, size_t data_size,
                 const unsigned int M, const unsigned int N) {
  __ggml_repack_q4_K_to_q4_K_8(W, repacked_W, data_size, M, N);
}

void unpack_q4_0(const void *in_q4_0x, void *out_q4_0, size_t data_size,
                 const unsigned int M, const unsigned int N) {
  Q4_0Utils::unpackBlocksQ4_0x8((const block_q4_0x8 *)in_q4_0x, data_size, M, N,
                                (block_q4_0 *)out_q4_0);
}

template <>
void softmax_row_inplace(float *qk_out, size_t start_row, size_t end_row,
                         size_t num_heads, float *sink) {
  nntrainer::avx2::softmax_row_inplace<float>(qk_out, start_row, end_row,
                                              num_heads, sink);
}

template <>
void softmax_row(float *qk_out, size_t start_row, size_t end_row,
                 size_t num_heads, float *sink) {
  nntrainer::avx2::softmax_row<float>(qk_out, start_row, end_row, num_heads,
                                      sink);
}

void compute_fp16vcache_fp32_transposed(int row_num, const float *in,
                                        const uint16_t *vcache, float *output,
                                        int num_cache_head, int gqa_size,
                                        int head_dim, size_t local_window_size,
                                        int head_start, int head_end) {
  nntrainer::avx2::compute_fp16vcache_fp32_transposed(
    row_num, in, vcache, output, num_cache_head, gqa_size, head_dim,
    local_window_size, head_start, head_end);
}

template <>
void compute_kcaches(const float *in, const uint16_t *kcache, float *output,
                     int num_rows, int num_cache_head, int head_dim,
                     int gqa_size, int tile_size, size_t local_window_size,
                     int head_start, int head_end) {
  nntrainer::avx2::compute_kcaches<uint16_t>(
    in, kcache, output, num_rows, num_cache_head, head_dim, gqa_size, tile_size,
    local_window_size, head_start, head_end);
}

void compute_rotary_emb_value(unsigned int width, unsigned int dim,
                              unsigned int half_, float *inout, void *output,
                              const float *cos_, const float *sin_,
                              bool only_convert_to_fp16) {
  nntrainer::avx2::compute_rotary_emb_value(width, dim, half_, inout, output,
                                            cos_, sin_, only_convert_to_fp16);
}

void rms_norm_wrt_width_fp32_intrinsic(const float *__restrict X,
                                       float *__restrict Y, size_t H, size_t W,
                                       float epsilon) {
  nntrainer::avx2::rms_norm_wrt_width_fp32_intrinsic(X, Y, H, W, epsilon);
}

template <>
void rms_norm_wrt_width_fp16_intrinsic(const float *__restrict X,
                                       float *__restrict Y, size_t H, size_t W,
                                       float epsilon) {
  __fallback_rms_norm_wrt_width_fp16_intrinsic(X, Y, H, W, epsilon);
}

template <>
void clamp(const float *input, float *output, size_t length, float lower_bound,
           float upper_bound) {
  nntrainer::avx2::clamp(input, output, length, lower_bound, upper_bound);
}

void create_q4_0_weights(const uint8_t *int4_weight, uint8_t *q4_0_weight) {
  nntrainer::avx2::create_q4_0_weights(int4_weight, q4_0_weight);
}

void transform_int4_osv32_isv2_to_q4_0(size_t N, size_t K,
                                       const uint8_t *osv32_weights,
                                       const uint16_t *osv32_scales,
                                       size_t scale_group_size,
                                       void *dst_q4_0x) {
#ifdef __AVX2__
  nntrainer::avx2::transform_int4_osv32_isv2_to_q4_0x8(
    N, K, osv32_weights, osv32_scales, scale_group_size, dst_q4_0x);
#else
  __fallback_transform_int4_osv32_isv2_to_q4_0(
    N, K, osv32_weights, osv32_scales, scale_group_size, 8, dst_q4_0x);
#endif
}
void quantize_kv_turboquant(const float *input, size_t num_elements,
                            uint8_t *out_packed, float *out_scales) {
#ifdef __AVX2__
  nntrainer::avx2::quantize_kv_turboquant(input, num_elements, out_packed,
                                          out_scales);
#else
  __fallback_quantize_kv_turboquant(input, num_elements, out_packed, out_scales);
#endif
}

void compute_kcaches_packed4(const float *query, const uint8_t *kcache_packed,
                             const float *kcache_scales, float *output,
                             int num_rows, int num_cache_head, int head_dim,
                             int gqa_size, int tile_size,
                             size_t local_window_size, int head_start,
                             int head_end) {
#ifdef __AVX2__
  nntrainer::avx2::compute_kcaches_packed4(
    query, kcache_packed, kcache_scales, output, num_rows, num_cache_head,
    head_dim, gqa_size, tile_size, local_window_size, head_start, head_end);
#else
  __fallback_compute_kcaches_packed4(query, kcache_packed, kcache_scales, output,
                                     num_rows, num_cache_head, head_dim,
                                     gqa_size, tile_size, local_window_size,
                                     head_start, head_end);
#endif
}

void compute_vcache_packed4_transposed(int row_num, const float *attn_weights,
                                       const uint8_t *vcache_packed,
                                       const float *vcache_scales,
                                       float *output, int num_cache_head,
                                       int gqa_size, int head_dim,
                                       size_t local_window_size, int head_start,
                                       int head_end) {
#ifdef __AVX2__
  nntrainer::avx2::compute_vcache_packed4_transposed(
    row_num, attn_weights, vcache_packed, vcache_scales, output, num_cache_head,
    gqa_size, head_dim, local_window_size, head_start, head_end);
#else
  __fallback_compute_vcache_packed4_transposed(
    row_num, attn_weights, vcache_packed, vcache_scales, output, num_cache_head,
    gqa_size, head_dim, local_window_size, head_start, head_end);
#endif
}

void quantize_kv_turboquant_rotated(const float *input, size_t num_elements,
                                    uint8_t *out_packed, float *out_scales,
                                    const float *signs, int head_dim,
                                    int num_heads) {
  __fallback_quantize_kv_turboquant_rotated(input, num_elements, out_packed,
                                            out_scales, signs, head_dim,
                                            num_heads);
}

void compute_kcaches_packed4_rotated(
  const float *query, const uint8_t *kcache_packed, const float *kcache_scales,
  float *output, int num_rows, int num_cache_head, int head_dim, int gqa_size,
  int tile_size, const float *signs, size_t local_window_size, int head_start,
  int head_end) {
  __fallback_compute_kcaches_packed4_rotated(
    query, kcache_packed, kcache_scales, output, num_rows, num_cache_head,
    head_dim, gqa_size, tile_size, signs, local_window_size, head_start,
    head_end);
}

void compute_vcache_packed4_transposed_rotated(
  int row_num, const float *attn_weights, const uint8_t *vcache_packed,
  const float *vcache_scales, float *output, int num_cache_head, int gqa_size,
  int head_dim, const float *signs, size_t local_window_size, int head_start,
  int head_end) {
  __fallback_compute_vcache_packed4_transposed_rotated(
    row_num, attn_weights, vcache_packed, vcache_scales, output, num_cache_head,
    gqa_size, head_dim, signs, local_window_size, head_start, head_end);
}

/** AVX2 horizontal sum helper (local to this file). */
static inline float hsum_avx2(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  lo = _mm_add_ps(lo, hi);
  __m128 shuf = _mm_movehdup_ps(lo);
  __m128 sums = _mm_add_ps(lo, shuf);
  shuf = _mm_movehl_ps(shuf, sums);
  sums = _mm_add_ss(sums, shuf);
  return _mm_cvtss_f32(sums);
}

/**
 * AVX2 Hadamard transform for power-of-2 sizes.
 * Uses AVX2 for the inner butterfly when block size >= 8.
 */
static void hadamard_transform_avx2(float *x, int n) {
  for (int len = 1; len < 8 && len < n; len <<= 1) {
    for (int i = 0; i < n; i += len << 1) {
      for (int j = 0; j < len; ++j) {
        float u = x[i + j];
        float v = x[i + j + len];
        x[i + j] = u + v;
        x[i + j + len] = u - v;
      }
    }
  }
  for (int len = 8; len < n; len <<= 1) {
    for (int i = 0; i < n; i += len << 1) {
      for (int j = 0; j < len; j += 8) {
        __m256 u = _mm256_loadu_ps(x + i + j);
        __m256 v = _mm256_loadu_ps(x + i + j + len);
        _mm256_storeu_ps(x + i + j, _mm256_add_ps(u, v));
        _mm256_storeu_ps(x + i + j + len, _mm256_sub_ps(u, v));
      }
    }
  }
  float inv_sqrt_n = 1.0f / std::sqrt((float)n);
  __m256 vinv = _mm256_set1_ps(inv_sqrt_n);
  int i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256 v = _mm256_loadu_ps(x + i);
    _mm256_storeu_ps(x + i, _mm256_mul_ps(v, vinv));
  }
  for (; i < n; ++i)
    x[i] *= inv_sqrt_n;
}

/**
 * AVX2 nibble unpack: 4 packed bytes → 8 x int32 nibble indices.
 */
static inline __m256i avx2_unpack_nibbles_4bytes(const uint8_t *packed) {
  uint32_t raw;
  std::memcpy(&raw, packed, 4);
  return _mm256_setr_epi32(
    raw & 0x0F, (raw >> 4) & 0x0F, (raw >> 8) & 0x0F, (raw >> 12) & 0x0F,
    (raw >> 16) & 0x0F, (raw >> 20) & 0x0F, (raw >> 24) & 0x0F,
    (raw >> 28) & 0x0F);
}

/** AVX2 centroid lookup using gather instruction. */
static inline __m256 avx2_centroid_lookup(__m256i idx,
                                          const float *centroids) {
  return _mm256_i32gather_ps(centroids, idx, 4);
}

/**
 * AVX2 centroid dot product with dual accumulators for ILP.
 */
static inline float avx2_centroid_dot(const float *rq, const uint8_t *packed,
                                      const float *centroids, int head_dim) {
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();

  int d = 0;
  for (; d + 16 <= head_dim; d += 16) {
    __m256i idx0 = avx2_unpack_nibbles_4bytes(packed + d / 2);
    __m256i idx1 = avx2_unpack_nibbles_4bytes(packed + d / 2 + 4);
    __m256 vals0 = avx2_centroid_lookup(idx0, centroids);
    __m256 vals1 = avx2_centroid_lookup(idx1, centroids);
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(rq + d), vals0, acc0);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(rq + d + 8), vals1, acc1);
  }
  for (; d + 8 <= head_dim; d += 8) {
    __m256i idx = avx2_unpack_nibbles_4bytes(packed + d / 2);
    __m256 vals = avx2_centroid_lookup(idx, centroids);
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(rq + d), vals, acc0);
  }

  float sum = hsum_avx2(_mm256_add_ps(acc0, acc1));
  for (; d + 2 <= head_dim; d += 2) {
    uint8_t byte = packed[d / 2];
    sum += rq[d] * centroids[byte & 0x0F];
    sum += rq[d + 1] * centroids[(byte >> 4) & 0x0F];
  }
  return sum;
}

void quantize_kv_turboquant_v2(const float *input, uint8_t *out_packed,
                               float *out_norms, const float *rot_signs,
                               int head_dim, int num_heads) {
  const LloydMaxCodebook &cb = get_codebook(head_dim);
  std::vector<float> rotated(head_dim);

  for (int h = 0; h < num_heads; ++h) {
    const float *head_in = input + h * head_dim;
    uint8_t *head_out = out_packed + h * head_dim / 2;

    // 1. Compute norm with AVX2
    __m256 norm_acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= head_dim; i += 8) {
      __m256 v = _mm256_loadu_ps(head_in + i);
      norm_acc = _mm256_fmadd_ps(v, v, norm_acc);
    }
    float norm_sq = hsum_avx2(norm_acc);
    for (; i < head_dim; ++i)
      norm_sq += head_in[i] * head_in[i];
    float norm = std::sqrt(norm_sq);
    out_norms[h] = norm;

    // 2. Normalize + multiply by rot_signs (AVX2)
    float inv_norm = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;
    __m256 vinv = _mm256_set1_ps(inv_norm);
    i = 0;
    for (; i + 8 <= head_dim; i += 8) {
      __m256 v = _mm256_loadu_ps(head_in + i);
      __m256 s = _mm256_loadu_ps(rot_signs + i);
      _mm256_storeu_ps(rotated.data() + i, _mm256_mul_ps(_mm256_mul_ps(v, vinv), s));
    }
    for (; i < head_dim; ++i)
      rotated[i] = head_in[i] * inv_norm * rot_signs[i];

    // 3. Hadamard transform (AVX2)
    hadamard_transform_avx2(rotated.data(), head_dim);

    // 4. Lloyd-Max quantize + pack (scalar, since boundary search is branchy)
    for (int d = 0; d < head_dim; d += 2) {
      uint8_t q0 = lloydmax_quantize(rotated[d], cb);
      uint8_t q1 = 8;
      if (d + 1 < head_dim)
        q1 = lloydmax_quantize(rotated[d + 1], cb);
      head_out[d / 2] = (q1 << 4) | q0;
    }
  }
}

void compute_kcaches_packed4_v2(
  const float *query, const uint8_t *kcache_packed, const float *kcache_norms,
  float *output, int num_rows, int num_cache_head, int head_dim, int gqa_size,
  int tile_size, const float *rot_signs, size_t local_window_size,
  int head_start, int head_end) {
  const LloydMaxCodebook &cb = get_codebook(head_dim);
  int actual_head_end = (head_end < 0) ? num_cache_head : head_end;
  int start_row =
    (size_t)num_rows < local_window_size ? 0 : num_rows - local_window_size;
  int row_cnt =
    (size_t)num_rows < local_window_size ? num_rows : local_window_size;
  int packed_row_bytes = num_cache_head * head_dim / 2;
  float inv_sqrt_d = 1.0f / std::sqrt((float)head_dim);

  // Pre-rotate queries: rotated_Q = H(D*Q)
  std::vector<float> rotated_queries(gqa_size * head_dim);

  for (int n = head_start; n < actual_head_end; ++n) {
    // Rotate all GQA queries for this KV head
    for (int g = 0; g < gqa_size; ++g) {
      const float *q_ptr = query + n * gqa_size * head_dim + g * head_dim;
      float *rq = rotated_queries.data() + g * head_dim;

      // AVX2 element-wise multiply by rot_signs
      int i = 0;
      for (; i + 8 <= head_dim; i += 8) {
        __m256 vq = _mm256_loadu_ps(q_ptr + i);
        __m256 vs = _mm256_loadu_ps(rot_signs + i);
        _mm256_storeu_ps(rq + i, _mm256_mul_ps(vq, vs));
      }
      for (; i < head_dim; ++i)
        rq[i] = q_ptr[i] * rot_signs[i];

      hadamard_transform_avx2(rq, head_dim);
    }

    for (int t_row = 0; t_row < row_cnt; ++t_row) {
      int row = start_row + t_row;
      const uint8_t *packed_ptr =
        kcache_packed + row * packed_row_bytes + n * head_dim / 2;
      float norm = kcache_norms[row * num_cache_head + n];

      for (int g = 0; g < gqa_size; ++g) {
        const float *rq = rotated_queries.data() + g * head_dim;
        float sum = avx2_centroid_dot(rq, packed_ptr, cb.centroids, head_dim);
        output[t_row * num_cache_head * gqa_size + n * gqa_size + g] =
          sum * norm * inv_sqrt_d;
      }
    }
  }
}
void compute_vcache_packed4_v2(
  int row_num, const float *attn_weights, const uint8_t *vcache_packed,
  const float *vcache_norms, float *output, int num_cache_head, int gqa_size,
  int head_dim, const float *rot_signs, size_t local_window_size,
  int head_start, int head_end) {
  const LloydMaxCodebook &cb = get_codebook(head_dim);
  int actual_head_end = (head_end < 0) ? num_cache_head : head_end;
  int packed_row_bytes = num_cache_head * head_dim / 2;
  int j_start = (size_t)row_num < local_window_size
                  ? 0
                  : row_num + 1 - (int)local_window_size;

  int num_blocks = head_dim / 8;
  int rem = head_dim % 8;

  std::vector<float> acc(head_dim);

  for (int n = head_start; n < actual_head_end; ++n) {
    for (int h = 0; h < gqa_size; ++h) {
      // AVX2 accumulators
      std::vector<__m256> sumVec(num_blocks, _mm256_setzero_ps());
      // Scalar accumulators for remainder
      float sumRem[8] = {};

      for (int j = j_start; j <= row_num; ++j) {
        float a_val =
          attn_weights[((j - j_start) * num_cache_head + n) * gqa_size + h];
        float norm = vcache_norms[j * num_cache_head + n];
        float scale = a_val * norm;
        __m256 vscale = _mm256_set1_ps(scale);

        const uint8_t *packed_ptr =
          vcache_packed + j * packed_row_bytes + n * head_dim / 2;

        for (int b = 0; b < num_blocks; ++b) {
          int d = b * 8;
          __m256i idx = avx2_unpack_nibbles_4bytes(packed_ptr + d / 2);
          __m256 vals = avx2_centroid_lookup(idx, cb.centroids);
          sumVec[b] = _mm256_fmadd_ps(vscale, vals, sumVec[b]);
        }

        // Scalar remainder
        for (int r = 0; r < rem; r += 2) {
          int dd = num_blocks * 8 + r;
          uint8_t byte = packed_ptr[dd / 2];
          sumRem[r] += scale * cb.centroids[byte & 0x0F];
          if (r + 1 < rem)
            sumRem[r + 1] += scale * cb.centroids[(byte >> 4) & 0x0F];
        }
      }

      // Store accumulated centroids into acc buffer for Hadamard
      for (int b = 0; b < num_blocks; ++b) {
        _mm256_storeu_ps(acc.data() + b * 8, sumVec[b]);
      }
      for (int r = 0; r < rem; ++r) {
        acc[num_blocks * 8 + r] = sumRem[r];
      }

      // Single inverse rotation: output = D * H * acc
      hadamard_transform_avx2(acc.data(), head_dim);
      int out_base = (n * gqa_size + h) * head_dim;

      // AVX2 multiply by rot_signs and store
      int i = 0;
      for (; i + 8 <= head_dim; i += 8) {
        __m256 va = _mm256_loadu_ps(acc.data() + i);
        __m256 vs = _mm256_loadu_ps(rot_signs + i);
        _mm256_storeu_ps(output + out_base + i, _mm256_mul_ps(va, vs));
      }
      for (; i < head_dim; ++i)
        output[out_base + i] = acc[i] * rot_signs[i];
    }
  }
}

} /* namespace nntrainer */
