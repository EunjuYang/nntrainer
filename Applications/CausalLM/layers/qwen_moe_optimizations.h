// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   qwen_moe_optimizations.h
 * @date   09 June 2025
 * @brief  SIMD and performance optimizations for MoE layer
 * @author Optimized version
 */

#ifndef __QWEN_MOE_OPTIMIZATIONS_H__
#define __QWEN_MOE_OPTIMIZATIONS_H__

#include <immintrin.h>
#include <cstddef>
#include <vector>

namespace causallm {

// Optimized SIMD-based matrix multiplication for small matrices
inline void optimized_gemm_small(const float* A, const float* B, float* C,
                                 size_t M, size_t N, size_t K) {
  // Use AVX2 for vectorized operations
  const size_t simd_width = 8; // AVX2 processes 8 floats at once
  
  for (size_t i = 0; i < M; ++i) {
    size_t j = 0;
    
    // Process 8 elements at a time with AVX2
    for (; j + simd_width <= N; j += simd_width) {
      __m256 sum = _mm256_setzero_ps();
      
      for (size_t k = 0; k < K; ++k) {
        __m256 a = _mm256_broadcast_ss(&A[i * K + k]);
        __m256 b = _mm256_loadu_ps(&B[k * N + j]);
        sum = _mm256_fmadd_ps(a, b, sum);
      }
      
      _mm256_storeu_ps(&C[i * N + j], sum);
    }
    
    // Handle remaining elements
    for (; j < N; ++j) {
      float sum = 0.0f;
      for (size_t k = 0; k < K; ++k) {
        sum += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = sum;
    }
  }
}

// Optimized batched matrix multiplication
inline void batched_gemm_optimized(const std::vector<const float*>& A_batch,
                                   const float* B,
                                   std::vector<float*>& C_batch,
                                   size_t batch_size, size_t M, size_t N, size_t K) {
  #pragma omp parallel for schedule(dynamic, 1)
  for (size_t b = 0; b < batch_size; ++b) {
    optimized_gemm_small(A_batch[b], B, C_batch[b], M, N, K);
  }
}

// SIMD-optimized vector addition
inline void simd_vector_add(float* dst, const float* src, size_t size) {
  const size_t simd_width = 8;
  size_t simd_iterations = size / simd_width;
  
  // Process 8 elements at a time
  for (size_t i = 0; i < simd_iterations; ++i) {
    __m256 dst_vec = _mm256_loadu_ps(dst + i * simd_width);
    __m256 src_vec = _mm256_loadu_ps(src + i * simd_width);
    dst_vec = _mm256_add_ps(dst_vec, src_vec);
    _mm256_storeu_ps(dst + i * simd_width, dst_vec);
  }
  
  // Handle remaining elements
  for (size_t i = simd_iterations * simd_width; i < size; ++i) {
    dst[i] += src[i];
  }
}

// SIMD-optimized vector multiplication by scalar
inline void simd_vector_scale(float* vec, float scalar, size_t size) {
  const size_t simd_width = 8;
  size_t simd_iterations = size / simd_width;
  __m256 scalar_vec = _mm256_set1_ps(scalar);
  
  // Process 8 elements at a time
  for (size_t i = 0; i < simd_iterations; ++i) {
    __m256 vec_data = _mm256_loadu_ps(vec + i * simd_width);
    vec_data = _mm256_mul_ps(vec_data, scalar_vec);
    _mm256_storeu_ps(vec + i * simd_width, vec_data);
  }
  
  // Handle remaining elements
  for (size_t i = simd_iterations * simd_width; i < size; ++i) {
    vec[i] *= scalar;
  }
}

// Cache-friendly blocked matrix multiplication
inline void blocked_gemm(const float* A, const float* B, float* C,
                        size_t M, size_t N, size_t K,
                        size_t block_size = 64) {
  // Zero out C
  std::fill(C, C + M * N, 0.0f);
  
  // Blocked multiplication for better cache utilization
  for (size_t i = 0; i < M; i += block_size) {
    for (size_t k = 0; k < K; k += block_size) {
      for (size_t j = 0; j < N; j += block_size) {
        // Compute block
        size_t i_end = std::min(i + block_size, M);
        size_t k_end = std::min(k + block_size, K);
        size_t j_end = std::min(j + block_size, N);
        
        for (size_t ii = i; ii < i_end; ++ii) {
          for (size_t kk = k; kk < k_end; ++kk) {
            float a_val = A[ii * K + kk];
            for (size_t jj = j; jj < j_end; ++jj) {
              C[ii * N + jj] += a_val * B[kk * N + jj];
            }
          }
        }
      }
    }
  }
}

} // namespace causallm

#endif // __QWEN_MOE_OPTIMIZATIONS_H__