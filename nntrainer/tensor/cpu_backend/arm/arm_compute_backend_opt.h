// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Optimization for MoE Layer
 *
 * @file arm_compute_backend_opt.h
 * @brief Optimized ARM backend functions for MoE layer
 */

#ifndef __ARM_COMPUTE_BACKEND_OPT_H__
#define __ARM_COMPUTE_BACKEND_OPT_H__
#ifdef __cplusplus

#include <cstdint>
#include <vector>
#include <tensor_dim.h>

namespace nntrainer {

/**
 * @brief Optimized batch copy with NEON
 * @param batch_size Number of batches to copy
 * @param elem_size Size of each element batch
 * @param src_offsets Array of source offsets
 * @param src Source data pointer
 * @param dst Destination data pointer
 */
void batch_copy_indexed(size_t batch_size, size_t elem_size,
                        const unsigned int* src_offsets,
                        const float* src, float* dst);

/**
 * @brief Optimized weighted accumulation: dst += src * weight
 * @param N Number of elements
 * @param src Source data
 * @param dst Destination data  
 * @param weight Weight scalar
 */
void weighted_add_i(const unsigned int N, const float* src, float* dst, 
                    float weight);

/**
 * @brief Batch weighted accumulation with different weights
 * @param batch_size Number of batches
 * @param elem_size Size of each batch
 * @param src Source data
 * @param dst_offsets Destination offsets
 * @param dst Destination data
 * @param weights Array of weights for each batch
 */
void batch_weighted_add(size_t batch_size, size_t elem_size,
                        const float* src, const unsigned int* dst_offsets,
                        float* dst, const float* weights);

/**
 * @brief Optimized add for contiguous tensors
 * @param N Number of elements
 * @param X First input
 * @param Y Second input  
 * @param Z Output (can be same as X or Y for in-place)
 */
void add_contiguous(const unsigned int N, const float* X, const float* Y, 
                   float* Z);

/**
 * @brief Parallel GEMM for multiple small matrices
 * @param num_matrices Number of matrices to process
 * @param M Row dimension
 * @param N Column dimension  
 * @param K Inner dimension
 * @param A_list List of A matrices
 * @param B_list List of B matrices
 * @param C_list List of C matrices
 * @param alpha Scalar multiplier
 * @param beta Scalar for output
 */
void batch_gemm(unsigned int num_matrices, unsigned int M, unsigned int N,
               unsigned int K, const std::vector<const float*>& A_list,
               const std::vector<const float*>& B_list,
               std::vector<float*>& C_list, float alpha = 1.0f, 
               float beta = 0.0f);

/**
 * @brief Fused SwiGLU operation for batch processing
 * @param batch_size Number of batches
 * @param width Width of each batch
 * @param gate_out Gate output data
 * @param up_out Up projection output
 * @param output Output tensor
 */
void batch_swiglu(unsigned int batch_size, unsigned int width,
                 float* gate_out, float* up_out, float* output);

/**
 * @brief Memory prefetch for expert weights
 * @param addr Address to prefetch
 * @param size Size in bytes to prefetch
 */
void prefetch_weights(const void* addr, size_t size);

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __ARM_COMPUTE_BACKEND_OPT_H__ */