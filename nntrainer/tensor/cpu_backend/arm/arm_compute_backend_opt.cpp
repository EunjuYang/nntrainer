// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Optimization for MoE Layer
 *
 * @file arm_compute_backend_opt.cpp
 * @brief Optimized ARM backend functions implementation
 */

#include <arm_compute_backend_opt.h>
#include <arm_neon.h>
#include <cstring>
#include <omp.h>
#include <algorithm>

namespace nntrainer {

void batch_copy_indexed(size_t batch_size, size_t elem_size,
                        const unsigned int* src_offsets,
                        const float* src, float* dst) {
    #pragma omp parallel for schedule(static) if(batch_size > 8)
    for (size_t i = 0; i < batch_size; ++i) {
        const float* src_ptr = src + src_offsets[i] * elem_size;
        float* dst_ptr = dst + i * elem_size;
        
        size_t j = 0;
        // Process 16 floats at a time using NEON
        for (; j + 16 <= elem_size; j += 16) {
            float32x4x4_t data = vld1q_f32_x4(src_ptr + j);
            vst1q_f32_x4(dst_ptr + j, data);
        }
        // Process 4 floats at a time
        for (; j + 4 <= elem_size; j += 4) {
            float32x4_t data = vld1q_f32(src_ptr + j);
            vst1q_f32(dst_ptr + j, data);
        }
        // Handle remaining elements
        for (; j < elem_size; ++j) {
            dst_ptr[j] = src_ptr[j];
        }
    }
}

void weighted_add_i(const unsigned int N, const float* src, float* dst, 
                   float weight) {
    float32x4_t weight_vec = vdupq_n_f32(weight);
    unsigned int i = 0;
    
    // Unroll by 16 for better performance
    for (; i + 16 <= N; i += 16) {
        float32x4x4_t src_data = vld1q_f32_x4(src + i);
        float32x4x4_t dst_data = vld1q_f32_x4(dst + i);
        
        dst_data.val[0] = vmlaq_f32(dst_data.val[0], src_data.val[0], weight_vec);
        dst_data.val[1] = vmlaq_f32(dst_data.val[1], src_data.val[1], weight_vec);
        dst_data.val[2] = vmlaq_f32(dst_data.val[2], src_data.val[2], weight_vec);
        dst_data.val[3] = vmlaq_f32(dst_data.val[3], src_data.val[3], weight_vec);
        
        vst1q_f32_x4(dst + i, dst_data);
    }
    
    // Process remaining 4-element blocks
    for (; i + 4 <= N; i += 4) {
        float32x4_t src_data = vld1q_f32(src + i);
        float32x4_t dst_data = vld1q_f32(dst + i);
        dst_data = vmlaq_f32(dst_data, src_data, weight_vec);
        vst1q_f32(dst + i, dst_data);
    }
    
    // Handle remaining elements
    for (; i < N; ++i) {
        dst[i] += src[i] * weight;
    }
}

void batch_weighted_add(size_t batch_size, size_t elem_size,
                        const float* src, const unsigned int* dst_offsets,
                        float* dst, const float* weights) {
    #pragma omp parallel for schedule(guided) if(batch_size > 4)
    for (size_t b = 0; b < batch_size; ++b) {
        const float* src_ptr = src + b * elem_size;
        float* dst_ptr = dst + dst_offsets[b] * elem_size;
        float weight = weights[b];
        
        float32x4_t weight_vec = vdupq_n_f32(weight);
        size_t i = 0;
        
        // Process 16 floats at a time
        for (; i + 16 <= elem_size; i += 16) {
            float32x4x4_t src_data = vld1q_f32_x4(src_ptr + i);
            float32x4x4_t dst_data = vld1q_f32_x4(dst_ptr + i);
            
            dst_data.val[0] = vmlaq_f32(dst_data.val[0], src_data.val[0], weight_vec);
            dst_data.val[1] = vmlaq_f32(dst_data.val[1], src_data.val[1], weight_vec);
            dst_data.val[2] = vmlaq_f32(dst_data.val[2], src_data.val[2], weight_vec);
            dst_data.val[3] = vmlaq_f32(dst_data.val[3], src_data.val[3], weight_vec);
            
            vst1q_f32_x4(dst_ptr + i, dst_data);
        }
        
        // Process 4 floats at a time
        for (; i + 4 <= elem_size; i += 4) {
            float32x4_t src_data = vld1q_f32(src_ptr + i);
            float32x4_t dst_data = vld1q_f32(dst_ptr + i);
            dst_data = vmlaq_f32(dst_data, src_data, weight_vec);
            vst1q_f32(dst_ptr + i, dst_data);
        }
        
        // Handle remaining elements
        for (; i < elem_size; ++i) {
            dst_ptr[i] += src_ptr[i] * weight;
        }
    }
}

void add_contiguous(const unsigned int N, const float* X, const float* Y, 
                   float* Z) {
    size_t i = 0;
    
    // Process 16 floats at a time
    for (; i + 16 <= N; i += 16) {
        float32x4x4_t x_data = vld1q_f32_x4(X + i);
        float32x4x4_t y_data = vld1q_f32_x4(Y + i);
        
        x_data.val[0] = vaddq_f32(x_data.val[0], y_data.val[0]);
        x_data.val[1] = vaddq_f32(x_data.val[1], y_data.val[1]);
        x_data.val[2] = vaddq_f32(x_data.val[2], y_data.val[2]);
        x_data.val[3] = vaddq_f32(x_data.val[3], y_data.val[3]);
        
        vst1q_f32_x4(Z + i, x_data);
    }
    
    // Process 4 floats at a time
    for (; i + 4 <= N; i += 4) {
        float32x4_t x_data = vld1q_f32(X + i);
        float32x4_t y_data = vld1q_f32(Y + i);
        vst1q_f32(Z + i, vaddq_f32(x_data, y_data));
    }
    
    // Handle remaining elements
    for (; i < N; ++i) {
        Z[i] = X[i] + Y[i];
    }
}

void batch_gemm(unsigned int num_matrices, unsigned int M, unsigned int N,
               unsigned int K, const std::vector<const float*>& A_list,
               const std::vector<const float*>& B_list,
               std::vector<float*>& C_list, float alpha, float beta) {
    // Use parallel processing for multiple matrices
    #pragma omp parallel for schedule(dynamic) if(num_matrices > 2)
    for (unsigned int idx = 0; idx < num_matrices; ++idx) {
        const float* A = A_list[idx];
        const float* B = B_list[idx];
        float* C = C_list[idx];
        
        // For small matrices, use optimized NEON implementation
        // For larger matrices, fall back to BLAS or optimized kernels
        if (M * N * K < 1024) {
            // Small matrix - direct NEON implementation
            for (unsigned int m = 0; m < M; ++m) {
                for (unsigned int n = 0; n < N; ++n) {
                    float32x4_t sum = vdupq_n_f32(0.0f);
                    unsigned int k = 0;
                    
                    // Vectorized inner loop
                    for (; k + 4 <= K; k += 4) {
                        float32x4_t a_vec = vld1q_f32(&A[m * K + k]);
                        float32x4_t b_vec = {B[k * N + n], B[(k+1) * N + n], 
                                            B[(k+2) * N + n], B[(k+3) * N + n]};
                        sum = vmlaq_f32(sum, a_vec, b_vec);
                    }
                    
                    // Reduce and handle remainder
                    float result = vaddvq_f32(sum);
                    for (; k < K; ++k) {
                        result += A[m * K + k] * B[k * N + n];
                    }
                    
                    C[m * N + n] = alpha * result + beta * C[m * N + n];
                }
            }
        } else {
            // For larger matrices, use existing optimized GEMM
            // This would call the existing sgemm implementation
            extern void sgemm(const unsigned int, bool, bool, 
                            const unsigned int, const unsigned int, const unsigned int,
                            const float, const float*, const unsigned int,
                            const float*, const unsigned int, 
                            const float, float*, const unsigned int);
            
            sgemm(0, false, false, M, N, K, alpha, A, K, B, N, beta, C, N);
        }
    }
}

void batch_swiglu(unsigned int batch_size, unsigned int width,
                 float* gate_out, float* up_out, float* output) {
    #pragma omp parallel for schedule(static) if(batch_size > 2)
    for (unsigned int b = 0; b < batch_size; ++b) {
        float* gate_ptr = gate_out + b * width;
        float* up_ptr = up_out + b * width;
        float* out_ptr = output + b * width;
        
        unsigned int i = 0;
        // Process 4 elements at a time with NEON
        for (; i + 4 <= width; i += 4) {
            float32x4_t gate = vld1q_f32(gate_ptr + i);
            float32x4_t up = vld1q_f32(up_ptr + i);
            
            // SiLU activation: gate / (1 + exp(-gate))
            float32x4_t neg_gate = vnegq_f32(gate);
            float32x4_t exp_neg = vexpq_f32(neg_gate);  // Requires NEON exp implementation
            float32x4_t one = vdupq_n_f32(1.0f);
            float32x4_t denom = vaddq_f32(one, exp_neg);
            float32x4_t silu = vdivq_f32(gate, denom);
            
            // Multiply with up projection
            float32x4_t result = vmulq_f32(silu, up);
            vst1q_f32(out_ptr + i, result);
        }
        
        // Handle remaining elements
        for (; i < width; ++i) {
            float g = gate_ptr[i];
            out_ptr[i] = (g / (1.0f + expf(-g))) * up_ptr[i];
        }
    }
}

void prefetch_weights(const void* addr, size_t size) {
    const char* ptr = static_cast<const char*>(addr);
    const size_t cache_line_size = 64;  // ARM typical cache line size
    
    for (size_t offset = 0; offset < size; offset += cache_line_size) {
        __builtin_prefetch(ptr + offset, 0, 3);  // Read, high temporal locality
    }
}

// Helper function for NEON exp (simplified version)
static inline float32x4_t vexpq_f32(float32x4_t x) {
    // This is a simplified approximation
    // For production, use a proper NEON exp implementation
    float32x4_t one = vdupq_n_f32(1.0f);
    float32x4_t result = one;
    float32x4_t term = x;
    
    // Taylor series approximation (simplified)
    for (int i = 1; i < 8; ++i) {
        result = vaddq_f32(result, term);
        term = vmulq_f32(term, vdivq_f32(x, vdupq_n_f32((float)(i + 1))));
    }
    
    return result;
}

} // namespace nntrainer