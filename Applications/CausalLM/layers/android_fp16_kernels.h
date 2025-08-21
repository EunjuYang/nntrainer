// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics
 *
 * @file   android_fp16_kernels.h
 * @date   2025
 * @brief  Android-specific FP16 optimized kernels using NEON intrinsics
 * @author Optimized for Android FP16
 */

#ifndef __ANDROID_FP16_KERNELS_H__
#define __ANDROID_FP16_KERNELS_H__

#ifdef __ANDROID__
#ifdef ENABLE_FP16

#include <arm_neon.h>
#include <cstdint>
#include <cmath>

#ifdef USE__FP16
#define _FP16 __fp16
#else
#define _FP16 _Float16
#endif

namespace nntrainer {

/**
 * @brief Optimized FP16 compute_kcaches for Android with NEON
 * @param[in] A float* input vector A
 * @param[in] B _FP16* input vector B (native FP16)
 * @param[out] output float* output vector
 * @param[in] num_rows number of rows
 * @param[in] N number of chunks
 * @param[in] chunk_size size of chunk
 * @param[in] group_size size of group
 * @param[in] tile_size size of tile
 */
template <>
inline void compute_kcaches(const float *A, const _FP16 *B, float *output,
                           int num_rows, int N, int chunk_size, int group_size,
                           int tile_size) {
  int row_stride = N * chunk_size;
  const int group_stride = group_size * chunk_size;
  const int tile_count = (num_rows + tile_size - 1) / tile_size;

  for (int n = 0; n < N; ++n) {
    for (int t = 0; t < tile_count; ++t) {
      int row_tile_start = t * tile_size;
      int tile_rows = std::min(tile_size, num_rows - row_tile_start);

      for (int g = 0; g < group_size; ++g) {
        const float *a_ptr = A + n * group_stride + g * chunk_size;
        
        for (int row = 0; row < tile_rows; ++row) {
          const _FP16 *b_row = B + (row_tile_start + row) * row_stride + n * chunk_size;
          
          float sum = 0.0f;
          int i = 0;
          
#ifdef USE_NEON
          // Use NEON for vectorized FP16 to FP32 conversion and computation
          float32x4_t acc = vdupq_n_f32(0.0f);
          
          for (; i + 8 <= chunk_size; i += 8) {
            // Load 8 FP16 values and convert to FP32
            float16x8_t b_fp16 = vld1q_f16(reinterpret_cast<const float16_t*>(b_row + i));
            float32x4_t b_low = vcvt_f32_f16(vget_low_f16(b_fp16));
            float32x4_t b_high = vcvt_f32_f16(vget_high_f16(b_fp16));
            
            // Load corresponding float values
            float32x4_t a_low = vld1q_f32(a_ptr + i);
            float32x4_t a_high = vld1q_f32(a_ptr + i + 4);
            
            // Multiply and accumulate
            acc = vmlaq_f32(acc, a_low, b_low);
            acc = vmlaq_f32(acc, a_high, b_high);
          }
          
          // Horizontal sum
          float32x2_t sum_low = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
          sum_low = vpadd_f32(sum_low, sum_low);
          sum += vget_lane_f32(sum_low, 0);
#endif
          
          // Handle remaining elements
          for (; i < chunk_size; ++i) {
            sum += a_ptr[i] * static_cast<float>(b_row[i]);
          }
          
          output[(row_tile_start + row) * N * group_size + n * group_size + g] =
            sum / std::sqrt(static_cast<float>(chunk_size));
        }
      }
    }
  }
}

/**
 * @brief Optimized FP16 vcache computation for Android
 * @param[in] row_num row number
 * @param[in] in float* input vector
 * @param[in] vcache _FP16* cache (native FP16)
 * @param[out] output float* output vector
 * @param[in] num_cache_head number of cache heads
 * @param[in] gqa_size group query attention size
 * @param[in] head_dim head dimension
 */
inline void compute_fp16vcache_fp32_transposed_native(
    int row_num, const float *in, const _FP16 *vcache, float *output,
    int num_cache_head, int gqa_size, int head_dim) {
  
  for (int n = 0; n < num_cache_head; ++n) {
    for (int h = 0; h < gqa_size; ++h) {
      for (int d = 0; d < head_dim; ++d) {
        float sum = 0.0f;
        
#ifdef USE_NEON
        // Vectorized computation using NEON
        int r = 0;
        float32x4_t acc = vdupq_n_f32(0.0f);
        
        for (; r + 4 <= row_num + 1; r += 4) {
          // Load attention scores
          float32x4_t attn = vld1q_f32(in + r * num_cache_head * gqa_size + 
                                       n * gqa_size + h);
          
          // Load and convert FP16 cache values to FP32
          float16x4_t cache_fp16;
          float cache_vals[4];
          for (int k = 0; k < 4; ++k) {
            int cache_idx = r + k;
            if (cache_idx <= row_num) {
              cache_vals[k] = static_cast<float>(
                vcache[cache_idx * num_cache_head * head_dim + n * head_dim + d]);
            } else {
              cache_vals[k] = 0.0f;
            }
          }
          float32x4_t cache_f32 = vld1q_f32(cache_vals);
          
          // Multiply and accumulate
          acc = vmlaq_f32(acc, attn, cache_f32);
        }
        
        // Sum the accumulator
        float32x2_t sum_low = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
        sum_low = vpadd_f32(sum_low, sum_low);
        sum += vget_lane_f32(sum_low, 0);
        
        // Handle remaining elements
        for (; r <= row_num; ++r) {
          float attn_val = in[r * num_cache_head * gqa_size + n * gqa_size + h];
          float cache_val = static_cast<float>(
            vcache[r * num_cache_head * head_dim + n * head_dim + d]);
          sum += attn_val * cache_val;
        }
#else
        // Scalar fallback
        for (int r = 0; r <= row_num; ++r) {
          float attn_val = in[r * num_cache_head * gqa_size + n * gqa_size + h];
          float cache_val = static_cast<float>(
            vcache[r * num_cache_head * head_dim + n * head_dim + d]);
          sum += attn_val * cache_val;
        }
#endif
        
        output[(n * gqa_size + h) * head_dim + d] = sum;
      }
    }
  }
}

/**
 * @brief Optimized rotary embedding with FP16 output for Android
 * @param[in] width width of the tensor
 * @param[in] dim dimension
 * @param[in] half_ half dimension
 * @param[in/out] inout float* input (also used as output for FP32)
 * @param[out] output _FP16* output for FP16 mode
 * @param[in] cos_ cosine values
 * @param[in] sin_ sine values
 * @param[in] only_convert_to_fp16 if true, only convert to FP16
 */
inline void compute_rotary_emb_value(unsigned int width, unsigned int dim,
                                     unsigned int half_, float *inout, 
                                     _FP16 *output, const float *cos_,
                                     const float *sin_, bool only_convert_to_fp16) {
  for (unsigned int w = 0; w < width; w += dim) {
#ifdef USE_NEON
    unsigned int k = 0;
    for (; k + 4 < half_; k += 4) {
      unsigned int i0 = w + k;
      unsigned int i1 = w + k + half_;
      
      // Load float values
      float32x4_t a = vld1q_f32(&inout[i0]);
      float32x4_t b = vld1q_f32(&inout[i1]);
      
      if (only_convert_to_fp16) {
        // Direct conversion to FP16
        float16x4_t a_fp16 = vcvt_f16_f32(a);
        float16x4_t b_fp16 = vcvt_f16_f32(b);
        
        vst1_f16(reinterpret_cast<float16_t*>(output + i0), a_fp16);
        vst1_f16(reinterpret_cast<float16_t*>(output + i1), b_fp16);
      } else {
        // Apply rotary embedding
        float32x4_t cos_v = vld1q_f32(&cos_[k]);
        float32x4_t sin_v = vld1q_f32(&sin_[k]);
        
        float32x4_t out0 = vsubq_f32(vmulq_f32(a, cos_v), vmulq_f32(b, sin_v));
        float32x4_t out1 = vaddq_f32(vmulq_f32(a, sin_v), vmulq_f32(b, cos_v));
        
        if (output != nullptr) {
          // Convert to FP16 and store
          float16x4_t out0_fp16 = vcvt_f16_f32(out0);
          float16x4_t out1_fp16 = vcvt_f16_f32(out1);
          
          vst1_f16(reinterpret_cast<float16_t*>(output + i0), out0_fp16);
          vst1_f16(reinterpret_cast<float16_t*>(output + i1), out1_fp16);
        } else {
          // Store as FP32
          vst1q_f32(&inout[i0], out0);
          vst1q_f32(&inout[i1], out1);
        }
      }
    }
    
    // Handle remaining elements
    for (; k < half_; ++k) {
      unsigned int i0 = w + k;
      unsigned int i1 = w + k + half_;
      float a = inout[i0];
      float b = inout[i1];
      
      if (only_convert_to_fp16) {
        output[i0] = static_cast<_FP16>(a);
        output[i1] = static_cast<_FP16>(b);
      } else {
        float c = cos_[k];
        float s = sin_[k];
        
        float out0 = a * c - b * s;
        float out1 = a * s + b * c;
        
        if (output != nullptr) {
          output[i0] = static_cast<_FP16>(out0);
          output[i1] = static_cast<_FP16>(out1);
        } else {
          inout[i0] = out0;
          inout[i1] = out1;
        }
      }
    }
#else
    // Scalar fallback
    for (unsigned int k = 0; k < half_; ++k) {
      unsigned int i0 = w + k;
      unsigned int i1 = w + k + half_;
      float a = inout[i0];
      float b = inout[i1];
      
      if (only_convert_to_fp16) {
        output[i0] = static_cast<_FP16>(a);
        output[i1] = static_cast<_FP16>(b);
      } else {
        float c = cos_[k];
        float s = sin_[k];
        
        float out0 = a * c - b * s;
        float out1 = a * s + b * c;
        
        if (output != nullptr) {
          output[i0] = static_cast<_FP16>(out0);
          output[i1] = static_cast<_FP16>(out1);
        } else {
          inout[i0] = out0;
          inout[i1] = out1;
        }
      }
    }
#endif
  }
}

} // namespace nntrainer

#endif // ENABLE_FP16
#endif // __ANDROID__

#endif // __ANDROID_FP16_KERNELS_H__