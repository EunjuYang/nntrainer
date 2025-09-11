/**
 * Copyright (C) 2025 Optimization for ARM/Android
 * 
 * @file qwen_moe_layer_cached_opt.h
 * @brief Optimized helper functions for MoE layer on ARM
 */

#ifndef __QWEN_MOE_LAYER_CACHED_OPT_H__
#define __QWEN_MOE_LAYER_CACHED_OPT_H__

#include <cstring>
#include <memory>
#include <vector>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace causallm {
namespace opt {

// Memory pool for expert outputs to reduce allocation overhead
class TensorMemoryPool {
public:
    static TensorMemoryPool& getInstance() {
        static thread_local TensorMemoryPool instance;
        return instance;
    }
    
    float* getBuffer(size_t size) {
        if (buffer_size_ < size) {
            buffer_.reset(new float[size]);
            buffer_size_ = size;
        }
        return buffer_.get();
    }
    
private:
    std::unique_ptr<float[]> buffer_;
    size_t buffer_size_ = 0;
};

// Optimized memory copy functions
inline void optimized_memcpy(float* dst, const float* src, size_t size) {
#ifdef __ARM_NEON
    size_t i = 0;
    // Process 64 bytes at a time (16 floats)
    for (; i + 16 <= size; i += 16) {
        float32x4x4_t data = vld1q_f32_x4(src + i);
        vst1q_f32_x4(dst + i, data);
    }
    // Process 16 bytes at a time (4 floats)
    for (; i + 4 <= size; i += 4) {
        float32x4_t data = vld1q_f32(src + i);
        vst1q_f32(dst + i, data);
    }
    // Handle remaining elements
    for (; i < size; ++i) {
        dst[i] = src[i];
    }
#else
    std::memcpy(dst, src, size * sizeof(float));
#endif
}

// Optimized weighted accumulation: dst += src * weight
inline void weighted_add(float* dst, const float* src, float weight, size_t size) {
#ifdef __ARM_NEON
    float32x4_t weight_vec = vdupq_n_f32(weight);
    size_t i = 0;
    
    // Unroll by 16 for better performance
    for (; i + 16 <= size; i += 16) {
        float32x4x4_t src_data = vld1q_f32_x4(src + i);
        float32x4x4_t dst_data = vld1q_f32_x4(dst + i);
        
        dst_data.val[0] = vmlaq_f32(dst_data.val[0], src_data.val[0], weight_vec);
        dst_data.val[1] = vmlaq_f32(dst_data.val[1], src_data.val[1], weight_vec);
        dst_data.val[2] = vmlaq_f32(dst_data.val[2], src_data.val[2], weight_vec);
        dst_data.val[3] = vmlaq_f32(dst_data.val[3], src_data.val[3], weight_vec);
        
        vst1q_f32_x4(dst + i, dst_data);
    }
    
    // Process remaining 4-element blocks
    for (; i + 4 <= size; i += 4) {
        float32x4_t src_data = vld1q_f32(src + i);
        float32x4_t dst_data = vld1q_f32(dst + i);
        dst_data = vmlaq_f32(dst_data, src_data, weight_vec);
        vst1q_f32(dst + i, dst_data);
    }
    
    // Handle remaining elements
    for (; i < size; ++i) {
        dst[i] += src[i] * weight;
    }
#else
    for (size_t i = 0; i < size; ++i) {
        dst[i] += src[i] * weight;
    }
#endif
}

// Optimized vector addition: dst += src
inline void vector_add(float* dst, const float* src, size_t size) {
#ifdef __ARM_NEON
    size_t i = 0;
    
    // Process 16 floats at a time
    for (; i + 16 <= size; i += 16) {
        float32x4x4_t src_data = vld1q_f32_x4(src + i);
        float32x4x4_t dst_data = vld1q_f32_x4(dst + i);
        
        dst_data.val[0] = vaddq_f32(dst_data.val[0], src_data.val[0]);
        dst_data.val[1] = vaddq_f32(dst_data.val[1], src_data.val[1]);
        dst_data.val[2] = vaddq_f32(dst_data.val[2], src_data.val[2]);
        dst_data.val[3] = vaddq_f32(dst_data.val[3], src_data.val[3]);
        
        vst1q_f32_x4(dst + i, dst_data);
    }
    
    // Process 4 floats at a time
    for (; i + 4 <= size; i += 4) {
        float32x4_t src_data = vld1q_f32(src + i);
        float32x4_t dst_data = vld1q_f32(dst + i);
        vst1q_f32(dst + i, vaddq_f32(dst_data, src_data));
    }
    
    // Handle remaining elements
    for (; i < size; ++i) {
        dst[i] += src[i];
    }
#else
    for (size_t i = 0; i < size; ++i) {
        dst[i] += src[i];
    }
#endif
}

// Cache prefetch hints for ARM
inline void prefetch_data(const void* addr) {
#ifdef __ARM_NEON
    __builtin_prefetch(addr, 0, 3);  // Read, high temporal locality
#endif
}

// Batch processing optimization parameters
struct OptimizationParams {
    static constexpr size_t CACHE_LINE_SIZE = 64;
    static constexpr size_t MIN_PARALLEL_TOKENS = 4;
    static constexpr size_t MIN_PARALLEL_EXPERTS = 2;
    static constexpr size_t PREFETCH_DISTANCE = 256;
};

} // namespace opt
} // namespace causallm

#endif // __QWEN_MOE_LAYER_CACHED_OPT_H__