/**
 * Copyright (C) 2020 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @file	gpt_oss_moe_layer_cached_optimized.cpp
 * @date	05 Sep 2025
 * @brief	Optimized Mixture of Expert Layer Class for Gpt-Oss model
 * @author	Optimized version
 */

#include <acti_func.h>
#include <algorithm>
#include <cmath>
#include <gpt_oss_moe_layer_cached.h>
#include <node_exporter.h>
#include <omp.h>
#include <stdexcept>
#include <immintrin.h>  // For SIMD optimizations

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

// Optimized swiglu function using SIMD
inline void swiglu_optimized(const unsigned int N, float *X, const float *Y, 
                             const float *Z, float alpha) {
    const int simd_width = 8;
    const __m256 alpha_vec = _mm256_set1_ps(alpha);
    const __m256 one_vec = _mm256_set1_ps(1.0f);
    
    unsigned int i = 0;
    
    // SIMD optimized main loop
    for (; i + simd_width <= N; i += simd_width) {
        __m256 y_vec = _mm256_loadu_ps(Y + i);
        __m256 z_vec = _mm256_loadu_ps(Z + i);
        
        // Compute gate * alpha
        __m256 gate_alpha = _mm256_mul_ps(y_vec, alpha_vec);
        
        // Approximate sigmoid using fast exp approximation
        // sigmoid(x) ≈ 1 / (1 + exp(-x))
        __m256 neg_gate = _mm256_sub_ps(_mm256_setzero_ps(), gate_alpha);
        
        // Fast exp approximation for better performance
        neg_gate = _mm256_max_ps(neg_gate, _mm256_set1_ps(-88.0f));
        neg_gate = _mm256_min_ps(neg_gate, _mm256_set1_ps(88.0f));
        
        __m256 exp_val = _mm256_exp_ps(neg_gate);
        __m256 sigmoid = _mm256_div_ps(one_vec, _mm256_add_ps(one_vec, exp_val));
        
        // Compute swiglu: gate * sigmoid(gate * alpha) * (up + 1)
        __m256 z_plus_one = _mm256_add_ps(z_vec, one_vec);
        __m256 result = _mm256_mul_ps(y_vec, sigmoid);
        result = _mm256_mul_ps(result, z_plus_one);
        
        _mm256_storeu_ps(X + i, result);
    }
    
    // Handle remaining elements
    for (; i < N; ++i) {
        float sigmoid = 1.0f / (1.0f + std::exp(-alpha * Y[i]));
        X[i] = Y[i] * sigmoid * (Z[i] + 1.0f);
    }
}

// Optimized clamp function using SIMD
inline void clamp_optimized(float *data, size_t length, float lower, float upper) {
    const int simd_width = 8;
    const __m256 lower_vec = _mm256_set1_ps(lower);
    const __m256 upper_vec = _mm256_set1_ps(upper);
    
    size_t i = 0;
    
    // SIMD optimized main loop
    for (; i + simd_width <= length; i += simd_width) {
        __m256 val = _mm256_loadu_ps(data + i);
        val = _mm256_max_ps(val, lower_vec);
        val = _mm256_min_ps(val, upper_vec);
        _mm256_storeu_ps(data + i, val);
    }
    
    // Handle remaining elements
    for (; i < length; ++i) {
        data[i] = std::max(lower, std::min(upper, data[i]));
    }
}

// Batch processing version of expert forward
inline void compute_expert_forward_batch(
    const nntrainer::Tensor &input, nntrainer::Tensor &expert_output,
    const std::vector<std::pair<unsigned, float>> &token_assignments,
    const nntrainer::Tensor &gate_proj, const nntrainer::Tensor &up_proj,
    const nntrainer::Tensor &down_proj, const nntrainer::Tensor &gate_bias,
    const nntrainer::Tensor &up_bias, const nntrainer::Tensor &down_bias,
    unsigned int hidden_size) {
    
    const unsigned intermediate_size = gate_proj.width();
    const unsigned num_tokens = token_assignments.size();
    
    if (num_tokens == 0) return;
    
    // Pre-allocate all intermediate tensors at once to improve cache locality
    nntrainer::TensorDim batch_input_dim({num_tokens, 1, 1, hidden_size}, 
                                         input.getTensorType());
    nntrainer::TensorDim batch_intermediate_dim({num_tokens, 1, 1, intermediate_size},
                                                input.getTensorType());
    nntrainer::TensorDim batch_output_dim({num_tokens, 1, 1, hidden_size},
                                          input.getTensorType());
    
    // Batch tensors for better cache utilization
    nntrainer::Tensor batch_input(batch_input_dim);
    nntrainer::Tensor batch_gate_out(batch_intermediate_dim);
    nntrainer::Tensor batch_up_out(batch_intermediate_dim);
    nntrainer::Tensor batch_down_out(batch_output_dim);
    
    // Step 1: Gather input tokens into batch (improves cache locality)
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < num_tokens; ++i) {
        const unsigned token_idx = token_assignments[i].first;
        size_t src_offset = token_idx * hidden_size;
        size_t dst_offset = i * hidden_size;
        
        // Use SIMD-optimized memcpy for better performance
        std::memcpy(batch_input.getData<float>() + dst_offset,
                   input.getData<float>() + src_offset,
                   hidden_size * sizeof(float));
    }
    
    // Step 2: Batch matrix multiplication for gate projection
    batch_input.dot(gate_proj, batch_gate_out);
    
    // Step 3: Batch matrix multiplication for up projection
    batch_input.dot(up_proj, batch_up_out);
    
    // Step 4: Apply biases and activations in a single pass
    float *gate_data = batch_gate_out.getData<float>();
    float *up_data = batch_up_out.getData<float>();
    const float *gate_bias_data = gate_bias.getData<float>();
    const float *up_bias_data = up_bias.getData<float>();
    
    const size_t total_intermediate = num_tokens * intermediate_size;
    
    // Fused bias addition and clamping using SIMD
    #pragma omp parallel
    {
        #pragma omp for schedule(static) nowait
        for (size_t token = 0; token < num_tokens; ++token) {
            size_t offset = token * intermediate_size;
            
            // Add bias and clamp for gate
            for (size_t j = 0; j < intermediate_size; ++j) {
                gate_data[offset + j] += gate_bias_data[j];
            }
            clamp_optimized(gate_data + offset, intermediate_size, 
                          std::numeric_limits<float>::lowest(), limit);
            
            // Add bias and clamp for up
            for (size_t j = 0; j < intermediate_size; ++j) {
                up_data[offset + j] += up_bias_data[j];
            }
            clamp_optimized(up_data + offset, intermediate_size, -limit, limit);
        }
    }
    
    // Step 5: Apply swiglu activation (optimized version)
    float *acti_out = batch_gate_out.getData<float>(); // Reuse gate_out for activation
    
    #pragma omp parallel for schedule(static)
    for (size_t token = 0; token < num_tokens; ++token) {
        size_t offset = token * intermediate_size;
        swiglu_optimized(intermediate_size, 
                        acti_out + offset,
                        gate_data + offset,
                        up_data + offset,
                        alpha);
    }
    
    // Step 6: Down projection
    batch_gate_out.dot(down_proj, batch_down_out);  // Reuse gate_out as input
    
    // Step 7: Add down bias and apply weights, then scatter to output
    float *down_data = batch_down_out.getData<float>();
    const float *down_bias_data = down_bias.getData<float>();
    
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < num_tokens; ++i) {
        const unsigned token_idx = token_assignments[i].first;
        const float weight = token_assignments[i].second;
        
        size_t batch_offset = i * hidden_size;
        size_t output_offset = token_idx * hidden_size;
        
        float *token_output = expert_output.getData<float>() + output_offset;
        const float *token_down = down_data + batch_offset;
        
        // Fused bias addition, weight multiplication, and accumulation
        for (size_t j = 0; j < hidden_size; ++j) {
            token_output[j] += (token_down[j] + down_bias_data[j]) * weight;
        }
    }
}

// Wrapper to maintain compatibility with original interface
inline void CachedSlimGptOssMoELayer::compute_expert_forward(
    const nntrainer::Tensor &input, nntrainer::Tensor &expert_output,
    const std::vector<std::pair<unsigned, float>> &token_assignments,
    const nntrainer::Tensor &gate_proj, const nntrainer::Tensor &up_proj,
    const nntrainer::Tensor &down_proj, const nntrainer::Tensor &gate_bias,
    const nntrainer::Tensor &up_bias, const nntrainer::Tensor &down_bias,
    unsigned int hidden_size) {
    
    // Use batch processing for better performance
    compute_expert_forward_batch(input, expert_output, token_assignments,
                                 gate_proj, up_proj, down_proj,
                                 gate_bias, up_bias, down_bias, hidden_size);
}

} // namespace causallm