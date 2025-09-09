// Simple usage example for FP16 softmax functions with overloading
#include <arm_neon.h>
#include "neon_impl.h"

using namespace nntrainer::neon;

void example_usage() {
    const size_t num_rows = 10;
    const size_t num_heads = 16;
    
    // Allocate data
    __fp16 *qk_out = new __fp16[num_rows * num_heads];
    __fp16 *sink_fp16 = new __fp16[num_heads];
    float *sink_fp32 = new float[num_heads];
    
    // Initialize data...
    
    // ========================================
    // 오버로딩이 자동으로 올바른 함수를 선택
    // ========================================
    
    // Case 1: FP16 input with FP16 sink - 템플릿 특수화 버전 호출
    softmax_row_inplace(qk_out, 0, num_rows, num_heads, sink_fp16);
    
    // Case 2: FP16 input with FP32 sink - 오버로드된 버전 호출
    softmax_row_inplace(qk_out, 0, num_rows, num_heads, sink_fp32);
    
    // Case 3: FP16 input without sink - nullptr 전달
    // FP16 버전 사용
    softmax_row_inplace(qk_out, 0, num_rows, num_heads, (__fp16*)nullptr);
    
    // FP32 버전 사용 (내부에서 nullptr 체크)
    softmax_row_inplace(qk_out, 0, num_rows, num_heads, (float*)nullptr);
    
    // ========================================
    // Non-inplace 버전도 동일하게 동작
    // ========================================
    
    // FP16 sink
    softmax_row(qk_out, 0, num_rows, num_heads, sink_fp16);
    
    // FP32 sink
    softmax_row(qk_out, 0, num_rows, num_heads, sink_fp32);
    
    // nullptr
    softmax_row(qk_out, 0, num_rows, num_heads, (__fp16*)nullptr);
    softmax_row(qk_out, 0, num_rows, num_heads, (float*)nullptr);
    
    // Clean up
    delete[] qk_out;
    delete[] sink_fp16;
    delete[] sink_fp32;
}

// 실제 사용 시나리오
void attention_computation(__fp16 *query, __fp16 *key, __fp16 *value,
                          float *attention_sink,  // FP32 sink for better precision
                          size_t seq_len, size_t num_heads, size_t head_dim) {
    
    // QK^T computation...
    __fp16 *qk_scores = new __fp16[seq_len * seq_len * num_heads];
    
    // Apply softmax with FP32 sink
    // 컴파일러가 자동으로 float* 파라미터를 보고 올바른 오버로드 선택
    for (size_t batch = 0; batch < seq_len; ++batch) {
        softmax_row_inplace(&qk_scores[batch * seq_len * num_heads], 
                           0, seq_len, num_heads, attention_sink);
    }
    
    // Continue with attention computation...
    delete[] qk_scores;
}