// Usage example for FP16 softmax functions
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
    
    // Case 1: FP16 input with FP16 sink
    softmax_row_inplace(qk_out, 0, num_rows, num_heads, sink_fp16);
    
    // Case 2: FP16 input without sink (nullptr)
    softmax_row_inplace(qk_out, 0, num_rows, num_heads, (__fp16*)nullptr);
    
    // Case 3: FP16 input with FP32 sink - use the specific function name
    softmax_row_inplace_fp16_fp32(qk_out, 0, num_rows, num_heads, sink_fp32);
    
    // Similarly for non-inplace versions:
    
    // Case 1: FP16 input with FP16 sink
    softmax_row(qk_out, 0, num_rows, num_heads, sink_fp16);
    
    // Case 2: FP16 input without sink (nullptr)
    softmax_row(qk_out, 0, num_rows, num_heads, (__fp16*)nullptr);
    
    // Case 3: FP16 input with FP32 sink - use the specific function name
    softmax_row_fp16_fp32(qk_out, 0, num_rows, num_heads, sink_fp32);
    
    // Clean up
    delete[] qk_out;
    delete[] sink_fp16;
    delete[] sink_fp32;
}

// Alternative: Create wrapper functions if you want automatic type deduction
namespace wrapper {
    // Wrapper for mixed precision case
    inline void softmax_row_inplace(__fp16 *qk_out, size_t start_row, 
                                    size_t end_row, size_t num_heads, 
                                    float *sink) {
        if (sink == nullptr) {
            // Call the FP16 version without sink
            ::nntrainer::neon::softmax_row_inplace(qk_out, start_row, end_row, 
                                                   num_heads, (__fp16*)nullptr);
        } else {
            // Call the mixed precision version
            ::nntrainer::neon::softmax_row_inplace_fp16_fp32(qk_out, start_row, 
                                                             end_row, num_heads, sink);
        }
    }
    
    inline void softmax_row(__fp16 *qk_out, size_t start_row, 
                           size_t end_row, size_t num_heads, 
                           float *sink) {
        if (sink == nullptr) {
            // Call the FP16 version without sink
            ::nntrainer::neon::softmax_row(qk_out, start_row, end_row, 
                                          num_heads, (__fp16*)nullptr);
        } else {
            // Call the mixed precision version
            ::nntrainer::neon::softmax_row_fp16_fp32(qk_out, start_row, 
                                                     end_row, num_heads, sink);
        }
    }
}