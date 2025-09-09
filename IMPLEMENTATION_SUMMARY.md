# FP16 Softmax with FP32 Sink Implementation Summary

## Overview
This implementation adds support for mixed-precision softmax operations in the NEON CPU backend, specifically handling `__fp16` input/output with `float` sink values.

## Files Modified

### 1. `/workspace/nntrainer/tensor/cpu_backend/arm/neon_impl_fp16.cpp`

#### New Functions Added:

1. **`softmax_row_inplace(__fp16 *qk_out, size_t start_row, size_t end_row, size_t num_heads, float *sink)`**
   - Performs in-place softmax on FP16 data with FP32 sink
   - Converts FP16 to FP32 for computation to maintain precision
   - Includes sink values in max and sum calculations
   - Optimized with NEON vectorization

2. **`softmax_row(__fp16 *qk_out, size_t start_row, size_t end_row, size_t num_heads, float *sink)`**
   - Non-inplace version of softmax with FP32 sink
   - Similar optimization and precision handling as inplace version

#### Fixed Functions:

1. **`softmax_row_inplace(__fp16 *qk_out, size_t start_row, size_t end_row, size_t num_heads, __fp16 *sink)`**
   - Fixed to properly include sink values in max calculation
   - Fixed to include exp(sink - max) in sum initialization
   - Now correctly handles attention sink tokens

2. **`softmax_row(__fp16 *qk_out, size_t start_row, size_t end_row, size_t num_heads, __fp16 *sink)`**
   - Similar fixes as the inplace version

### 2. `/workspace/nntrainer/tensor/cpu_backend/arm/neon_impl.h`

Added new function declarations:
```cpp
#ifdef ENABLE_FP16
void softmax_row_inplace(__fp16 *qk_out, size_t start_row, size_t end_row,
                         size_t num_heads, float *sink);
void softmax_row(__fp16 *qk_out, size_t start_row, size_t end_row,
                 size_t num_heads, float *sink);
#endif
```

## Key Implementation Details

### Mixed Precision Handling
- Input data (`qk_out`) is in FP16 format for memory efficiency
- Sink values are in FP32 for better precision
- Internal computations use FP32 to avoid numerical issues
- Results are converted back to FP16 for storage

### Algorithm Flow
1. **Max Calculation**: Find maximum value across each column, including sink values
2. **Exp and Sum**: Compute exp(x - max) for numerical stability, including sink in sum
3. **Normalization**: Divide each exp value by the sum to get softmax

### NEON Optimization
- Processes 8 FP16 values at once using `float16x8_t`
- Converts to FP32 using `vcvt_f32_f16` for computation
- Uses vectorized exp function `exp_ps` for FP32 values
- Converts back to FP16 using `vcvt_f16_f32`

### Correctness of Sink Handling
The implementation correctly handles attention sink tokens by:
1. Including sink values when finding the maximum (for numerical stability)
2. Adding exp(sink - max) to the sum (sink contributes to normalization)
3. This ensures the softmax properly accounts for the sink token's attention weight

## Testing
A test file was created (`test_softmax_fp16.cpp`) that:
- Compares the NEON implementation against a reference implementation
- Tests both inplace and non-inplace versions
- Verifies correctness with random input data

## Performance Considerations
- Vectorized operations process 8 FP16 values simultaneously
- Memory access patterns are optimized for cache efficiency
- Mixed precision approach balances memory usage and numerical accuracy

## Usage Example
```cpp
// FP16 attention scores
__fp16 *qk_out = ...;  // shape: [seq_len, num_heads]

// FP32 sink values
float *sink = ...;     // shape: [num_heads]

// Apply softmax with sink
nntrainer::neon::softmax_row_inplace(qk_out, 0, seq_len, num_heads, sink);
```

## Notes
- The implementation requires ARM NEON with FP16 support
- Compile with `-march=armv8-a+fp16` flag on ARM systems
- The `ENABLE_FP16` macro must be defined