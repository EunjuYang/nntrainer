// Test file for FP16 softmax with FP32 sink
#include <iostream>
#include <vector>
#include <cmath>
#include <arm_neon.h>
#include <algorithm>

// Forward declarations
namespace nntrainer::neon {
void softmax_row_inplace(__fp16 *qk_out, size_t start_row, size_t end_row,
                         size_t num_heads, float *sink);
void softmax_row(__fp16 *qk_out, size_t start_row, size_t end_row,
                 size_t num_heads, float *sink);
}

// Helper function to print fp16 values
void print_fp16_array(const char* name, __fp16* arr, size_t rows, size_t cols) {
    std::cout << name << ":\n";
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            std::cout << static_cast<float>(arr[r * cols + c]) << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

// Helper function to print float values
void print_float_array(const char* name, float* arr, size_t size) {
    std::cout << name << ": ";
    for (size_t i = 0; i < size; ++i) {
        std::cout << arr[i] << " ";
    }
    std::cout << "\n\n";
}

// Reference softmax implementation for verification
void reference_softmax_with_sink(__fp16 *qk_out, size_t start_row, size_t end_row,
                                 size_t num_heads, float *sink) {
    for (size_t c = 0; c < num_heads; ++c) {
        // Find max including sink
        float max_val = sink[c];
        for (size_t r = start_row; r < end_row; ++r) {
            max_val = std::max(max_val, static_cast<float>(qk_out[r * num_heads + c]));
        }
        
        // Compute sum including sink
        float sum = std::exp(sink[c] - max_val);
        for (size_t r = start_row; r < end_row; ++r) {
            sum += std::exp(static_cast<float>(qk_out[r * num_heads + c]) - max_val);
        }
        
        // Apply softmax
        for (size_t r = start_row; r < end_row; ++r) {
            float val = static_cast<float>(qk_out[r * num_heads + c]);
            qk_out[r * num_heads + c] = static_cast<__fp16>(std::exp(val - max_val) / sum);
        }
    }
}

int main() {
    std::cout << "Testing FP16 Softmax with FP32 Sink\n";
    std::cout << "====================================\n\n";
    
    // Test parameters
    const size_t num_rows = 4;
    const size_t num_heads = 16;  // Test with 16 heads (multiple of 8)
    const size_t start_row = 0;
    const size_t end_row = num_rows;
    
    // Allocate memory
    std::vector<__fp16> qk_out_inplace(num_rows * num_heads);
    std::vector<__fp16> qk_out_non_inplace(num_rows * num_heads);
    std::vector<__fp16> qk_out_reference(num_rows * num_heads);
    std::vector<float> sink(num_heads);
    
    // Initialize with test data
    for (size_t i = 0; i < num_rows * num_heads; ++i) {
        float val = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
        qk_out_inplace[i] = static_cast<__fp16>(val);
        qk_out_non_inplace[i] = static_cast<__fp16>(val);
        qk_out_reference[i] = static_cast<__fp16>(val);
    }
    
    for (size_t i = 0; i < num_heads; ++i) {
        sink[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    }
    
    std::cout << "Input data:\n";
    print_fp16_array("qk_out (first 4 rows)", qk_out_inplace.data(), 
                     std::min(size_t(4), num_rows), num_heads);
    print_float_array("sink", sink.data(), num_heads);
    
    // Test inplace version
    std::cout << "Testing softmax_row_inplace with FP32 sink...\n";
    nntrainer::neon::softmax_row_inplace(qk_out_inplace.data(), start_row, end_row, 
                                         num_heads, sink.data());
    
    // Test non-inplace version
    std::cout << "Testing softmax_row with FP32 sink...\n";
    nntrainer::neon::softmax_row(qk_out_non_inplace.data(), start_row, end_row, 
                                 num_heads, sink.data());
    
    // Compute reference
    std::cout << "Computing reference...\n";
    reference_softmax_with_sink(qk_out_reference.data(), start_row, end_row, 
                                num_heads, sink.data());
    
    // Print results
    std::cout << "\nResults:\n";
    print_fp16_array("qk_out_inplace (first 4 rows)", qk_out_inplace.data(), 
                     std::min(size_t(4), num_rows), num_heads);
    print_fp16_array("qk_out_non_inplace (first 4 rows)", qk_out_non_inplace.data(), 
                     std::min(size_t(4), num_rows), num_heads);
    print_fp16_array("qk_out_reference (first 4 rows)", qk_out_reference.data(), 
                     std::min(size_t(4), num_rows), num_heads);
    
    // Verify results
    std::cout << "Verification:\n";
    float max_error_inplace = 0.0f;
    float max_error_non_inplace = 0.0f;
    
    for (size_t i = 0; i < num_rows * num_heads; ++i) {
        float ref = static_cast<float>(qk_out_reference[i]);
        float inplace = static_cast<float>(qk_out_inplace[i]);
        float non_inplace = static_cast<float>(qk_out_non_inplace[i]);
        
        max_error_inplace = std::max(max_error_inplace, std::abs(inplace - ref));
        max_error_non_inplace = std::max(max_error_non_inplace, std::abs(non_inplace - ref));
    }
    
    std::cout << "Max error (inplace vs reference): " << max_error_inplace << "\n";
    std::cout << "Max error (non-inplace vs reference): " << max_error_non_inplace << "\n";
    
    // Check if errors are within acceptable tolerance (considering FP16 precision)
    const float tolerance = 1e-3f;  // FP16 has limited precision
    
    if (max_error_inplace < tolerance && max_error_non_inplace < tolerance) {
        std::cout << "\n✓ Test PASSED: All implementations match within tolerance\n";
        return 0;
    } else {
        std::cout << "\n✗ Test FAILED: Implementations differ beyond tolerance\n";
        return 1;
    }
}