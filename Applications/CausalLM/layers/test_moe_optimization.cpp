/**
 * @file test_moe_optimization.cpp
 * @brief Test and benchmark for MoE layer optimization
 */

#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <iomanip>

// Simple benchmark framework
class Benchmark {
public:
    static void run_benchmark(const std::string& name, std::function<void()> func, int iterations = 100) {
        std::cout << "Running benchmark: " << name << std::endl;
        
        // Warmup
        for (int i = 0; i < 10; ++i) {
            func();
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            func();
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double avg_time = duration.count() / static_cast<double>(iterations);
        
        std::cout << "  Average time: " << std::fixed << std::setprecision(2) 
                  << avg_time << " μs" << std::endl;
        std::cout << "  Total time: " << duration.count() / 1000.0 << " ms" << std::endl;
        std::cout << "  Throughput: " << (iterations * 1000000.0) / duration.count() 
                  << " ops/sec" << std::endl;
    }
};

// Test functions for optimization verification
void test_memory_copy_optimization() {
    const size_t sizes[] = {128, 256, 512, 1024, 2048, 4096};
    
    for (size_t size : sizes) {
        std::vector<float> src(size);
        std::vector<float> dst(size);
        
        // Initialize with random data
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        for (size_t i = 0; i < size; ++i) {
            src[i] = dis(gen);
        }
        
        std::string test_name = "Memory Copy (size=" + std::to_string(size) + ")";
        
        Benchmark::run_benchmark(test_name, [&]() {
#ifdef __ARM_NEON
            // NEON optimized copy
            float* dst_ptr = dst.data();
            const float* src_ptr = src.data();
            size_t i = 0;
            for (; i + 16 <= size; i += 16) {
                float32x4x4_t data = vld1q_f32_x4(src_ptr + i);
                vst1q_f32_x4(dst_ptr + i, data);
            }
            for (; i + 4 <= size; i += 4) {
                float32x4_t data = vld1q_f32(src_ptr + i);
                vst1q_f32(dst_ptr + i, data);
            }
            for (; i < size; ++i) {
                dst_ptr[i] = src_ptr[i];
            }
#else
            std::memcpy(dst.data(), src.data(), size * sizeof(float));
#endif
        });
    }
}

void test_weighted_accumulation() {
    const size_t size = 1024;
    std::vector<float> src(size);
    std::vector<float> dst(size);
    float weight = 0.5f;
    
    // Initialize with random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (size_t i = 0; i < size; ++i) {
        src[i] = dis(gen);
        dst[i] = dis(gen);
    }
    
    Benchmark::run_benchmark("Weighted Accumulation", [&]() {
#ifdef __ARM_NEON
        float32x4_t weight_vec = vdupq_n_f32(weight);
        size_t i = 0;
        for (; i + 16 <= size; i += 16) {
            float32x4x4_t src_data = vld1q_f32_x4(src.data() + i);
            float32x4x4_t dst_data = vld1q_f32_x4(dst.data() + i);
            
            dst_data.val[0] = vmlaq_f32(dst_data.val[0], src_data.val[0], weight_vec);
            dst_data.val[1] = vmlaq_f32(dst_data.val[1], src_data.val[1], weight_vec);
            dst_data.val[2] = vmlaq_f32(dst_data.val[2], src_data.val[2], weight_vec);
            dst_data.val[3] = vmlaq_f32(dst_data.val[3], src_data.val[3], weight_vec);
            
            vst1q_f32_x4(dst.data() + i, dst_data);
        }
        for (; i < size; ++i) {
            dst[i] += src[i] * weight;
        }
#else
        for (size_t i = 0; i < size; ++i) {
            dst[i] += src[i] * weight;
        }
#endif
    });
}

int main() {
    std::cout << "=== MoE Layer Optimization Benchmark ===" << std::endl;
    std::cout << "Platform: ";
#ifdef __ARM_NEON
    std::cout << "ARM with NEON" << std::endl;
#else
    std::cout << "Generic (no NEON)" << std::endl;
#endif
    std::cout << std::endl;
    
    test_memory_copy_optimization();
    std::cout << std::endl;
    
    test_weighted_accumulation();
    std::cout << std::endl;
    
    std::cout << "=== Benchmark Complete ===" << std::endl;
    
    return 0;
}