#include <chrono>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <cstdlib>

// Simple Q4_0 performance test
void test_q4_0_performance() {
    const int M = 512;  // Typical batch size
    const int N = 4096; // Typical hidden dimension
    const int K = 4096; // Typical input dimension
    const int num_iterations = 100;
    
    // Allocate matrices
    std::vector<float> A(M * K);
    std::vector<float> B(N * K);  // This would be quantized
    std::vector<float> C(M * N);
    
    // Initialize with random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (auto& val : A) val = dist(gen);
    for (auto& val : B) val = dist(gen);
    
    // Test different thread configurations
    std::vector<int> thread_counts = {1, 2, 4, 6, 8, 12, 16};
    
    std::cout << "Testing Q4_0 Performance with different thread counts\n";
    std::cout << "Matrix dimensions: M=" << M << ", N=" << N << ", K=" << K << "\n\n";
    
    for (int threads : thread_counts) {
        omp_set_num_threads(threads);
        
        std::vector<double> timings;
        
        for (int iter = 0; iter < num_iterations; ++iter) {
            auto start = std::chrono::high_resolution_clock::now();
            
            // Simulate GEMM operation with OpenMP
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
                    float sum = 0.0f;
                    for (int k = 0; k < K; ++k) {
                        sum += A[i * K + k] * B[j * K + k];
                    }
                    C[i * N + j] = sum;
                }
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            timings.push_back(duration / 1000.0); // Convert to milliseconds
        }
        
        // Calculate statistics
        double mean = 0.0;
        for (double t : timings) mean += t;
        mean /= timings.size();
        
        double variance = 0.0;
        for (double t : timings) {
            variance += (t - mean) * (t - mean);
        }
        variance /= timings.size();
        double stddev = std::sqrt(variance);
        
        std::sort(timings.begin(), timings.end());
        double min_time = timings[0];
        double max_time = timings[timings.size() - 1];
        double median = timings[timings.size() / 2];
        double p95 = timings[static_cast<int>(timings.size() * 0.95)];
        
        std::cout << "Threads: " << threads << "\n";
        std::cout << "  Mean: " << mean << " ms\n";
        std::cout << "  StdDev: " << stddev << " ms (CV: " << (stddev/mean*100) << "%)\n";
        std::cout << "  Min: " << min_time << " ms\n";
        std::cout << "  Max: " << max_time << " ms\n";
        std::cout << "  Median: " << median << " ms\n";
        std::cout << "  P95: " << p95 << " ms\n";
        std::cout << "  Variance ratio (max/min): " << (max_time/min_time) << "\n\n";
    }
}

// Test thread affinity
void test_thread_affinity() {
    std::cout << "Testing Thread Affinity Settings\n";
    std::cout << "================================\n\n";
    
    // Test different OMP environment settings
    const char* proc_bind_values[] = {"false", "true", "master", "close", "spread"};
    
    for (const char* bind_val : proc_bind_values) {
        setenv("OMP_PROC_BIND", bind_val, 1);
        std::cout << "OMP_PROC_BIND=" << bind_val << "\n";
        
        #pragma omp parallel num_threads(4)
        {
            #pragma omp single
            {
                std::cout << "  Active threads: " << omp_get_num_threads() << "\n";
            }
        }
    }
}

// Test scheduling policies
void test_scheduling_policies() {
    const int N = 10000;
    std::vector<float> data(N);
    
    std::cout << "\nTesting Scheduling Policies\n";
    std::cout << "===========================\n\n";
    
    // Static scheduling
    {
        auto start = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) {
            data[i] = std::sin(i) * std::cos(i);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "Static scheduling: " << duration << " us\n";
    }
    
    // Dynamic scheduling
    {
        auto start = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for schedule(dynamic, 16)
        for (int i = 0; i < N; ++i) {
            data[i] = std::sin(i) * std::cos(i);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "Dynamic scheduling (chunk=16): " << duration << " us\n";
    }
    
    // Guided scheduling
    {
        auto start = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for schedule(guided)
        for (int i = 0; i < N; ++i) {
            data[i] = std::sin(i) * std::cos(i);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "Guided scheduling: " << duration << " us\n";
    }
}

int main() {
    std::cout << "Q4_0 Performance Analysis Tool\n";
    std::cout << "==============================\n\n";
    
    // Print system info
    std::cout << "System Information:\n";
    std::cout << "  Max threads: " << omp_get_max_threads() << "\n";
    std::cout << "  Processors: " << omp_get_num_procs() << "\n";
    
    const char* omp_schedule = std::getenv("OMP_SCHEDULE");
    if (omp_schedule) {
        std::cout << "  OMP_SCHEDULE: " << omp_schedule << "\n";
    }
    
    const char* omp_num_threads = std::getenv("OMP_NUM_THREADS");
    if (omp_num_threads) {
        std::cout << "  OMP_NUM_THREADS: " << omp_num_threads << "\n";
    }
    
    std::cout << "\n";
    
    // Run tests
    test_q4_0_performance();
    test_thread_affinity();
    test_scheduling_policies();
    
    return 0;
}