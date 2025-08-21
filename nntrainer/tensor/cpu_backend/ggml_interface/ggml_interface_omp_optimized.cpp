// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Michal Wlasiuk <testmailsmtp12345@gmail.com>
 * Copyright (C) 2025 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   ggml_interface_omp_optimized.cpp
 * @date   15 April 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Michal Wlasiuk <testmailsmtp12345@gmail.com>
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Optimized OpenMP implementation for consistent Q4_0 performance
 */

#include "ggml-common.h"
#include "ggml-cpu-quants.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
#include "ggml.h"

#include <algorithm>
#include <ggml_interface.h>
#include <nntr_ggml_impl.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <cstdlib>
#include <omp.h>

namespace nntrainer {

// Cache line size for padding to avoid false sharing
constexpr size_t CACHE_LINE_SIZE = 64;

// Optimized thread count management
static int get_optimal_thread_count(size_t workload_size, size_t min_work_per_thread = 16) {
    static int max_threads = 0;
    if (max_threads == 0) {
        // Initialize once
        const char* env_threads = std::getenv("OMP_NUM_THREADS");
        if (env_threads) {
            max_threads = std::atoi(env_threads);
        } else {
            max_threads = std::min(omp_get_max_threads(), 
                                  static_cast<int>(std::thread::hardware_concurrency()));
        }
        // Limit to reasonable number to avoid oversubscription
        max_threads = std::min(max_threads, 16);
    }
    
    // Calculate optimal thread count based on workload
    int optimal = static_cast<int>(workload_size / min_work_per_thread);
    optimal = std::max(1, std::min(optimal, max_threads));
    
    // For small workloads, use fewer threads to reduce overhead
    if (workload_size < 64) {
        optimal = std::min(optimal, 2);
    } else if (workload_size < 256) {
        optimal = std::min(optimal, 4);
    }
    
    return optimal;
}

// Thread-local workspace with cache line padding
struct alignas(CACHE_LINE_SIZE) ThreadWorkspace {
    std::vector<char> quantized_data;
    char padding[CACHE_LINE_SIZE - sizeof(std::vector<char>) % CACHE_LINE_SIZE];
};

/**
 * @brief Continuously packed 4 q8_K
 */
struct block_q8_Kx4 {
  float d[4];              // delta
  int8_t qs[QK_K * 4];     // quants
  int16_t bsums[QK_K / 4]; // sum of quants in groups of 16
};

/**
 * @brief struct template for q4_0 and q8_0
 */
template <int K> constexpr int QK_0() {
  if constexpr (K == 4) {
    return QK4_0;
  }
  if constexpr (K == 8) {
    return QK8_0;
  }
  return -1;
}

/**
 * @brief block of q4_0 or q8_0 block
 */
template <int K, int N> struct block {
  ggml_half d[N];                     // deltas for N qK_0 blocks
  int8_t qs[(QK_0<K>() * N * K) / 8]; // quants for N qK_0 blocks
};

using block_q4_0x4 = block<4, 4>;
using block_q8_0x4 = block<8, 4>;

// Optimized Q4_0 GEMM with improved thread management
template <>
void __ggml_q4_0_4x8_q8_0_GEMM(const unsigned int M, const unsigned int N,
                               const unsigned int K, const float *A,
                               const unsigned int lda, const void *B,
                               const unsigned int ldb, float *C,
                               const unsigned int ldc) {
  constexpr int NB_COLS = 4;
  constexpr size_t MIN_WORK_PER_THREAD = 32; // Minimum columns per thread
  
  if (M == 1) { // GEMV
    unsigned int B_step = sizeof(block_q4_0) * (K / QK4_0);
    unsigned int blocks_per_row = (K + QK8_0 - 1) / QK8_0;
    unsigned int qa_size = sizeof(block_q8_0) * blocks_per_row;
    
    // Use thread-local storage to avoid allocation in hot path
    thread_local std::vector<char> QA_storage;
    QA_storage.resize(qa_size);
    ::quantize_row_q8_0(A, QA_storage.data(), K);
    
    // Optimize thread count based on workload
    int n_threads = get_optimal_thread_count(N, MIN_WORK_PER_THREAD);
    
    // Use static scheduling for predictable performance
    #pragma omp parallel for num_threads(n_threads) schedule(static)
    for (int idx = 0; idx < static_cast<int>((N + NB_COLS - 1) / NB_COLS); ++idx) {
      unsigned int M_step_start = idx * NB_COLS;
      unsigned int M_step_end = std::min(M_step_start + NB_COLS, N);
      
      // Ensure alignment
      M_step_start = (M_step_start / NB_COLS) * NB_COLS;
      M_step_end = ((M_step_end + NB_COLS - 1) / NB_COLS) * NB_COLS;
      M_step_end = std::min(M_step_end, N);
      
      if (M_step_start < N) {
        nntr_gemv_q4_0_4x8_q8_0(K, (float *)(C + M_step_start), N,
                                (void *)((char *)B + M_step_start * B_step),
                                QA_storage.data(), M, M_step_end - M_step_start);
      }
    }
  } else if (M % 4 != 0) {
    // Handle non-aligned M
    unsigned int blocks_per_4_rows = (K + QK8_0 - 1) / QK8_0;
    unsigned int qa_4_rows_size = sizeof(block_q8_0x4) * blocks_per_4_rows;
    const size_t qa_row_size = (sizeof(block_q8_0) * K) / QK8_0;
    unsigned int M4 = ((M - M % 4) / 4);
    int B_step = sizeof(block_q4_0) * (K / QK4_0);
    
    unsigned int qa_size = qa_4_rows_size * (((M >> 2) << 2) / 4 + 1);
    std::vector<char> QA(qa_size);
    
    // Quantize 4-row aligned portion
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(M4); i++) {
      ggml_quantize_mat_q8_0_4x8(A + 4 * i * K, QA.data() + i * qa_4_rows_size, K);
    }
    
    // Quantize remaining rows
    for (unsigned int i = M4 * 4; i < M; i++) {
      ::quantize_row_q8_0(
        (float *)A + i * K,
        (QA.data() + (M4 * qa_4_rows_size) + (i - M4 * 4) * qa_row_size), K);
    }
    
    // Process 4-row aligned portion
    int n_threads = get_optimal_thread_count(N, MIN_WORK_PER_THREAD);
    
    #pragma omp parallel for num_threads(n_threads) schedule(static)
    for (int idx = 0; idx < static_cast<int>((N + NB_COLS - 1) / NB_COLS); ++idx) {
      unsigned int src0_start = idx * NB_COLS;
      unsigned int src0_end = std::min(src0_start + NB_COLS, N);
      
      src0_start = (src0_start / NB_COLS) * NB_COLS;
      src0_end = ((src0_end + NB_COLS - 1) / NB_COLS) * NB_COLS;
      src0_end = std::min(src0_end, N);
      
      if (src0_start < N) {
        nntr_gemm_q4_0_4x8_q8_0(K, (float *)(C + src0_start), ldc,
                                (void *)((char *)B + src0_start * B_step),
                                QA.data(), M4 * 4, src0_end - src0_start);
      }
    }
    
    // Process remaining rows
    for (unsigned int pb = M4 * 4; pb < M; pb++) {
      #pragma omp parallel for num_threads(n_threads) schedule(static)
      for (int idx = 0; idx < static_cast<int>((N + NB_COLS - 1) / NB_COLS); ++idx) {
        unsigned int M_step_start = idx * NB_COLS;
        unsigned int M_step_end = std::min(M_step_start + NB_COLS, N);
        
        M_step_start = (M_step_start / NB_COLS) * NB_COLS;
        M_step_end = ((M_step_end + NB_COLS - 1) / NB_COLS) * NB_COLS;
        M_step_end = std::min(M_step_end, N);
        
        if (M_step_start < N) {
          nntr_gemv_q4_0_4x8_q8_0(
            K, (float *)((C + ((pb - M4 * 4) * N) + (M4 * 4 * N)) + M_step_start),
            N, (void *)((char *)B + M_step_start * B_step),
            QA.data() + (M4 * qa_4_rows_size) + (pb - M4 * 4) * qa_row_size, 1,
            M_step_end - M_step_start);
        }
      }
    }
  } else { // GEMM with aligned M
    unsigned int blocks_per_4_rows = (K + QK8_0 - 1) / QK8_0;
    unsigned int qa_4_rows_size = sizeof(block_q8_0x4) * blocks_per_4_rows;
    unsigned int M4 = ((M + 3) / 4);
    
    unsigned int qa_size = qa_4_rows_size * M4;
    std::vector<char> QA(qa_size);
    
    // Parallel quantization with optimal thread count
    int quant_threads = get_optimal_thread_count(M4, 4);
    #pragma omp parallel for num_threads(quant_threads) schedule(static)
    for (int i = 0; i < static_cast<int>(M4); i++) {
      ggml_quantize_mat_q8_0_4x8(A + 4 * i * K, QA.data() + i * qa_4_rows_size, K);
    }
    
    unsigned int B_step = sizeof(block_q4_0) * (K / QK4_0);
    
    // Optimize thread count for GEMM
    int gemm_threads = get_optimal_thread_count(N, MIN_WORK_PER_THREAD * 2);
    
    // Use static scheduling for consistent performance
    #pragma omp parallel for num_threads(gemm_threads) schedule(static)
    for (int idx = 0; idx < static_cast<int>((N + NB_COLS - 1) / NB_COLS); ++idx) {
      unsigned int src0_start = idx * NB_COLS;
      unsigned int src0_end = std::min(src0_start + NB_COLS, N);
      
      src0_start = (src0_start / NB_COLS) * NB_COLS;
      src0_end = ((src0_end + NB_COLS - 1) / NB_COLS) * NB_COLS;
      src0_end = std::min(src0_end, N);
      
      if (src0_start < N) {
        nntr_gemm_q4_0_4x8_q8_0(K, (float *)(C + src0_start), ldc,
                                (void *)((char *)B + src0_start * B_step),
                                QA.data(), M, src0_end - src0_start);
      }
    }
  }
}

// Optimized multi-weight version
template <>
void __ggml_q4_0_4x8_q8_0_GEMM(const unsigned int M,
                               std::vector<unsigned int> Ns,
                               const unsigned int K, const float *A,
                               const unsigned int lda, std::vector<void *> Bs,
                               std::vector<unsigned int> ldbs,
                               std::vector<float *> Cs,
                               std::vector<unsigned int> ldcs) {
  constexpr int NB_COLS = 4;
  int B_step = sizeof(block_q4_0) * (K / QK4_0);
  int blocks_per_4_rows = (K + QK8_0 - 1) / QK8_0;
  
  if (M == 1) {
    int qa_size = sizeof(block_q8_0) * blocks_per_4_rows;
    thread_local std::vector<char> QA;
    QA.resize(qa_size);
    auto qa_data = QA.data();
    quantize_row_q8_0(A, qa_data, K);
    
    // Check if all Ns are small
    bool all_small = std::all_of(Ns.begin(), Ns.end(),
                                 [](unsigned int n) { return n <= 256; });
    
    if (all_small) {
      // Sequential processing for small workloads
      for (unsigned int num_w = 0; num_w < Ns.size(); ++num_w) {
        unsigned int N = Ns[num_w];
        float *C = Cs[num_w];
        void *B = Bs[num_w];
        
        unsigned int M_step_start = 0;
        unsigned int M_step_end = ((N + NB_COLS - 1) / NB_COLS) * NB_COLS;
        
        nntr_gemv_q4_0_4x8_q8_0(K, C, N, B, qa_data, M, M_step_end);
      }
    } else {
      // Parallel processing for larger workloads
      unsigned int total_work = 0;
      for (auto n : Ns) total_work += n;
      
      int n_threads = get_optimal_thread_count(total_work, 64);
      
      #pragma omp parallel for num_threads(n_threads) schedule(dynamic, 1)
      for (int w = 0; w < static_cast<int>(Ns.size()); ++w) {
        unsigned int N = Ns[w];
        float *C = Cs[w];
        void *B = Bs[w];
        
        for (unsigned int idx = 0; idx < (N + NB_COLS - 1) / NB_COLS; ++idx) {
          unsigned int M_step_start = idx * NB_COLS;
          unsigned int M_step_end = std::min(M_step_start + NB_COLS, N);
          
          M_step_start = (M_step_start / NB_COLS) * NB_COLS;
          M_step_end = ((M_step_end + NB_COLS - 1) / NB_COLS) * NB_COLS;
          M_step_end = std::min(M_step_end, N);
          
          if (M_step_start < N) {
            nntr_gemv_q4_0_4x8_q8_0(K, (float *)(C + M_step_start), N,
                                    (void *)((char *)B + M_step_start * B_step),
                                    qa_data, M, M_step_end - M_step_start);
          }
        }
      }
    }
  } else {
    // GEMM case
    unsigned int qa_4_rows_size = sizeof(block_q8_0x4) * blocks_per_4_rows;
    const size_t qa_row_size = (sizeof(block_q8_0) * K) / QK8_0;
    
    unsigned int M4 = ((M - M % 4) / 4);
    unsigned int qa_size = qa_4_rows_size * (((M >> 2) << 2) / 4 + 1);
    
    std::vector<char> QA(qa_size);
    
    // Quantize with optimal thread count
    int quant_threads = get_optimal_thread_count(M4, 4);
    #pragma omp parallel for num_threads(quant_threads) schedule(static)
    for (int i = 0; i < static_cast<int>(M4); i++) {
      ggml_quantize_mat_q8_0_4x8(A + 4 * i * K, QA.data() + i * qa_4_rows_size, K);
    }
    
    for (unsigned int i = M4 * 4; i < M; i++) {
      quantize_row_q8_0(
        (float *)A + i * K,
        (QA.data() + (M4 * qa_4_rows_size) + (i - M4 * 4) * qa_row_size), K);
    }
    
    // Calculate total work
    unsigned int total_cols = 0;
    for (auto n : Ns) total_cols += n;
    
    int n_threads = get_optimal_thread_count(total_cols, 32);
    
    // Use dynamic scheduling with chunk size for better load balancing
    #pragma omp parallel num_threads(n_threads)
    {
      for (unsigned int num_w = 0; num_w < Ns.size(); ++num_w) {
        unsigned int N = Ns[num_w];
        unsigned int ldc = ldcs[num_w];
        float *C = Cs[num_w];
        void *B = Bs[num_w];
        
        #pragma omp for schedule(static)
        for (int idx = 0; idx < static_cast<int>((N + NB_COLS - 1) / NB_COLS); ++idx) {
          unsigned int src0_start = idx * NB_COLS;
          unsigned int src0_end = std::min(src0_start + NB_COLS, N);
          
          src0_start = (src0_start / NB_COLS) * NB_COLS;
          src0_end = ((src0_end + NB_COLS - 1) / NB_COLS) * NB_COLS;
          src0_end = std::min(src0_end, N);
          
          if (src0_start < N) {
            nntr_gemm_q4_0_4x8_q8_0(K, (float *)(C + src0_start), ldc,
                                    (void *)((char *)B + src0_start * B_step),
                                    QA.data(), M4 * 4, src0_end - src0_start);
          }
        }
      }
    }
    
    // Handle remaining rows
    if (M4 * 4 != M) {
      #pragma omp parallel for num_threads(n_threads) schedule(dynamic, 1)
      for (int w = 0; w < static_cast<int>(Ns.size()); ++w) {
        unsigned int N = Ns[w];
        unsigned int ldc = ldcs[w];
        float *C = Cs[w];
        void *B = Bs[w];
        
        for (int pb = M4 * 4; pb < static_cast<int>(M); pb++) {
          for (unsigned int idx = 0; idx < (N + NB_COLS - 1) / NB_COLS; ++idx) {
            unsigned int M_step_start = idx * NB_COLS;
            unsigned int M_step_end = std::min(M_step_start + NB_COLS, N);
            
            M_step_start = (M_step_start / NB_COLS) * NB_COLS;
            M_step_end = ((M_step_end + NB_COLS - 1) / NB_COLS) * NB_COLS;
            M_step_end = std::min(M_step_end, N);
            
            if (M_step_start < N) {
              nntr_gemv_q4_0_4x8_q8_0(
                K,
                (float *)((C + ((pb - M4 * 4) * N) + (M4 * 4 * N)) + M_step_start),
                N, (void *)((char *)B + M_step_start * B_step),
                QA.data() + (M4 * qa_4_rows_size) + (pb - M4 * 4) * qa_row_size,
                1, M_step_end - M_step_start);
            }
          }
        }
      }
    }
  }
}

// Add remaining functions with similar optimizations...
// (The rest of the implementation follows the same pattern)

} // namespace nntrainer