// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   neon_setting.h
 * @date   18 Jan 2024
 * @see    https://github.com/nntrainer/nntrainer
 *         https://arxiv.org/abs/1706.03762
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This file is for OpenMP setting
 *
 */

#include <algorithm>
#include <omp.h>

/// @note This variable should be optimized by user
/// @todo Must find a general solution to optimize the functionality of
/// multithreading : determining the combination of #threads and size of
/// (M x K) x (K x N) GEMM
/**
 * @brief Function for setting the number of threads to use for GEMM
 *
 * @return size_t& num_threads
 */
inline size_t &GEMM_NUM_THREADS() {
  static size_t num_threads = 1;
  return num_threads;
}
/**
 * @brief Set the gemm num threads
 *
 * @param n num_threads to set
 */
inline void set_gemm_num_threads(size_t n) { GEMM_NUM_THREADS() = n; }
/**
 * @brief Get the gemm num threads
 *
 * @return size_t num_threads
 */
inline size_t get_gemm_num_threads() { return GEMM_NUM_THREADS(); }
/**
 * @brief Function for setting the number of threads to use for GEMV
 *
 * @return size_t& num_threads
 */
inline size_t &GEMV_NUM_THREADS() {
  static size_t num_threads = 1;
  return num_threads;
}
/**
 * @brief Set the gemv num threads
 *
 * @param n num_threads to set
 */
inline void set_gemv_num_threads(size_t n) { GEMV_NUM_THREADS() = n; }
/**
 * @brief Get the gemv num threads
 *
 * @return size_t num_threads
 */
inline size_t get_gemv_num_threads() { return GEMV_NUM_THREADS(); }
/**
 * @brief Select optimal thread count for GEMV based on matrix dimensions.
 * Uses compile-time OMP_NUM_THREADS if defined, otherwise applies a
 * work-size based heuristic suitable for mobile/ARM targets.
 *
 * @param M number of rows
 * @param N number of columns
 * @return size_t optimal thread count
 */
inline size_t select_gemv_num_threads(uint32_t M, uint32_t N) {
#ifdef OMP_NUM_THREADS
  return OMP_NUM_THREADS;
#endif
  size_t work_size = static_cast<size_t>(M) * N;
  if (work_size < 256 * 1024)
    return 1;
  if (work_size < 1024 * 1024)
    return 2;
  return 4;
}
