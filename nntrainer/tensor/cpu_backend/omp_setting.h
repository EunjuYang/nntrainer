// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   omp_setting.h
 * @date   18 Jan 2024
 * @see    https://github.com/nntrainer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This file is for OpenMP thread count settings shared across all CPU
 *         backends (ARM, x86, etc.)
 *
 */

#ifndef __OMP_SETTING_H__
#define __OMP_SETTING_H__

#include <cstdlib>
#include <thread>

/**
 * @brief Get runtime thread count from OMP_NUM_THREADS environment variable.
 * Falls back to hardware_concurrency()/2 if not set.
 * Supports dynamic override: if set_runtime_omp_num_threads() has been called,
 * returns that value instead.
 *
 * @return int thread count to use
 */
inline int &_omp_num_threads_override() {
  static int override_val = 0;
  return override_val;
}

inline int get_runtime_omp_num_threads() {
  int override_val = _omp_num_threads_override();
  if (override_val > 0)
    return override_val;

  const char *env = std::getenv("OMP_NUM_THREADS");
  if (env != nullptr) {
    int val = std::atoi(env);
    if (val > 0) {
      return val;
    }
  }
  int hw = static_cast<int>(std::thread::hardware_concurrency());
  return (hw > 1) ? hw / 2 : 1;
}

/**
 * @brief Dynamically override the OMP thread count.
 * Set to 0 to revert to OMP_NUM_THREADS env var / hardware default.
 *
 * @param n num_threads to set (0 to clear override)
 */
inline void set_runtime_omp_num_threads(int n) {
  _omp_num_threads_override() = n;
}

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

#endif /* __OMP_SETTING_H__ */
