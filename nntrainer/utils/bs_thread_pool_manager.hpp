// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   bs_threadpool_manager.hpp
 * @date   20 May 2025
 * @see    https://github.com/nntrainer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  BS threadpool manager class header file
 */

#ifndef THREAD_POOL_MANAGER_HPP
#define THREAD_POOL_MANAGER_HPP

#pragma once
#include "bs_thread_pool.h"
#include "singleton.h"

#include <cstdlib>
#include <thread>

namespace nntrainer {

/**
 * @brief Get the thread pool size, respecting OMP_NUM_THREADS to avoid
 * oversubscription when both BS thread pool and OMP coexist.
 *
 * @return unsigned int thread count for the BS thread pool
 */
inline unsigned int get_bs_pool_thread_count() {
  const char *env = std::getenv("OMP_NUM_THREADS");
  if (env != nullptr) {
    int val = std::atoi(env);
    if (val > 0)
      return static_cast<unsigned int>(val);
  }
  unsigned int hw = std::thread::hardware_concurrency();
  return (hw > 1) ? hw / 2 : 1;
}
/**
 * @brief ThreadPoolManager is a singleton class that manages a thread pool
 *
 */
class ThreadPoolManager : public Noncopyable, public Nonmovable {
public:
  /**
   * @brief Select optimal number of thread to use in K-quantized GEMM and GEMV
   *
   * @param M M for GEMM (M != 1) or GEMV (M = 1)
   * @param N N for GEMM or GEMV
   * @param K K for GEMM or GEMV
   * @return std::size_t number of thread to use
   */
  std::size_t select_k_quant_thread_count(unsigned int M, unsigned int N,
                                          unsigned int K);

  BS::thread_pool<> &getThreadPool() { return pool_; }

  /**
   * @brief Construct a new Thread Pool Manager object
   *
   */
  ThreadPoolManager() : pool_(get_bs_pool_thread_count()) {}
  /**
   * @brief Destroy the Thread Pool Manager object
   *
   */
  ~ThreadPoolManager() = default;

private:
  BS::thread_pool<> pool_;
};
} // namespace nntrainer

#endif // THREAD_POOL_MANAGER_HPP
