// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   omp_setting.h
 * @date   19 Mar 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang
 * @bug    No known bugs except for NYI items
 * @brief  This file is for OpenMP thread count settings shared across all CPU
 *         backends (ARM, x86, etc.)
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
inline int &nntr_omp_num_threads_override() {
  static int override_val = 0;
  return override_val;
}

/**
 * @brief Get runtime omp num threads
 */
inline int get_runtime_omp_num_threads() {
  static int cached = []() {
    const char *env = std::getenv("OMP_NUM_THREADS");
    if (env) {
      int val = std::atoi(env);
      if (val > 0)
        return val;
    }
    int hw = static_cast<int>(std::thread::hardware_concurrency());
    return (hw > 1) ? hw / 2 : 1;
  }();

  int override_val = nntr_omp_num_threads_override();
  return (override_val > 0) ? override_val : cached;
}

/**
 * @brief Dynamically override the OMP thread count.
 * Set to 0 to revert to OMP_NUM_THREADS env var / hardware default.
 *
 * @param n num_threads to set (0 to clear override)
 */
inline void set_runtime_omp_num_threads(int n) {
  nntr_omp_num_threads_override() = n;
}

#endif /* __OMP_SETTING_H__ */
