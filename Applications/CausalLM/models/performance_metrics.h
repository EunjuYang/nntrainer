// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    performance_metrics.h
 * @date    24 Mar 2026
 * @brief   Performance metrics definitions shared between models and API layers
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Eunju Yang <ej.yang@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#ifndef __CAUSAL_LM_PERFORMANCE_METRICS_H__
#define __CAUSAL_LM_PERFORMANCE_METRICS_H__

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Performance Metrics
 */
typedef struct {
  unsigned int prefill_tokens;
  double prefill_duration_ms;
  unsigned int generation_tokens;
  double generation_duration_ms;
  double total_duration_ms;
  double initialization_duration_ms;
  size_t peak_memory_kb;
} PerformanceMetrics;

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

/**
 * @brief Get peak memory usage in KB
 */
size_t getPeakMemoryKb();

#endif // __cplusplus

#endif // __CAUSAL_LM_PERFORMANCE_METRICS_H__
