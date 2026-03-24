// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    performance_metrics.cpp
 * @date    24 Mar 2026
 * @brief   Performance metrics implementation
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Eunju Yang <ej.yang@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#include "performance_metrics.h"

#ifdef _WIN32
#include <psapi.h>
#include <windows.h>
#else
#include <sys/resource.h>
#endif

size_t getPeakMemoryKb() {
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS pmc;
  if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
    return (size_t)(pmc.PeakWorkingSetSize / 1024);
  }
  return 0;
#else
  struct rusage rusage;
  if (getrusage(RUSAGE_SELF, &rusage) == 0) {
    return (size_t)(rusage.ru_maxrss);
  }
  return 0;
#endif
}
