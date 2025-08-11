// SPDX-License-Identifier: Apache-2.0
/**
 * @file   android_memory_optimizer.h
 * @brief  Android-specific memory optimization utilities for large models
 */

#ifndef __ANDROID_MEMORY_OPTIMIZER_H__
#define __ANDROID_MEMORY_OPTIMIZER_H__

#include <cstddef>
#include <memory>
#include <vector>

namespace causallm {

class AndroidMemoryOptimizer {
public:
  /**
   * @brief Get optimal chunk size for Android device
   * @return Optimal chunk size in bytes
   */
  static size_t getOptimalChunkSize();

  /**
   * @brief Check available memory on Android device
   * @return Available memory in bytes
   */
  static size_t getAvailableMemory();

  /**
   * @brief Configure memory hints for large model loading
   * @param total_size Total model size in bytes
   */
  static void configureMemoryHints(size_t total_size);

  /**
   * @brief Preload critical model parts into memory
   * @param fd File descriptor
   * @param critical_offsets Offsets of critical parts
   * @param sizes Sizes of critical parts
   */
  static void preloadCriticalParts(int fd, 
                                   const std::vector<size_t>& critical_offsets,
                                   const std::vector<size_t>& sizes);

  /**
   * @brief Enable memory compression if supported
   */
  static void enableMemoryCompression();

  /**
   * @brief Set process priority for better memory allocation
   */
  static void setHighPriority();

private:
  static constexpr size_t DEFAULT_CHUNK_SIZE = 64 * 1024 * 1024; // 64MB chunks
  static constexpr size_t MIN_CHUNK_SIZE = 16 * 1024 * 1024;     // 16MB minimum
  static constexpr size_t MAX_CHUNK_SIZE = 256 * 1024 * 1024;    // 256MB maximum
};

} // namespace causallm

#endif // __ANDROID_MEMORY_OPTIMIZER_H__