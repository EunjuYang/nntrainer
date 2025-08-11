// SPDX-License-Identifier: Apache-2.0
/**
 * @file   android_memory_optimizer.cpp
 * @brief  Android-specific memory optimization implementation
 */

#include "android_memory_optimizer.h"

#ifdef __ANDROID__
#include <android/log.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>
#include <fcntl.h>
#include <fstream>
#include <sstream>
#include <string>

#define LOG_TAG "NNTrainer-CausalLM"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)

namespace causallm {

size_t AndroidMemoryOptimizer::getOptimalChunkSize() {
  // Get available memory
  size_t available = getAvailableMemory();
  
  // Use 1/8 of available memory as chunk size, bounded by min/max
  size_t chunk_size = available / 8;
  
  if (chunk_size < MIN_CHUNK_SIZE) {
    chunk_size = MIN_CHUNK_SIZE;
  } else if (chunk_size > MAX_CHUNK_SIZE) {
    chunk_size = MAX_CHUNK_SIZE;
  }
  
  // Align to page size
  long page_size = sysconf(_SC_PAGESIZE);
  chunk_size = (chunk_size / page_size) * page_size;
  
  LOGI("Optimal chunk size: %zu MB", chunk_size / (1024 * 1024));
  return chunk_size;
}

size_t AndroidMemoryOptimizer::getAvailableMemory() {
  std::ifstream meminfo("/proc/meminfo");
  std::string line;
  size_t available_kb = 0;
  
  while (std::getline(meminfo, line)) {
    if (line.find("MemAvailable:") == 0) {
      std::istringstream iss(line);
      std::string label;
      iss >> label >> available_kb;
      break;
    }
  }
  
  size_t available_bytes = available_kb * 1024;
  LOGI("Available memory: %zu MB", available_bytes / (1024 * 1024));
  return available_bytes;
}

void AndroidMemoryOptimizer::configureMemoryHints(size_t total_size) {
  // Increase file descriptor limit
  struct rlimit rl;
  if (getrlimit(RLIMIT_NOFILE, &rl) == 0) {
    rl.rlim_cur = rl.rlim_max;
    setrlimit(RLIMIT_NOFILE, &rl);
  }
  
  // Try to increase virtual memory limit
  if (getrlimit(RLIMIT_AS, &rl) == 0) {
    rl.rlim_cur = rl.rlim_max;
    setrlimit(RLIMIT_AS, &rl);
  }
  
  // Configure memory overcommit (requires root)
  int fd = open("/proc/sys/vm/overcommit_memory", O_WRONLY);
  if (fd != -1) {
    write(fd, "1", 1);  // Always overcommit
    close(fd);
  }
  
  // Configure swappiness (prefer keeping pages in memory)
  fd = open("/proc/sys/vm/swappiness", O_WRONLY);
  if (fd != -1) {
    write(fd, "10", 2);  // Low swappiness value
    close(fd);
  }
  
  LOGI("Memory hints configured for %zu MB model", total_size / (1024 * 1024));
}

void AndroidMemoryOptimizer::preloadCriticalParts(int fd,
                                                  const std::vector<size_t>& critical_offsets,
                                                  const std::vector<size_t>& sizes) {
  long page_size = sysconf(_SC_PAGESIZE);
  
  for (size_t i = 0; i < critical_offsets.size() && i < sizes.size(); ++i) {
    size_t offset = critical_offsets[i];
    size_t size = sizes[i];
    
    // Align to page boundaries
    size_t aligned_offset = (offset / page_size) * page_size;
    size_t diff = offset - aligned_offset;
    size_t aligned_size = size + diff;
    
    // Map and immediately touch pages to load them
    void* ptr = mmap(nullptr, aligned_size, PROT_READ, MAP_PRIVATE, fd, aligned_offset);
    if (ptr != MAP_FAILED) {
      // Use madvise to hint kernel about access pattern
      madvise(ptr, aligned_size, MADV_WILLNEED);
      madvise(ptr, aligned_size, MADV_SEQUENTIAL);
      
      // Touch first byte of each page to trigger loading
      for (size_t j = 0; j < aligned_size; j += page_size) {
        volatile char dummy = ((char*)ptr)[j];
        (void)dummy;
      }
      
      // Keep the mapping for now (will be unmapped when replaced)
      LOGI("Preloaded %zu KB at offset %zu", size / 1024, offset);
    }
  }
}

void AndroidMemoryOptimizer::enableMemoryCompression() {
  // Enable ZRAM compression if available (requires root)
  int fd = open("/sys/block/zram0/comp_algorithm", O_WRONLY);
  if (fd != -1) {
    write(fd, "lz4", 3);  // Use LZ4 for fast compression
    close(fd);
    LOGI("Memory compression enabled (LZ4)");
  }
  
  // Configure ZRAM size
  fd = open("/sys/block/zram0/disksize", O_WRONLY);
  if (fd != -1) {
    // Set ZRAM to 2GB
    write(fd, "2147483648", 10);
    close(fd);
  }
}

void AndroidMemoryOptimizer::setHighPriority() {
  // Set nice value to give process higher priority
  if (nice(-10) == -1) {
    LOGW("Failed to set process priority");
  }
  
  // Set I/O priority (requires CAP_SYS_ADMIN)
  // syscall(__NR_ioprio_set, 1, 0, 0x6000); // IOPRIO_CLASS_RT
  
  // Set CPU affinity to performance cores if possible
  // This is device-specific and would need more sophisticated detection
  
  LOGI("Process priority adjusted");
}

} // namespace causallm

#else // !__ANDROID__

namespace causallm {

// Stub implementations for non-Android platforms
size_t AndroidMemoryOptimizer::getOptimalChunkSize() {
  return DEFAULT_CHUNK_SIZE;
}

size_t AndroidMemoryOptimizer::getAvailableMemory() {
  return 4ULL * 1024 * 1024 * 1024; // Default to 4GB
}

void AndroidMemoryOptimizer::configureMemoryHints(size_t total_size) {
  // No-op on non-Android
}

void AndroidMemoryOptimizer::preloadCriticalParts(int fd,
                                                  const std::vector<size_t>& critical_offsets,
                                                  const std::vector<size_t>& sizes) {
  // No-op on non-Android
}

void AndroidMemoryOptimizer::enableMemoryCompression() {
  // No-op on non-Android
}

void AndroidMemoryOptimizer::setHighPriority() {
  // No-op on non-Android
}

} // namespace causallm

#endif // __ANDROID__