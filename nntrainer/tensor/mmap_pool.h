// SPDX-License-Identifier: Apache-2.0
/**
 * @file   mmap_pool.h
 * @brief  MMap pooling for efficient tensor activation/deactivation
 */

#ifndef __MMAP_POOL_H__
#define __MMAP_POOL_H__

#include <unordered_map>
#include <list>
#include <mutex>
#include <chrono>
#include <cstddef>

namespace nntrainer {

/**
 * @class MMapPool
 * @brief Manages a pool of mmap'd regions to reduce syscall overhead
 */
class MMapPool {
public:
  struct MMapEntry {
    void* addr;           // Mapped address
    size_t size;          // Size of mapping
    int fd;               // File descriptor
    off_t offset;         // File offset
    std::chrono::steady_clock::time_point last_access;
    bool in_use;          // Currently in use
    int access_count;     // Number of times accessed
  };

  /**
   * @brief Get singleton instance
   */
  static MMapPool& getInstance() {
    static MMapPool instance;
    return instance;
  }

  /**
   * @brief Request a mapped region (may reuse existing mapping)
   * @param fd File descriptor
   * @param offset File offset
   * @param size Size to map
   * @return Mapped pointer or MAP_FAILED
   */
  void* requestMapping(int fd, off_t offset, size_t size);

  /**
   * @brief Release a mapping (may keep it cached)
   * @param addr Mapped address
   * @param should_keep_cached Whether to keep in cache
   */
  void releaseMapping(void* addr, bool should_keep_cached = true);

  /**
   * @brief Clear old unused mappings
   * @param max_age_ms Maximum age in milliseconds
   */
  void clearOldMappings(int max_age_ms = 5000);

  /**
   * @brief Set maximum pool size
   * @param max_size Maximum total size of cached mappings
   */
  void setMaxPoolSize(size_t max_size) { max_pool_size_ = max_size; }

  /**
   * @brief Get statistics about pool usage
   */
  struct Stats {
    size_t total_mappings;
    size_t active_mappings;
    size_t cached_mappings;
    size_t total_size;
    size_t hit_count;
    size_t miss_count;
  };
  
  Stats getStats() const;

  /**
   * @brief Clear all cached mappings
   */
  void clearAll();

  /**
   * @brief Enable/disable pooling
   */
  void setEnabled(bool enabled) { enabled_ = enabled; }

private:
  MMapPool() : max_pool_size_(256 * 1024 * 1024), // 256MB default
               enabled_(true), hit_count_(0), miss_count_(0) {}
  
  ~MMapPool() { clearAll(); }

  // Prevent copying
  MMapPool(const MMapPool&) = delete;
  MMapPool& operator=(const MMapPool&) = delete;

  // Generate key for mapping lookup
  std::string generateKey(int fd, off_t offset, size_t size) const;

  // Evict least recently used mappings if needed
  void evictLRU();

  // Check if we should keep this mapping cached
  bool shouldCache(const MMapEntry& entry) const;

private:
  mutable std::mutex mutex_;
  std::unordered_map<std::string, MMapEntry> mappings_;  // Key -> Entry
  std::unordered_map<void*, std::string> addr_to_key_;   // Address -> Key
  std::list<std::string> lru_list_;                      // LRU order
  std::unordered_map<std::string, std::list<std::string>::iterator> lru_map_;
  
  size_t max_pool_size_;
  size_t current_pool_size_ = 0;
  bool enabled_;
  
  // Statistics
  size_t hit_count_;
  size_t miss_count_;
};

} // namespace nntrainer

#endif // __MMAP_POOL_H__