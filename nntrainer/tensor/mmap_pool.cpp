// SPDX-License-Identifier: Apache-2.0
/**
 * @file   mmap_pool.cpp
 * @brief  MMap pooling implementation
 */

#include "mmap_pool.h"
#include <sys/mman.h>
#include <unistd.h>
#include <sstream>
#include <iostream>

#ifdef __ANDROID__
#include <android/log.h>
#define LOG_TAG "NNTrainer-MMapPool"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#else
#define LOGI(...) std::cout << __VA_ARGS__ << std::endl
#endif

namespace nntrainer {

std::string MMapPool::generateKey(int fd, off_t offset, size_t size) const {
  std::stringstream ss;
  ss << fd << "_" << offset << "_" << size;
  return ss.str();
}

void* MMapPool::requestMapping(int fd, off_t offset, size_t size) {
  if (!enabled_) {
    // Pooling disabled, just do regular mmap
    return mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, offset);
  }

  std::lock_guard<std::mutex> lock(mutex_);
  
  std::string key = generateKey(fd, offset, size);
  
  // Check if we already have this mapping
  auto it = mappings_.find(key);
  if (it != mappings_.end() && !it->second.in_use) {
    // Cache hit!
    hit_count_++;
    it->second.in_use = true;
    it->second.last_access = std::chrono::steady_clock::now();
    it->second.access_count++;
    
    // Update LRU
    lru_list_.erase(lru_map_[key]);
    lru_list_.push_back(key);
    lru_map_[key] = --lru_list_.end();
    
#ifdef __ANDROID__
    // Re-advise kernel since we're reusing the mapping
    madvise(it->second.addr, it->second.size, MADV_WILLNEED);
#endif
    
    LOGI("MMapPool: Cache hit for offset %ld, size %zu", offset, size);
    return it->second.addr;
  }
  
  // Cache miss - need to create new mapping
  miss_count_++;
  
  // Check if we need to evict
  if (current_pool_size_ + size > max_pool_size_) {
    evictLRU();
  }
  
  // Create new mapping
#ifdef __ANDROID__
  void* addr = mmap(nullptr, size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, offset);
#else
  void* addr = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, offset);
#endif
  
  if (addr == MAP_FAILED) {
    return MAP_FAILED;
  }
  
#ifdef __ANDROID__
  // Optimize for Android
  madvise(addr, size, MADV_SEQUENTIAL);
  madvise(addr, size, MADV_WILLNEED);
  
  // Lock small mappings
  if (size < 8 * 1024 * 1024) { // Less than 8MB
    mlock(addr, size);
  }
#endif
  
  // Add to pool
  MMapEntry entry;
  entry.addr = addr;
  entry.size = size;
  entry.fd = fd;
  entry.offset = offset;
  entry.last_access = std::chrono::steady_clock::now();
  entry.in_use = true;
  entry.access_count = 1;
  
  mappings_[key] = entry;
  addr_to_key_[addr] = key;
  current_pool_size_ += size;
  
  // Add to LRU
  lru_list_.push_back(key);
  lru_map_[key] = --lru_list_.end();
  
  LOGI("MMapPool: Created new mapping for offset %ld, size %zu", offset, size);
  return addr;
}

void MMapPool::releaseMapping(void* addr, bool should_keep_cached) {
  if (!enabled_ || !should_keep_cached) {
    // Just unmap immediately
    // Need to find size first
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = addr_to_key_.find(addr);
    if (it != addr_to_key_.end()) {
      auto& entry = mappings_[it->second];
      size_t size = entry.size;
      
#ifdef __ANDROID__
      if (size < 8 * 1024 * 1024) {
        munlock(addr, size);
      }
      madvise(addr, size, MADV_DONTNEED);
#endif
      
      munmap(addr, size);
      
      // Remove from pool
      current_pool_size_ -= size;
      lru_list_.erase(lru_map_[it->second]);
      lru_map_.erase(it->second);
      mappings_.erase(it->second);
      addr_to_key_.erase(it);
    } else {
      // Not in pool, just unmap with page-aligned size guess
      // This shouldn't happen in normal operation
      munmap(addr, 4096);
    }
    return;
  }
  
  std::lock_guard<std::mutex> lock(mutex_);
  
  auto it = addr_to_key_.find(addr);
  if (it == addr_to_key_.end()) {
    // Not found in pool
    return;
  }
  
  std::string key = it->second;
  auto& entry = mappings_[key];
  
  entry.in_use = false;
  entry.last_access = std::chrono::steady_clock::now();
  
  // Decide if we should keep it cached based on access pattern
  if (!shouldCache(entry)) {
    // Remove from cache
#ifdef __ANDROID__
    if (entry.size < 8 * 1024 * 1024) {
      munlock(entry.addr, entry.size);
    }
    madvise(entry.addr, entry.size, MADV_DONTNEED);
#endif
    
    munmap(entry.addr, entry.size);
    
    current_pool_size_ -= entry.size;
    lru_list_.erase(lru_map_[key]);
    lru_map_.erase(key);
    mappings_.erase(key);
    addr_to_key_.erase(it);
    
    LOGI("MMapPool: Removed mapping from cache (low access count)");
  } else {
#ifdef __ANDROID__
    // Advise kernel we might not need this soon
    madvise(entry.addr, entry.size, MADV_DONTNEED);
#endif
    LOGI("MMapPool: Kept mapping in cache for reuse");
  }
}

void MMapPool::evictLRU() {
  while (current_pool_size_ > max_pool_size_ * 0.75 && !lru_list_.empty()) {
    // Find least recently used inactive mapping
    for (auto it = lru_list_.begin(); it != lru_list_.end(); ++it) {
      auto& entry = mappings_[*it];
      if (!entry.in_use) {
        // Evict this one
#ifdef __ANDROID__
        if (entry.size < 8 * 1024 * 1024) {
          munlock(entry.addr, entry.size);
        }
#endif
        
        munmap(entry.addr, entry.size);
        
        current_pool_size_ -= entry.size;
        addr_to_key_.erase(entry.addr);
        mappings_.erase(*it);
        lru_map_.erase(*it);
        lru_list_.erase(it);
        
        LOGI("MMapPool: Evicted mapping to free space");
        break;
      }
    }
    
    // Safety check to prevent infinite loop
    if (lru_list_.empty()) break;
  }
}

bool MMapPool::shouldCache(const MMapEntry& entry) const {
  // Keep in cache if:
  // 1. Accessed more than once
  // 2. Small size (likely to be reused)
  // 3. Recently accessed
  
  if (entry.access_count > 1) return true;
  if (entry.size < 4 * 1024 * 1024) return true; // Less than 4MB
  
  auto now = std::chrono::steady_clock::now();
  auto age = std::chrono::duration_cast<std::chrono::milliseconds>(
    now - entry.last_access).count();
  
  if (age < 1000) return true; // Used within last second
  
  return false;
}

void MMapPool::clearOldMappings(int max_age_ms) {
  std::lock_guard<std::mutex> lock(mutex_);
  
  auto now = std::chrono::steady_clock::now();
  std::vector<std::string> to_remove;
  
  for (auto& [key, entry] : mappings_) {
    if (!entry.in_use) {
      auto age = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - entry.last_access).count();
      
      if (age > max_age_ms) {
        to_remove.push_back(key);
      }
    }
  }
  
  for (const auto& key : to_remove) {
    auto& entry = mappings_[key];
    
#ifdef __ANDROID__
    if (entry.size < 8 * 1024 * 1024) {
      munlock(entry.addr, entry.size);
    }
#endif
    
    munmap(entry.addr, entry.size);
    
    current_pool_size_ -= entry.size;
    addr_to_key_.erase(entry.addr);
    lru_list_.erase(lru_map_[key]);
    lru_map_.erase(key);
    mappings_.erase(key);
  }
  
  if (!to_remove.empty()) {
    LOGI("MMapPool: Cleared %zu old mappings", to_remove.size());
  }
}

MMapPool::Stats MMapPool::getStats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  
  Stats stats;
  stats.total_mappings = mappings_.size();
  stats.active_mappings = 0;
  stats.cached_mappings = 0;
  stats.total_size = current_pool_size_;
  stats.hit_count = hit_count_;
  stats.miss_count = miss_count_;
  
  for (const auto& [key, entry] : mappings_) {
    if (entry.in_use) {
      stats.active_mappings++;
    } else {
      stats.cached_mappings++;
    }
  }
  
  return stats;
}

void MMapPool::clearAll() {
  std::lock_guard<std::mutex> lock(mutex_);
  
  for (auto& [key, entry] : mappings_) {
#ifdef __ANDROID__
    if (entry.size < 8 * 1024 * 1024) {
      munlock(entry.addr, entry.size);
    }
#endif
    
    munmap(entry.addr, entry.size);
  }
  
  mappings_.clear();
  addr_to_key_.clear();
  lru_list_.clear();
  lru_map_.clear();
  current_pool_size_ = 0;
  
  LOGI("MMapPool: Cleared all mappings");
}

} // namespace nntrainer