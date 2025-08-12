// SPDX-License-Identifier: Apache-2.0
/**
 * @file   expert_prefetcher.h
 * @brief  Intelligent prefetching for MoE expert weights
 */

#ifndef __EXPERT_PREFETCHER_H__
#define __EXPERT_PREFETCHER_H__

#include <vector>
#include <thread>
#include <atomic>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <unordered_map>

namespace causallm {

/**
 * @class ExpertPrefetcher
 * @brief Prefetches expert weights based on predicted access patterns
 */
class ExpertPrefetcher {
public:
  struct PrefetchRequest {
    int expert_id;
    int layer_id;
    float priority;  // Higher priority = prefetch sooner
    void* tensor_ptr;
    size_t size;
  };

  /**
   * @brief Initialize prefetcher with number of experts and layers
   */
  ExpertPrefetcher(int num_experts, int num_layers);
  
  ~ExpertPrefetcher();

  /**
   * @brief Start prefetching thread
   */
  void start();

  /**
   * @brief Stop prefetching thread
   */
  void stop();

  /**
   * @brief Record expert access pattern for learning
   * @param layer_id Current layer
   * @param expert_ids Experts accessed in this layer
   */
  void recordAccess(int layer_id, const std::vector<int>& expert_ids);

  /**
   * @brief Predict next experts to be accessed
   * @param current_layer Current layer being processed
   * @param top_k Number of experts to predict
   * @return Vector of predicted expert IDs
   */
  std::vector<int> predictNextExperts(int current_layer, int top_k = 3);

  /**
   * @brief Schedule prefetch for specific experts
   * @param requests Prefetch requests to schedule
   */
  void schedulePrefetch(const std::vector<PrefetchRequest>& requests);

  /**
   * @brief Check if expert is already prefetched
   * @param layer_id Layer ID
   * @param expert_id Expert ID
   * @return True if already in memory
   */
  bool isPrefetched(int layer_id, int expert_id) const;

  /**
   * @brief Get prefetch hit rate statistics
   */
  struct Stats {
    size_t total_accesses;
    size_t prefetch_hits;
    size_t prefetch_misses;
    float hit_rate;
    float avg_prefetch_time_ms;
  };
  
  Stats getStats() const;

  /**
   * @brief Enable/disable prefetching
   */
  void setEnabled(bool enabled) { enabled_ = enabled; }

  /**
   * @brief Set maximum memory for prefetching
   */
  void setMaxMemory(size_t max_bytes) { max_prefetch_memory_ = max_bytes; }

private:
  // Prefetching worker thread
  void prefetchWorker();

  // Update access pattern history
  void updateHistory(int layer_id, const std::vector<int>& expert_ids);

  // Calculate expert affinity scores
  std::vector<float> calculateAffinityScores(int layer_id) const;

  // Evict least useful prefetched data
  void evictLRU();

private:
  int num_experts_;
  int num_layers_;
  bool enabled_;
  std::atomic<bool> running_;
  
  // Prefetch thread
  std::thread prefetch_thread_;
  std::mutex queue_mutex_;
  std::condition_variable queue_cv_;
  std::priority_queue<PrefetchRequest> prefetch_queue_;
  
  // Access pattern tracking
  std::unordered_map<int, std::vector<std::vector<int>>> access_history_;
  std::unordered_map<std::string, int> pattern_frequency_;
  
  // Prefetched data tracking
  struct PrefetchedData {
    void* addr;
    size_t size;
    std::chrono::steady_clock::time_point timestamp;
    int access_count;
  };
  std::unordered_map<std::string, PrefetchedData> prefetched_;
  mutable std::mutex prefetch_mutex_;
  
  // Memory management
  size_t max_prefetch_memory_;
  std::atomic<size_t> current_prefetch_memory_;
  
  // Statistics
  mutable std::atomic<size_t> total_accesses_;
  mutable std::atomic<size_t> prefetch_hits_;
  mutable std::atomic<size_t> prefetch_misses_;
  std::chrono::milliseconds total_prefetch_time_;
};

} // namespace causallm

#endif // __EXPERT_PREFETCHER_H__