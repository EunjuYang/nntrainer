// NNTrainer Main Thread Execution Patch
// This patch modifies NNTrainer to run entirely in the main thread without creating worker threads
// Suitable for WebAssembly compilation and single-threaded environments

// ============================================================================
// File: nntrainer/dataset/databuffer.h
// Add these members to the DataBuffer class private section (around line 165):
// ============================================================================
/*
private:
  // ... existing members ...
  
  // Main thread execution state - add these new members:
  DataProducer::Generator main_thread_generator; // generator for main thread
  unsigned int main_thread_size = 0; // size of dataset
  unsigned int main_thread_current_idx = 0; // current index
  bool main_thread_shuffle = false; // whether to shuffle
  std::vector<unsigned int> main_thread_idxes; // shuffled indices
  std::shared_ptr<IterationQueue> main_thread_iq; // iteration queue
*/

// ============================================================================
// File: nntrainer/dataset/databuffer.cpp
// Replace the startFetchWorker method (starting around line 71):
// ============================================================================

#include <databuffer.h>
#include <algorithm>
#include <numeric>

std::future<std::shared_ptr<IterationQueue>>
DataBuffer::startFetchWorker(const std::vector<TensorDim> &input_dims,
                             const std::vector<TensorDim> &label_dims,
                             bool shuffle) {
  NNTR_THROW_IF(!producer, std::runtime_error) << "producer does not exist";
  NNTR_THROW_IF(input_dims.empty(), std::runtime_error)
    << "There must be at least one input";

  auto q_size = std::get<PropsBufferSize>(*db_props);
  auto iq = std::make_shared<IterationQueue>(q_size, input_dims, label_dims);
  auto generator = producer->finalize(input_dims, label_dims);
  auto size = producer->size(input_dims, label_dims);
  iq_view = iq;

  // Store generator and state for on-demand generation in main thread
  main_thread_generator = generator;
  main_thread_size = size;
  main_thread_current_idx = 0;
  main_thread_shuffle = shuffle;
  main_thread_iq = iq;
  
  if (shuffle == true && size != DataProducer::SIZE_UNDEFINED) {
    main_thread_idxes.resize(size);
    std::iota(main_thread_idxes.begin(), main_thread_idxes.end(), 0);
    std::shuffle(main_thread_idxes.begin(), main_thread_idxes.end(), rng);
  }

  // Pre-fill the first iteration to make it available immediately
  fillOneIteration();

  // Return immediately fulfilled future (no thread creation)
  std::promise<std::shared_ptr<IterationQueue>> promise;
  promise.set_value(iq);
  return promise.get_future();
}

// ============================================================================
// Add this new helper method to DataBuffer class:
// ============================================================================

void DataBuffer::fillOneIteration() {
  if (!main_thread_generator || !main_thread_iq) {
    return;
  }
  
  // Check if we've reached the end
  if (main_thread_size != DataProducer::SIZE_UNDEFINED && 
      main_thread_current_idx >= main_thread_size) {
    // Signal end of data
    main_thread_iq->notifyEndOfRequestEmpty();
    return;
  }
  
  // Get the batch size from the iteration queue
  unsigned int batch_size = main_thread_iq->getBatchSize();
  
  // Fill one iteration worth of samples
  for (unsigned int s = 0; s < batch_size; ++s) {
    if (main_thread_size != DataProducer::SIZE_UNDEFINED && 
        main_thread_current_idx >= main_thread_size) {
      // Partial batch or end of data
      if (s == 0) {
        main_thread_iq->notifyEndOfRequestEmpty();
      }
      break;
    }
    
    auto sample_view = main_thread_iq->requestEmptySlotNonBlocking();
    if (sample_view.isEmpty()) {
      break;
    }
    
    auto &sample = sample_view.get();
    try {
      unsigned int idx = main_thread_current_idx;
      if (main_thread_shuffle && !main_thread_idxes.empty()) {
        idx = main_thread_idxes[main_thread_current_idx];
      }
      
      bool last = false;
      if (main_thread_size == DataProducer::SIZE_UNDEFINED) {
        // Generator mode - check for last sample
        last = main_thread_generator(main_thread_current_idx, 
                                    sample.getInputsRef(), 
                                    sample.getLabelsRef());
      } else {
        // Fixed size mode
        main_thread_generator(idx, sample.getInputsRef(), sample.getLabelsRef());
      }
      
      main_thread_current_idx++;
      
      if (last) {
        main_thread_iq->notifyEndOfRequestEmpty();
        break;
      }
    } catch (std::exception &e) {
      ml_loge("Fetching sample failed, Error: %s", e.what());
      throw;
    }
  }
}

// ============================================================================
// Replace the fetch method (around line 155):
// ============================================================================

ScopedView<Iteration> DataBuffer::fetch() {
  NNTR_THROW_IF(!producer, std::runtime_error) << "producer does not exist";
  auto iq = iq_view.lock();
  NNTR_THROW_IF(!iq, std::runtime_error)
    << "Cannot fetch, either fetcher is not running or fetcher has ended and "
       "invalidated";
  
  // Try to get a filled iteration
  auto result = iq->requestFilledSlotNonBlocking();
  
  // If no filled iteration is available and we haven't reached the end,
  // generate one iteration on-demand
  if (result.isEmpty() && main_thread_generator) {
    fillOneIteration();
    result = iq->requestFilledSlotNonBlocking();
  }
  
  return result;
}

// ============================================================================
// File: nntrainer/dataset/iteration_queue.h
// Add these methods to IterationQueue class (around line 240):
// ============================================================================
/*
public:
  // Add non-blocking version of requestEmptySlot
  ScopedView<Sample> requestEmptySlotNonBlocking();
  
  // Add non-blocking version of requestFilledSlot  
  ScopedView<Iteration> requestFilledSlotNonBlocking();
  
  // Getter for batch size
  unsigned int getBatchSize() const { return batch_size; }
  
  // Make batch_size accessible
  // Move from private to public or add the getter above
*/

// ============================================================================
// File: nntrainer/dataset/iteration_queue.h
// Add this method to ViewQueue template class (around line 72):
// ============================================================================
/*
template <typename T>
class ViewQueue {
public:
  // ... existing methods ...
  
  // Add non-blocking pop method:
  T* tryPop() {
    std::unique_lock<std::shared_mutex> lk(q_mutex);
    if (q.empty()) {
      return nullptr;
    }
    auto ptr = q.front();
    q.pop();
    return ptr;
  }
};
*/

// ============================================================================
// File: nntrainer/dataset/iteration_queue.cpp
// Add these non-blocking methods (around line 92):
// ============================================================================

ScopedView<Sample> IterationQueue::requestEmptySlotNonBlocking() {
  std::scoped_lock lg(empty_mutex);
  auto current_flow_state = flow_state.load();
  
  if (current_flow_state != FlowState::FLOW_STATE_OPEN) {
    return ScopedView<Sample>(nullptr);
  }

  if (being_filled == nullptr ||
      current_iterator + 1 == being_filled->get().end()) {
    // Try non-blocking pop
    being_filled = empty_q.tryPop();
    if (being_filled == nullptr) {
      // No empty slots available
      return ScopedView<Sample>(nullptr);
    }
    being_filled->reset();
    num_being_filled++;
    current_iterator = being_filled->get().begin();
  } else {
    current_iterator++;
  }

  auto view = ScopedView<Sample>(
    &(*current_iterator),
    [current_being_filed = this->being_filled] {
      current_being_filed->markSampleFilled();
    },
    [this, current_being_filled = this->being_filled] {
      std::unique_lock lg(empty_mutex);
      this->markEmpty(current_being_filled);
      num_being_filled--;
      notify_emptied_cv.notify_all();
    });
  return view;
}

ScopedView<Iteration> IterationQueue::requestFilledSlotNonBlocking() {
  std::scoped_lock lock(filled_mutex);

  if (flow_state.load() == FlowState::FLOW_STATE_STOPPED) {
    return ScopedView<Iteration>(nullptr);
  }

  // Try non-blocking pop
  auto iteration = filled_q.tryPop();
  if (iteration == nullptr) {
    if (flow_state.load() == FlowState::FLOW_STATE_STOP_REQUESTED) {
      // End of data
      auto stop_request_state = FlowState::FLOW_STATE_STOP_REQUESTED;
      bool exchange_result = flow_state.compare_exchange_strong(
        stop_request_state, FlowState::FLOW_STATE_STOPPED);
      if (!exchange_result) {
        ml_logw("Failed to transition to STOPPED state");
      }
    }
    return ScopedView<Iteration>(nullptr);
  }

  return ScopedView<Iteration>(
    &iteration->get(), [this, iteration] { markEmpty(iteration); },
    [this, iteration] {
      std::unique_lock lock(filled_mutex);
      flow_state.store(FlowState::FLOW_STATE_STOPPED);
      markEmpty(iteration);
    });
}

// ============================================================================
// USAGE NOTES:
// ============================================================================
/*
This patch modifies NNTrainer to run entirely in the main thread by:

1. Replacing async thread creation with immediate promise fulfillment
2. Implementing on-demand data generation in the fetch() method
3. Adding non-blocking versions of queue operations
4. Maintaining state for incremental data generation

To apply this patch:

1. Add the new member variables to DataBuffer class in databuffer.h
2. Replace the startFetchWorker method in databuffer.cpp
3. Replace the fetch method in databuffer.cpp
4. Add the fillOneIteration helper method to databuffer.cpp
5. Add tryPop() to ViewQueue template in iteration_queue.h
6. Add the non-blocking methods to IterationQueue class
7. Add the declarations to iteration_queue.h

The modified version will:
- Generate data on-demand when fetch() is called
- Never create worker threads
- Work correctly with WebAssembly
- Maintain the same API interface

Key changes:
- No std::async calls
- No blocking on condition variables
- Data is generated incrementally as needed
- State is maintained between fetch() calls
*/