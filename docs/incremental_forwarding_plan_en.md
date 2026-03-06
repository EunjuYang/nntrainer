# Incremental Forwarding Refactoring Plan

## 1. Problem Analysis of the Current Implementation

### 1.1 Current Architecture Summary

The current incremental forwarding passes a **single `(from, to)` pair** identically to all layers through the call chain `NeuralNetwork` -> `NetworkGraph` -> `LayerNode` -> `Layer`:

```
NeuralNetwork::incremental_forwarding(from, to, ...)
  └─ NetworkGraph::incremental_forwarding(from, to, ...)
       └─ for (layer : all_layers)
            forwarding_op(layer, training)   // from, to are fixed at lambda capture time
              └─ node->incremental_forwarding(from, to, training)
                   └─ layer->incremental_forwarding(context, from, to, training)
```

Core problem: `from` and `to` are **fixed at capture time** in the `forwarding_op` lambda, so every layer receives the same values.

```cpp
// neuralnet.cpp:449-461
std::function<void(std::shared_ptr<LayerNode>, bool)> forwarding_op =
  [this, from, to, ...](std::shared_ptr<LayerNode> node, bool training) -> void {
    node->incremental_forwarding(from, to, training);  // same from, to for all layers
  };
```

**Scope of impact:** A total of **39 files** implement `incremental_forwarding`:
- Core layers (nntrainer/layers/): embedding, fc, attention, multi_head_attention, addition, concat, layer_normalization, multiout, etc. — 9 files
- CL layers (nntrainer/layers/cl_layers/): addition_cl, concat_cl, fc_cl, reshape_cl, rmsnorm_cl, swiglu_cl, transpose_cl, etc. — 7 files
- CausalLM app (Applications/CausalLM/layers/): mha_core, qkv_layer, rms_norm, swiglu, lm_head, embedding_layer, embedding_normalize_layer, embedding_pooling_layer, reshaped_rms_norm, tie_word_embedding, etc. — 10 files
- CausalLM models: gpt_oss_moe_layer_cached, gpt_oss_moe_layer, qwen_moe_layer_cached, qwen_moe_layer, qwen_moe_layer_fsu, etc. — 5 files
- LLaMA app: custom_multi_head_attention_layer, rms_norm, rotary_embedding, swiglu, etc. — 4 files
- Graph/model: network_graph, neuralnet, layer_node, etc. — 3 files
- Tests: layers_golden_tests — 1 file

---

### 1.2 Problem 1: Cannot Handle Different from/to Per Batch

**Symptom:** When there are multiple batches in the input, each batch may be at a different sequence position (e.g., batch 0 is at step 5, batch 1 is at step 3).

**Limitations of the current code:**

```cpp
// neuralnet.cpp:481-496 - only validates input tensor batch size, no per-batch from/to
sharedConstTensors
NeuralNetwork::incremental_forwarding(unsigned int from, unsigned int to,
                                      sharedConstTensors input, ...) {
  auto current_batch = model_graph.getBatchSize();
  NNTR_THROW_IF(input[0]->batch() != current_batch, ...);
  // from, to are scalars → applied identically to all batches
}
```

Even inside each layer, the batch loop uses the same from/to:

```cpp
// fc_layer.cpp - same from, to for slicing within the batch loop
for (unsigned int b = 0; b < hidden_.batch(); ++b) {
  // input_step_dim.height(to - from);  ← same for all batches
  Tensor input_step = input_.getSharedDataTensor(input_step_dim, ...);
}
```

**Concrete example:**
- Batch 0: already processed 10 tokens, needs to process the 11th token → `from=10, to=11`
- Batch 1: already processed 5 tokens, needs to process the 6th token → `from=5, to=6`
- The current architecture can only pass a single `(from, to)`, making it impossible to handle different sequence positions across batches

**Tensor memory layout issue:** When per-batch from/to differ, a single contiguous slice cannot be obtained via `getSharedDataTensor()`. Batch 0 needs to write at offset 10 while batch 1 needs to write at offset 5, requiring separate slices per batch. The current pattern of applying `step_dim.height(to - from)` uniformly across all batches breaks down.

---

### 1.3 Problem 2: Cannot Propagate from/to Between Layers

**Symptom:** After a preceding layer transforms/consumes from/to, there is no mechanism to pass the modified from/to to downstream layers.

**Limitation of the current code — Embedding:**

```cpp
// embedding.cpp:123-128 - internally resets to from=0, to=1
if (from) {
  NNTR_THROW_IF(to - from != 1, std::invalid_argument);
  from = 0;  // only modifies local variable, not propagated to the next layer
  to = 1;
}
```

The Embedding layer receives `from=10, to=11` and internally processes it as `from=0, to=1`, but:
- The valid range of the output tensor is `[0, 1)` (tensor with height 1)
- The next FC layer still receives the original `from=10, to=11`
- The FC layer tries to read data at height positions 10~11 using these values → **mismatch**

**Limitation of the current code — MultiOut:**

```cpp
// multiout_layer.cpp:42-71 - similarly resets to from=0, to=1
void MultiOutLayer::incremental_forwarding(RunLayerContext &context,
                                           unsigned int from, unsigned int to,
                                           bool training) {
  if (from) {
    NNTR_THROW_IF(to - from != 1, std::invalid_argument);
    from = 0;  // local reset, not propagated to downstream layers
    to = 1;
  }
  // ... comment notes this only works with batch size 1
  // @todo: set reset stride as false. This implementation only works when
  // batch size is 1
}
```

**Limitation of the current code — MHACoreLayer (CausalLM app):**

```cpp
// Applications/CausalLM/layers/mha_core.cpp:209-224
unsigned int from = _from;
unsigned int to = _to;

if (to >= max_timestep) {
  if (!_from) {
    throw std::invalid_argument("...");
  } else {
    // KV cache overflow → shift
    cache_shift = true;
    from = max_timestep - 1;  // only modifies local variable
    to = max_timestep;
  }
}
```

MHACoreLayer modifies from/to on KV cache overflow, but this information is not propagated to subsequent layers.

**Asymmetric usage in the Attention layer:**

```cpp
// attention_layer.cpp - Query uses (to-from), Key/Value use (to)
query_step_dim.height(to - from);   // only the newly input portion
value_step_dim.height(to);          // entire accumulated range (KV cache)
key_step_dim.height(to);
```

The Attention layer **interprets from/to differently** for Query vs Key/Value inputs.
This is a case that cannot be correctly represented with a single (from, to).

---

### 1.4 Problem 3: Different from/to for Multi-Input Layers

**Symptom:** When a single layer receives multiple inputs, the valid range (from/to) of each input may differ.

**Concrete example — Multi-Head Attention:**
- Input 0 (Query): `from=10, to=11` (1 new token)
- Input 1 (Key): `from=0, to=11` (entire range including KV cache)
- Input 2 (Value): `from=0, to=11` (entire range including KV cache)

Currently, a single `(from, to)` is passed, so the MHA layer hardcodes the distinction internally:

```cpp
// multi_head_attention_layer.cpp:637-642
projected_query_step_dim.height(to - from);   // hardcoded: only new portion for Query
projected_key_step_dim.height(to);             // hardcoded: entire range for Key
projected_value_step_dim.height(to);           // hardcoded: entire range for Value
```

This hardcoding breaks if the input order of MHA changes or if other types of multi-input layers are added.

**Concrete example — Addition layer (residual connection):**
- Input 0 (skip connection): `from=10, to=11` (corresponding position from the original input)
- Input 1 (MHA output): `from=0, to=1` (range reset by MHA)
- Currently: both inputs receive the same from=10, to=11, causing incorrect offset when accessing the MHA output

---

## 2. Proposed Solution

### 2.1 Core Idea: Propagate Per-Input, Per-Batch `from/to` as Context Metadata

Manage from/to as **metadata attached to RunLayerContext** rather than function arguments, allowing each tensor (= each connection) to carry independent from/to information.

### 2.2 New Data Structure: `IncrementalInfo`

```cpp
/**
 * @file   incremental_info.h
 * @brief  Per-tensor incremental forwarding range information
 *
 * Supports different from/to per batch for each input/output tensor
 */
struct IncrementalInfo {
  // per-batch from/to vectors
  std::vector<unsigned int> from;  // size == batch_size
  std::vector<unsigned int> to;    // size == batch_size

  /** @brief Default constructor: empty state (non-incremental) */
  IncrementalInfo() = default;

  /** @brief Check if all batches have the same from/to */
  bool is_uniform() const {
    if (from.empty()) return true;
    for (size_t i = 1; i < from.size(); ++i)
      if (from[i] != from[0] || to[i] != to[0]) return false;
    return true;
  }

  /** @brief Check if valid incremental info exists */
  bool isValid() const { return !from.empty(); }

  /** @brief Get from for a specific batch (ignores batch if uniform) */
  unsigned int getFrom(unsigned int batch = 0) const {
    return from.size() == 1 ? from[0] : from[batch];
  }

  /** @brief Get to for a specific batch */
  unsigned int getTo(unsigned int batch = 0) const {
    return to.size() == 1 ? to[0] : to[batch];
  }

  /** @brief Get step size (to - from) for a specific batch */
  unsigned int getStepSize(unsigned int batch = 0) const {
    return getTo(batch) - getFrom(batch);
  }

  /** @brief Only callable when step size is uniform */
  unsigned int getUniformFrom() const { return from[0]; }
  unsigned int getUniformTo() const { return to[0]; }

  /** @brief Create with a single (from, to) — backward compatible */
  static IncrementalInfo uniform(unsigned int f, unsigned int t,
                                  unsigned int batch_size = 1) {
    IncrementalInfo info;
    info.from.assign(batch_size, f);
    info.to.assign(batch_size, t);
    return info;
  }

  /** @brief Create with per-batch from/to vectors */
  static IncrementalInfo perBatch(std::vector<unsigned int> f,
                                   std::vector<unsigned int> t) {
    IncrementalInfo info;
    info.from = std::move(f);
    info.to = std::move(t);
    return info;
  }
};
```

### 2.3 Interface Changes

#### 2.3.1 Layer Interface (layer_devel.h)

```cpp
// Existing (kept as deprecated)
virtual void incremental_forwarding(RunLayerContext &context,
                                    unsigned int from, unsigned int to,
                                    bool training) {
  forwarding(context, training);
}

// New addition — default implementation delegates to existing signature (backward compat)
virtual void incremental_forwarding(RunLayerContext &context,
                                    bool training) {
  // If an existing layer only overrode (context, from, to, training),
  // this default implementation ensures existing code works as-is
  auto &info = context.getInputIncrementalInfo(0);
  if (info.isValid() && info.is_uniform()) {
    incremental_forwarding(context, info.getUniformFrom(),
                           info.getUniformTo(), training);
  } else {
    forwarding(context, training);
  }
  // Default behavior: propagate input IncrementalInfo to outputs as-is
  for (unsigned int i = 0; i < context.getNumOutputs(); ++i) {
    if (context.getNumInputs() > 0) {
      context.setOutputIncrementalInfo(i, context.getInputIncrementalInfo(0));
    }
  }
}
```

**Key point:** The new signature's default implementation calls the existing signature, so layers that only overrode `(context, from, to, training)` work without modification.

#### 2.3.2 RunLayerContext Extension (layer_context.h)

```cpp
class RunLayerContext {
  // Added to existing members:
private:
  std::vector<IncrementalInfo> input_incremental_info;   // per-input
  std::vector<IncrementalInfo> output_incremental_info;  // per-output

public:
  /// Get incremental range for an input tensor
  const IncrementalInfo& getInputIncrementalInfo(unsigned int idx) const;

  /// Set incremental range for an input tensor (called during graph propagation)
  void setInputIncrementalInfo(unsigned int idx, const IncrementalInfo &info);

  /// Set incremental range for an output tensor (called by the layer during forwarding)
  void setOutputIncrementalInfo(unsigned int idx, const IncrementalInfo &info);

  /// Get incremental range for an output tensor (propagated as input to the next layer)
  const IncrementalInfo& getOutputIncrementalInfo(unsigned int idx) const;

  /// Initialize IncrementalInfo vector sizes (at finalizeContext time)
  void initIncrementalInfo(unsigned int num_inputs, unsigned int num_outputs);

  /// Reset all IncrementalInfo (at the start of each forwarding pass)
  void resetIncrementalInfo();
};
```

#### 2.3.3 LayerNode Changes (layer_node.h/cpp)

```cpp
// Existing retained (backward compat wrapper)
void incremental_forwarding(unsigned int from, unsigned int to,
                            bool training = true);

// New addition
void incremental_forwarding(bool training = true);
```

```cpp
// layer_node.cpp — new implementation
void LayerNode::incremental_forwarding(bool training) {
  loss->set(run_context->getRegularizationLoss());
  PROFILE_TIME_START(forward_event_key);
  layer->incremental_forwarding(*run_context, training);
  PROFILE_TIME_END(forward_event_key);
  // ... (same validation/loss logic as existing)
}

// Existing implementation → converted to wrapper
void LayerNode::incremental_forwarding(unsigned int from, unsigned int to,
                                       bool training) {
  // Set uniform IncrementalInfo for all inputs
  for (unsigned int i = 0; i < run_context->getNumInputs(); ++i) {
    run_context->setInputIncrementalInfo(
      i, IncrementalInfo::uniform(from, to));
  }
  incremental_forwarding(training);
}
```

#### 2.3.4 NetworkGraph Changes (network_graph.h/cpp)

```cpp
// Existing retained (backward compat wrapper)
sharedConstTensors incremental_forwarding(
  unsigned int from, unsigned int to, bool training, ...);

// New addition
sharedConstTensors incremental_forwarding(
  bool training,
  std::function<void(std::shared_ptr<LayerNode>, bool)> forwarding_op = ...,
  std::function<bool(void *userdata)> stop_cb = ...,
  void *user_data = nullptr);

// New — IncrementalInfo propagation method
void propagateIncrementalInfo(const std::shared_ptr<LayerNode> &node);
```

#### 2.3.5 NeuralNetwork Changes (neuralnet.h/cpp)

```cpp
// Existing retained (backward compat wrapper)
sharedConstTensors incremental_forwarding(unsigned int from, unsigned int to, ...);
sharedConstTensors incremental_inference(sharedConstTensors X, ...,
                                          unsigned int from, unsigned int to);

// New addition — IncrementalInfo-based
sharedConstTensors incremental_forwarding(
  const std::vector<IncrementalInfo> &input_incremental_info,
  bool training = true, ...);
sharedConstTensors incremental_inference(
  sharedConstTensors X,
  const std::vector<IncrementalInfo> &input_incremental_info, ...);
```

---

### 2.4 IncrementalInfo Propagation Mechanism

#### Core Execution Flow:

```
1. NeuralNetwork sets IncrementalInfo for input layers
     ↓
2. NetworkGraph traverses layers in topological order:
   for (layer : layers) {
     // a) Copy previous layer's output IncrementalInfo to current layer's input IncrementalInfo
     propagateIncrementalInfo(layer);

     // b) Execute layer (layer queries from/to through context internally)
     layer->incremental_forwarding(training);
     //   - Query each input's range via context.getInputIncrementalInfo(i)
     //   - Perform computation
     //   - Set output range via context.setOutputIncrementalInfo(i, ...)
   }
```

#### Propagation Rule Implementation (network_graph.cpp):

```cpp
void NetworkGraph::propagateIncrementalInfo(
    const std::shared_ptr<LayerNode> &node) {
  auto &run_context = node->getRunContext();

  // Input layers (first layers of the network) are already set by NeuralNetwork
  if (node->getNumInputConnections() == 0)
    return;

  // For each input connection of the node:
  for (unsigned int i = 0; i < node->getNumInputConnections(); ++i) {
    // Find the previous node's output index using graph connection info
    auto [prev_node_idx, prev_output_idx] = getInputConnection(node, i);
    auto &prev_node = getLayerNode(prev_node_idx);
    auto &prev_context = prev_node->getRunContext();

    // Copy previous node's output_idx-th output IncrementalInfo →
    // current node's i-th input IncrementalInfo
    run_context.setInputIncrementalInfo(
      i, prev_context.getOutputIncrementalInfo(prev_output_idx));
  }
}
```

#### Propagation Flow Example — Transformer Decoder Block:

```
[Input Token IDs]
  IncrementalInfo: from=10, to=11 (batch 0)

  ↓ Embedding Layer
  Input info: from=10, to=11
  Internal processing: embed token_id[10] → output height=1
  Output info: from=0, to=1  ← layer transforms the range

  ↓ propagateIncrementalInfo()
  ↓ (Embedding's output info copied to FC's input info)

  ↓ FC Layer (query projection)
  Input info: from=0, to=1  ← receives correct range
  Internal processing: matmul on height [0,1) slice
  Output info: from=0, to=1  ← propagated as-is

  ↓ MultiOut Layer (branches query to MHA and residual)
  Input info: from=0, to=1
  Output[0] info: from=0, to=1  (→ MHA's query input)
  Output[1] info: from=0, to=1  (→ residual add input)

  ↓ Multi-Head Attention
  Input[0] (query) info:  from=0, to=1   ← new token
  Input[1] (key) info:    from=0, to=1   ← (info transformed by layer with KV cache)
  Input[2] (value) info:  from=0, to=1
  Internal: add new K,V to KV cache, cache range is [0, 11)
            attention computation: Q[0:1] x K[0:11]^T → V[0:11]
  Output info: from=0, to=1  ← based on query

  ↓ Addition Layer (residual)
  Input[0] info: from=0, to=1  (MHA output)
  Input[1] info: from=0, to=1  (skip connection)  ← now matches!
  Output info: from=0, to=1
```

---

### 2.5 Per-Layer Changes

#### 2.5.1 Default Behavior (layer_devel.h default implementation)

As described in 2.3.1. Layers that only override `(context, from, to, training)` work without modification.

#### 2.5.2 Layers Requiring Migration

Layers that need migration to the new interface are those that **transform from/to internally**, **require different ranges per input**, or **need per-batch processing**:

| Priority | Layer | Reason for Migration | Key Change |
|----------|-------|---------------------|------------|
| **P0** | `MultiHeadAttentionLayer` | Requires different from/to per input (Q vs K/V) | Query each input independently via `getInputIncrementalInfo(0/1/2)` |
| **P0** | `AttentionLayer` | Requires different from/to per input | Same |
| **P0** | `EmbeddingLayer` | Transforms from/to for output | Set transformed range via `setOutputIncrementalInfo` |
| **P0** | `MultiOutLayer` | Transforms from/to + remove batch=1 restriction | Propagate input info to all outputs |
| **P1** | `FullyConnectedLayer` | Support per-batch from/to | Use `info.getFrom(b)` in batch loop |
| **P1** | `AdditionLayer` | Validate info consistency across multiple inputs | Query per-input info then element-wise add |
| **P1** | `ConcatLayer` | Axis-dependent from/to summation logic | Compute output info based on concat axis |
| **P1** | `LayerNormalizationLayer` | Per-batch support | Query range from info |
| **P2** | CausalLM `MHACoreLayer` | Per-batch + cache shift propagation | Most complex transformation logic |
| **P2** | CausalLM remaining layers | Per-batch support | Switch to info-based query |
| **P2** | CL layers | Same | Same |
| **P2** | LLaMA layers | Same | Same |
| **P3** | Layers that only override existing signature | No change needed — works via default implementation | No change (incremental migration later) |

#### 2.5.3 Detailed Migration Example — Embedding

```cpp
void EmbeddingLayer::incremental_forwarding(RunLayerContext &context,
                                            bool training) {
  auto &info = context.getInputIncrementalInfo(0);
  const Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  Tensor &output_ = context.getOutput(SINGLE_INOUT_IDX);

  unsigned int batch_size = input_.batch();

  for (unsigned int b = 0; b < batch_size; ++b) {
    unsigned int b_from = info.getFrom(b);
    unsigned int b_to = info.getTo(b);
    unsigned int step_size = b_to - b_from;

    NNTR_THROW_IF(b_from && step_size != 1, std::invalid_argument)
      << "incremental step size is not 1";

    unsigned int actual_from = b_from ? 0 : b_from;
    unsigned int actual_to = b_from ? 1 : b_to;

    // Per-batch tensor slicing & embedding lookup
    for (unsigned int i = actual_from; i < actual_to; ++i) {
      // ... existing embedding logic (for batch b)
    }
  }

  // Set output range: embedding resets from
  if (info.is_uniform()) {
    unsigned int step = info.getStepSize(0);
    unsigned int out_from = info.getFrom(0) ? 0 : info.getFrom(0);
    unsigned int out_to = info.getFrom(0) ? step : info.getTo(0);
    context.setOutputIncrementalInfo(0,
      IncrementalInfo::uniform(out_from, out_to, batch_size));
  } else {
    // Set per-batch output range
    std::vector<unsigned int> out_from(batch_size), out_to(batch_size);
    for (unsigned int b = 0; b < batch_size; ++b) {
      out_from[b] = info.getFrom(b) ? 0 : info.getFrom(b);
      out_to[b] = info.getFrom(b) ? info.getStepSize(b) : info.getTo(b);
    }
    context.setOutputIncrementalInfo(0,
      IncrementalInfo::perBatch(out_from, out_to));
  }
}
```

#### 2.5.4 Detailed Migration Example — Multi-Head Attention

```cpp
void MultiHeadAttentionLayer::incremental_forwarding(
    RunLayerContext &context, bool training) {
  // Query IncrementalInfo independently for each input
  auto &query_info = context.getInputIncrementalInfo(0);
  auto &key_info   = context.getInputIncrementalInfo(1);
  auto &value_info = context.getInputIncrementalInfo(2);

  // Use each input's actual range without hardcoding
  for (unsigned int b = 0; b < batch_size; ++b) {
    unsigned int q_from = query_info.getFrom(b);
    unsigned int q_to   = query_info.getTo(b);
    unsigned int k_from = key_info.getFrom(b);
    unsigned int k_to   = key_info.getTo(b);
    unsigned int v_from = value_info.getFrom(b);
    unsigned int v_to   = value_info.getTo(b);

    // Query projection: q_from ~ q_to range
    // Key/Value projection + cache: k_from ~ k_to range
    // Attention: Q[q_from:q_to] x K[0:k_to]^T → O[q_from:q_to]
    // ...
  }

  // Output range = Query's range (attention output has the same sequence length as query)
  context.setOutputIncrementalInfo(0, query_info);
}
```

---

### 2.6 Tensor Slicing Strategy for Per-Batch from/to

Changes to tensor memory access patterns when per-batch from/to differ:

#### Current Pattern (uniform from/to):
```cpp
// Apply the same step_dim to all batches
TensorDim step_dim = dim;
step_dim.height(to - from);
for (unsigned int b = 0; b < batch_size; ++b) {
  Tensor step = tensor.getSharedDataTensor(step_dim,
    b * dim.getFeatureLen() + from * dim.width());
  // ... compute on step
}
```

#### New Pattern (per-batch from/to):
```cpp
for (unsigned int b = 0; b < batch_size; ++b) {
  unsigned int b_from = info.getFrom(b);
  unsigned int b_to = info.getTo(b);

  TensorDim step_dim = dim;
  step_dim.batch(1);
  step_dim.height(b_to - b_from);

  Tensor step = tensor.getSharedDataTensor(step_dim,
    b * dim.getFeatureLen() + b_from * dim.width(), true);
  // ... compute on step
}
```

**Note:** When step_size differs across batches in per-batch mode, batched BLAS operations cannot be used directly, requiring fallback to per-batch individual operations. Performance optimizations include:
- Check `is_uniform()` first, and use the existing pattern for uniform cases
- Only use the per-batch loop for non-uniform cases

---

### 2.7 Backward Compatibility Strategy

#### Core Principle: Existing code must compile and work without modification

1. **Retain existing signatures (marked as deprecated)**
   ```cpp
   // layer_devel.h - retain existing virtual method
   [[deprecated("Use incremental_forwarding(context, training) instead")]]
   virtual void incremental_forwarding(RunLayerContext &context,
                                       unsigned int from, unsigned int to,
                                       bool training) {
     forwarding(context, training);
   }
   ```

2. **New signature's default implementation delegates to existing signature**
   - Layers that overrode `(context, from, to, training)` work without modification
   - The new interface's default extracts from/to from context and calls the existing method

3. **Retain existing public APIs for NeuralNetwork and NetworkGraph**
   - Existing `incremental_forwarding(from, to, ...)` → internally converts to `IncrementalInfo::uniform`
   - New overloads are added alongside

---

### 2.8 Phased Implementation Plan

#### Phase 1: Add Infrastructure (maintain backward compatibility, no changes to existing behavior)
1. Define `IncrementalInfo` struct (`incremental_info.h`)
2. Add IncrementalInfo storage/query API to `RunLayerContext`
3. Add new virtual method to `layer_devel.h` (retain existing method)
4. Add new `incremental_forwarding(bool)` to `LayerNode`
5. Implement `propagateIncrementalInfo()` in `NetworkGraph`
6. Add `NetworkGraph::incremental_forwarding(bool, ...)` overload
7. Add IncrementalInfo-based overloads to `NeuralNetwork`

**At this point, existing code works without modification. The new interface is also available.**

#### Phase 2: Core Layer Migration
1. `EmbeddingLayer` → new interface (from/to transformation + output info setting)
2. `MultiOutLayer` → new interface (batch>1 support)
3. `MultiHeadAttentionLayer` → new interface (per-input info)
4. `AttentionLayer` → new interface (per-input info)
5. `FullyConnectedLayer` → new interface (per-batch support)
6. `AdditionLayer`, `ConcatLayer`, `LayerNormalizationLayer`

#### Phase 3: Application Layer Migration
1. CausalLM layers (MHACoreLayer, etc.)
2. LLaMA layers
3. CL layers

#### Phase 4: Cleanup
1. Add `[[deprecated]]` to existing signatures
2. Update tests (layers_golden_tests.cpp)
3. Update documentation

---

### 2.9 Files to Modify

#### Phase 1 (Infrastructure)
| File | Change Description |
|------|-------------------|
| `nntrainer/layers/incremental_info.h` (new) | Define `IncrementalInfo` struct |
| `nntrainer/layers/layer_context.h` | Add IncrementalInfo members/methods to `RunLayerContext` |
| `nntrainer/layers/layer_context.cpp` | Implement IncrementalInfo-related logic |
| `nntrainer/layers/layer_devel.h` | Add new virtual `incremental_forwarding(context, training)` |
| `nntrainer/layers/layer_node.h` | Declare new `incremental_forwarding(training)` |
| `nntrainer/layers/layer_node.cpp` | Implement new method, convert existing method to wrapper |
| `nntrainer/graph/network_graph.h` | Declare `propagateIncrementalInfo()`, new `incremental_forwarding()` |
| `nntrainer/graph/network_graph.cpp` | Implement propagation logic + new forwarding loop |
| `nntrainer/models/neuralnet.h` | Declare new overloads |
| `nntrainer/models/neuralnet.cpp` | Implement new overloads, convert existing methods to wrappers |

#### Phase 2 (Core Layers)
| File | Change Description |
|------|-------------------|
| `nntrainer/layers/embedding.cpp` | Migrate to new interface |
| `nntrainer/layers/multiout_layer.cpp` | New interface + batch>1 support |
| `nntrainer/layers/multi_head_attention_layer.cpp` | Use per-input info |
| `nntrainer/layers/attention_layer.cpp` | Use per-input info |
| `nntrainer/layers/fc_layer.cpp` | Use per-batch info |
| `nntrainer/layers/addition_layer.cpp` | Validate per-input info |
| `nntrainer/layers/concat_layer.cpp` | Axis-dependent info summation |
| `nntrainer/layers/layer_normalization_layer.cpp` | Info-based range query |

#### Phase 3 (Application Layers)
| File | Change Description |
|------|-------------------|
| `Applications/CausalLM/layers/*.cpp` (10 files) | Migrate to new interface |
| `Applications/CausalLM/models/*.cpp` (5 files) | Migrate to new interface |
| `Applications/LLaMA/jni/*.cpp` (4 files) | Migrate to new interface |
| `nntrainer/layers/cl_layers/*.cpp` (7 files) | Migrate to new interface |

#### Phase 4 (Cleanup)
| File | Change Description |
|------|-------------------|
| `test/unittest/layers/layers_golden_tests.cpp` | Add tests for new interface |
| `nntrainer/tensor/cpu_backend/fallback/fallback_internal.cpp` | Update if needed |

---

## 3. Summary

### Root Cause of the Current Problem
**A single scalar `(from, to)` is passed as function arguments and applied identically to all layers, all batches, and all inputs**

### Core Solution
**Manage `IncrementalInfo` as RunLayerContext metadata to support per-input, per-batch ranges with automatic propagation during graph traversal**

### Resolution for the 3 Problems:

| Problem | Solution |
|---------|----------|
| Different from/to per batch | `IncrementalInfo.from/to` are `vector<uint>` (per-batch) |
| from/to propagation between layers | `propagateIncrementalInfo()` automatically propagates output→input; layers set transformed ranges via `setOutputIncrementalInfo()` |
| Different from/to for multiple inputs | Independent query per input via `getInputIncrementalInfo(idx)` |

### Design Principles:
1. **Backward compatible:** Existing code compiles and runs without modification
2. **Incremental migration:** Each phase can be deployed independently
3. **Layer autonomy:** Each layer is responsible for interpreting input info and setting output info
4. **Performance preservation:** Maintains identical performance path for uniform cases
