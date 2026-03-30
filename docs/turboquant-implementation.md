# TurboQuant Implementation in nntrainer

## 1. Overview

This document describes the TurboQuant KV cache compression implementation in nntrainer,
based on the paper ["TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
(Zandieh et al., arXiv:2504.19874)](https://arxiv.org/abs/2504.19874).

TurboQuant is a data-oblivious, calibration-free KV cache quantization method for
decoder-only transformers. It compresses Key/Value cache vectors to 4-bit per coordinate
using a combination of:

1. **Per-head norm extraction** (norm preservation)
2. **Randomized Hadamard Transform (RHT)** for coordinate decorrelation
3. **Lloyd-Max optimal scalar codebook** for MSE-minimal quantization

This reduces KV cache memory by approximately **4x** compared to FP16 storage,
enabling longer context lengths on memory-constrained devices.

---

## 2. TurboQuant Paper Summary

### 2.1 Paper Information

- **Title**: TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate
- **Authors**: Amir Zandieh (Google Research), Majid Daliri (NYU), Majid Hadian (Google DeepMind), Vahab Mirrokni (Google Research)
- **Published**: ICLR 2026 (arXiv: April 2025)

### 2.2 Core Idea

In decoder-only transformers, KV cache stores one key/value vector per token per layer per head.
With long context windows, the memory footprint becomes dominant.
TurboQuant addresses this by compressing each KV vector to sub-byte precision while preserving
inner product (attention score) fidelity.

### 2.3 Paper Algorithm (Two-Stage)

The paper proposes a two-stage approach:

**Stage 1 - MSE-Optimal Quantization (Algorithm 1):**

Given an input vector `x ∈ R^d`:

1. **Norm extraction**: Compute `‖x‖₂`, store as metadata. Normalize: `x̂ = x / ‖x‖₂`
2. **Random rotation**: Apply Randomized Hadamard Transform: `y = (1/√d) · H · D · x̂`
   - `D` = diagonal sign matrix (deterministic random ±1 signs)
   - `H` = Walsh-Hadamard matrix (orthogonal, O(d log d) butterfly computation)
   - After rotation, each coordinate of the unit vector follows a **Beta((d-1)/2, (d-1)/2)** distribution on [-1, 1], regardless of the input direction
3. **Lloyd-Max scalar quantization**: Quantize each rotated coordinate independently using a precomputed optimal codebook matched to the Beta distribution
4. **Pack**: Store quantized indices in packed format (e.g., 2 elements per byte for 4-bit)

**Dequantization:**

1. Look up centroid values from codebook indices
2. Apply inverse rotation: `x̂_recon = D · (1/√d) · H · centroids`
3. Rescale by stored norm: `x_recon = ‖x‖₂ · x̂_recon`

**Stage 2 - QJL Residual Correction (Optional):**

Apply a 1-bit Quantized Johnson-Lindenstrauss (QJL) transform to the quantization residual
to produce an unbiased inner product estimator. However, community testing found that
**MSE-only (Stage 1 alone) performs better in practice** because QJL's increased variance
disrupts softmax top-1 ranking despite eliminating bias.

### 2.4 Key Paper Claims

| Property | Description |
|----------|-------------|
| **Data-oblivious** | No calibration data needed; codebook depends only on dimension `d` |
| **Near-optimal** | Within ~2.7x of information-theoretic lower bound |
| **Online** | Each vector quantized independently (no batch dependency) |
| **Bit-widths tested** | 2-bit, 3-bit, 4-bit per coordinate |
| **Recommended** | 3.5 bits for quality-neutral, 2.5 bits for marginal degradation |
| **Models tested** | Llama-2, Llama-3, Mistral, and others |
| **Baselines compared** | KIVI, KVQuant, QJL, RaBitQ, Product Quantization |

---

## 3. What Is Implemented

### 3.1 Implemented Features (v2 - Paper Algorithm 1)

The nntrainer implementation covers the **Stage 1** (MSE-optimal quantization) of the paper:

| Paper Component | Implementation Status | Location |
|----------------|----------------------|----------|
| Per-head norm extraction | **Implemented** | `turboquant_utils.h:150-154` |
| Unit vector normalization | **Implemented** | `turboquant_utils.h:157-159` |
| Randomized Hadamard Transform (RHT) | **Implemented** | `turboquant_utils.h:45-59, 72-77` |
| Deterministic sign generation (D matrix) | **Implemented** | `turboquant_utils.h:62-69` |
| Lloyd-Max codebook (4-bit, 16 levels) | **Implemented** | `turboquant_utils.h:90-133` |
| Codebook for d=64 (Beta(31.5, 31.5)) | **Implemented** | `turboquant_utils.h:96-105` |
| Codebook for d=128 (Beta(63.5, 63.5)) | **Implemented** | `turboquant_utils.h:108-117` |
| 4-bit nibble packing (2 elements/byte) | **Implemented** | `turboquant_utils.h:162-169` |
| Full dequantization pipeline | **Implemented** | `turboquant_utils.h:175-190` |
| Fused Q*K^T with packed cache | **Implemented** | `fallback_internal.cpp:867-915` |
| Fused Attn*V with packed cache | **Implemented** | `fallback_internal.cpp:917-961` |
| MHA integration (CausalLM) | **Implemented** | `mha_core.cpp:750-929` |
| Multi-backend (Fallback/AVX2/NEON) | **Partially** | v1 on all backends, v2 on fallback |
| Sliding window attention support | **Implemented** | All compute kernels |
| GQA (Grouped Query Attention) support | **Implemented** | All compute kernels |
| Prefill (multi-token) path | **Implemented** | `mha_core.cpp:857-928` |
| Decode (single-token) path | **Implemented** | `mha_core.cpp:812-856` |

### 3.2 What Is NOT Implemented

| Paper Component | Status | Notes |
|----------------|--------|-------|
| **Stage 2: QJL residual correction** | Not implemented | Community consensus: MSE-only is better in practice |
| **2-bit / 3-bit codebooks** | Not implemented | Only 4-bit (16-level) codebook provided |
| **Asymmetric K/V bit allocation** | Not implemented | Same 4-bit used for both K and V |
| **Codebooks for d ≠ 64, 128** | Not implemented | Only d=64 and d=128 supported |
| **Training / fine-tuning with TurboQuant** | Not implemented | Inference-only (forwarding is no-op) |
| **FP16 compute path with TurboQuant** | Not implemented | TurboQuant path is FP32-only |
| **AVX2/NEON optimized v2 kernels** | Not implemented | v2 only has fallback (scalar) backend |
| **Sink token support with TurboQuant** | Not implemented | `use_sink` and `use_turboquant` are mutually exclusive in practice |

---

## 4. Detailed Comparison: Paper vs Implementation

### 4.1 Quantization Granularity

| Aspect | Paper | nntrainer |
|--------|-------|-----------|
| **Granularity** | Per-head (each attention head quantized independently) | Per-head (same) |
| **Norm storage** | 1 float per head per token | 1 float per head per token (FP32) |
| **Bit-width** | 2, 3, 3.5, 4 bits tested | **4-bit only** |
| **Compression ratio** | ~3.8x (4-bit) to ~7.3x (2-bit) vs FP16 | **~4x** (4-bit vs FP16) |

### 4.2 Rotation Matrix

| Aspect | Paper | nntrainer |
|--------|-------|-----------|
| **Transform** | Randomized Hadamard Transform (RHT) | Walsh-Hadamard Transform + random sign vector (same) |
| **Complexity** | O(d log d) | O(d log d) (same) |
| **Sign generation** | Random ±1 diagonal | Deterministic LCG with seed `0xDEADBEEF` |
| **Sign sharing** | One per head dimension | **Shared across all heads and all layers** (single sign vector of size `head_dim`) |

The paper suggests that signs can be shared across heads since the distribution is the same.
The implementation uses a single global sign vector per model (generated once at `finalize()`).

### 4.3 Lloyd-Max Codebook

| Aspect | Paper | nntrainer |
|--------|-------|-----------|
| **Distribution** | Beta((d-1)/2, (d-1)/2) on [-1, 1] | Beta(31.5, 31.5) for d=64, Beta(63.5, 63.5) for d=128 |
| **Levels** | Variable (4, 8, 16 levels for 2, 3, 4 bits) | **16 levels (4-bit) only** |
| **Codebook type** | Precomputed centroids + boundaries | Same: 16 centroids + 15 boundaries (compile-time constants) |
| **Quantization** | Linear boundary search | Linear boundary search (same) |
| **Symmetry** | Symmetric around 0 | Symmetric around 0 (same) |

### 4.4 Compute Optimizations

The implementation includes compute optimizations not described in the paper:

1. **Pre-rotated queries**: Instead of dequantizing each K cache entry, the implementation
   pre-rotates the query vector: `Q_rot = H · D · Q`. Then the dot product becomes:
   ```
   Q · dequant(K) = Q · (norm · D · H · c) = norm · (H · D · Q)^T · c
   ```
   This eliminates per-row Hadamard transforms during the Q*K^T dot product computation.

2. **Accumulated V in rotated domain**: For attention-weighted value computation:
   ```
   Σ attn[j] · dequant(V[j]) = D · H · Σ(attn[j] · norm[j] · c[j])
   ```
   Centroids are accumulated in the rotated domain, and a single inverse rotation is applied at the end.

### 4.5 KV Cache Memory Layout

| Component | Non-TurboQuant | TurboQuant |
|-----------|---------------|------------|
| **K cache** | `[B, 1, max_T, H_KV × d]` FP16/UINT16 | `[B, 1, max_T, H_KV × d / 2]` UINT8 |
| **V cache** | `[B, 1, max_T, H_KV × d]` FP16/UINT16 | `[B, 1, max_T, H_KV × d / 2]` UINT8 |
| **K norms** | N/A | `[B, 1, max_T, H_KV]` FP32 |
| **V norms** | N/A | `[B, 1, max_T, H_KV]` FP32 |

For a model with `H_KV=4, d=128, max_T=4096`:
- **Non-TurboQuant**: `4096 × 512 × 2 bytes × 2 (K+V) = 8 MB` per batch (FP16)
- **TurboQuant**: `4096 × 256 × 1 byte × 2 + 4096 × 4 × 4 bytes × 2 = 2.18 MB` per batch
- **Savings**: ~3.7x memory reduction

---

## 5. Implementation Version History (v1 → v2)

The codebase contains two versions of TurboQuant:

### 5.1 v1 (Legacy, Backward Compatibility)

- **Scheme**: 3-bit uniform quantization + 1-bit sign per element
- **Granularity**: Per-group scale (group size = 32)
- **Packing**: `[sign(1) | data(3)]` per nibble, 2 elements per byte
- **Scale**: `absmax / 3.0f` per group
- **Dequantization**: `scale × (q_val - 4.0f)`
- **SIMD backends**: Fallback, AVX2, NEON all implemented

### 5.2 v2 (Current, Paper Algorithm 1)

- **Scheme**: Norm + Hadamard rotation + Lloyd-Max codebook
- **Granularity**: Per-head norm (1 float per head)
- **Packing**: `[data(4)]` per nibble, 2 elements per byte (pure 4-bit indices)
- **Codebook**: Precomputed Beta-distribution-optimal centroids
- **SIMD backends**: Fallback only (no AVX2/NEON optimized v2 kernels yet)

The MHA core layer (`mha_core.cpp`) **uses v2 exclusively** when `use_turboquant=true`.

---

## 6. Code Architecture for Applications/CausalLM

### 6.1 End-to-End Flow

```
┌─────────────────────────────────────────────────────────┐
│  nntr_config.json: "use_turboquant": true               │
└───────────────────────┬─────────────────────────────────┘
                        │ (1) Config parsing
                        ▼
┌─────────────────────────────────────────────────────────┐
│  transformer.cpp::setupParameters()                     │
│    USE_TURBOQUANT = nntr_cfg["use_turboquant"]          │
└───────────────────────┬─────────────────────────────────┘
                        │ (2) Layer construction
                        ▼
┌─────────────────────────────────────────────────────────┐
│  qwen3_causallm.cpp::createAttentionLayer()             │
│    withKey("use_turboquant", USE_TURBOQUANT ? ...)      │
│    createLayer("mha_core", a_params)                    │
└───────────────────────┬─────────────────────────────────┘
                        │ (3) Layer finalize
                        ▼
┌─────────────────────────────────────────────────────────┐
│  mha_core.cpp::finalize()                               │
│    - Allocate UINT8 packed KV cache tensors             │
│    - Allocate FP32 per-head norm tensors                │
│    - Generate rotation signs (seed=0xDEADBEEF)          │
└───────────────────────┬─────────────────────────────────┘
                        │ (4) Inference
                        ▼
┌─────────────────────────────────────────────────────────┐
│  mha_core.cpp::incremental_forwarding()                 │
│    → one_batch_incremental_forwarding_turboquant()      │
│                                                         │
│    Step 1: RoPE on Q and K                              │
│    Step 2: quantize_kv_turboquant_v2(K) → packed cache  │
│    Step 3: quantize_kv_turboquant_v2(V) → packed cache  │
│    Step 4: compute_kcaches_packed4_v2(Q, K_packed)      │
│            → attention scores                           │
│    Step 5: softmax                                      │
│    Step 6: compute_vcache_packed4_v2(attn, V_packed)    │
│            → attention output                           │
└─────────────────────────────────────────────────────────┘
```

### 6.2 File-by-File Description

#### Configuration Layer

| File | Role |
|------|------|
| `Applications/CausalLM/res/<model>/nntr_config.json` | Model config JSON; add `"use_turboquant": true` to enable |
| `Applications/CausalLM/models/transformer.h` | Declares `bool USE_TURBOQUANT = false` member |
| `Applications/CausalLM/models/transformer.cpp` | Reads `use_turboquant` from JSON config (line 113-115) |

#### Model Layer Construction

| File | Role |
|------|------|
| `Applications/CausalLM/models/qwen3/qwen3_causallm.cpp` | Passes `use_turboquant` property to `mha_core` layer (line 93) |

#### MHA Core Layer (Main Integration Point)

| File | Role |
|------|------|
| `Applications/CausalLM/layers/mha_core.h` | Defines `UseTurboQuant` property class; declares `one_batch_incremental_forwarding_turboquant()` |
| `Applications/CausalLM/layers/mha_core.cpp` | Full TurboQuant inference implementation |

Key sections in `mha_core.cpp`:

| Lines | Function |
|-------|----------|
| 154 | Read `use_turboquant` property |
| 156-198 | Allocate UINT8 packed cache + FP32 norm tensors + generate rotation signs |
| 331-353 | Dispatch to TurboQuant forwarding path |
| 444-472 | Replicate norm tensors across batches (prefill) |
| 750-929 | `one_batch_incremental_forwarding_turboquant()` - full attention pipeline |
| 774-788 | Key quantization with v2 |
| 791-806 | Value quantization with v2 |
| 812-856 | Single-token decode path (OpenMP parallelized) |
| 857-928 | Multi-token prefill path (thread pool parallelized) |

#### TurboQuant Core Algorithms

| File | Role |
|------|------|
| `nntrainer/tensor/cpu_backend/turboquant_utils.h` | All quantization/dequantization algorithms, Hadamard transform, codebooks |

Key functions:

| Function | Description |
|----------|-------------|
| `hadamard_transform(x, n)` | In-place Walsh-Hadamard Transform, O(n log n) |
| `generate_random_signs(signs, n, seed)` | Deterministic ±1 sign vector via LCG |
| `apply_rotation(input, output, signs, n)` | Forward: output = (1/√n) · H · D · input |
| `apply_inverse_rotation(data, signs, n)` | Inverse: data = D · (1/√n) · H · data |
| `lloydmax_quantize(val, cb)` | Boundary search → 4-bit index |
| `turboquant_quantize_head(...)` | Full v2 quantize: norm → normalize → rotate → quantize → pack |
| `turboquant_dequantize_head(...)` | Full v2 dequantize: unpack → centroid → inverse rotate → rescale |
| `get_codebook(head_dim)` | Return appropriate codebook for d=64 or d=128 |

#### Backend Dispatch & Kernels

| File | Role |
|------|------|
| `nntrainer/tensor/cpu_backend/cpu_backend.h` | Extern declarations for dispatch functions (lines 1511-1627) |
| `nntrainer/tensor/cpu_backend/fallback/fallback_internal.h` | Fallback function declarations |
| `nntrainer/tensor/cpu_backend/fallback/fallback_internal.cpp` | Scalar C++ implementations for v1, v1-rotated, and v2 |
| `nntrainer/tensor/cpu_backend/x86/avx2_impl.h` | AVX2 declarations (v1 only) |
| `nntrainer/tensor/cpu_backend/x86/avx2_impl.cpp` | AVX2 SIMD implementations (v1 only) |
| `nntrainer/tensor/cpu_backend/arm/neon_impl.h` | NEON declarations (v1 only) |
| `nntrainer/tensor/cpu_backend/arm/neon_impl.cpp` | NEON SIMD implementations (v1 only) |

Backend dispatch functions (called from `mha_core.cpp`):

| Function | v1 | v1-rotated | v2 |
|----------|----|------------|-----|
| `quantize_kv_turboquant[_v2]` | Fallback/AVX2/NEON | Fallback | Fallback |
| `compute_kcaches_packed4[_v2]` | Fallback/AVX2/NEON | Fallback | Fallback |
| `compute_vcache_packed4_transposed[_v2]` | Fallback/AVX2/NEON | Fallback | Fallback |

#### Tests

| File | Coverage |
|------|----------|
| `test/unittest/unittest_turboquant.cpp` | v1 pack/unpack, dequantization, round-trip error bounds, Q*K^T accuracy vs FP32, sliding window, per-head quantization, dispatch consistency |
| `test/unittest/unittest_turboquant_mha_integration.cpp` | Full MHA pipeline (quantize → Q*K^T → softmax → Attn*V), prefill & decode modes, v1 vs v2 vs FP32 comparison, GQA support, LLM-scale stress tests |
| `test/unittest/unittest_turboquant_bench.cpp` | Micro-benchmarks for v2: quantize latency, decode at various context lengths (32-2048), Hadamard transform, Lloyd-Max quantization |

---

## 7. How to Enable TurboQuant

To enable TurboQuant for a CausalLM model, add the following to the model's `nntr_config.json`:

```json
{
  "use_turboquant": true
}
```

**Requirements:**
- The model must have `head_dim` of 64 or 128 (for Lloyd-Max codebook support)
- The model must use `mha_core` layer type (currently Qwen3 models)
- Inference-only (training `forwarding()` is not implemented for TurboQuant path)

**Currently supported model architectures:**
- Qwen3 (`Applications/CausalLM/models/qwen3/qwen3_causallm.cpp`)

Other model architectures (Gemma3, Qwen2, GPT-OSS) do not yet pass the `use_turboquant`
property to their MHA layers.

---

## 8. Known Limitations and Future Work

1. **v2 SIMD optimization**: The v2 (Lloyd-Max) kernels only have scalar fallback implementations.
   AVX2 and NEON optimized kernels would significantly improve performance.

2. **Limited bit-widths**: Only 4-bit quantization is implemented. The paper shows
   3-bit and 2-bit are viable with moderate quality trade-offs.

3. **No QJL correction**: Stage 2 of the paper (QJL residual) is not implemented.
   While community consensus suggests MSE-only is sufficient, the option could be
   provided for research purposes.

4. **Fixed head dimensions**: Codebooks are only precomputed for d=64 and d=128.
   Adding codebooks for d=256 would extend support to more model architectures.

5. **Model support**: Only Qwen3 passes the `use_turboquant` flag.
   Extending to Gemma3, Qwen2, and other architectures requires adding the
   property pass-through in their respective `createAttentionLayer()` methods.

6. **No asymmetric K/V quantization**: The paper and community findings suggest
   keys benefit from higher precision than values. An asymmetric scheme
   (e.g., 4-bit keys + 3-bit values) could further reduce memory.
