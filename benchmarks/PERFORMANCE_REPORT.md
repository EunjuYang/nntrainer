# Attention Kernel Optimization — Performance Report

## Environment

- **Platform**: x86_64 (AVX2 + FMA + F16C)
- **Build**: Meson/Ninja, `-O2`, OpenMP enabled
- **Iterations**: 100 per benchmark
- **Date**: 2026-02-24

## Summary of Optimizations

| # | Optimization | Affected Kernels | Backends |
|---|-------------|-----------------|----------|
| 1 | Eliminate redundant `exp()` in softmax (delegate non-inplace → inplace) | `softmax_row`, `softmax_row_with_sink` | AVX2, ARM NEON (FP32+FP16) |
| 2 | Restructure `compute_kcaches` loop (FP16→FP32 conversion once per KV row) | `compute_kcaches` | AVX2, ARM NEON (FP32+FP16) |
| 3 | Precompute `1/√d` outside hot loops | `compute_kcaches` | AVX2, ARM NEON (FP32+FP16) |
| 4 | OpenMP parallelization of RoPE | `apply_rotary_emb_tensor_v2` | Application layer (all backends) |
| 5 | Stack-allocate softmax temporaries (max/sum arrays) | `softmax_row_inplace`, `softmax_row_with_sink_inplace` | AVX2, ARM NEON (FP32+FP16) |
| 6 | Stack-allocate vcache FP32 conversion buffer | `compute_fp16vcache_fp32_transposed` | AVX2 |

## Correctness Verification

All optimized kernels produce bit-identical (or within FP rounding tolerance)
results compared to naive scalar reference implementations.

### Qwen2-0.5B (14 Q-heads, 2 KV-heads, dim=64, GQA=7)

| Kernel | Context | Max Abs Diff | Status |
|--------|---------|-------------|--------|
| softmax_row | 1 | 0.00e+00 | OK |
| softmax_row | 64 | 3.73e-09 | OK |
| softmax_row | 256 | 9.31e-10 | OK |
| softmax_row | 512 | 4.66e-10 | OK |
| compute_kcaches | 1 | 1.19e-07 | OK |
| compute_kcaches | 64 | 2.98e-07 | OK |
| compute_kcaches | 256 | 3.58e-07 | OK |
| compute_kcaches | 512 | 3.58e-07 | OK |
| compute_vcache | 1 | 0.00e+00 | OK |
| compute_vcache | 64 | 0.00e+00 | OK |
| compute_vcache | 256 | 0.00e+00 | OK |
| compute_vcache | 512 | 0.00e+00 | OK |
| compute_rotary_emb | — | 5.96e-08 | OK |

### Qwen3-4B (32 Q-heads, 8 KV-heads, dim=128, GQA=4)

| Kernel | Context | Max Abs Diff | Status |
|--------|---------|-------------|--------|
| softmax_row | 1 | 0.00e+00 | OK |
| softmax_row | 64 | 3.73e-09 | OK |
| softmax_row | 256 | 9.31e-10 | OK |
| softmax_row | 512 | 4.66e-10 | OK |
| compute_kcaches | 1 | 1.19e-07 | OK |
| compute_kcaches | 64 | 6.56e-07 | OK |
| compute_kcaches | 256 | 5.36e-07 | OK |
| compute_kcaches | 512 | 5.36e-07 | OK |
| compute_vcache | 1 | 0.00e+00 | OK |
| compute_vcache | 64 | 0.00e+00 | OK |
| compute_vcache | 256 | 0.00e+00 | OK |
| compute_vcache | 512 | 0.00e+00 | OK |
| compute_rotary_emb | — | 1.19e-07 | OK |

## Performance Measurements (x86_64 / AVX2)

### Qwen2-0.5B

| Kernel | ctx=1 | ctx=64 | ctx=256 | ctx=512 |
|--------|-------|--------|---------|---------|
| softmax_row | < 0.1 µs | 1.9 µs | 12.5 µs | 26.7 µs |
| compute_kcaches | < 0.1 µs | 7.4 µs | 24.1 µs | 43.0 µs |
| compute_vcache | < 0.1 µs | 5.7 µs | 25.1 µs | 49.4 µs |
| compute_rotary_emb | < 0.1 µs | — | — | — |

### Qwen3-4B

| Kernel | ctx=1 | ctx=64 | ctx=256 | ctx=512 |
|--------|-------|--------|---------|---------|
| softmax_row | < 0.1 µs | 2.0 µs | 12.4 µs | 31.0 µs |
| compute_kcaches | < 0.1 µs | 24.9 µs | 86.5 µs | 193.5 µs |
| compute_vcache | < 0.1 µs | 23.1 µs | 101.7 µs | 204.1 µs |
| compute_rotary_emb | < 0.1 µs | — | — | — |

## Optimization Details

### 1. Softmax: Eliminate Redundant exp() Calls

**Problem**: The original `softmax_row` (non-inplace) computed `exp(x - max)` twice
per element — once during the sum-accumulation pass and once during the
normalization pass. `exp()` is the most expensive SIMD operation in softmax.

**Fix**: Delegate `softmax_row` → `softmax_row_inplace`, which computes `exp()`
once and stores the result in-place, then divides by the column sum.

**Impact**: ~50% reduction in `exp()` calls inside softmax.

### 2. compute_kcaches: Loop Restructuring

**Problem**: The original inner-loop order was `[n (cache head), g (GQA group),
t_row (context)]`, causing the same FP16 KV-cache row to be converted to FP32
`gqa_size` times (once per GQA group).

**Fix**: Reorder to `[n, t_row, g]` so FP16→FP32 conversion happens once per
KV-cache row, shared across all GQA groups.

**Impact**: Reduces FP16→FP32 conversions by `gqa_size×` (7× for Qwen2-0.5B,
4× for Qwen3-4B).

### 3. compute_kcaches: Precompute inv_sqrt

**Problem**: `1.0f / sqrt(head_dim)` was recomputed inside the innermost loop
for every dot-product result.

**Fix**: Precompute `inv_sqrt_head_dim` once before the loop and multiply.

**Impact**: Eliminates `O(rows × heads × gqa)` redundant `sqrt()` + division
operations.

### 4. RoPE: OpenMP Parallelization

**Problem**: `apply_rotary_emb_tensor_v2` uses triple-nested loops over
`[batch, channel, height]` that are entirely independent, but ran single-threaded.

**Fix**: Flatten the 3 loops into a single index and add
`#pragma omp parallel for schedule(static)` with `if(total_iters > 1)` guard.

**Impact**: Near-linear scaling with available cores for multi-head/multi-batch
scenarios.

### 5. Softmax: Stack Allocation for Temporaries

**Problem**: `softmax_row_inplace` allocated `max_vals[]` and `sum_vals[]` on the
heap via `new float[num_heads]`. This function is called thousands of times per
token during autoregressive decoding, incurring repeated `malloc`/`free` overhead.

**Fix**: Use stack-allocated `alignas(32) float[128]` buffers for typical
`num_heads` (≤128), falling back to heap only for unusually large configurations.

**Impact**: Eliminates heap allocation overhead in the hot decoding path.

### 6. vcache: Stack Allocation for FP32 Buffer

**Problem**: `compute_fp16vcache_fp32_transposed` allocated a temporary FP32
conversion buffer on the heap in every call.

**Fix**: Use stack-allocated `alignas(32) float[256]` for typical `head_dim`
values, falling back to heap for large dimensions.

**Impact**: Eliminates heap allocation in the vcache matmul hot path.

## Notes

- **ARM NEON**: Optimizations 1–3 and 5 were applied to both `neon_impl.cpp`
  (FP32 path) and `neon_impl_fp16.cpp` (FP16 path). ARM NEON uses `alignas(16)`
  for 128-bit NEON alignment instead of `alignas(32)`.
- **Fallback backend**: All attention kernel functions in the fallback backend
  are NYI stubs (`std::runtime_error`). No changes were made.
- **Tolerance**: All max absolute differences are within expected FP32 rounding
  tolerance (< 1e-6), confirming numerical correctness is preserved.
