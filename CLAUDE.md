# CLAUDE.md

## Project Overview

NNTrainer is a C++17 neural network training and inference framework for edge devices (phones, embedded systems). It supports on-device training (transfer learning, few-shot learning) and efficient LLM inference with memory optimization techniques (FSU, MoE caching). Licensed under Apache 2.0.

Repository: https://github.com/nnstreamer/nntrainer
Platforms: Linux, Tizen, Android, Windows

## Build and Test

Build system: **Meson + Ninja** (meson >= 0.55.0)

```bash
# Initialize submodules (required for googletest, openblas, iniparser, etc.)
git submodule sync && git submodule update --init --recursive

# Configure (default: release build, werror=true)
meson setup build

# Build
ninja -C build

# Run all tests
ninja test -C build
# Or run specific test suite
meson test -C build --suite unittests
```

### Key meson options

Pass with `-D<option>=<value>` to `meson setup`:

| Option | Default | Description |
|--------|---------|-------------|
| `platform` | none | Target: none, tizen, yocto, android, windows |
| `enable-test` | true | Build and run tests |
| `enable-fp16` | false | FP16 half-precision support |
| `enable-opencl` | false | OpenCL GPU backend |
| `enable-capi` / `enable-ccapi` | varies | C / C++ public API |
| `enable-blas` | true | BLAS backend (OpenBLAS) |
| `enable-openmp` | varies | OpenMP parallelism |
| `enable-fsu` | false | Flash Storage Utilization for LLM inference |
| `enable-transformer` | false | Transformer model support |
| `arm-arch` | armv7l | ARM architecture target |

See `meson_options.txt` for the full list.

## Code Style and Conventions

### Formatting

- **clang-format v14** using the repo's `.clang-format`. Run it on all `.cpp` and `.c` files.
- 80-column limit, 2-space indent, no tabs, Attach (K&R) brace style.
- Header files (`.h`) have relaxed rules: indent and 80-col may differ.
- **Warnings are treated as errors** (`werror=true`). Never weaken warning flags.

### Naming

| Element | Convention | Example |
|---------|-----------|---------|
| Classes/Structs | `PascalCase` | `Conv2DLayer`, `NeuralNetwork` |
| Functions/Methods | `camelCase` | `forwarding()`, `calcDerivative()` |
| Constants/Macros | `UPPER_SNAKE_CASE` | `CONV2D_DIM`, `ML_ERROR_BAD_ADDRESS` |
| Files | `snake_case` | `conv2d_layer.cpp`, `tensor_base.h` |

### Namespaces

- Core library: `nntrainer` (sub: `nntrainer::props`, `nntrainer::opencl`, `nntrainer::exception`)
- Public C++ API: `ml::train` (in `api/ccapi/`)

### Header guards

Double-underscore style: `#ifndef __FILE_NAME_H__`

### License headers

Every new file needs an SPDX header with Doxygen tags:

```cpp
// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) <year> <author> <email>
 *
 * @file   filename.cpp
 * @date   DD Month YYYY
 * @brief  Brief description
 * @see    https://github.com/nntrainer/nntrainer
 * @author Name <email>
 * @bug    No known bugs except for NYI items
 */
```

### Language standards

- C++ code: **C++17** (`cpp_std=c++17`)
- C code: **C89** (`c_std=gnu89`)

### Error handling

- Use `NNTR_THROW_IF(pred, exception_type)` macro for throwing exceptions.
- Prefer RAII and clear ownership. Error paths must release resources.

## Directory Structure

```
nntrainer/           Core library
  layers/            Layer implementations (conv2d, fc, attention, rnn, etc.)
    cl_layers/       OpenCL GPU layer implementations
    loss/            Loss function layers
  tensor/            Tensor classes, memory management, quantization
    cpu_backend/     CPU-specific backends (ARM NEON, x86, fallback)
    cl_operations/   OpenCL tensor operations
  models/            NeuralNetwork class, model loading/saving
  graph/             NetworkGraph, graph traversal, connections
  compiler/          Graph compilers, interpreters (INI, TFLite, ONNX), realizers
  optimizers/        SGD, Adam, AdamW, Lion, LR schedulers
  dataset/           Data producers and buffering
  opencl/            OpenCL runtime management
  utils/             Utilities (properties, profiling, SIMD helpers)
api/
  ccapi/             Public C++ API (ml::train namespace)
  capi/              Public C API (Tizen-compatible)
test/
  unittest/          GTest unit tests (layers/, models/, datasets/, etc.)
  input_gen/         Python test input generators
  test_models/       Model configs and golden data
Applications/        Example apps (ResNet, LLaMA, MNIST, VGG, YOLOv3, etc.)
docs/                Documentation
subprojects/         Dependencies (googletest, openblas, iniparser, ruy, CLBlast)
```

## Architecture Notes

### Layer hierarchy

All layers derive from `nntrainer::Layer` (defined in `layer_devel.h`). Layers with weights extend `LayerImpl`. Each layer implements `finalize()` (shape inference), `forwarding()`, `calcDerivative()`, and `calcGrad()`. OpenCL layers live in `cl_layers/` and extend `LayerImplCl`.

### Model pipeline

INI/TFLite/ONNX config -> **Interpreter** (parses to GraphRepresentation) -> **Realizers** (graph transforms: activation fusion, BN folding, flatten, multiout) -> **Compiler** (produces executable NetworkGraph) -> **NeuralNetwork** (training/inference engine).

### Tensor design

`Tensor` is a type-erasing wrapper over `TensorBase` subclasses (`FloatTensor`, `HalfTensor`, `Int4Tensor`, quantized variants like `Q4_0Tensor`, `Q4_K_Tensor`). Supports NCHW and NHWC formats. Memory is managed by `Manager` with optimized memory planners for reuse.

### Property system

Layers and optimizers use a typed property system (`base_properties.h`). Properties are configured via string key-value pairs (e.g., `"unit=5"`, `"weight_decay=0.0"`).

## Testing Patterns

- Tests use **GTest/GMock**. Test files follow `unittest_<component>.cpp` naming.
- Layer tests use **parameterized golden tests**: provide a layer factory, properties, input shape, and a `.nnlayergolden` reference file. See `layers_common_tests.h`.
- Negative test cases should outnumber positive ones for new features.
- Test timeout: 90 seconds (configurable via `test-timeout` meson option).

## Commit and PR Workflow

- **Signed-off-by** is mandatory on every commit (`git commit -s`).
- PRs require **at least 2 committer approvals** and all CI must pass.
- Keep commits focused and descriptive: describe the bug and fix, or describe the new feature.
- CI runs on Ubuntu (22.04/24.04), Tizen, Android, Windows.

## Review Priorities (descending)

1. Correctness & API/ABI safety
2. CI compatibility & portability
3. Memory / ownership & error handling
4. Performance (only when relevant)
5. Testability & coverage

See `.cursorrules` for detailed review comment formatting guidelines.

## Key References

- [Getting Started](docs/getting-started.md) - build prerequisites and setup
- [Coding Convention](docs/coding-convention.md) - code style rules
- [How to Contribute](docs/contributing.md) - PR process and merge criteria
- [INI Configuration](docs/configuration-ini.md) - model config format
- [How to Use Testcases](docs/how-to-use-testcases.md) - testing guide
