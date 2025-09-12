# NNTrainer Meson Build Options Guide

This document provides a comprehensive guide to all available Meson build options for the NNTrainer project.

## Table of Contents

- [Quick Reference Tables](#quick-reference-tables)
- [Platform Configuration](#platform-configuration)
- [API Configuration](#api-configuration)
- [Applications and Testing](#applications-and-testing)
- [Backend Acceleration](#backend-acceleration)
- [Debugging and Profiling](#debugging-and-profiling)
- [ML API Integration](#ml-api-integration)
- [Advanced Configuration](#advanced-configuration)
- [Usage Examples](#usage-examples)

## Quick Reference Tables

### Platform Configuration

| Option | Type | Default | Choices | Description |
|--------|------|---------|---------|-------------|
| `platform` | combo | `none` | `none`, `tizen`, `yocto`, `android`, `windows` | Target platform for build |
| `tizen-version-major` | integer | 9999 | 4-9999 | Tizen major version (9999 = not Tizen) |
| `tizen-version-minor` | integer | 0 | 0-9999 | Tizen minor version |
| `enable-tizen-feature-check` | boolean | true | - | Enable Tizen feature checking |

### API Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enable-capi` | feature | `auto` | Enable C API (requires enable-ccapi) |
| `enable-ccapi` | boolean | true | Enable C++ API |

### Applications and Testing

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enable-app` | boolean | true | Build example applications |
| `install-app` | boolean | true | Install built applications |
| `enable-test` | boolean | true | Build unit tests |
| `test-timeout` | integer | 60 | Test timeout in seconds |
| `reduce-tolerance` | boolean | true | Reduce test tolerance |
| `enable-long-test` | boolean | false | Enable long-running tests |

### Backend Acceleration

| Option | Type | Default | Range | Description |
|--------|------|---------|-------|-------------|
| `enable-blas` | boolean | true | - | Enable OpenBLAS acceleration |
| `openblas-num-threads` | integer | 0 | 0-9999 | OpenBLAS thread count (0=auto) |
| `enable-cublas` | boolean | false | - | Enable NVIDIA CUDA cuBLAS |
| `enable-opencl` | boolean | false | - | Enable OpenCL GPU acceleration |
| `opencl-kernel-path` | string | `nntrainer_opencl_kernels` | - | OpenCL kernel files path |
| `enable-openmp` | boolean | true | - | Enable OpenMP parallel processing |
| `omp-num-threads` | integer | 6 | 0-9999 | OpenMP thread count |
| `nntr-num-threads` | integer | 1 | 0-9999 | NNTrainer internal thread count |
| `enable-fp16` | boolean | false | - | Enable 16-bit floating point |
| `enable-biqgemm` | boolean | false | - | Enable BiQGEMM quantized operations |
| `biqgemm-path` | string | `../BiQGEMM` | - | BiQGEMM library path |
| `hgemm-experimental-kernel` | boolean | false | - | Enable experimental half-precision GEMM |
| `enable-mmap` | boolean | true | - | Enable memory mapping |
| `mmap-read` | boolean | true | - | Use memory mapping for file reading |

### Debugging and Profiling

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enable-debug` | boolean | false | Enable debug mode |
| `enable-profile` | boolean | false | Enable performance profiling |
| `enable-trace` | boolean | false | Enable execution tracing |
| `enable-logging` | boolean | true | Enable logging functionality |
| `enable-benchmarks` | boolean | false | Build benchmark programs |

### ML API Integration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `ml-api-support` | feature | `auto` | Enable ML API integration |
| `capi-ml-common-actual` | string | `capi-ml-common` | ML Common API dependency name |
| `capi-ml-inference-actual` | string | `capi-ml-inference` | ML Inference API dependency name |
| `enable-nnstreamer-backbone` | boolean | false | Enable NNStreamer backbone |
| `enable-nnstreamer-tensor-filter` | feature | `auto` | Build NNStreamer tensor filter plugin |
| `enable-nnstreamer-tensor-trainer` | feature | `auto` | Build NNStreamer tensor trainer plugin |
| `nnstreamer-subplugin-install-path` | string | `lib/nnstreamer` | NNStreamer subplugin install path |

### Backend Interpreters

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enable-tflite-backbone` | boolean | true | Enable TensorFlow Lite backbone |
| `enable-tflite-interpreter` | boolean | true | Enable TensorFlow Lite interpreter |
| `enable-onnx-interpreter` | boolean | false | Enable ONNX interpreter |
| `enable-ggml` | boolean | false | Enable GGML backend |
| `ggml-thread-backend` | string | `mixed` | GGML thread backend type |

### Advanced Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enable-transformer` | boolean | false | Enable Transformer model support |
| `enable-npu` | boolean | false | Enable NPU (Neural Processing Unit) |
| `use_gym` | boolean | false | Enable OpenAI Gym environment |
| `enable-fsu` | boolean | false | Enable Flash Storage Utilization |
| `fsu-path` | string | '' | FSU path |
| `libiomp_root` | string | `./libiomp_win` | Intel OpenMP library root (Windows) |

## Platform Configuration

### `platform`
- **Type**: combo
- **Default**: `none`
- **Choices**: `none`, `tizen`, `yocto`, `android`, `windows`
- **Description**: Specifies the target platform for building.
  - `none`: Generic Linux/Unix environment
  - `tizen`: Samsung Tizen OS build
  - `yocto`: Yocto Project-based embedded systems
  - `android`: Android NDK build
  - `windows`: Windows platform

**Usage Example**:
```bash
meson setup build -Dplatform=android
```

### Tizen-Related Options

#### `tizen-version-major`
- **Type**: integer
- **Range**: 4 ~ 9999
- **Default**: 9999 (indicates not Tizen)
- **Description**: Specifies the Tizen major version.

#### `tizen-version-minor`
- **Type**: integer
- **Range**: 0 ~ 9999
- **Default**: 0
- **Description**: Specifies the Tizen minor version.

#### `enable-tizen-feature-check`
- **Type**: boolean
- **Default**: true
- **Description**: Enables feature checking on Tizen platform.

## API Configuration

### `enable-capi`
- **Type**: feature
- **Default**: `auto`
- **Description**: Enables C API. Required for NNStreamer integration.
- **Dependencies**: `enable-ccapi` must be enabled

### `enable-ccapi`
- **Type**: boolean
- **Default**: true
- **Description**: Enables C++ API. Required when using CAPI.

## Applications and Testing

### `enable-app`
- **Type**: boolean
- **Default**: true
- **Description**: Builds example applications.

### `install-app`
- **Type**: boolean
- **Default**: true
- **Description**: Installs built applications.

### `enable-test`
- **Type**: boolean
- **Default**: true
- **Description**: Builds unit tests.

### `test-timeout`
- **Type**: integer
- **Default**: 60
- **Description**: Sets test timeout in seconds.

### `reduce-tolerance`
- **Type**: boolean
- **Default**: true
- **Description**: Reduces tolerance in tests.

### `enable-long-test`
- **Type**: boolean
- **Default**: false
- **Description**: Enables long-running tests.

## Backend Acceleration

### BLAS Acceleration

#### `enable-blas`
- **Type**: boolean
- **Default**: true
- **Description**: Enables linear algebra acceleration using OpenBLAS.

#### `openblas-num-threads`
- **Type**: integer
- **Range**: 0 ~ 9999
- **Default**: 0 (automatic)
- **Description**: Specifies the number of threads for OpenBLAS.

### CUDA Acceleration

#### `enable-cublas`
- **Type**: boolean
- **Default**: false
- **Description**: Enables GPU acceleration using NVIDIA CUDA cuBLAS.

### OpenCL Acceleration

#### `enable-opencl`
- **Type**: boolean
- **Default**: false
- **Description**: Enables GPU acceleration using OpenCL.

#### `opencl-kernel-path`
- **Type**: string
- **Default**: `nntrainer_opencl_kernels`
- **Description**: Specifies the path to OpenCL kernel files.

### Multi-threading

#### `enable-openmp`
- **Type**: boolean
- **Default**: true
- **Description**: Enables parallel processing using OpenMP.

#### `omp-num-threads`
- **Type**: integer
- **Range**: 0 ~ 9999
- **Default**: 6
- **Description**: Specifies the number of threads for OpenMP.

#### `nntr-num-threads`
- **Type**: integer
- **Range**: 0 ~ 9999
- **Default**: 1
- **Description**: Specifies the number of threads for NNTrainer internals.

### Special Optimizations

#### `enable-fp16`
- **Type**: boolean
- **Default**: false
- **Description**: Enables 16-bit floating point operations (supported on ARM, x86_64).

#### `enable-biqgemm`
- **Type**: boolean
- **Default**: false
- **Description**: Enables quantized matrix multiplication using BiQGEMM library.

#### `biqgemm-path`
- **Type**: string
- **Default**: `../BiQGEMM`
- **Description**: Specifies the path to BiQGEMM library.

#### `hgemm-experimental-kernel`
- **Type**: boolean
- **Default**: false
- **Description**: Enables experimental half-precision GEMM kernel.

### Memory Management

#### `enable-mmap`
- **Type**: boolean
- **Default**: true
- **Description**: Enables memory mapping.

#### `mmap-read`
- **Type**: boolean
- **Default**: true
- **Description**: Uses memory mapping for file reading.

## Debugging and Profiling

### `enable-debug`
- **Type**: boolean
- **Default**: false
- **Description**: Enables debug mode.

### `enable-profile`
- **Type**: boolean
- **Default**: false
- **Description**: Enables performance profiling.

### `enable-trace`
- **Type**: boolean
- **Default**: false
- **Description**: Enables execution tracing.

### `enable-logging`
- **Type**: boolean
- **Default**: true
- **Description**: Enables logging functionality.

### `enable-benchmarks`
- **Type**: boolean
- **Default**: false
- **Description**: Builds benchmark programs.

## ML API Integration

### `ml-api-support`
- **Type**: feature
- **Default**: `auto`
- **Description**: Enables ML API integration. Required for NNStreamer integration.

### `capi-ml-common-actual`
- **Type**: string
- **Default**: `capi-ml-common`
- **Description**: Specifies the actual name of ML Common API dependency.

### `capi-ml-inference-actual`
- **Type**: string
- **Default**: `capi-ml-inference`
- **Description**: Specifies the actual name of ML Inference API dependency.

### NNStreamer Integration

#### `enable-nnstreamer-backbone`
- **Type**: boolean
- **Default**: false
- **Description**: Enables NNStreamer backbone.

#### `enable-nnstreamer-tensor-filter`
- **Type**: feature
- **Default**: `auto`
- **Description**: Builds NNStreamer tensor filter plugin.

#### `enable-nnstreamer-tensor-trainer`
- **Type**: feature
- **Default**: `auto`
- **Description**: Builds NNStreamer tensor trainer plugin.

#### `nnstreamer-subplugin-install-path`
- **Type**: string
- **Default**: `lib/nnstreamer`
- **Description**: Specifies NNStreamer subplugin installation path.

## Backend Interpreters

### `enable-tflite-backbone`
- **Type**: boolean
- **Default**: true
- **Description**: Enables TensorFlow Lite backbone.

### `enable-tflite-interpreter`
- **Type**: boolean
- **Default**: true
- **Description**: Enables TensorFlow Lite interpreter.

### `enable-onnx-interpreter`
- **Type**: boolean
- **Default**: false
- **Description**: Enables ONNX interpreter.

### `enable-ggml`
- **Type**: boolean
- **Default**: false
- **Description**: Enables GGML (Georgi Gerganov Machine Learning) backend.

#### `ggml-thread-backend`
- **Type**: string
- **Default**: `mixed`
- **Description**: Specifies GGML thread backend type.

## Advanced Configuration

### `enable-transformer`
- **Type**: boolean
- **Default**: false
- **Description**: Enables Transformer model support.

### `enable-npu`
- **Type**: boolean
- **Default**: false
- **Description**: Enables NPU (Neural Processing Unit) support.

### `use_gym`
- **Type**: boolean
- **Default**: false
- **Description**: Enables OpenAI Gym environment support.

### Flash Storage Utilization

#### `enable-fsu`
- **Type**: boolean
- **Default**: false
- **Description**: Enables Flash Storage Utilization.

#### `fsu-path`
- **Type**: string
- **Default**: '' (empty string)
- **Description**: Specifies FSU path.

### Windows-Specific Configuration

#### `libiomp_root`
- **Type**: string
- **Default**: `./libiomp_win`
- **Description**: Specifies Intel OpenMP library root path on Windows.

## Usage Examples

### Basic Build
```bash
meson setup build
ninja -C build
```

### Android Build
```bash
meson setup build -Dplatform=android -Denable-app=false
ninja -C build
```

### High-Performance Build (All Accelerations Enabled)
```bash
meson setup build \
  -Denable-blas=true \
  -Denable-openmp=true \
  -Denable-fp16=true \
  -Dopenblas-num-threads=8 \
  -Domp-num-threads=8
ninja -C build
```

### Developer Build (Debug Enabled)
```bash
meson setup build \
  -Denable-debug=true \
  -Denable-profile=true \
  -Denable-trace=true \
  -Denable-test=true \
  -Denable-benchmarks=true
ninja -C build
```

### Tizen Build
```bash
meson setup build \
  -Dplatform=tizen \
  -Dtizen-version-major=6 \
  -Dtizen-version-minor=0 \
  -Denable-capi=enabled
ninja -C build
```

### Minimal Build (Embedded Environment)
```bash
meson setup build \
  -Denable-app=false \
  -Denable-test=false \
  -Denable-logging=false \
  -Denable-blas=false \
  -Denable-openmp=false
ninja -C build
```

### GPU Acceleration Build
```bash
meson setup build \
  -Denable-opencl=true \
  -Denable-cublas=true \
  -Denable-fp16=true
ninja -C build
```

### NNStreamer Integration Build
```bash
meson setup build \
  -Dml-api-support=enabled \
  -Denable-capi=enabled \
  -Denable-nnstreamer-backbone=true \
  -Denable-nnstreamer-tensor-filter=enabled \
  -Denable-nnstreamer-tensor-trainer=enabled
ninja -C build
```

## Build Configuration Matrix

The following table shows recommended configurations for different use cases:

| Use Case | Platform | Key Options | Description |
|----------|----------|-------------|-------------|
| **Production Server** | `none` | `-Denable-blas=true -Denable-openmp=true -Denable-logging=false` | High performance, minimal logging |
| **Mobile Development** | `android` | `-Denable-app=false -Denable-test=false -Denable-fp16=true` | Optimized for mobile resources |
| **Embedded IoT** | `none` | `-Denable-app=false -Denable-blas=false -Denable-openmp=false` | Minimal resource usage |
| **Research/Development** | `none` | `-Denable-debug=true -Denable-profile=true -Denable-benchmarks=true` | Full debugging and profiling |
| **Tizen Application** | `tizen` | `-Denable-capi=enabled -Denable-tizen-feature-check=true` | Tizen-specific features |
| **NNStreamer Pipeline** | `none` | `-Dml-api-support=enabled -Denable-nnstreamer-backbone=true` | NNStreamer integration |

## Important Notes

### 1. **Dependency Requirements**
Some options have interdependencies:
- `enable-capi` requires `enable-ccapi` to be enabled
- `mmap-read` and `enable-transformer` cannot be used simultaneously
- GPU acceleration options require appropriate drivers and libraries

### 2. **Platform-Specific Limitations**
- **Android**: Some tests and applications are not supported
- **Windows**: Additional library setup may be required
- **Tizen**: Requires specific Tizen SDK and feature permissions

### 3. **Performance Considerations**
- Debug options can significantly impact performance in production builds
- Thread count should be adjusted based on available CPU cores
- GPU acceleration requires compatible hardware and drivers

### 4. **Memory Usage**
- `enable-fp16` reduces memory usage but may affect accuracy
- `enable-mmap` improves memory efficiency for large model loading
- Backend interpreters (TFLite, ONNX, GGML) have different memory footprints

### 5. **Build Time and Size**
- Enabling all features increases build time and binary size
- Consider disabling unused features for production builds
- Test builds benefit from parallel compilation (`ninja -j$(nproc)`)

## Troubleshooting

### Common Build Issues

| Issue | Likely Cause | Solution |
|-------|--------------|----------|
| OpenBLAS not found | Missing system dependencies | Install `libopenblas-dev` or use subproject |
| CUDA compilation fails | Missing CUDA toolkit | Install NVIDIA CUDA SDK |
| Test timeouts | Insufficient system resources | Increase `test-timeout` value |
| Android build fails | NDK not configured | Set up Android NDK environment |

### Environment Variables

Some options can be influenced by environment variables:
- `CC`, `CXX`: Compiler selection
- `CFLAGS`, `CXXFLAGS`: Additional compilation flags  
- `LDFLAGS`: Additional linking flags
- `PKG_CONFIG_PATH`: Package configuration paths

## Additional Resources

- [NNTrainer Official Documentation](https://github.com/nnstreamer/nntrainer)
- [Build Guide](docs/getting-started.md)
- [Example Usage](docs/how-to-run-examples.md)
- [Contributing Guidelines](CONTRIBUTING.md)