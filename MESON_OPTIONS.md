# NNTrainer Meson Build Options Guide

이 문서는 NNTrainer 프로젝트의 Meson 빌드 시스템에서 사용할 수 있는 모든 옵션들에 대한 상세한 설명을 제공합니다.

## 목차

- [플랫폼 설정](#플랫폼-설정)
- [API 설정](#api-설정)
- [애플리케이션 및 테스트](#애플리케이션-및-테스트)
- [백엔드 가속](#백엔드-가속)
- [디버깅 및 프로파일링](#디버깅-및-프로파일링)
- [ML API 통합](#ml-api-통합)
- [고급 설정](#고급-설정)
- [예제 사용법](#예제-사용법)

## 플랫폼 설정

### `platform`
- **타입**: combo (선택형)
- **선택지**: `none`, `tizen`, `yocto`, `android`, `windows`
- **기본값**: `none`
- **설명**: 빌드 대상 플랫폼을 지정합니다.
  - `none`: 일반적인 Linux/Unix 환경
  - `tizen`: Samsung Tizen OS용 빌드
  - `yocto`: Yocto Project 기반 임베디드 시스템
  - `android`: Android NDK 빌드
  - `windows`: Windows 플랫폼

**사용 예시**:
```bash
meson setup build -Dplatform=android
```

### Tizen 관련 옵션

#### `tizen-version-major`
- **타입**: integer
- **범위**: 4 ~ 9999
- **기본값**: 9999 (Tizen이 아님을 의미)
- **설명**: Tizen 메이저 버전을 지정합니다.

#### `tizen-version-minor`
- **타입**: integer
- **범위**: 0 ~ 9999
- **기본값**: 0
- **설명**: Tizen 마이너 버전을 지정합니다.

#### `enable-tizen-feature-check`
- **타입**: boolean
- **기본값**: true
- **설명**: Tizen 플랫폼에서 기능 검사를 활성화합니다.

## API 설정

### `enable-capi`
- **타입**: feature
- **기본값**: `auto`
- **설명**: C API를 활성화합니다. NNStreamer와의 연동을 위해 필요합니다.
- **의존성**: `enable-ccapi`가 활성화되어야 함

### `enable-ccapi`
- **타입**: boolean
- **기본값**: true
- **설명**: C++ API를 활성화합니다. CAPI 사용 시 필수입니다.

## 애플리케이션 및 테스트

### `enable-app`
- **타입**: boolean
- **기본값**: true
- **설명**: 예제 애플리케이션들을 빌드합니다.

### `install-app`
- **타입**: boolean
- **기본값**: true
- **설명**: 빌드된 애플리케이션들을 설치합니다.

### `enable-test`
- **타입**: boolean
- **기본값**: true
- **설명**: 단위 테스트를 빌드합니다.

### `test-timeout`
- **타입**: integer
- **기본값**: 60
- **설명**: 테스트 타임아웃 시간(초)을 설정합니다.

### `reduce-tolerance`
- **타입**: boolean
- **기본값**: true
- **설명**: 테스트에서 허용 오차를 줄입니다.

### `enable-long-test`
- **타입**: boolean
- **기본값**: false
- **설명**: 시간이 오래 걸리는 테스트들을 활성화합니다.

## 백엔드 가속

### BLAS 가속

#### `enable-blas`
- **타입**: boolean
- **기본값**: true
- **설명**: OpenBLAS를 사용한 선형대수 연산 가속을 활성화합니다.

#### `openblas-num-threads`
- **타입**: integer
- **범위**: 0 ~ 9999
- **기본값**: 0 (자동)
- **설명**: OpenBLAS에서 사용할 스레드 수를 지정합니다.

### CUDA 가속

#### `enable-cublas`
- **타입**: boolean
- **기본값**: false
- **설명**: NVIDIA CUDA cuBLAS를 사용한 GPU 가속을 활성화합니다.

### OpenCL 가속

#### `enable-opencl`
- **타입**: boolean
- **기본값**: false
- **설명**: OpenCL을 사용한 GPU 가속을 활성화합니다.

#### `opencl-kernel-path`
- **타입**: string
- **기본값**: `nntrainer_opencl_kernels`
- **설명**: OpenCL 커널 파일들의 경로를 지정합니다.

### 멀티스레딩

#### `enable-openmp`
- **타입**: boolean
- **기본값**: true
- **설명**: OpenMP를 사용한 병렬 처리를 활성화합니다.

#### `omp-num-threads`
- **타입**: integer
- **범위**: 0 ~ 9999
- **기본값**: 6
- **설명**: OpenMP에서 사용할 스레드 수를 지정합니다.

#### `nntr-num-threads`
- **타입**: integer
- **범위**: 0 ~ 9999
- **기본값**: 1
- **설명**: NNTrainer 내부에서 사용할 스레드 수를 지정합니다.

### 특수 최적화

#### `enable-fp16`
- **타입**: boolean
- **기본값**: false
- **설명**: 16비트 부동소수점 연산을 활성화합니다. (ARM, x86_64에서 지원)

#### `enable-biqgemm`
- **타입**: boolean
- **기본값**: false
- **설명**: BiQGEMM 라이브러리를 사용한 양자화 행렬 곱셈을 활성화합니다.

#### `biqgemm-path`
- **타입**: string
- **기본값**: `../BiQGEMM`
- **설명**: BiQGEMM 라이브러리의 경로를 지정합니다.

#### `hgemm-experimental-kernel`
- **타입**: boolean
- **기본값**: false
- **설명**: 실험적인 half-precision GEMM 커널을 활성화합니다.

### 메모리 관리

#### `enable-mmap`
- **타입**: boolean
- **기본값**: true
- **설명**: 메모리 매핑을 활성화합니다.

#### `mmap-read`
- **타입**: boolean
- **기본값**: true
- **설명**: 파일 읽기에 메모리 매핑을 사용합니다.

## 디버깅 및 프로파일링

### `enable-debug`
- **타입**: boolean
- **기본값**: false
- **설명**: 디버그 모드를 활성화합니다.

### `enable-profile`
- **타입**: boolean
- **기본값**: false
- **설명**: 성능 프로파일링을 활성화합니다.

### `enable-trace`
- **타입**: boolean
- **기본값**: false
- **설명**: 실행 추적을 활성화합니다.

### `enable-logging`
- **타입**: boolean
- **기본값**: true
- **설명**: 로깅 기능을 활성화합니다.

### `enable-benchmarks`
- **타입**: boolean
- **기본값**: false
- **설명**: 벤치마크 프로그램들을 빌드합니다.

## ML API 통합

### `ml-api-support`
- **타입**: feature
- **기본값**: `auto`
- **설명**: ML API와의 통합을 활성화합니다. NNStreamer와의 연동에 필요합니다.

### `capi-ml-common-actual`
- **타입**: string
- **기본값**: `capi-ml-common`
- **설명**: ML Common API 의존성의 실제 이름을 지정합니다.

### `capi-ml-inference-actual`
- **타입**: string
- **기본값**: `capi-ml-inference`
- **설명**: ML Inference API 의존성의 실제 이름을 지정합니다.

### NNStreamer 통합

#### `enable-nnstreamer-backbone`
- **타입**: boolean
- **기본값**: false
- **설명**: NNStreamer 백본을 활성화합니다.

#### `enable-nnstreamer-tensor-filter`
- **타입**: feature
- **기본값**: `auto`
- **설명**: NNStreamer tensor filter 플러그인을 빌드합니다.

#### `enable-nnstreamer-tensor-trainer`
- **타입**: feature
- **기본값**: `auto`
- **설명**: NNStreamer tensor trainer 플러그인을 빌드합니다.

#### `nnstreamer-subplugin-install-path`
- **타입**: string
- **기본값**: `lib/nnstreamer`
- **설명**: NNStreamer 서브플러그인 설치 경로를 지정합니다.

## 백엔드 인터프리터

### `enable-tflite-backbone`
- **타입**: boolean
- **기본값**: true
- **설명**: TensorFlow Lite 백본을 활성화합니다.

### `enable-tflite-interpreter`
- **타입**: boolean
- **기본값**: true
- **설명**: TensorFlow Lite 인터프리터를 활성화합니다.

### `enable-onnx-interpreter`
- **타입**: boolean
- **기본값**: false
- **설명**: ONNX 인터프리터를 활성화합니다.

### `enable-ggml`
- **타입**: boolean
- **기본값**: false
- **설명**: GGML(Georgi Gerganov Machine Learning) 백엔드를 활성화합니다.

#### `ggml-thread-backend`
- **타입**: string
- **기본값**: `mixed`
- **설명**: GGML 스레드 백엔드 타입을 지정합니다.

## 고급 설정

### `enable-transformer`
- **타입**: boolean
- **기본값**: false
- **설명**: Transformer 모델 지원을 활성화합니다.

### `enable-npu`
- **타입**: boolean
- **기본값**: false
- **설명**: NPU(Neural Processing Unit) 지원을 활성화합니다.

### `use_gym`
- **타입**: boolean
- **기본값**: false
- **설명**: OpenAI Gym 환경 지원을 활성화합니다.

### Flash Storage Utilization

#### `enable-fsu`
- **타입**: boolean
- **기본값**: false
- **설명**: Flash Storage Utilization을 활성화합니다.

#### `fsu-path`
- **타입**: string
- **기본값**: '' (빈 문자열)
- **설명**: FSU 경로를 지정합니다.

### Windows 전용 설정

#### `libiomp_root`
- **타입**: string
- **기본값**: `./libiomp_win`
- **설명**: Windows에서 Intel OpenMP 라이브러리의 루트 경로를 지정합니다.

## 예제 사용법

### 기본 빌드
```bash
meson setup build
ninja -C build
```

### Android 빌드
```bash
meson setup build -Dplatform=android -Denable-app=false
ninja -C build
```

### 고성능 빌드 (모든 가속 활성화)
```bash
meson setup build \
  -Denable-blas=true \
  -Denable-openmp=true \
  -Denable-fp16=true \
  -Dopenblas-num-threads=8 \
  -Domp-num-threads=8
ninja -C build
```

### 개발자 빌드 (디버깅 활성화)
```bash
meson setup build \
  -Denable-debug=true \
  -Denable-profile=true \
  -Denable-trace=true \
  -Denable-test=true \
  -Denable-benchmarks=true
ninja -C build
```

### Tizen 빌드
```bash
meson setup build \
  -Dplatform=tizen \
  -Dtizen-version-major=6 \
  -Dtizen-version-minor=0 \
  -Denable-capi=enabled
ninja -C build
```

### 최소 빌드 (임베디드 환경)
```bash
meson setup build \
  -Denable-app=false \
  -Denable-test=false \
  -Denable-logging=false \
  -Denable-blas=false \
  -Denable-openmp=false
ninja -C build
```

## 주의사항

1. **의존성 관계**: 일부 옵션들은 서로 의존관계가 있습니다.
   - `enable-capi`를 사용하려면 `enable-ccapi`가 활성화되어야 합니다.
   - `mmap-read`와 `enable-transformer`는 동시에 사용할 수 없습니다.

2. **플랫폼별 제한사항**:
   - Android에서는 일부 테스트와 애플리케이션이 지원되지 않습니다.
   - Windows에서는 추가적인 라이브러리 설정이 필요할 수 있습니다.

3. **성능 고려사항**:
   - 디버깅 옵션들은 성능에 영향을 줄 수 있으므로 프로덕션 빌드에서는 비활성화하는 것을 권장합니다.
   - 스레드 수 설정은 시스템의 CPU 코어 수에 맞춰 조정하세요.

4. **메모리 사용량**:
   - `enable-fp16`은 메모리 사용량을 줄일 수 있지만 정확도에 영향을 줄 수 있습니다.
   - `enable-mmap`은 대용량 모델 로딩 시 메모리 효율성을 향상시킵니다.

## 더 많은 정보

- [NNTrainer 공식 문서](https://github.com/nnstreamer/nntrainer)
- [빌드 가이드](docs/getting-started.md)
- [예제 실행 방법](docs/how-to-run-examples.md)