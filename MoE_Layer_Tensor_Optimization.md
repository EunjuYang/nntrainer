# MoE Layer Optimization using Tensor Operations

## 개요
레이어 코드에 직접 SIMD intrinsic을 사용하는 대신, NNTrainer의 Tensor operation과 ARM backend를 활용하여 최적화를 수행했습니다. 이 접근 방식은 코드의 유지보수성과 이식성을 향상시키면서도 성능을 최적화합니다.

## 주요 최적화 전략

### 1. **Tensor Operation 활용**
- 직접적인 SIMD 코드 대신 Tensor 클래스의 최적화된 연산 사용
- `copyData()`, `add_i()`, `multiply()`, `dot()` 등의 기존 최적화된 메서드 활용
- Backend에서 자동으로 NEON/AVX 등의 SIMD 명령어 활용

### 2. **Backend 함수 확장**
ARM backend에 MoE 레이어에 특화된 최적화 함수 추가:

```cpp
// arm_compute_backend_opt.h/cpp
- batch_copy_indexed(): 인덱스 기반 배치 복사
- weighted_add_i(): 가중치 적용 누적 연산
- batch_weighted_add(): 배치 단위 가중치 누적
- batch_swiglu(): 배치 SwiGLU 활성화
- prefetch_weights(): 가중치 프리페치
```

### 3. **메모리 관리 최적화**
```cpp
// Thread-local 메모리 풀 사용
static thread_local nntrainer::Tensor gate_out;
static thread_local nntrainer::Tensor acti_out;
static thread_local std::vector<nntrainer::Tensor> expert_outputs;
```

### 4. **Zero-copy 최적화**
```cpp
// Shared tensor 활용으로 불필요한 복사 제거
token_input = input.getSharedDataTensor(token_input_dim, token_offset, true);
```

### 5. **병렬 처리 개선**
```cpp
// 조건부 병렬화로 오버헤드 최소화
#pragma omp parallel for schedule(guided) if(size > threshold)
```

## 구현 상세

### compute_expert_forward 최적화

1. **배치 입력 준비**
   - Tensor의 `getSharedDataTensor()`와 `copyData()` 활용
   - Contiguous 메모리 체크로 최적 경로 선택

2. **행렬 연산**
   - Tensor의 `dot()` 연산 활용 (내부적으로 GEMM 최적화)
   - Backend에서 자동으로 NEON/BLAS 선택

3. **활성화 함수**
   - Backend의 `swiglu()` 함수 직접 호출
   - NEON 최적화된 구현 자동 활용

4. **가중치 누적**
   - Tensor의 `multiply()`와 `add_i()` 조합
   - Backend에서 SIMD 자동 활용

### incremental_forwarding 최적화

1. **Expert 캐싱**
   - Thread-local 메모리 풀로 할당 오버헤드 감소
   - LRU 캐시 관리 최적화

2. **병렬 Expert 처리**
   - OpenMP guided 스케줄링
   - 조건부 병렬화로 작은 작업 오버헤드 방지

3. **Expert 출력 결합**
   - Tensor의 `copyData()`와 `add_i()` 활용
   - Contiguous 메모리에 대해 최적화된 경로

## 성능 이점

### 코드 품질
- **유지보수성**: 플랫폼별 SIMD 코드 제거로 유지보수 용이
- **이식성**: ARM/x86 등 다양한 플랫폼 자동 지원
- **재사용성**: 기존 최적화된 Tensor operation 활용

### 성능
- **SIMD 자동 활용**: Backend에서 플랫폼별 최적화 자동 적용
- **메모리 효율**: Zero-copy와 메모리 풀 활용
- **캐시 효율**: Contiguous 메모리 접근 패턴

## 컴파일 및 테스트

### 컴파일 옵션
```bash
# Android NDK
-march=armv8-a+fp16
-mtune=cortex-a76  
-O3
-fopenmp
-ffast-math
-DENABLE_FP16  # FP16 지원 활성화
```

### 테스트 방법
```bash
# 단위 테스트
cd /workspace
mkdir build && cd build
cmake .. -DENABLE_FP16=ON -DENABLE_OPENMP=ON
make test_moe_layer

# 벤치마크
./test_moe_layer --benchmark
```

## 추가 최적화 가능 영역

1. **FP16 지원 확장**
   - HalfTensor 활용으로 메모리 대역폭 절감
   - ARM FP16 연산 활용

2. **Quantization 통합**
   - INT8/INT4 quantized tensor 지원
   - QInt8Tensor, QInt4Tensor 활용

3. **GPU 오프로딩**
   - OpenCL backend 활용
   - 큰 배치 처리 시 GPU 활용

4. **Dynamic Shape 최적화**
   - 런타임 shape 변경에 대한 효율적 처리
   - 메모리 재할당 최소화

## 결론

Tensor operation 기반 최적화를 통해:
- 코드의 가독성과 유지보수성 향상
- 플랫폼 독립적인 최적화 달성
- Backend의 기존 최적화 자동 활용
- 예상 성능 향상: 2-2.5배 (플랫폼 및 워크로드에 따라 다름)

이 접근 방식은 NNTrainer의 설계 철학에 부합하며, 향후 다른 레이어 최적화에도 적용 가능한 패턴을 제공합니다.