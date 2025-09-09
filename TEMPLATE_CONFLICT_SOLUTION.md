# 템플릿 충돌 해결 방안

## 문제
```
candidate template ignored: deduced conflicting types for parameter T ('__fp16' vs. 'float')
```
템플릿 파라미터 `T`가 `__fp16`과 `float` 중 무엇인지 결정할 수 없는 문제

## 원인
```cpp
template <typename T>
void softmax_row_inplace(T *qk_out, ..., T *sink);

// 호출 시
__fp16 *data;
float *sink;
softmax_row_inplace(data, ..., sink);  // T = __fp16? float? 충돌!
```

## 해결 방법
명시적인 함수 오버로드를 모든 레벨에 추가

### 1. neon_impl.h / neon_impl_fp16.cpp
```cpp
// 템플릿 특수화 (T = __fp16)
template <>
void softmax_row_inplace(__fp16 *qk_out, ..., __fp16 *sink);

// 명시적 오버로드 (mixed precision)
void softmax_row_inplace(__fp16 *qk_out, ..., float *sink);
```

### 2. arm_compute_backend.h / arm_compute_backend.cpp
```cpp
// 템플릿 선언
template <typename T = float>
void softmax_row_inplace(T *qk_out, ..., T *sink = nullptr);

#ifdef ENABLE_FP16
// 명시적 오버로드 선언
void softmax_row_inplace(__fp16 *qk_out, ..., float *sink);
#endif

// cpp 파일에 구현 추가
void softmax_row_inplace(__fp16 *qk_out, ..., float *sink) {
  neon::softmax_row_inplace(qk_out, ..., sink);
}
```

### 3. cpu_backend.h
```cpp
// 템플릿 선언
template <typename T = float>
extern void softmax_row_inplace(T *qk_out, ..., T *sink = nullptr);

#ifdef ENABLE_FP16
// 명시적 오버로드 선언
extern void softmax_row_inplace(_FP16 *qk_out, ..., float *sink);
#endif
```

## 수정된 파일들
1. `/workspace/nntrainer/tensor/cpu_backend/cpu_backend.h` - 오버로드 선언 추가
2. `/workspace/nntrainer/tensor/cpu_backend/arm/arm_compute_backend.h` - 오버로드 선언 추가
3. `/workspace/nntrainer/tensor/cpu_backend/arm/arm_compute_backend.cpp` - 구현 추가
4. `/workspace/nntrainer/tensor/cpu_backend/arm/neon_impl.h` - 이미 추가됨
5. `/workspace/nntrainer/tensor/cpu_backend/arm/neon_impl_fp16.cpp` - 이미 구현됨

## 사용법
```cpp
#include <cpu_backend.h>

__fp16 *data = ...;
float *sink_fp32 = ...;
__fp16 *sink_fp16 = ...;

// 이제 모두 정상 동작 (컴파일 에러 없음)
softmax_row_inplace(data, 0, 10, 16, sink_fp32);  // 오버로드 버전
softmax_row_inplace(data, 0, 10, 16, sink_fp16);  // 템플릿 특수화
softmax_row_inplace(data, 0, 10, 16, nullptr);    // nullptr OK
```

## 핵심 포인트
- 템플릿과 오버로드를 함께 사용할 때는 모든 레벨에서 일관되게 선언
- Mixed type의 경우 명시적 오버로드로 템플릿 추론 모호성 제거
- 각 백엔드 레벨에서 단순 포워딩만 수행