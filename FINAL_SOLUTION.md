# 최종 해결 방안: 함수 오버로딩을 통한 Mixed Precision Softmax

## 구현 방식
sink의 타입에 따라 함수를 오버로딩하여 사용자가 간단하게 사용할 수 있도록 구현했습니다.

## 구현된 함수들

### 1. FP16 입력 + FP16 sink (템플릿 특수화)
```cpp
template <>
void softmax_row_inplace(__fp16 *qk_out, size_t start_row, size_t end_row,
                         size_t num_heads, __fp16 *sink);

template <>
void softmax_row(__fp16 *qk_out, size_t start_row, size_t end_row,
                 size_t num_heads, __fp16 *sink);
```

### 2. FP16 입력 + FP32 sink (함수 오버로딩)
```cpp
void softmax_row_inplace(__fp16 *qk_out, size_t start_row, size_t end_row,
                         size_t num_heads, float *sink);

void softmax_row(__fp16 *qk_out, size_t start_row, size_t end_row,
                 size_t num_heads, float *sink);
```

## 사용법 - 매우 간단!

```cpp
__fp16 *qk_out = ...;     // FP16 attention scores
__fp16 *sink_fp16 = ...;  // FP16 sink
float *sink_fp32 = ...;   // FP32 sink

// 컴파일러가 자동으로 올바른 함수 선택
softmax_row_inplace(qk_out, 0, rows, heads, sink_fp16);  // FP16 버전
softmax_row_inplace(qk_out, 0, rows, heads, sink_fp32);  // FP32 버전

// nullptr도 처리 가능
softmax_row_inplace(qk_out, 0, rows, heads, (float*)nullptr);
```

## 장점

1. **사용 편의성**: 사용자가 함수 이름을 구분할 필요 없음
2. **타입 안정성**: 컴파일러가 타입에 따라 자동 선택
3. **일관된 인터페이스**: 모든 softmax 함수가 동일한 이름 사용
4. **nullptr 지원**: 두 버전 모두 nullptr 처리 가능

## nullptr 처리

```cpp
// FP32 sink 버전
void softmax_row_inplace(__fp16 *qk_out, ..., float *sink) {
  if (sink == nullptr) {
    // sink 없이 일반 softmax 수행
    return softmax_row_inplace_no_sink(qk_out, ...);
  } else {
    // FP32 sink 포함한 softmax 수행
    return softmax_row_inplace_with_fp32_sink(qk_out, ..., sink);
  }
}
```

## 내부 구현 구조

```
사용자 API (오버로딩)
├── softmax_row_inplace(__fp16*, ..., __fp16*)  [템플릿 특수화]
│   ├── softmax_row_inplace_no_sink()           [sink == nullptr]
│   └── softmax_row_inplace_with_fp16_sink()    [sink != nullptr]
│
└── softmax_row_inplace(__fp16*, ..., float*)   [함수 오버로딩]
    ├── softmax_row_inplace_no_sink()           [sink == nullptr]
    └── softmax_row_inplace_with_fp32_sink()    [sink != nullptr]
```

## 실제 사용 예시

```cpp
void attention_layer(__fp16 *qk_scores, float *attention_sink,
                    size_t seq_len, size_t num_heads) {
    // 단순히 호출하면 컴파일러가 float* 타입을 보고
    // 자동으로 mixed precision 버전 선택
    softmax_row_inplace(qk_scores, 0, seq_len, num_heads, attention_sink);
    
    // FP16 sink를 사용하고 싶다면
    __fp16 sink_fp16[num_heads];
    // ... initialize sink_fp16 ...
    softmax_row_inplace(qk_scores, 0, seq_len, num_heads, sink_fp16);
}
```

## 결론
함수 오버로딩을 통해 사용자 코드를 간단하게 유지하면서도 타입 안정성을 보장하는 구현을 완성했습니다. 사용자는 sink의 타입만 결정하면 컴파일러가 자동으로 적절한 함수를 선택합니다.