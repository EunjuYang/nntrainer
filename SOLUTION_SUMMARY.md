# 해결 방안: FP16/FP32 Mixed Precision Softmax

## 문제점
`nullptr`을 전달할 때 함수 오버로딩 해결에 모호성이 발생할 수 있는 문제가 있었습니다:
- `softmax_row_inplace(__fp16*, ..., __fp16*)` 
- `softmax_row_inplace(__fp16*, ..., float*)`

`nullptr`을 전달하면 컴파일러가 어떤 함수를 선택해야 할지 알 수 없습니다.

## 해결 방안

### 1. 명확한 함수 이름 사용
FP16 입력과 FP32 sink를 사용하는 함수에 별도의 이름을 부여:
- `softmax_row_inplace_fp16_fp32()` - FP16 입력, FP32 sink (inplace)
- `softmax_row_fp16_fp32()` - FP16 입력, FP32 sink (non-inplace)

### 2. 구현된 함수들

#### FP16 입력 + FP16 sink (템플릿 특수화)
```cpp
template <>
void softmax_row_inplace(__fp16 *qk_out, size_t start_row, size_t end_row,
                         size_t num_heads, __fp16 *sink);

template <>
void softmax_row(__fp16 *qk_out, size_t start_row, size_t end_row,
                 size_t num_heads, __fp16 *sink);
```

#### FP16 입력 + FP32 sink (별도 함수)
```cpp
void softmax_row_inplace_fp16_fp32(__fp16 *qk_out, size_t start_row, 
                                   size_t end_row, size_t num_heads, float *sink);

void softmax_row_fp16_fp32(__fp16 *qk_out, size_t start_row, 
                           size_t end_row, size_t num_heads, float *sink);
```

## 사용 방법

### 직접 호출
```cpp
// FP16 sink 사용
__fp16 *sink_fp16 = ...;
softmax_row_inplace(qk_out, 0, rows, heads, sink_fp16);

// FP32 sink 사용 - 명확한 함수 이름 사용
float *sink_fp32 = ...;
softmax_row_inplace_fp16_fp32(qk_out, 0, rows, heads, sink_fp32);

// Sink 없이 사용
softmax_row_inplace(qk_out, 0, rows, heads, (__fp16*)nullptr);
```

### 선택적: Wrapper 함수 사용
필요한 경우 자동 타입 추론을 위한 wrapper 함수를 만들 수 있습니다:
```cpp
namespace wrapper {
    void softmax_row_inplace(__fp16 *qk_out, size_t start_row, 
                            size_t end_row, size_t num_heads, float *sink) {
        if (sink == nullptr) {
            ::nntrainer::neon::softmax_row_inplace(qk_out, start_row, end_row, 
                                                   num_heads, (__fp16*)nullptr);
        } else {
            ::nntrainer::neon::softmax_row_inplace_fp16_fp32(qk_out, start_row, 
                                                             end_row, num_heads, sink);
        }
    }
}
```

## 장점
1. **컴파일 에러 방지**: 함수 이름이 명확하여 오버로딩 모호성 없음
2. **명확한 의도**: 함수 이름에서 입력/출력 타입이 명확함
3. **유연성**: 필요시 wrapper로 편의성 제공 가능
4. **성능**: 런타임 오버헤드 없음

## 구현 세부사항
- FP32 sink는 반드시 non-null이어야 함 (nullptr 체크 제거)
- FP16 계산은 정밀도를 위해 내부적으로 FP32로 변환
- NEON 벡터화로 8개의 FP16 값을 동시 처리
- Sink 값은 max 계산과 sum 초기화에 포함됨