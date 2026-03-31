# Sentence Transformer API Support

CausalLM API에 Sentence Transformer(임베딩) 모델 실행 기능을 추가한 변경사항을 설명합니다.

## 배경

기존 CausalLM API(`causal_lm_api.h`)는 텍스트 생성(CausalLM) 모델만 지원했습니다.
`runModel()` 내부에서 `dynamic_cast<CausalLM*>`를 통해 출력 텍스트를 가져오는 구조였기 때문에,
SentenceTransformer 모델을 로드하더라도 임베딩 결과를 추출할 수 없었습니다.

SentenceTransformer 모델 클래스와 구현체(Qwen3Embedding, Qwen2Embedding, EmbeddingGemma)는
이미 `models/` 디렉토리에 존재했지만, API 레이어에서 이를 활용하는 코드가 빠져있었습니다.

## 변경 파일 목록

| 파일 | 변경 요약 |
|------|----------|
| `api/causal_lm_api.h` | 임베딩 모델 타입, 결과 구조체, 새 API 함수 선언 |
| `models/transformer.h` | `getBatchSize()`, `getEmbeddingDim()` getter 추가 |
| `api/causal_lm_api.cpp` | Factory 등록, architecture resolution, `runEmbeddingModel()` 구현 |
| `api/test_api.cpp` | 임베딩 모델 테스트 지원 |
| `meson.build` | 코어 라이브러리 / API 라이브러리 / 앱 빌드 분리 |

## 변경 내용 상세

### 1. 공개 API 확장 (`causal_lm_api.h`)

#### ModelType enum 확장

```c
typedef enum {
  CAUSAL_LM_MODEL_QWEN3_0_6B = 0,
  CAUSAL_LM_MODEL_QWEN3_EMBEDDING = 1,  // NEW
  CAUSAL_LM_MODEL_QWEN2_EMBEDDING = 2,  // NEW
  CAUSAL_LM_MODEL_GEMMA_EMBEDDING = 3,  // NEW
} ModelType;
```

#### 임베딩 결과 구조체

```c
typedef struct {
  float *embeddings;           // 임베딩 벡터 (batch_size * embedding_dim)
  unsigned int embedding_dim;  // 임베딩 차원
  unsigned int batch_size;     // 배치 크기
} EmbeddingResult;
```

#### 새 API 함수

```c
// 임베딩 모델 실행 (Sentence Transformer용)
ErrorCode runEmbeddingModel(const char *inputTextPrompt,
                            EmbeddingResult *result);

// 임베딩 결과 메모리 해제
void freeEmbeddingResult(EmbeddingResult *result);
```

> **설계 판단**: `runModel()`을 확장하지 않고 별도 함수를 추가한 이유:
> - CausalLM은 `const char**`(텍스트), SentenceTransformer는 `float*`(임베딩)를 반환 -> 출력 타입이 근본적으로 다름
> - C API에서 union/void*로 반환하면 타입 안전성이 떨어짐
> - 기존 `runModel()` 사용자에 대한 하위 호환성 유지

### 2. Transformer base getter 추가 (`transformer.h`)

API 레이어에서 임베딩 차원과 배치 크기에 접근하기 위해 public getter를 추가했습니다.

```cpp
unsigned int getBatchSize() const { return BATCH_SIZE; }
int getEmbeddingDim() const { return DIM; }
```

### 3. API 구현 (`causal_lm_api.cpp`)

#### 3-1. 임베딩 모델 Factory 등록

`main.cpp`에만 있던 임베딩 모델 Factory 등록을 API에도 추가:

```cpp
// register_models() 내부에 추가
causallm::Factory::Instance().registerModel("Qwen3Embedding", ...);
causallm::Factory::Instance().registerModel("Qwen2Embedding", ...);
causallm::Factory::Instance().registerModel("EmbeddingGemma", ...);
```

#### 3-2. Architecture Resolution

`main.cpp`의 `resolve_architecture()` 로직을 API에 포팅.
`nntr_cfg["model_type"]`이 `"embedding"`이면 architecture 이름을 임베딩 클래스로 변환:

| config.json architecture | 변환 결과 |
|-------------------------|----------|
| `Qwen3ForCausalLM` | `Qwen3Embedding` |
| `Gemma3ForCausalLM` / `Gemma3TextModel` | `EmbeddingGemma` |
| `Qwen2Model` | `Qwen2Embedding` |

#### 3-3. `loadModel()` 수정

- architecture resolution 호출 추가
- `module_config_path`를 모델 디렉토리 기준 상대경로로 resolve

#### 3-4. `runModel()` 수정

SentenceTransformer 모델에 대해 `runModel()` 호출 시 명시적 에러 반환:
```
"runModel() is not supported for this model type. Use runEmbeddingModel() for embedding models."
```

#### 3-5. `runEmbeddingModel()` 구현

```
1. dynamic_cast<SentenceTransformer*>로 모델 타입 확인
2. SentenceTransformer::encode() 호출 -> vector<float*> 반환
3. EmbeddingResult에 결과 복사 (batch_size * embedding_dim)
4. encode() 원본 메모리 해제
```

#### 3-6. `getPerformanceMetrics()` 수정

기존: `dynamic_cast<CausalLM*>`에만 의존 -> SentenceTransformer에서 실패
변경: Transformer base의 `getPerformanceMetrics()`를 직접 사용, CausalLM인 경우에만 `hasRun()` 체크

### 4. 테스트 앱 (`test_api.cpp`)

모델 이름에 `EMBEDDING`이 포함되면 자동으로 임베딩 모드로 분기:

- **CausalLM 모델**: 기존대로 `runModel()` -> 텍스트 출력
- **임베딩 모델**: `runEmbeddingModel()` -> 임베딩 벡터 출력 (처음 10개 차원 표시)

```bash
# CausalLM (기존)
./test_api QWEN3-0.6B "Tell me a joke" 1 W4A32

# Sentence Transformer (신규)
./test_api QWEN3-EMBEDDING "What is AI?" 0 UNKNOWN
```

### 5. 빌드 구조 분리 (`meson.build`)

기존에는 API 코드와 모델 코드가 하나의 `libcausallm.so`에 섞여있었습니다.
이를 4개의 독립 빌드 타겟으로 분리했습니다:

```
libcausallm.so        코어 라이브러리 (모델, 트랜스포머, 토크나이저)
  |
  +-- libcausallm_api.so   API 라이브러리 (C API 레이어)
  |     |
  |     +-- test_api        API 테스트 앱
  |
  +-- nntr_causallm        Main CLI 앱 (Factory 직접 사용)
```

| 타겟 | 소스 | 의존성 |
|------|------|--------|
| `libcausallm.so` | tokenizer, llm_util, models | nntrainer, layers |
| `libcausallm_api.so` | causal_lm_api.cpp, model_config.cpp | libcausallm |
| `nntr_causallm` | main.cpp | libcausallm |
| `test_api` | test_api.cpp | libcausallm_api |

## API 사용 예시

### CausalLM (텍스트 생성)

```c
Config config = {true, false, true};
setOptions(config);
loadModel(CAUSAL_LM_BACKEND_CPU, CAUSAL_LM_MODEL_QWEN3_0_6B,
          CAUSAL_LM_QUANTIZATION_W4A32);

const char *output = NULL;
runModel("Tell me a joke", &output);
printf("Output: %s\n", output);
```

### Sentence Transformer (임베딩)

```c
Config config = {false, false, false};
setOptions(config);
loadModel(CAUSAL_LM_BACKEND_CPU, CAUSAL_LM_MODEL_QWEN3_EMBEDDING,
          CAUSAL_LM_QUANTIZATION_UNKNOWN);

EmbeddingResult result;
runEmbeddingModel("What is machine learning?", &result);

printf("Embedding dim: %u, batch: %u\n", result.embedding_dim, result.batch_size);
for (unsigned int i = 0; i < result.embedding_dim && i < 5; i++) {
    printf("  [%u] = %f\n", i, result.embeddings[i]);
}

freeEmbeddingResult(&result);
```

## 에러 처리

| 상황 | 반환 에러코드 |
|------|-------------|
| 임베딩 모델에 `runModel()` 호출 | `CAUSAL_LM_ERROR_INVALID_PARAMETER` |
| CausalLM 모델에 `runEmbeddingModel()` 호출 | `CAUSAL_LM_ERROR_INVALID_PARAMETER` |
| 모델 미로드 상태에서 실행 | `CAUSAL_LM_ERROR_NOT_INITIALIZED` |
| 임베딩 추론 중 예외 발생 | `CAUSAL_LM_ERROR_INFERENCE_FAILED` |
