# NNTrainer CausalLM 애플리케이션 OJT 가이드

## 목차
1. [프로젝트 개요](#프로젝트-개요)
2. [NNTrainer 프레임워크 소개](#nntrainer-프레임워크-소개)
3. [CausalLM 애플리케이션 구조](#causallm-애플리케이션-구조)
4. [파일 구조 상세 분석](#파일-구조-상세-분석)
5. [핵심 컴포넌트 설명](#핵심-컴포넌트-설명)
6. [실행 방법](#실행-방법)
7. [커스텀 레이어 개발 가이드](#커스텀-레이어-개발-가이드)
8. [모델 확장 가이드](#모델-확장-가이드)

---

## 프로젝트 개요

### NNTrainer란?
NNTrainer는 **임베디드 디바이스에서 신경망 모델을 학습할 수 있는 소프트웨어 프레임워크**입니다. 제한된 리소스를 가진 디바이스에서도 효율적으로 모델을 파인튜닝하고 개인화할 수 있도록 설계되었습니다.

### CausalLM 애플리케이션
CausalLM은 NNTrainer를 사용하여 **Transformer 기반 Causal Language Model(인과 언어 모델)**을 실행하는 애플리케이션입니다. 현재는 **추론(Inference) 모드만 지원**하며, 향후 Parameter-efficient Training 모드를 지원할 예정입니다.

### 지원 모델
- **Llama**
- **Qwen3** (1.7b/4b/7b/14b)
- **Qwen3MoE** (30b-A3b)
- 사용자 정의 모델도 커스텀 레이어를 통해 지원 가능

---

## NNTrainer 프레임워크 소개

### 주요 특징
1. **온디바이스 학습**: 서버 없이 디바이스에서 직접 모델 학습 및 파인튜닝
2. **효율적인 리소스 활용**: 제한된 메모리와 연산 자원을 효율적으로 사용
3. **다양한 레이어 지원**: CNN, RNN, Transformer 등 다양한 레이어 제공
4. **C/C++ API**: Tizen, Ubuntu, Android 등 다양한 플랫폼 지원

### 아키텍처
```
[Input] → [Embedding] → [Decoder Block × N] → [RMSNorm] → [LMHead] → [Output]
```

---

## CausalLM 애플리케이션 구조

### 전체 구조도
```
Applications/CausalLM/
├── main.cpp                    # 메인 진입점
├── causal_lm.h/cpp             # 기본 CausalLM 클래스
├── qwen3_causallm.h/cpp        # Qwen3 모델 구현
├── qwen3_moe_causallm.h/cpp    # Qwen3MoE 모델 구현
├── factory.h                   # 팩토리 패턴 구현
├── llm_util.hpp/cpp            # 유틸리티 함수
├── layers/                     # 커스텀 레이어
│   ├── embedding_layer.h/cpp
│   ├── mha_core.h/cpp          # Multi-Head Attention Core
│   ├── rms_norm.h/cpp          # RMS Normalization
│   ├── reshaped_rms_norm.h/cpp # Reshaped RMS Norm (Qwen3)
│   ├── swiglu.h/cpp            # SwiGLU 활성화 함수
│   └── tie_word_embedding.h/cpp # Word Embedding 공유
└── res/                        # 모델 리소스
    ├── qwen3-4b/
    │   ├── config.json
    │   ├── generation_config.json
    │   ├── nntr_config.json
    │   └── nntr_qwen3_4b_fp32.bin
    └── qwen3-30b-a3b/
```

---

## 파일 구조 상세 분석

### 1. main.cpp
**역할**: 애플리케이션의 진입점

**주요 기능**:
- 모델 팩토리에 다양한 CausalLM 모델 등록
- 명령줄 인자 파싱 (모델 경로, 입력 프롬프트)
- JSON 설정 파일 로드 (config.json, generation_config.json, nntr_config.json)
- 모델 초기화 및 실행

**코드 흐름**:
```cpp
1. Factory에 모델 등록 (LlamaForCausalLM, Qwen3ForCausalLM, Qwen3MoeForCausalLM)
2. 설정 파일 로드
3. Factory를 통해 모델 인스턴스 생성
4. 모델 초기화 및 가중치 로드
5. 추론 실행
```

### 2. causal_lm.h/cpp
**역할**: CausalLM의 기본 클래스 구현

**핵심 멤버 변수**:
- `ModelHandle model`: NNTrainer 모델 핸들
- `tokenizer`: 토크나이저 인스턴스
- `NUM_VOCAB`, `DIM`, `NUM_LAYERS` 등 모델 하이퍼파라미터
- `BATCH_SIZE`, `MAX_SEQ_LEN`, `NUM_TO_GENERATE` 등 실행 파라미터

**주요 메서드**:

#### `CausalLM::setupParameters()`
- JSON 설정 파일에서 모델 파라미터 추출
- NNTrainer 설정 파라미터 초기화
- Generation 설정 파라미터 초기화

#### `CausalLM::constructModel()`
모델 구조를 레이어 단위로 구성:
1. **Input Layer**: 입력 레이어 생성
2. **Embedding Layer**: 토큰 임베딩 레이어
3. **Transformer Decoder Blocks**: N개의 디코더 블록 생성
4. **RMSNorm**: 최종 정규화 레이어
5. **LMHead**: 언어 모델 헤드 (출력 레이어)

#### `CausalLM::createTransformerDecoderBlock()`
Transformer 디코더 블록 구성:
```
Input
  ↓
RMSNorm (Attention Norm)
  ↓
Attention (Multi-Head Attention)
  ↓
Addition (Residual Connection)
  ↓
RMSNorm (FFN Norm)
  ↓
MLP (Feed Forward Network with SwiGLU)
  ↓
Addition (Residual Connection)
  ↓
Output
```

#### `CausalLM::createAttention()`
Multi-Head Attention 구조:
1. **Q, K, V Projection**: Fully Connected 레이어로 Query, Key, Value 생성
2. **MHA Core**: 실제 어텐션 연산 수행 (RoPE, KV-Cache 포함)
3. **O Projection**: Output projection 레이어

#### `CausalLM::createMlp()`
Feed Forward Network 구조:
1. **FFN Up**: Up projection 레이어
2. **FFN Gate**: Gate projection 레이어
3. **SwiGLU**: SwiGLU 활성화 함수 적용
4. **FFN Down**: Down projection 레이어

#### `CausalLM::run()`
추론 실행 메인 로직:
1. **Input Preparation**: 프롬프트 토크나이징 및 입력 준비
2. **Prefill Phase**: 초기 시퀀스에 대한 병렬 처리
3. **Token Generation Phase**: 토큰을 하나씩 생성 (Auto-regressive)
4. **Output Decoding**: 생성된 토큰을 디코딩하여 출력

**Prefill vs Generation**:
- **Prefill**: 초기 입력 시퀀스를 한 번에 처리 (병렬 처리 가능)
- **Generation**: 생성된 토큰을 하나씩 처리 (순차 처리)

### 3. qwen3_causallm.h/cpp
**역할**: Qwen3 모델 특화 구현

**기본 CausalLM과의 차이점**:
- `createAttention()` 오버라이드: Q와 K에 대해 **ReshapedRMSNorm** 적용
- `registerCustomLayers()` 오버라이드: `ReshapedRMSNormLayer` 등록

**Qwen3의 특징**:
- Q와 K projection 후에 각각 RMSNorm을 적용
- Reshaped RMSNorm은 헤드 차원 단위로 정규화 수행

### 4. factory.h
**역할**: 팩토리 패턴 구현으로 다양한 모델 타입 지원

**설계 패턴**: Singleton + Factory Pattern

**사용 방법**:
```cpp
// 모델 등록
Factory::Instance().registerModel("ModelName", creator_function);

// 모델 생성
auto model = Factory::Instance().create("ModelName", cfg, gen_cfg, nntr_cfg);
```

### 5. llm_util.hpp/cpp
**역할**: LLM 관련 유틸리티 함수 제공

**주요 함수**:

#### `generate_multi_tokens()`
- Logits에서 여러 토큰 생성
- Repetition penalty, bad words penalty 적용

#### `applyRepetitionPenalty()`
- 반복 토큰에 대한 페널티 적용

#### `applyBadWordsPenalty()`
- 금지 단어에 대한 페널티 적용

#### `applyTKP()`
- Temperature, Top-K, Top-P 샘플링 적용

#### `withKey()`
- 레이어 속성 문자열 생성 헬퍼 함수

---

## 핵심 컴포넌트 설명

### 1. 커스텀 레이어

#### EmbeddingLayer (`layers/embedding_layer.h`)
**역할**: 토큰 ID를 벡터 임베딩으로 변환

**특징**:
- `in_dim`: 어휘 크기 (vocab_size)
- `out_dim`: 임베딩 차원 (hidden_size)
- `incremental_forwarding()` 지원: 토큰 단위 생성 시 사용

#### MHACoreLayer (`layers/mha_core.h`)
**역할**: Multi-Head Attention의 핵심 연산 수행

**주요 기능**:
1. **RoPE (Rotary Position Embedding)**: 위치 인코딩 적용
2. **KV-Cache**: Key-Value 캐시 관리로 추론 속도 향상
3. **Sliding Window Attention**: 긴 시퀀스 처리 지원
4. **GQA (Grouped Query Attention)**: 메모리 효율적인 어텐션

**KV-Cache 동작**:
- Prefill 단계: 전체 시퀀스의 K, V를 캐시에 저장
- Generation 단계: 새로운 토큰의 K, V만 계산하여 캐시에 추가

#### RMSNormLayer (`layers/rms_norm.h`)
**역할**: Root Mean Square Normalization 수행

**수식**:
```
RMSNorm(x) = x / sqrt(mean(x^2) + eps)
```

#### ReshapedRMSNormLayer (`layers/reshaped_rms_norm.h`)
**역할**: Qwen3에서 사용하는 Reshaped RMS Normalization

**차이점**: 헤드 차원 단위로 정규화 수행

#### SwiGLULayer (`layers/swiglu.h`)
**역할**: SwiGLU 활성화 함수 적용

**수식**:
```
SwiGLU(x, gate) = Swish(x) * gate
Swish(x) = x * sigmoid(x)
```

#### TieWordEmbedding (`layers/tie_word_embedding.h`)
**역할**: Word Embedding과 LMHead의 가중치 공유

**효과**: 모델 파라미터 수 감소 및 메모리 절약

### 2. 토크나이저
**역할**: 텍스트와 토큰 ID 간 변환

**사용 라이브러리**: HuggingFace Tokenizers (C++ 바인딩)

**주요 기능**:
- `Encode()`: 텍스트 → 토큰 ID 리스트
- `Decode()`: 토큰 ID 리스트 → 텍스트

### 3. 설정 파일

#### config.json
HuggingFace 모델 설정 파일:
- `vocab_size`: 어휘 크기
- `hidden_size`: 은닉층 차원
- `num_hidden_layers`: 레이어 수
- `num_attention_heads`: 어텐션 헤드 수
- `intermediate_size`: FFN 중간 차원
- `max_position_embeddings`: 최대 위치 임베딩
- `rope_theta`: RoPE theta 값
- `rms_norm_eps`: RMSNorm epsilon 값

#### generation_config.json
생성 관련 설정:
- `eos_token_id`: 종료 토큰 ID
- `bos_token_id`: 시작 토큰 ID
- `top_k`: Top-K 샘플링
- `top_p`: Top-P (nucleus) 샘플링
- `temperature`: 온도 파라미터
- `do_sample`: 샘플링 여부

#### nntr_config.json
NNTrainer 실행 설정:
```json
{
    "model_tensor_type": "FP32-FP32",      // 텐서 타입
    "model_file_name": "model.bin",        // 가중치 파일명
    "embedding_dtype": "FP32",             // 임베딩 데이터 타입
    "fc_layer_dtype": "FP32",              // FC 레이어 데이터 타입
    "batch_size": 1,                       // 배치 크기
    "init_seq_len": 1024,                  // 초기 시퀀스 길이
    "max_seq_len": 2048,                   // 최대 시퀀스 길이
    "num_to_generate": 512,                // 생성할 토큰 수
    "tokenizer_file": "path/to/tokenizer.json",
    "sample_input": "default prompt",
    "fsu": false,                          // FSU (Flash Storage Unit) 사용 여부
    "fsu_lookahead": 2                     // FSU lookahead 값
}
```

---

## 실행 방법

### 1. 빌드
```bash
cd /workspace
meson build
ninja -C build
```

### 2. 모델 준비
HuggingFace에서 모델을 다운로드하여 `res/{model_name}/` 디렉토리에 배치:
- `config.json`
- `generation_config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `vocab.json`
- `nntr_config.json`
- NNTrainer 가중치 바이너리 파일

### 3. 가중치 변환
HuggingFace 모델 가중치를 NNTrainer 형식으로 변환:
```bash
cd res/{model_name}/
python weight_converter.py
```

### 4. 실행
```bash
cd build/Applications/CausalLM
./nntr_causallm /path/to/res/qwen3-4b/ [input_prompt]
```

**예시**:
```bash
./nntr_causallm /tmp/nntrainer/Applications/CausalLM/res/qwen3-4b/ \
  "<|im_start|>user\nWhat is AI?<|im_end|>\n<|im_start|>assistant\n"
```

---

## 커스텀 레이어 개발 가이드

### 레이어 개발 단계

#### 1. 헤더 파일 작성 (`layers/my_layer.h`)
```cpp
#include <layer_impl.h>

namespace causallm {

class MyLayer : public nntrainer::LayerImpl {
public:
    MyLayer();
    ~MyLayer() = default;
    
    void finalize(nntrainer::InitLayerContext &context) override;
    void forwarding(nntrainer::RunLayerContext &context, bool training) override;
    void incremental_forwarding(nntrainer::RunLayerContext &context,
                               unsigned int from, unsigned int to,
                               bool training) override;
    void setProperty(const std::vector<std::string> &values) override;
    const std::string getType() const override { return MyLayer::type; }
    
    inline static const std::string type = "my_layer";
    
private:
    // Properties and internal state
};

}
```

#### 2. 구현 파일 작성 (`layers/my_layer.cpp`)
- `finalize()`: 레이어 초기화, 텐서 차원 설정
- `forwarding()`: 학습/추론 시 순전파
- `incremental_forwarding()`: 토큰 단위 생성 시 사용

#### 3. 레이어 등록
`CausalLM::registerCustomLayers()`에서 등록:
```cpp
app_context->registerFactory(nntrainer::createLayer<causallm::MyLayer>);
```

#### 4. 빌드 시스템에 추가
`layers/meson.build`에 레이어 추가

### 레이어 속성 정의
`causallm_common_properties.h` 또는 레이어 헤더에 속성 클래스 정의:
```cpp
namespace props {
class MyProperty : public nntrainer::Property<int> {
public:
    MyProperty(int value = 0) { set(value); };
    static constexpr const char *key = "my_property";
    using prop_tag = nntrainer::int_prop_tag;
};
}
```

---

## 모델 확장 가이드

### 새로운 모델 추가하기

#### 1. 모델 클래스 생성
```cpp
// my_model_causallm.h
#include <causal_lm.h>

namespace causallm {
class MyModelCausalLM : public CausalLM {
public:
    MyModelCausalLM(json &cfg, json &generation_cfg, json &nntr_cfg);
    
    // 필요한 메서드 오버라이드
    std::vector<LayerHandle> createTransformerDecoderBlock(...) override;
    void registerCustomLayers() override;
};
}
```

#### 2. Factory에 등록
`main.cpp`에서 등록:
```cpp
Factory::Instance().registerModel(
    "MyModelForCausalLM",
    [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<causallm::MyModelCausalLM>(
            cfg, generation_cfg, nntr_cfg);
    });
```

#### 3. 모델 특화 레이어 구현
모델에 특화된 레이어가 필요한 경우 커스텀 레이어 개발

### 모델별 차이점 처리

#### Attention 구조 차이
- `createAttention()` 오버라이드하여 모델별 어텐션 구조 구현

#### Normalization 차이
- LayerNorm vs RMSNorm
- Reshaped RMSNorm (Qwen3)

#### Activation 함수 차이
- ReLU, GELU, SwiGLU 등

---

## 주요 개념 정리

### 1. Incremental Inference
토큰을 하나씩 생성하는 추론 방식:
- Prefill: 초기 시퀀스 병렬 처리
- Generation: 토큰 단위 순차 처리
- KV-Cache를 활용하여 이전 계산 결과 재사용

### 2. KV-Cache
Key-Value 캐시로 추론 속도 향상:
- Prefill 단계에서 전체 K, V 계산 및 저장
- Generation 단계에서 새로운 토큰의 K, V만 계산

### 3. RoPE (Rotary Position Embedding)
회전 위치 임베딩:
- 상대적 위치 정보를 인코딩
- `rope_theta` 파라미터로 주기 조절

### 4. GQA (Grouped Query Attention)
그룹화된 쿼리 어텐션:
- 여러 Query 헤드가 하나의 Key/Value 헤드 공유
- 메모리 사용량 감소

### 5. Sliding Window Attention
슬라이딩 윈도우 어텐션:
- 긴 시퀀스 처리 시 윈도우 크기 제한
- 메모리 효율성 향상

---

## 디버깅 팁

### 1. 모델 구조 확인
```cpp
#ifdef DEBUG
model->summarize(std::cout, ML_TRAIN_SUMMARY_MODEL);
#endif
```

### 2. 레이어 이름 규칙
- `layer{N}_attention_norm`: N번째 레이어의 어텐션 정규화
- `layer{N}_attention`: N번째 레이어의 어텐션
- `layer{N}_decoder_output`: N번째 레이어의 출력

### 3. 텐서 차원 확인
- Input: `[batch, 1, seq_len]` 또는 `[batch, 1, 1]` (incremental)
- Embedding output: `[batch, 1, seq_len, hidden_size]`
- Attention output: `[batch, 1, seq_len, hidden_size]`

### 4. 일반적인 오류
- **차원 불일치**: 레이어 간 입력/출력 차원 확인
- **가중치 로드 실패**: 가중치 파일 경로 및 형식 확인
- **메모리 부족**: `MAX_SEQ_LEN`, `BATCH_SIZE` 조정

---

## 성능 최적화

### 1. 메모리 최적화
- `fsu` 옵션 사용: Flash Storage Unit 활용
- `fsu_lookahead` 조정: Lookahead 크기 최적화
- 데이터 타입 선택: FP16 사용 시 메모리 절약

### 2. 속도 최적화
- KV-Cache 활용
- Incremental inference 사용
- 배치 크기 조정

### 3. 모델 최적화
- Weight sharing (Tie Word Embeddings)
- GQA 사용
- Sliding Window Attention

---

## 참고 자료

### 공식 문서
- [NNTrainer GitHub](https://github.com/nnstreamer/nntrainer)
- [Getting Started Guide](../getting-started.md)
- [How to Create Model](../how-to-create-model.md)

### 관련 논문
- [A New Frontier of AI: On-Device AI Training and Personalization](https://dl.acm.org/doi/abs/10.1145/3639477.3639716)
- [NNTrainer: Light-Weight On-Device Training Framework](https://arxiv.org/pdf/2206.04688.pdf)

### 코드 참고
- HuggingFace Transformers: [Qwen3 Modeling](https://github.com/huggingface/transformers/blob/v4.52.3/src/transformers/models/qwen3/modeling_qwen3.py)

---

## FAQ

### Q1: 새로운 모델을 추가하려면?
A: `CausalLM`을 상속받아 모델 클래스를 만들고, Factory에 등록하세요. 모델별 차이점은 가상 함수를 오버라이드하여 구현합니다.

### Q2: 커스텀 레이어를 만들려면?
A: `nntrainer::LayerImpl`을 상속받아 레이어를 구현하고, `registerCustomLayers()`에서 등록하세요.

### Q3: 추론 속도를 높이려면?
A: KV-Cache를 활용하고, 적절한 배치 크기를 설정하며, FSU 옵션을 사용하세요.

### Q4: 메모리 부족 문제는?
A: `MAX_SEQ_LEN`을 줄이거나, `fsu` 옵션을 사용하거나, FP16 데이터 타입을 사용하세요.

---

## 결론

이 문서는 NNTrainer CausalLM 애플리케이션의 구조와 사용법을 상세히 설명합니다. 새로운 모델을 추가하거나 커스텀 레이어를 개발할 때 이 가이드를 참고하시기 바랍니다.

추가 질문이나 개선 사항이 있으면 이슈를 등록하거나 PR을 제출해 주세요.
