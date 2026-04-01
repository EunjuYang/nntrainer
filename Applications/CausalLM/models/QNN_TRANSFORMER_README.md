# QNN Transformer 모델 가이드

## 개요

`QNNTransformer`는 QNN (Qualcomm Neural Network) 바이너리를 사용하여 Transformer 모델을 추론하는 클래스입니다. QNN SDK로 컴파일된 모델 바이너리(`.bin`)를 로드하고, nntrainer 프레임워크의 레이어 시스템을 통해 실행합니다.

주요 특징:
- **Prefill/Generation 이중 모델 구조**: 프롬프트 처리용(Prefill)과 토큰 생성용(Generation) 모델을 분리하여 최적화
- **Config 기반 자동 설정**: 3개의 JSON 설정 파일에서 모든 파라미터를 자동으로 읽어옴
- **QNN 바이너리 직접 실행**: `qnn_graph` 레이어를 통해 QNN 바이너리 그래프를 직접 실행

---

## 아키텍처

### 클래스 구조

```
Transformer (transformer.h)
    │
QNNTransformer (qnn_transformer.h)
    │
    ├── prefill_model   (ModelHandle)  ← 프롬프트 전체 처리
    └── generation_model (ModelHandle) ← 토큰 단위 생성
```

`QNNTransformer`는 `Transformer` 베이스 클래스를 상속하며, 기존 Transformer의 레이어별 구성(Attention, MLP 등) 대신 **QNN 바이너리 그래프 전체를 하나의 레이어로 실행**합니다.

### 모델 구조 (Prefill / Generation 공통)

```
[Embedding Layer]  ─────────────────┐
                                    │
[Non-Embedding Input Layers] ───────┤  (position_ids, attention_mask 등)
                                    │
                              [QNN Graph Layer]  ← QNN 바이너리 실행
                                    │
                               [Output]
```

---

## 모델 동작 흐름

`QNNTransformer`의 전체 실행 흐름은 다음과 같습니다:

```
1. 생성자
   └── setupParameters(cfg, generation_cfg, nntr_cfg)
       └── 3개 JSON 파일에서 모든 파라미터 자동 로드

2. initialize()
   ├── Engine에 QNN Context 등록
   │   └── Engine::registerContext("libqnn_context.so", "")
   │
   ├── Prefill 모델 구성
   │   ├── Embedding 레이어 추가 (input_shape: 1:sequence_length)
   │   ├── Non-Embedding Input 레이어들 추가
   │   ├── QNN Graph 레이어 추가 (prefill 그래프)
   │   ├── 모델 속성 설정 (batch_size, model_tensor_type 등)
   │   ├── compile(INFERENCE)
   │   └── initialize(INFERENCE)
   │
   └── Generation 모델 구성
       ├── Embedding 레이어 추가 (input_shape: 1:1)
       ├── Non-Embedding Input 레이어들 추가
       ├── QNN Graph 레이어 추가 (generation 그래프)
       ├── 모델 속성 설정
       ├── compile(INFERENCE)
       └── initialize(INFERENCE)

3. load_weight(weight_path)
   ├── prefill_model.load(model_path, QNN)    ← QNN 바이너리 로드
   ├── prefill_model.load(embedding_path)      ← 임베딩 가중치 로드
   ├── prefill_model.allocate()
   │
   ├── generation_model.load(model_path, QNN)
   ├── generation_model.load(embedding_path)
   └── generation_model.allocate()

4. run(prompt)
   └── Prefill → Generation 반복으로 텍스트 생성
```

---

## Config 파일 자동 읽기 메커니즘

`QNNTransformer`는 생성자에서 `setupParameters()` 메서드를 호출하여 **3개의 JSON 설정 파일에서 모든 파라미터를 자동으로 읽어옵니다**. QNN SDK로 모델을 컴파일하면 생성되는 config 파일들의 값을 그대로 사용할 수 있도록 설계되었습니다.

### 생성자에서의 자동 로드

```cpp
QNNTransformer(json &cfg, json &generation_cfg, json &nntr_cfg) :
  Transformer(cfg, generation_cfg, nntr_cfg, ModelType::MODEL) {
  setupParameters(cfg, generation_cfg, nntr_cfg);  // ← 자동 파라미터 로드
}
```

### 3개 JSON 설정 파일

#### 1) `config.json` - QNN 그래프 및 모델 구조 설정

QNN SDK로 모델을 컴파일할 때 생성되는 설정 파일입니다. Prefill/Generation 그래프의 입출력 사양과 모델 아키텍처 파라미터를 포함합니다.

**Prefill 그래프 설정:**

| JSON 키 | 타입 | 설명 |
|---------|------|------|
| `prefill_graph_name` | string | Prefill QNN 그래프 이름 |
| `prefill_input_names` | string | 입력 텐서 이름들 (쉼표 구분) |
| `prefill_output_names` | string | 출력 텐서 이름들 |
| `prefill_in_quant` | string | 입력 양자화 파라미터 |
| `prefill_out_quant` | string | 출력 양자화 파라미터 |
| `prefill_in_dim` | string | 입력 차원 |
| `prefill_out_dim` | string | 출력 차원 |
| `prefill_in_data_format` | string | 입력 데이터 형식 |
| `prefill_out_data_format` | string | 출력 데이터 형식 |
| `prefill_out_tensor_format` | string | 출력 텐서 형식 |
| `prefill_non_embed_input_names` | string[] | Embedding 외 추가 입력 이름 목록 |
| `prefill_non_embed_input_dims` | string[] | Embedding 외 추가 입력 차원 목록 |

**Generation 그래프 설정:**

| JSON 키 | 타입 | 설명 |
|---------|------|------|
| `generation_graph_name` | string | Generation QNN 그래프 이름 |
| `generation_input_names` | string | 입력 텐서 이름들 |
| `generation_output_names` | string | 출력 텐서 이름들 |
| `generation_in_quant` | string | 입력 양자화 파라미터 |
| `generation_out_quant` | string | 출력 양자화 파라미터 |
| `generation_in_dim` | string | 입력 차원 |
| `generation_out_dim` | string | 출력 차원 |
| `generation_in_data_format` | string | 입력 데이터 형식 |
| `generation_out_data_format` | string | 출력 데이터 형식 |
| `generation_out_tensor_format` | string | 출력 텐서 형식 |
| `generation_non_embed_input_names` | string[] | Embedding 외 추가 입력 이름 목록 |
| `generation_non_embed_input_dims` | string[] | Embedding 외 추가 입력 차원 목록 |

**모델 아키텍처 파라미터:**

| JSON 키 | 타입 | 설명 |
|---------|------|------|
| `num_hidden_layers` | int | Transformer 히든 레이어 수 |
| `max_window_layers` | int | 최대 윈도우 레이어 수 |
| `hidden_size` | int | 히든 차원 크기 |
| `sequence_length` | int | Prefill 시퀀스 길이 |
| `vocab_size` | int | 어휘 크기 |
| `max_seq_len` | int | 최대 시퀀스 길이 |
| `sliding_window` | int | 슬라이딩 윈도우 크기 |
| `local_rope_theta` | float | 로컬 RoPE theta 값 |
| `rope_theta` | float | RoPE theta 값 |
| `context_size` | int | 컨텍스트 크기 |
| `pos_dim` | int | 위치 임베딩 차원 |
| `head_dim` | int | 어텐션 헤드 차원 |
| `lora_sizes` | int[] | LoRA 크기 목록 |

#### 2) `generation_config.json` - 텍스트 생성 파라미터

토큰 생성 시 사용되는 샘플링 및 디코딩 설정입니다.

| JSON 키 | 타입 | 설명 |
|---------|------|------|
| `padding_token` | int | 패딩 토큰 ID |
| `eos_token_id` | int | 종료 토큰 ID |
| `temperature` | float | 샘플링 온도 |
| `top_k` | int | Top-K 샘플링 값 |
| `top_p` | float | Top-P (Nucleus) 샘플링 값 |
| `repetition_penalty` | float | 반복 패널티 |
| `logit_scale` | float | 로짓 스케일 값 |
| `logit_offset` | int | 로짓 오프셋 값 |

#### 3) `nntr_config.json` - nntrainer 경로 설정

nntrainer 프레임워크에서 사용하는 파일 경로 설정입니다.

| JSON 키 | 타입 | 설명 |
|---------|------|------|
| `model_file_name` | string | QNN 모델 바이너리 파일 경로 (`.bin`) |
| `embedding_file_name` | string | 임베딩 가중치 파일 경로 |
| `tokenizer_file` | string | 토크나이저 파일 경로 |

**예시** (`nntr_config.json`):

```json
{
  "model_type": "CausalLM",
  "model_tensor_type": "FP32-FP32",
  "model_file_name": "model.bin",
  "embedding_file_name": "embedding.bin",
  "tokenizer_file": "/path/to/tokenizer.json",
  "num_to_generate": 512,
  "max_seq_len": 2048,
  "batch_size": 1
}
```

### `setupParameters()` 동작 방식

`setupParameters()` 메서드는 각 JSON 객체에서 키를 직접 읽어 대응하는 멤버 변수에 매핑합니다:

```cpp
void QNNTransformer::setupParameters(json &cfg, json &generation_cfg,
                                     json &nntr_cfg) {
  // nntr_config.json에서 경로 정보 읽기
  model_path = nntr_cfg["model_file_name"].get<std::string>();
  embedding_path = nntr_cfg["embedding_file_name"].get<std::string>();
  tokenizer_path = nntr_cfg["tokenizer_file"].get<std::string>();

  // config.json에서 Prefill 그래프 설정 읽기
  prefill_graph_name = cfg["prefill_graph_name"].get<std::string>();
  prefill_input_names = cfg["prefill_input_names"].get<std::string>();
  // ... (모든 prefill/generation 파라미터 자동 매핑)

  // config.json에서 모델 아키텍처 파라미터 읽기
  num_hidden_layers = cfg["num_hidden_layers"].get<int>();
  hidden_size = cfg["hidden_size"].get<int>();
  // ...

  // generation_config.json에서 생성 파라미터 읽기
  temperature = generation_cfg["temperature"].get<float>();
  top_k = generation_cfg["top_k"].get<int>();
  // ...
}
```

> **중요**: JSON 파일에 정의된 키 이름은 코드의 멤버 변수와 1:1로 대응됩니다. QNN SDK에서 모델 컴파일 시 생성되는 config 파일의 키 이름을 그대로 사용하므로, 별도의 변환 없이 자동으로 파라미터가 설정됩니다.

---

## Prefill vs Generation 모델

QNNTransformer는 효율적인 추론을 위해 두 개의 모델을 사용합니다.

| 항목 | Prefill 모델 | Generation 모델 |
|------|-------------|----------------|
| **용도** | 입력 프롬프트 전체를 한 번에 처리 | 토큰을 하나씩 생성 |
| **Embedding input_shape** | `1:sequence_length` | `1:1` |
| **QNN 그래프** | `prefill_graph_name` | `generation_graph_name` |
| **입출력 설정** | `prefill_*` 파라미터 사용 | `generation_*` 파라미터 사용 |
| **실행 시점** | 추론 시작 시 1회 | Prefill 이후 반복 실행 |

두 모델 모두 동일한 구조를 가집니다:

```
1. Embedding 레이어 (inputs_embeds)
   - in_dim: vocab_size
   - out_dim: hidden_size

2. Non-Embedding Input 레이어들 (0~N개)
   - position_ids, attention_mask 등
   - config의 non_embed_input_names/dims에서 자동 설정

3. QNN Graph 레이어
   - QNN 바이너리 그래프를 실행하는 핵심 레이어
```

---

## QNN Graph 레이어 속성

`qnn_graph` 레이어 생성 시 다음 속성들이 config에서 자동으로 설정됩니다:

```cpp
LayerHandle qnn_layer = createLayer("qnn_graph", {
  withKey("name", graph_name),           // QNN 그래프 이름
  withKey("path", model_path),           // QNN 바이너리 파일 경로
  withKey("dim", out_dim),               // 출력 차원
  withKey("tensor_dtype", out_data_format),   // 출력 텐서 데이터 타입
  withKey("tensor_type", out_tensor_format),  // 출력 텐서 형식
  withKey("input_layers", input_names),       // 입력 레이어 이름들
  withKey("input_quant_param", in_quant),     // 입력 양자화 파라미터
  withKey("output_quant_param", out_quant),   // 출력 양자화 파라미터
  withKey("engine", "qnn")                    // QNN 엔진 사용 지정
});
```

| 속성 | 설명 |
|------|------|
| `name` | 바이너리에 포함된 QNN 그래프를 식별하는 이름 |
| `path` | QNN 바이너리 파일(`.bin`) 경로 |
| `dim` | 출력 텐서 차원 |
| `tensor_dtype` | 출력 데이터 형식 (e.g., FLOAT32, UINT16) |
| `tensor_type` | 출력 텐서 형식 |
| `input_layers` | 이 레이어의 입력이 되는 레이어 이름들 |
| `input_quant_param` | 입력 텐서의 양자화 파라미터 (scale, offset) |
| `output_quant_param` | 출력 텐서의 양자화 파라미터 |
| `engine` | 사용할 컴퓨트 엔진 (`"qnn"` 지정 시 QNNContext 사용) |

---

## 새로운 QNN 모델 추가하기

QNNTransformer를 참고하여 새로운 QNN 기반 모델을 추가하는 절차입니다.

### 1. QNN SDK로 모델 컴파일

QNN SDK를 사용하여 모델을 컴파일하면 다음 파일들이 생성됩니다:
- 모델 바이너리 파일 (`.bin`) - Prefill/Generation 그래프 포함
- `config.json` - 그래프 입출력 사양 및 모델 파라미터

### 2. Config 파일 준비

3개의 JSON 설정 파일을 준비합니다:

```
my_model/
├── config.json              ← QNN 그래프 설정 (SDK 생성물 기반)
├── generation_config.json   ← 생성 파라미터
└── nntr_config.json         ← 모델/임베딩/토크나이저 경로
```

### 3. QNNTransformer 상속 (필요 시)

모델 구조가 다르다면 `QNNTransformer`를 상속하여 `initialize()`를 오버라이드합니다:

```cpp
class MyQNNModel : public QNNTransformer {
public:
  MyQNNModel(json &cfg, json &gen_cfg, json &nntr_cfg)
    : QNNTransformer(cfg, gen_cfg, nntr_cfg) {}

  void initialize() override {
    // 커스텀 모델 구성
  }

  void run(const WSTR prompt, ...) override {
    // 커스텀 실행 로직
  }
};
```

### 4. 사용 예시

```cpp
// JSON 설정 파일 로드
json cfg = causallm::LoadJsonFile("path/to/config.json");
json gen_cfg = causallm::LoadJsonFile("path/to/generation_config.json");
json nntr_cfg = causallm::LoadJsonFile("path/to/nntr_config.json");

// 모델 생성 및 초기화
QNNTransformer model(cfg, gen_cfg, nntr_cfg);
model.initialize();
model.load_weight("");
model.run("Hello, world!");
```

---

## 관련 파일 목록

| 파일 | 설명 |
|------|------|
| `Applications/CausalLM/models/qnn_transformer.h` | QNNTransformer 클래스 선언 |
| `Applications/CausalLM/models/qnn_transformer.cpp` | QNNTransformer 구현 |
| `Applications/CausalLM/models/transformer.h` | Transformer 베이스 클래스 |
| `nntrainer/qnn_context.h` | QNNContext (QNN 백엔드 관리) |
| `nntrainer/engine.h` | Engine (Context 등록/관리) |
| `Applications/CausalLM/res/qwen2/qwen2-0.5b/nntr_config.json` | 설정 파일 예시 |
