# Incremental Forwarding 리팩토링 계획

## 1. 현재 구현의 문제 분석

### 1.1 현재 구조 요약

현재 incremental forwarding은 `NeuralNetwork` → `NetworkGraph` → `LayerNode` → `Layer`로 이어지는 호출 체인에서 **단일 `(from, to)` 쌍**이 모든 레이어에 동일하게 전달되는 구조:

```
NeuralNetwork::incremental_forwarding(from, to, ...)
  └─ NetworkGraph::incremental_forwarding(from, to, ...)
       └─ for (layer : all_layers)
            forwarding_op(layer, training)   // from, to는 lambda 캡처로 고정
              └─ node->incremental_forwarding(from, to, training)
                   └─ layer->incremental_forwarding(context, from, to, training)
```

핵심 문제: `forwarding_op` lambda에서 `from`, `to`가 **캡처 시점에 고정**되어 모든 레이어가 동일한 값을 받음.

```cpp
// neuralnet.cpp:449-461
std::function<void(std::shared_ptr<LayerNode>, bool)> forwarding_op =
  [this, from, to, ...](std::shared_ptr<LayerNode> node, bool training) -> void {
    node->incremental_forwarding(from, to, training);  // 모든 레이어에 동일한 from, to
  };
```

**영향 범위:** `incremental_forwarding`을 구현하고 있는 파일은 총 **39개**:
- 코어 레이어 (nntrainer/layers/): embedding, fc, attention, multi_head_attention, addition, concat, layer_normalization, multiout 등 9개
- CL 레이어 (nntrainer/layers/cl_layers/): addition_cl, concat_cl, fc_cl, reshape_cl, rmsnorm_cl, swiglu_cl, transpose_cl 등 7개
- CausalLM 앱 (Applications/CausalLM/layers/): mha_core, qkv_layer, rms_norm, swiglu, lm_head, embedding_layer, embedding_normalize_layer, embedding_pooling_layer, reshaped_rms_norm, tie_word_embedding 등 10개
- CausalLM 모델: gpt_oss_moe_layer_cached, gpt_oss_moe_layer, qwen_moe_layer_cached, qwen_moe_layer, qwen_moe_layer_fsu 등 5개
- LLaMA 앱: custom_multi_head_attention_layer, rms_norm, rotary_embedding, swiglu 등 4개
- 그래프/모델: network_graph, neuralnet, layer_node 등 3개
- 테스트: layers_golden_tests 1개

---

### 1.2 문제 1: 배치별 다른 from/to 처리 불가

**현상:** 입력 배치가 여러 개일 때, 각 배치마다 시퀀스 진행 상태가 다를 수 있음 (e.g., 배치 0은 step 5, 배치 1은 step 3).

**현재 코드의 한계:**

```cpp
// neuralnet.cpp:481-496 - 입력 텐서의 batch 크기만 검증, per-batch from/to 없음
sharedConstTensors
NeuralNetwork::incremental_forwarding(unsigned int from, unsigned int to,
                                      sharedConstTensors input, ...) {
  auto current_batch = model_graph.getBatchSize();
  NNTR_THROW_IF(input[0]->batch() != current_batch, ...);
  // from, to는 스칼라 → 모든 배치에 동일 적용
}
```

각 레이어 내부에서도 배치 루프를 돌지만 동일한 from/to를 사용:

```cpp
// fc_layer.cpp - 배치 루프 내에서 동일한 from, to로 슬라이싱
for (unsigned int b = 0; b < hidden_.batch(); ++b) {
  // input_step_dim.height(to - from);  ← 모든 batch에 동일
  Tensor input_step = input_.getSharedDataTensor(input_step_dim, ...);
}
```

**구체적 예시:**
- Batch 0: 이미 10 토큰 처리, 11번째 토큰 처리 필요 → `from=10, to=11`
- Batch 1: 이미 5 토큰 처리, 6번째 토큰 처리 필요 → `from=5, to=6`
- 현재 구조에서는 단일 `(from, to)`만 전달 가능하므로, 배치 간 서로 다른 시퀀스 위치를 처리할 수 없음

**텐서 메모리 레이아웃 문제:** per-batch from/to가 다르면 `getSharedDataTensor()`로 단일 연속 슬라이스를 잡을 수 없음. batch 0은 offset 10에, batch 1은 offset 5에 쓰기해야 하므로, 배치별 별도 슬라이스가 필요함. 현재 `step_dim.height(to - from)`으로 모든 배치에 동일 height를 적용하는 패턴이 깨짐.

---

### 1.3 문제 2: 레이어 간 from/to 전파 불가

**현상:** 앞 레이어가 from/to를 변환/소비한 뒤, 변경된 from/to를 뒤 레이어에 전달할 메커니즘이 없음.

**현재 코드의 한계 - Embedding:**

```cpp
// embedding.cpp:123-128 - 내부적으로 from=0, to=1로 리셋
if (from) {
  NNTR_THROW_IF(to - from != 1, std::invalid_argument);
  from = 0;  // 로컬 변수만 변경, 다음 레이어에 전파 안됨
  to = 1;
}
```

Embedding 레이어가 `from=10, to=11`을 받아 내부적으로 `from=0, to=1`로 처리하지만:
- 출력 텐서의 유효 범위는 `[0, 1)` (height 1짜리 텐서)
- 다음 FC 레이어는 여전히 원래의 `from=10, to=11`을 받음
- FC 레이어는 이 값으로 height 10~11 위치에서 데이터를 읽으려 함 → **불일치**

**현재 코드의 한계 - MultiOut:**

```cpp
// multiout_layer.cpp:42-71 - 동일하게 from=0, to=1로 리셋
void MultiOutLayer::incremental_forwarding(RunLayerContext &context,
                                           unsigned int from, unsigned int to,
                                           bool training) {
  if (from) {
    NNTR_THROW_IF(to - from != 1, std::invalid_argument);
    from = 0;  // 로컬 리셋, 하류 레이어에 전파 안됨
    to = 1;
  }
  // ... batch size 1에서만 동작한다는 주석 있음
  // @todo: set reset stride as false. This implementation only works when
  // batch size is 1
}
```

**현재 코드의 한계 - MHACoreLayer (CausalLM 앱):**

```cpp
// Applications/CausalLM/layers/mha_core.cpp:209-224
unsigned int from = _from;
unsigned int to = _to;

if (to >= max_timestep) {
  if (!_from) {
    throw std::invalid_argument("...");
  } else {
    // KV cache가 넘침 → 시프트
    cache_shift = true;
    from = max_timestep - 1;  // 로컬 변수만 변경
    to = max_timestep;
  }
}
```

MHACoreLayer는 KV cache 오버플로 시 from/to를 변경하지만, 이 정보가 후속 레이어로 전달되지 않음.

**Attention 레이어의 비대칭 사용:**

```cpp
// attention_layer.cpp - Query는 (to-from), Key/Value는 (to) 사용
query_step_dim.height(to - from);   // 새로 입력된 부분만
value_step_dim.height(to);          // 누적된 전체 (KV cache)
key_step_dim.height(to);
```

Attention은 입력 중 Query/Key/Value에 대해 from/to를 **서로 다르게 해석**.
이것은 현재 단일 (from, to)만으로는 올바르게 표현할 수 없는 경우임.

---

### 1.4 문제 3: 다중 입력 레이어의 서로 다른 from/to

**현상:** 하나의 레이어가 여러 입력을 받을 때, 각 입력의 유효 범위(from/to)가 다를 수 있음.

**구체적 예시 - Multi-Head Attention:**
- Input 0 (Query): `from=10, to=11` (새 토큰 1개)
- Input 1 (Key): `from=0, to=11` (KV cache 포함 전체)
- Input 2 (Value): `from=0, to=11` (KV cache 포함 전체)

현재는 단일 `(from, to)`가 전달되므로, MHA 레이어 내부에서 하드코딩으로 이를 구분:

```cpp
// multi_head_attention_layer.cpp:637-642
projected_query_step_dim.height(to - from);   // 하드코딩: Query는 새 부분만
projected_key_step_dim.height(to);             // 하드코딩: Key는 전체
projected_value_step_dim.height(to);           // 하드코딩: Value는 전체
```

이 하드코딩은 MHA의 입력 순서가 변경되거나, 다른 유형의 다중-입력 레이어가 추가되면 깨짐.

**구체적 예시 - Addition 레이어 (residual connection):**
- Input 0 (skip connection): `from=10, to=11` (원본 입력의 해당 위치)
- Input 1 (MHA 출력): `from=0, to=1` (MHA가 리셋한 범위)
- 현재: 두 입력 모두 동일한 from=10, to=11을 받으므로, MHA 출력 접근 시 잘못된 오프셋 사용

---

## 2. 해결 방안

### 2.1 핵심 아이디어: Per-Input, Per-Batch `from/to`를 컨텍스트 메타데이터로 전달

from/to를 함수 인자가 아닌 **RunLayerContext에 연결된 메타데이터**로 관리하여, 각 텐서(=각 연결)마다 독립적인 from/to 정보를 가지게 함.

### 2.2 새로운 데이터 구조: `IncrementalInfo`

```cpp
/**
 * @file   incremental_info.h
 * @brief  텐서별 incremental forwarding 범위 정보
 *
 * 각 입력/출력 텐서에 대해 batch별로 다른 from/to를 지원
 */
struct IncrementalInfo {
  // per-batch from/to vectors
  std::vector<unsigned int> from;  // size == batch_size
  std::vector<unsigned int> to;    // size == batch_size

  /** @brief 기본 생성자: 빈 상태 (non-incremental) */
  IncrementalInfo() = default;

  /** @brief 모든 batch가 동일한 from/to인지 */
  bool is_uniform() const {
    if (from.empty()) return true;
    for (size_t i = 1; i < from.size(); ++i)
      if (from[i] != from[0] || to[i] != to[0]) return false;
    return true;
  }

  /** @brief 유효한 incremental 정보가 있는지 */
  bool isValid() const { return !from.empty(); }

  /** @brief batch별 from 조회 (uniform인 경우 batch 무시) */
  unsigned int getFrom(unsigned int batch = 0) const {
    return from.size() == 1 ? from[0] : from[batch];
  }

  /** @brief batch별 to 조회 */
  unsigned int getTo(unsigned int batch = 0) const {
    return to.size() == 1 ? to[0] : to[batch];
  }

  /** @brief batch별 step size (to - from) */
  unsigned int getStepSize(unsigned int batch = 0) const {
    return getTo(batch) - getFrom(batch);
  }

  /** @brief uniform step size인 경우에만 호출 가능 */
  unsigned int getUniformFrom() const { return from[0]; }
  unsigned int getUniformTo() const { return to[0]; }

  /** @brief 단일 (from, to)로 생성 (backward compatible) */
  static IncrementalInfo uniform(unsigned int f, unsigned int t,
                                  unsigned int batch_size = 1) {
    IncrementalInfo info;
    info.from.assign(batch_size, f);
    info.to.assign(batch_size, t);
    return info;
  }

  /** @brief per-batch from/to 벡터로 생성 */
  static IncrementalInfo perBatch(std::vector<unsigned int> f,
                                   std::vector<unsigned int> t) {
    IncrementalInfo info;
    info.from = std::move(f);
    info.to = std::move(t);
    return info;
  }
};
```

### 2.3 인터페이스 변경

#### 2.3.1 Layer 인터페이스 (layer_devel.h)

```cpp
// 기존 (deprecated로 유지)
virtual void incremental_forwarding(RunLayerContext &context,
                                    unsigned int from, unsigned int to,
                                    bool training) {
  forwarding(context, training);
}

// 신규 추가 - 기본 구현은 기존 시그니처로 위임 (backward compat)
virtual void incremental_forwarding(RunLayerContext &context,
                                    bool training) {
  // 기존 레이어가 (context, from, to, training) 오버라이드만 했다면
  // 이 default 구현을 통해 기존 코드가 그대로 동작함
  auto &info = context.getInputIncrementalInfo(0);
  if (info.isValid() && info.is_uniform()) {
    incremental_forwarding(context, info.getUniformFrom(),
                           info.getUniformTo(), training);
  } else {
    forwarding(context, training);
  }
  // 출력 IncrementalInfo는 입력을 그대로 전파 (기본 동작)
  for (unsigned int i = 0; i < context.getNumOutputs(); ++i) {
    if (context.getNumInputs() > 0) {
      context.setOutputIncrementalInfo(i, context.getInputIncrementalInfo(0));
    }
  }
}
```

**핵심 포인트:** 새 시그니처의 default 구현이 기존 시그니처를 호출하므로, 기존에 `(context, from, to, training)`만 오버라이드한 레이어들은 수정 없이 동작함.

#### 2.3.2 RunLayerContext 확장 (layer_context.h)

```cpp
class RunLayerContext {
  // 기존 멤버에 추가:
private:
  std::vector<IncrementalInfo> input_incremental_info;   // per-input
  std::vector<IncrementalInfo> output_incremental_info;  // per-output

public:
  /// 입력 텐서의 incremental 범위 조회
  const IncrementalInfo& getInputIncrementalInfo(unsigned int idx) const;

  /// 입력 텐서의 incremental 범위 설정 (그래프 전파 시 호출)
  void setInputIncrementalInfo(unsigned int idx, const IncrementalInfo &info);

  /// 출력 텐서의 incremental 범위 설정 (레이어가 forwarding 시 호출)
  void setOutputIncrementalInfo(unsigned int idx, const IncrementalInfo &info);

  /// 출력 텐서의 incremental 범위 조회 (다음 레이어의 입력으로 전파)
  const IncrementalInfo& getOutputIncrementalInfo(unsigned int idx) const;

  /// IncrementalInfo 벡터 크기 초기화 (finalizeContext 시)
  void initIncrementalInfo(unsigned int num_inputs, unsigned int num_outputs);

  /// 모든 IncrementalInfo 초기화 (매 forwarding 시작 시)
  void resetIncrementalInfo();
};
```

#### 2.3.3 LayerNode 변경 (layer_node.h/cpp)

```cpp
// 기존 유지 (backward compat wrapper)
void incremental_forwarding(unsigned int from, unsigned int to,
                            bool training = true);

// 신규 추가
void incremental_forwarding(bool training = true);
```

```cpp
// layer_node.cpp - 신규 구현
void LayerNode::incremental_forwarding(bool training) {
  loss->set(run_context->getRegularizationLoss());
  PROFILE_TIME_START(forward_event_key);
  layer->incremental_forwarding(*run_context, training);
  PROFILE_TIME_END(forward_event_key);
  // ... (기존과 동일한 validation/loss 로직)
}

// 기존 구현 → wrapper로 변환
void LayerNode::incremental_forwarding(unsigned int from, unsigned int to,
                                       bool training) {
  // 모든 입력에 uniform IncrementalInfo 설정
  for (unsigned int i = 0; i < run_context->getNumInputs(); ++i) {
    run_context->setInputIncrementalInfo(
      i, IncrementalInfo::uniform(from, to));
  }
  incremental_forwarding(training);
}
```

#### 2.3.4 NetworkGraph 변경 (network_graph.h/cpp)

```cpp
// 기존 유지 (backward compat wrapper)
sharedConstTensors incremental_forwarding(
  unsigned int from, unsigned int to, bool training, ...);

// 신규 추가
sharedConstTensors incremental_forwarding(
  bool training,
  std::function<void(std::shared_ptr<LayerNode>, bool)> forwarding_op = ...,
  std::function<bool(void *userdata)> stop_cb = ...,
  void *user_data = nullptr);

// 신규 - IncrementalInfo 전파 메서드
void propagateIncrementalInfo(const std::shared_ptr<LayerNode> &node);
```

#### 2.3.5 NeuralNetwork 변경 (neuralnet.h/cpp)

```cpp
// 기존 유지 (backward compat wrapper)
sharedConstTensors incremental_forwarding(unsigned int from, unsigned int to, ...);
sharedConstTensors incremental_inference(sharedConstTensors X, ...,
                                          unsigned int from, unsigned int to);

// 신규 추가 - IncrementalInfo 기반
sharedConstTensors incremental_forwarding(
  const std::vector<IncrementalInfo> &input_incremental_info,
  bool training = true, ...);
sharedConstTensors incremental_inference(
  sharedConstTensors X,
  const std::vector<IncrementalInfo> &input_incremental_info, ...);
```

---

### 2.4 IncrementalInfo 전파 메커니즘

#### 핵심 동작 흐름:

```
1. NeuralNetwork에서 입력 레이어의 IncrementalInfo 설정
     ↓
2. NetworkGraph가 레이어를 토폴로지 순서로 순회:
   for (layer : layers) {
     // a) 이전 레이어의 출력 IncrementalInfo를 현재 레이어의 입력 IncrementalInfo로 복사
     propagateIncrementalInfo(layer);

     // b) 레이어 실행 (레이어 내부에서 context를 통해 from/to 조회)
     layer->incremental_forwarding(training);
     //   - context.getInputIncrementalInfo(i)로 각 입력의 범위 확인
     //   - 연산 수행
     //   - context.setOutputIncrementalInfo(i, ...)로 출력 범위 설정
   }
```

#### 전파 규칙 구현 (network_graph.cpp):

```cpp
void NetworkGraph::propagateIncrementalInfo(
    const std::shared_ptr<LayerNode> &node) {
  auto &run_context = node->getRunContext();

  // 입력 레이어 (네트워크의 첫 번째 레이어)는 NeuralNetwork에서 이미 설정되어 있음
  if (node->getNumInputConnections() == 0)
    return;

  // node의 각 입력 연결에 대해:
  for (unsigned int i = 0; i < node->getNumInputConnections(); ++i) {
    // 그래프 연결 정보로 이전 노드의 출력 인덱스 찾기
    auto [prev_node_idx, prev_output_idx] = getInputConnection(node, i);
    auto &prev_node = getLayerNode(prev_node_idx);
    auto &prev_context = prev_node->getRunContext();

    // 이전 노드의 output_idx번째 출력의 IncrementalInfo →
    // 현재 노드의 i번째 입력의 IncrementalInfo로 복사
    run_context.setInputIncrementalInfo(
      i, prev_context.getOutputIncrementalInfo(prev_output_idx));
  }
}
```

#### 전파 흐름 예시 - Transformer Decoder Block:

```
[Input Token IDs]
  IncrementalInfo: from=10, to=11 (batch 0)

  ↓ Embedding Layer
  입력 info: from=10, to=11
  내부 처리: token_id[10]를 임베딩 → 출력 height=1
  출력 info: from=0, to=1  ← 레이어가 변환

  ↓ propagateIncrementalInfo()
  ↓ (Embedding의 출력 info가 FC의 입력 info로 복사)

  ↓ FC Layer (query projection)
  입력 info: from=0, to=1  ← 정확한 범위 수신
  내부 처리: height [0,1) 슬라이스에 대해 matmul
  출력 info: from=0, to=1  ← 그대로 전파

  ↓ MultiOut Layer (query를 MHA와 residual로 분기)
  입력 info: from=0, to=1
  출력[0] info: from=0, to=1  (→ MHA의 query 입력)
  출력[1] info: from=0, to=1  (→ residual add 입력)

  ↓ Multi-Head Attention
  입력[0] (query) info:  from=0, to=1   ← 새 토큰
  입력[1] (key) info:    from=0, to=1   ← (KV cache를 가진 레이어에서 info 변환됨)
  입력[2] (value) info:  from=0, to=1
  내부: KV cache에 새 K,V 추가, cache 범위는 [0, 11)
        attention 계산: Q[0:1] x K[0:11]^T → V[0:11]
  출력 info: from=0, to=1  ← query 기준

  ↓ Addition Layer (residual)
  입력[0] info: from=0, to=1  (MHA 출력)
  입력[1] info: from=0, to=1  (skip connection)  ← 이제 일치!
  출력 info: from=0, to=1
```

---

### 2.5 레이어별 변경 사항

#### 2.5.1 기본 동작 (layer_devel.h default 구현)

이미 2.3.1에서 설명. 기존 `(context, from, to, training)` 오버라이드를 가진 레이어는 수정 없이 동작.

#### 2.5.2 마이그레이션이 필요한 레이어들

새 인터페이스로 마이그레이션해야 하는 레이어는 **from/to를 내부적으로 변환하거나**, **per-input 다른 범위가 필요하거나**, **per-batch 처리가 필요한** 레이어:

| 우선순위 | 레이어 | 마이그레이션 이유 | 변경 핵심 |
|---------|--------|-----------------|----------|
| **P0** | `MultiHeadAttentionLayer` | 입력별 다른 from/to 필요 (Q vs K/V) | `getInputIncrementalInfo(0/1/2)`로 각 입력 독립 조회 |
| **P0** | `AttentionLayer` | 입력별 다른 from/to 필요 | 동일 |
| **P0** | `EmbeddingLayer` | from/to를 변환하여 출력 | `setOutputIncrementalInfo`로 변환된 범위 설정 |
| **P0** | `MultiOutLayer` | from/to를 변환 + batch=1 제한 해제 | 입력 info를 모든 출력에 전파 |
| **P1** | `FullyConnectedLayer` | per-batch from/to 지원 | batch 루프에서 `info.getFrom(b)` 사용 |
| **P1** | `AdditionLayer` | 다중 입력의 info 일치 검증 | 입력별 info 조회 후 element-wise add |
| **P1** | `ConcatLayer` | axis별 from/to 합산 로직 | concat axis에 따른 출력 info 계산 |
| **P1** | `LayerNormalizationLayer` | per-batch 지원 | info에서 범위 조회 |
| **P2** | CausalLM `MHACoreLayer` | per-batch + cache shift 전파 | 가장 복잡한 변환 로직 |
| **P2** | CausalLM 나머지 레이어들 | per-batch 지원 | info 조회로 전환 |
| **P2** | CL 레이어들 | 동일 | 동일 |
| **P2** | LLaMA 레이어들 | 동일 | 동일 |
| **P3** | 기존 시그니처만 오버라이드한 레이어들 | default 구현으로 동작하므로 변경 불필요 | 변경 없음 (나중에 점진적 마이그레이션) |

#### 2.5.3 상세 마이그레이션 예시 - Embedding

```cpp
void EmbeddingLayer::incremental_forwarding(RunLayerContext &context,
                                            bool training) {
  auto &info = context.getInputIncrementalInfo(0);
  const Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  Tensor &output_ = context.getOutput(SINGLE_INOUT_IDX);

  unsigned int batch_size = input_.batch();

  for (unsigned int b = 0; b < batch_size; ++b) {
    unsigned int b_from = info.getFrom(b);
    unsigned int b_to = info.getTo(b);
    unsigned int step_size = b_to - b_from;

    NNTR_THROW_IF(b_from && step_size != 1, std::invalid_argument)
      << "incremental step size is not 1";

    unsigned int actual_from = b_from ? 0 : b_from;
    unsigned int actual_to = b_from ? 1 : b_to;

    // 배치별 텐서 슬라이스 & 임베딩 수행
    for (unsigned int i = actual_from; i < actual_to; ++i) {
      // ... 기존 임베딩 로직 (배치 b에 대해)
    }
  }

  // 출력 범위 설정: 임베딩은 from을 리셋
  if (info.is_uniform()) {
    unsigned int step = info.getStepSize(0);
    unsigned int out_from = info.getFrom(0) ? 0 : info.getFrom(0);
    unsigned int out_to = info.getFrom(0) ? step : info.getTo(0);
    context.setOutputIncrementalInfo(0,
      IncrementalInfo::uniform(out_from, out_to, batch_size));
  } else {
    // per-batch 출력 범위 설정
    std::vector<unsigned int> out_from(batch_size), out_to(batch_size);
    for (unsigned int b = 0; b < batch_size; ++b) {
      out_from[b] = info.getFrom(b) ? 0 : info.getFrom(b);
      out_to[b] = info.getFrom(b) ? info.getStepSize(b) : info.getTo(b);
    }
    context.setOutputIncrementalInfo(0,
      IncrementalInfo::perBatch(out_from, out_to));
  }
}
```

#### 2.5.4 상세 마이그레이션 예시 - Multi-Head Attention

```cpp
void MultiHeadAttentionLayer::incremental_forwarding(
    RunLayerContext &context, bool training) {
  // 각 입력에 대해 독립적으로 IncrementalInfo 조회
  auto &query_info = context.getInputIncrementalInfo(0);
  auto &key_info   = context.getInputIncrementalInfo(1);
  auto &value_info = context.getInputIncrementalInfo(2);

  // 하드코딩 없이 각 입력의 실제 범위를 사용
  for (unsigned int b = 0; b < batch_size; ++b) {
    unsigned int q_from = query_info.getFrom(b);
    unsigned int q_to   = query_info.getTo(b);
    unsigned int k_from = key_info.getFrom(b);
    unsigned int k_to   = key_info.getTo(b);
    unsigned int v_from = value_info.getFrom(b);
    unsigned int v_to   = value_info.getTo(b);

    // Query projection: q_from ~ q_to 범위
    // Key/Value projection + cache: k_from ~ k_to 범위
    // Attention: Q[q_from:q_to] x K[0:k_to]^T → O[q_from:q_to]
    // ...
  }

  // 출력 범위 = Query의 범위 (attention 출력은 query와 동일한 시퀀스 길이)
  context.setOutputIncrementalInfo(0, query_info);
}
```

---

### 2.6 Per-Batch from/to 시 텐서 슬라이싱 전략

per-batch from/to가 다를 때의 텐서 메모리 접근 패턴 변경:

#### 현재 패턴 (uniform from/to):
```cpp
// 모든 batch에 동일한 step_dim 적용
TensorDim step_dim = dim;
step_dim.height(to - from);
for (unsigned int b = 0; b < batch_size; ++b) {
  Tensor step = tensor.getSharedDataTensor(step_dim,
    b * dim.getFeatureLen() + from * dim.width());
  // ... step으로 연산
}
```

#### 새 패턴 (per-batch from/to):
```cpp
for (unsigned int b = 0; b < batch_size; ++b) {
  unsigned int b_from = info.getFrom(b);
  unsigned int b_to = info.getTo(b);

  TensorDim step_dim = dim;
  step_dim.batch(1);
  step_dim.height(b_to - b_from);

  Tensor step = tensor.getSharedDataTensor(step_dim,
    b * dim.getFeatureLen() + b_from * dim.width(), true);
  // ... step으로 연산
}
```

**주의:** per-batch에서 step_size가 배치마다 다르면 BLAS 배치 연산을 직접 사용할 수 없으므로, 배치별 개별 연산으로 fallback해야 함. 성능 최적화로는:
- `is_uniform()` 체크 후 uniform인 경우 기존 패턴 사용
- non-uniform인 경우만 per-batch 루프

---

### 2.7 Backward Compatibility 전략

#### 핵심 원칙: 기존 코드가 수정 없이 컴파일/동작해야 함

1. **기존 시그니처 유지 (deprecated 마킹)**
   ```cpp
   // layer_devel.h - 기존 virtual 메서드 유지
   [[deprecated("Use incremental_forwarding(context, training) instead")]]
   virtual void incremental_forwarding(RunLayerContext &context,
                                       unsigned int from, unsigned int to,
                                       bool training) {
     forwarding(context, training);
   }
   ```

2. **새 시그니처의 default 구현이 기존 시그니처로 위임**
   - 기존에 `(context, from, to, training)`을 오버라이드한 레이어는 수정 없이 동작
   - 새 인터페이스의 default가 context에서 from/to를 꺼내서 기존 메서드 호출

3. **NeuralNetwork, NetworkGraph의 기존 공개 API 유지**
   - 기존 `incremental_forwarding(from, to, ...)` → 내부에서 `IncrementalInfo::uniform`으로 변환
   - 새 오버로드 추가

---

### 2.8 단계별 구현 계획

#### Phase 1: 인프라 추가 (하위 호환성 유지, 기존 동작 변경 없음)
1. `IncrementalInfo` 구조체 정의 (`incremental_info.h`)
2. `RunLayerContext`에 IncrementalInfo 저장/조회 API 추가
3. `layer_devel.h`에 새 virtual 메서드 추가 (기존 메서드 유지)
4. `LayerNode`에 새 `incremental_forwarding(bool)` 추가
5. `NetworkGraph`에 `propagateIncrementalInfo()` 구현
6. `NetworkGraph::incremental_forwarding(bool, ...)` 추가
7. `NeuralNetwork`에 IncrementalInfo 기반 오버로드 추가

**이 시점에서 기존 코드는 수정 없이 동작. 새 인터페이스도 사용 가능.**

#### Phase 2: 코어 레이어 마이그레이션
1. `EmbeddingLayer` → 새 인터페이스 (from/to 변환 + 출력 info 설정)
2. `MultiOutLayer` → 새 인터페이스 (batch>1 지원)
3. `MultiHeadAttentionLayer` → 새 인터페이스 (per-input info)
4. `AttentionLayer` → 새 인터페이스 (per-input info)
5. `FullyConnectedLayer` → 새 인터페이스 (per-batch 지원)
6. `AdditionLayer`, `ConcatLayer`, `LayerNormalizationLayer`

#### Phase 3: 앱 레이어 마이그레이션
1. CausalLM 레이어들 (MHACoreLayer 등)
2. LLaMA 레이어들
3. CL 레이어들

#### Phase 4: 정리
1. 기존 시그니처에 `[[deprecated]]` 추가
2. 테스트 업데이트 (layers_golden_tests.cpp)
3. 문서 업데이트

---

### 2.9 수정 대상 파일 목록

#### Phase 1 (인프라)
| 파일 | 변경 내용 |
|------|----------|
| `nntrainer/layers/incremental_info.h` (신규) | `IncrementalInfo` 구조체 정의 |
| `nntrainer/layers/layer_context.h` | `RunLayerContext`에 IncrementalInfo 멤버/메서드 추가 |
| `nntrainer/layers/layer_context.cpp` | IncrementalInfo 관련 구현 |
| `nntrainer/layers/layer_devel.h` | 새 virtual `incremental_forwarding(context, training)` 추가 |
| `nntrainer/layers/layer_node.h` | 새 `incremental_forwarding(training)` 선언 |
| `nntrainer/layers/layer_node.cpp` | 새 메서드 구현, 기존 메서드를 wrapper로 변환 |
| `nntrainer/graph/network_graph.h` | `propagateIncrementalInfo()`, 새 `incremental_forwarding()` 선언 |
| `nntrainer/graph/network_graph.cpp` | 전파 로직 + 새 forwarding 루프 구현 |
| `nntrainer/models/neuralnet.h` | 새 오버로드 선언 |
| `nntrainer/models/neuralnet.cpp` | 새 오버로드 구현, 기존 메서드를 wrapper로 변환 |

#### Phase 2 (코어 레이어)
| 파일 | 변경 내용 |
|------|----------|
| `nntrainer/layers/embedding.cpp` | 새 인터페이스로 마이그레이션 |
| `nntrainer/layers/multiout_layer.cpp` | 새 인터페이스 + batch>1 지원 |
| `nntrainer/layers/multi_head_attention_layer.cpp` | per-input info 사용 |
| `nntrainer/layers/attention_layer.cpp` | per-input info 사용 |
| `nntrainer/layers/fc_layer.cpp` | per-batch info 사용 |
| `nntrainer/layers/addition_layer.cpp` | per-input info 검증 |
| `nntrainer/layers/concat_layer.cpp` | axis별 info 합산 |
| `nntrainer/layers/layer_normalization_layer.cpp` | info 기반 범위 조회 |

#### Phase 3 (앱 레이어)
| 파일 | 변경 내용 |
|------|----------|
| `Applications/CausalLM/layers/*.cpp` (10개) | 새 인터페이스로 마이그레이션 |
| `Applications/CausalLM/models/*.cpp` (5개) | 새 인터페이스로 마이그레이션 |
| `Applications/LLaMA/jni/*.cpp` (4개) | 새 인터페이스로 마이그레이션 |
| `nntrainer/layers/cl_layers/*.cpp` (7개) | 새 인터페이스로 마이그레이션 |

#### Phase 4 (정리)
| 파일 | 변경 내용 |
|------|----------|
| `test/unittest/layers/layers_golden_tests.cpp` | 새 인터페이스 테스트 추가 |
| `nntrainer/tensor/cpu_backend/fallback/fallback_internal.cpp` | 필요 시 업데이트 |

---

## 3. 요약

### 현재 문제의 근본 원인
**단일 스칼라 `(from, to)`가 함수 인자로 전달되어 모든 레이어, 모든 배치, 모든 입력에 동일하게 적용**

### 해결 핵심
**`IncrementalInfo`를 RunLayerContext 메타데이터로 관리하여 per-input, per-batch 범위를 지원하고, 그래프 순회 시 자동 전파**

### 3가지 문제에 대한 해결:

| 문제 | 해결 |
|------|------|
| 배치별 다른 from/to | `IncrementalInfo.from/to`가 `vector<uint>` (per-batch) |
| 레이어 간 from/to 전파 | `propagateIncrementalInfo()`가 출력→입력으로 자동 전파, 레이어가 `setOutputIncrementalInfo()`로 변환된 범위 설정 |
| 다중 입력의 다른 from/to | `getInputIncrementalInfo(idx)`로 입력별 독립 조회 |

### 설계 원칙:
1. **Backward compatible:** 기존 코드 수정 없이 컴파일/동작
2. **점진적 마이그레이션:** Phase별로 독립적으로 배포 가능
3. **레이어 자율성:** 각 레이어가 입력 info를 해석하고 출력 info를 설정하는 책임
4. **성능 보존:** uniform인 경우 기존과 동일한 성능 경로 유지
