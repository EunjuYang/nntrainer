## nntrainer Custom Layer 튜토리얼 (PowLayer 예제)

이 디렉터리는 **nntrainer에서 custom object(레이어, 옵티마이저 등)를 만드는 방법**을 보여주는 예제들을 모아둔 곳입니다.
이 문서는 nntrainer를 처음 접한 사람을 대상으로, **가장 단순한 custom layer를 직접 따라 만들어 보는 수업용 튜토리얼**을 제공합니다.

---

### 목표

- **목표 1**: nntrainer의 `Layer` 인터페이스 구조를 이해한다.
- **목표 2**: 입력 텐서를 \(f(x) = x^{n}\) 형태로 변환하는 **PowLayer**를 직접 구현해 본다.
- **목표 3**: 구현한 레이어를 **AppContext에 등록**하고, **INI 파일 / C++ API**에서 사용하는 방법을 배운다.

---

### 예제 개요

이 튜토리얼에서 사용하는 주요 파일은 아래와 같습니다.

- **레이어 구현 코드**
  - `Applications/Custom/pow.h` : PowLayer 클래스 선언
  - `Applications/Custom/pow.cpp` : PowLayer 동작 구현
- **커스텀 loss 레이어 (참고)**
  - `Applications/Custom/mae_loss.h`, `Applications/Custom/mae_loss.cpp`
- **레이어 등록 + 데모 실행 코드**
  - `Applications/Custom/LayerClient/jni/main.cpp`
  - `Applications/Custom/LayerClient/res/custom_layer_client.ini`

이 코드는 이미 저장소 안에 들어 있으므로, 수업에서는

1. 전체 구조를 설명하고,
2. 핵심 메서드 몇 개를 같이 읽어 보면서,
3. 직접 수정/실험해 보도록 진행하면 됩니다.

---

### 1. Custom Layer의 기본 구조 이해하기

nntrainer에서 새 레이어를 만들려면 **`nntrainer::Layer`를 상속**받고, 몇 가지 순수 가상 함수를 구현해야 합니다.
PowLayer의 선언부(`pow.h`)는 다음과 같이 생겼습니다.

```cpp
class PowLayer final : public nntrainer::Layer {
public:
  PowLayer(float exponent_ = 1) : Layer(), exponent(exponent_) {}
  ~PowLayer() {}

  void finalize(nntrainer::InitLayerContext &context) override;
  void forwarding(nntrainer::RunLayerContext &context, bool training) override;
  void calcDerivative(nntrainer::RunLayerContext &context) override;
  bool supportBackwarding() const override { return true; };

  void exportTo(nntrainer::Exporter &exporter,
                const ml::train::ExportMethods &method) const override {}

  const std::string getType() const override { return PowLayer::type; };
  void setProperty(const std::vector<std::string> &values) override;

  static constexpr const char *type = "custom_pow";

private:
  float exponent;
};
```

- **`getType()` / `type`**: 이 문자열(`"custom_pow"`)이 INI 파일과 C++ API에서 사용할 **레이어 이름**이 됩니다.
- **`setProperty()`**: INI나 API에서 전달된 문자열 속성을 파싱하여, 이 레이어의 `exponent` 값을 설정합니다.
- **`finalize()`**: 입력/출력 텐서의 shape, 내부 weight 등이 있다면 이 시점에 정의합니다.
- **`forwarding()`**: 순전파(Forward)를 구현합니다. 여기서는 입력 텐서의 각 원소에 `pow(x, exponent)`를 적용합니다.
- **`calcDerivative()`**: 역전파(Backward)에서 이 레이어의 미분 결과를 계산합니다.

수업에서는 우선 이 인터페이스를 그림으로 설명하고, 각 메서드가 언제/어디서 호출되는지를 간단한 다이어그램과 함께 소개하면 이해하기 좋습니다.

---

### 2. PowLayer 동작 구현 살펴보기

실제 동작은 `pow.cpp`에 구현되어 있습니다. 핵심 부분만 발췌해서 보면 다음과 같습니다.

#### 2-1. 속성 파싱 (`setProperty`)

```cpp
void PowLayer::setProperty(const std::vector<std::string> &values) {
  PowUtil::Entry e;

  for (auto &val : values) {
    e = PowUtil::getKeyValue(val); // "key=value" 문자열 파싱

    if (e.key != "exponent") {
      std::string msg = "[PowLayer] Unknown Layer Property Key for value " +
                        std::string(e.key);
      throw std::invalid_argument(msg);
    }

    exponent = std::stoi(e.value);
  }
}
```

- INI / API에서 `"exponent=3"` 같은 문자열이 들어오면, `exponent` 멤버 변수에 저장합니다.
- **실습 아이디어**: `exponent`값이 0 이하일 때 예외를 던지도록 검증 로직을 추가해 보게 할 수 있습니다.

#### 2-2. 출력 텐서 shape 설정 (`finalize`)

```cpp
void PowLayer::finalize(nntrainer::InitLayerContext &context) {
  context.setOutputDimensions(context.getInputDimensions());

  // 이 레이어는 학습 가능한 weight가 없으므로 추가 작업이 필요 없습니다.
}
```

- PowLayer는 입력과 출력의 shape가 동일합니다. 따라서 입력 차원을 그대로 출력으로 설정합니다.

#### 2-3. 순전파 구현 (`forwarding`)

```cpp
void PowLayer::forwarding(nntrainer::RunLayerContext &context, bool training) {
  static constexpr size_t SINGLE_INOUT_IDX = 0;

#ifdef DEBUG
  std::cout << "pow layer forward is called\n";
#endif

  context.getInput(SINGLE_INOUT_IDX)
    .pow(exponent, context.getOutput(SINGLE_INOUT_IDX));
}
```

- **핵심 한 줄**: `input.pow(exponent, output)`
  - 입력 텐서의 각 원소를 `exponent` 제곱한 값을 출력 텐서에 저장합니다.

#### 2-4. 역전파 구현 (`calcDerivative`)

```cpp
void PowLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  static constexpr size_t SINGLE_INOUT_IDX = 0;

#ifdef DEBUG
  std::cout << "pow layer backward is called\n";
#endif

  const nntrainer::Tensor &derivative_ =
    context.getIncomingDerivative(SINGLE_INOUT_IDX);
  nntrainer::Tensor &dx = context.getOutgoingDerivative(SINGLE_INOUT_IDX);

  derivative_.multiply(exponent, dx);
}
```

- 아주 단순화된 예제로, 들어온 미분값에 `exponent`를 곱해서 다시 내보냅니다.
- 실제 \(y = x^{n}\)의 정확한 미분은 \(\frac{dy}{dx} = n x^{n-1}\) 이지만, 여기서는 **구현을 단순화**하여 수업 예제로 쓰고 있습니다.
- **실습 아이디어**: 학생들에게 직접 \(n x^{n-1}\) 형태로 고쳐 보게 하면서, forward 값(`context.getOutput`)을 함께 사용하는 법을 알려줄 수 있습니다.

---

### 3. AppContext에 Custom Layer 등록하기

이제 구현한 `PowLayer`를 **nntrainer 런타임에 등록**해야 INI 파일 / C++ API에서 사용할 수 있습니다.
등록 코드는 `LayerClient/jni/main.cpp`에 있습니다.

```cpp
auto &ct_engine = nntrainer::Engine::Global();
auto app_context = static_cast<nntrainer::AppContext *>(
  ct_engine.getRegisteredContext("cpu"));

// registerFactory는 "레이어를 생성하는 함수"를 등록합니다.
app_context->registerFactory(nntrainer::createLayer<custom::PowLayer>);
app_context->registerFactory(nntrainer::createLayer<custom::MaeLossLayer>);
```

- `createLayer<custom::PowLayer>`는 내부적으로 `PowLayer::type` (즉, `"custom_pow"`)을 키로 사용합니다.
- 따라서 **등록 이후에는** INI 파일이나 C++ API에서 `Type = custom_pow` 같은 식으로 이 레이어를 사용할 수 있습니다.

수업에서는 여기서 **"nntrainer가 문자열 타입을 어떻게 실제 C++ 클래스와 연결하는지"**를 설명해 주면 좋습니다.

---

### 4. INI 파일에서 Custom Layer 사용하기

`LayerClient` 예제는 INI 기반 설정 파일인 `res/custom_layer_client.ini`를 사용합니다. PowLayer가 등장하는 부분은 아래와 같습니다.

```ini
[powlayer]
input_layers = inputlayer
Type = custom_pow  # PowLayer::getType() == "custom_pow"
exponent = 3       # setProperty("exponent=3")로 전달됨
```

- `Type = custom_pow` : 우리가 구현한 PowLayer를 사용한다는 의미입니다.
- `exponent = 3` : PowLayer의 `setProperty()`에 문자열로 전달되어, 내부 멤버 `exponent` 값이 3이 됩니다.

학생들에게는 INI 파일을 수정해 보게 하면서

- `exponent = 2`, `exponent = 5` 등으로 값을 바꿔 보고,
- 출력 결과(또는 디버그 로그)가 어떻게 달라지는지를 직접 확인하게 하면 이해가 빨라집니다.

---

### 5. C++ API에서 Custom Layer 사용하기

동일한 PowLayer를 **C++ API 방식**으로 사용할 수도 있습니다.
`LayerClient/jni/main.cpp`의 `api_model_run()` 함수 안에서 레이어 배열을 만드는 부분을 보면:

```cpp
layers = std::vector<std::shared_ptr<ml::train::Layer>>{
  ml::train::layer::Input({"name=inputlayer", "input_shape=1:1:100"}),
  ml::train::createLayer("custom_pow", {"name=powlayer", "exponent=3"}),
  ml::train::layer::FullyConnected(
    {"name=outputlayer", "input_layers=powlayer", "unit=10",
     "bias_initializer=zeros", "activation=softmax"}),
  ml::train::createLayer("mae_loss", {"name=mae_loss"})};
```

- `ml::train::createLayer("custom_pow", {...})` : 문자열 이름으로 custom 레이어를 생성합니다.
- 나머지 레이어(`Input`, `FullyConnected`, `mae_loss`)와 동일한 방식으로 연결할 수 있습니다.

수업 예제로는 이 부분을 복사해서 **새로운 작은 main 함수**를 만들어 보게 하거나,
PowLayer 대신 다른 custom layer(예: `NegativeLayer`, `SqrtLayer`와 비슷한 것)를 직접 만들어 끼워 넣어 보게 할 수 있습니다.

---

### 6. 예제 빌드 및 실행 방법

이 저장소는 이미 Meson/Ninja 빌드 구성이 되어 있습니다. 기본적인 흐름은 다음과 같습니다.

1. **프로젝트 전체 빌드** (상위 디렉터리에서)
   - (프로젝트 가이드에 맞게 `meson setup`, `ninja` 등을 실행)
2. **LayerClient 실행**

   ```bash
   $ Applications/Custom/LayerClient/jni/layer_client model
   ```

   - PowLayer forward/backward가 호출되면, (디버그 빌드에서) 터미널에 관련 로그가 출력됩니다.
   - INI 파일을 직접 지정하고 싶다면 `res/custom_layer_client.ini` 경로를 넘겨줄 수 있습니다.

실행 로그를 보면서 학생들에게

- Forward 시 입력/출력 텐서 shape,
- PowLayer가 어디에 끼어들어 있는지,
- Backward 시 미분 값이 어떻게 흘러가는지

를 설명하면 좋습니다.

---

### 7. 수업에서 활용할 수 있는 응용 과제 아이디어

- **과제 1**: PowLayer의 `calcDerivative()`를 실제 미분 \(n x^{n-1}\) 형태로 수정해 보기
- **과제 2**: `exponent`를 정수가 아니라 실수(float)로 받아, `2.5` 같은 값도 허용되도록 확장해 보기
- **과제 3**: 새로운 custom layer `SquareLayer`를 만들어, `x^2`만 지원하고 `setProperty()`를 사용하지 않는 단순 버전 만들어 보기
- **과제 4**: `MaeLossLayer` 구현을 살펴보고, MSE(Mean Squared Error) loss 레이어를 직접 구현해 보기

이 튜토리얼을 기반으로, nntrainer의 다른 예제(`Applications/Custom/LayerPlugin`, `Applications/CausalLM` 등)를 함께 살펴보면
실제 프로젝트에서 custom layer를 어떻게 확장/활용할 수 있는지 자연스럽게 이어서 설명할 수 있습니다.
