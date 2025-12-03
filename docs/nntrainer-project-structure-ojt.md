# NNTrainer 프로젝트 구조 종합 OJT 가이드

## 목차
1. [프로젝트 개요](#프로젝트-개요)
2. [전체 디렉토리 구조](#전체-디렉토리-구조)
3. [핵심 컴포넌트 상세 분석](#핵심-컴포넌트-상세-분석)
4. [빌드 시스템](#빌드-시스템)
5. [API 구조](#api-구조)
6. [개발 워크플로우](#개발-워크플로우)
7. [테스트 구조](#테스트-구조)
8. [예제 애플리케이션](#예제-애플리케이션)
9. [확장 가이드](#확장-가이드)

---

## 프로젝트 개요

### NNTrainer란?
**NNTrainer**는 **임베디드 디바이스에서 신경망 모델을 학습할 수 있는 경량 소프트웨어 프레임워크**입니다.

### 핵심 목표
- **온디바이스 학습**: 서버 없이 디바이스에서 직접 모델 학습 및 파인튜닝
- **리소스 효율성**: 제한된 메모리와 연산 자원을 효율적으로 활용
- **개인화**: 사용자 데이터로 모델을 개인화
- **크로스 플랫폼**: Tizen, Ubuntu, Android 등 다양한 플랫폼 지원

### 지원 기능
- 다양한 레이어 타입 (CNN, RNN, Transformer 등)
- 최적화 알고리즘 (SGD, Adam, AdamW)
- 손실 함수 (MSE, Cross Entropy 등)
- 데이터셋 로더
- 모델 저장/로드
- 추론 및 학습 모드

---

## 전체 디렉토리 구조

```
nntrainer/
├── api/                    # API 인터페이스
│   ├── capi/              # C API (Tizen 공식)
│   └── ccapi/             # C++ API
├── Applications/           # 예제 애플리케이션
│   ├── CausalLM/          # LLM 추론 예제
│   ├── MNIST/             # MNIST 분류 예제
│   ├── Resnet/            # ResNet 예제
│   ├── VGG/               # VGG 예제
│   ├── YOLOv2/            # YOLO v2 예제
│   ├── KNN/               # K-Nearest Neighbor 예제
│   ├── Custom/            # 커스텀 레이어/옵티마이저 예제
│   └── ...
├── nntrainer/             # 핵심 프레임워크 코드
│   ├── layers/            # 레이어 구현
│   ├── optimizers/        # 옵티마이저 구현
│   ├── tensor/            # 텐서 연산 및 메모리 관리
│   ├── models/            # 모델 관리
│   ├── dataset/           # 데이터셋 로더
│   ├── compiler/          # 모델 컴파일러 (INI, ONNX, TFLite)
│   ├── graph/             # 계산 그래프
│   ├── opencl/            # OpenCL 지원
│   └── utils/             # 유틸리티 함수
├── test/                  # 테스트 코드
│   ├── unittest/          # 단위 테스트
│   ├── ccapi/             # C++ API 테스트
│   └── tizen_capi/        # Tizen C API 테스트
├── benchmarks/            # 벤치마크 코드
├── docs/                  # 문서
├── debian/                # Debian 패키징
├── packaging/             # 패키징 관련
├── jni/                   # JNI 관련
├── nnstreamer/            # NNStreamer 통합
├── subprojects/           # 서브프로젝트 (의존성)
└── tools/                 # 개발 도구
```

---

## 핵심 컴포넌트 상세 분석

### 1. nntrainer/ - 핵심 프레임워크

#### 1.1 layers/ - 레이어 구현
**위치**: `nntrainer/layers/`

**역할**: 신경망의 기본 구성 요소인 레이어들을 구현

**주요 레이어 타입**:

##### 기본 레이어
- **InputLayer**: 입력 레이어
- **FullyConnectedLayer**: 완전 연결 레이어
- **Conv2DLayer**: 2D 컨볼루션 레이어
- **Conv1DLayer**: 1D 컨볼루션 레이어
- **Pooling2DLayer**: 2D 풀링 레이어 (Max, Average, Global)

##### 정규화 레이어
- **BatchNormalizationLayer**: 배치 정규화
- **LayerNormalizationLayer**: 레이어 정규화

##### 활성화 및 연산 레이어
- **ActivationLayer**: 활성화 함수 (ReLU, Sigmoid, Tanh, Softmax 등)
- **AdditionLayer**: 덧셈 연산
- **ConcatLayer**: 연결 연산
- **MultiplyLayer**: 곱셈 연산

##### 순환 신경망 레이어
- **RNNLayer**: RNN 레이어
- **LSTMLayer**: LSTM 레이어
- **GRULayer**: GRU 레이어
- **RNNCellLayer**, **LSTMCellLayer**, **GRUCellLayer**: 셀 단위 레이어
- **ZoneoutLSTMCellLayer**: Zoneout LSTM 셀

##### 어텐션 레이어
- **AttentionLayer**: 기본 어텐션 레이어
- **MultiHeadAttentionLayer**: Multi-Head Attention

##### 특수 레이어
- **EmbeddingLayer**: 임베딩 레이어
- **PositionalEncodingLayer**: 위치 인코딩
- **DropoutLayer**: 드롭아웃
- **FlattenLayer**: 평탄화 레이어
- **ReshapeLayer**: 형태 변경 레이어
- **PermuteLayer**: 차원 순서 변경

##### 백본 레이어
- **NNStreamerLayer**: NNStreamer 모델 래핑
- **TfLiteLayer**: TensorFlow Lite 모델 래핑

##### 손실 레이어 (`layers/loss/`)
- **MSELossLayer**: Mean Squared Error
- **CrossEntropySigmoidLossLayer**: Cross Entropy with Sigmoid
- **CrossEntropySoftmaxLossLayer**: Cross Entropy with Softmax
- **KLDLossLayer**: Kullback-Leibler Divergence

##### OpenCL 레이어 (`layers/cl_layers/`)
- GPU 가속을 위한 OpenCL 구현 레이어들

**레이어 개발 패턴**:
```cpp
class MyLayer : public LayerImpl {
public:
    void finalize(InitLayerContext &context) override;
    void forwarding(RunLayerContext &context, bool training) override;
    void calcDerivative(RunLayerContext &context) override;
    void calcGradient(RunLayerContext &context) override;
    void setProperty(const std::vector<std::string> &values) override;
    const std::string getType() const override;
};
```

#### 1.2 optimizers/ - 최적화 알고리즘
**위치**: `nntrainer/optimizers/`

**역할**: 경사 하강법 기반 최적화 알고리즘 구현

**지원 옵티마이저**:
- **SGD**: Stochastic Gradient Descent
- **Adam**: Adaptive Moment Estimation
- **AdamW**: Adam with decoupled weight decay

**학습률 스케줄러** (`optimizers/lr_scheduler_*.h`):
- **Constant**: 상수 학습률
- **Exponential**: 지수 감소
- **Step**: 단계별 감소
- **Cosine**: 코사인 감소
- **Linear**: 선형 감소

**옵티마이저 인터페이스**:
```cpp
class Optimizer {
public:
    virtual void applyGradient(RunOptimizerContext &context) = 0;
    virtual void setProperty(const std::vector<std::string> &values) = 0;
};
```

#### 1.3 tensor/ - 텐서 연산 및 메모리 관리
**위치**: `nntrainer/tensor/`

**역할**: 텐서 데이터 구조 및 연산, 메모리 관리

**주요 컴포넌트**:

##### 텐서 타입
- **Tensor**: 기본 FP32 텐서
- **HalfTensor**: FP16 텐서
- **Q4_0Tensor**, **Q4_KTensor**: 양자화 텐서 (4-bit)
- **Q6_KTensor**: 양자화 텐서 (6-bit)
- **UIntTensor**: 부호 없는 정수 텐서

##### 메모리 관리
- **MemoryPlanner**: 메모리 할당 계획
  - `BasicPlanner`: 기본 메모리 플래너
  - `OptimizedV1Planner`: 최적화 버전 1
  - `OptimizedV2Planner`: 최적화 버전 2
  - `OptimizedV3Planner`: 최적화 버전 3
- **MemoryPool**: 메모리 풀 관리
- **CachePool**: 캐시 풀 관리

##### 연산 백엔드 (`tensor/cpu_backend/`)
- **Fallback**: 기본 CPU 구현
- **x86**: x86 아키텍처 최적화 (AVX2)
- **arm**: ARM 아키텍처 최적화 (NEON)
- **CBLAS Interface**: CBLAS 라이브러리 인터페이스
- **GGML Interface**: GGML 라이브러리 인터페이스

##### OpenCL 연산 (`tensor/cl_operations/`)
- OpenCL을 통한 GPU 가속 연산

**텐서 차원 형식**:
- 4D 텐서: `[Batch, Channel, Height, Width]`
- 예: `1:3:224:224` = 배치 1, 채널 3, 높이 224, 너비 224

#### 1.4 models/ - 모델 관리
**위치**: `nntrainer/models/`

**역할**: 모델 생성, 컴파일, 학습, 추론 관리

**주요 클래스**:
- **NeuralNetwork**: 신경망 모델 클래스
- **Model**: 모델 인터페이스

**주요 기능**:
- 모델 생성 및 레이어 추가
- 모델 컴파일 및 초기화
- 학습 및 추론 실행
- 가중치 저장/로드
- 모델 요약 출력

#### 1.5 dataset/ - 데이터셋 로더
**위치**: `nntrainer/dataset/`

**역할**: 다양한 형식의 데이터셋 로딩

**지원 데이터셋 타입**:
- **FileDataset**: 파일 기반 데이터셋
- **RandomDataLoader**: 랜덤 데이터 생성기
- **GeneratorDataset**: 제너레이터 기반 데이터셋

**데이터 형식**:
- 바이너리 형식 (`.dat`)
- 텍스트 형식
- 커스텀 로더 지원

#### 1.6 compiler/ - 모델 컴파일러
**위치**: `nntrainer/compiler/`

**역할**: 다양한 형식의 모델 파일을 NNTrainer 형식으로 변환

**지원 형식**:
- **INI**: NNTrainer 설정 파일 (`.ini`)
- **ONNX**: ONNX 모델 (`.onnx`)
- **TensorFlow Lite**: TFLite 모델 (`.tflite`)

**주요 컴포넌트**:
- **IniInterpreter**: INI 파일 인터프리터
- **OnnxInterpreter**: ONNX 인터프리터
- **TfLiteInterpreter**: TFLite 인터프리터
- **Realizer**: 모델 구조 변환기

#### 1.7 graph/ - 계산 그래프
**위치**: `nntrainer/graph/`

**역할**: 레이어 간 연결 관계를 그래프로 관리

**주요 기능**:
- 레이어 간 의존성 분석
- 순서 결정 (Topological Sort)
- 다중 입력/출력 처리

#### 1.8 utils/ - 유틸리티
**위치**: `nntrainer/utils/`

**주요 유틸리티**:
- **BaseProperties**: 속성 관리
- **Profiler**: 성능 프로파일링
- **Tracer**: 디버깅 추적
- **ThreadPool**: 스레드 풀 관리
- **FP16**: FP16 변환 유틸리티
- **SIMD**: SIMD 최적화 유틸리티

---

### 2. api/ - API 인터페이스

#### 2.1 C API (capi/)
**위치**: `api/capi/`

**역할**: Tizen 플랫폼을 위한 C API 제공

**주요 헤더**: `api/capi/include/nntrainer.h`

**주요 함수**:
```c
// 모델 생성
ml_train_model_h ml_train_model_create(const char *config_file);
int ml_train_model_compile(ml_train_model_h model);
int ml_train_model_initialize(ml_train_model_h model);

// 학습
int ml_train_model_train(ml_train_model_h model);

// 추론
int ml_train_model_forward(ml_train_model_h model, ...);

// 가중치 관리
int ml_train_model_save(ml_train_model_h model, const char *path);
int ml_train_model_load(ml_train_model_h model, const char *path);
```

#### 2.2 C++ API (ccapi/)
**위치**: `api/ccapi/`

**역할**: C++ 플랫폼을 위한 고수준 API 제공

**주요 헤더**:
- `api/ccapi/include/model.h`: 모델 인터페이스
- `api/ccapi/include/layer.h`: 레이어 인터페이스
- `api/ccapi/include/optimizer.h`: 옵티마이저 인터페이스
- `api/ccapi/include/dataset.h`: 데이터셋 인터페이스
- `api/ccapi/include/tensor_api.h`: 텐서 API

**사용 예시**:
```cpp
#include <model.h>
#include <layer.h>
#include <optimizer.h>

auto model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);
model->addLayer(ml::train::createLayer("input", {"input_shape=1:1:784"}));
model->addLayer(ml::train::createLayer("fully_connected", {"unit=10"}));
auto optimizer = ml::train::createOptimizer("adam", {"learning_rate=0.001"});
model->setOptimizer(std::move(optimizer));
model->compile();
model->initialize();
model->train();
```

---

### 3. Applications/ - 예제 애플리케이션

**위치**: `Applications/`

**역할**: 다양한 사용 사례를 보여주는 예제 애플리케이션

#### 주요 예제

##### 3.1 CausalLM
- **목적**: LLM (Large Language Model) 추론
- **모델**: Qwen3, Llama 등
- **특징**: Transformer 기반 인과 언어 모델

##### 3.2 MNIST
- **목적**: 손글씨 숫자 분류
- **모델**: 간단한 CNN
- **데이터**: MNIST 데이터셋

##### 3.3 ResNet
- **목적**: 이미지 분류
- **모델**: ResNet 아키텍처
- **데이터**: CIFAR-10 등

##### 3.4 VGG
- **목적**: 이미지 분류
- **모델**: VGG 아키텍처

##### 3.5 YOLOv2/YOLOv3
- **목적**: 객체 탐지
- **모델**: YOLO 아키텍처

##### 3.6 KNN
- **목적**: K-Nearest Neighbor 분류
- **특징**: 학습 없는 분류 알고리즘

##### 3.7 Custom
- **목적**: 커스텀 레이어/옵티마이저 예제
- **구성**:
  - `LayerPlugin/`: 커스텀 레이어 플러그인
  - `OptimizerPlugin/`: 커스텀 옵티마이저 플러그인
  - `LayerClient/`: 커스텀 레이어 사용 예제

##### 3.8 TransferLearning
- **목적**: 전이 학습 예제
- **기법**: 사전 학습된 모델을 파인튜닝

##### 3.9 ReinforcementLearning
- **목적**: 강화 학습 예제
- **알고리즘**: Deep Q-Network (DQN)

##### 3.10 ProductRatings
- **목적**: 제품 평점 예측
- **모델**: 추천 시스템

---

### 4. test/ - 테스트 코드

**위치**: `test/`

**구조**:
```
test/
├── unittest/              # 단위 테스트
│   ├── layers/           # 레이어 테스트
│   ├── models/            # 모델 테스트
│   ├── datasets/          # 데이터셋 테스트
│   ├── compiler/          # 컴파일러 테스트
│   └── memory/            # 메모리 관리 테스트
├── ccapi/                 # C++ API 테스트
├── tizen_capi/            # Tizen C API 테스트
└── integration_tests/     # 통합 테스트
```

**테스트 프레임워크**:
- **GTest**: Google Test Framework
- **SSAT**: Simple Shell Automated Test

---

### 5. 빌드 시스템

#### 5.1 Meson 빌드 시스템
**빌드 도구**: Meson + Ninja

**주요 빌드 파일**:
- `meson.build`: 루트 빌드 파일
- 각 디렉토리의 `meson.build`: 서브프로젝트 빌드 설정

**빌드 옵션**:
```bash
meson build
ninja -C build
```

**플랫폼별 빌드**:
- **Tizen**: `meson build -Dplatform=tizen`
- **Android**: `meson build -Dplatform=android`
- **Ubuntu**: `meson build` (기본)

#### 5.2 의존성 관리
**서브프로젝트** (`subprojects/`):
- **googletest**: 테스트 프레임워크
- **iniparser**: INI 파일 파서
- **ggml**: 텐서 연산 라이브러리
- **clblast**: OpenCL BLAS
- **ruy**: 행렬 곱셈 최적화

**외부 의존성**:
- OpenBLAS
- TensorFlow Lite
- NNStreamer (선택적)

---

## API 구조

### API 계층 구조

```
┌─────────────────────────────────────┐
│   Applications (사용자 코드)          │
├─────────────────────────────────────┤
│   C++ API (ccapi/)                  │  ← 고수준 API
├─────────────────────────────────────┤
│   C API (capi/)                     │  ← Tizen 공식 API
├─────────────────────────────────────┤
│   Core Framework (nntrainer/)      │  ← 핵심 구현
└─────────────────────────────────────┘
```

### API 사용 흐름

#### C++ API 사용
```cpp
// 1. 모델 생성
auto model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);

// 2. 레이어 추가
model->addLayer(ml::train::createLayer("input", {"input_shape=1:1:784"}));
model->addLayer(ml::train::createLayer("fully_connected", {"unit=128"}));
model->addLayer(ml::train::createLayer("fully_connected", {"unit=10"}));

// 3. 옵티마이저 설정
auto optimizer = ml::train::createOptimizer("adam", {"learning_rate=0.001"});
model->setOptimizer(std::move(optimizer));

// 4. 모델 속성 설정
model->setProperty({"batch_size=32", "epochs=10"});

// 5. 컴파일 및 초기화
model->compile();
model->initialize();

// 6. 데이터셋 설정
auto dataset = ml::train::createDataset(...);
model->setDataset(ml::train::DatasetModeType::MODE_TRAIN, std::move(dataset));

// 7. 학습
model->train();

// 8. 가중치 저장
model->save("model.bin", ml::train::ModelFormat::MODEL_FORMAT_BIN);
```

#### C API 사용
```c
// 1. 모델 생성
ml_train_model_h model;
ml_train_model_create_with_file("model.ini", &model);

// 2. 컴파일 및 초기화
ml_train_model_compile(model, NULL);
ml_train_model_initialize(model, NULL);

// 3. 학습
ml_train_model_train(model, NULL, NULL);

// 4. 가중치 저장
ml_train_model_save(model, "model.bin", ML_TRAIN_MODEL_FORMAT_BIN);
```

---

## 개발 워크플로우

### 1. 새 레이어 추가하기

#### 단계 1: 레이어 클래스 생성
```cpp
// nntrainer/layers/my_layer.h
class MyLayer : public LayerImpl {
public:
    void finalize(InitLayerContext &context) override;
    void forwarding(RunLayerContext &context, bool training) override;
    void calcDerivative(RunLayerContext &context) override;
    void calcGradient(RunLayerContext &context) override;
    void setProperty(const std::vector<std::string> &values) override;
    const std::string getType() const override { return "my_layer"; }
    
    inline static const std::string type = "my_layer";
};
```

#### 단계 2: 레이어 구현
```cpp
// nntrainer/layers/my_layer.cpp
void MyLayer::finalize(InitLayerContext &context) {
    // 텐서 차원 설정, 가중치 초기화
}

void MyLayer::forwarding(RunLayerContext &context, bool training) {
    // 순전파 구현
}
```

#### 단계 3: 레이어 등록
```cpp
// nntrainer/layers/meson.build에 추가
my_layer_src = files('my_layer.cpp')
my_layer_dep = declare_dependency(...)
```

#### 단계 4: 테스트 작성
```cpp
// test/unittest/layers/my_layer_test.cpp
TEST(nntrainer_layer, my_layer_01) {
    // 테스트 코드
}
```

### 2. 새 옵티마이저 추가하기

#### 단계 1: 옵티마이저 클래스 생성
```cpp
// nntrainer/optimizers/my_optimizer.h
class MyOptimizer : public Optimizer {
public:
    void applyGradient(RunOptimizerContext &context) override;
    void setProperty(const std::vector<std::string> &values) override;
};
```

#### 단계 2: 옵티마이저 구현
```cpp
// nntrainer/optimizers/my_optimizer.cpp
void MyOptimizer::applyGradient(RunOptimizerContext &context) {
    // 경사 적용 로직
}
```

### 3. 새 애플리케이션 추가하기

#### 단계 1: 디렉토리 생성
```
Applications/MyApp/
├── jni/
│   ├── main.cpp
│   └── meson.build
└── res/
    └── model.ini
```

#### 단계 2: 메인 코드 작성
```cpp
// Applications/MyApp/jni/main.cpp
#include <model.h>
#include <layer.h>

int main() {
    auto model = ml::train::createModel(...);
    // 모델 구성
    return 0;
}
```

#### 단계 3: 빌드 파일 작성
```meson
# Applications/MyApp/jni/meson.build
executable('nntrainer_myapp',
    'main.cpp',
    dependencies: [nntrainer_ccapi_dep],
    install: true
)
```

#### 단계 4: 루트 빌드 파일에 추가
```meson
# Applications/meson.build
subdir('MyApp/jni')
```

---

## 테스트 구조

### 테스트 실행
```bash
# 전체 테스트 실행
ninja -C build test

# 특정 테스트 실행
./build/test/unittest/layers/test_layers
```

### 테스트 작성 가이드
- **양성 테스트**: 정상 동작 확인
- **음성 테스트**: 오류 처리 확인
- **경계값 테스트**: 경계 조건 확인

**테스트 비율**: 음성 테스트 ≥ 양성 테스트

---

## 예제 애플리케이션

### 애플리케이션 실행 방법

#### 1. 빌드
```bash
meson build
ninja -C build
```

#### 2. 실행
```bash
# MNIST 예제
cd build/Applications/MNIST/jni
./nntrainer_mnist

# CausalLM 예제
cd build/Applications/CausalLM
./nntr_causallm /path/to/model/
```

### 애플리케이션별 특징

| 애플리케이션 | 목적 | 주요 레이어 | 데이터셋 |
|------------|------|-----------|---------|
| MNIST | 손글씨 분류 | Conv2D, FC | MNIST |
| ResNet | 이미지 분류 | ResNet 블록 | CIFAR-10 |
| CausalLM | 텍스트 생성 | Transformer | 텍스트 |
| YOLOv2 | 객체 탐지 | YOLO | 이미지 |
| KNN | 분류 | CentroidKNN | 특징 벡터 |

---

## 확장 가이드

### 1. 커스텀 레이어 플러그인
**위치**: `Applications/Custom/LayerPlugin/`

**특징**:
- 동적 라이브러리로 컴파일
- 런타임에 로드 가능
- 메인 프레임워크 재컴파일 불필요

**사용 방법**:
```cpp
// 레이어 플러그인 로드
app_context->loadCustomLayer("libmy_layer.so");

// 레이어 사용
model->addLayer(ml::train::createLayer("my_layer", {...}));
```

### 2. 커스텀 옵티마이저 플러그인
**위치**: `Applications/Custom/OptimizerPlugin/`

**특징**:
- 동적 라이브러리로 컴파일
- 런타임에 로드 가능

### 3. 백본 모델 통합
**INI 파일에서 백본 사용**:
```ini
[backbone_layer]
backbone = resnet.ini
trainable = false
```

**외부 프레임워크 백본**:
- TensorFlow Lite: `.tflite` 파일 직접 사용
- NNStreamer: 다양한 형식 지원 (`.pb`, `.pt`, `.circle` 등)

---

## 설정 파일 형식

### INI 파일 구조
```ini
[Model]
type = NeuralNetwork
epochs = 10
loss = cross
save_path = "model.bin"
batch_size = 32

[Optimizer]
type = adam
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-7

[LearningRateScheduler]
type = constant
learning_rate = 0.001

[train_set]
type = file
path = "train.dat"

[valid_set]
type = file
path = "valid.dat"

[layer1]
type = input
input_shape = 1:28:28

[layer2]
type = conv2d
input_layers = layer1
filters = 32
kernel_size = 3,3
activation = relu

[layer3]
type = fully_connected
input_layers = layer2
unit = 10
activation = softmax
```

---

## 메모리 관리

### 메모리 플래너
- **BasicPlanner**: 기본 메모리 할당
- **OptimizedV1/V2/V3Planner**: 최적화된 메모리 할당

### 메모리 최적화 기법
- **Lazy Evaluation**: 지연 계산으로 메모리 복사 최소화
- **Memory Pool**: 메모리 풀을 통한 재사용
- **Cache Pool**: 캐시 풀을 통한 효율적 관리

---

## 성능 최적화

### CPU 최적화
- **SIMD**: AVX2, NEON 등 SIMD 명령어 활용
- **멀티스레딩**: OpenMP를 통한 병렬 처리
- **CBLAS**: BLAS 라이브러리 활용

### GPU 최적화
- **OpenCL**: GPU 가속 연산
- **CLBlast**: OpenCL BLAS 라이브러리

### 양자화
- **FP16**: 반정밀도 부동소수점
- **INT4/INT6**: 정수 양자화

---

## 디버깅 및 프로파일링

### 프로파일러 사용
```cpp
#include <profiler.h>

Profiler profiler;
profiler.start("operation_name");
// 연산 수행
profiler.end("operation_name");
profiler.print();
```

### 트레이서 사용
```cpp
#include <tracer.h>

Tracer tracer;
tracer.trace("layer_name", tensor_data);
```

### 모델 요약 출력
```cpp
model->summarize(std::cout, ML_TRAIN_SUMMARY_MODEL);
```

---

## 플랫폼별 특성

### Tizen
- **API**: C API 사용
- **빌드**: GBS (GNU Build System)
- **패키징**: RPM 패키지

### Android
- **API**: C++ API 또는 JNI
- **빌드**: NDK 빌드
- **패키징**: AAR 또는 APK

### Ubuntu/Linux
- **API**: C++ API
- **빌드**: Meson
- **패키징**: DEB 패키지

---

## 참고 자료

### 공식 문서
- [Getting Started Guide](getting-started.md)
- [How to Create Model](how-to-create-model.md)
- [Configuration INI](configuration-ini.md)
- [Coding Convention](coding-convention.md)
- [Contributing Guide](contributing.md)

### 외부 리소스
- [GitHub Repository](https://github.com/nnstreamer/nntrainer)
- [NNStreamer Project](https://github.com/nnstreamer/nnstreamer)
- [Tizen Developer](https://developer.tizen.org/)

### 논문
- [A New Frontier of AI: On-Device AI Training and Personalization](https://dl.acm.org/doi/abs/10.1145/3639477.3639716)
- [NNTrainer: Light-Weight On-Device Training Framework](https://arxiv.org/pdf/2206.04688.pdf)

---

## FAQ

### Q1: 새 레이어를 추가하려면?
A: `nntrainer/layers/`에 레이어 클래스를 만들고, `LayerImpl`을 상속받아 구현하세요. 빌드 파일에 추가하고 테스트를 작성하세요.

### Q2: 모델을 파일로 저장하려면?
A: `model->save(path, format)` 메서드를 사용하세요. 바이너리 형식(`MODEL_FORMAT_BIN`) 또는 INI 형식을 지원합니다.

### Q3: 외부 모델을 사용하려면?
A: ONNX 또는 TFLite 형식의 모델을 `compiler/`를 통해 변환하거나, 백본 레이어를 사용하여 직접 통합할 수 있습니다.

### Q4: 성능을 최적화하려면?
A: SIMD 최적화, 멀티스레딩, GPU 가속(OpenCL), 양자화 등을 활용하세요. 프로파일러로 병목 지점을 찾아 최적화하세요.

### Q5: 메모리 부족 문제는?
A: 메모리 플래너를 최적화된 버전으로 변경하거나, 배치 크기를 줄이거나, 양자화를 사용하세요.

---

## 결론

이 문서는 NNTrainer 프로젝트의 전체 구조를 종합적으로 설명합니다. 각 컴포넌트의 역할과 상호작용을 이해하고, 새로운 기능을 추가하거나 기존 기능을 확장할 때 이 가이드를 참고하시기 바랍니다.

추가 질문이나 개선 사항이 있으면 이슈를 등록하거나 PR을 제출해 주세요.
