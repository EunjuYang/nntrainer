# NNTrainer API - Embedding

## Overview

NNTrainer provides an Embedding API that allows users to run embedding models
and retrieve embedding vectors for given input token indices. This API is
designed for models that contain an embedding layer, enabling efficient
token-to-vector lookups using pre-trained or trained embedding weights.

### Embedding Layer

The embedding layer (`ML_TRAIN_LAYER_TYPE_EMBEDDING`) maps discrete token
indices to dense vector representations. It is defined by two key properties:

- **`in_dim`**: Vocabulary size (the number of unique tokens)
- **`out_dim`**: Embedding dimension (the size of each embedding vector)

For an input sequence of token indices with shape `(batch, 1, 1, seq_len)`,
the embedding layer produces output vectors with shape
`(batch, 1, seq_len, out_dim)`.

---

## C++ API

### `Model::embedding()`

```cpp
virtual std::vector<float *>
embedding(unsigned int batch, const std::vector<float *> &input) = 0;
```

#### Description

Runs the model in inference mode and returns embedding vectors for the given
input token indices. The model should contain an embedding layer and must be
compiled, initialized, and have weights loaded before calling this method.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `batch` | `unsigned int` | Batch size of the current input |
| `input` | `const std::vector<float *> &` | Input data containing token indices (stored as float values) |

#### Return Value

Returns `std::vector<float *>` containing pointers to the output embedding
vectors. The output memory is managed by the model and must **not** be freed
by the caller.

#### Usage Example (C++)

```cpp
#include <model.h>
#include <layer.h>

// 1. Create the model
auto model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);

// 2. Add an input layer
auto input_layer = ml::train::createLayer("input",
  {"name=input", "input_shape=1:1:1:10"});
model->addLayer(input_layer);

// 3. Add an embedding layer
auto embedding_layer = ml::train::createLayer("embedding",
  {"name=embedding", "in_dim=1000", "out_dim=128", "input_layers=input"});
model->addLayer(embedding_layer);

// 4. Compile and initialize the model
model->compile(ml::train::ExecutionMode::INFERENCE);
model->initialize(ml::train::ExecutionMode::INFERENCE);

// 5. Load pre-trained weights (if available)
// model->load("embedding_weights.bin");

// 6. Prepare input data (token indices as float)
unsigned int batch = 1;
unsigned int seq_len = 10;
float input_data[10] = {1.0f, 5.0f, 23.0f, 42.0f, 7.0f,
                         100.0f, 55.0f, 3.0f, 88.0f, 12.0f};
std::vector<float *> input = {input_data};

// 7. Run embedding
std::vector<float *> output = model->embedding(batch, input);

// 8. Access embedding vectors
// output[0] points to embedding vectors of shape (1, 1, 10, 128)
// output[0][0..127] = embedding vector for token 1
// output[0][128..255] = embedding vector for token 5
// ...
```

---

## C API

### `ml_train_model_run_embedding()`

```c
int ml_train_model_run_embedding(ml_train_model_h model,
                                  const float *input,
                                  unsigned int input_length,
                                  float *output,
                                  unsigned int *output_length);
```

#### Description

Runs an embedding model and retrieves embedding vectors for the given input
token indices. The model must contain an embedding layer and should be compiled
before calling this function.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `ml_train_model_h` | The NNTrainer model handle |
| `input` | `const float *` | Input data containing token indices (as float) |
| `input_length` | `unsigned int` | Number of elements in the input array |
| `output` | `float *` | Pre-allocated output buffer for embedding vectors |
| `output_length` | `unsigned int *` | Receives the number of elements written to output |

#### Return Values

| Value | Description |
|-------|-------------|
| `ML_ERROR_NONE` | Successful |
| `ML_ERROR_NOT_SUPPORTED` | Not supported |
| `ML_ERROR_INVALID_PARAMETER` | Invalid parameter |

#### Usage Example (C)

```c
#include <nntrainer.h>
#include <stdlib.h>
#include <stdio.h>

int main() {
  int status;
  ml_train_model_h model;
  ml_train_layer_h input_layer, embedding_layer;

  // 1. Create the model
  status = ml_train_model_construct(&model);

  // 2. Create and configure input layer
  status = ml_train_layer_create(&input_layer, ML_TRAIN_LAYER_TYPE_INPUT);
  status = ml_train_layer_set_property(input_layer,
    "name=input", "input_shape=1:1:1:10", NULL);
  status = ml_train_model_add_layer(model, input_layer);

  // 3. Create and configure embedding layer
  status = ml_train_layer_create(&embedding_layer,
    ML_TRAIN_LAYER_TYPE_EMBEDDING);
  status = ml_train_layer_set_property(embedding_layer,
    "name=embedding", "in_dim=1000", "out_dim=128",
    "input_layers=input", NULL);
  status = ml_train_model_add_layer(model, embedding_layer);

  // 4. Compile model
  status = ml_train_model_compile(model, NULL);

  // 5. Prepare input (token indices as float)
  float input_data[10] = {1.0f, 5.0f, 23.0f, 42.0f, 7.0f,
                           100.0f, 55.0f, 3.0f, 88.0f, 12.0f};
  unsigned int input_length = 10;

  // 6. Allocate output buffer (seq_len * out_dim)
  unsigned int output_buf_size = 10 * 128;  // seq_len * embedding_dim
  float *output = (float *)malloc(output_buf_size * sizeof(float));
  unsigned int output_length = 0;

  // 7. Run embedding
  status = ml_train_model_run_embedding(model, input_data, input_length,
                                         output, &output_length);
  if (status == ML_ERROR_NONE) {
    printf("Embedding output length: %u\n", output_length);
    // output[0..127] = embedding vector for token 1
    // output[128..255] = embedding vector for token 5
    // ...
  }

  // 8. Cleanup
  free(output);
  ml_train_model_destroy(model);
  return 0;
}
```

---

## Differences Between `embedding()` and `inference()`

| Feature | `embedding()` | `inference()` |
|---------|---------------|---------------|
| Purpose | Get embedding vectors from token indices | General model inference |
| Input | Token indices (integer values stored as float) | Model-dependent input data |
| Label required | No | Optional |
| Typical model | Models with embedding layer | Any trained model |
| Output | Dense embedding vectors | Model prediction output |

The `embedding()` API is essentially a specialized wrapper around `inference()`
that is optimized for the embedding use case:
- No label data is required
- The API is simpler with fewer parameters
- It makes the intent clear when working with embedding models

---

## Model Architecture for Embedding

A typical embedding model consists of:

```
Input Layer (token indices) -> Embedding Layer -> [Optional: Additional Layers]
```

### Configuration File Example (INI format)

```ini
[model]
type = NeuralNetwork
batch_size = 1

[input]
type = input
input_shape = 1:1:1:10

[embedding]
type = embedding
in_dim = 1000
out_dim = 128
input_layers = input
```

### Key Properties

| Property | Description | Example |
|----------|-------------|---------|
| `in_dim` | Vocabulary size (number of unique tokens) | `1000` |
| `out_dim` | Embedding vector dimension | `128` |
| `input_shape` | Shape of input tensor `(channel:height:width)` | `1:1:1:10` (10 tokens) |
