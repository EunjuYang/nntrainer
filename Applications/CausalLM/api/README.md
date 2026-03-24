# CausalLM API

This directory contains the C API for CausalLM application, designed to provide a simple interface for loading and running Large Language Models (LLMs) and Embedding Models on various backends, including Android.

## Overview

The API provides functionality to:
- Initialize and configure the CausalLM environment.
- Load pre-trained models with specific quantization settings.
- Run text generation (`runModel`) or embedding inference (`runModelFloat`) using a unified `loadModel` interface.
- Retrieve performance metrics (token counts, duration).

## Build & Integration

The CausalLM API is built as a separate shared library `libcausallm_api.so`, which depends on the core logic in `libcausallm_core.so`.

### Build Artifacts (Android)

- **`libcausallm_core.so`**: Contains the core LLM implementation (Model, Layers, etc.).
- **`libcausallm_api.so`**: Contains the C API implementation (`causal_lm_api.cpp`) and configuration helpers (`model_config.cpp`).

### Linking

When integrating this API into your application (e.g., via JNI), you must link against both libraries:
1.  `libcausallm_api.so`
2.  `libcausallm_core.so`
3.  `libnntrainer.so` (Dependency)

## Directory Structure & Model Loading

The API strictly relies on registered model types and quantization settings to locate model files. There are two modes of loading, depending on how the model is registered within the library.

**Path Convention:** `./models/{ModelKey}{QuantizationSuffix}/`

- **ModelKey**: Derived from `ModelType` (e.g., `qwen3-0.6b`).
- **QuantizationSuffix**: Derived from `ModelQuantizationType` (e.g., `-w16a16`).

### 1. Internal/Embedded Configuration (Pre-configured)

Some model configurations (including architecture, tokenizer settings, and generation parameters) are embedded directly into the CausalLM library code (via `model_config.cpp`). This protects the model specifications and simplifies deployment.

- **Required Files:**
    - **Weight Binary File**: The actual model weights (e.g., `qwen3-0.6b-fp32.bin`). The filename is hardcoded in the internal configuration.
    - **Tokenizer Files**: `tokenizer.json` / `vocab.json` (if required by the specific tokenizer implementation).

- **Ignored Files:**
    - `config.json`, `nntr_config.json`, `generation_config.json` are **NOT** loaded from the disk even if they exist.

### 2. External/File-based Configuration

For registered model types that do not have a specific hardcoded configuration for the requested quantization, the API falls back to loading configuration files from the directory.

- **Required Files:**
    - **`config.json`**: Model architecture configuration (HuggingFace format).
    - **`nntr_config.json`**: NNTrainer specific configuration.
        - Must contain `"model_file_name"` field pointing to the binary weight file.
    - **Weight Binary File**: The file specified in `nntr_config.json`.
    - **`generation_config.json`**: (Optional) Generation parameters.
    - **Tokenizer Files**: `tokenizer.json` / `vocab.json`.

### 3. Embedding Model Configuration

Embedding models support both internal and external configuration, following the same two-mode pattern as CausalLM models.

#### 3a. Internal/Embedded Configuration (Pre-configured)

When an embedding model is registered via `model_config.cpp` with `modules_config` set, all configuration — including the module pipeline — is embedded in the library code. Only the tokenizer and weight files are needed on disk:

```
./models/embedding-qwen3-w16a16/
  ├── embedding-qwen3-fp16.bin  # Weight file (name hardcoded in internal config)
  └── tokenizer.json            # Tokenizer file
```

- `config.json`, `nntr_config.json`, `modules.json`, and module config directories are **NOT** needed.
- If `modules_config` is not set but `module_config_path` is set, `modules.json` and module config files are loaded from disk.

See `model_config.cpp` for a complete example (`register_embedding_qwen3()`).

#### 3b. External/File-based Configuration

When no internal configuration is registered for the requested quantization type, all configuration files must be present:

```
./models/embedding-qwen3-w16a16/
  ├── config.json               # Model architecture config (HuggingFace format)
  ├── nntr_config.json          # NNTrainer config (model_type must be "Embedding")
  ├── modules.json              # Module pipeline (Transformer -> Pooling -> Normalize)
  ├── 1_Pooling/config.json     # Pooling module config
  ├── 2_Normalize/config.json   # Normalize module config (optional)
  ├── pytorch_model.bin         # Weight file (name specified in nntr_config.json)
  └── tokenizer.json            # Tokenizer file
```

**Note:** When `debug_mode` is enabled in `setOptions`, the API will attempt to validate the existence of the required files for the resolved mode during initialization.

## API Reference

The main header file is `causal_lm_api.h`.

### Enums

#### `ErrorCode`
Return codes for API functions.
- `CAUSAL_LM_ERROR_NONE`: Success.
- `CAUSAL_LM_ERROR_INVALID_PARAMETER`: Invalid argument provided.
- `CAUSAL_LM_ERROR_MODEL_LOAD_FAILED`: Failed to load the model.
- `CAUSAL_LM_ERROR_INFERENCE_FAILED`: Inference execution failed.
- `CAUSAL_LM_ERROR_NOT_INITIALIZED`: API not initialized or model not loaded.
- `CAUSAL_LM_ERROR_INFERENCE_NOT_RUN`: Metrics requested before inference run.

#### `BackendType`
Target backend for computation.
- `CAUSAL_LM_BACKEND_CPU`: CPU execution (default).
- `CAUSAL_LM_BACKEND_GPU`: GPU execution (Planned).
- `CAUSAL_LM_BACKEND_NPU`: NPU execution (Planned).

#### `ModelType`
Supported pre-defined model types. Includes both text generation and embedding models.

**Text Generation (CausalLM):**
Supported pre-defined model types.
- `CAUSAL_LM_MODEL_QWEN3_0_6B`: Qwen3 0.6B model.

**Embedding (SentenceTransformer):**
- `CAUSAL_LM_MODEL_EMBEDDING_QWEN3`: Qwen3-based embedding model.

#### `ModelQuantizationType`
Supported quantization formats.
- `CAUSAL_LM_QUANTIZATION_W4A32`: 4-bit weights, 32-bit activations.
- `CAUSAL_LM_QUANTIZATION_W16A16`: 16-bit weights, 16-bit activations (FP16).
- `CAUSAL_LM_QUANTIZATION_W8A16`: 8-bit weights, 16-bit activations.
- `CAUSAL_LM_QUANTIZATION_W32A32`: 32-bit weights, 32-bit activations (FP32).

### Functions

#### `ErrorCode setOptions(Config config)`
Sets global configuration options.
- **config**: Structure containing options like `use_chat_template`, `debug_mode`, and `verbose`.

#### `ErrorCode loadModel(BackendType compute, ModelType modeltype, ModelQuantizationType quant_type)`
Loads a registered model. Works for both text generation and embedding models.
- **compute**: Backend to use.
- **modeltype**: Model type enum (CausalLM or Embedding).
- **quant_type**: Quantization type.

#### `ErrorCode runModel(const char *inputTextPrompt, const char **outputText)`
Runs text generation on the loaded CausalLM model.
- **inputTextPrompt**: The input text/prompt.
- **outputText**: Pointer to store the result string.

#### `ErrorCode runModel(const char *inputTextPrompt, float **outputData, unsigned int *outputDim, unsigned int *outputLength)`
Runs embedding inference on the loaded embedding model.
- **inputTextPrompt**: The input text to encode.
- **outputData**: Pointer to receive the embedding vector data. Managed by the API, do not free.
- **outputDim**: Pointer to receive the embedding dimension (e.g., 1024).
- **outputLength**: Pointer to receive the number of embedding vectors (batch size).
- Returns `CAUSAL_LM_ERROR_INVALID_PARAMETER` if a CausalLM model is loaded. Use `runModel()` instead.

#### `ErrorCode getPerformanceMetrics(PerformanceMetrics *metrics)`
Retrieves performance metrics of the last run.
- **metrics**: Pointer to `PerformanceMetrics` struct to be filled.
- `prefill_tokens`, `prefill_duration_ms`: Stats for prompt processing.
- `generation_tokens`, `generation_duration_ms`: Stats for token generation.
- `total_duration_ms`: Total execution time from start to finish.
- `peak_memory_kb`: Peak resident set size (memory usage) in KB.

## Usage Example (Text Generation)

```c
#include "causal_lm_api.h"
#include <stdio.h>

int main() {
    // 1. Set Options
    Config config;
    config.use_chat_template = true;
    config.debug_mode = false;
    config.verbose = false;
    setOptions(config);

    // 2. Load Model
    // Automatically looks for files in "./models/qwen3-0.6b-w16a16/"
    ErrorCode err = loadModel(CAUSAL_LM_BACKEND_CPU, 
                              CAUSAL_LM_MODEL_QWEN3_0_6B, 
                              CAUSAL_LM_QUANTIZATION_W16A16);
    
    if (err != CAUSAL_LM_ERROR_NONE) {
        printf("Failed to load model\n");
        return -1;
    }

    // 3. Run Inference
    const char* output = NULL;
    err = runModel("Hello, how are you?", &output);
    
    if (err == CAUSAL_LM_ERROR_NONE) {
        printf("Response: %s\n", output);
    }

    // 4. Check Metrics
    PerformanceMetrics metrics;
    if (getPerformanceMetrics(&metrics) == CAUSAL_LM_ERROR_NONE) {
        printf("Generated %d tokens in %.2f ms\n", 
               metrics.generation_tokens, metrics.generation_duration_ms);
    }

    return 0;
}
```

## Usage Example (Embedding)

```c
#include "causal_lm_api.h"
#include <stdio.h>

int main() {
    // 1. Set Options
    Config config;
    config.use_chat_template = false;
    config.debug_mode = false;
    config.verbose = false;
    setOptions(config);

    // 2. Load Embedding Model (same loadModel API)
    ErrorCode err = loadModel(CAUSAL_LM_BACKEND_CPU,
                              CAUSAL_LM_MODEL_EMBEDDING_QWEN3,
                              CAUSAL_LM_QUANTIZATION_W16A16);

    if (err != CAUSAL_LM_ERROR_NONE) {
        printf("Failed to load embedding model\n");
        return -1;
    }

    // 3. Run Inference (float vector output)
    float *emb_data = NULL;
    unsigned int emb_dim = 0;
    unsigned int emb_length = 0;

    err = runModelFloat("Hello, world!", &emb_data, &emb_dim, &emb_length);

    if (err == CAUSAL_LM_ERROR_NONE) {
        printf("Embedding dim: %u, batch: %u\n", emb_dim, emb_length);
        printf("Embedding: [");
        for (unsigned int i = 0; i < 5 && i < emb_dim; ++i) {
            printf("%.6f%s", emb_data[i], (i < 4) ? ", " : "");
        }
        printf(", ...]\n");
    }

    // Note: runModel() would return CAUSAL_LM_ERROR_INVALID_PARAMETER
    //       for embedding models.

    return 0;
}
```

## Registering Embedding Models with Internal Configuration

To embed model configuration parameters directly in the library (avoiding `config.json` / `nntr_config.json` on disk), add a registration function in `model_config.cpp`. This is useful for protecting model specifications and simplifying deployment.

### Example: `model_config.cpp`

```c
#include "model_config_internal.h"
#include <climits>
#include <cstring>

static void register_embedding_qwen3() {
    // 1. Architecture Config (from config.json)
    ModelArchConfig ac;
    memset(&ac, 0, sizeof(ModelArchConfig));

    ac.vocab_size = 151936;
    ac.hidden_size = 1024;
    ac.intermediate_size = 3072;
    ac.num_hidden_layers = 28;
    ac.num_attention_heads = 16;
    ac.head_dim = 128;
    ac.num_key_value_heads = 8;
    ac.max_position_embeddings = 32768;
    ac.rope_theta = 1000000.0f;
    ac.rms_norm_eps = 1e-06f;
    ac.tie_word_embeddings = true;
    ac.sliding_window = UINT_MAX;
    // Architecture must match the base model (not the embedding variant)
    strncpy(ac.architecture, "Qwen3ForCausalLM",
            sizeof(ac.architecture) - 1);
    ac.bos_token_id = 151643;
    ac.eos_token_ids[0] = 151645;
    ac.num_eos_token_ids = 1;

    registerModelArchitecture("Embedding-Qwen3-Arch", ac);

    // 2. Runtime Config (from nntr_config.json)
    ModelRuntimeConfig rc;
    memset(&rc, 0, sizeof(ModelRuntimeConfig));

    rc.batch_size = 1;
    strncpy(rc.model_type, "Embedding", sizeof(rc.model_type) - 1);
    strncpy(rc.model_tensor_type, "FP16-FP16",
            sizeof(rc.model_tensor_type) - 1);
    rc.init_seq_len = 512;
    rc.max_seq_len = 8192;
    rc.num_to_generate = 0;  // Not used for embedding
    strncpy(rc.embedding_dtype, "FP16", sizeof(rc.embedding_dtype) - 1);
    strncpy(rc.fc_layer_dtype, "FP16", sizeof(rc.fc_layer_dtype) - 1);
    strncpy(rc.model_file_name, "embedding-qwen3-fp16.bin",
            sizeof(rc.model_file_name) - 1);
    strncpy(rc.tokenizer_file, "tokenizer.json",
            sizeof(rc.tokenizer_file) - 1);

    // 3. Inline modules pipeline (replaces modules.json + module configs)
    //    Each entry has: idx, type, and optional config object
    strncpy(rc.modules_config,
      "["
      "{\"idx\":0,\"type\":\"sentence_transformers.models.Transformer\"},"
      "{\"idx\":1,\"type\":\"sentence_transformers.models.Pooling\","
       "\"config\":{\"word_embedding_dimension\":1024,"
                   "\"pooling_mode_mean_tokens\":true}},"
      "{\"idx\":2,\"type\":\"sentence_transformers.models.Normalize\"}"
      "]",
      sizeof(rc.modules_config) - 1);

    registerModel("EMBEDDING-QWEN3-W16A16", "Embedding-Qwen3-Arch", rc);
    registerModel("EMBEDDING-QWEN3", "Embedding-Qwen3-Arch", rc);  // alias
}

int register_builtin_model_configs() {
    register_qwen3_0_6b();
    register_embedding_qwen3();
    return 0;
}
```

### What Gets Embedded vs. Loaded from Disk

| Parameter Source | Internal Config (with `modules_config`) | External Config |
|---|---|---|
| Architecture params (vocab_size, hidden_size, ...) | Embedded in code | `config.json` |
| Runtime params (batch_size, max_seq_len, ...) | Embedded in code | `nntr_config.json` |
| Module pipeline + module configs | Embedded in code (`modules_config`) | `modules.json` + `{N}_Module/config.json` |
| Tokenizer | `tokenizer.json` from disk | `tokenizer.json` from disk |
| Model weights | Binary file from disk | Binary file from disk |
