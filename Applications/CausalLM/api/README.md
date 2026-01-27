# CausalLM API

This directory contains the C API for CausalLM application, designed to provide a simple interface for loading and running Large Language Models (LLMs) on various backends, including Android.

## Overview

The API provides functionality to:
- Initialize and configure the CausalLM environment.
- Load pre-trained models with specific quantization settings.
- Run inference (text generation) given a prompt.
- Retrieve performance metrics (token counts, duration).

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
Pre-defined model types.
- `CAUSAL_LM_MODEL_UNKNOWN`: Use custom path or name.
- `CAUSAL_LM_MODEL_QWEN3_0_6B`: Qwen3 0.6B model.

#### `ModelQuantizationType`
Supported quantization formats.
- `CAUSAL_LM_QUANTIZATION_W4A32`: 4-bit weights, 32-bit activations.
- `CAUSAL_LM_QUANTIZATION_W16A16`: 16-bit weights, 16-bit activations (FP16).
- `CAUSAL_LM_QUANTIZATION_W8A16`: 8-bit weights, 16-bit activations.
- `CAUSAL_LM_QUANTIZATION_W32A32`: 32-bit weights, 32-bit activations (FP32).

### Functions

#### `ErrorCode setOptions(Config config)`
Sets global configuration options.
- **config**: Structure containing options like `use_chat_template` and `debug_mode`.

#### `ErrorCode loadModel(BackendType compute, ModelType modeltype, ModelQuantizationType quant_type, const char *model_name_or_path)`
Loads a model.
- **compute**: Backend to use.
- **modeltype**: Specific model enum or `UNKNOWN`.
- **quant_type**: Quantization type.
- **model_name_or_path**: Path to the model directory or model name.

#### `ErrorCode runModel(const char *inputTextPrompt, const char **outputText)`
Runs inference on the loaded model.
- **inputTextPrompt**: The input text/prompt.
- **outputText**: Pointer to store the result string.

#### `ErrorCode getPerformanceMetrics(PerformanceMetrics *metrics)`
Retrieves performance metrics of the last run.
- **metrics**: Pointer to `PerformanceMetrics` struct to be filled.
- `prefill_tokens`, `prefill_duration_ms`: Stats for prompt processing.
- `generation_tokens`, `generation_duration_ms`: Stats for token generation.

## Usage Example

```c
#include "causal_lm_api.h"
#include <stdio.h>

int main() {
    // 1. Set Options
    Config config;
    config.use_chat_template = true;
    config.debug_mode = false;
    setOptions(config);

    // 2. Load Model
    // Assuming model files are in "./models/qwen3-0.6b-w16a16/"
    ErrorCode err = loadModel(CAUSAL_LM_BACKEND_CPU, 
                              CAUSAL_LM_MODEL_UNKNOWN, 
                              CAUSAL_LM_QUANTIZATION_W16A16, 
                              "./models/qwen3-0.6b-w16a16/");
    
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
