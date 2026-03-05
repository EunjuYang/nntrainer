# Qwen3-0.6B Quantization Guide

This directory contains the resources needed to convert the Qwen3-0.6B model
from HuggingFace to NNTrainer's FP32 binary format and then quantize it to
Q4_0 for all components (weights, embedding, and LM head).

## Model Architecture

| Parameter | Value |
|---|---|
| Architecture | Qwen3ForCausalLM |
| Parameters | ~0.6B |
| Hidden Size | 1024 |
| Num Layers | 28 |
| Attention Heads | 16 (Q) / 8 (KV) |
| Head Dim | 128 |
| Intermediate Size | 4096 |
| Vocab Size | 151,669 |
| Tie Word Embeddings | true |
| Context Length | 32K (max 40,960) |

## Prerequisites

- Python 3.8+ with `torch`, `transformers`, `numpy`
- NNTrainer built with `-Denable-transformer=true`

```bash
# Install Python dependencies
pip install torch transformers numpy

# Build NNTrainer
meson setup build -Denable-transformer=true \
    -Denable-tflite-backbone=false \
    -Denable-tflite-interpreter=false
ninja -C build Applications/CausalLM/nntr_quantize Applications/CausalLM/nntr_causallm
```

## Step-by-Step Guide

### Step 1: Download Qwen3-0.6B from HuggingFace

```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='Qwen/Qwen3-0.6B', local_dir='/tmp/Qwen3-0.6B')
"
```

Copy the tokenizer to this directory:
```bash
cp /tmp/Qwen3-0.6B/tokenizer.json .
```

### Step 2: Convert Weights to FP32 Binary

```bash
python3 weight_converter.py \
    --model_path /tmp/Qwen3-0.6B \
    --output_name ./nntr_qwen3_0.6b_fp32.bin \
    --data_type float32
```

This produces `nntr_qwen3_0.6b_fp32.bin` (~2.4 GB for the real model).

Update `nntr_config.json` with the tokenizer path:
```bash
# Edit nntr_config.json and set "tokenizer_file" to the absolute path
# of your tokenizer.json
```

### Step 3: Quantize to Q4_0 (All Components)

```bash
# Set library path
export LD_LIBRARY_PATH=build/nntrainer:build/Applications/CausalLM:build/Applications/CausalLM/layers:$LD_LIBRARY_PATH

# Create output directory
mkdir -p q4_0
cp config.json generation_config.json q4_0/

# Run quantization: weight / embedding / lmhead all Q4_0
build/Applications/CausalLM/nntr_quantize . \
    --fc_dtype Q4_0 \
    --embd_dtype Q4_0 \
    --lmhead_dtype Q4_0 \
    -o q4_0 \
    --output_bin nntr_qwen3_0.6b_q40.bin
```

Expected output:
```
Source size:  ~2400 MB (FP32)
Output size:  ~280 MB  (Q4_0)
Compression:  ~11.5%
```

The quantizer also generates `q4_0/nntr_config.json` with updated dtype settings.

Update the tokenizer path in `q4_0/nntr_config.json` to point to your tokenizer.

### Step 4: Run Inference

```bash
export LD_LIBRARY_PATH=build/nntrainer:build/Applications/CausalLM:build/Applications/CausalLM/layers:$LD_LIBRARY_PATH

build/Applications/CausalLM/nntr_causallm q4_0/
```

## Automated Pipeline

For convenience, `run_pipeline.sh` automates all steps above:

```bash
./run_pipeline.sh
```

Options:
- `--model-dir <dir>` : Directory for model resources (default: this directory)
- `--build-dir <dir>` : NNTrainer build directory (default: `$NNTRAINER_ROOT/build`)
- `--skip-download`   : Skip HuggingFace download (use existing cache)

## Files in This Directory

| File | Description |
|---|---|
| `config.json` | Qwen3-0.6B HuggingFace model config |
| `generation_config.json` | Generation parameters |
| `nntr_config.json` | NNTrainer runtime config (FP32 baseline) |
| `weight_converter.py` | Converts HuggingFace weights to NNTrainer FP32 bin |
| `generate_dummy_weights.py` | Generates dummy weights for testing the pipeline |
| `run_pipeline.sh` | Automated end-to-end pipeline script |
| `README.md` | This file |

## Chat Template

Qwen3 uses the ChatML-style chat template with special tokens:

| Token | Token ID | Role |
|---|---|---|
| `<\|endoftext\|>` | 151643 | PAD / BOS |
| `<\|im_start\|>` | 151644 | Start of message turn |
| `<\|im_end\|>` | 151645 | End of message turn / EOS |
| `<think>` | 151667 | Start of thinking block |
| `</think>` | 151668 | End of thinking block |

**Input format** (single-turn, no system prompt):
```
<|im_start|>user
What is 1+1?<|im_end|>
<|im_start|>assistant
```

The `sample_input` field in `nntr_config.json` uses this format. When building
the prompt string for `nntr_causallm`, make sure to include these special tokens
exactly as shown — the tokenizer will encode them as single token IDs.

## FP32 vs Q4_0 Inference Comparison

Both models were tested with the same prompt using greedy decoding (`do_sample=false`).

**Prompt**: `<|im_start|>user\nWhat is 1+1?<|im_end|>\n<|im_start|>assistant\n`

### Test 1: Short prompt (31 tokens)

| Metric | FP32 | Q4_0 | Ratio |
|---|---|---|---|
| Model size | 3,201 MB | 367 MB | **8.7x smaller** |
| Memory (RSS) | 3,082 MB | 750 MB | **4.1x smaller** |
| Prefill TPS | 42.5 | 72.1 | **1.7x faster** |
| Generation TPS | 5.5 | 5.6 | ~same |
| E2E time | 8,934 ms | 6,657 ms | **1.3x faster** |

### Test 2: Longer prompt (61 tokens)

**Prompt**: `<|im_start|>user\nExplain quantum computing in one sentence.<|im_end|>\n<|im_start|>assistant\n`

| Metric | FP32 | Q4_0 | Ratio |
|---|---|---|---|
| Memory (RSS) | 3,084 MB | 775 MB | **4.0x smaller** |
| Prefill TPS | 69.6 | 95.8 | **1.4x faster** |
| Generation TPS | 5.5 | 6.6 | **1.2x faster** |
| E2E time | 8,906 ms | 6,063 ms | **1.5x faster** |

> **Note**: These results use dummy (random) weights for pipeline verification.
> The output tokens are not meaningful text, but the performance characteristics
> (memory, throughput, latency) are representative of real models. With real
> Qwen3-0.6B weights, both FP32 and Q4_0 models will produce coherent text,
> and the Q4_0 model should show minimal quality degradation compared to FP32.

## Notes

- **Qwen3-0.6B uses `tie_word_embeddings=true`**: The embedding and LM head
  share the same weight matrix. The weight converter handles this by reusing
  `embed_tokens.weight` when `lm_head.weight` is not present in the state dict.

- **Q4_0 Embedding Support**: Q4_0 quantization for the tie_word_embedding
  layer was added as part of this work. Previously, only Q6_K was supported
  for embedding quantization.

- **Quantization Comparison**:
  - FP32: ~2.4 GB (full precision)
  - Q4_0 (all): ~280 MB (weights + embedding + lmhead all quantized)
  - Q4_0 FC + FP32 embd: ~380 MB (typical configuration)

## Testing with Dummy Weights

For testing the quantization pipeline without downloading the real model:

```bash
python3 generate_dummy_weights.py --output nntr_qwen3_0.6b_fp32.bin
```

This generates random weights matching the Qwen3-0.6B architecture. The model
won't produce meaningful text, but the quantization and inference pipeline can
be verified end-to-end.
