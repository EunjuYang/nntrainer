#!/bin/bash
# ==============================================================================
# Qwen3-0.6B Quantization Pipeline
# ==============================================================================
# This script automates the full pipeline:
#   1. Download Qwen3-0.6B from HuggingFace
#   2. Convert weights to FP32 NNTrainer bin format
#   3. Quantize to Q4_0 (weight/embedding/lmhead all Q4_0)
#   4. Run inference to verify the quantized model
#
# Prerequisites:
#   - Python 3.8+ with pip
#   - torch, transformers, numpy (installed automatically)
#   - nntrainer built with -Denable-transformer=true
#
# Usage:
#   ./run_pipeline.sh [--model-dir <dir>] [--build-dir <dir>] [--skip-download]
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NNTRAINER_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
BUILD_DIR="${NNTRAINER_ROOT}/build"
MODEL_DIR="${SCRIPT_DIR}"
HF_MODEL_ID="Qwen/Qwen3-0.6B"
HF_CACHE_DIR="/tmp/Qwen3-0.6B"
SKIP_DOWNLOAD=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-dir) MODEL_DIR="$2"; shift 2;;
        --build-dir) BUILD_DIR="$2"; shift 2;;
        --skip-download) SKIP_DOWNLOAD=true; shift;;
        --help|-h)
            echo "Usage: $0 [--model-dir <dir>] [--build-dir <dir>] [--skip-download]"
            echo ""
            echo "Options:"
            echo "  --model-dir <dir>   Directory for model resources (default: script dir)"
            echo "  --build-dir <dir>   NNTrainer build directory (default: \$NNTRAINER_ROOT/build)"
            echo "  --skip-download     Skip HuggingFace model download (use existing cache)"
            exit 0;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

NNTR_QUANTIZE="${BUILD_DIR}/Applications/CausalLM/nntr_quantize"
NNTR_CAUSALLM="${BUILD_DIR}/Applications/CausalLM/nntr_causallm"

FP32_BIN="nntr_qwen3_0.6b_fp32.bin"
Q4_BIN="nntr_qwen3_0.6b_q40.bin"

echo "============================================================"
echo "  Qwen3-0.6B Quantization Pipeline"
echo "============================================================"
echo "  NNTrainer root:  ${NNTRAINER_ROOT}"
echo "  Build dir:       ${BUILD_DIR}"
echo "  Model dir:       ${MODEL_DIR}"
echo "  HF model:        ${HF_MODEL_ID}"
echo ""

# --------------------------------------------------------------------------
# Step 0: Check prerequisites
# --------------------------------------------------------------------------
echo "[0/4] Checking prerequisites..."

if [ ! -f "${NNTR_QUANTIZE}" ]; then
    echo "ERROR: nntr_quantize not found at ${NNTR_QUANTIZE}"
    echo "  Build with: meson setup build -Denable-transformer=true && ninja -C build"
    exit 1
fi

if [ ! -f "${NNTR_CAUSALLM}" ]; then
    echo "ERROR: nntr_causallm not found at ${NNTR_CAUSALLM}"
    exit 1
fi

python3 -c "import torch; import transformers; import numpy" 2>/dev/null || {
    echo "Installing Python dependencies..."
    pip3 install torch transformers numpy
}

echo "  Prerequisites OK."
echo ""

# --------------------------------------------------------------------------
# Step 1: Download Qwen3-0.6B from HuggingFace
# --------------------------------------------------------------------------
echo "[1/4] Downloading Qwen3-0.6B from HuggingFace..."

if [ "${SKIP_DOWNLOAD}" = true ] && [ -d "${HF_CACHE_DIR}" ]; then
    echo "  Skipping download (using cached model at ${HF_CACHE_DIR})"
else
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='${HF_MODEL_ID}', local_dir='${HF_CACHE_DIR}')
print('Download complete.')
"
fi

# Copy tokenizer to model directory
if [ -f "${HF_CACHE_DIR}/tokenizer.json" ]; then
    cp "${HF_CACHE_DIR}/tokenizer.json" "${MODEL_DIR}/tokenizer.json"
    echo "  Copied tokenizer.json to ${MODEL_DIR}"
fi

echo ""

# --------------------------------------------------------------------------
# Step 2: Convert weights to FP32 NNTrainer bin format
# --------------------------------------------------------------------------
echo "[2/4] Converting weights to FP32 NNTrainer format..."

if [ -f "${MODEL_DIR}/${FP32_BIN}" ]; then
    echo "  FP32 bin already exists, skipping conversion."
else
    python3 "${SCRIPT_DIR}/weight_converter.py" \
        --model_path "${HF_CACHE_DIR}" \
        --output_name "${MODEL_DIR}/${FP32_BIN}" \
        --data_type float32

    FP32_SIZE=$(du -h "${MODEL_DIR}/${FP32_BIN}" | cut -f1)
    echo "  FP32 bin created: ${MODEL_DIR}/${FP32_BIN} (${FP32_SIZE})"
fi

# Update nntr_config.json with tokenizer path
python3 -c "
import json, os
config_path = '${MODEL_DIR}/nntr_config.json'
with open(config_path) as f:
    cfg = json.load(f)
cfg['tokenizer_file'] = '${MODEL_DIR}/tokenizer.json'
cfg['model_file_name'] = '${FP32_BIN}'
with open(config_path, 'w') as f:
    json.dump(cfg, f, indent=4)
print('  Updated nntr_config.json')
"
echo ""

# --------------------------------------------------------------------------
# Step 3: Quantize to Q4_0 (weight/embedding/lmhead all Q4_0)
# --------------------------------------------------------------------------
echo "[3/4] Quantizing to Q4_0 (all components)..."

# Set library path for runtime shared libraries
export LD_LIBRARY_PATH="${BUILD_DIR}/nntrainer:${BUILD_DIR}/Applications/CausalLM:${BUILD_DIR}/Applications/CausalLM/layers:${BUILD_DIR}/Applications/CausalLM/models/qwen3:${BUILD_DIR}/Applications/CausalLM/models/qwen3_moe:${BUILD_DIR}/Applications/CausalLM/models/qwen3_slim_moe:${BUILD_DIR}/Applications/CausalLM/models/qwen3_cached_slim_moe:${BUILD_DIR}/Applications/CausalLM/models/gpt_oss:${BUILD_DIR}/Applications/CausalLM/models/gpt_oss_cached_slim:${BUILD_DIR}/Applications/CausalLM/models/gemma3:${BUILD_DIR}/Applications/CausalLM/models/qwen2:${LD_LIBRARY_PATH:-}"

Q4_OUTPUT_DIR="${MODEL_DIR}/q4_0"
mkdir -p "${Q4_OUTPUT_DIR}"

# Copy config files to output dir
cp "${MODEL_DIR}/config.json" "${Q4_OUTPUT_DIR}/config.json"
cp "${MODEL_DIR}/generation_config.json" "${Q4_OUTPUT_DIR}/generation_config.json"

"${NNTR_QUANTIZE}" "${MODEL_DIR}" \
    --fc_dtype Q4_0 \
    --embd_dtype Q4_0 \
    --lmhead_dtype Q4_0 \
    -o "${Q4_OUTPUT_DIR}" \
    --output_bin "${Q4_BIN}"

# Update the output nntr_config.json with tokenizer path
if [ -f "${Q4_OUTPUT_DIR}/nntr_config.json" ]; then
    python3 -c "
import json
config_path = '${Q4_OUTPUT_DIR}/nntr_config.json'
with open(config_path) as f:
    cfg = json.load(f)
cfg['tokenizer_file'] = '${MODEL_DIR}/tokenizer.json'
with open(config_path, 'w') as f:
    json.dump(cfg, f, indent=4)
"
fi

echo ""

# --------------------------------------------------------------------------
# Step 4: Run inference with quantized model
# --------------------------------------------------------------------------
echo "[4/4] Running inference with quantized model..."

"${NNTR_CAUSALLM}" "${Q4_OUTPUT_DIR}"

echo ""
echo "============================================================"
echo "  Pipeline complete!"
echo "============================================================"
echo ""
echo "  FP32 model:       ${MODEL_DIR}/${FP32_BIN}"
echo "  Q4_0 model:       ${Q4_OUTPUT_DIR}/${Q4_BIN}"
echo "  Q4_0 config:      ${Q4_OUTPUT_DIR}/nntr_config.json"
echo ""
echo "  To run inference:"
echo "    export LD_LIBRARY_PATH=${BUILD_DIR}/nntrainer:${BUILD_DIR}/Applications/CausalLM:\$LD_LIBRARY_PATH"
echo "    ${NNTR_CAUSALLM} ${Q4_OUTPUT_DIR}"
echo ""
