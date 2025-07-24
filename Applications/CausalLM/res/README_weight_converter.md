# Weight Converter for nntrainer

This script converts PyTorch model weights to nntrainer-compatible binary format with GGML quantization.

## Features

- **GGML-compatible quantization**:
  - Q4_0: 4-bit quantization with block size 32
  - Q6_K: 6-bit quantization with superblock size 256
  
- **CPU backend optimization**:
  - Q4_0x4: 4-way SIMD repacking
  - Q4_0x8: 8-way SIMD repacking
  
- **Performance optimizations**:
  - Parallel processing with multiprocessing
  - GPU acceleration with CuPy (optional)
  - JIT compilation with Numba (optional)
  - Vectorized operations

## Installation

```bash
# Required
pip install torch transformers numpy tqdm psutil

# Optional for better performance
pip install cupy-cuda12x  # For GPU acceleration
pip install numba         # For JIT compilation
```

## Usage

### Basic conversion

Convert a model to Q4_0 format:
```bash
python weight_converter.py /path/to/model output.bin
```

Convert a model to Q6_K format:
```bash
python weight_converter.py /path/to/model output.bin --quant-type q6_k
```

### With CPU backend optimization

Convert to Q4_0 with 8-way SIMD repacking:
```bash
python weight_converter.py /path/to/model output.bin --quant-type q4_0 --repack-format q4_0_8
```

Convert to Q4_0 with 4-way SIMD repacking:
```bash
python weight_converter.py /path/to/model output.bin --quant-type q4_0 --repack-format q4_0_4
```

### Testing and benchmarking

Run tests:
```bash
python weight_converter.py dummy dummy --test
```

Run performance benchmark:
```bash
python weight_converter.py dummy dummy --benchmark
```

## Examples

### Convert a local model
```bash
# Assuming you have a model in ./qwen3-0.5b/
python weight_converter.py ./qwen3-0.5b/ qwen3-0.5b-q4_0.bin --quant-type q4_0
```

### Convert with optimal CPU backend settings
```bash
# For nntrainer CPU backend with 8-way SIMD
python weight_converter.py ./qwen3-0.5b/ qwen3-0.5b-q4_0_8.bin --quant-type q4_0 --repack-format q4_0_8
```

### Convert from Hugging Face model
```bash
# Download and convert in one step
python weight_converter.py Qwen/Qwen2.5-0.5B qwen2.5-0.5b-q4_0.bin --quant-type q4_0
```

## Output format

The output binary file contains:
1. Magic number (4 bytes): "NNTR"
2. Version (4 bytes): 1
3. Model configuration (variable length)
4. Weight tensors with quantization

### Supported tensor types:
- Type 3: Q4_0 repacked (with --repack-format)
- Type 4: Q4_0 standard
- Type 5: Q6_K
- Type 6: Float32 (for vectors, biases, etc.)

## Performance

Typical conversion speeds (on CPU with 8 cores):
- Q4_0: ~50-100M parameters/second
- Q6_K: ~30-60M parameters/second

With GPU acceleration (CuPy):
- Q4_0: ~200-400M parameters/second
- Q6_K: ~150-300M parameters/second

## Compression ratios

- Q4_0: ~8x compression (4 bits per weight)
- Q6_K: ~5.3x compression (6 bits per weight)

## Notes

- The script automatically detects and uses GPU if CuPy is installed
- Repacking (--repack-format) is only supported for Q4_0
- For best performance, ensure the number of rows in weight matrices is divisible by 8 when using q4_0_8 repacking