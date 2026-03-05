#!/usr/bin/env python3
"""
Generate dummy FP32 weights matching Qwen3-0.6B architecture.
Used for testing the quantization pipeline without downloading the real model.

Qwen3-0.6B Architecture:
  - vocab_size: 151669
  - hidden_size: 1024
  - num_hidden_layers: 28
  - num_attention_heads: 16
  - num_key_value_heads: 8
  - head_dim: 128
  - intermediate_size: 4096
  - tie_word_embeddings: true
"""

import numpy as np
import argparse
import os

def generate_dummy_qwen3_06b(output_path, dtype="float32"):
    """Generate dummy weights matching the Qwen3-0.6B binary layout."""

    vocab_size = 151669
    hidden_size = 1024
    num_layers = 28
    num_q_heads = 16
    num_kv_heads = 8
    head_dim = 128
    intermediate_size = 4096

    q_proj_size = num_q_heads * head_dim   # 16 * 128 = 2048
    kv_proj_size = num_kv_heads * head_dim  # 8 * 128 = 1024
    o_proj_size = num_q_heads * head_dim   # 2048

    total_params = 0

    with open(output_path, "wb") as f:
        # Embedding: [vocab_size, hidden_size]
        w = np.random.randn(vocab_size, hidden_size).astype(dtype) * 0.02
        w.tofile(f)
        total_params += w.size
        print(f"  embed_tokens: {w.shape} ({w.nbytes / 1024 / 1024:.1f} MB)")

        for layer_idx in range(num_layers):
            prefix = f"  layer{layer_idx}"

            # input_layernorm: [hidden_size]
            w = np.ones(hidden_size, dtype=dtype)
            w.tofile(f)
            total_params += w.size

            # Q projection: [hidden_size, q_proj_size] (transposed)
            w = np.random.randn(hidden_size, q_proj_size).astype(dtype) * 0.02
            w.tofile(f)
            total_params += w.size

            # Q norm: [head_dim]
            w = np.ones(head_dim, dtype=dtype)
            w.tofile(f)
            total_params += w.size

            # K projection: [hidden_size, kv_proj_size] (transposed)
            w = np.random.randn(hidden_size, kv_proj_size).astype(dtype) * 0.02
            w.tofile(f)
            total_params += w.size

            # K norm: [head_dim]
            w = np.ones(head_dim, dtype=dtype)
            w.tofile(f)
            total_params += w.size

            # V projection: [hidden_size, kv_proj_size] (transposed)
            w = np.random.randn(hidden_size, kv_proj_size).astype(dtype) * 0.02
            w.tofile(f)
            total_params += w.size

            # O projection: [o_proj_size, hidden_size] (transposed)
            w = np.random.randn(o_proj_size, hidden_size).astype(dtype) * 0.02
            w.tofile(f)
            total_params += w.size

            # post_attention_layernorm: [hidden_size]
            w = np.ones(hidden_size, dtype=dtype)
            w.tofile(f)
            total_params += w.size

            # MLP up_proj: [hidden_size, intermediate_size] (transposed)
            w = np.random.randn(hidden_size, intermediate_size).astype(dtype) * 0.02
            w.tofile(f)
            total_params += w.size

            # MLP gate_proj: [hidden_size, intermediate_size] (transposed)
            w = np.random.randn(hidden_size, intermediate_size).astype(dtype) * 0.02
            w.tofile(f)
            total_params += w.size

            # MLP down_proj: [intermediate_size, hidden_size] (transposed)
            w = np.random.randn(intermediate_size, hidden_size).astype(dtype) * 0.02
            w.tofile(f)
            total_params += w.size

            if (layer_idx + 1) % 7 == 0:
                print(f"  layer {layer_idx} done...")

        # Final norm: [hidden_size]
        w = np.ones(hidden_size, dtype=dtype)
        w.tofile(f)
        total_params += w.size

        # LM head: [hidden_size, vocab_size] (transposed)
        # Even with tie_word_embeddings=true, the bin file includes the lm_head
        w = np.random.randn(hidden_size, vocab_size).astype(dtype) * 0.02
        w.tofile(f)
        total_params += w.size
        print(f"  lm_head: [{hidden_size}, {vocab_size}] ({w.nbytes / 1024 / 1024:.1f} MB)")

    file_size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"\nTotal parameters: {total_params:,}")
    print(f"File size: {file_size_mb:.1f} MB")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dummy Qwen3-0.6B weights for testing")
    parser.add_argument("--output", type=str, default="nntr_qwen3_0.6b_fp32.bin",
                        help="Output binary file path")
    parser.add_argument("--dtype", type=str, default="float32",
                        help="Data type (float32, float16)")
    args = parser.parse_args()

    print("Generating dummy Qwen3-0.6B weights...")
    generate_dummy_qwen3_06b(args.output, args.dtype)
