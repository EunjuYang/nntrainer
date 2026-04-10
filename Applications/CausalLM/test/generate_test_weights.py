#!/usr/bin/env python3
## @file generate_test_weights.py
## @brief Generate dummy weight files for CausalLM unit tests.
##        Supports any Qwen3-compatible model config.
##        Weights are filled with small random values to avoid numerical issues.
## @usage python3 generate_test_weights.py --config tiny --output /tmp/test_weights.bin
## @usage python3 generate_test_weights.py --config qwen3-0.6b --output /tmp/qwen3_0.6b_weights.bin

import argparse
import json
import numpy as np
import os
import sys

# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "tiny": {
        "vocab_size": 32,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "head_dim": 16,
        "num_key_value_heads": 2,
        "tie_word_embeddings": False,
        "has_qk_norm": True,
    },
    "qwen3-0.6b": {
        "vocab_size": 151936,
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "head_dim": 128,
        "num_key_value_heads": 8,
        "tie_word_embeddings": True,
        "has_qk_norm": True,
    },
    "qwen3-4b": {
        "vocab_size": 151936,
        "hidden_size": 3584,
        "intermediate_size": 18944,
        "num_hidden_layers": 36,
        "num_attention_heads": 28,
        "head_dim": 128,
        "num_key_value_heads": 8,
        "tie_word_embeddings": False,
        "has_qk_norm": True,
    },
}


def generate_weight(shape, dtype="float32", scale=0.02):
    """Generate a small random weight tensor."""
    return np.random.randn(*shape).astype(dtype) * scale


def generate_qwen3_weights(config, output_path, dtype="float32", seed=42):
    """
    Generate dummy weights in nntrainer binary format for a Qwen3 model.

    Weight order matches the nntrainer model layer construction order:
      1. Embedding: [vocab_size, hidden_size]
      2. Per layer:
         a. attention_norm (rms_norm gamma): [hidden_size]
         b. Q projection: [hidden_size, num_heads * head_dim]
         c. Q norm gamma: [head_dim]  (Qwen3-specific)
         d. K projection: [hidden_size, num_kv_heads * head_dim]
         e. K norm gamma: [head_dim]  (Qwen3-specific)
         f. V projection: [hidden_size, num_kv_heads * head_dim]
         g. O projection: [num_heads * head_dim, hidden_size]
         h. ffn_norm (rms_norm gamma): [hidden_size]
         i. up_proj: [hidden_size, intermediate_size]
         j. gate_proj: [hidden_size, intermediate_size]
         k. down_proj: [intermediate_size, hidden_size]
      3. output_norm (rms_norm gamma): [hidden_size]
      4. lm_head: [hidden_size, vocab_size]  (skipped if tie_word_embeddings)
    """
    np.random.seed(seed)

    vocab_size = config["vocab_size"]
    hidden_size = config["hidden_size"]
    intermediate_size = config["intermediate_size"]
    num_layers = config["num_hidden_layers"]
    num_heads = config["num_attention_heads"]
    head_dim = config["head_dim"]
    num_kv_heads = config["num_key_value_heads"]
    tie_word_embeddings = config.get("tie_word_embeddings", False)
    has_qk_norm = config.get("has_qk_norm", True)

    total_params = 0

    with open(output_path, "wb") as f:
        # 1. Embedding
        w = generate_weight((vocab_size, hidden_size), dtype)
        w.tofile(f)
        total_params += w.size
        print(f"  embedding: {w.shape} ({w.size:,} params)")

        # 2. Transformer layers
        for layer_idx in range(num_layers):
            layer_params = 0

            # a. attention_norm
            w = generate_weight((hidden_size,), dtype)
            w.tofile(f)
            layer_params += w.size

            # b. Q projection [hidden_size, num_heads * head_dim]
            w = generate_weight((hidden_size, num_heads * head_dim), dtype)
            w.tofile(f)
            layer_params += w.size

            # c. Q norm [head_dim] (Qwen3-specific)
            if has_qk_norm:
                w = generate_weight((head_dim,), dtype)
                w.tofile(f)
                layer_params += w.size

            # d. K projection [hidden_size, num_kv_heads * head_dim]
            w = generate_weight((hidden_size, num_kv_heads * head_dim), dtype)
            w.tofile(f)
            layer_params += w.size

            # e. K norm [head_dim] (Qwen3-specific)
            if has_qk_norm:
                w = generate_weight((head_dim,), dtype)
                w.tofile(f)
                layer_params += w.size

            # f. V projection [hidden_size, num_kv_heads * head_dim]
            w = generate_weight((hidden_size, num_kv_heads * head_dim), dtype)
            w.tofile(f)
            layer_params += w.size

            # g. O projection [num_heads * head_dim, hidden_size]
            w = generate_weight((num_heads * head_dim, hidden_size), dtype)
            w.tofile(f)
            layer_params += w.size

            # h. ffn_norm
            w = generate_weight((hidden_size,), dtype)
            w.tofile(f)
            layer_params += w.size

            # i. up_proj [hidden_size, intermediate_size]
            w = generate_weight((hidden_size, intermediate_size), dtype)
            w.tofile(f)
            layer_params += w.size

            # j. gate_proj [hidden_size, intermediate_size]
            w = generate_weight((hidden_size, intermediate_size), dtype)
            w.tofile(f)
            layer_params += w.size

            # k. down_proj [intermediate_size, hidden_size]
            w = generate_weight((intermediate_size, hidden_size), dtype)
            w.tofile(f)
            layer_params += w.size

            total_params += layer_params
            if layer_idx < 3 or layer_idx == num_layers - 1:
                print(f"  layer {layer_idx}: {layer_params:,} params")
            elif layer_idx == 3:
                print(f"  ...")

        # 3. output_norm
        w = generate_weight((hidden_size,), dtype)
        w.tofile(f)
        total_params += w.size
        print(f"  output_norm: {w.shape} ({w.size:,} params)")

        # 4. lm_head (skip if tie_word_embeddings)
        if not tie_word_embeddings:
            w = generate_weight((hidden_size, vocab_size), dtype)
            w.tofile(f)
            total_params += w.size
            print(f"  lm_head: {w.shape} ({w.size:,} params)")
        else:
            print(f"  lm_head: skipped (tie_word_embeddings=true)")

    file_size = os.path.getsize(output_path)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"File size: {file_size:,} bytes ({file_size / 1024 / 1024:.1f} MB)")
    return output_path


def generate_dummy_tokenizer(output_path):
    """Generate a minimal BPE tokenizer JSON for testing."""
    tokenizer = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": i,
                "content": f"[TOK_{i}]",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
            for i in range(3)
        ],
        "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
        "post_processor": None,
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": "[TOK_0]",
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "vocab": {f"[TOK_{i}]": i for i in range(3)},
            "merges": [],
        },
    }
    with open(output_path, "w") as f:
        json.dump(tokenizer, f, indent=2)
    print(f"Dummy tokenizer saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate dummy weights for CausalLM unit tests"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="tiny",
        choices=list(MODEL_CONFIGS.keys()),
        help="Model configuration to use",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output weight file path"
    )
    parser.add_argument(
        "--tokenizer-output",
        type=str,
        default=None,
        help="Output tokenizer file path",
    )
    parser.add_argument("--dtype", type=str, default="float32", help="Data type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    config = MODEL_CONFIGS[args.config]

    if args.output is None:
        args.output = f"/tmp/nntrainer_test_{args.config}_weights.bin"

    print(f"Generating {args.config} weights...")
    print(f"Config: {json.dumps(config, indent=2)}")
    generate_qwen3_weights(config, args.output, args.dtype, args.seed)

    if args.tokenizer_output:
        generate_dummy_tokenizer(args.tokenizer_output)
