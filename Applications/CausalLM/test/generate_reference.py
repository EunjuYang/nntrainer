#!/usr/bin/env python3
## @file generate_reference.py
## @brief Generate reference outputs for CausalLM verification tests.
##        Downloads a HuggingFace model, converts weights to nntrainer format,
##        runs inference, and saves reference token IDs and logits.
## @author Eunju Yang <ej.yang@samsung.com>

import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Add parent directory so we can import weight converter
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "res", "qwen3", "qwen3-0.6b")
)
from weight_converter import save_qwen3_for_nntrainer


def generate_reference(model_path, output_dir, num_generate=10):
    """Generate reference outputs from HuggingFace model.

    Args:
        model_path: Path to HuggingFace model (local or hub ID)
        output_dir: Directory to save reference data and converted weights
        num_generate: Number of tokens to generate for comparison
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"[1/5] Loading model from {model_path}...")
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, trust_remote_code=True
    )
    model.eval()

    # Save config files to output directory using HF API
    print("[2/5] Saving config files...")
    config.to_json_file(os.path.join(output_dir, "config.json"))
    tokenizer.save_pretrained(output_dir)

    # Save generation_config if available
    if hasattr(model, "generation_config"):
        model.generation_config.to_json_file(
            os.path.join(output_dir, "generation_config.json")
        )
    else:
        # Create minimal generation_config
        gen_cfg = {
            "bos_token_id": config.bos_token_id,
            "eos_token_id": config.eos_token_id,
            "do_sample": False,
        }
        with open(os.path.join(output_dir, "generation_config.json"), "w") as f:
            json.dump(gen_cfg, f, indent=4)

    # Convert weights to nntrainer format
    print("[3/5] Converting weights to nntrainer format...")
    weight_file = os.path.join(output_dir, "nntr_qwen3_0.6b_fp32.bin")
    with open(weight_file, "wb") as f:
        save_qwen3_for_nntrainer(
            model.state_dict(), config.num_hidden_layers, "float32", f
        )

    # Update nntr_config.json with correct paths
    nntr_config_src = os.path.join(
        os.path.dirname(__file__),
        "..",
        "res",
        "qwen3",
        "qwen3-0.6b",
        "nntr_config.json",
    )
    with open(nntr_config_src, "r") as f:
        nntr_config = json.load(f)

    nntr_config["tokenizer_file"] = os.path.join(output_dir, "tokenizer.json")
    nntr_config["num_to_generate"] = num_generate

    with open(os.path.join(output_dir, "nntr_config.json"), "w") as f:
        json.dump(nntr_config, f, indent=4)

    # Tokenize the sample input
    sample_input = nntr_config["sample_input"]
    print(f"[4/5] Tokenizing input: {repr(sample_input)}")
    input_ids = tokenizer.encode(sample_input, return_tensors="pt")

    # Save input token IDs
    input_ids_np = input_ids.numpy().flatten().astype(np.int32)
    np.save(os.path.join(output_dir, "reference_input_ids.npy"), input_ids_np)
    print(f"  Input token IDs ({len(input_ids_np)}): {input_ids_np.tolist()}")

    # Run inference and collect reference outputs
    print(f"[5/5] Running reference inference ({num_generate} tokens)...")
    generated_ids = []
    prefill_logits = None

    with torch.no_grad():
        # Prefill: forward pass on all input tokens
        outputs = model(input_ids)
        logits = outputs.logits  # [batch, seq_len, vocab_size]

        # Get logits for the last input token position (next token prediction)
        last_logits = logits[0, -1, :]  # [vocab_size]
        prefill_logits = last_logits.numpy().astype(np.float32)

        # Save top-K logit indices and values for verification
        top_k = 10
        top_values, top_indices = torch.topk(last_logits, top_k)
        print(f"  Prefill top-{top_k} tokens: {top_indices.tolist()}")
        print(f"  Prefill top-{top_k} logits: {top_values.tolist()}")

        # Greedy decode: argmax
        next_token = torch.argmax(last_logits).item()
        generated_ids.append(next_token)
        print(f"  Prefill -> token {next_token}: {repr(tokenizer.decode([next_token]))}")

        # Generate subsequent tokens one by one
        current_ids = torch.cat(
            [input_ids, torch.tensor([[next_token]])], dim=1
        )

        for step in range(1, num_generate):
            outputs = model(current_ids)
            step_logits = outputs.logits[0, -1, :]
            next_token = torch.argmax(step_logits).item()
            generated_ids.append(next_token)
            current_ids = torch.cat(
                [current_ids, torch.tensor([[next_token]])], dim=1
            )
            print(
                f"  Step {step} -> token {next_token}: {repr(tokenizer.decode([next_token]))}"
            )

    # Save reference data
    generated_ids_np = np.array(generated_ids, dtype=np.int32)
    np.save(os.path.join(output_dir, "reference_generated_ids.npy"), generated_ids_np)
    np.save(os.path.join(output_dir, "reference_prefill_logits.npy"), prefill_logits)

    # Save top-K logit info for C++ test
    top_k_data = {
        "top_indices": top_indices.tolist(),
        "top_values": top_values.tolist(),
    }
    with open(os.path.join(output_dir, "reference_topk.json"), "w") as f:
        json.dump(top_k_data, f)

    print(f"\nReference data saved to {output_dir}:")
    print(f"  - reference_input_ids.npy: {len(input_ids_np)} token IDs")
    print(f"  - reference_generated_ids.npy: {len(generated_ids_np)} generated tokens")
    print(f"  - reference_prefill_logits.npy: {len(prefill_logits)} logit values")
    print(f"  - reference_topk.json: top-{top_k} logit info")
    print(f"  - nntr_qwen3_0.6b_fp32.bin: converted weights")
    print(f"  Generated text: {repr(tokenizer.decode(generated_ids))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate reference data for CausalLM verification"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="HuggingFace model path or hub ID",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/tmp/causallm_test/qwen3-0.6b",
        help="Output directory for reference data",
    )
    parser.add_argument(
        "--num_generate",
        type=int,
        default=10,
        help="Number of tokens to generate",
    )
    args = parser.parse_args()

    generate_reference(args.model_path, args.output_dir, args.num_generate)
