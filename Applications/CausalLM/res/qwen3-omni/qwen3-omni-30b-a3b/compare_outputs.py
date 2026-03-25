## @file compare_outputs.py
## @brief Comparison script for validating Qwen3-Omni-30B-A3B-Thinking
##        thinker text model outputs between HuggingFace and nntrainer.
##
## Usage:
##   1. Generate reference outputs from HuggingFace:
##      python compare_outputs.py --mode generate --model_path Qwen/Qwen3-Omni-30B-A3B-Thinking
##
##   2. Compare with nntrainer outputs:
##      python compare_outputs.py --mode compare \
##          --ref_dir ./reference_outputs \
##          --nntr_logits_file ./nntr_logits.bin
##
##   3. Run full pipeline (generate + display for manual comparison):
##      python compare_outputs.py --mode generate --model_path Qwen/Qwen3-Omni-30B-A3B-Thinking

import argparse
import json
import os

import numpy as np
import torch


def generate_reference_outputs(model_path, output_dir, prompt, max_new_tokens=32):
    """Generate reference outputs from HuggingFace model."""
    from transformers import AutoConfig, AutoTokenizer

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"Loading model from {model_path}...")
    try:
        from transformers import Qwen3OmniMoeThinkerForConditionalGeneration
        model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float32, trust_remote_code=True
        )
    except (ImportError, Exception):
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float32, trust_remote_code=True
        )
    model.eval()

    print(f"\nInput prompt: {repr(prompt)}")
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    print(f"Input token ids: {input_ids.tolist()}")

    # Save input token ids
    np.array(input_ids.numpy(), dtype=np.int32).tofile(
        os.path.join(output_dir, "input_ids.bin")
    )

    # Get logits for the input (prefill)
    print("\nRunning prefill forward pass...")
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits  # [batch, seq_len, vocab_size]

    print(f"Logits shape: {logits.shape}")
    print(f"Logits dtype: {logits.dtype}")
    print(f"Logits[0, -1, :10]: {logits[0, -1, :10].tolist()}")

    # Save prefill logits
    np.array(logits.numpy(), dtype=np.float32).tofile(
        os.path.join(output_dir, "prefill_logits.bin")
    )

    # Save last token logits (most important for generation)
    last_logits = logits[0, -1, :]  # [vocab_size]
    np.array(last_logits.numpy(), dtype=np.float32).tofile(
        os.path.join(output_dir, "last_token_logits.bin")
    )

    # Get top-k predictions
    topk_values, topk_indices = torch.topk(last_logits, k=10)
    print(f"\nTop-10 predictions for next token:")
    for i, (val, idx) in enumerate(zip(topk_values, topk_indices)):
        token_str = tokenizer.decode([idx.item()])
        print(f"  {i+1}. token_id={idx.item():6d}  logit={val.item():8.4f}  '{token_str}'")

    # Save top-k info
    topk_info = {
        "topk_indices": topk_indices.tolist(),
        "topk_values": topk_values.tolist(),
        "topk_tokens": [tokenizer.decode([idx.item()]) for idx in topk_indices],
    }
    with open(os.path.join(output_dir, "topk_predictions.json"), "w") as f:
        json.dump(topk_info, f, indent=2, ensure_ascii=False)

    # Generate text
    print(f"\nGenerating {max_new_tokens} tokens...")
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy for reproducibility
            temperature=1.0,
        )

    generated_text = tokenizer.decode(generated[0], skip_special_tokens=False)
    print(f"\nGenerated text:\n{generated_text}")

    # Save generated token ids and text
    np.array(generated.numpy(), dtype=np.int32).tofile(
        os.path.join(output_dir, "generated_ids.bin")
    )
    with open(os.path.join(output_dir, "generated_text.txt"), "w") as f:
        f.write(generated_text)

    # Save metadata
    metadata = {
        "model_path": model_path,
        "prompt": prompt,
        "input_length": input_ids.shape[1],
        "logits_shape": list(logits.shape),
        "vocab_size": logits.shape[-1],
        "max_new_tokens": max_new_tokens,
        "generated_length": generated.shape[1],
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nReference outputs saved to {output_dir}/")
    return output_dir


def compare_outputs(ref_dir, nntr_logits_file, tolerance=1e-3):
    """Compare nntrainer outputs against HuggingFace reference outputs."""

    print(f"Loading reference from {ref_dir}/...")
    ref_last_logits = np.fromfile(
        os.path.join(ref_dir, "last_token_logits.bin"), dtype=np.float32
    )

    with open(os.path.join(ref_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)

    print(f"  Reference vocab_size: {metadata['vocab_size']}")
    print(f"  Reference input_length: {metadata['input_length']}")

    print(f"\nLoading nntrainer logits from {nntr_logits_file}...")
    nntr_logits = np.fromfile(nntr_logits_file, dtype=np.float32)

    # Handle shape: nntrainer may output [seq_len * vocab_size] or [vocab_size]
    vocab_size = metadata["vocab_size"]
    if nntr_logits.size == vocab_size:
        nntr_last_logits = nntr_logits
    elif nntr_logits.size % vocab_size == 0:
        seq_len = nntr_logits.size // vocab_size
        nntr_logits_reshaped = nntr_logits.reshape(seq_len, vocab_size)
        nntr_last_logits = nntr_logits_reshaped[-1, :]
    else:
        raise ValueError(
            f"nntrainer logits size {nntr_logits.size} is not compatible "
            f"with vocab_size {vocab_size}"
        )

    print(f"  nntrainer last_logits shape: {nntr_last_logits.shape}")

    # Compute comparison metrics
    abs_diff = np.abs(ref_last_logits - nntr_last_logits)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)
    median_diff = np.median(abs_diff)

    # Cosine similarity
    ref_norm = np.linalg.norm(ref_last_logits)
    nntr_norm = np.linalg.norm(nntr_last_logits)
    cosine_sim = np.dot(ref_last_logits, nntr_last_logits) / (ref_norm * nntr_norm + 1e-8)

    # Top-k agreement
    ref_topk = np.argsort(ref_last_logits)[-10:][::-1]
    nntr_topk = np.argsort(nntr_last_logits)[-10:][::-1]
    topk_overlap = len(set(ref_topk) & set(nntr_topk))

    print(f"\n{'='*60}")
    print(f" Comparison Results")
    print(f"{'='*60}")
    print(f"  Max absolute difference:    {max_diff:.6f}")
    print(f"  Mean absolute difference:   {mean_diff:.6f}")
    print(f"  Median absolute difference: {median_diff:.6f}")
    print(f"  Cosine similarity:          {cosine_sim:.8f}")
    print(f"  Top-10 overlap:             {topk_overlap}/10")
    print(f"  Argmax match:               {ref_topk[0] == nntr_topk[0]}")
    print(f"{'='*60}")

    print(f"\n  Reference  top-5: {ref_topk[:5].tolist()}")
    print(f"  nntrainer  top-5: {nntr_topk[:5].tolist()}")

    print(f"\n  Reference  logits[:5]: {ref_last_logits[:5].tolist()}")
    print(f"  nntrainer  logits[:5]: {nntr_last_logits[:5].tolist()}")

    # Pass/Fail
    passed = cosine_sim > 0.99 and topk_overlap >= 8
    status = "PASS" if passed else "FAIL"
    print(f"\n  Overall: [{status}] (cosine > 0.99 and top-10 overlap >= 8)")

    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare Qwen3-Omni-30B-A3B outputs between HuggingFace and nntrainer"
    )
    parser.add_argument(
        "--mode", type=str, required=True,
        choices=["generate", "compare"],
        help="'generate' to create reference outputs, 'compare' to compare"
    )
    parser.add_argument(
        "--model_path", type=str,
        default="Qwen/Qwen3-Omni-30B-A3B-Thinking",
        help="HuggingFace model ID or local path (for generate mode)"
    )
    parser.add_argument(
        "--ref_dir", type=str, default="./reference_outputs",
        help="Directory to save/load reference outputs"
    )
    parser.add_argument(
        "--nntr_logits_file", type=str, default="",
        help="Path to nntrainer output logits binary file (for compare mode)"
    )
    parser.add_argument(
        "--prompt", type=str,
        default="<|im_start|>user\nGive me a short introduction to large language model.<|im_end|>\n<|im_start|>assistant\n",
        help="Input prompt for generation"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=32,
        help="Maximum new tokens to generate"
    )
    args = parser.parse_args()

    if args.mode == "generate":
        generate_reference_outputs(
            args.model_path, args.ref_dir, args.prompt, args.max_new_tokens
        )
    elif args.mode == "compare":
        if not args.nntr_logits_file:
            parser.error("--nntr_logits_file is required for compare mode")
        compare_outputs(args.ref_dir, args.nntr_logits_file)
