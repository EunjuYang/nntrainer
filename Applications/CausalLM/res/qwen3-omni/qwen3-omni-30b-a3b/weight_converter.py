## @file weight_converter.py
## @brief weight conversion script for Qwen3-Omni-MoE thinker model
## @note  This script downloads the Qwen3-Omni-30B-A3B-Thinking model from
##        HuggingFace and converts the thinker text weights to nntrainer format.
##
##        Key differences from standard Qwen3-MoE weight converter:
##        1. Fused gate_up_proj: Omni stores all expert gate+up projections
##           in a single tensor [num_experts, 2*moe_intermediate_size, hidden_size].
##           We split this into separate gate_proj and up_proj per expert.
##        2. Mixed dense/MoE layers: Some layers may use dense MLP instead of MoE,
##           controlled by decoder_sparse_step and mlp_only_layers config.
##        3. Weight prefix: Thinker model uses "model." prefix for text backbone.

import argparse
import json
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer


def is_moe_layer(layer_idx, config):
    """Check if a layer uses MoE based on decoder_sparse_step and mlp_only_layers."""
    mlp_only_layers = getattr(config, 'mlp_only_layers', []) or []
    decoder_sparse_step = getattr(config, 'decoder_sparse_step', 1)
    num_experts = getattr(config, 'num_experts', 0)

    if layer_idx in mlp_only_layers:
        return False
    return num_experts > 0 and (layer_idx + 1) % decoder_sparse_step == 0


def save_qwen3_omni_moe_for_nntrainer(params, config, dtype, file):
    """Convert and save Qwen3-Omni-MoE thinker weights to nntrainer format."""

    n_layers = config.num_hidden_layers
    n_experts = config.num_experts
    moe_intermediate_size = config.moe_intermediate_size

    def save_weight(weight_name, is_transpose=False):
        print(f"  {weight_name} {params[weight_name].shape}")
        if is_transpose:
            np.array(params[weight_name].permute(1, 0), dtype=dtype).tofile(file)
        else:
            np.array(params[weight_name], dtype=dtype).tofile(file)

    def save_tensor(tensor, name="", is_transpose=False):
        """Save a raw tensor (not from params dict)."""
        print(f"  [tensor] {name} {tensor.shape}")
        if is_transpose:
            np.array(tensor.permute(1, 0), dtype=dtype).tofile(file)
        else:
            np.array(tensor, dtype=dtype).tofile(file)

    def save_projection(layer_name, proj_name):
        """Helper function to handle base/lora weight saving."""
        lora_key = f"{layer_name}{proj_name}.lora_A.default.weight"
        if lora_key in params:
            save_weight(f"{layer_name}{proj_name}.base_layer.weight", True)
            save_weight(f"{layer_name}{proj_name}.lora_A.default.weight", True)
            save_weight(f"{layer_name}{proj_name}.lora_B.default.weight", True)
        else:
            save_weight(f"{layer_name}{proj_name}.weight", True)

    def save_attention(layer_name):
        """Save attention layer weights (Q/K/V/O with Q/K norm)."""
        # Save Q/K/V/O projections
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            save_projection(layer_name, f"self_attn.{proj}")
            # Qwen3 Q/K norm
            proj_norm_name = f"{layer_name}self_attn.{proj[0]}_norm.weight"
            if proj_norm_name in params:
                save_weight(proj_norm_name)

    def save_feed_forward_moe(layer_name):
        """Save MoE feed forward weights (fused gate_up_proj → split per expert)."""
        # Router gate weight
        save_weight(f"{layer_name}mlp.gate.weight", True)

        # Fused expert weights: gate_up_proj [num_experts, 2*moe_intermediate_size, hidden_size]
        gate_up_key = f"{layer_name}mlp.experts.gate_up_proj"
        down_key = f"{layer_name}mlp.experts.down_proj"

        gate_up_proj = params[gate_up_key]  # [num_experts, 2*intermediate, hidden]
        down_proj = params[down_key]  # [num_experts, hidden, intermediate]

        print(f"  {gate_up_key} {gate_up_proj.shape} (fused)")
        print(f"  {down_key} {down_proj.shape}")

        for expert_id in range(n_experts):
            # Split fused gate_up_proj into gate_proj and up_proj
            expert_gate_up = gate_up_proj[expert_id]  # [2*intermediate, hidden]
            gate_proj = expert_gate_up[:moe_intermediate_size, :]  # [intermediate, hidden]
            up_proj = expert_gate_up[moe_intermediate_size:, :]  # [intermediate, hidden]
            expert_down = down_proj[expert_id]  # [hidden, intermediate]

            # Save in nntrainer order: up, gate, down (each transposed)
            save_tensor(up_proj.permute(1, 0), f"expert_{expert_id}_up_proj")
            save_tensor(gate_proj.permute(1, 0), f"expert_{expert_id}_gate_proj")
            save_tensor(expert_down.permute(1, 0), f"expert_{expert_id}_down_proj")

    def save_feed_forward_dense(layer_name):
        """Save dense MLP feed forward weights (standard up/gate/down projections)."""
        for proj in ["up_proj", "gate_proj", "down_proj"]:
            save_projection(layer_name, f"mlp.{proj}")

    ####################################################################
    # Save embedding layer
    save_weight("model.embed_tokens.weight")

    # Process all layers
    for layer_idx in range(n_layers):
        layer_prefix = f"model.layers.{layer_idx}."
        print(f"\n=== Layer {layer_idx} ({'MoE' if is_moe_layer(layer_idx, config) else 'Dense'}) ===")

        # Attention norm
        save_weight(f"{layer_prefix}input_layernorm.weight")

        # Attention
        save_attention(layer_prefix)

        # FFN norm
        save_weight(f"{layer_prefix}post_attention_layernorm.weight")

        # FFN (MoE or Dense)
        if is_moe_layer(layer_idx, config):
            save_feed_forward_moe(layer_prefix)
        else:
            save_feed_forward_dense(layer_prefix)

    # Save final layers
    save_weight("model.norm.weight")
    save_weight("lm_head.weight", True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Qwen3-Omni-30B-A3B-Thinking thinker weights to nntrainer format"
    )
    parser.add_argument(
        "--model_path", type=str,
        default="Qwen/Qwen3-Omni-30B-A3B-Thinking",
        help="HuggingFace model ID or local path"
    )
    parser.add_argument(
        "--output_name", type=str,
        default="./nntr_qwen3_omni_30b_a3b_fp32.bin",
        help="Output binary file name"
    )
    parser.add_argument(
        "--data_type", type=str, default="float32",
        choices=["float32", "float16"],
        help="Data type for output weights"
    )
    args = parser.parse_args()

    data_dtype = args.data_type
    model_path = args.model_path
    output_name = args.output_name

    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"Loading config from {model_path}...")
    full_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # Extract thinker text config
    if hasattr(full_config, 'thinker_config'):
        thinker_config = full_config.thinker_config
        if hasattr(thinker_config, 'text_config'):
            text_config = thinker_config.text_config
        else:
            text_config = thinker_config
    elif hasattr(full_config, 'text_config'):
        text_config = full_config.text_config
    else:
        text_config = full_config

    print(f"\nText config:")
    print(f"  hidden_size: {text_config.hidden_size}")
    print(f"  num_hidden_layers: {text_config.num_hidden_layers}")
    print(f"  num_experts: {text_config.num_experts}")
    print(f"  moe_intermediate_size: {text_config.moe_intermediate_size}")
    print(f"  decoder_sparse_step: {getattr(text_config, 'decoder_sparse_step', 1)}")
    print(f"  mlp_only_layers: {getattr(text_config, 'mlp_only_layers', [])}")

    print(f"\nLoading model from {model_path}...")
    print("  (This may take a while for large models)")

    # Load only the thinker model for text-only inference
    from transformers import AutoModelForCausalLM
    try:
        # Try loading as thinker conditional generation model
        from transformers import Qwen3OmniMoeThinkerForConditionalGeneration
        model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float32, trust_remote_code=True
        )
    except (ImportError, Exception):
        # Fallback: load with AutoModel
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float32, trust_remote_code=True
        )
    model.eval()

    # Get state dict and filter to text model weights only
    state_dict = model.state_dict()

    # Detect prefix: check if keys start with "model.model." (nested) or "model."
    sample_key = next(iter(state_dict.keys()))
    if sample_key.startswith("model.model."):
        # Nested: model.model.layers -> model.layers
        print("\nDetected nested prefix 'model.model.' - remapping keys...")
        remapped = {}
        for key, value in state_dict.items():
            if key.startswith("model.model."):
                new_key = key.replace("model.model.", "model.", 1)
                remapped[new_key] = value
            elif key.startswith("model.") and not key.startswith("model.model."):
                # Skip non-text model weights (e.g., model.audio_tower, model.visual)
                continue
            elif key == "lm_head.weight":
                remapped[key] = value
            else:
                # Keep other keys as-is (lm_head, etc.)
                remapped[key] = value
        state_dict = remapped
    else:
        # Filter out non-text weights (audio_tower, visual, etc.)
        text_keys = {k: v for k, v in state_dict.items()
                     if k.startswith("model.embed_tokens") or
                        k.startswith("model.layers") or
                        k.startswith("model.norm") or
                        k.startswith("lm_head")}
        if text_keys:
            state_dict = text_keys

    print(f"\nTotal text weights: {len(state_dict)} tensors")
    print(f"Sample keys: {list(state_dict.keys())[:5]}")

    print(f"\nConverting weights to {output_name}...")
    with open(output_name, "wb") as f_model:
        save_qwen3_omni_moe_for_nntrainer(state_dict, text_config, data_dtype, f_model)

    print(f"\nDone! Output saved to {output_name}")
