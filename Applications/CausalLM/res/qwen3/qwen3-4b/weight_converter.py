## @file weight_converter.py
## @brief weight conversion script for qwen3 model
## @author Eunju Yang <ej.yang@samsung.com>

import argparse
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

total_size = 0
def save_qwen3_for_nntrainer(params, n_layers, dtype, file):  
    """Convert and save weights as nntrainer format for multi-head attention model"""  
      
    def save_weight(weight):
        np.array(weight, dtype=dtype).tofile(file)  

    def save_projection(layer_name, proj_name):  
        """Helper function to handle base/lora weight saving"""  
        lora_key = f"{layer_name}{proj_name}.lora_A.default.weight"  
        if lora_key in params:  
            save_weight(params[f"{layer_name}{proj_name}.base_layer.weight"].permute(1, 0))  
            save_weight(params[f"{layer_name}{proj_name}.lora_A.default.weight"].permute(1, 0))  
            save_weight(params[f"{layer_name}{proj_name}.lora_B.default.weight"].permute(1, 0))  
        else:  
            save_weight(params[f"{layer_name}{proj_name}.weight"].permute(1, 0))  

    def save_attention(layer_name):
        """Save attention layer weights"""
        save_weight(params[f"{layer_name}input_layernorm.weight"])

        # Save QKV projections grouped (matches fused QKVLayer weight order)
        for proj in ["q_proj", "k_proj", "v_proj"]:
            save_projection(layer_name, f"self_attn.{proj}")

        # Save Q/K norms after projections (matches topological sort order)
        for norm_prefix in ["q", "k"]:
            norm_name = f"{layer_name}self_attn.{norm_prefix}_norm.weight"
            if norm_name in params:
                print(norm_name)
                save_weight(params[norm_name])

        # Save O projection
        save_projection(layer_name, "self_attn.o_proj")

    def save_feed_forward(layer_name):  
        """Save feed forward layer weights"""  
        save_weight(params[f"{layer_name}post_attention_layernorm.weight"])  
          
        # Save MLP projections using helper  
        for proj in ["up_proj", "gate_proj", "down_proj"]:  
            save_projection(layer_name, f"mlp.{proj}")  

    # Save embedding layer  
    save_weight(params["model.embed_tokens.weight"])  

    # Process all layers  
    for layer_idx in range(n_layers):  
        layer_prefix = f"model.layers.{layer_idx}."  
        save_attention(layer_prefix)  
        save_feed_forward(layer_prefix)  

    # Save final layers  
    save_weight(params["model.norm.weight"])  
    save_weight(params["lm_head.weight"].permute(1, 0))  


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./Qwen3-4b")
    parser.add_argument("--output_name", type=str, default="./nntr_qwen3_4b_fp32.bin")
    parser.add_argument("--data_type", type=str, default="float32")
    args = parser.parse_args()
    
    data_dtype = args.data_type
    model_path = args.model_path
    output_name = args.output_name
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="float", trust_remote_code=True)
    model.eval()

    with open(output_name, "wb") as f_model :
        save_qwen3_for_nntrainer(model.state_dict(), config.num_hidden_layers, data_dtype, f_model)
