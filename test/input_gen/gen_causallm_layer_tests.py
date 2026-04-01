#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Generate golden test data for CausalLM custom layers.

Golden file format (binary, sequential):
  For each tensor: [size_as_int32] [data_as_float32_array]

Order:
  1. Initial weights
  2. Inputs
  3. Outputs (after forward)
  4. Gradients (for trainable weights only)
  5. Weights (after backward - same as initial if frozen)
  6. Derivatives (outgoing derivatives / input gradients)

Note: incoming derivative is set to 2.0 in the test framework.
"""

import numpy as np
import os

# Fix seed for reproducibility (matches recorder.py SEED=1234)
SEED = 1234
np.random.seed(SEED)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def write_tensor(f, tensor):
    """Write a tensor in nnlayergolden format: [size_int32][data_float32]"""
    tensor = tensor.astype(np.float32)
    np.array(tensor.size, dtype=np.int32).tofile(f)
    tensor.tofile(f)


def rand_input(shape):
    """Generate random integer input [0, 10), matching recorder.py _rand_like"""
    return np.random.randint(0, 10, shape).astype(np.float32)


def gen_rms_norm_golden(input_shape=(2, 3, 3, 3), epsilon=1e-3,
                        filename="causallm_rmsnorm.nnlayergolden"):
    """Generate golden data for CausalLM RMS Norm layer with backward pass.

    RMS Norm forward: y = gamma * x / rms
    where rms = sqrt(mean(x^2, axis=-1, keepdim=True) + epsilon)

    Backward (incoming_deriv = 2.0):
    gamma_dy = gamma * incoming_deriv
    c = mean(gamma_dy * x, axis=-1, keepdim=True)
    dx = inv_rms * (gamma_dy - x * c * inv_rms^2)
    """
    width = input_shape[-1]

    # Initial weights: gamma = ones
    gamma = np.ones((1, 1, 1, width), dtype=np.float32)

    # Input
    x = rand_input(input_shape)

    # Forward pass
    mean_sq = np.mean(x * x, axis=-1, keepdims=True)  # mean(x^2)
    inv_rms = 1.0 / np.sqrt(mean_sq + epsilon)
    output = x * inv_rms * gamma

    # Backward pass
    # incoming derivative = 2.0
    incoming_deriv = np.full_like(output, 2.0)
    # gamma_dy = gamma * incoming_deriv
    gamma_dy = gamma * incoming_deriv
    # c = mean(gamma_dy * x, axis=-1, keepdims=True)
    c = np.mean(gamma_dy * x, axis=-1, keepdims=True)
    # dx = inv_rms * (gamma_dy - x * c * inv_rms^2)
    dx = inv_rms * (gamma_dy - x * c * inv_rms * inv_rms)

    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "wb") as f:
        # 1. Initial weights
        write_tensor(f, gamma)
        # 2. Inputs
        write_tensor(f, x)
        # 3. Outputs
        write_tensor(f, output)
        # 4. Gradients (gamma is NOT trainable -> nothing to write)
        # 5. Weights (unchanged)
        write_tensor(f, gamma)
        # 6. Derivatives
        write_tensor(f, dx)

    print(f"Generated: {filepath}")
    print(f"  input shape: {x.shape}, gamma shape: {gamma.shape}")
    print(f"  output sample: {output.flat[:5]}")
    print(f"  derivative sample: {dx.flat[:5]}")

    return filepath


def gen_lm_head_golden(input_shape=(2, 1, 1, 10), unit=5,
                       filename="causallm_lmhead.nnlayergolden"):
    """Generate golden data for CausalLM LM Head layer with backward pass.

    LM Head forward: output = input @ weight (+ bias)
    Weight shape: (1, 1, in_width, unit) for NCHW

    Backward (incoming_deriv = 2.0):
    dx = dy @ W^T
    """
    in_width = input_shape[-1]

    # Weight: shape (1, 1, in_width, unit)
    weight = rand_input((1, 1, in_width, unit))
    # Bias: disabled for this test

    # Input
    x = rand_input(input_shape)

    # Forward: output = x @ weight
    # x shape: (batch, 1, 1, in_width), weight: (1, 1, in_width, unit)
    # dot: (batch, 1, 1, in_width) x (1, 1, in_width, unit) -> (batch, 1, 1, unit)
    x_2d = x.reshape(-1, in_width)
    w_2d = weight.reshape(in_width, unit)
    out_2d = x_2d @ w_2d
    output = out_2d.reshape(input_shape[0], 1, 1, unit)

    # Backward: incoming_deriv = 2.0
    incoming_deriv = np.full_like(output, 2.0)
    # dx = dy @ W^T
    dy_2d = incoming_deriv.reshape(-1, unit)
    dx_2d = dy_2d @ w_2d.T
    dx = dx_2d.reshape(input_shape)

    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "wb") as f:
        # 1. Initial weights (weight is trainable by default in LmHead)
        write_tensor(f, weight)
        # 2. Inputs
        write_tensor(f, x)
        # 3. Outputs
        write_tensor(f, output)
        # 4. Gradients (weight gradient: dW = x^T @ dy)
        dw_2d = x_2d.T @ dy_2d
        dw = dw_2d.reshape(1, 1, in_width, unit)
        write_tensor(f, dw)
        # 5. Weights (unchanged - no optimizer step)
        write_tensor(f, weight)
        # 6. Derivatives
        write_tensor(f, dx)

    print(f"Generated: {filepath}")
    print(f"  input shape: {x.shape}, weight shape: {weight.shape}")
    print(f"  output shape: {output.shape}")
    print(f"  derivative sample: {dx.flat[:5]}")

    return filepath


def gen_swiglu_golden(input_shape=(2, 1, 1, 10),
                      filename="causallm_swiglu.nnlayergolden"):
    """Generate golden data for CausalLM SwiGLU layer with backward pass.

    SwiGLU forward: output = swish(gate) * up
    where swish(x) = x * sigmoid(x)

    Two inputs: gate (input_idx=0), up (input_idx=1)

    Backward (incoming_deriv = 2.0):
    d_up = swish(gate) * dy
    d_gate = up * swish'(gate) * dy
    where swish'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    """
    # Inputs (no weights for SwiGLU)
    gate = rand_input(input_shape)
    up = rand_input(input_shape)

    # Forward
    sigmoid_gate = 1.0 / (1.0 + np.exp(-gate))
    swish_gate = gate * sigmoid_gate
    output = swish_gate * up

    # Backward: incoming_deriv = 2.0
    incoming_deriv = np.full_like(output, 2.0)

    # d_up = swish(gate) * dy
    d_up = swish_gate * incoming_deriv

    # swish'(gate) = sigmoid(gate) + gate * sigmoid(gate) * (1 - sigmoid(gate))
    swish_prime = sigmoid_gate + gate * sigmoid_gate * (1.0 - sigmoid_gate)
    # d_gate = up * swish'(gate) * dy
    d_gate = up * swish_prime * incoming_deriv

    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "wb") as f:
        # 1. Initial weights (none)
        # 2. Inputs (gate, up)
        write_tensor(f, gate)
        write_tensor(f, up)
        # 3. Outputs
        write_tensor(f, output)
        # 4. Gradients (no trainable weights)
        # 5. Weights (none)
        # 6. Derivatives (d_gate, d_up)
        write_tensor(f, d_gate)
        write_tensor(f, d_up)

    print(f"Generated: {filepath}")
    print(f"  gate shape: {gate.shape}, up shape: {up.shape}")
    print(f"  output sample: {output.flat[:5]}")
    print(f"  d_gate sample: {d_gate.flat[:5]}")
    print(f"  d_up sample: {d_up.flat[:5]}")

    return filepath


def gen_tie_word_embedding_golden(input_shape=(2, 1, 1, 10), unit=5,
                                  filename="causallm_tiewordembedding.nnlayergolden"):
    """Generate golden data for CausalLM TieWordEmbedding layer (lm_head mode).

    TieWordEmbedding in lm_head mode: output = input @ weight^T
    Weight shape: (1, 1, unit, in_width) — stored transposed vs LmHead.

    Backward (incoming_deriv = 2.0):
    dx = dy @ weight  (since forward was input @ weight^T)
    """
    in_width = input_shape[-1]

    # Weight: shape (1, 1, unit, in_width) — note: transposed vs LmHead
    weight = rand_input((1, 1, unit, in_width))

    # Input
    x = rand_input(input_shape)

    # Forward: output = x @ weight^T
    x_2d = x.reshape(-1, in_width)
    w_2d = weight.reshape(unit, in_width)  # (unit, in_width)
    out_2d = x_2d @ w_2d.T  # (batch, unit)
    output = out_2d.reshape(input_shape[0], 1, 1, unit)

    # Backward: incoming_deriv = 2.0
    incoming_deriv = np.full_like(output, 2.0)
    # dx = dy @ weight (not transposed, since forward used weight^T)
    dy_2d = incoming_deriv.reshape(-1, unit)
    dx_2d = dy_2d @ w_2d  # (batch, in_width)
    dx = dx_2d.reshape(input_shape)

    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "wb") as f:
        # 1. Initial weights
        write_tensor(f, weight)
        # 2. Inputs
        write_tensor(f, x)
        # 3. Outputs
        write_tensor(f, output)
        # 4. Gradients (weight gradient: dW = (x^T @ dy)^T = dy^T @ x for transposed weight)
        dw_2d = dy_2d.T @ x_2d  # (unit, in_width)
        dw = dw_2d.reshape(1, 1, unit, in_width)
        write_tensor(f, dw)
        # 5. Weights (unchanged)
        write_tensor(f, weight)
        # 6. Derivatives
        write_tensor(f, dx)

    print(f"Generated: {filepath}")
    print(f"  input shape: {x.shape}, weight shape: {weight.shape}")
    print(f"  output shape: {output.shape}")
    print(f"  derivative sample: {dx.flat[:5]}")

    return filepath


if __name__ == "__main__":
    gen_rms_norm_golden()
    gen_lm_head_golden()
    gen_swiglu_golden()
    gen_tie_word_embedding_golden()
