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


if __name__ == "__main__":
    gen_rms_norm_golden()
