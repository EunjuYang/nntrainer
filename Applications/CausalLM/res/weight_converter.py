"""
SPDX-License-Identifier: Apache-2.0
Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>

@file weights_converter.py
@date 08 May 2025
@this script is tested on transformers 4.53.2
@note Qwen3-moe-30b-a3b
@author Eunju Yang <ej.yang@samsung.com>
@author SeungBasek Hong <sb92.hong@samsung.com>
"""
import torch
import numpy as np
from dataclasses import dataclass
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
from functools import partial
import io
import struct
import os
import psutil

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("GPU acceleration available with CuPy")
except ImportError:
    GPU_AVAILABLE = False
    print("CuPy not available. Using CPU-only optimizations")

# Try to import Numba for JIT compilation
try:
    from numba import jit, prange, njit
    import numba
    NUMBA_AVAILABLE = True
    print("Numba JIT compilation available")
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba not available. Using pure NumPy")

# GGML constants
QK4_0 = 32
QK6_K = 256

# Optimized memory pool for NumPy
if not GPU_AVAILABLE:
    # Pre-allocate memory pools to reduce allocation overhead
    MEMORY_POOL_SIZE = 1024 * 1024 * 100  # 100MB pool
    try:
        np_mempool = np.empty(MEMORY_POOL_SIZE, dtype=np.uint8)
    except:
        np_mempool = None

def get_optimal_num_workers():
    """Get optimal number of workers based on system resources"""
    cpu_count = mp.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # Use fewer workers if memory is limited
    if memory_gb < 16:
        return min(cpu_count // 2, 4)
    elif memory_gb < 32:
        return min(cpu_count - 2, 8)
    else:
        return min(cpu_count - 2, 16)

# Numba-accelerated functions if available
if NUMBA_AVAILABLE:
    @njit(parallel=True, fastmath=True)
    def quantize_q4_0_block_numba(values, absmax, scale):
        """Numba-accelerated Q4_0 block quantization"""
        quantized = np.empty(32, dtype=np.int8)
        
        for i in prange(32):
            if absmax < 1e-8:
                quantized[i] = 0
            else:
                v = values[i] / scale
                # Round to nearest integer
                q = int(np.round(v))
                # Clamp to [-8, 7]
                if q > 7:
                    q = 7
                elif q < -8:
                    q = -8
                quantized[i] = q
        
        return quantized
    
    @njit(parallel=True, fastmath=True)
    def pack_nibbles_numba(quantized):
        """Numba-accelerated nibble packing"""
        packed = np.empty(16, dtype=np.uint8)
        
        for i in prange(16):
            low = quantized[i * 2] + 8
            high = quantized[i * 2 + 1] + 8
            packed[i] = (low & 0xF) | ((high & 0xF) << 4)
        
        return packed

def quantize_row_q4_0_ggml(tensor):
    """GGML Q4_0 quantization - optimized version"""
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    
    batch_size, n = tensor.shape
    result = []
    
    # Convert to numpy for faster processing
    if tensor.is_cuda and GPU_AVAILABLE:
        data = cp.asarray(tensor.cpu().numpy())
    else:
        data = tensor.numpy()
    
    for b in range(batch_size):
        row_data = data[b] if GPU_AVAILABLE else data[b]
        output = bytearray()
        
        # Process blocks in parallel using vectorized operations
        num_blocks = n // QK4_0
        
        if GPU_AVAILABLE:
            # GPU-accelerated version
            for block_idx in range(num_blocks):
                start = block_idx * QK4_0
                block = row_data[start:start + QK4_0]
                
                # Compute scale
                absmax = cp.max(cp.abs(block))
                scale = absmax / 7.0 if absmax > 0 else 0.0
                
                # Quantize
                if scale > 0:
                    quantized = cp.round(block / scale).astype(cp.int8)
                    quantized = cp.clip(quantized, -8, 7)
                else:
                    quantized = cp.zeros(QK4_0, dtype=cp.int8)
                
                # Pack nibbles
                quantized_np = quantized.get()
                packed = np.zeros(16, dtype=np.uint8)
                for i in range(16):
                    low = quantized_np[i * 2] + 8
                    high = quantized_np[i * 2 + 1] + 8
                    packed[i] = (low & 0xF) | ((high & 0xF) << 4)
                
                # Write scale and packed data
                output.extend(struct.pack('e', np.float16(scale)))
                output.extend(packed.tobytes())
        else:
            # CPU version with vectorization
            # Pre-allocate arrays for better performance
            scales = np.zeros(num_blocks, dtype=np.float16)
            all_packed = np.zeros((num_blocks, 16), dtype=np.uint8)
            
            # Vectorized computation of scales
            blocks = row_data.reshape(num_blocks, QK4_0)
            absmaxes = np.max(np.abs(blocks), axis=1)
            scales = np.where(absmaxes > 0, absmaxes / 7.0, 0.0).astype(np.float16)
            
            # Process each block
            for block_idx in range(num_blocks):
                block = blocks[block_idx]
                scale = scales[block_idx]
                
                if scale > 0:
                    if NUMBA_AVAILABLE:
                        quantized = quantize_q4_0_block_numba(block, absmaxes[block_idx], scale)
                        packed = pack_nibbles_numba(quantized)
                    else:
                        # Vectorized quantization
                        quantized = np.round(block / scale).astype(np.int8)
                        quantized = np.clip(quantized, -8, 7)
                        
                        # Vectorized packing
                        quantized_offset = quantized + 8
                        low_nibbles = quantized_offset[::2] & 0xF
                        high_nibbles = (quantized_offset[1::2] & 0xF) << 4
                        packed = (low_nibbles | high_nibbles).astype(np.uint8)
                else:
                    packed = np.zeros(16, dtype=np.uint8)
                
                all_packed[block_idx] = packed
            
            # Write all data at once
            for block_idx in range(num_blocks):
                output.extend(struct.pack('e', scales[block_idx]))
                output.extend(all_packed[block_idx].tobytes())
        
        result.append(bytes(output))
    
    return result[0] if len(result) == 1 else result

def repack_q4_0_to_q4_0_4(q4_0_data, rows, cols):
    """
    Repack Q4_0 data to Q4_0x4 format for 4-way SIMD optimization
    
    Args:
        q4_0_data: Original Q4_0 quantized data (bytes)
        rows: Number of rows (M)
        cols: Number of columns (K)
    
    Returns:
        Repacked data in Q4_0x4 format
    """
    nrows_interleaved = 4
    block_size = 2 + 16  # 2 bytes for scale + 16 bytes for packed values
    blocks_per_row = cols // QK4_0
    
    # Check alignment requirements
    if rows % nrows_interleaved != 0:
        raise ValueError(f"Number of rows ({rows}) must be divisible by {nrows_interleaved}")
    if cols % QK4_0 != 0:
        raise ValueError(f"Number of columns ({cols}) must be divisible by {QK4_0}")
    
    # Parse input data
    input_data = np.frombuffer(q4_0_data, dtype=np.uint8)
    
    # Output buffer
    output = bytearray()
    
    # Process in groups of 4 rows
    for row_group in range(0, rows, nrows_interleaved):
        # For each block position
        for block_idx in range(blocks_per_row):
            # Collect 4 blocks (one from each row in the group)
            blocks = []
            for i in range(nrows_interleaved):
                row = row_group + i
                if row < rows:
                    # Calculate offset for this block
                    offset = (row * blocks_per_row + block_idx) * block_size
                    block_data = input_data[offset:offset + block_size]
                    blocks.append(block_data)
                else:
                    # Padding with zeros if needed
                    blocks.append(np.zeros(block_size, dtype=np.uint8))
            
            # Interleave the blocks
            # First, all scales (4 x 2 bytes)
            for block in blocks:
                output.extend(block[:2])  # Scale (FP16)
            
            # Then, all packed values (4 x 16 bytes)
            for block in blocks:
                output.extend(block[2:])  # Packed quantized values
    
    return bytes(output)

def repack_q4_0_to_q4_0_8(q4_0_data, rows, cols):
    """
    Repack Q4_0 data to Q4_0x8 format for 8-way SIMD optimization
    
    Args:
        q4_0_data: Original Q4_0 quantized data (bytes)
        rows: Number of rows (M)
        cols: Number of columns (K)
    
    Returns:
        Repacked data in Q4_0x8 format
    """
    nrows_interleaved = 8
    block_size = 2 + 16  # 2 bytes for scale + 16 bytes for packed values
    blocks_per_row = cols // QK4_0
    
    # Check alignment requirements
    if rows % nrows_interleaved != 0:
        raise ValueError(f"Number of rows ({rows}) must be divisible by {nrows_interleaved}")
    if cols % QK4_0 != 0:
        raise ValueError(f"Number of columns ({cols}) must be divisible by {QK4_0}")
    
    # Parse input data
    input_data = np.frombuffer(q4_0_data, dtype=np.uint8)
    
    # Output buffer
    output = bytearray()
    
    # Process in groups of 8 rows
    for row_group in range(0, rows, nrows_interleaved):
        # For each block position
        for block_idx in range(blocks_per_row):
            # Collect 8 blocks (one from each row in the group)
            blocks = []
            for i in range(nrows_interleaved):
                row = row_group + i
                if row < rows:
                    # Calculate offset for this block
                    offset = (row * blocks_per_row + block_idx) * block_size
                    block_data = input_data[offset:offset + block_size]
                    blocks.append(block_data)
                else:
                    # Padding with zeros if needed
                    blocks.append(np.zeros(block_size, dtype=np.uint8))
            
            # Interleave the blocks
            # First, all scales (8 x 2 bytes)
            for block in blocks:
                output.extend(block[:2])  # Scale (FP16)
            
            # Then, all packed values (8 x 16 bytes)
            for block in blocks:
                output.extend(block[2:])  # Packed quantized values
    
    return bytes(output)

def quantize_and_repack_q4_0(tensor, repack_format="q4_0"):
    """
    Quantize tensor to Q4_0 and optionally repack for SIMD optimization
    
    Args:
        tensor: Input tensor to quantize
        repack_format: One of "q4_0" (no repacking), "q4_0_4", or "q4_0_8"
    
    Returns:
        Quantized (and optionally repacked) data
    """
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    
    rows, cols = tensor.shape
    
    # First, quantize to standard Q4_0
    quantized_data = quantize_row_q4_0_ggml(tensor)
    
    if isinstance(quantized_data, list):
        # Concatenate all rows
        quantized_data = b''.join(quantized_data)
    
    # Apply repacking if requested
    if repack_format == "q4_0":
        return quantized_data
    elif repack_format == "q4_0_4":
        return repack_q4_0_to_q4_0_4(quantized_data, rows, cols)
    elif repack_format == "q4_0_8":
        return repack_q4_0_to_q4_0_8(quantized_data, rows, cols)
    else:
        raise ValueError(f"Unknown repack format: {repack_format}")

def quantize_row_q6_k_ggml(tensor):
    """GGML Q6_K quantization - optimized version"""
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    
    batch_size, n = tensor.shape
    result = []
    
    # Convert to numpy for faster processing
    if tensor.is_cuda and GPU_AVAILABLE:
        data = cp.asarray(tensor.cpu().numpy())
    else:
        data = tensor.numpy()
    
    for b in range(batch_size):
        row_data = data[b] if GPU_AVAILABLE else data[b]
        output = bytearray()
        
        num_blocks = n // QK6_K
        
        for block_idx in range(num_blocks):
            start = block_idx * QK6_K
            superblock = row_data[start:start + QK6_K]
            
            if GPU_AVAILABLE:
                # GPU version
                superblock_cp = superblock if isinstance(superblock, cp.ndarray) else cp.asarray(superblock)
                
                # Find global scale
                absmax = cp.max(cp.abs(superblock_cp))
                if absmax < 1e-8:
                    # Zero block - fast path
                    output.extend(b'\x00' * 210)
                    continue
                
                inv_scale = 63.0 / absmax
                d = absmax / 63.0
                
                # Quantize to 6-bit
                quantized = cp.round(superblock_cp * inv_scale).astype(cp.int8)
                quantized = cp.clip(quantized, -32, 31)
                quantized = (quantized + 32).astype(cp.uint8)
                
                # Convert to CPU for packing
                quantized_np = quantized.get()
            else:
                # CPU version with vectorization
                absmax = np.max(np.abs(superblock))
                if absmax < 1e-8:
                    # Zero block - fast path
                    output.extend(b'\x00' * 210)
                    continue
                
                inv_scale = 63.0 / absmax
                d = absmax / 63.0
                
                # Vectorized quantization
                quantized = np.round(superblock * inv_scale).astype(np.int8)
                quantized = np.clip(quantized, -32, 31)
                quantized = (quantized + 32).astype(np.uint8)
                quantized_np = quantized
            
            # Pack 6-bit values efficiently using vectorized operations
            # Lower 4 bits (vectorized)
            ql = np.zeros(128, dtype=np.uint8)
            for i in range(128):
                idx = i * 2
                ql[i] = (quantized_np[idx] & 0xF) | ((quantized_np[idx + 1] & 0xF) << 4)
            
            # Upper 2 bits (vectorized)
            qh = np.zeros(64, dtype=np.uint8)
            for i in range(64):
                base_idx = i * 4
                qh[i] = ((quantized_np[base_idx] >> 4) & 0x3) | \
                        (((quantized_np[base_idx + 1] >> 4) & 0x3) << 2) | \
                        (((quantized_np[base_idx + 2] >> 4) & 0x3) << 4) | \
                        (((quantized_np[base_idx + 3] >> 4) & 0x3) << 6)
            
            # Compute scales for each group (vectorized)
            scales = np.zeros(16, dtype=np.int8)
            groups = superblock.reshape(16, 16)
            
            if absmax > 0:
                group_scales = np.max(np.abs(groups), axis=1) / d
                group_scales = np.round(group_scales * 127.0 / 63.0).astype(np.int8)
                group_scales = np.clip(group_scales, -128, 127)
                scales = group_scales
            
            # Write block
            output.extend(ql.tobytes())
            output.extend(qh.tobytes())
            output.extend(scales.tobytes())
            output.extend(struct.pack('e', np.float16(d)))
        
        result.append(bytes(output))
    
    return result[0] if len(result) == 1 else result

# Parallel processing functions
def process_weight_batch(weight_batch, quant_type):
    """Process a batch of weights in parallel"""
    if quant_type == "q4_0":
        return [quantize_row_q4_0_ggml(w) for w in weight_batch]
    elif quant_type == "q6_k":
        return [quantize_row_q6_k_ggml(w) for w in weight_batch]
    else:
        raise ValueError(f"Unknown quantization type: {quant_type}")

def quantize_weights_parallel(weights, quant_type, batch_size=32):
    """Quantize weights in parallel with optimal batching"""
    if not isinstance(weights, torch.Tensor):
        weights = torch.tensor(weights)
    
    if weights.dim() == 1:
        weights = weights.unsqueeze(0)
    
    num_rows = weights.shape[0]
    num_workers = get_optimal_num_workers()
    
    # Use ProcessPoolExecutor for CPU-bound tasks
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        
        # Split weights into batches
        for i in range(0, num_rows, batch_size):
            batch = weights[i:i+batch_size]
            future = executor.submit(process_weight_batch, batch, quant_type)
            futures.append(future)
        
        # Collect results
        results = []
        for future in tqdm(futures, desc=f"Quantizing {quant_type}"):
            batch_results = future.result()
            results.extend(batch_results)
    
    return results

@dataclass
class ProcessingContext:
    """Context for efficient tensor processing"""
    use_gpu: bool = False
    num_workers: int = None
    batch_size: int = 32
    
    def __post_init__(self):
        if self.num_workers is None:
            self.num_workers = get_optimal_num_workers()
        self.use_gpu = GPU_AVAILABLE and torch.cuda.is_available()

# Main conversion functions with optimizations
def convert_model_optimized(model_path, output_path, quant_type="q4_0", repack_format=None):
    """
    Optimized model conversion with parallel processing
    
    Args:
        model_path: Path to the model
        output_path: Output path for quantized model
        quant_type: Quantization type ("q4_0" or "q6_k")
        repack_format: For Q4_0, can be None, "q4_0_4", or "q4_0_8"
    """
    print(f"Loading model from {model_path}...")
    
    # Load model with optimizations
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if GPU_AVAILABLE else torch.float32,
        device_map="auto" if GPU_AVAILABLE else None,
        low_cpu_mem_usage=True
    )
    
    ctx = ProcessingContext()
    print(f"Using {ctx.num_workers} workers for parallel processing")
    
    # Process weights
    quantized_weights = {}
    
    for name, param in tqdm(model.named_parameters(), desc="Processing layers"):
        if param.requires_grad:
            param = param.detach()
        
        # Convert to appropriate format
        if param.dim() == 2 and quant_type == "q4_0" and repack_format:
            # Use repacking for Q4_0
            quantized = quantize_and_repack_q4_0(param, repack_format)
            quantized_weights[name] = quantized
        elif param.dim() == 2:
            # Matrix - process rows in parallel
            quantized = quantize_weights_parallel(param, quant_type, ctx.batch_size)
            quantized_weights[name] = quantized
        else:
            # Vector or other shape - process directly
            quantized_weights[name] = param.cpu().numpy()
    
    # Save quantized weights
    print(f"Saving quantized model to {output_path}...")
    torch.save(quantized_weights, output_path)
    print("Conversion complete!")

# Test functions for repacking
def test_q4_0_repacking():
    """Test Q4_0 repacking functionality"""
    print("Testing Q4_0 repacking...")
    
    # Create test data
    test_tensor = torch.randn(8, 256)  # 8 rows, 256 columns
    
    # Test standard Q4_0
    q4_0_standard = quantize_and_repack_q4_0(test_tensor, "q4_0")
    print(f"Standard Q4_0 size: {len(q4_0_standard)} bytes")
    
    # Test Q4_0x4
    q4_0_4 = quantize_and_repack_q4_0(test_tensor, "q4_0_4")
    print(f"Q4_0x4 size: {len(q4_0_4)} bytes")
    
    # Test Q4_0x8
    q4_0_8 = quantize_and_repack_q4_0(test_tensor, "q4_0_8")
    print(f"Q4_0x8 size: {len(q4_0_8)} bytes")
    
    # All sizes should be the same (just different layout)
    assert len(q4_0_standard) == len(q4_0_4) == len(q4_0_8)
    
    print("✓ Q4_0 repacking test passed!")

# Benchmark function
def benchmark_quantization():
    """Benchmark quantization performance"""
    sizes = [(1024, 1024), (4096, 4096), (8192, 8192)]
    
    for rows, cols in sizes:
        print(f"\nBenchmarking {rows}x{cols} matrix:")
        data = torch.randn(rows, cols, dtype=torch.float32)
        
        # Q4_0 benchmark
        start = time.time()
        _ = quantize_weights_parallel(data, "q4_0")
        q4_time = time.time() - start
        print(f"  Q4_0: {q4_time:.3f}s ({rows*cols/q4_time/1e6:.1f} M elements/s)")
        
        # Q6_K benchmark
        start = time.time()
        _ = quantize_weights_parallel(data, "q6_k")
        q6_time = time.time() - start
        print(f"  Q6_K: {q6_time:.3f}s ({rows*cols/q6_time/1e6:.1f} M elements/s)")

def save_model_for_nntrainer(model_path, output_path, quant_type="q4_0", repack_format=None):
    """
    Convert and save a model in nntrainer format
    
    Args:
        model_path: Path to the input model
        output_path: Path for the output file
        quant_type: Quantization type ("q4_0" or "q6_k")
        repack_format: For Q4_0, can be None, "q4_0_4", or "q4_0_8"
    """
    print(f"Loading model from {model_path}...")
    
    # Load model
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if GPU_AVAILABLE else torch.float32,
        device_map="auto" if GPU_AVAILABLE else None,
        low_cpu_mem_usage=True
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Get model configuration
    model_config = {
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": getattr(config, "num_key_value_heads", config.num_attention_heads),
        "max_position_embeddings": config.max_position_embeddings,
        "rms_norm_eps": getattr(config, "rms_norm_eps", 1e-6),
        "rope_theta": getattr(config, "rope_theta", 10000.0),
        "quantization_type": quant_type,
        "repack_format": repack_format if quant_type == "q4_0" else None
    }
    
    print(f"Model configuration:")
    for key, value in model_config.items():
        print(f"  {key}: {value}")
    
    print(f"\nConverting weights to {quant_type} format...")
    if repack_format:
        print(f"  with repacking to {repack_format}")
    
    # Open output file
    with open(output_path, "wb") as f:
        # Write magic number
        f.write(struct.pack("I", 0x4E4E5452))  # "NNTR"
        
        # Write version
        f.write(struct.pack("I", 1))
        
        # Write model configuration
        f.write(struct.pack("I", len(model_config)))
        for key, value in model_config.items():
            # Write key
            key_bytes = key.encode('utf-8')
            f.write(struct.pack("I", len(key_bytes)))
            f.write(key_bytes)
            
            # Write value
            if isinstance(value, int):
                f.write(struct.pack("B", 0))  # Type: int
                f.write(struct.pack("I", value))
            elif isinstance(value, float):
                f.write(struct.pack("B", 1))  # Type: float
                f.write(struct.pack("f", value))
            elif isinstance(value, str) or value is None:
                f.write(struct.pack("B", 2))  # Type: string
                value_str = str(value) if value is not None else ""
                value_bytes = value_str.encode('utf-8')
                f.write(struct.pack("I", len(value_bytes)))
                f.write(value_bytes)
        
        # Write weights
        ctx = ProcessingContext()
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nTotal parameters: {total_params/1e6:.2f}M")
        
        # Count number of weight tensors
        weight_count = len(list(model.named_parameters()))
        f.write(struct.pack("I", weight_count))
        
        # Process and write each weight
        with tqdm(total=weight_count, desc="Converting weights") as pbar:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param = param.detach()
                
                # Write tensor name
                name_bytes = name.encode('utf-8')
                f.write(struct.pack("I", len(name_bytes)))
                f.write(name_bytes)
                
                # Write tensor shape
                shape = list(param.shape)
                f.write(struct.pack("I", len(shape)))
                for dim in shape:
                    f.write(struct.pack("I", dim))
                
                # Convert and write tensor data
                if param.dim() == 2 and quant_type in ["q4_0", "q6_k"]:
                    # Quantize matrix weights
                    if quant_type == "q4_0" and repack_format:
                        # Use repacking for Q4_0
                        quantized = quantize_and_repack_q4_0(param, repack_format)
                        f.write(struct.pack("B", 3))  # Type: Q4_0 repacked
                        f.write(struct.pack("I", len(quantized)))
                        f.write(quantized)
                    else:
                        # Standard quantization
                        quantized = quantize_weights_parallel(param, quant_type, ctx.batch_size)
                        if isinstance(quantized, list):
                            quantized_data = b''.join(quantized)
                        else:
                            quantized_data = quantized
                        
                        type_id = 4 if quant_type == "q4_0" else 5  # Q4_0 or Q6_K
                        f.write(struct.pack("B", type_id))
                        f.write(struct.pack("I", len(quantized_data)))
                        f.write(quantized_data)
                else:
                    # Store as float32 (vectors, biases, etc.)
                    data = param.cpu().numpy().astype(np.float32)
                    f.write(struct.pack("B", 6))  # Type: float32
                    f.write(struct.pack("I", data.nbytes))
                    f.write(data.tobytes())
                
                pbar.update(1)
    
    file_size = os.path.getsize(output_path)
    print(f"\nConversion complete!")
    print(f"Output file: {output_path}")
    print(f"File size: {file_size/1024/1024:.2f} MB")
    print(f"Compression ratio: {total_params * 4 / file_size:.2f}x")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert model weights to nntrainer format")
    parser.add_argument("model_path", type=str, help="Path to the input model")
    parser.add_argument("output_path", type=str, help="Path for the output file")
    parser.add_argument("--quant-type", type=str, default="q4_0", 
                       choices=["q4_0", "q6_k"],
                       help="Quantization type (default: q4_0)")
    parser.add_argument("--repack-format", type=str, default=None,
                       choices=[None, "q4_0_4", "q4_0_8"],
                       help="Repacking format for Q4_0 (default: None)")
    parser.add_argument("--test", action="store_true",
                       help="Run tests instead of conversion")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run benchmark instead of conversion")
    
    args = parser.parse_args()
    
    if args.test:
        # Run tests
        print("Running quantization tests...")
        test_q4_0_repacking()
        print("\nAll tests passed!")
    elif args.benchmark:
        # Run benchmark
        print("Running quantization benchmark...")
        benchmark_quantization()
    else:
        # Perform conversion
        if args.repack_format and args.quant_type != "q4_0":
            print("Warning: Repacking is only supported for Q4_0 quantization")
            args.repack_format = None
        
        save_model_for_nntrainer(
            args.model_path,
            args.output_path,
            args.quant_type,
            args.repack_format
        )
