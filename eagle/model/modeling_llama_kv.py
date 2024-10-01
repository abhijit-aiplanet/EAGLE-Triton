# Source: https://github.com/huggingface/transformers/blob/v4.31-release/src/transformers/models/llama/modeling_llama.py
# Modifications are denoted by the symbol: [MODIFIED]


""" PyTorch LLaMA model."""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# [MODIFIED] Import from transformer library
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers import LlamaConfig

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


import triton
import triton.language as tl
import torch

@triton.jit
def causal_mask_kernel(
    mask, 
    tgt_len,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_idx = tl.program_id(1) * BLOCK_SIZE + col_offsets

    # Compute mask values based on causal condition
    mask_vals = tl.where(col_idx >= row_idx, 0.0, -3.4e38)

    # Compute the linear offsets into the mask tensor
    offsets = row_idx * tgt_len + col_idx

    # Store the mask values at the computed offsets
    tl.store(mask + offsets, mask_vals, mask=col_idx < tgt_len)


def _make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
):
    bsz, tgt_len = input_ids_shape
    mask = torch.empty((tgt_len, tgt_len), dtype=torch.float32, device=device)
    
    BLOCK_SIZE = 1024
    grid = (tgt_len, (tgt_len + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    causal_mask_kernel[grid](
        mask, 
        tgt_len,
        BLOCK_SIZE
    )
    
    if past_key_values_length > 0:
        past_mask = torch.zeros(
            (tgt_len, past_key_values_length), dtype=torch.float32, device=device
        )
        mask = torch.cat([past_mask, mask], dim=-1)
    
    # Convert to the desired dtype after all computations
    mask = mask.to(dtype)
    
    return mask[None, None, :, :].expand(
        bsz, 1, tgt_len, tgt_len + past_key_values_length
    )

@triton.jit
def expand_mask_kernel(
        expanded_mask_ptr, 
        mask_ptr, 
        src_len, 
        tgt_len, 
        bsz, 
        BLOCK_SIZE: tl.constexpr
):
    b_idx = tl.program_id(0)  # Get batch index
    t_idx = tl.program_id(1)  # Get target sequence index
    s_idx = tl.arange(0, BLOCK_SIZE)  # Get a block of source sequence indices
    
    # Compute mask expansion for each batch and target index
    mask_value = tl.load(mask_ptr + b_idx * src_len + s_idx, mask=s_idx < src_len)
    
    # Instead of using expand, we'll broadcast the mask value to all target indices
    expanded_value = 1.0 - mask_value
    
    # Store the result in the expanded_mask tensor
    tl.store(expanded_mask_ptr + b_idx * tgt_len * src_len + t_idx * src_len + s_idx, 
             expanded_value, mask=s_idx < src_len)

def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expand attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]` using Triton.
    Args:
        mask (torch.Tensor): The attention mask tensor of shape `[bsz, seq_len]`.
        dtype (torch.dtype): The data type of the mask.
        tgt_len (Optional[int], optional): The target sequence length. If None, it defaults to the source sequence length.
    Returns:
        torch.Tensor: The expanded mask tensor.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    
    # Initialize the expanded mask
    expanded_mask = torch.empty((bsz, tgt_len, src_len), dtype=dtype, device=mask.device)
    
    BLOCK_SIZE = 1024  # Define the block size for the Triton kernel
    grid = lambda meta: (bsz, tgt_len, triton.cdiv(src_len, BLOCK_SIZE))
    
    # Launch Triton kernel for expanding the mask
    expand_mask_kernel[grid](
        expanded_mask, mask, src_len, tgt_len, bsz, BLOCK_SIZE
    )
    
    # Apply the mask fill (can still use standard PyTorch here)
    expanded_mask = expanded_mask.to(dtype)
    return expanded_mask.masked_fill(expanded_mask.to(torch.bool), torch.finfo(dtype).min)

import torch.nn as nn
import torch



# Triton kernel for RMS normalization
@triton.jit
def rms_norm_kernel(
        hidden_states_ptr, 
        normed_states_ptr, 
        weight_ptr, 
        eps_ptr, 
        hidden_size, 
        BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)  # Get row index for each batch
    col_idx = tl.arange(0, BLOCK_SIZE)  # Get block of indices for hidden size

    # Load hidden states
    hidden_vals = tl.load(hidden_states_ptr + row_idx * hidden_size + col_idx, mask=col_idx < hidden_size, other=0.0)

    # Compute variance
    variance = tl.sum(hidden_vals * hidden_vals, axis=0) / hidden_size
    inv_stddev = 1.0 / tl.sqrt(variance + tl.load(eps_ptr))

    # Normalize the hidden states
    normed_vals = hidden_vals * inv_stddev

    # Scale with the weight parameter
    weight_vals = tl.load(weight_ptr + col_idx, mask=col_idx < hidden_size)
    normed_vals = normed_vals * weight_vals

    # Store the normalized and scaled values
    tl.store(normed_states_ptr + row_idx * hidden_size + col_idx, normed_vals, mask=col_idx < hidden_size)


class LlamaRMSNorm(nn.Module):
    """
    LlamaRMSNorm is equivalent to T5LayerNorm, using Triton to accelerate computation.

    Args:
        hidden_size (int): The size of the hidden states.
        eps (float, optional): A small value to prevent division by zero. Default is 1e-6.
    """

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = torch.tensor([eps], dtype=torch.float32).cuda()
        self.hidden_size = hidden_size

    def forward(self, hidden_states):
        """
        Apply LlamaRMSNorm to the input hidden states using Triton for GPU acceleration.

        Args:
            hidden_states (torch.Tensor): Input hidden states.

        Returns:
            torch.Tensor: The normalized and scaled hidden states.
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)  # Convert to float32 for stability

        # Initialize the output tensor
        normed_states = torch.empty_like(hidden_states)

        # Launch Triton kernel
        BLOCK_SIZE = 1024  # You can tune this based on your hardware
        grid = lambda meta: (hidden_states.size(0),)  # One block per row (batch element)

        rms_norm_kernel[grid](
            hidden_states,
            normed_states,
            self.weight,
            self.variance_epsilon,
            self.hidden_size,
            BLOCK_SIZE
        )

        return normed_states.to(input_dtype)  # Convert back to original dtype if needed

import torch
import triton
import triton.language as tl

@triton.jit
def rotary_embedding_kernel(
    emb_ptr, 
    cos_ptr, 
    sin_ptr, 
    seq_len, 
    dim, 
    BLOCK_SIZE: tl.constexpr
):
    # Sequence index for the batch
    seq_idx = tl.program_id(0)
    
    # Dimension index block for parallelism
    dim_idx = tl.arange(0, BLOCK_SIZE)
    
    # Offset for the current sequence
    offset = seq_idx * dim + dim_idx
    
    # Load emb values
    emb_val = tl.load(emb_ptr + offset, mask=dim_idx < dim)
    
    # Cast to float32 for computation
    emb_val_f32 = emb_val.to(tl.float32)
    
    # Calculate cosine and sine
    cos_val = tl.cos(emb_val_f32)
    sin_val = tl.sin(emb_val_f32)
    
    # Store the calculated cos and sin values (casting back to original dtype)
    tl.store(cos_ptr + offset, cos_val.to(emb_val.dtype), mask=dim_idx < dim)
    tl.store(sin_ptr + offset, sin_val.to(emb_val.dtype), mask=dim_idx < dim)

class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq)
        
        # Initialize with float32, we'll convert to the appropriate dtype in _set_cos_sin_cache
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=device,
            dtype=torch.float32,
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        if device.type == "cpu":
            raise ValueError("Triton kernels require GPU tensors. Please use a CUDA device.")
        
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).contiguous()
    
        # Convert to the desired dtype
        emb = emb.to(dtype)
        
        # Allocate memory for cosine and sine buffers
        cos_cache = torch.empty_like(emb, dtype=dtype, device=device)
        sin_cache = torch.empty_like(emb, dtype=dtype, device=device)
    
        BLOCK_SIZE = 1024  # Adjust this based on your hardware for optimal performance
        
        # Grid size adjusted for parallelism across sequence length
        grid = (seq_len,)
    
        # Launch the Triton kernel to calculate cos and sin embeddings
        rotary_embedding_kernel[grid](
            emb, 
            cos_cache, 
            sin_cache, 
            seq_len, 
            emb.shape[1],  # This is the dimension size
            BLOCK_SIZE
        )
        
        # Register the buffers for future access
        self.register_buffer("cos_cached", cos_cache.unsqueeze(0).unsqueeze(0), persistent=False)
        self.register_buffer("sin_cached", sin_cache.unsqueeze(0).unsqueeze(0), persistent=False)

    def forward(self, x, seq_len=None):
        if x.device.type == "cpu":
            raise ValueError("Input tensor must be on a CUDA device for Triton kernel execution.")
        
        # Ensure that cos_cached and sin_cached are in the same dtype and device as x
        if self.cos_cached.dtype != x.dtype or self.cos_cached.device != x.device:
            self._set_cos_sin_cache(seq_len=self.max_seq_len_cached, device=x.device, dtype=x.dtype)
        
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        
        return (
            self.cos_cached[:, :, :seq_len, ...],
            self.sin_cached[:, :, :seq_len, ...],
        )

# Helper function to check if CUDA is available
def cuda_is_available():
    return torch.cuda.is_available()

# Helper function to get the current CUDA device
def get_cuda_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Triton kernel to calculate cosine and sine embeddings with scaling
@triton.jit
def linear_scaling_rotary_embedding_kernel(
        scaled_freqs_ptr,
        cos_ptr,
        sin_ptr,
        seq_len,
        dim,
        scaling_factor,
        BLOCK_SIZE: tl.constexpr):
    
    seq_idx = tl.program_id(0)  # Sequence index for each batch
    dim_idx = tl.arange(0, BLOCK_SIZE)  # Dimension index block for parallelism

    # Calculate scaled frequency for each sequence index and dimension
    freq_val = tl.load(scaled_freqs_ptr + seq_idx * dim + dim_idx, mask=dim_idx < dim)
    freq_val = freq_val / scaling_factor  # Apply linear scaling

    # Calculate cosine and sine of the scaled frequency
    cos_val = tl.cos(freq_val)
    sin_val = tl.sin(freq_val)

    # Store the calculated cos and sin values
    tl.store(cos_ptr + seq_idx * dim + dim_idx, cos_val, mask=dim_idx < dim)
    tl.store(sin_ptr + seq_idx * dim + dim_idx, sin_val, mask=dim_idx < dim)


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """
    LlamaRotaryEmbedding extended with linear scaling using Triton for faster embedding computation.

    Args:
        dim (int): The dimension of the embedding.
        max_position_embeddings (int, optional): The maximum number of position embeddings. Default is 2048.
        base (int, optional): The base value for the rotational embeddings. Default is 10000.
        device (str or torch.device, optional): The device where the embeddings should be stored. Default is None.
        scaling_factor (float, optional): The scaling factor for the embeddings. Default is 1.0.
    """

    def __init__(
            self,
            dim,
            max_position_embeddings=2048,
            base=10000,
            device=None,
            scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        """
        Set the cosine and sine cache for the rotary embeddings using Triton.

        Args:
            seq_len (int): The sequence length.
            device (str or torch.device): The device where the cache should be stored.
            dtype: The data type for the cache.
        """
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor  # Apply the scaling factor

        # Compute frequency embeddings
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        # Allocate memory for cosine and sine buffers
        cos_cache = torch.empty_like(emb, dtype=dtype, device=device)
        sin_cache = torch.empty_like(emb, dtype=dtype, device=device)

        BLOCK_SIZE = 1024  # You can adjust this based on your hardware for optimal performance
        grid = lambda meta: (seq_len,)  # Grid parallelizes across the sequence length

        # Launch the Triton kernel to calculate cos and sin embeddings with scaling
        linear_scaling_rotary_embedding_kernel[grid](
            emb, 
            cos_cache, 
            sin_cache, 
            seq_len, 
            self.dim, 
            self.scaling_factor,
            BLOCK_SIZE
        )

        # Register the buffers for future access
        self.register_buffer("cos_cached", cos_cache[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", sin_cache[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        """
        Forward pass of the LlamaLinearScalingRotaryEmbedding module.

        Args:
            x (torch.Tensor): Input tensor of shape [bs, num_attention_heads, seq_len, head_size].
            seq_len (int): The sequence length. If greater than the cached length, the cache will be updated.

        Returns:
            tuple: A tuple containing two tensors, the cosine and sine embeddings, both of shape [1, 1, seq_len, dim].
        """
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


# Triton kernel for applying cosine and sine embeddings to query and key tensors
@triton.jit
def rotary_pos_emb_kernel(q_ptr, k_ptr, cos_ptr, sin_ptr, position_ids_ptr, embed_dim, BLOCK_SIZE: tl.constexpr):
    seq_idx = tl.program_id(0)  # Sequence index for each batch
    dim_idx = tl.arange(0, BLOCK_SIZE)  # Dimension index block for parallelism

    # Load query and key values
    q_val = tl.load(q_ptr + seq_idx * embed_dim + dim_idx, mask=dim_idx < embed_dim)
    k_val = tl.load(k_ptr + seq_idx * embed_dim + dim_idx, mask=dim_idx < embed_dim)

    # Load cosine and sine values for the current position
    cos_val = tl.load(cos_ptr + tl.load(position_ids_ptr + seq_idx) * embed_dim + dim_idx, mask=dim_idx < embed_dim)
    sin_val = tl.load(sin_ptr + tl.load(position_ids_ptr + seq_idx) * embed_dim + dim_idx, mask=dim_idx < embed_dim)

    # Rotate half the dimensions of the query and key tensors
    half = embed_dim // 2
    q1 = q_val[..., :half]
    q2 = q_val[..., half:]
    k1 = k_val[..., :half]
    k2 = k_val[..., half:]

    # Apply rotary embeddings
    q_embed = (q1 * cos_val[:half]) + (-q2 * sin_val[:half])
    k_embed = (k1 * cos_val[:half]) + (-k2 * sin_val[:half])

    # Store the updated query and key embeddings
    tl.store(q_ptr + seq_idx * embed_dim + dim_idx, q_embed, mask=dim_idx < embed_dim)
    tl.store(k_ptr + seq_idx * embed_dim + dim_idx, k_embed, mask=dim_idx < embed_dim)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """
    LlamaRotaryEmbedding extended with Dynamic NTK scaling using Triton for efficient rotary embedding computation.
    """

    def __init__(
            self,
            dim,
            max_position_embeddings=2048,
            base=10000,
            device=None,
            scaling_factor=1.0,
    ):
        """
        Initialize the LlamaDynamicNTKScalingRotaryEmbedding.

        Args:
            dim (int): The dimensionality of the embedding.
            max_position_embeddings (int, optional): Maximum number of position embeddings. Default is 2048.
            base (int, optional): Base value for scaling calculations. Default is 10000.
            device: The device to place tensors on. If None, uses the default device.
            scaling_factor (float, optional): Scaling factor for NTK scaling. Default is 1.0.
        """
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        """
        Set the cached values for cosine and sine.

        Args:
            seq_len (int): The sequence length.
            device: The device to place tensors on.
            dtype: The data type of tensors.
        """
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                    (self.scaling_factor * seq_len / self.max_position_embeddings)
                    - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                    base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
            )
            self.register_buffer("inv_freq", inv_freq)

        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False
        )


def rotate_half_triton(x_ptr, embed_dim, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel for rotating half the hidden dimensions.

    Args:
        x_ptr (torch.Tensor): Input tensor.
        embed_dim (int): Embedding dimension.
        BLOCK_SIZE (int): Block size for Triton kernel.
    """
    seq_idx = tl.program_id(0)  # Sequence index
    dim_idx = tl.arange(0, BLOCK_SIZE)  # Dimension index block

    half = embed_dim // 2
    x1 = tl.load(x_ptr + seq_idx * embed_dim + dim_idx, mask=dim_idx < half)
    x2 = tl.load(x_ptr + seq_idx * embed_dim + dim_idx + half, mask=dim_idx < half)

    # Rotate half the dimensions
    tl.store(x_ptr + seq_idx * embed_dim + dim_idx, -x2, mask=dim_idx < half)
    tl.store(x_ptr + seq_idx * embed_dim + dim_idx + half, x1, mask=dim_idx < half)


def apply_rotary_pos_emb_triton(q, k, cos, sin, position_ids):
    """
    Apply rotary position embeddings to query and key tensors using Triton for parallel computation.

    Args:
        q (torch.Tensor): Query tensor.
        k (torch.Tensor): Key tensor.
        cos (torch.Tensor): Cosine values.
        sin (torch.Tensor): Sine values.
        position_ids (torch.Tensor): Position IDs.

    Returns:
        torch.Tensor: Query and key tensors with rotary position embeddings applied.
    """
    embed_dim = q.shape[-1]

    # Triton block size
    BLOCK_SIZE = 1024
    grid = lambda meta: (q.size(0),)  # Parallelize over batch size

    # Apply rotary embeddings using Triton
    rotary_pos_emb_kernel[grid](
        q,
        k,
        cos,
        sin,
        position_ids,
        embed_dim,
        BLOCK_SIZE
    )

    return q, k


# Triton kernel for MLP projection and activation
@triton.jit
def mlp_kernel(x_ptr, gate_proj_ptr, up_proj_ptr, down_proj_ptr, act_fn, intermediate_size, hidden_size, BLOCK_SIZE: tl.constexpr):
    seq_idx = tl.program_id(0)  # Sequence index for the batch
    dim_idx = tl.arange(0, BLOCK_SIZE)  # Block of dimensions for parallelism

    # Load input tensor
    x_val = tl.load(x_ptr + seq_idx * hidden_size + dim_idx, mask=dim_idx < hidden_size)

    # Apply gate_proj and up_proj (linear layers)
    gate_proj_val = tl.dot(x_val, gate_proj_ptr + dim_idx)
    up_proj_val = tl.dot(x_val, up_proj_ptr + dim_idx)

    # Apply activation function
    if act_fn == 'relu':
        gate_proj_val = tl.max(0, gate_proj_val)
    elif act_fn == 'gelu':
        gate_proj_val = tl.gelu(gate_proj_val)

    # Multiply activated gate_proj with up_proj
    intermediate_val = gate_proj_val * up_proj_val

    # Apply down_proj (linear layer)
    down_proj_val = tl.dot(intermediate_val, down_proj_ptr + dim_idx)

    # Store result back into the output tensor
    tl.store(x_ptr + seq_idx * hidden_size + dim_idx, down_proj_val, mask=dim_idx < hidden_size)


class LlamaMLP(nn.Module):
    """
    LlamaMLP is a multi-layer perceptron module used in the Llama model with Triton acceleration.
    """

    def __init__(self, config):
        super().__init__()
        self.pretraining_tp = config.pretraining_tp
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = config.hidden_act

    def forward(self, x):
        """
        Forward pass of the MLP with Triton acceleration.
        """
        BLOCK_SIZE = 1024  # You can tune this for your hardware
        grid = lambda meta: (x.size(0),)  # Parallelize over batch size

        if self.pretraining_tp > 1:
            slice_size = self.intermediate_size // self.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice_size, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice_size, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice_size, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.pretraining_tp)],
                dim=-1,
            )
            up_proj = torch.cat(
                [F.linear(x, up_proj_slices[i]) for i in range(self.pretraining_tp)],
                dim=-1,
            )

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice_size, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i])
                for i in range(self.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            # Use Triton for the main MLP operations
            mlp_kernel[grid](
                x,
                self.gate_proj.weight,
                self.up_proj.weight,
                self.down_proj.weight,
                self.act_fn,
                self.intermediate_size,
                self.hidden_size,
                BLOCK_SIZE
            )

        return down_proj


# Triton kernel for repeating key and value tensors
@triton.jit
def repeat_kv_kernel(x_ptr, y_ptr, num_kv_heads, slen, head_dim, n_rep, BLOCK_SIZE: tl.constexpr):
    batch_idx = tl.program_id(0)  # Batch index
    head_idx = tl.program_id(1)  # Head index for repetition
    seq_idx = tl.arange(0, BLOCK_SIZE)  # Sequence index block

    # Load original hidden states
    x_val = tl.load(x_ptr + batch_idx * num_kv_heads * slen * head_dim + seq_idx, mask=seq_idx < slen * head_dim)

    # Repeat key and value tensors along head dimension
    for rep in range(n_rep):
        y_offset = batch_idx * num_kv_heads * n_rep * slen * head_dim + head_idx * slen * head_dim + rep * slen * head_dim
        tl.store(y_ptr + y_offset + seq_idx, x_val, mask=seq_idx < slen * head_dim)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key and value tensors n times along the specified dimension using Triton.

    Args:
        hidden_states (torch.Tensor): Input tensor with shape (batch, num_key_value_heads, seqlen, head_dim).
        n_rep (int): Number of times to repeat.

    Returns:
        torch.Tensor: Repeated tensor with shape (batch, num_key_value_heads * n_rep, seqlen, head_dim).
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states

    # Allocate output tensor
    output = torch.empty((batch, num_key_value_heads * n_rep, slen, head_dim), device=hidden_states.device, dtype=hidden_states.dtype)

    # Triton block size and grid setup
    BLOCK_SIZE = 1024
    grid = lambda meta: (batch, num_key_value_heads)

    # Launch Triton kernel to perform the repetition
    repeat_kv_kernel[grid](
        hidden_states,
        output,
        num_key_value_heads,
        slen,
        head_dim,
        n_rep,
        BLOCK_SIZE
    )

    return output



@triton.jit
def linear_proj_kernel(x_ptr, weight_ptr, out_ptr, in_features, out_features: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    seq_idx = tl.program_id(0)  # Sequence index
    dim_idx = tl.arange(0, BLOCK_SIZE)  # Block of dimensions for parallelism
    
    # Load input tensor (1D slice corresponding to a single sequence, in_features long)
    x_val = tl.load(x_ptr + seq_idx * in_features + dim_idx, mask=dim_idx < in_features)
    
    # Load weight tensor as a matrix (ensure correct broadcasting across out_features)
    weight = tl.load(weight_ptr + dim_idx * out_features + tl.arange(0, out_features), mask=(dim_idx < in_features))
    
    # Perform matrix-vector multiplication
    result = tl.dot(x_val, weight)
    
    # Store result in the output tensor
    tl.store(out_ptr + seq_idx * out_features + dim_idx, result, mask=dim_idx < out_features)



# Triton kernel for attention weights calculation (query * key^T)
@triton.jit
def attn_weights_kernel(query_ptr, key_ptr, out_ptr, head_dim, BLOCK_SIZE: tl.constexpr):
    batch_idx = tl.program_id(0)  # Batch index
    seq_idx = tl.program_id(1)  # Sequence index
    dim_idx = tl.arange(0, BLOCK_SIZE)  # Block of dimensions for parallelism

    # Load query and key
    query_val = tl.load(query_ptr + batch_idx * head_dim + dim_idx, mask=dim_idx < head_dim)
    key_val = tl.load(key_ptr + seq_idx * head_dim + dim_idx, mask=dim_idx < head_dim)

    # Compute attention weights (dot product)
    result = tl.dot(query_val, key_val)

    # Store the result in the output tensor
    tl.store(out_ptr + batch_idx * head_dim + seq_idx, result)


class LlamaAttention(nn.Module):
    """
    LlamaAttention is a multi-headed attention module based on the 'Attention Is All You Need' paper, 
    with Triton acceleration for linear projections and attention weights.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.pretraining_tp = config.pretraining_tp
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim, max_position_embeddings=self.max_position_embeddings, base=self.config.rope_theta
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.pretraining_tp > 1:
            key_value_slicing = (
                self.num_key_value_heads * self.head_dim
            ) // self.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [
                F.linear(hidden_states, query_slices[i])
                for i in range(self.pretraining_tp)
            ]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [
                F.linear(hidden_states, key_slices[i])
                for i in range(self.pretraining_tp)
            ]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [
                F.linear(hidden_states, value_slices[i])
                for i in range(self.pretraining_tp)
            ]
            value_states = torch.cat(value_states, dim=-1)

        else:
            # Apply Triton kernel for linear projections
            BLOCK_SIZE = 1024  # Adjust this for performance
            grid = lambda meta: (hidden_states.size(0),)  # Parallelize over batch size

            query_states = torch.empty((bsz, q_len, self.num_heads * self.head_dim), device=hidden_states.device, dtype=hidden_states.dtype)
            key_states = torch.empty((bsz, q_len, self.num_key_value_heads * self.head_dim), device=hidden_states.device, dtype=hidden_states.dtype)
            value_states = torch.empty((bsz, q_len, self.num_key_value_heads * self.head_dim), device=hidden_states.device, dtype=hidden_states.dtype)

            linear_proj_kernel[grid](
                hidden_states,
                self.q_proj.weight,
                query_states,
                self.hidden_size,
                self.num_heads * self.head_dim,
                BLOCK_SIZE
            )

            linear_proj_kernel[grid](
                hidden_states,
                self.k_proj.weight,
                key_states,
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                BLOCK_SIZE
            )

            linear_proj_kernel[grid](
                hidden_states,
                self.v_proj.weight,
                value_states,
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                BLOCK_SIZE
            )

        # Reshape query, key, and value tensors
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Compute rotary positional embeddings
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        # Triton kernel for computing attention weights
        attn_weights = torch.empty((bsz, self.num_heads, q_len, kv_seq_len), device=hidden_states.device, dtype=hidden_states.dtype)

        attn_weights_kernel[grid](
            query_states,
            key_states,
            attn_weights,
            self.head_dim,
            BLOCK_SIZE
        )

        attn_weights = attn_weights / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights += attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # Compute attention output
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)

        # Apply output projection
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# Triton kernel for layer normalization
@triton.jit
def layernorm_kernel(x_ptr, out_ptr, weight_ptr, bias_ptr, epsilon, hidden_size, BLOCK_SIZE: tl.constexpr):
    batch_idx = tl.program_id(0)
    dim_idx = tl.arange(0, BLOCK_SIZE)

    x = tl.load(x_ptr + batch_idx * hidden_size + dim_idx, mask=dim_idx < hidden_size)
    
    mean = tl.sum(x, axis=0) / hidden_size
    x_mean_sub = x - mean
    var = tl.sum(x_mean_sub * x_mean_sub, axis=0) / hidden_size
    inv_std = 1.0 / tl.sqrt(var + tl.load(epsilon))  # Load the value from the var pointer
    
    norm_x = x_mean_sub * inv_std
    if weight_ptr:
        weight = tl.load(weight_ptr + dim_idx, mask=dim_idx < hidden_size)
        norm_x = norm_x * weight
    if bias_ptr:
        bias = tl.load(bias_ptr + dim_idx, mask=dim_idx < hidden_size)
        norm_x = norm_x + bias
    
    tl.store(out_ptr + batch_idx * hidden_size + dim_idx, norm_x, mask=dim_idx < hidden_size)



class LlamaDecoderLayer(nn.Module):
    """
    LlamaDecoderLayer represents a single layer of the Llama decoder with Triton acceleration.
    """

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)  # Self-attention with Triton acceleration
        self.mlp = LlamaMLP(config)  # MLP with Triton acceleration
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # Layernorm
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # Layernorm

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Forward pass for the LlamaDecoderLayer with Triton optimization.

        Args:
            hidden_states (torch.FloatTensor): Input tensor of shape `(batch, seq_len, embed_dim)`.
            attention_mask (torch.FloatTensor, optional): Attention mask.
            position_ids (torch.LongTensor, optional): Positional IDs tensor.
            past_key_value (Tuple[torch.FloatTensor], optional): Cached past key and value projection states.
            output_attentions (bool, optional): Whether or not to return the attentions tensors of all attention layers.
            use_cache (bool, optional): If set to `True`, `past_key_values` key-value states are returned.

        Returns:
            Tuple: Output hidden states and optional attention weights and past key-value states.
        """

        residual = hidden_states
        bsz, seq_len, hidden_size = hidden_states.size()

        # Triton-accelerated input layer normalization
        normalized_hidden_states = torch.empty_like(hidden_states)
        BLOCK_SIZE = 1024  # Adjust block size for performance
        grid = lambda meta: (bsz,)
        layernorm_kernel[grid](
            hidden_states,
            normalized_hidden_states,
            self.input_layernorm.weight,
            None,
            self.input_layernorm.variance_epsilon,
            hidden_size,
            BLOCK_SIZE
        )

        # Self Attention with Triton acceleration
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=normalized_hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Post-attention layer normalization using Triton
        residual = hidden_states
        normalized_hidden_states = torch.empty_like(hidden_states)
        layernorm_kernel[grid](
            hidden_states,
            normalized_hidden_states,
            self.post_attention_layernorm.weight,
            None,
            self.post_attention_layernorm.variance_epsilon,
            hidden_size,
            BLOCK_SIZE
        )

        # Fully connected feed-forward MLP with Triton acceleration
        hidden_states = self.mlp(normalized_hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs



LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LlamaModel):
            module.gradient_checkpointing = value


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(
            self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                # inputs_embeds.dtype,
                torch.float32,  # [MODIFIED] force to cast to float32
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )


        if hasattr(self, "tree_mask") and self.tree_mask is not None:
            tree_mask = self.tree_mask
            tree_len = tree_mask.size(-1)
            combined_attention_mask[:, :, -tree_len:, -tree_len:][
                tree_mask == 0
                ] = combined_attention_mask.min()

        return combined_attention_mask

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values=None,  # [MODIFIED] past_key_value is KVCache class
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=inputs_embeds.device,
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            # if idx==16:
            #     print(idx)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC
    )
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values=None,  # [MODIFIED] past_key_value is KVCache class
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(
                self.vocab_size // self.pretraining_tp, dim=0
            )
            logits = [
                F.linear(hidden_states, lm_head_slices[i])
                for i in range(self.pretraining_tp)
            ]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            inputs_embeds=None,
            **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past


@add_start_docstrings(
    """
    The LLaMa Model transformer with a sequence classification head on top (linear layer).

    [`LlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError(
                "Cannot handle batch sizes > 1 if no padding token is defined."
            )
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (
                        torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
                ).to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[
            torch.arange(batch_size, device=logits.device), sequence_lengths
        ]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                        labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    pooled_logits.view(-1, self.num_labels), labels.view(-1)
                )
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
