# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch LLaMA model."""
import copy
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import math
from typing import List, Optional, Tuple, Union
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN


try:
    from .configs import EConfig
    from .utils_c import *
    from .choices import *
except:
    from configs import EConfig
    from utils_c import *
    from choices import *
    from utils import prepare_logits_processor


import triton
import triton.language as tl
import torch

# Updated _make_causal_mask function using Triton
@triton.jit
def _make_causal_mask_triton(mask_ptr, tgt_len, dtype_min, stride_m, stride_n):
    row_idx = tl.program_id(0)
    col_idx = tl.arange(0, tgt_len)

    mask_val = tl.load(mask_ptr + row_idx * stride_m + col_idx * stride_n, mask=col_idx < tgt_len)
    mask_val = tl.where(col_idx < row_idx + 1, 0.0, dtype_min)
    tl.store(mask_ptr + row_idx * stride_m + col_idx * stride_n, mask_val)

def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention with Triton optimization.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)

    # Triton kernel launch
    dtype_min = torch.finfo(dtype).min
    mask_ptr = mask.data_ptr()
    stride_m, stride_n = mask.stride()
    grid = (tgt_len,)
    _make_causal_mask_triton[(grid,)](mask_ptr, tgt_len, dtype_min, stride_m, stride_n)

    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Updated _expand_mask function using Triton
@triton.jit
def _expand_mask_triton(mask_ptr, inverted_mask_ptr, bsz, src_len, tgt_len, dtype_min, stride_m):
    b_idx = tl.program_id(0)
    s_idx = tl.arange(0, src_len)

    mask_val = tl.load(mask_ptr + b_idx * stride_m + s_idx, mask=s_idx < src_len)
    mask_exp = tl.where(mask_val > 0, 0.0, 1.0)
    
    for t_idx in range(tgt_len):
        tl.store(inverted_mask_ptr + b_idx * stride_m + t_idx * src_len + s_idx, 
                 tl.where(mask_exp > 0, dtype_min, 0.0), mask=s_idx < src_len)

def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]` using Triton optimization.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask

    # Triton kernel launch
    dtype_min = torch.finfo(dtype).min
    mask_ptr = expanded_mask.data_ptr()
    inverted_mask_ptr = inverted_mask.data_ptr()
    stride_m = mask.stride(0)
    
    grid = (bsz,)
    _expand_mask_triton[(grid,)](mask_ptr, inverted_mask_ptr, bsz, src_len, tgt_len, dtype_min, stride_m)

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), dtype_min)


# Updated repeat_kv function using Triton
@triton.jit
def repeat_kv_triton(hidden_ptr, out_ptr, bsz, num_kv_heads, n_rep, slen, head_dim, stride_b, stride_h, stride_s, stride_d):
    b_idx = tl.program_id(0)
    h_idx = tl.arange(0, num_kv_heads * n_rep)

    for s_idx in range(slen):
        val = tl.load(hidden_ptr + b_idx * stride_b + h_idx * stride_h + s_idx * stride_s + tl.arange(0, head_dim))
        tl.store(out_ptr + b_idx * stride_b + h_idx * stride_h + s_idx * stride_s + tl.arange(0, head_dim), val)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Efficient key-value repeating using Triton.
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states

    hidden_states_expanded = torch.empty(
        (batch, num_key_value_heads * n_rep, slen, head_dim), device=hidden_states.device, dtype=hidden_states.dtype
    )

    # Triton kernel launch
    hidden_ptr = hidden_states.data_ptr()
    out_ptr = hidden_states_expanded.data_ptr()
    stride_b, stride_h, stride_s, stride_d = hidden_states.stride()

    grid = (batch,)
    repeat_kv_triton[(grid,)](hidden_ptr, out_ptr, batch, num_key_value_heads, n_rep, slen, head_dim, stride_b, stride_h, stride_s, stride_d)

    return hidden_states_expanded


# rotate_half function remains as-is since it's already efficient in current form
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

# Triton kernel to apply rotary positional embedding
@triton.jit
def apply_rotary_pos_emb_triton(
    q_ptr, k_ptr, cos_ptr, sin_ptr, out_q_ptr, out_k_ptr,
    seq_len, dim, head_dim, stride_q, stride_k, stride_o, stride_c, stride_s
):
    batch_id = tl.program_id(0)
    seq_id = tl.program_id(1)
    dim_id = tl.arange(0, dim)

    # Load cos and sin values
    cos = tl.load(cos_ptr + seq_id * stride_c + dim_id)
    sin = tl.load(sin_ptr + seq_id * stride_s + dim_id)

    # Load q and k values
    q = tl.load(q_ptr + batch_id * stride_q + seq_id * head_dim + dim_id)
    k = tl.load(k_ptr + batch_id * stride_k + seq_id * head_dim + dim_id)

    # Rotate half and apply positional embedding
    q_rotated = tl.cat((-q[..., dim // 2:], q[..., :dim // 2]), dim=-1)
    k_rotated = tl.cat((-k[..., dim // 2:], k[..., :dim // 2]), dim=-1)

    # Compute the embedded q and k
    q_embed = q * cos + q_rotated * sin
    k_embed = k * cos + k_rotated * sin

    # Store the results
    tl.store(out_q_ptr + batch_id * stride_o + seq_id * head_dim + dim_id, q_embed)
    tl.store(out_k_ptr + batch_id * stride_o + seq_id * head_dim + dim_id, k_embed)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """
    Optimized version of the rotary position embedding application using Triton.
    """
    # Squeeze the first two dimensions of cos and sin
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]

    # Gather the cos and sin based on position_ids
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]

    batch_size, seq_len, head_dim = q.shape
    dim = cos.shape[-1]

    # Prepare output tensors for q and k embeddings
    q_embed = torch.empty_like(q)
    k_embed = torch.empty_like(k)

    # Triton kernel launch
    q_ptr = q.data_ptr()
    k_ptr = k.data_ptr()
    cos_ptr = cos.data_ptr()
    sin_ptr = sin.data_ptr()
    out_q_ptr = q_embed.data_ptr()
    out_k_ptr = k_embed.data_ptr()

    # Get the strides for the respective tensors
    stride_q = q.stride(0)
    stride_k = k.stride(0)
    stride_o = q_embed.stride(0)
    stride_c = cos.stride(0)
    stride_s = sin.stride(0)

    grid = (batch_size, seq_len)
    apply_rotary_pos_emb_triton[(grid,)](q_ptr, k_ptr, cos_ptr, sin_ptr, out_q_ptr, out_k_ptr, seq_len, dim, head_dim, stride_q, stride_k, stride_o, stride_c, stride_s)

    return q_embed, k_embed



import triton
import triton.language as tl
import torch

# Triton kernel to compute cos and sin cache for rotary embeddings
@triton.jit
def compute_rotary_cos_sin_kernel(t_ptr, inv_freq_ptr, cos_ptr, sin_ptr, seq_len, dim, stride_t, stride_f, stride_c, stride_s):
    seq_id = tl.program_id(0)
    dim_id = tl.arange(0, dim)

    # Load time steps and inverse frequency
    t = tl.load(t_ptr + seq_id * stride_t)
    inv_freq = tl.load(inv_freq_ptr + dim_id)

    # Compute frequency products
    freqs = t * inv_freq

    # Calculate cos and sin
    cos = tl.math.cos(freqs)
    sin = tl.math.sin(freqs)

    # Store the results in the output buffers
    tl.store(cos_ptr + seq_id * stride_c + dim_id, cos)
    tl.store(sin_ptr + seq_id * stride_s + dim_id, sin)


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build cos/sin cache for the default max sequence length
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        # Prepare tensors for time steps and frequencies
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        cos_cached = torch.empty((seq_len, self.dim), dtype=dtype, device=device)
        sin_cached = torch.empty((seq_len, self.dim), dtype=dtype, device=device)

        # Triton kernel launch for cos/sin computation
        t_ptr = t.data_ptr()
        inv_freq_ptr = self.inv_freq.data_ptr()
        cos_ptr = cos_cached.data_ptr()
        sin_ptr = sin_cached.data_ptr()

        # Define strides for Triton access
        stride_t = t.stride(0)
        stride_f = self.inv_freq.stride(0)
        stride_c = cos_cached.stride(0)
        stride_s = sin_cached.stride(0)

        # Launch the Triton kernel
        grid = (seq_len,)
        compute_rotary_cos_sin_kernel[grid](
            t_ptr, inv_freq_ptr, cos_ptr, sin_ptr, seq_len, self.dim, stride_t, stride_f, stride_c, stride_s
        )

        # Store cos and sin cache
        emb_cos = torch.cat((cos_cached, cos_cached), dim=-1).unsqueeze(0).unsqueeze(0)
        emb_sin = torch.cat((sin_cached, sin_cached), dim=-1).unsqueeze(0).unsqueeze(0)
        self.register_buffer("cos_cached", emb_cos, persistent=False)
        self.register_buffer("sin_cached", emb_sin, persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

# Triton kernel for computing cos and sin cache with scaling factor
@triton.jit
def compute_rotary_cos_sin_scaling_kernel(t_ptr, inv_freq_ptr, scaling_factor, cos_ptr, sin_ptr, seq_len, dim, stride_t, stride_f, stride_c, stride_s):
    seq_id = tl.program_id(0)
    dim_id = tl.arange(0, dim)

    # Load time steps and inverse frequency
    t = tl.load(t_ptr + seq_id * stride_t) / scaling_factor
    inv_freq = tl.load(inv_freq_ptr + dim_id)

    # Compute frequency products
    freqs = t * inv_freq

    # Calculate cos and sin
    cos = tl.math.cos(freqs)
    sin = tl.math.sin(freqs)

    # Store the results in the output buffers
    tl.store(cos_ptr + seq_id * stride_c + dim_id, cos)
    tl.store(sin_ptr + seq_id * stride_s + dim_id, sin)


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling."""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        # Prepare tensors for time steps and frequencies
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        cos_cached = torch.empty((seq_len, self.dim), dtype=dtype, device=device)
        sin_cached = torch.empty((seq_len, self.dim), dtype=dtype, device=device)

        # Triton kernel launch for cos/sin computation with scaling factor
        t_ptr = t.data_ptr()
        inv_freq_ptr = self.inv_freq.data_ptr()
        cos_ptr = cos_cached.data_ptr()
        sin_ptr = sin_cached.data_ptr()

        # Define strides for Triton access
        stride_t = t.stride(0)
        stride_f = self.inv_freq.stride(0)
        stride_c = cos_cached.stride(0)
        stride_s = sin_cached.stride(0)

        # Launch the Triton kernel
        grid = (seq_len,)
        compute_rotary_cos_sin_scaling_kernel[grid](
            t_ptr, inv_freq_ptr, self.scaling_factor, cos_ptr, sin_ptr, seq_len, self.dim, stride_t, stride_f, stride_c, stride_s
        )

        # Store cos and sin cache
        emb_cos = torch.cat((cos_cached, cos_cached), dim=-1).unsqueeze(0).unsqueeze(0)
        emb_sin = torch.cat((sin_cached, sin_cached), dim=-1).unsqueeze(0).unsqueeze(0)
        self.register_buffer("cos_cached", emb_cos, persistent=False)
        self.register_buffer("sin_cached", emb_sin, persistent=False)


# Triton kernel for computing cos and sin cache with dynamic NTK scaling
@triton.jit
def compute_rotary_cos_sin_dynamic_ntk_kernel(t_ptr, inv_freq_ptr, cos_ptr, sin_ptr, seq_len, dim, stride_t, stride_f, stride_c, stride_s):
    seq_id = tl.program_id(0)
    dim_id = tl.arange(0, dim)

    # Load time steps and inverse frequency
    t = tl.load(t_ptr + seq_id * stride_t)
    inv_freq = tl.load(inv_freq_ptr + dim_id)

    # Compute frequency products
    freqs = t * inv_freq

    # Calculate cos and sin
    cos = tl.math.cos(freqs)
    sin = tl.math.sin(freqs)

    # Store the results in the output buffers
    tl.store(cos_ptr + seq_id * stride_c + dim_id, cos)
    tl.store(sin_ptr + seq_id * stride_s + dim_id, sin)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling."""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            # Dynamic NTK scaling logic
            base = self.base * (
                    (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Prepare tensors for time steps and frequencies
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        cos_cached = torch.empty((seq_len, self.dim), dtype=dtype, device=device)
        sin_cached = torch.empty((seq_len, self.dim), dtype=dtype, device=device)

        # Triton kernel launch for cos/sin computation with dynamic NTK scaling
        t_ptr = t.data_ptr()
        inv_freq_ptr = self.inv_freq.data_ptr()
        cos_ptr = cos_cached.data_ptr()
        sin_ptr = sin_cached.data_ptr()

        # Define strides for Triton access
        stride_t = t.stride(0)
        stride_f = self.inv_freq.stride(0)
        stride_c = cos_cached.stride(0)
        stride_s = sin_cached.stride(0)

        # Launch the Triton kernel
        grid = (seq_len,)
        compute_rotary_cos_sin_dynamic_ntk_kernel[grid](
            t_ptr, inv_freq_ptr, cos_ptr, sin_ptr, seq_len, self.dim, stride_t, stride_f, stride_c, stride_s
        )

        # Store cos and sin cache
        emb_cos = torch.cat((cos_cached, cos_cached), dim=-1).unsqueeze(0).unsqueeze(0)
        emb_sin = torch.cat((sin_cached, sin_cached), dim=-1).unsqueeze(0).unsqueeze(0)
        self.register_buffer("cos_cached", emb_cos, persistent=False)
        self.register_buffer("sin_cached", emb_sin, persistent=False)

class LlamaAttentionTriton(nn.Module):
    """Multi-headed attention with Triton optimization."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(f"hidden_size must be divisible by num_heads "
                             f"(got `hidden_size`: {self.hidden_size}, `num_heads`: {self.num_heads})")

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    @triton.jit
    def triton_matmul(A, B, C, BLOCK_SIZE: tl.constexpr):
        """Triton kernel for matrix multiplication."""
        pid = tl.program_id(0)
        # Block index
        i = pid // BLOCK_SIZE
        j = pid % BLOCK_SIZE
        # Each thread processes one element of the output matrix
        acc = 0.0
        for k in range(0, BLOCK_SIZE):
            acc += A[i, k] * B[k, j]
        C[i, j] = acc

    @triton.jit
    def triton_softmax(Q, K, V, out, BLOCK_SIZE: tl.constexpr):
        """Triton kernel for softmax."""
        pid = tl.program_id(0)
        # Indexing into Q, K, V
        i = pid // BLOCK_SIZE
        j = pid % BLOCK_SIZE
        # Dot product and scaling
        dot_prod = Q[i, j] * K[j, i]
        scaled = dot_prod / tl.math.sqrt(BLOCK_SIZE)
        # Compute softmax
        out[i, j] = tl.math.exp(scaled) / tl.math.sum(tl.math.exp(scaled), axis=1)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None, past_key_value: Optional[Tuple[torch.Tensor]] = None,
                output_attentions: bool = False, use_cache: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # Triton-optimized matmul
        BLOCK_SIZE = 16
        attn_weights = torch.empty((bsz, self.num_heads, q_len, kv_seq_len), device=hidden_states.device)
        self.triton_matmul(query_states, key_states.transpose(2, 3), attn_weights, BLOCK_SIZE=BLOCK_SIZE)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Triton-optimized softmax
        attn_output = torch.empty_like(attn_weights)
        self.triton_softmax(attn_weights, key_states, value_states, attn_output, BLOCK_SIZE=BLOCK_SIZE)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights if output_attentions else None, None



class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # Initialize the weight matrices
        self.gate_proj = torch.nn.Parameter(torch.empty(self.hidden_size, self.intermediate_size))
        self.up_proj = torch.nn.Parameter(torch.empty(self.hidden_size, self.intermediate_size))
        self.down_proj = torch.nn.Parameter(torch.empty(self.intermediate_size, self.hidden_size))

        self.act_fn = ACT2FN[config.hidden_act]
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.gate_proj, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.up_proj, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.down_proj, a=math.sqrt(5))

    @triton.jit
    def triton_matmul(A_ptr, B_ptr, C_ptr, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        # Define the IDs of the program (block indices)
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        # Initialize offsets for the blocks
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        # Load blocks of A and B
        A = tl.load(A_ptr + offs_m[:, None] * K + offs_k[None, :], mask=offs_m[:, None] < M)
        B = tl.load(B_ptr + offs_k[:, None] * N + offs_n[None, :], mask=offs_n[None] < N)

        # Initialize accumulator
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Block matrix multiplication
        for k in range(0, K, BLOCK_K):
            A_block = tl.load(A + k * BLOCK_K, mask=offs_k[None, :] < K)
            B_block = tl.load(B + k * BLOCK_K, mask=offs_k[:, None] < K)
            acc += tl.dot(A_block, B_block)

        # Store the result
        C = tl.store(C_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.size()

        # Ensure input sizes are compatible
        assert hidden_size == self.hidden_size, f"Expected input hidden size {self.hidden_size}, got {hidden_size}"

        # Define block sizes for efficient tiling
        BLOCK_SIZE = 128

        # Define Triton kernel grid
        grid = lambda META: (triton.cdiv(batch_size * seq_len, META['BLOCK_M']),
                             triton.cdiv(self.intermediate_size, META['BLOCK_N']))

        # Allocate output buffers
        gate_output = torch.empty((batch_size, seq_len, self.intermediate_size), device=x.device, dtype=x.dtype)
        up_output = torch.empty((batch_size, seq_len, self.intermediate_size), device=x.device, dtype=x.dtype)
        down_output = torch.empty((batch_size, seq_len, self.hidden_size), device=x.device, dtype=x.dtype)

        # Perform gate_proj and up_proj operations in parallel using Triton
        self.triton_matmul[grid](
            x, self.gate_proj, gate_output, batch_size * seq_len, self.intermediate_size, self.hidden_size,
            BLOCK_M=BLOCK_SIZE, BLOCK_N=BLOCK_SIZE, BLOCK_K=BLOCK_SIZE
        )
        self.triton_matmul[grid](
            x, self.up_proj, up_output, batch_size * seq_len, self.intermediate_size, self.hidden_size,
            BLOCK_M=BLOCK_SIZE, BLOCK_N=BLOCK_SIZE, BLOCK_K=BLOCK_SIZE
        )

        # Apply activation function
        intermediate_states = self.act_fn(gate_output) * up_output

        # Perform down_proj operation using Triton
        self.triton_matmul[grid](
            intermediate_states, self.down_proj, down_output, batch_size * seq_len, self.hidden_size, self.intermediate_size,
            BLOCK_M=BLOCK_SIZE, BLOCK_N=BLOCK_SIZE, BLOCK_K=BLOCK_SIZE
        )

        return down_output


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm but optimized with Triton.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @triton.jit
    def rmsnorm_kernel(X_ptr, W_ptr, Y_ptr, hidden_size, eps, BLOCK_SIZE: tl.constexpr):
        """
        RMSNorm kernel: calculates normalization and scales the result.
        Args:
            X_ptr: Input tensor pointer.
            W_ptr: Weight tensor pointer.
            Y_ptr: Output tensor pointer.
            hidden_size: Size of the last dimension (feature size).
            eps: Small epsilon value to prevent division by zero.
            BLOCK_SIZE: Block size for tiling, must be divisible by hidden_size.
        """
        # Program ID
        pid = tl.program_id(0)

        # Compute offsets
        block_start = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        
        # Load the data block
        x = tl.load(X_ptr + block_start)
        
        # Calculate the variance (mean of squared elements)
        var = tl.sum(x * x) / hidden_size
        norm_factor = tl.rsqrt(var + eps)
        
        # Normalize the input
        x_normalized = x * norm_factor
        
        # Load the corresponding weight and apply it
        weight = tl.load(W_ptr + block_start)
        y = x_normalized * weight
        
        # Store the result
        tl.store(Y_ptr + block_start, y)

    def forward(self, hidden_states):
        # Define block size for the kernel, ensure it's divisible by hidden_size
        BLOCK_SIZE = 128  # Adjust based on your hardware and hidden size

        # Ensure the last dimension of hidden_states is divisible by BLOCK_SIZE
        assert self.hidden_size % BLOCK_SIZE == 0, "hidden_size must be divisible by BLOCK_SIZE"

        # Allocate output tensor
        output = torch.empty_like(hidden_states)

        # Define grid for Triton kernel
        grid = lambda META: (hidden_states.numel() // self.hidden_size,)

        # Execute the Triton kernel
        self.rmsnorm_kernel[grid](
            hidden_states,  # Input tensor
            self.weight,  # Weight parameter
            output,  # Output tensor
            self.hidden_size,  # Size of the last dimension (feature size)
            self.eps,  # Epsilon for variance normalization
            BLOCK_SIZE=BLOCK_SIZE  # Block size for efficient tiling
        )

        return output


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(config)
        self.index = index
        if self.index != 0:
            self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        if self.index != 0:
            hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class I(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.ones(1, dtype=torch.float32))

    def forward(self, x):
        return x + self.dummy - self.dummy  # (also tried x+self.dummy)


def len_list(x, n):
    return [i for i in x if len(i) <= n]


class Model(nn.Module):
    def __init__(self, config, load_emb=False, path=None, bias=True, total_tokens=63, depth=5, top_k=8, threshold=1.0):
        super().__init__()

        self.gradient_checkpointing = True
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        if load_emb:
            from safetensors import safe_open
            import json
            try:
                with open(os.path.join(path, "model.safetensors.index.json"), "r") as f:
                    index_json = json.loads(f.read())
                    emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
                with safe_open(os.path.join(path, emb_path),
                               framework="pt",
                               device="cpu") as f:
                    tensor_slice = f.get_slice("model.embed_tokens.weight")
                    vocab_size, hidden_dim = tensor_slice.get_shape()
                    tensor = tensor_slice[:, :hidden_dim].float()
            except:
                with open(os.path.join(path, "pytorch_model.bin.index.json"), "r") as f:
                    index_json = json.loads(f.read())
                    emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
                weights = torch.load(os.path.join(path, emb_path))
                tensor = weights["model.embed_tokens.weight"].float()
            self.embed_tokens.weight.data = tensor

        self.top_k = top_k
        self.total_tokens = total_tokens - 1
        self.depth = depth
        self.threshold = math.log(threshold)
        # print("total_tokens",total_tokens)
        # print("depth",depth)
        # print("top_k",top_k)
        # print("threshold",threshold)

        self.layers = nn.ModuleList([LlamaDecoderLayer(config, index) for index in range(config.num_hidden_layers)])
        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=bias)
        self.act = ACT2FN[config.hidden_act]
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        for param in self.embed_tokens.parameters():
            param.requires_grad = False

    def init_tree(self):
        self.tree_mask_init = torch.eye(self.top_k, device=self.embed_tokens.weight.device)[None, None]
        self.position_ids = torch.zeros(self.top_k, device=self.embed_tokens.weight.device, dtype=torch.long)
        self.tree_mask_init = self.tree_mask_init.to(self.embed_tokens.weight.device)

    def reset(self):
        self.tree_mask = None

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
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
            expanded_attn_mask = _expand_mask(attention_mask, torch.float32, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        # [MODIFIED] add tree mask
        if hasattr(self, "tree_mask") and self.tree_mask is not None:
            tree_mask = self.tree_mask
            _, _, tree_shape0, tree_shape1 = tree_mask.shape
            combined_attention_mask[:, :, -tree_shape0:, -tree_shape1:][
                tree_mask == 0
                ] = torch.finfo(torch.float32).min

        return combined_attention_mask

    def forward(
            self,
            hidden_states,
            input_ids,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            std=None
    ):
        batch_size, seq_length, _ = hidden_states.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0

        with torch.no_grad():
            inputs_embeds = self.embed_tokens(input_ids)
            # inputs_embeds = inputs_embeds.detach()

        # if std is not None:
        #     noise = torch.randn(inputs_embeds.size(),device=inputs_embeds.device) * std
        #     inputs_embeds=inputs_embeds+noise

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if position_ids is None:
            device = hidden_states.device if hidden_states is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        #position_ids=position_ids//4
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=hidden_states.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), hidden_states, past_key_values_length
        )

        # if self.gradient_checkpointing and self.training:
        #    if use_cache:
        #        use_cache = False

        # hidden_states=self.act(self.fc(torch.cat((inputs_embeds,hidden_states),dim=-1)))
        inputs_embeds = inputs_embeds.to(hidden_states.dtype)
        hidden_states = self.fc(torch.cat((inputs_embeds, hidden_states), dim=-1))

        all_hidden_states = () if output_hidden_states else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
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

        if use_cache:
            return hidden_states, next_decoder_cache

        return hidden_states

    def reset_kv(self):
        self.stable_kv = None

    @torch.no_grad()
    def topK_genrate(self, hidden_states, input_ids, head, logits_processor):

        input_ids = input_ids.to(hidden_states.device)
        total_tokens = self.total_tokens
        depth = self.depth
        top_k = self.top_k

        sample_token = input_ids[:, -1]

        scores_list = []
        parents_list = []
        ss_token = []

        input_ids = input_ids[:, 1:]
        input_ids = input_ids.to(hidden_states.device)

        len_posi = input_ids.shape[1]
        self.reset()

        # with Timer("draft many"):
        if hasattr(self, "stable_kv") and self.stable_kv is not None:
            kv_len = self.stable_kv[0][0].shape[2]
            out_hidden, past_key_values = self(hidden_states, input_ids=input_ids[:, kv_len:],
                                               past_key_values=self.stable_kv, use_cache=True)
        else:
            out_hidden, past_key_values = self(hidden_states, input_ids=input_ids, use_cache=True)
        self.stable_kv = past_key_values
        last_hidden = out_hidden[:, -1]

        last_headout = head(last_hidden)

        last_p = self.logsoftmax(last_headout)
        top = torch.topk(last_p, top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values
        scores = topk_p[0]
        scores_list.append(scores[None])
        parents_list.append(torch.zeros(1, dtype=torch.long, device=scores.device))
        ss_token.append(topk_index)
        input_ids = topk_index
        input_hidden = last_hidden[None].repeat(1, top_k, 1)
        tree_mask = self.tree_mask_init
        topk_cs_index = torch.arange(top_k, device=self.embed_tokens.weight.device)

        # 4
        for i in range(depth):
            self.tree_mask = tree_mask
            position_ids = len_posi + self.position_ids
            # with Timer("draft one"):
            out_hidden, past_key_values = self(input_hidden, input_ids=input_ids, past_key_values=past_key_values,
                                               position_ids=position_ids, use_cache=True)
            len_posi += 1

            # with Timer("sort1"):
            bias1 = top_k if i > 0 else 0
            bias2 = max(0, i - 1)
            bias = 1 + top_k ** 2 * bias2 + bias1
            parents = (topk_cs_index + bias)
            parents_list.append(parents)

            last_headout = head(out_hidden[0])
            last_p = self.logsoftmax(last_headout)

            top = torch.topk(last_p, top_k, dim=-1)
            topk_index, topk_p = top.indices, top.values

            cu_scores = topk_p + scores[:, None]

            topk_cs = torch.topk(cu_scores.view(-1), top_k, dim=-1)
            topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
            scores = topk_cs_p

            out_ids = topk_cs_index // top_k
            input_hidden = out_hidden[:, out_ids]
            # with Timer("2index"):
            #     in_ids = topk_cs_index % top_k
            #     input_ids = topk_index[out_ids, in_ids][None]
            # with Timer("1index"):
            input_ids = topk_index.view(-1)[topk_cs_index][None]
            # print(input_ids.equal(input_ids0))

            ss_token.append(topk_index)
            scores_list.append(cu_scores)
            tree_mask = torch.cat((tree_mask[:, :, out_ids], self.tree_mask_init), dim=3)

            # if self.threshold < 0 and cu_scores.max() < self.threshold:
            #     break

        # del parents_list,scores_list,ss_token
        # return draft_tokens, mask_index,tree_mask,tree_position_ids

        # with Timer("post"):

        scores_list = torch.cat(scores_list, dim=0).view(-1)
        ss_token_list = torch.cat(ss_token, dim=0).view(-1)
        top_scores = torch.topk(scores_list, total_tokens, dim=-1)
        top_scores_index = top_scores.indices
        top_scores_index = torch.sort(top_scores_index).values

        draft_tokens = ss_token_list[top_scores_index]
        draft_tokens = torch.cat((sample_token, draft_tokens), dim=0)

        draft_parents = torch.cat(parents_list, dim=0)[top_scores_index // top_k].long()
        mask_index = torch.searchsorted(top_scores_index, draft_parents - 1, right=False)
        # mask_index[(top_scores_index[mask_index]!=draft_parents - 1)]=-1
        mask_index[draft_parents == 0] = -1
        mask_index = mask_index + 1
        mask_index_list = mask_index.tolist()
        # with Timer("mask"):
        tree_mask = torch.eye(total_tokens + 1).bool()
        tree_mask[:, 0] = True
        for i in range(total_tokens):
            tree_mask[i + 1].add_(tree_mask[mask_index_list[i]])

        # with Timer("mask1"):
        #     tree_mask0 = [[False for _ in range(total_tokens + 1)] for _ in range(total_tokens + 1)]
        #     tree_mask0[0][0] = True
        #     for i in range(total_tokens):
        #         #tree_mask0[i + 1][0]=True
        #         tree_mask0[i + 1][i + 1] = True
        #         p=mask_index_list[i]
        #         tree_mask0[i + 1][p] = True
        #         while p:
        #             p=mask_index_list[p-1]
        #             tree_mask0[i + 1][p] = True
        #     tree_mask0 = torch.tensor(tree_mask0, dtype=torch.bool)
        #
        # print(tree_mask0.equal(tree_mask))
        tree_position_ids = torch.sum(tree_mask, dim=1) - 1

        tree_mask = tree_mask.float()[None, None]
        draft_tokens = draft_tokens[None]

        del parents_list, scores_list, ss_token, ss_token_list, draft_parents

        # with Timer("retrieve"):

        max_depth = torch.max(tree_position_ids) + 1
        noleaf_index = torch.unique(mask_index).tolist()
        noleaf_num = len(noleaf_index) - 1
        leaf_num = total_tokens - noleaf_num

        retrieve_indices = torch.zeros(leaf_num, max_depth.item(), dtype=torch.long) - 1
        retrieve_indices = retrieve_indices.tolist()

        rid = 0
        position_ids_list = tree_position_ids.tolist()

        for i in range(total_tokens + 1):
            if i not in noleaf_index:
                cid = i
                depth = position_ids_list[i]
                for j in reversed(range(depth + 1)):
                    retrieve_indices[rid][j] = cid
                    cid = mask_index_list[cid - 1]
                rid += 1

        if logits_processor is not None:
            maxitem = total_tokens + 5

            def custom_sort(lst):
                # sort_keys=[len(list)]
                sort_keys = []
                for i in range(len(lst)):
                    sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
                return sort_keys

            retrieve_indices = sorted(retrieve_indices, key=custom_sort)

        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
        del mask_index, mask_index_list, noleaf_index, noleaf_num, leaf_num, max_depth, rid
        tree_position_ids = tree_position_ids.to(hidden_states.device)

        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids

    @torch.no_grad()
    def acc(self, data, head, max_length=5):
        hidden_states = data["hidden_states"]
        input_ids = data["input_ids"]
        # attention_mask=data["attention_mask"]
        loss_mask = data["loss_mask"]
        sample_mask = data["sample_mask"]
        target = data["target"]
        total = [0 for _ in range(max_length)]
        correct = [0 for _ in range(max_length)]
        bs, sl = hidden_states.shape[0], hidden_states.shape[1]
        target_headout = head(target)
        hidden_states_headout = head(hidden_states)

        for i in range(bs):
            for j in range(sl):
                if loss_mask[i, j] == 0:
                    continue
                single_hidden_states = hidden_states[i, :j]
                single_input_ids = input_ids[i, :j]

                single_hidden_states = single_hidden_states[None, :, :]
                single_input_ids = single_input_ids[None, :]
                for k in range(max_length):
                    tmp_in_target_headout = hidden_states_headout[i, single_hidden_states.shape[1] - 1]
                    tmp_out_target_headout = target_headout[i, single_hidden_states.shape[1] - 1]
                    target_in_token = torch.argmax(tmp_in_target_headout)
                    target_out_token = torch.argmax(tmp_out_target_headout)
                    tmp_token = input_ids[i, single_hidden_states.shape[1] - 1]
                    tmp_sample_mask = sample_mask[i, single_hidden_states.shape[1] - 1]
                    if not (target_in_token == tmp_token):
                        break
                    out_hidden = self(single_hidden_states, input_ids=single_input_ids)
                    last_hidden = out_hidden[:, -1]
                    last_headout = head(last_hidden)
                    token = torch.argmax(last_headout)
                    total[k] += 1
                    if token == target_out_token:
                        correct[k] += 1
                    else:
                        for kk in range(k, max_length):
                            total[kk] += 1
                        break

                    single_hidden_states = torch.cat((single_hidden_states, out_hidden[:, -1:]), dim=1)
                    single_input_ids = torch.cat(
                        (single_input_ids, torch.tensor([[token]]).to(single_input_ids.device)), dim=1)

        acc = [correct[i] / total[i] for i in range(len(correct))]
        return acc


class Vhead(nn.Module):
    def __init__(self, ins=6566, outs=32000):
        super().__init__()
        self.fc = nn.Linear(ins, outs, bias=False)

    def forward(self, x):
        return self.fc(x)


import torch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    config = EConfig.from_pretrained('config.json')
    model = Model(config, load_emb=False)
    print(model)
