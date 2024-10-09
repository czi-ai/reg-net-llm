### VS: This is eric's flash attention transormer implementation 
## from https://github.com/aqlaboratory/esm_training/tree/main
## I've mostly focused on standardizing the API so we have a single FlassAttention class, and then multiple transformer classes that use it.


# TransformerLayer adapted from:
#    https://github.com/facebookresearch/esm/blob/main/esm/modules.py
# FlashMHA adopted from:
#    https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/flash_attention.py

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Note: RotaryEmbedding from FlashAttention is incompatible with ESM.
#  I took elements from both to get consistent results and optimize speed.
#  Both rotary embedding implementations are likely correct.


import torch
import torch.nn as nn
from einops import rearrange, repeat
from typing import Tuple 
import math
import torch.nn.functional as F
import importlib
from flash_attn.flash_attn_interface import flash_attn_varlen_kvpacked_func
from flash_attn.bert_padding import unpad_input
import loralib as lora
import importlib
import torch
from typing import Tuple
from einops import repeat
from fa2_custom_mask import flash_attention_custom_mask


def rotate_half(x):
    "from https://github.com/facebookresearch/esm"
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


#@torch.jit.script   # would require setting shape to static (or finite number of shapes)
def apply_rotary_pos_emb(x, cos, sin, seq_dimension: int = -2):
    "from https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/rotary.py"
    # NOTE: This could probably be moved to Triton

    # Handle a possible sequence length mismatch in between q and k
    cos = cos[:x.shape[seq_dimension], :]
    sin = sin[:x.shape[seq_dimension], :]
    if seq_dimension == -3:
        cos = cos[:, None, :]
        sin = sin[:, None, :]
    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbeddingESM(torch.nn.Module):
    """
    The rotary position embeddings from RoFormer_ (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.
    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration
    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox
    .. warning: Please note that this embedding is not registered on purpose, as it is transformative
        (it does not create the embedding dimension) and will likely be picked up (imported) on a ad-hoc basis
    """

    def __init__(self, dim: int, *_, **__):
        super().__init__()
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(self, x, seq_dimension=1):
        seq_len = x.shape[seq_dimension]

        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if (seq_len != self._seq_len_cached or self._cos_cached.device != x.device
            or self._cos_cached.dtype != x.dtype
        ):
            self._seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dimension], device=x.device, dtype=self.inv_freq.dtype)
            # Don't do einsum, it converts fp32 to fp16
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            # FlashAttention repeat (d 2) is upscaling, (2 d) is repeating channel
            # self._cos_cached = repeat(torch.cos(freqs).to(x.dtype), '... d -> ... (d 2)')
            # self._sin_cached = repeat(torch.sin(freqs).to(x.dtype), '... d -> ... (d 2)')
            
            self._cos_cached = repeat(torch.cos(freqs).to(x.dtype), '... d -> ... (2 d)')
            self._sin_cached = repeat(torch.sin(freqs).to(x.dtype), '... d -> ... (2 d)')
            # possibly another way:
            # self._cos_cached = torch.cos(freqs).to(x.dtype).view(freqs.shape[0],1,freqs.shape[1]).expand(-1, 2, -1).contiguous().view(freqs.shape[0], -1)
            # self._sin_cached = torch.sin(freqs).to(x.dtype).view(freqs.shape[0],1,freqs.shape[1]).expand(-1, 2, -1).contiguous().view(freqs.shape[0], -1)

        return self._cos_cached, self._sin_cached

    def forward(self, q: torch.Tensor, k: torch.Tensor,
                seq_dimension=-2) -> Tuple[torch.Tensor, torch.Tensor]:
        assert seq_dimension in [-2, -3]  # Either (bs, h, s, d) or (bs, s, h, d)
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(
            k, seq_dimension=seq_dimension
        )

        return (
            apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached, seq_dimension),
            apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached, seq_dimension),
        )



#@torch.jit.script  # would require setting shape to static (or finite number of shapes)

## Keeping these just in case
def gelu(x):
    """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different
    (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))    


class SwiGLUB(nn.Module):
    """SwisGLU activation with trainable per-channel beta, combine with fc1
    Replaces the first linear layer of FFN.
    Beta allows swish function to interpolate between linear(0) and relu(inf)
    SWISH: A SELF-GATED ACTIVATION FUNCTION arXiv:1710.05941
    GLU Variants Improve Transformer  arXiv:2002.05202v1
    """
    def __init__(self, dim_in, dim_out, bias=True):
        super().__init__()
        self.dim_out = dim_out
        self.beta = nn.Parameter(torch.ones(dim_in))
        self.linear = nn.Linear(dim_in, dim_out*2, bias=bias)
    
    def forward(self, x):
        x[..., :self.dim_out] *= self.beta  # gate
        x = self.linear(x)
        return F.silu(x[..., :self.dim_out]) * x[..., self.dim_out:]


class SwiGLU(nn.Module):
    """SwisGLU activation , combine with fc1
    Replaces the first linear layer of FFN.
    SWISH: A SELF-GATED ACTIVATION FUNCTION arXiv:1710.05941
    GLU Variants Improve Transformer  arXiv:2002.05202v1
    """
    def __init__(self, dim_in, dim_out, bias=True):
        super().__init__()
        self.dim_out = dim_out
        self.linear = nn.Linear(dim_in, dim_out*2, bias=bias)
    
    def forward(self, x):
        x = self.linear(x)
        return F.silu(x[..., :self.dim_out]) * x[..., self.dim_out:]  # gate * x

### Note: this implementation is a little more general than Erics, but is likely 
### slower due to the split WQKV into 3 linear layers instead of, when it could be 2(q, kv)
### the reasonf for this is so that the MHA class takes a q,k,v as as separate inputs,
### which lets us have the 1 MHA func. Then we stack k and v together, so we can use 
### flash_attn_varlen_kvpacked_func, but it may be better to add flash_attn_varlen_func  - 
### this would save the torch.stack call, but I don't know if it would be faster overall

## Removing factory kwargs

class FusedWQKV(nn.Module):
    def __init__(self, d_model,nhead,use_flash_attn, init_scheme = "kaiming_uniform", lora_qv_rank = None,bias=False, device=None, dtype=None):
        super(FusedWQKV, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        if lora_qv_rank is not None:
            self.Wqkv = lora.MergedLinear(d_model, 3*d_model, r=lora_qv_rank,
                                            enable_lora=[True, False, True])
        else:
            self.Wqkv = nn.Linear(d_model, 3 * d_model, bias=bias, **factory_kwargs)
        self.nhead = nhead
        self.use_flash_attn = use_flash_attn
        if init_scheme == "xavier_uniform":
            nn.init.xavier_uniform_(self.Wqkv.weight)
        elif init_scheme == "xavier_normal":
            nn.init.xavier_normal_(self.Wqkv.weight)
        elif init_scheme == "xn_dim":
            nn.init.xavier_normal_(self.Wqkv.weight, gain = 2/math.sqrt(d_model))
    def forward(self, x):
        qkv = self.Wqkv(x)
        if self.use_flash_attn:
            q, k, v = rearrange(qkv, 'b s (three h d) -> three b s h d', three=3, h=self.nhead)
        else:
            q, k, v = rearrange(qkv, 'b s (three h d) -> three b h s d',
                            three=3, h=self.nhead)
        return q,k,v


### fuse in positional embedding to q, k ,v #####
class FusedWQKVwithPE(nn.Module):
    def __init__(self, d_model, d_emb, nhead, use_flash_attn, init_scheme = "kaiming_uniform", lora_qv_rank = None,bias=False, device=None, dtype=None):
        super(FusedWQKVwithPE, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        if lora_qv_rank is not None:
            self.Wqkv = lora.MergedLinear(d_model, 3*d_model, r=lora_qv_rank,
                                            enable_lora=[True, False, True])
        else:
            self.Wqkv = nn.Linear(d_model, 3 * d_model, bias=bias, **factory_kwargs)
        self.Wp = nn.Linear(d_emb, d_model, bias=False, **factory_kwargs)
        self.nhead = nhead
        self.use_flash_attn = use_flash_attn
        if init_scheme == "xavier_uniform":
            nn.init.xavier_uniform_(self.Wqkv.weight)
            nn.init.xavier_uniform_(self.Wp.weight)
        elif init_scheme == "xavier_normal":
            nn.init.xavier_normal_(self.Wqkv.weight)
            nn.init.xavier_normal_(self.Wp.weight)
        elif init_scheme == "xn_dim":
            nn.init.xavier_normal_(self.Wqkv.weight, gain = 2/math.sqrt(d_model))
            nn.init.xavier_normal_(self.Wp.weight, gain = 2/math.sqrt(d_emb))
    def forward(self, x, p):
        p = self.Wp(p)
        h = x + p
        qkv = self.Wqkv(h)
        if self.use_flash_attn:
            q, k, v = rearrange(qkv, 'b s (three h d) -> three b s h d', three=3, h=self.nhead)
        else:
            q, k, v = rearrange(qkv, 'b s (three h d) -> three b h s d',
                            three=3, h=self.nhead)
        return q,k,v

class WQKV(nn.Module):
    
    def __init__(self, d_model,nhead, use_flash_attn,init_scheme = "kaiming_uniform",  lora_qv_rank = None, bias=False ):
        super(WQKV, self).__init__()
        if lora_qv_rank is not None:
            self.Wq = lora.MergedLinear(d_model, d_model, r=lora_qv_rank)
            self.Wk = nn.Linear(d_model, d_model, bias=bias)
            self.Wv = lora.MergedLinear(d_model, d_model, r=lora_qv_rank)
        else:
            self.Wq = nn.Linear(d_model, d_model, bias=bias)
            self.Wk = nn.Linear(d_model, d_model, bias=bias)
            self.Wv = nn.Linear(d_model, d_model, bias=bias)
        self.use_flash_attn = use_flash_attn
        self.nhead = nhead
        if init_scheme == "xavier_uniform":
            nn.init.xavier_uniform_(self.Wq.weight)
            nn.init.xavier_uniform_(self.Wk.weight)
            nn.init.xavier_uniform_(self.Wv.weight)
        elif init_scheme == "xavier_normal":
            nn.init.xavier_normal_(self.Wq.weight)
            nn.init.xavier_normal_(self.Wk.weight)
            nn.init.xavier_normal_(self.Wv.weight)
        elif init_scheme == "xn_dim":
            nn.init.xavier_normal_(self.Wq.weight, gain = 2/math.sqrt(d_model))
            nn.init.xavier_normal_(self.Wk.weight, gain = 2/math.sqrt(d_model))
            nn.init.xavier_normal_(self.Wv.weight, gain = 2/math.sqrt(d_model))

    def forward(self, q, k, v ):
        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)
        ## match FusedWQKV format 
        if self.use_flash_attn:
            q = rearrange(q, 'b s (h d) -> b s h d',  h=self.nhead)
            k = rearrange(k, 'b s (h d) -> b s h d',  h=self.nhead)
            v = rearrange(v, 'b s (h d) -> b s h d',  h=self.nhead)
        else:
            q= rearrange(q, 'b s (h d) -> b h s d',
                            h=self.nhead)
            k= rearrange(k, 'b s (h d) -> b h s d',
                            h=self.nhead)
            v= rearrange(v, 'b s (h d) -> b h s d',
                            h=self.nhead)
                            

        return q,k,v


class FlashMHASelfMaskKV(nn.Module):
    def __init__(self, d_model, num_heads, batch_first, attention_dropout,mode="self", bias=True, causal=False, 
                 use_rotary_emb=None, device = None, dtype = None) -> None:
        super(FlashMHASelfMaskKV, self).__init__()
        assert batch_first
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.d_model = d_model
        self.causal = causal
        self.dropout_p = attention_dropout
        self.mode = mode 

        self.num_heads = num_heads
        assert self.d_model % num_heads == 0, f"emb {self.d_model} must be divisible by num_heads {num_heads}"
        self.head_dim = self.d_model // num_heads
        # assert self.head_dim in [16, 32, 64, 128], "Only support head_dim == 16, 32, 64, or 128"
        assert (self.head_dim % 8 == 0) & (self.head_dim <= 128), 'heads divisible by 8'
        self.scaling = self.head_dim ** -0.5

        self.use_rotary_emb = use_rotary_emb
        if use_rotary_emb:
            self.rot_emb = RotaryEmbeddingESM(self.head_dim)

        

        self.out_proj = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)

    def forward(self, q,k,v, key_padding_mask=None):
        """x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
        key_padding_mask: bool tensor of shape (batch, seqlen)
        Credit: some elements adopted from OpenFold:
        https://github.com/aqlaboratory/openfold/blob/feed4ae22edf899b37bee49293fff902bdd64e2d/openfold/model/primitives.py#L660
        """
        dtype = q.dtype

        b_size, s_size, _, _ = q.shape
        q_cu_seqlens = torch.arange(
            0, (b_size + 1) * s_size, step=s_size, dtype=torch.int32, device=q.device
        )

        if self.use_rotary_emb:
            q, k = self.rot_emb(q, k, seq_dimension=-3)
        q = rearrange(q.type(dtype), 'b s h d -> (b s) h d',
                        h=self.num_heads)
        q = q * self.scaling

        # [b s 2 h d]
        kv = torch.stack([k.type(dtype), v], dim=2)

        if key_padding_mask is not None:
            kv = rearrange(kv, 'b s two h d -> b s (two h d)',
                            two=2, h=self.num_heads)
            key_padding_mask = key_padding_mask.type(dtype)
            kv_unpad, _, kv_cu_seqlens, kv_max_s = unpad_input(
                kv, key_padding_mask
            )
            kv_unpad = rearrange(kv_unpad, 'nnz (two h d) -> nnz two h d',
                                    two=2, h=self.num_heads)
        elif self.mode == "cross" and (key_padding_mask is None):
            ## case where we want to use cross attention but don't have a key_padding_mask
            ## This is not the most efficient way to do this,
            kv_unpad = rearrange(kv, 'b s two h d -> (b s) two h d',
                                     two=2, h=self.num_heads)
            kv_cu_seqlens = torch.arange(
                0, (b_size + 1) * k.shape[1], step=k.shape[1],
                dtype=torch.int32, device=q.device
            )
            kv_max_s = k.shape[1]
        else:
            ### self attention, no pads 
            kv_unpad = rearrange(kv, 'b s two h d -> (b s) two h d',
                                    two=2, h=self.num_heads)
            kv_cu_seqlens = q_cu_seqlens
            kv_max_s = s_size

        context = flash_attn_varlen_kvpacked_func(
            q,
            kv_unpad,
            q_cu_seqlens,
            kv_cu_seqlens,
            s_size,
            kv_max_s,
            dropout_p=self.dropout_p ,
            softmax_scale=1.,  # apply on q above
        )
        context = rearrange(context, '(b s) h d -> b s (h d)',
                            b=b_size, h=self.num_heads)
        return self.out_proj(context)


class CustomTorchMHASelf(nn.Module):

    def __init__(self, d_model, num_heads, bias=True, batch_first=True, attention_dropout=0.0,
                 causal=False, use_rotary_emb=None,device=None, dtype=None) -> None:
        assert batch_first
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(CustomTorchMHASelf, self).__init__()
        self.d_model = d_model
        self.causal = causal
        self.dropout_p = attention_dropout

        self.num_heads = num_heads
        assert self.d_model % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.d_model // num_heads
        # assert self.head_dim in [16, 32, 64, 128], "Only support head_dim == 16, 32, 64, or 128"
        assert (self.head_dim % 8 == 0) & (self.head_dim <= 128), 'heads divisible by 8'
        self.scaling = self.head_dim ** -0.5

        self.use_rotary_emb = use_rotary_emb
        if use_rotary_emb:
            self.rot_emb = RotaryEmbeddingESM(self.head_dim)

        self.out_proj = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)

    def forward(self, q,k,v, key_padding_mask=None):
        """x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
        key_padding_mask: bool tensor of shape (batch, seqlen)
        Credit: some elements adopted from OpenFold:
        https://github.com/aqlaboratory/openfold/blob/feed4ae22edf899b37bee49293fff902bdd64e2d/openfold/model/primitives.py#L660
        """
        ## input is b h s d
        b_size, _, s_size, _ = q.shape

        if self.use_rotary_emb:
            q, k = self.rot_emb(q, k, seq_dimension=-2)
        # scaling happens in scaled_dot_product_attention
        # q = q * self.scaling

        if key_padding_mask is not None:
            key_padding_mask = rearrange(~key_padding_mask, 'b s -> b 1 1 s')

        
        context = F.scaled_dot_product_attention(q, k, v, attn_mask=key_padding_mask, dropout_p=self.dropout_p, is_causal=self.causal)

        context = rearrange(context, 'b h s d -> b s (h d)',
                            b=b_size, h=self.num_heads)
        return self.out_proj(context)

# Allows custom masking at a batch level. Mask shape is (1, H, L, L) where H is num heads, L is seq length
class FlashMHACustomMasking(nn.Module):
    def __init__(self, d_model, num_heads, batch_first, attention_dropout, bias=True, device = None, dtype = None) -> None:
        super(FlashMHACustomMasking, self).__init__()
        assert batch_first
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.d_model = d_model
        self.dropout_p = attention_dropout

        self.num_heads = num_heads
        assert self.d_model % num_heads == 0, f"emb {self.d_model} must be divisible by num_heads {num_heads}"
        self.head_dim = self.d_model // num_heads
        # assert self.head_dim in [16, 32, 64, 128], "Only support head_dim == 16, 32, 64, or 128"
        assert (self.head_dim % 8 == 0) & (self.head_dim <= 128), 'heads divisible by 8'
        self.scaling = self.head_dim ** -0.5        

        self.out_proj = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)

    def forward(self, q,k,v, attention_mask=None):
        """
        q, k, v: (batch, heads, seqlen, model dim)
        attention mask: (1, heads, seqlen, seqlen)
        package used: https://github.com/alexzhang13/flashattention2-custom-mask/blob/main/fa2_custom_mask/fa2_custom_mask.py
        ONLY COMPATIABLE WITH AMPERE DEVICE AND UP. V100 DOES NOT WORK.
        """
        dtype = q.dtype
        q = rearrange(q.type(dtype), 'b s h d -> b h s d', h=self.num_heads)
        k = rearrange(k.type(dtype), 'b s h d -> b h s d', h=self.num_heads)
        v = rearrange(k.type(dtype), 'b s h d -> b h s d', h=self.num_heads)
        batch_size, seq_len, _, _ = q.shape
        attention_mask = torch.broadcast_to(attention_mask, (batch_size, self.num_heads, seq_len, seq_len))
        context = flash_attention_custom_mask(
            q,
            k,
            v,
            mask=attention_mask, # (B, H, L, L)å
            sm_scale=self.scaling
        )
        context = rearrange(context, '(b s) h d -> b s (h d)', b=q.shape[0], h=self.num_heads)
        return self.out_proj(context)

class FlashTransformerEncoderLayer(nn.Module):
    """ 
    We assume transformer encoder layer is for the same sequence
    so use the FusedWqkv 
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout ,
                 activation,batch_first,use_flash_attn = True, use_attn_mask = False, use_PE=False,
                 layer_norm_eps = 1e-5, norm_first = False,lora_qv_rank = None, use_rotary_emb = False, 
                 init_scheme = "kaiming_uniform"
                 ):
        super(FlashTransformerEncoderLayer, self).__init__()
        if activation == 'esm-gelu':
            self.activation = gelu
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation.startswith('SwiGLU'):  # combined with fc1
            self.activation = None
        else:
            raise ValueError(f'TransformerLayer {activation} not implemented')
        ## MHA
        self.use_attn_mask = use_attn_mask
        if use_flash_attn:
            self.self_attention = FlashMHASelfMaskKV(
                d_model=d_model, num_heads = nhead, batch_first = batch_first, attention_dropout=dropout, use_rotary_emb=use_rotary_emb
            )
            if use_attn_mask:
                self.self_attention = FlashMHACustomMasking(
                    d_model=d_model, num_heads = nhead, batch_first = batch_first, attention_dropout=dropout
                )
        else:
            self.self_attention = CustomTorchMHASelf(
                d_model=d_model, num_heads = nhead, batch_first = batch_first, attention_dropout=dropout, use_rotary_emb=use_rotary_emb
            )
    
        ## Wqkv:
        self.use_PE = use_PE
        if use_PE:
            self.wqkv = FusedWQKVwithPE(d_model,nhead,use_flash_attn, lora_qv_rank = lora_qv_rank, init_scheme = init_scheme, bias=True)
        else:
            self.wqkv = FusedWQKV(d_model,nhead,use_flash_attn, lora_qv_rank = lora_qv_rank, init_scheme = init_scheme, bias=True)
        ## feedforward projection
        if activation.startswith('SwiGLU'):
            self.ff = eval(activation)(d_model, dim_feedforward)
        else:
            self.ff = nn.Sequential(
                nn.Linear(d_model, dim_feedforward), 
                self.activation,
                nn.Linear(dim_feedforward, d_model) )
        
        self.dropout_ff = nn.Dropout(dropout)
        self.norm_first = norm_first
        self.ln1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.ln2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
    def forward(self, qkv, p=None, key_padding_mask = None, attn_mask = None):
        x = qkv
        if self.use_PE:
            assert p != None
        
        if self.use_attn_mask:
            assert attn_mask != None
            
        if self.norm_first:
            x = self.ln1(x)
            # infusing PE or not
            if self.use_PE:
                q,k,v = self.wqkv(x, p)
            else:
                q,k,v = self.wqkv(x)
            
            # mask attention weights or not
            if self.use_attn_mask:
                x = x + self.self_attention(q,k,v,attn_mask)
            else:
                x = x + self.self_attention(q,k,v,key_padding_mask)
            
            # layer norm -> linear network
            x = self.ln2(x)
            x = x + self.dropout_ff(self.ff(x))
        else:
            if self.use_PE:
                q,k,v = self.wqkv(x, p)
            else:
                q,k,v = self.wqkv(x)
            
            if self.use_attn_mask:
                x = x + self.self_attention(q,k,v,attn_mask)
            else:
                x = x + self.self_attention(q,k,v,key_padding_mask)

            # linear network -> layer norm
            x = self.ln1(x)
            x = x + self.dropout_ff(self.ff(x))
            x = self.ln2(x)
        return x

class FlashTransformerDecoderLayer(nn.Module):
    """ 
    We assume transformer encoder layer is for the same sequence
    so use the FusedWqkv 
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout ,activation,batch_first,use_flash_attn = True, layer_norm_eps = 1e-5, norm_first = False, use_rotary_emb = False,lora_qv_rank = None, init_scheme = "kaiming_uniform"
                 ):
        super(FlashTransformerEncoderLayer, self).__init__()
        if activation == 'esm-gelu':
            self.activation = gelu
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation.startswith('SwiGLU'):  # combined with fc1
            self.activation = None
        else:
            raise ValueError(f'TransformerLayer {activation} not implemented')
        ## MHA
        if use_flash_attn:
            self.self_attention = FlashMHASelfMaskKV(
                d_model=d_model, num_heads = nhead, batch_first = batch_first, attention_dropout=dropout, use_rotary_emb=use_rotary_emb
            )

            self.cross_attention = FlashMHASelfMaskKV(
                d_model=d_model, num_heads = nhead, batch_first = batch_first, attention_dropout=dropout, use_rotary_emb=use_rotary_emb,mode = "cross"
            )
        else:
            self.self_attention = CustomTorchMHASelf(
                d_model=d_model, num_heads = nhead, batch_first = batch_first, attention_dropout=dropout, use_rotary_emb=use_rotary_emb
            )

            self.cross_attention = CustomTorchMHASelf(
                d_model=d_model, num_heads = nhead, batch_first = batch_first, attention_dropout=dropout, use_rotary_emb=use_rotary_emb
            )

        ## Wqkv:
        self.wqkv = WQKV(d_model,nhead, use_flash_attn= use_flash_attn, lora_qv_rank = lora_qv_rank,init_scheme = init_scheme, bias=True)
        ## feedforward projection
        if activation.startswith('SwiGLU'):
            self.ff = eval(activation)(d_model, dim_feedforward)
        else:
            self.ff = nn.Sequential(
                nn.Linear(d_model, dim_feedforward), 
                self.activation,
                nn.Linear(dim_feedforward, d_model) )
        
        self.dropout_ff = nn.Dropout(dropout)
        self.norm_first = norm_first
        self.ln1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.ln2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.ln3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
    def forward(self, q,kv, key_padding_mask=None):
        

        if self.norm_first:
            q = self.ln1(q)
            x,k,v = self.wqkv(q, kv, kv)
            x = x + self.self_attention(q,k,v,key_padding_mask)
            x = self.ln2(x)
            x = x + self.cross_attention(x,kv,kv,key_padding_mask)
            x = self.ln3(x)
            x = x + self.dropout_ff(self.ff(x))
        else:
            x,k,v = self.wqkv(q, kv, kv)
            x = x + self.self_attention(q,k,v,key_padding_mask)
            x = self.ln1(x)
            x = x + self.cross_attention(x,kv,kv,key_padding_mask)
            x = self.ln2(x)
            x = x + self.dropout_ff(self.ff(x))
            x = self.ln3(x)
        return x     

class FlashTransformerCrossAttnLayer(nn.Module):
    """ 
    We assume transformer encoder layer is for the same sequence
    so use the FusedWqkv 
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout , activation,batch_first,use_flash_attn = True,layer_norm_eps = 1e-5, norm_first = False, use_rotary_emb = False, lora_qv_rank = None, init_scheme = "kaiming_uniform"):
        super(FlashTransformerCrossAttnLayer, self).__init__()
        if activation == 'esm-gelu':
            self.activation = gelu
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation.startswith('SwiGLU'):  # combined with fc1
            self.activation = None
        else:
            raise ValueError(f'TransformerLayer {activation} not implemented')
        ## MHA
        ##NOTE: rotary not set up for cross attn, always set to false
        ## arg is just included for compatibility with other classes
        if use_flash_attn:
            self.cross_attention = FlashMHASelfMaskKV(
                d_model=d_model, num_heads = nhead, batch_first = batch_first, attention_dropout=dropout, use_rotary_emb=False, mode  =  "cross"
            )
        else:
            self.cross_attention = CustomTorchMHASelf(
                d_model=d_model, num_heads = nhead, batch_first = batch_first, attention_dropout=dropout, use_rotary_emb=False
            )

        ## Wqkv:
        self.wqkv = WQKV(d_model,nhead, use_flash_attn = use_flash_attn, bias=True, lora_qv_rank = lora_qv_rank, init_scheme = init_scheme)
        ## feedforward projection
        if activation.startswith('SwiGLU'):
            self.ff = eval(activation)(d_model, dim_feedforward)
        else:
            self.ff = nn.Sequential(
                nn.Linear(d_model, dim_feedforward), 
                self.activation,
                nn.Linear(dim_feedforward, d_model) )
        
        self.dropout_ff = nn.Dropout(dropout)
        self.norm_first = norm_first
        self.ln1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.ln2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(self, x,kv, key_padding_mask = None):
        
        if self.norm_first:
            x = self.ln1(x)
            q,k,v = self.wqkv(x, kv, kv)
            x = x + self.cross_attention(q,k,v,key_padding_mask)
            x = self.ln2(x)
            x = x + self.dropout_ff(self.ff(x))
        else:
            q,k,v = self.wqkv(x, kv, kv)
            x = x + self.cross_attention(q,k,v,key_padding_mask)
            x = self.ln1(x)
            x = x + self.dropout_ff(self.ff(x))
            x = self.ln2(x)
        return x
