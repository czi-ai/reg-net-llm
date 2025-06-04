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
from scGraphLLM.graph_op import _chebyshev_diffusion
from scGraphLLM.MLP_modules import PerturbEmbedding
from scGraphLLM._globals import *

## Keeping these just in case
def gelu(x):
    """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different
    (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))    



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
    def __init__(self, d_model, nhead, use_flash_attn, init_scheme = "kaiming_uniform", lora_qv_rank = None,bias=False, device=None, dtype=None):
        super(FusedWQKVwithPE, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        if lora_qv_rank is not None:
            self.Wqkv = lora.MergedLinear(d_model, 3*d_model, r=lora_qv_rank,
                                            enable_lora=[True, False, True])
        else:
            self.Wqkv = nn.Linear(d_model, 3 * d_model, bias=bias, **factory_kwargs)
        self.Wp = None  # Initialize Wp as None
        self.bias = bias
        self.nhead = nhead
        self.use_flash_attn = use_flash_attn
        if init_scheme == "xavier_uniform":
            nn.init.xavier_uniform_(self.Wqkv.weight)
        elif init_scheme == "xavier_normal":
            nn.init.xavier_normal_(self.Wqkv.weight)
        elif init_scheme == "xn_dim":
            nn.init.xavier_normal_(self.Wqkv.weight, gain = 2/math.sqrt(d_model))
    def forward(self, x, p):
        if self.Wp is None:
            self.Wp = nn.Linear(p.size(-1), x.size(-1), bias=self.bias).to(x.device)
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
    def __init__(
            self, 
            d_model, 
            num_heads, 
            batch_first, 
            attention_dropout, 
            mode="self", 
            bias=True, 
            causal=False, 
            diffusion_kernel_attn=False, 
            fine_tuning=False,
            use_rotary_emb=None, 
            device = None, 
            dtype = None
        ) -> None:
        #TODO: rename forward inputs for diffusion 
        super(FlashMHASelfMaskKV, self).__init__()
        assert batch_first
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.d_model = d_model
        self.causal = causal
        self.diffusion_kernel_attn= diffusion_kernel_attn
        self.dropout_p = attention_dropout
        self.mode = mode 
        self.num_heads = num_heads
        assert self.d_model % num_heads == 0, f"emb {self.d_model} must be divisible by num_heads {num_heads}"
        self.head_dim = self.d_model // num_heads
        # assert self.head_dim in [16, 32, 64, 128], "Only support head_dim == 16, 32, 64, or 128"
        assert (self.head_dim % 8 == 0) & (self.head_dim <= 128), 'heads divisible by 8'
        self.scaling = self.head_dim ** -0.5
        self.fine_tuning = fine_tuning
        self.use_rotary_emb = use_rotary_emb

        self.out_proj = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)
        if fine_tuning:
            self.perturb_emb = PerturbEmbedding(max_hop=4, embed_dim=self.d_model, 
                                            hidden_dim=100, output_dim=self.d_model)

    def forward(self, q, k,v, key_padding_mask=None, 
                edge_index_list=None, num_nodes_list=None, perturb_one_hot=None):
        """
        Credit: some elements adopted from OpenFold:
        https://github.com/aqlaboratory/openfold/blob/feed4ae22edf899b37bee49293fff902bdd64e2d/openfold/model/primitives.py#L660
        """
        
        q = q.bfloat16()
        k = k.bfloat16()
        v = v.bfloat16()
        dtype = q.dtype

        b_size, s_size, h_size, d_size = q.shape
        q_cu_seqlens = torch.arange(
            0, (b_size + 1) * s_size, step=s_size, dtype=torch.int32, device=q.device
        )

        if self.use_rotary_emb:
            q, k = self.rot_emb(q, k, seq_dimension=-3)
            
        if self.diffusion_kernel_attn:
            q_genes = q[:, 1:, :, :]
            q_cls = q[:, 0, :, :]
            q_genes_diffused = _chebyshev_diffusion(edge_index_list, num_nodes_list, q_genes, k=64, beta=BETA)
            
            # shift query by perturbational embedding if observing perturb seq data
            if self.fine_tuning:
                assert perturb_one_hot is not None, "need perturbation labels"
                perturb_bias = self.perturb_emb(edge_index_list, num_nodes_list, perturb_one_hot)
                assert q_genes_diffused.shape == perturb_bias.shape
                q_genes_diffused += perturb_bias
                
            q_final = torch.cat([q_cls.unsqueeze(1), q_genes_diffused], dim=1)
            q_final = q_final.bfloat16()
                
            
        q_final = rearrange(q_final.type(dtype), 'b s h d -> (b s) h d',
                            h=self.num_heads)
        q_final = q_final * self.scaling

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
            q_final,
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
        context = context.to(torch.float32)
        return self.out_proj(context)


class FlashTransformerEncoderLayer(nn.Module):
    """ 
    We assume transformer encoder layer is for the same sequence
    so use the FusedWqkv 
    """
    def __init__(
            self, 
            d_model, 
            nhead, 
            dim_feedforward, 
            dropout,
            activation, 
            batch_first, 
            use_flash_attn=True, 
            diffusion_kernel_attn=False, 
            use_PE=False,
            fine_tuning=False, 
            layer_norm_eps=1e-5, 
            norm_first=False,
            lora_qv_rank=None, 
            use_rotary_emb=False, 
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
        self.diffusion_kernel_attn = diffusion_kernel_attn
        self.fine_tuning = fine_tuning
        
        self.self_attention = FlashMHASelfMaskKV(
            d_model=d_model, num_heads = nhead, batch_first = batch_first, 
            attention_dropout=dropout, use_rotary_emb=use_rotary_emb, 
            diffusion_kernel_attn=diffusion_kernel_attn, fine_tuning=self.fine_tuning
        )
    
        ## Wqkv:
        self.use_PE = use_PE
        if use_PE:
            self.wqkv = FusedWQKVwithPE(d_model,nhead,use_flash_attn, lora_qv_rank = lora_qv_rank, 
                                        init_scheme = init_scheme, bias=True)
        else:
            self.wqkv = FusedWQKV(d_model,nhead,use_flash_attn, 
                                  lora_qv_rank = lora_qv_rank, init_scheme = init_scheme, bias=True)
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
        
    def forward(self, qkv, p=None, key_padding_mask=None, edge_index_list=None, 
                num_nodes_list=None, perturb_one_hot=None):
        x = qkv
        if self.use_PE:
            assert p is not None, "Positional encoding tensor must be provided when use_PE is True."
        
        if self.diffusion_kernel_attn:
            assert edge_index_list is not None, "Graph must be provided if using kernelized attention"
            assert num_nodes_list is not None, "Number of nodes must be provided if using kernelized attention"
            
        if self.norm_first:
            x = self.ln1(x)
            # infusing PE or not
            if self.use_PE:
                q, k, v = self.wqkv(x, p)
            else:
                q, k, v = self.wqkv(x)
            
            # mask attention weights or not
            if self.diffusion_kernel_attn:
                x = x + self.self_attention(q, k, v, 
                                            edge_index_list=edge_index_list, 
                                            num_nodes_list=num_nodes_list,
                                            perturb_one_hot=perturb_one_hot)
            else:
                x = x + self.self_attention(q, k, v, key_padding_mask=key_padding_mask)
            
            # layer norm -> linear network
            x = self.ln2(x)
            x = x + self.dropout_ff(self.ff(x))
        else:
            if self.use_PE:
                q, k, v = self.wqkv(x, p)
            else:
                q, k, v = self.wqkv(x)
            
            if self.diffusion_kernel_attn:
                x = x + self.self_attention(q, k, v, 
                                            edge_index_list=edge_index_list, 
                                            num_nodes_list=num_nodes_list,
                                            perturb_one_hot=perturb_one_hot)
            else:
                x = x + self.self_attention(q, k, v, key_padding_mask=key_padding_mask)

            # linear network -> layer norm
            x = self.ln1(x)
            x = x + self.dropout_ff(self.ff(x))
            x = self.ln2(x)
        return x
