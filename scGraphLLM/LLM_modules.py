from flash_attn.modules.block import Block
from flash_attn.modules.mha import MHA
from flash_attn.modules.mlp import FusedMLP, Mlp
from functools import partial
import torch
import torch.nn as nn

## instantiate flash attention transformer block, with fused MLP, layernorm, and activation operations 
def init_transformer_block(config):
  ## instantiate partials for transformer block
  mixer_cls = partial(
        MHA,
        num_heads=config.num_attention_heads,
        cross_attn=False,
        dropout=config.attention_probs_dropout_prob,
        causal=False,
        fused_bias_fc=True,
        use_flash_attn=True,
        return_residual=False
    )
  assert config.hidden_act in ["gelu_new", "gelu_fast", "gelu_pytorch_tanh"], (
            "fused_mlp only " "supports approximate gelu"
        )
  mlp_cls = partial(
          FusedMLP,
          hidden_features=config.mlp_intermediate_size,
          activation = config.hidden_act,
          return_residual=False
      )
  ## using default LayerNorm eps for now
  block = Block(
      config.dim_encoder,
      mixer_cls,
      mlp_cls,
      prenorm=True,
      resid_dropout1=config.hidden_dropout_prob,
      resid_dropout2=config.hidden_dropout_prob,
      fused_dropout_add_ln=True,
      return_residual=False
  )
  return block 


class MLMEncoder(torch.nn.Module):
  def __init__(self, config):
    super(MLMEncoder, self).__init__()
    self.config = config
    self.encoder_stack = nn.ModuleList([init_transformer_block(config) for _ in range(config.num_layers_encoder)])
  def forward(self, x, key_padding_mask):
    for layer in self.encoder_stack:
      x = layer(x, key_padding_mask=key_padding_mask)
    return x