import math
import torch
from models import LitScGraphLLM
from GNN_modules import *
from MLP_modules import *
from _globals import * ## these define the indices for the special tokens 
from transformer_modules import FlashTransformerEncoderLayer

class FlashTRAN(LitScGraphLLM):
    def __init__(self, config):
        super().__init__(config)
        tconfig = config.transformer_config

        print(tconfig.input_dim)

        self.transformer_encoder1 = FlashTransformerEncoderLayer(tconfig.input_dim, tconfig.num_heads, tconfig.feed_dim, tconfig.dropout, tconfig.activation, tconfig.batch_first)
        self.transformer_encoder2 = FlashTransformerEncoderLayer(tconfig.input_dim, tconfig.num_heads, tconfig.feed_dim, tconfig.dropout, tconfig.activation, tconfig.batch_first)

        self.link_prediction_head = LinkPredictHead(in_dim=config.model_config.node_embedding_dim *2 ,
                                               hidden_dim=config.model_config.gnn_config.hidden_dims[0])
        self.node_embedding = torch.nn.Embedding(config.model_config.num_genes + config.model_config.num_ranks, 
                                                 config.model_config.node_embedding_dim, padding_idx=PAD_IDX)
        self.gene_prediction_head = RobertaLMHead(config.model_config.node_embedding_dim*2, config.model_config.num_genes)
        self.rank_prediction_head = RobertaLMHead(config.model_config.node_embedding_dim*2, config.model_config.num_ranks)
        self.optim_config = config.optim_config
        self.loss_config = config.loss_config

        # self.pos_emb = config.pos_emb

    def forward(self, batch):
        # adding in more explicit annotations of shapes and workflow
        # some definitions: 
        # n = the number of cells in a batch; 
        # G = the total number of unique genes across the whole dataset; g = the number of unique genes in a batch ; g_dim = the dimension of the gene embeddings;
        # R = the total number of unique  *expressed genes* across the whole dataset; r_dim = the dimension of the rank embeddings; 
        # L_dim = the dimension of the learned cell embeddings(r_dim + g_dim);
        # e= the number of edges in a batch

        orig_gene_id = batch["orig_gene_id"]
        orig_rank_id = batch["orig_rank_indices"]
        mask_locs = [batch["gene_mask"], batch["rank_mask"], batch["both_mask"]]

        node_indices = torch.stack([orig_gene_id, orig_rank_id]).permute(1, 2, 0)
        node_indices = node_indices.type(torch.long)

        # Shapes: node_indices: n x g x 2
        node_embeddings = self.node_embedding(node_indices).flatten(2) ## maps n x g x embed_dim

        # Positional encoding
        # if pos_emb:
        #   pass
        
        # take in node embeddings with shape nodes x edim and return the same sized, updated node embeddings
        node_embeddings = self.transformer_encoder1(node_embeddings) # no shape changes, just updates inputs.
        node_embeddings = self.transformer_encoder2(node_embeddings) # no shape changes, just updates inputs.

        return node_embeddings, orig_gene_id, orig_rank_id, mask_locs



# class PositionalEncoding(nn.Module):

#     def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#         pe = torch.zeros(max_len, 1, d_model)
#         pe[:, 0, 0::2] = torch.sin(position * div_term)
#         pe[:, 0, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)

#     def forward(self, x: Tensor) -> Tensor:
        # """
        # Arguments:
        #     x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        # """
        # x = x + self.pe[:x.size(0)]
        # return self.dropout(x)