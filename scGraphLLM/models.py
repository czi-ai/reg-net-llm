# For the model class
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
from GNN_modules import *
from transformer_modules import FlashTransformerEncoderLayer
import lightning.pytorch as pl
from data import pad_make_masks

class LitScGraphLLM(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.gnn_encoder = baseGCN(**config.gnn_config)
        self.node_embedding = torch.nn.Embedding(config.node_embedding_size, config.node_embedding_dim)
        self.rank_embedding = torch.nn.Embedding(config.rank_embedding_size, config.rank_embedding_dim)
        self.mlm_encoder = FlashTransformerEncoderLayer(**config.mlm_config)
    def forward(self, batch ):
        node_indices, edge_list,edge_weights, rank_data = batch
        node_embeddings = self.node_embedding(node_indices)
        ## wasn't getting the gnn to work so jus commented it out for now
        ## but needs to basically take in node embeddings with shape nodes x edim and return the same sized, updated node embeddings
        #node_embeddings, edge_index, edge_weight, attn_weight = self.gnn_encoder(node_embeddings, edge_list, edge_weights)
        ranks, gene_indices = rank_data # both are nested tensors
        rank_embeddings, _ = pad_make_masks([self.rank_embedding(ranks[i]) for i in range(ranks.size(0))])
        gene_embeddings, attn_mask = pad_make_masks([node_embeddings[gene_indices[i]] for i in range(gene_indices.size(0))])
        full_cell_embedding = torch.cat([gene_embeddings, rank_embeddings], dim=2)
        return self.mlm_encoder(full_cell_embedding, attn_mask)


