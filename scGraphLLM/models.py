# For the model class
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
from GNN_modules import *
from MLP_modules import *
from transformer_modules import FlashTransformerEncoderLayer
import lightning.pytorch as pl
from data import pad_make_masks
from _globals import *

class LitScGraphLLM(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.gnn_encoder = baseGCN(**config.gnn_config)
        self.node_embedding = torch.nn.Embedding(config.node_embedding_size, config.node_embedding_dim)
        self.rank_embedding = torch.nn.Embedding(config.rank_embedding_size, config.rank_embedding_dim)
        self.mlm_encoder = FlashTransformerEncoderLayer(**config.mlm_config)
        self.prediction_head = RobertaLMHead(config.rank_embedding_size*2, config.node_embedding_size)
        self.optim_config = config.optim_config
    def forward(self, batch ):
        node_indices, edge_list,edge_weights, rank_data = batch
        node_embeddings = self.node_embedding(node_indices)
        ## wasn't getting the gnn to work so jus commented it out for now
        ## but needs to basically take in node embeddings with shape nodes x edim and return the same sized, updated node embeddings
        #node_embeddings, edge_index, edge_weight, attn_weight = self.gnn_encoder(node_embeddings, edge_list, edge_weights)
        ranks, rank_global_gene_indices, rank_local_gene_indices = rank_data # both are nested tensors
        rank_embeddings, _ = pad_make_masks([self.rank_embedding(ranks[i]) for i in range(ranks.size(0))])
        gene_embeddings, attn_mask = pad_make_masks([node_embeddings[rank_local_gene_indices[i]] for i in range(rank_local_gene_indices.size(0))])
        masked_full_cell_embedding, mask_locs = self.mask_tensor(torch.cat([gene_embeddings, rank_embeddings], dim=2))
        learned_cell_embedding = self.mlm_encoder(masked_full_cell_embedding, attn_mask)
        

        return learned_cell_embedding, rank_global_gene_indices, mask_locs
    def train_step(self, batch, batch_idx):
        learned_cell_embedding, rank_global_gene_indices, mask_locs = self(batch)
        predicted_gene_id= self.prediction_head(learned_cell_embedding)
        loss = self.mlm_loss(predicted_gene_id, rank_global_gene_indices, mask_locs)
        self.log('train_loss', loss)
        return loss
    def mask_tensor(self, tensor,mask_ratio=0.15):
        masked_tensor = tensor.clone()
        total_vectors = tensor.size(0) * tensor.size(1)
        num_to_mask = int(total_vectors * mask_ratio)
        batch_indices = torch.randint(0, tensor.size(0), (num_to_mask,))
        seq_indices = torch.randint(0, tensor.size(1), (num_to_mask,))
        mask_value = torch.cat([
            self.node_embedding(torch.tensor(MASK_IDX, device = tensor.device, dtype = torch.long)),
            self.rank_embedding(torch.tensor(MASK_IDX, device = tensor.device, dtype = torch.long))
            ])
        for i in range(num_to_mask):
            masked_tensor[batch_indices[i], seq_indices[i], :] = mask_value.clone()
        mask_locations = torch.zeros_like(tensor, dtype=torch.bool)
        mask_locations[batch_indices, seq_indices, :] = True
        return masked_tensor, mask_locations
    def mlm_loss(self, predicted_gene_id, rank_global_gene_indices, mask_locs):
        loss = F.cross_entropy(predicted_gene_id[mask_locs], rank_global_gene_indices[mask_locs[:,:,0]])
        return loss
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.optim_config)
        return optimizer




