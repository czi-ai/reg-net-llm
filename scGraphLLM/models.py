# For the model class
import torch
import torch.nn as nn
import torch.nn.functional as F
from GNN_modules import *
from MLP_modules import *
from transformer_modules import FlashTransformerEncoderLayer
import lightning.pytorch as pl
from data import pad_make_masks
from _globals import *

class LitScGraphLLM(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.gnn_encoder = baseGCN(**config.model_config.gnn_config)
        self.node_embedding = torch.nn.Embedding(config.model_config.node_embedding_size, config.model_config.node_embedding_dim)
        self.rank_embedding = torch.nn.Embedding(config.model_config.rank_embedding_size, config.model_config.rank_embedding_dim)
        self.mlm_encoder = FlashTransformerEncoderLayer(**config.model_config.mlm_config)
        self.prediction_head = RobertaLMHead(config.model_config.rank_embedding_dim*2, config.model_config.node_embedding_size)
        self.optim_config = config.optim_config
        self.loss_config = config.loss_config
    def forward(self, batch ):
        node_indices, edge_list,edge_weights, rank_data = batch
        node_embeddings = self.node_embedding(node_indices)
        ## wasn't getting the gnn to work so jus commented it out for now
        ## but needs to basically take in node embeddings with shape nodes x edim and return the same sized, updated node embeddings
        #node_embeddings, edge_index, edge_weight, attn_weight = self.gnn_encoder(node_embeddings, edge_list, edge_weights)
        ranks, rank_global_gene_indices, rank_local_gene_indices = rank_data # both are nested tensors
        rank_embeddings= pad_make_masks([self.rank_embedding(ranks[i]) for i in range(ranks.size(0))], return_mask=False)
        gene_embeddings, attn_mask = pad_make_masks([node_embeddings[rank_local_gene_indices[i]] for i in range(rank_local_gene_indices.size(0))])
        global_gene_indices = pad_make_masks([rank_global_gene_indices[i] for i in range(rank_global_gene_indices.size(0))],  return_mask=False)
        ### NOTE: this masking setup will allow padded tokens to be included in the masked language modeling task
        ### we don't want to include this, so we will eventually need to refactor
        masked_full_cell_embedding, mask_locs = self.mask_tensor(torch.cat([gene_embeddings, rank_embeddings], dim=2))
        learned_cell_embedding = self.mlm_encoder(masked_full_cell_embedding, attn_mask)
        return learned_cell_embedding, global_gene_indices, mask_locs
    def training_step(self, batch, batch_idx):
        learned_cell_embedding, rank_global_gene_indices, mask_locs = self(batch)
        predicted_gene_id= self.prediction_head(learned_cell_embedding)
        loss = self.mlm_loss(predicted_gene_id, rank_global_gene_indices, mask_locs)
        self.log('train_loss', loss)
        return loss
    def mask_tensor(self, tensor,mask_ratio=0.15):
        """
        Given a tensor, mask a ratio of the vectors in the tensor
        This method of sampling allows for the same vector to be masked multiple times but also allows for the same vector to be masked multiple times
        """
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
        return masked_tensor, (batch_indices, seq_indices)
    def mlm_loss(self, predicted_gene_id, rank_global_gene_indices, mask_locs):
        batch_indices, seq_indices=mask_locs
        masked_predictions = predicted_gene_id[batch_indices, seq_indices, :]
        labels = rank_global_gene_indices[batch_indices, seq_indices]
        loss = F.cross_entropy(masked_predictions,labels)
        return loss
    def configure_optimizers(self):
        optim_fn = self.optim_config["optimizer"]
        optimizer = optim_fn(self.parameters(), **self.optim_config.args)
        return optimizer




