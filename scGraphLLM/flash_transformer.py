# For the model class
import torch
import torch.nn as nn
import torch.nn.functional as F
from GNN_modules import *
from MLP_modules import *
from transformer_modules import *
import lightning.pytorch as pl
from _globals import * ## these define the indices for the special tokens 
from torch_geometric.utils import negative_sampling

class FlashTRAN(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        
        self.transformer_encoder = FlashTransformerEncoderLayer(config.model_config.input_dim, 
                                                                config.model_config.num_heads, 
                                                                config.model_config.feed_dim, 
                                                                config.model_config.dropout, 
                                                                config.model_config.activation, 
                                                                config.model_config.batch_first)
        
        self.node_embedding = torch.nn.Embedding(config.model_config.num_genes + config.model_config.num_ranks, 
                                                 config.model_config.node_embedding_dim, 
                                                 padding_idx=PAD_IDX)
        
        self.gene_prediction_head = RobertaLMHead(config.model_config.node_embedding_dim*2, 
                                                  config.model_config.num_genes)
        
        self.rank_prediction_head = RobertaLMHead(config.model_config.node_embedding_dim*2, 
                                                  config.model_config.num_ranks)
        
        self.optim_config = config.optim_config
        self.loss_config = config.loss_config
        
    def forward(self, batch):
        # Extract data from batch
        orig_gene_id = batch["orig_gene_id"]
        orig_rank_id = batch["orig_rank_indices"]
        mask_locs = [batch["gene_mask"], batch["rank_mask"], batch["both_mask"]]
        
        # Get individual embeddings, then the combined (gene and rank) embedding
        node_embedding = self.node_embedding(orig_gene_id)
        rank_embedding = self.node_embedding(orig_rank_id)
        combined_embedding = torch.concat([node_embedding, rank_embedding], dim=2)
        
        # Pass combined embedding through Transformer layer:
        # Take in node embeddings with shape nodes x edim and return the same sized, updated node embeddings
        combined_embedding = self.transformer_encoder(combined_embedding) # no shape changes

        return combined_embedding, orig_gene_id, orig_rank_id, mask_locs
    
    def _step(self, batch, batch_idx):
        # Forward Pass: 
        # Get learned cell embedding, target gene IDs, target rank IDs, and mask locations (IDs?)
        learned_cell_embedding, target_gene_ids, target_rank_ids, mask_locs = self(batch)
        
        # Predict the missing gene (based on the cell embedding)
        predicted_gene_id= self.gene_prediction_head(learned_cell_embedding) 
        # Predict the rank of this missing gene (based on the cell embedding)
        predicted_rank_id= self.rank_prediction_head(learned_cell_embedding)
        
        # Extract mask locations
        gene_mask_locs, rank_mask_locs, both_mask_locs = mask_locs
        
        # ---- Loss ----
        # Technically we could run the only/both together in a single pass, but this way we can track each one separately
        # Loss (gene)
        L_mlm_gene_ONLY = self.mlm_loss(predicted_gene_id, target_gene_ids, gene_mask_locs)
        L_mlm_gene_BOTH = self.mlm_loss(predicted_gene_id, target_gene_ids, both_mask_locs)

        # Loss (rank)
        target_rank_ids = target_rank_ids - NUM_GENES # Shift the rank indices to start from 0
        L_mlm_rank_ONLY = self.mlm_loss(predicted_rank_id, target_rank_ids, rank_mask_locs)
        L_mlm_rank_BOTH = self.mlm_loss(predicted_rank_id, target_rank_ids, both_mask_locs)
        
        # TOTAL LOSS: MLM (gene ONLY + gene BOTH + rank ONLY + rank BOTH)
        loss = L_mlm_gene_ONLY + L_mlm_gene_BOTH + L_mlm_rank_ONLY + L_mlm_rank_BOTH # + L_g
        
        # ---- Perplexity ----
        # Gene & Rank perplexity calculated independently
        gene_pp = self.pseudo_perp(predicted_gene_id, target_gene_ids, gene_mask_locs | both_mask_locs)
        rank_pp = self.pseudo_perp(predicted_rank_id, target_rank_ids, rank_mask_locs | both_mask_locs)
        
        
        # ---- LOGGING ----
        if type(batch) == dict:
            subset = batch['dataset_name'][0]
        else:
            subset = batch.dataset_name[0]

        self.log(f'{subset}_loss', loss, batch_size=1, add_dataloader_idx=False)
        self.log(f'{subset}_mlm_geneonly_loss', L_mlm_gene_ONLY, batch_size=gene_mask_locs.sum(), add_dataloader_idx=False)
        self.log(f'{subset}_mlm_geneboth_loss', L_mlm_gene_BOTH, batch_size=(gene_mask_locs|both_mask_locs).sum(), add_dataloader_idx=False) 
        self.log(f'{subset}_mlm_rankonly_loss', L_mlm_rank_ONLY, batch_size=rank_mask_locs.sum(), add_dataloader_idx=False)
        self.log(f'{subset}_mlm_rankboth_loss', L_mlm_rank_BOTH, batch_size=(rank_mask_locs|both_mask_locs).sum(), add_dataloader_idx=False )
        self.log(f"{subset}_gene_perplexity", gene_pp, batch_size=1, add_dataloader_idx=False)
        self.log(f"{subset}_rank_perplexity", rank_pp, batch_size=1, add_dataloader_idx=False)
        
        return loss
        
    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        return loss
        

    def validation_step(self, batch, batch_idx, dataloader_idx):
        loss = self._step(batch, batch_idx)
        return loss


    def mlm_loss(self, predicted_gene_id, rank_global_gene_indices, mask_locs):
        masked_predictions = predicted_gene_id[mask_locs, :] # because we record the location of the masked tokens, we can retrieve just the masked tokens, and collapse the first dimension, ie mapping from n x r x G to m x G where m is the number of masked tokens
        labels = rank_global_gene_indices[mask_locs]
        loss = F.cross_entropy(masked_predictions, labels)
        return loss

    # def alignment_loss():
    #     pass

    def pseudo_perp(self, predicted_gene_id, rank_global_gene_indices, mask_locs):
        masked_predictions = predicted_gene_id[mask_locs, :] ## because we record the location of the masked tokens, we cna retrieve just the masked tokens, and collapse the first dimension, ie mapping from n x r x G to m x G where m is the number of masked tokens
        labels = rank_global_gene_indices[mask_locs]

        # Get softmax probabilities
        sft = nn.Softmax(dim=1)
        probabilities = sft(masked_predictions)

        L = labels.shape[0]
        # Get the summed log-likelihoods
        sum_llk = sum([torch.log(probabilities[i, labels[i]]) for i in range(L)]) # sum of log likelihood # get the model's softmax probability for the correct token
        # Pseudo-perplexity calculation
        pp = torch.exp(-(1/L)*sum_llk)
        return pp

    def configure_optimizers(self):
        optim_fn = self.optim_config["optimizer"]
        optimizer = optim_fn(self.parameters(), **self.optim_config.args)
        return optimizer
