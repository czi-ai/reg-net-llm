import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
import lightning.pytorch as pl

from scGraphLLM.GNN_modules import *
from scGraphLLM.MLP_modules import *
from scGraphLLM._globals import * ## these define the indices for the special tokens 
from scGraphLLM.transformer_modules import *

class LitScGraphLLM(pl.LightningModule):
    def __init__(self, config, pad_node=PAD_GENE_IDX):
        super().__init__()
        self.gene_embedding = torch.nn.Embedding(
            num_embeddings=config.model_config.num_genes, 
            embedding_dim=config.model_config.node_embedding_dim, 
            padding_idx=pad_node
        )
        
        self.rank_embedding = torch.nn.Embedding(
            num_embeddings=config.model_config.num_ranks, 
            embedding_dim=config.model_config.node_embedding_dim, 
            padding_idx=PAD_RANK_IDX
        )
        self.rank_prediction_head = RobertaLMHead(config.model_config.node_embedding_dim*2, config.model_config.num_ranks)
        self.optim_config = config.optim_config
        self.loss_config = config.loss_config

        
    def forward(self, batch):
        pass
    
    def _step(self, batch, batch_idx):
        learned_cell_embedding, target_gene_ids, target_expression_ids, mask_locs, edge_index_list, num_nodes_list = self(batch)
        predicted_expression_id = self.rank_prediction_head(learned_cell_embedding)
        gene_mask_locs, expression_mask_locs, rank_mask_locs = mask_locs
        L_mlm_rankonly = self.mlm_loss(predicted_expression_id, target_expression_ids, rank_mask_locs)
        loss = L_mlm_rankonly
        rank_pp = self.pseudo_perp(predicted_expression_id, target_expression_ids, rank_mask_locs)
        if type(batch) == dict:
            subset = batch["dataset_name"][0]
        else:
            subset = batch.dataset_name[0]

        self.log(f'{subset}_loss', loss, batch_size=1, add_dataloader_idx=False)
        self.log(f'{subset}_mlm_rankonly_loss', L_mlm_rankonly, batch_size=expression_mask_locs.sum(), add_dataloader_idx=False)
        self.log(f"{subset}_rank_perplexity", rank_pp, batch_size=1, add_dataloader_idx=False)
        return loss
        
    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        return loss
        

    def validation_step(self, batch, batch_idx, dataloader_idx):
        loss = self._step(batch, batch_idx)
        return loss


    def mlm_loss(self, predicted_gene_id, rank_global_gene_indices, mask_locs):
        masked_predictions = predicted_gene_id[mask_locs, :] ## because we record the location of the masked tokens, we cna retrieve just the masked tokens, and collapse the first dimension, ie mapping from n x r x G to m x G where m is the number of masked tokens
        labels = rank_global_gene_indices[mask_locs]
        loss = F.cross_entropy(masked_predictions,labels)
        return loss

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

# -------------------------
# Pre-training Model
# -------------------------

class GDTransformer(LitScGraphLLM):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.tconfig = config.transformer_config

        self.transformer_encoder = nn.ModuleList()
        for i in range(self.tconfig.num_encoder_layers):
            #TODO: refactor config to specify whether to use GK diffusion on each layer
            # Below if-statement ensures the application of GK diffusion to the desired layers
            if type(self.tconfig.use_flash_attn) == list: # Check if ONLY specified transformer layers will use GK diffusion
                use_attn = False # Default assumes no GK diffusion on this layer
                if i in self.tconfig.use_flash_attn: # If this transformer layer should use GK diffusion
                    use_attn = True
            else:
                use_attn = self.tconfig.use_flash_attn # Where self.tconfig.use_flash_attn is a boolean value  

            self.transformer_encoder.append(
                FlashTransformerEncoderLayer(
                    self.tconfig.transformer_dim.input_dim, 
                    self.tconfig.num_heads, 
                    self.tconfig.transformer_dim.feed_dim, 
                    self.tconfig.dropout, 
                    self.tconfig.activation, 
                    self.tconfig.batch_first,
                    diffusion_kernel_attn=self.tconfig.use_attn_mask,
                    use_PE=self.tconfig.use_pe,
                    use_flash_attn=self.tconfig.use_flash_attn,
                    fine_tuning=self.tconfig.fine_tuning,
                )   
            )
        
        self.use_attn_mask = self.tconfig.use_attn_mask
        self.use_PE = self.tconfig.use_pe

    def forward(self, batch):        
        orig_gene_id = batch["orig_gene_id"]
        orig_rank_indices = batch["orig_rank_indices"]
        mask = batch["rank_mask"] # SELECT MASK OPTION HERE
        pe = batch["spectral_pe"].to(torch.float32) if self.use_PE else None # shape = (batch_size, seq_len, d_emb)
        edge_index_list = batch["edge_index"]
        num_nodes_list = batch["num_nodes"]
        
        # IMPORTANT: Copy/clone the gene IDs and expression tensors, altering the originals will alter the training labels
        gene_ids = orig_gene_id.clone()
        expression = orig_rank_indices.clone()
        
        # # Mask specified gene IDs and expression values
        # gene_ids[mask] = torch.tensor(MASK_GENE_IDX, dtype=gene_ids.dtype)
        # expression[mask] = torch.tensor(MASK_RANK_IDX, dtype=gene_ids.dtype)
        
        # shape assertions for graph features
        if self.use_PE:
            assert pe.shape[1] == gene_ids.shape[1], f"Expect seqlen to be {gene_ids.shape[1]}, Got {pe.shape[1]}"
        
        mask_locs = [batch["gene_mask"], batch["rank_mask"], batch["both_mask"]]
        
        node_embedding = self.gene_embedding(gene_ids) 
        rank_embedding = self.rank_embedding(expression)
        
        combined_embedding = torch.concat([node_embedding, rank_embedding], dim=2)
        
        for encoder_layer in self.transformer_encoder:
            combined_embedding = encoder_layer(
                combined_embedding, 
                p=pe, 
                edge_index_list=edge_index_list, 
                num_nodes_list=num_nodes_list
            )
        
        # We have the learned cell embedding, no more need for MASKED gene_ids & expression
        del gene_ids
        del expression
        
        # IMPORTANT: Make sure to return the correct orig_gene_id & orig_rank_indices as these are the un-altered, unmasked training labels (do not return gene_ids & expression)
        return combined_embedding, orig_gene_id, orig_rank_indices, mask_locs, edge_index_list, num_nodes_list


# ------------------------------
# Fine-Tuning Perturbation Model
# ------------------------------
class Perturb_GDTransformer(GDTransformer):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.expression_pred_head = RobertaLMHead(
            config.model_config.node_embedding_dim * 2, 1
        )
        self.optim_config = config.optim_config
        
        # Freeze encoder weights if specified
        if config.model_config.freeze_encoder:
            for param in self.transformer_encoder.parameters():
                param.requires_grad = False

    def forward(self, batch):
        x_c = batch["ctrl_exp"]
        x_p = batch["perturb_exp"]
        r_p = batch['perturb_one_hot']
        edge_index_list = batch["edge_index"]
        num_nodes_list = batch["num_nodes"]
        pe = batch["spectral_pe"].to(torch.float32) if self.use_PE else None
        
        # FIXME - WE SPLIT THE NODE_EMBEDDING INTO GENE/RANK_EMBEDDING
        ctrl_exp_embedding = self.node_embedding(x_c)
        
        if self.tconfig.num_encoder_layers == 1:
            pert_exp_embedding = self.transformer_encoder(ctrl_exp_embedding, p=pe, 
                                                     edge_index_list=edge_index_list, 
                                                     num_nodes_list=num_nodes_list,
                                                     perturb_one_hot=r_p)
        else:
            for encoder_layer in self.transformer_encoder:
                pert_exp_embedding = encoder_layer(ctrl_exp_embedding, p=pe, 
                                               edge_index_list=edge_index_list, 
                                               num_nodes_list=num_nodes_list,
                                               perturb_one_hot=r_p)
        x_p_hat = self.expression_pred_head(pert_exp_embedding).squeeze()
        assert x_p_hat.shape == x_p.shape
        return x_c, x_p, x_p_hat
    
    def MMD(self, x_p, x_p_hat, bandwidth=1.0):
        gamma = 1.0 / (2 * bandwidth**2)
        yy = torch.cdist(x_p, x_p, p=2)**2
        xx = torch.cdist(x_p_hat, x_p_hat, p=2)**2
        xy = torch.cdist(x_p, x_p_hat, p=2)**2
        
        # embed with RBF
        k_xx = torch.exp(-gamma * xx)
        k_yy = torch.exp(-gamma * yy)
        k_xy = torch.exp(-gamma * xy)
        
        return k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()
    
    def ATE_alignment(self, x_c, x_p, x_p_hat):
        # expectation over cells
        ate_gt = torch.abs(x_p - x_c).mean()
        ate_pred = torch.abs(x_p_hat - x_c).mean()
        return 1 - torch.dot(ate_gt, ate_pred) / (torch.norm(ate_gt) * torch.norm(ate_pred))
        
    def fine_tune_loss(self, x_c, x_p, x_p_hat, bandwidth=1.0):
        mmd = self.MMD(x_p, x_p_hat, bandwidth)
        ate_align = self.ATE_alignment(x_c, x_p, x_p_hat)
        loss = mmd + ate_align
        return loss, mmd, ate_align
        
    def training_step(self, batch, batch_idx):
        x_c, x_p, x_p_hat = self(batch)
        loss, mmd, ate_align = self.fine_tune_loss(x_c, x_p, x_p_hat)
        self.log("train_loss", loss, batch_size=x_c.shape[0], add_dataloader_idx=False)
        self.log("mmd_train", mmd, batch_size=x_c.shape[0], add_dataloader_idx=False)
        self.log("ATE_align_train", ate_align, batch_size=x_c.shape[0], add_dataloader_idx=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x_c, x_p, x_p_hat = self(batch)
        loss, mmd, ate_align = self.fine_tune_loss(x_c, x_p, x_p_hat)
        self.log("val_loss", loss, batch_size=x_c.shape[0], add_dataloader_idx=False)
        self.log("mmd_val", mmd, batch_size=x_c.shape[0], add_dataloader_idx=False)
        self.log("ATE_align_val", ate_align, batch_size=x_c.shape[0], add_dataloader_idx=False)
        return loss
    
    def configure_optimizers(self):
        optim_fn = self.optim_config["optimizer"]
        optimizer = optim_fn(self.parameters(), **self.optim_config.args)
        return optimizer


