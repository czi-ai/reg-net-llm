import torch
import torch.nn as nn
import torch.nn.functional as F
from GNN_modules import *
from MLP_modules import *
import lightning.pytorch as pl
from _globals import * ## these define the indices for the special tokens 
from torch_geometric.utils import negative_sampling
from transformer_modules import *

class LitScGraphLLM(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        #self.link_prediction_head = LinkPredictHead(config.model_config.node_embedding_dim * 2, 1)
        self.node_embedding = torch.nn.Embedding(config.model_config.num_genes + config.model_config.num_ranks, 
                                                 config.model_config.node_embedding_dim, padding_idx=PAD_IDX)
        #self.gene_prediction_head = RobertaLMHead(config.model_config.node_embedding_dim*2, config.model_config.num_genes)
        self.rank_prediction_head = RobertaLMHead(config.model_config.node_embedding_dim*2, config.model_config.num_ranks)
        self.optim_config = config.optim_config
        self.loss_config = config.loss_config
        
    def forward(self, batch):
        pass
    
    def _step(self, batch, batch_idx):
        learned_cell_embedding,  target_gene_ids, target_rank_ids, mask_locs, edge_index_list, num_nodes_list = self(batch)
        #predicted_gene_id= self.gene_prediction_head(learned_cell_embedding) 
        predicted_rank_id= self.rank_prediction_head(learned_cell_embedding)
        gene_mask_locs, rank_mask_locs, both_mask_locs = mask_locs
        ## technically we could run the only/both together in a single pass, but this way we can track each one separately
        #L_mlm_geneonly = self.mlm_loss(predicted_gene_id, target_gene_ids, gene_mask_locs)
        #L_mlm_geneboth = self.mlm_loss(predicted_gene_id, target_gene_ids, both_mask_locs)

        target_rank_ids = target_rank_ids - NUM_GENES ## shift the rank indices to start from 0
        L_mlm_rankonly = self.mlm_loss(predicted_rank_id, target_rank_ids, rank_mask_locs)
        L_mlm_rankboth = self.mlm_loss(predicted_rank_id, target_rank_ids, both_mask_locs)
        #L_g = self.link_pred_loss(learned_cell_embedding, mask_locs[0], edge_index_list) # only for gene masks
        #loss = L_mlm_geneonly + L_mlm_geneboth + L_mlm_rankonly + L_mlm_rankboth + L_g
        loss = L_mlm_rankonly + L_mlm_rankboth
        #gene_pp = self.pseudo_perp(predicted_gene_id, target_gene_ids, gene_mask_locs | both_mask_locs)
        rank_pp = self.pseudo_perp(predicted_rank_id, target_rank_ids, rank_mask_locs | both_mask_locs)
        if type(batch) == dict:
            subset = batch["dataset_name"][0]
        else:
            subset = batch.dataset_name[0]

        self.log(f'{subset}_loss', loss, batch_size=1, add_dataloader_idx=False)
        #self.log(f'{subset}_mlm_geneonly_loss', L_mlm_geneonly, batch_size=gene_mask_locs.sum(), add_dataloader_idx=False)
        #self.log(f'{subset}_mlm_geneboth_loss', L_mlm_geneboth, batch_size=(gene_mask_locs|both_mask_locs).sum(), add_dataloader_idx=False) 
        self.log(f'{subset}_mlm_rankonly_loss', L_mlm_rankonly, batch_size=rank_mask_locs.sum(), add_dataloader_idx=False)
        self.log(f'{subset}_mlm_rankboth_loss', L_mlm_rankboth, batch_size=(rank_mask_locs|both_mask_locs).sum(), add_dataloader_idx=False )
        #self.log(f"{subset}_gene_perplexity", gene_pp, batch_size=1, add_dataloader_idx=False)
        self.log(f"{subset}_rank_perplexity", rank_pp, batch_size=1, add_dataloader_idx=False)
        #self.log(f"{subset}_link_pred_loss", L_g, batch_size=(rank_mask_locs|both_mask_locs).sum(), add_dataloader_idx=False)
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

    def link_pred_loss(self, node_embedding, mask_locs, edge_index_list):
        pos_out = []
        neg_out = []
        
        batch_size, num_nodes, embed_dim = node_embedding.shape
        predictor = self.link_prediction_head
        device = node_embedding.device

        for batch in range(batch_size):
            masked_nodes = torch.where(mask_locs[batch])[0]
            if masked_nodes.numel() == 0:
                continue
            masked_nodes = masked_nodes.to(device)
            edge_index = edge_index_list[batch].to(device)
            masked_nodes_bool = torch.zeros(num_nodes, dtype=torch.bool, device=device)
            masked_nodes_bool[masked_nodes] = True
            src_nodes = edge_index[0]
            dst_nodes = edge_index[1]
            # this line need to be changed to this for the "unmasked" data
            #edge_mask = (masked_nodes_bool[src_nodes]) & (masked_nodes_bool[dst_nodes]) 
            edge_mask = (~masked_nodes_bool[src_nodes]) & (~masked_nodes_bool[dst_nodes])
            pos_edge_index = edge_index[:, edge_mask]
            if pos_edge_index.size(1) == 0:
                continue

            num_neg_samples = pos_edge_index.size(1)
            neg_edge_index = negative_sampling(
                edge_index=edge_index,
                num_nodes=num_nodes,
                num_neg_samples=num_neg_samples,
                method='sparse'
            ).to(device)

            src_emb_pos = node_embedding[batch, pos_edge_index[0]]
            dst_emb_pos = node_embedding[batch, pos_edge_index[1]]

            pos_scores = predictor(src_emb_pos, dst_emb_pos) 
            pos_out.append(pos_scores)

            src_emb_neg = node_embedding[batch, neg_edge_index[0]]
            dst_emb_neg = node_embedding[batch, neg_edge_index[1]]

            neg_scores = predictor(src_emb_neg, dst_emb_neg)
            neg_out.append(neg_scores)

        if pos_out:
            pos_out = torch.cat(pos_out, dim=0)
            neg_out = torch.cat(neg_out, dim=0)

            # Loss calculation
            pos_loss = -torch.log(pos_out + 1e-10).mean()
            neg_loss = -torch.log(1 - neg_out + 1e-10).mean()
            return pos_loss + neg_loss
        else:
            # Return zero loss if there are no valid masked nodes
            return torch.tensor(0.0, device=device)

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
    def __init__(self, config):
        super().__init__(config)
        self.tconfig = config.transformer_config
        
        if self.tconfig.num_encoder_layers == 1:
            self.transformer_encoder = FlashTransformerEncoderLayer(
                    self.tconfig.transformer_dim.input_dim, 
                    self.tconfig.num_heads, 
                    self.tconfig.transformer_dim.feed_dim, 
                    self.tconfig.dropout, 
                    self.tconfig.activation, 
                    self.tconfig.batch_first,
                    use_attn_mask=self.tconfig.use_attn_mask,
                    use_PE=self.tconfig.use_pe,
                    use_flash_attn=self.tconfig.use_flash_attn,
                    fine_tuning=self.tconfig.fine_tuning,
                )
        else:
            self.transformer_encoder = nn.ModuleList()
            for i in range(self.tconfig.num_encoder_layers):
                self.transformer_encoder.append(
                    FlashTransformerEncoderLayer(
                        self.tconfig.transformer_dim.input_dim, 
                        self.tconfig.num_heads, 
                        self.tconfig.transformer_dim.feed_dim, 
                        self.tconfig.dropout, 
                        self.tconfig.activation, 
                        self.tconfig.batch_first,
                        use_attn_mask=self.tconfig.use_attn_mask,
                        use_PE=self.tconfig.use_pe,
                        use_flash_attn=self.tconfig.use_flash_attn,
                        fine_tuning=self.tconfig.fine_tuning,
                    )   
                )
        self.node_embedding = torch.nn.Embedding(config.model_config.num_genes + config.model_config.num_ranks, 
                                                 config.model_config.node_embedding_dim, padding_idx=PAD_IDX)
        #self.gene_prediction_head = RobertaLMHead(config.model_config.node_embedding_dim*2, config.model_config.num_genes)
        self.rank_prediction_head = RobertaLMHead(config.model_config.node_embedding_dim*2, config.model_config.num_ranks)
        self.optim_config = config.optim_config
        self.loss_config = config.loss_config
        self.use_attn_mask = self.tconfig.use_attn_mask
        self.use_PE = self.tconfig.use_pe

    def forward(self, batch):
        orig_gene_id = batch["orig_gene_id"]
        orig_rank_id = batch["orig_rank_indices"]
        pe = batch["spectral_pe"].to(torch.float32) if self.use_PE else None # shape = (batch_size, seq_len, d_emb)
        edge_index_list = batch["edge_index"]
        num_nodes_list = batch["num_nodes"]
        
        # shape assertions for graph features
        if self.use_PE:
            assert pe.shape[1] == orig_gene_id.shape[1], f"Expect seqlen to be {orig_gene_id.shape[1]}, Got {pe.shape[1]}"
        
        mask_locs = [batch["gene_mask"], batch["rank_mask"], batch["both_mask"]]
        
        node_embedding = self.node_embedding(orig_gene_id) 
        rank_embedding = self.node_embedding(orig_rank_id)
        
        combined_embedding = torch.concat([node_embedding, rank_embedding], dim=2)
        
        if self.tconfig.num_encoder_layers == 1:
            combined_embedding = self.transformer_encoder(combined_embedding, p=pe, 
                                                          edge_index_list=edge_index_list, 
                                                          num_nodes_list=num_nodes_list)
        else:
            for encoder_layer in self.transformer_encoder:
                combined_embedding = encoder_layer(combined_embedding, p=pe, 
                                               edge_index_list=edge_index_list, 
                                               num_nodes_list=num_nodes_list)
        
        return combined_embedding, orig_gene_id, orig_rank_id, mask_locs, edge_index_list, num_nodes_list


# ------------------------------
# Fine-Tuning Perturbation Model
# ------------------------------
class Perturb_GDTransformer(GDTransformer):
    def __init__(self, config):
        super().__init__(config)
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


