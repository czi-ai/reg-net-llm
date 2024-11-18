# For the model class
import torch
from GNN_modules import *
from MLP_modules import *
from transformer_modules import *
import lightning.pytorch as pl
from _globals import * ## these define the indices for the special tokens 
from models import LitScGraphLLM

class GDTransformer(LitScGraphLLM):
    def __init__(self, config):
        super().__init__(config)
        tconfig = config.transformer_config
        self.transformer_encoder = FlashTransformerEncoderLayer(tconfig.input_dim, 
                                                                tconfig.num_heads, 
                                                                tconfig.feed_dim, 
                                                                tconfig.dropout, 
                                                                tconfig.activation, 
                                                                tconfig.batch_first,
                                                                use_attn_mask=tconfig.use_attn_mask,
                                                                use_PE=tconfig.use_pe,
                                                                use_flash_attn=tconfig.use_flash_attn)        
        self.node_embedding = torch.nn.Embedding(config.model_config.num_genes + config.model_config.num_ranks, 
                                                 config.model_config.node_embedding_dim, padding_idx=PAD_IDX)
        self.gene_prediction_head = RobertaLMHead(config.model_config.node_embedding_dim*2, config.model_config.num_genes)
        self.rank_prediction_head = RobertaLMHead(config.model_config.node_embedding_dim*2, config.model_config.num_ranks)
        self.optim_config = config.optim_config
        self.loss_config = config.loss_config
        self.use_attn_mask = tconfig.use_attn_mask
        self.use_PE = tconfig.use_pe

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
        combined_embedding = self.transformer_encoder(combined_embedding, p=pe, 
                                                      edge_index_list=edge_index_list, 
                                                      num_nodes_list=num_nodes_list)
        return combined_embedding, orig_gene_id, orig_rank_id, mask_locs, edge_index_list, num_nodes_list