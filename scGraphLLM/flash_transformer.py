# For the model class
import torch
import lightning.pytorch as pl
from scGraphLLM.GNN_modules import *
from scGraphLLM.MLP_modules import *
from scGraphLLM.transformer_modules import *
from scGraphLLM._globals import * ## these define the indices for the special tokens 
from scGraphLLM.models import LitScGraphLLM


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
                    use_flash_attn=self.tconfig.use_flash_attn
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
                        use_flash_attn=self.tconfig.use_flash_attn
                    )   
                )
        self.node_embedding = torch.nn.Embedding(config.model_config.num_genes + config.model_config.num_ranks, 
                                                 config.model_config.node_embedding_dim, padding_idx=PAD_IDX)
        self.gene_prediction_head = RobertaLMHead(config.model_config.node_embedding_dim*2, config.model_config.num_genes)
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