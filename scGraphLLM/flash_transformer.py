from models import LitScGraphLLM
from GNN_modules import *
from MLP_modules import *
from _globals import * ## these define the indices for the special tokens 
from transformer_modules import FlashTransformerEncoderLayer

class FlashTRAN(LitScGraphLLM):
    def __init__(self, config):
        super().__init__(config)
        tconfig = config.transformer_config

        self.transformer_encoder = FlashTransformerEncoderLayer(tconfig.input_dim, tconfig.num_heads, tconfig.feed_dim, tconfig.dropout, tconfig.activation, tconfig.batch_first)

        # YOU LEFT OFF HERE - CONFIG VARIABLES MIGHT BE OFF
        # YOU PROBABLY HAVE TO REPLACE "model_config" BELOW WITH "gnn_config"!!!!
        self.link_prediction_head = LinkPredictHead(in_dim=config.model_config.node_embedding_dim *2 ,
                                               hidden_dim=config.model_config.gnn_config.hidden_dims[0])
        self.node_embedding = torch.nn.Embedding(config.model_config.num_genes + config.model_config.num_ranks, 
                                                 config.model_config.node_embedding_dim, padding_idx=PAD_IDX)
        self.gene_prediction_head = RobertaLMHead(config.model_config.node_embedding_dim*2, config.model_config.num_genes)
        self.rank_prediction_head = RobertaLMHead(config.model_config.node_embedding_dim*2, config.model_config.num_ranks)
        self.optim_config = config.optim_config
        self.loss_config = config.loss_config

    def forward(self, batch):
        ## adding in more explicit annotations of shapes and workflow
        ## some definitions: 
        ## n = the number of cells in a batch; 
        ## G = the total number of unique genes across the whole dataset; g = the number of unique genes in a batch ; g_dim = the dimension of the gene embeddings;
        ## R= the total number of unique  *expressed genes* across the whole dataset; r_dim = the dimension of the rank embeddings; 
        ## L_dim = the dimension of the learned cell embeddings(r_dim + g_dim);
        ## e= the number of edges in a batch

        orig_gene_id = batch["orig_gene_id"]
        orig_rank_id = batch["orig_rank_indices"]
        mask_locs = [batch["gene_mask"], batch["rank_mask"], batch["both_mask"]]

        node_indices = torch.stack([orig_gene_id, orig_rank_id]).permute(1, 2, 0)
        node_indices = node_indices.type(torch.long)

        ## Shapes: node_indices: n x g x 2
        node_embeddings = self.node_embedding(node_indices).flatten(2) ## maps n x g x embed_dim
        
        ## take in node embeddings with shape nodes x edim and return the same sized, updated node embeddings
        node_embeddings = self.transformer_encoder(node_embeddings) ## no shape changes, just updates inputs.

        return node_embeddings, orig_gene_id, orig_rank_id, mask_locs


class GraphTransformer(LitScGraphLLM):
    def __init__(self, config):
        super().__init__(config)
        tconfig = config.graph_transformer_config
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
        ## adding in more explicit annotations of shapes and workflow
        ## some definitions: 
        ## n = the number of cells in a batch; 
        ## G = the total number of unique genes across the whole dataset; g = the number of unique genes in a batch ; g_dim = the dimension of the gene embeddings;
        ## R= the total number of unique  *expressed genes* across the whole dataset; r_dim = the dimension of the rank embeddings; 
        ## L_dim = the dimension of the learned cell embeddings(r_dim + g_dim);
        ## e= the number of edges in a batch

        orig_gene_id = batch.orig_gene_id
        orig_rank_id = batch.orig_rank_indices
        pe = batch.pe if self.use_PE else None
        K = batch.diffusion_kernel if self.use_attn_mask else None
        L = batch.laplacian if self.use_attn_mask else None
        
        # shape assertions for graph features
        if self.use_PE:
            assert pe.shape[0] == batch.x.shape[0], f"Expect number of token to be {batch.x.shape[0]}, Got {pe.shape[0]}"
        
        # stack different masks for different heads
        if self.use_attn_mask:
            assert K.shape == (batch.x.shape[0], batch.x.shape[0]), f"Expect shape of K to be {(batch.x.shape[0], batch.x.shape[0])}, Got {K.shape}"
            assert L.shape == (batch.x.shape[0], batch.x.shape[0]), f"Expect shape of L to be {(batch.x.shape[0], batch.x.shape[0])}, Got {L.shape}"
            num_heads = self.transformer_encoder.num_heads
            half_heads = num_heads // 2
            attn_mask = torch.zeros((1, num_heads, K.shape[0], K.shape[1]), device=K.device)
            attn_mask[:, :half_heads, :, :] = K.unsqueeze(0).expand(half_heads, -1, -1)
            attn_mask[:, half_heads:, :, :] = L.unsqueeze(0).expand(half_heads, -1, -1)
            
        mask_locs = [batch.gene_mask, batch.rank_mask, batch.both_mask]
        node_indices = torch.stack([orig_gene_id, orig_rank_id]).permute(1, 2, 0)
        node_indices = node_indices.type(torch.long)

        ## Shapes: node_indices: n x g x 2
        node_embeddings = self.node_embedding(node_indices).flatten(2) ## maps n x g x embed_dim     
        ## take in node embeddings with shape nodes x edim and return the same sized, updated node embeddings
        node_embeddings = self.transformer_encoder(node_embeddings, p=pe, attn_mask=attn_mask)

        return node_embeddings, orig_gene_id, orig_rank_id, mask_locs