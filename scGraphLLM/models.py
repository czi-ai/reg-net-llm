# For the model class
import torch
import torch.nn as nn
import torch.nn.functional as F
from GNN_modules import *
from MLP_modules import *
from transformer_modules import FlashTransformerEncoderLayer
import lightning.pytorch as pl
from data import pad_make_masks
from _globals import * ## these define the indices for the special tokens 
from torch_geometric.utils import negative_sampling

class LitScGraphLLM(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.gnn_encoder = GNN(**config.model_config.gnn_config)

        self.link_pred_decoder = LinkPredictor(in_dim=config.model_config.node_embedding_dim + config.model_config.rank_embedding_dim, 
                                               hidden_dim=config.model_config.gnn_config.hidden_dims[0])
        self.node_embedding = torch.nn.Embedding(config.model_config.node_embedding_size, config.model_config.node_embedding_dim, padding_idx=PAD_IDX)
        self.rank_embedding = torch.nn.Embedding(config.model_config.rank_embedding_size, config.model_config.rank_embedding_dim, padding_idx=PAD_IDX)

        self.mlm_encoder = FlashTransformerEncoderLayer(**config.model_config.mlm_config)
        self.prediction_head = RobertaLMHead(config.model_config.rank_embedding_dim*2, config.model_config.node_embedding_size)
        self.optim_config = config.optim_config
        self.loss_config = config.loss_config
    def forward(self, batch):
        ## adding in more explicit annotations of shapes and workflow
        ## some definitions: 
        ## G = the total number of unique genes across the whole dataset; g = the number of unique genes in a batch ; n = the number of cells in a batch; 
        ## R= the total number of unique  *expressed genes* across the whole dataset; r = the number of genes expressed in the cell with the most expressed genes, in this batch; the cells that have fewer expressed genes will have 0 padding so that the tensors are the same size
        ## G_dim = the dimension of the gene embeddings; R_dim = the dimension of the node embeddings; L_dim  = the dimension of the LLM embeddings equalto G_dim + R_dim
        ## e= the number of edges in a batch
        node_indices, edge_list,edge_weights, rank_data = batch
        ## Shapes: node_indices: gx1, ; edge_list: 2xe; edge_weights: e; 
        node_embeddings = self.node_embedding(node_indices) ## maps g x 1 to g x G_dim
        
        ## take in node embeddings with shape nodes x edim and return the same sized, updated node embeddings
        node_embeddings = self.gnn_encoder(node_embeddings, edge_list, edge_weights) ## no shape changes, just updates inputs.

        ranks, rank_global_gene_indices, rank_local_gene_indices, attn_mask = rank_data # 
        # these are now padded tensors
        # ranks is the integer rank of an expressed gene in a cell; both rank_global_gene_indices and rank_local_gene_indices map gene identity to an integer value; each row is a cell with a different number of expressed genes, and we have n rows. 
        ## the reason we have both rank_global_gene_indices and rank_local_gene_indices is because we need the rank_local_gene_indices to get the gene embeddings outputed by the gnn/initial node embeddings; we need the rank_global_gene_indices so that we know the identity of the gene for the MLM tasks

        ##TODO: This can be refactored to not use a for loop, if we pad the rank indices before hand
        rank_embeddings= self.rank_embedding(ranks) ## this individually maps each row in the NestedTensor to embeddings from the rank embedding layer, then pads then an concatenates them to a tensor of shape n x r x R_dim
        
        ##TODO: This should be able to be refactored as an indexing operation. To accomplish this,we'll need to append the <pad>, <mask> token values to the node_embedding tensors. This will letus use rank_global_gene_indices to index into the node_embeddings tensor. 
        node_embeddings = torch.cat([
            self.node_embedding(torch.tensor(PAD_IDX, device = node_embeddings.device, dtype = torch.long)).unsqueeze(0),
            self.node_embedding(torch.tensor(MASK_IDX, device = node_embeddings.device, dtype = torch.long)).unsqueeze(0),
            self.node_embedding(torch.tensor(ZERO_IDX, device = node_embeddings.device, dtype = torch.long)).unsqueeze(0),
            node_embeddings
            ], dim=0) ## this adds the <PAD> token to the node_embeddings tensor, so that we can index into it with the rank_global_gene_indices tensor. This is a n x G_dim tensor
        gene_embeddings = node_embeddings[rank_local_gene_indices] 
        ## here, the individual integers in each row of rank_local_gene_indices are mapped to the embedding stored at that position in node_embeddings. We do a similar looping operation as above, but we also return an attention mask that corresponds to the padding. This function outputs an n x r x G_dim tensor and an n x r attention mask tensor
        global_gene_indices = rank_global_gene_indices
        

        ### TODO: this masking setup will allow padded tokens to be included in the masked language modeling task. In order to fix this,We'll need to add the masking to the batch generation step in the dataloader. We also should also roughly standardize the tokens in each batch, or include a scaling factor for the loss based on the number of tokens in each batch 
        masked_full_cell_embedding, mask_locs = self.mask_tensor(torch.cat([gene_embeddings, rank_embeddings], dim=2)) ## the rank embedding and gene embedding layers are concatenated together and masked. in the masking operation, 15% of the token embedding vectors are replaced with <MASK> values, which is a concatenation of the <MASK> token from the node_embedding layer and the <MASK> token from the rank_embedding layer. This is a n x r x L_dim tensor
        learned_cell_embedding = self.mlm_encoder(masked_full_cell_embedding, attn_mask) ## this outputs an n x r x L_dim tensor
        return learned_cell_embedding, node_embedding, global_gene_indices,local_gene_to_node_index, mask_locs, edge_list

        
    def training_step(self, batch, batch_idx):
        learned_cell_embedding, node_embedding, rank_global_gene_indices,local_gene_to_node_index, mask_locs, edge_list = self(batch)
        predicted_gene_id= self.prediction_head(learned_cell_embedding) ## this maps the n x r x L_dim tensor to an n x r x G tensor
        L_mlm = self.mlm_loss(predicted_gene_id, rank_global_gene_indices, mask_locs)
        L_g = self.link_pred_loss(node_embedding, mask_locs, rank_global_gene_indices, local_gene_to_node_index, edge_list)
        loss = L_mlm + L_g
        pp = self.pseudo_perp(predicted_gene_id, rank_global_gene_indices, mask_locs)
        self.log('train_loss', loss)
        self.log('MLM_loss', L_mlm)
        self.log('Link_pred_loss', L_g)
        self.log("train_perplexity", pp, batch_size=1)
        return loss

    def validation_step(self, batch, batch_idx):
        learned_cell_embedding, rank_global_gene_indices, mask_locs = self(batch)
        predicted_gene_id= self.prediction_head(learned_cell_embedding) ## this maps the n x r x L_dim tensor to an n x r x G tensor
        loss = self.mlm_loss(predicted_gene_id, rank_global_gene_indices, mask_locs)
        pp = self.pseudo_perp(predicted_gene_id, rank_global_gene_indices, mask_locs)
        self.log('val_loss', loss, batch_size=1)
        self.log("val_perplexity", pp, batch_size=1)
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
        # for i in range(num_to_mask):
        masked_tensor[batch_indices, seq_indices, :] = mask_value
        return masked_tensor, (batch_indices, seq_indices)

    def mlm_loss(self, predicted_gene_id, rank_global_gene_indices, mask_locs):
        batch_indices, seq_indices=mask_locs
        masked_predictions = predicted_gene_id[batch_indices, seq_indices, :] ## because we record the location of the masked tokens, we cna retrieve just the masked tokens, and collapse the first dimension, ie mapping from n x r x G to m x G where m is the number of masked tokens
        labels = rank_global_gene_indices[batch_indices, seq_indices]
        loss = F.cross_entropy(masked_predictions,labels)
        return loss

    # pass in adj    
    def link_pred_loss(self, node_embedding, mask_locs, global_gene_index, local_gene_index, edge_index):
        pos_out = []
        neg_out = []
        batch_indices, seq_indices=mask_locs
        predictor = self.link_pred_decoder
        node_embeddings = node_embedding
        global_masked_nodes = global_gene_index[batch_indices, seq_indices]
        local_masked_nodes = local_gene_index[batch_indices, seq_indices]
        map_dict = dict(zip(global_masked_nodes.detach().cpu().numpy(), local_masked_nodes.detach().cpu().numpy() ))

        for global_m, local_m in zip(global_masked_nodes, local_masked_nodes):

            # Positive examples
            pos_neighbors = edge_index[1, edge_index[0] == global_m]

            # skip if no connections
            if pos_neighbors.size(0) == 0:
                continue
            
            local_ids = [map_dict[n] for n in pos_neighbors]
            pos_scores = predictor(node_embeddings[local_m, :].repeat(len(pos_neighbors), 1), node_embeddings[local_ids, :])
            pos_out.append(pos_scores)

            # Negative examples - sampled randomly
            neg_neighbors = negative_sampling(edge_index, num_nodes=node_embeddings.size(0), num_neg_samples=pos_neighbors.size(0))
            local_neg_ids = [map_dict[n] for n in neg_neighbors]
            neg_scores = predictor(node_embeddings[local_m, :].repeat(len(neg_neighbors), 1), node_embeddings[local_neg_ids, :])
            neg_out.append(neg_scores)

        pos_out = torch.cat(pos_out, dim=0)
        neg_out = torch.cat(neg_out, dim=0)
    
        # Loss calculation
        pos_loss = -torch.log(pos_out + 1e-10).mean()
        neg_loss = -torch.log(1 - neg_out + 1e-10).mean()
        return pos_loss + neg_loss

    def pseudo_perp(self, predicted_gene_id, rank_global_gene_indices, mask_locs):
        batch_indices, seq_indices=mask_locs
        masked_predictions = predicted_gene_id[batch_indices, seq_indices, :] ## because we record the location of the masked tokens, we cna retrieve just the masked tokens, and collapse the first dimension, ie mapping from n x r x G to m x G where m is the number of masked tokens
        labels = rank_global_gene_indices[batch_indices, seq_indices]

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




