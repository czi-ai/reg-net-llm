# For the model class
import torch
import torch.nn as nn
import torch.nn.functional as F
from GNN_modules import *
from MLP_modules import *
import lightning.pytorch as pl
from _globals import * ## these define the indices for the special tokens 
from torch_geometric.utils import negative_sampling

class LitScGraphLLM(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.gnn_encoder = GNN(**config.model_config.gnn_config)

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
        node_indices, edge_list,edge_weights = batch.x, batch.edge_index, batch.edge_weight
        mask_locs = [batch.gene_mask, batch.rank_mask, batch.both_mask]
        node_indices = node_indices.type(torch.long)
        ## Shapes: node_indices: n x g x 2, ; edge_list: 2xe; edge_weights: e; 
        node_embeddings = self.node_embedding(node_indices).flatten(1) ## maps n x g x 
        
        ## take in node embeddings with shape nodes x edim and return the same sized, updated node embeddings
        node_embeddings = self.gnn_encoder(node_embeddings, edge_list, edge_weights) ## no shape changes, just updates inputs.

        gene_ids = batch.orig_gene_id
        rank_ids = batch.orig_rank_indices
        return node_embeddings,edge_list, gene_ids, rank_ids, mask_locs
    
    def _step(self, batch, batch_idx):
        learned_cell_embedding, edge_list,  target_gene_ids, target_rank_ids, mask_locs = self(batch)
        predicted_gene_id= self.gene_prediction_head(learned_cell_embedding) 
        predicted_rank_id= self.rank_prediction_head(learned_cell_embedding)
        gene_mask_locs, rank_mask_locs, both_mask_locs = mask_locs
        ## technically we could run the only/both together in a single pass, but this way we can track each one separately
        L_mlm_geneonly = self.mlm_loss(predicted_gene_id, target_gene_ids, gene_mask_locs)
        L_mlm_geneboth = self.mlm_loss(predicted_gene_id, target_gene_ids, both_mask_locs)

        target_rank_ids = target_rank_ids - NUM_GENES ## shift the rank indices to start from 0
        L_mlm_rankonly = self.mlm_loss(predicted_rank_id, target_rank_ids, rank_mask_locs)
        L_mlm_rankboth = self.mlm_loss(predicted_rank_id, target_rank_ids, both_mask_locs)
        #L_g = self.link_pred_loss(learned_cell_embedding, mask_locs, target_gene_ids, target_rank_ids, edge_list)
        loss = L_mlm_geneonly + L_mlm_geneboth + L_mlm_rankonly + L_mlm_rankboth #+ L_g
        gene_pp = self.pseudo_perp(predicted_gene_id, target_gene_ids, gene_mask_locs | both_mask_locs)
        rank_pp = self.pseudo_perp(predicted_rank_id, target_rank_ids, rank_mask_locs | both_mask_locs)
        subset = batch.dataset_name[0]
        self.log(f'{subset}_loss', loss)
        self.log(f'{subset}_mlm_geneonly_loss', L_mlm_geneonly, batch_size=gene_mask_locs.sum())
        self.log(f'{subset}_mlm_geneboth_loss', L_mlm_geneboth, batch_size=(gene_mask_locs|both_mask_locs).sum() )
        self.log(f'{subset}_mlm_rankonly_loss', L_mlm_rankonly, batch_size=rank_mask_locs.sum())
        self.log(f'{subset}_mlm_rankboth_loss', L_mlm_rankboth, batch_size=(rank_mask_locs|both_mask_locs).sum() )
        #self.log('train_link_pred_loss', L_g)
        self.log(f"{subset}_gene_perplexity", gene_pp, batch_size=1)
        self.log(f"{subset}_rank_perplexity", rank_pp, batch_size=1)
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

    # pass in adj    
    def link_pred_loss(self, node_embedding, mask_locs, global_gene_index, local_gene_index, edge_index):
        pos_out = []
        neg_out = []
        batch_indices, seq_indices=mask_locs
        predictor = self.link_pred_decoder
        node_embeddings = node_embedding
        global_masked_nodes = global_gene_index[batch_indices, seq_indices]
        #local_masked_nodes = local_gene_index[batch_indices, seq_indices]
        #map_dict = dict(zip(global_masked_nodes.detach().cpu().numpy(), local_masked_nodes.detach().cpu().numpy()))
        #for global_m, local_m in zip(global_masked_nodes, local_masked_nodes):
        for global_m in global_masked_nodes:

            # Positive examples
            pos_neighbors = edge_index[1, edge_index[0] == global_m]
            #pos_neighbors = torch.tensor(list(set(pos_neighbors).intersection(list(global_masked_nodes.detach().cpu().numpy()))))
            # skip if no connections
            if len(pos_neighbors) == 0:
                continue

            #local_ids = [map_dict[n.item()] for n in pos_neighbors]
            pos_scores = predictor(node_embeddings[global_m, :].repeat(len(pos_neighbors), 1), node_embeddings[pos_neighbors, :])
            pos_out.append(pos_scores)

            # Negative examples - sampled randomly
            neg_neighbors = negative_sampling(edge_index, num_nodes=node_embeddings.size(0), num_neg_samples=pos_neighbors.size(0)).view(-1)
            #local_neg_ids = [map_dict[n] for n in neg_neighbors]
            neg_scores = predictor(node_embeddings[global_m, :].repeat(len(neg_neighbors), 1), node_embeddings[neg_neighbors, :])
            neg_out.append(neg_scores)

        pos_out = torch.cat(pos_out, dim=0)
        neg_out = torch.cat(neg_out, dim=0)
    
        # Loss calculation
        pos_loss = -torch.log(pos_out + 1e-10).mean()
        neg_loss = -torch.log(1 - neg_out + 1e-10).mean()
        return pos_loss + neg_loss

    def alignment_loss():
        pass

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




