# For functions like loss, hyperparam containers, pre/post processing

import torch
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
import torch_geometric.nn as nn
from torch_geometric.loader import NeighborLoader, NeighborSampler
import pandas as pd
import scanpy as sc
import numpy as np


# ARACNe3 to torch geometric edge list
def _aracne_to_edge_list(network, genes):
    edges = network[['regulator.values', 'target.values', 'mi.values']]
    gene_to_node_index = {string: index for index, string in enumerate(genes)}
    edges['regulator.values'] = edges['regulator.values'].map(gene_to_node_index)
    edges['target.values'] = edges['target.values'].map(gene_to_node_index)
    edge_list = torch.tensor(np.array(edges[['regulator.values', 'target.values']])).T
    edge_weights = torch.tensor(np.array(edges['mi.values']))
    return edge_list, edge_weights

# Sampling n_batch equally sized neighbourhoods per layer of GNN without replacement, weighed by MI
def node_batching(adata, network, genes, batch_size=64, 
                 neigborhood_size=20, num_hops=2, with_label=False):
    edge_list, edge_weights = _aracne_to_edge_list(network=network, genes=genes)
    node_embedding = torch.tensor(adata.X.T.todense())
    graph_dataset = Data(x = node_embedding, edge_index=edge_list, edge_attr=edge_weights)
    
    if with_label:
        node_labels = torch.tensor(adata.var['labels'])
        graph_dataset = Data(x = node_embedding, edge_index=edge_list, y=node_labels)
    
    dataloader = NeighborLoader(data=graph_dataset, replace=False, num_neighbors=[neigborhood_size] * num_hops, 
                                subgraph_type="bidirectional", disjoint=False,
                                weight_attr="edge_attr", batch_size=batch_size
                                )
    
    return dataloader
    
    
    






