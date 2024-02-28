import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import NeighborLoader
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
                                weight_attr="edge_attr", batch_size=batch_size, shuffle=True
                                )
    
    return dataloader

# Some functions for simulating data for GNN testing
def simulate_data(num_classes=2, graphs_per_class=5, num_nodes_per_graph=10, 
                 num_edges_per_graph=5, node_embedding_dim=5):
      data_list = []
      num_nodes = num_nodes_per_graph
      for class_id in range(num_classes):
           for _ in range(graphs_per_class):
            edge_index = torch.randint(0, num_nodes, (2, num_edges_per_graph))
            mask = edge_index[0] != edge_index[1]
            edge_index = edge_index[:, mask]
            edge_weights = torch.rand(num_edges_per_graph,)
            node_features = torch.randn(num_nodes, node_embedding_dim) + class_id * 10
            data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_weights, y=class_id)
            data_list.append(data)
      return data_list

class CombinedDataset(InMemoryDataset):
    def __init__(self, data_list):
        super(CombinedDataset, self).__init__('.')
        self.data, self.slices = self.collate(data_list)

    def _download(self):
        pass

    def _process(self):
        pass
    
    
    






