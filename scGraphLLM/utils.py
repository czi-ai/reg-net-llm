import torch
from torch_geometric.data import Data, InMemoryDataset, DataLoader, Batch
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import MessagePassing
import numpy as np

# Ground truth message passing mechnism
class WeightedAverageConv(MessagePassing):
    def __init__(self):
        super(WeightedAverageConv, self).__init__(aggr='add')

    def forward(self, x, edge_index, edge_weight):
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j

# Customized combination dataset that returns node embedding and rank embedding
class CombinationDataset(Data):
    def __init__(self, edge_index, edge_weight, node_embedding, rank_embedding):
        super(CombinationDataset, self).__init__()
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.x = node_embedding
        self.rank_embedding = rank_embedding

# Custom collecter
def collate_all(data_list):
    batch = Batch.from_data_list([Data(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_weight) for data in data_list])
    rank_embedding = torch.cat([data.rank_embedding for data in data_list], dim=0)
    return batch, rank_embedding

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
# GENE BY CELL
def node_batching(node_embedding, ranks, network, genes, batch_size=64, 
                 neigborhood_size=-1, num_hops=1):
    edge_list, edge_weights = _aracne_to_edge_list(network=network, genes=genes)
    combined_data = CombinationDataset(edge_index=edge_list, edge_weight=edge_weights, 
                                       node_embedding=node_embedding, rank_embedding=ranks)
    
    dataloader = NeighborLoader(data=combined_data, replace=False, num_neighbors=[neigborhood_size] * num_hops, 
                                input_nodes=None, subgraph_type="bidirectional", disjoint=False,
                                weight_attr="edge_weight", batch_size=batch_size, shuffle=True
                                )
    
    return dataloader, combined_data

# Some functions for simulating data for GNN testing
def simulate_data(num_classes=2, graphs_per_class=5, num_nodes_per_graph=10, 
                 num_edges_per_graph=5, node_embedding_dim=5):
      data_list = []
      num_nodes = num_nodes_per_graph
      class_noise = torch.randn(num_classes,node_embedding_dim)
      for class_id in range(num_classes):
           for _ in range(graphs_per_class):
            edge_index = torch.randint(0, num_nodes, (2, num_edges_per_graph))
            mask = edge_index[0] != edge_index[1]
            edge_index = edge_index[:, mask]
            edge_weight = torch.rand(edge_index.shape[-1])
            node_features = torch.randn(num_nodes, node_embedding_dim) + class_noise[class_id, :] * 2
            conv = WeightedAverageConv()
            for _ in range(10):
                 node_features = conv(node_features, edge_index, edge_weight)
                 node_features = torch.nn.functional.normalize(node_features, p=2, dim=-1)
            edge_weight = torch.tensor(edge_weight)
            data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_weight, y=class_id)
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
    
    
    






