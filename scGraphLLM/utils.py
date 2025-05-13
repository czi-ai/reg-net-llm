import torch
from torch_geometric.data import Data, InMemoryDataset, DataLoader, Batch
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import MessagePassing
import numpy as np
import pandas as pd 
from scGraphLLM._globals import ZERO_IDX
from scGraphLLM._globals import ZERO_IDX


# Ground truth message passing mechnism
class WeightedAverageConv(MessagePassing):
    def __init__(self):
        super(WeightedAverageConv, self).__init__(aggr='add')

    def forward(self, x, edge_index, edge_weight):
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j
# Custom collecter
def collate_all(data_list):
    batch = Batch.from_data_list([Data(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_weight) for data in data_list])
    expression_embedding = torch.cat([data.expression_embedding for data in data_list], dim=0)
    return batch, expression_embedding
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

def update_mconfig_from_dict(mconfig, sweep_dict, ignore_keys={}):
    sweep_keys = [k for k in sweep_dict.keys() if k not in ignore_keys ]
    for skey in sweep_keys:
        key_path = skey.split("-")
        c_dict = mconfig
        for _key in key_path[:-1]:
            c_dict = c_dict[_key]
        ## preserve original datatype of parameter
        orig_dtype = type(c_dict[key_path[-1]])
        c_dict[key_path[-1]]=orig_dtype(sweep_dict[skey])
    return mconfig

class CombinedDataset(InMemoryDataset):
    def __init__(self, data_list):
        super(CombinedDataset, self).__init__('.')
        self.data, self.slices = self.collate(data_list)

    def _download(self):
        pass

    def _process(self):
        pass

# mask edges from graph
def random_edge_mask(edge_index, mask_ratio=0.15):
    device = edge_index.device
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    E = edge_index.size(1)
    
    undirected_pairs = {}
    for i in range(E):
        u, v = src[i], dst[i]
        mn, mx = (u, v) if u <= v else (v, u)
        if (mn, mx) not in undirected_pairs:
            undirected_pairs[(mn, mx)] = []
        undirected_pairs[(mn, mx)].append(i)
    
    unique_pairs = list(undirected_pairs.keys())  
    num_unique = len(unique_pairs)
    num_mask = int(mask_ratio * num_unique)
    
    perm = torch.randperm(num_unique, device=device)
    masked_pairs_indices = perm[:num_mask]
    masked_pairs = [unique_pairs[i.item()] for i in masked_pairs_indices]
    masked_indices = []
    for pair in masked_pairs:
        masked_indices.extend(undirected_pairs[pair]) 
    masked_indices = torch.tensor(masked_indices, device=device, dtype=torch.long)
    
    masked_edge_index = edge_index[:, masked_indices]

    return masked_edge_index
    
    






