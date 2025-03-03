# Graph network modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GATConv
# from scGraphLLM.MLP_modules import AttentionMLP
from torch_geometric.utils import to_dense_adj, dense_to_sparse, get_laplacian, add_self_loops
import networkx as nx
import random


class GNN(nn.Module):
  def __init__(self, input_dim, hidden_dims, conv_dim, out_dim, num_heads, num_heads_attn_network=2,
                activation=nn.GELU(), dropout_rate=0.5):
    super().__init__()
    self.input_dim = input_dim
    self.hidden_dims = hidden_dims
    self.conv_dim = conv_dim
    self.num_heads = num_heads
    self.num_heads_attn_network = num_heads_attn_network
    self.out_dim = out_dim
    self.f = activation
    self.dropout = nn.Dropout(dropout_rate)

    # GAT layers
    self.layers = nn.ModuleList([GATConv(input_dim, hidden_dims[0], heads=num_heads[0], 
                                         concat=False, dropout=dropout_rate)]) # Because of PE 
    for i in range(1, len(hidden_dims)):
      self.layers.append(GATConv(hidden_dims[i - 1], hidden_dims[i], heads=num_heads[i], 
                                 concat=False, dropout=dropout_rate))
    self.layers.append(GATConv(hidden_dims[-1], conv_dim, heads=num_heads[-1], 
                               concat=False, dropout=dropout_rate))

    # Layer Norm layers
    self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dims[0])])
    for i in range(1, len(hidden_dims)):
      self.layer_norms.append(nn.LayerNorm(hidden_dims[i]))
    self.layer_norms.append(nn.LayerNorm(conv_dim))

    # Final linear layer
    self.output_layer = nn.Linear(conv_dim, out_dim)
    
    # attention network
    #self.attn_network = AttentionMLP(in_size=conv_dim, hidden_size=64, num_heads=self.num_heads_attn_network)

  def forward(self, X, edge_index, edge_weight):
    # positional embedding based on shortest path distance
    # pos_embedding = self.PE(edge_index, X.shape[0])
    # X = torch.cat([X, pos_embedding], dim=1)

    # Virutal node connected to every other nodes
    virtual_node = torch.mean(X, dim=0, keepdim=True)
    X = torch.cat([X, virtual_node], dim=0)
    virtual_node_idx = X.size(0) - 1
    virtual_edges = torch.stack([torch.arange(virtual_node_idx), 
                                 torch.full((virtual_node_idx,), virtual_node_idx)], dim=0).to(X.device)
    edge_index = torch.cat([edge_index, virtual_edges, virtual_edges.flip(0)], dim=1)

    h = X
    for i, layer in enumerate(self.layers):
      h = layer(h, edge_index)
      h = self.layer_norms[i](h)
      h = self.f(h)
      h = self.dropout(h)

    virtual_node_embedding = h[-1].unsqueeze(0).repeat(h.shape[0]-1, 1) 
    node_embedding = h[:-1]
    # virtual_node_embedding is broadcasted to (G, D) from (1, D)
    assert virtual_node_embedding.shape == node_embedding.shape
    combined_embedding = torch.stack([node_embedding, virtual_node_embedding], dim=1) # shape = (G, 2, D)
    #final_node_embedding, _ = self.attn_network(combined_embedding)
    h = self.output_layer(final_node_embedding)
    return h


# Graph smoothness reg
def graph_smoothness(x, adj):
  edge_index, weight = dense_to_sparse(adj)
  laplacian_id, laplacian_weight = get_laplacian(edge_index, edge_weight=weight, 
                                                 normalization='sym', num_nodes=x.shape[0])
  L = to_dense_adj(laplacian_id, edge_attr=laplacian_weight)
  prod = x.t() @ L @ x
  smoothness_loss = torch.trace(prod.squeeze())
  return smoothness_loss

def attention_sparsity(attn):
  return attn.norm(p = 1)

# random walk to get sequence of nodes from graph
def generate_node_sequences(G, num_walks, walk_length):
    walks = []
    nodes = list(G.nodes())
    for _ in range(num_walks):
        random.shuffle(nodes)
        for node in nodes:
            walk = [node]
            while len(walk) < walk_length:
                cur = walk[-1]
                neighbors = list(G.neighbors(cur))
                if len(neighbors) > 0:
                    walk.append(random.choice(neighbors))
                else:
                    break
            walks.append(walk)
    return walks
