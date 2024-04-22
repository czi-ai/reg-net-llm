# Graph network modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GCNConv, BatchNorm, global_mean_pool, GATConv
from MLP_modules import MLPAttention, MLPCrossAttention
from torch_geometric.utils import to_dense_adj, dense_to_sparse, get_laplacian
  
# Vanilla graph conovolution network
class baseGCN(nn.Module):
  def __init__(self, input_dim, hidden_dims, conv_dim, out_dim, activation = nn.LeakyReLU()):
    super().__init__()
    self.input_dim = input_dim
    self.hidden_dims = hidden_dims
    self.conv_dim = conv_dim
    self.out_dim = out_dim
    self.f = activation

    # First layer
    layers = [GCNConv(self.input_dim, self.hidden_dims[0])]
    bns = [BatchNorm(self.hidden_dims[0])]

    # Hidden layers
    for i in range(1, len(self.hidden_dims)):
      layers.append(GCNConv(self.hidden_dims[i - 1], self.hidden_dims[i]))
      bns.append(self.hidden_dims[i])
    
    # Last conv and bn layers
    layers.append(GCNConv(self.hidden_dims[-1], self.conv_dim))
    bns.append(BatchNorm(self.conv_dim))

    self.layers = nn.ModuleList(layers)
    self.bns = nn.ModuleList(bns)

    # MLP head
    self.output_layer = nn.Linear(self.conv_dim, self.out_dim)

  def forward(self, X, A, W):
    h = X
    for i, layer in enumerate(self.layers):
      if i < len(self.layers) - 1:
        h = layer(h, A, W)
        h = self.bns[i](h)
        h = self.f(h)
      else:
        h = layer(h)
    return h

class GNN(nn.Module):
  def __init__(self, input_dim, hidden_dims, conv_dim, out_dim, num_heads, 
                activation=nn.LeakyReLU(), dropout_rate=0.5):
    super().__init__()
    self.input_dim = input_dim
    self.hidden_dims = hidden_dims
    self.conv_dim = conv_dim
    self.num_heads = num_heads
    self.out_dim = out_dim
    self.f = activation
    self.dropout = nn.Dropout(dropout_rate)

    # GAT layers
    self.layers = nn.ModuleList([GATConv(input_dim, hidden_dims[0], heads=num_heads[0], concat=True, dropout=dropout_rate)])
    for i in range(1, len(hidden_dims)):
      self.layers.append(GATConv(hidden_dims[i - 1] * num_heads[i - 1], hidden_dims[i], heads=num_heads[i], concat=True, dropout=dropout_rate))
    self.layers.append(GATConv(hidden_dims[-1] * num_heads[-1], conv_dim, heads=num_heads[-1], concat=False, dropout=dropout_rate))

    # Batch Norm layers
    self.bns = nn.ModuleList([BatchNorm(hidden_dims[0] * num_heads[0])])
    for i in range(1, len(hidden_dims)):
      self.bns.append(BatchNorm(hidden_dims[i] * num_heads[i]))
    self.bns.append(BatchNorm(conv_dim))

    # Final linear layer
    self.output_layer = nn.Linear(conv_dim, out_dim)

    # Dimension matching for residual connections
    self.dim_match = nn.ModuleList()
    initial_out_dim = hidden_dims[0] * num_heads[0]
    self.dim_match.append(nn.Linear(input_dim, initial_out_dim) if input_dim != initial_out_dim else nn.Identity())
    for i in range(1, len(hidden_dims)):
      prev_dim = hidden_dims[i - 1] * num_heads[i - 1]
      curr_dim = hidden_dims[i] * num_heads[i]
      self.dim_match.append(nn.Linear(prev_dim, curr_dim) if prev_dim != curr_dim else nn.Identity())

  def forward(self, X, edge_index, edge_weight):
    h = X
    for i, (layer, match) in enumerate(zip(self.layers, self.dim_match)):
      h_prev = match(h)
      h = layer(h, edge_index, edge_weight)
      h = self.bns[i](h)
      h = self.f(h)
      h = self.dropout(h)
      # residual connection
      if i > 0: 
        h += h_prev
    h = self.output_layer(h)
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


    


      
    

