# Graph network modules
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch_geometric
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GCNConv, SAGPooling, BatchNorm
from scGraphLLM.MLP_modules import MLPAttention
from scGraphLLM.config.model_config import GNNConfig
from torch_geometric.utils import to_dense_adj, to_edge_index

# Edge convolution layer. Aggregate mean information along edges within a neighbourhood
class edgeConv(MessagePassing):
  def __init__(self, in_channels, out_channels):
    super().__init__(aggr='mean')
    self.linear_net = nn.Sequential(
      nn.Linear(2 * in_channels, out_channels),
      nn.ReLU(),
      nn.Linear(out_channels, out_channels)
    )
  
  def forward(self, x, A):
    return self.propagate(A, x=x)
  
  def message(self, x_i, x_j):
    return self.linear_net(torch.cat([x_i, x_j - x_i], dim=1)) # Node feature and relative node features wrt edge
  
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
    
    # Last layer
    layers.append(GCNConv(self.hidden_dims[-1], self.conv_dim))
    layers.append(nn.Linear(self.conv_dim, self.out_dim))
    bns.append(BatchNorm(self.conv_dim))
    self.layers = nn.ModuleList(layers)
    self.bns = nn.ModuleList(bns)

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

# structural learning GNN through metric learning and attention
class GCN_attn(baseGCN):
   def __init__(self, num_nodes):
    super().__init__()
    self.num_nodes = num_nodes
    attention_networks = [MLPAttention(self.num_nodes, 64)]
    
    for i in range(1, len(self.hidden_dims)):
      attention_networks.append(MLPAttention(self.num_nodes, 64))
    
    attention_networks.append(MLPAttention(self.num_nodes, 64))
    self.attention_networks = nn.ModuleList(attention_networks)

   def forward(self, X, A, W):
     h = X
     final_A = None
     for i, layer in enumerate(self.layers):
       if i < len(self.layers) - 1:
         A_orig = to_dense_adj(A, edge_attr=W)
         A_star = torch.sigmoid(h @ h.T)
         A_merged, _ = self.attention_networks[i](torch.stack([A_orig, A_star], dim=1))
         final_A = A_merged
         A_merged = to_edge_index(A_merged)
         h = layer(h, A_merged, W)
         h = self.bns[i](h)
         h = self.f(h)
       else:
         h = layer(h)
     return h, final_A

# Siamese GNN with contrastive learning
class contrastiveGNN(nn.Module):
    def __init__(self, config=GNNConfig()):
      super(contrastiveGNN, self).__init__()
      self.config = config
      self.GNN = GCN_attn(config.input_dim, config.hidden_dims, config.conv_dim, config.out_dim, config.num_nodes)
      self.siamese_net = nn.Sequential(
        nn.Linear(config.num_graphs * config.out_dim, config.latent_dim),
        nn.LeakyReLU(),
        nn.Linear(config.latent_dim, config.out_dim)
      )
      self.cross_graph_attention = MLPAttention(config.out_dim, hidden_size=64)

    def forward_one_branch(self, x, A, W):
      return self.GNN(x, A, W)
      
    def forward(self, xs):
      # Obtain embeddings from each graph
      embeddings = []
      graphs = []
      for data in xs:
        emb, graph = self.forward_one_branch(data.x, data.edge_index, data.edge_attr)
        embeddings.append(emb)
        graphs.append(graph)
      
      # Get siamese embeddings
      all_embeddings = torch.cat(embeddings, dim=1)
      siamese_embeddings = self.siamese_net(all_embeddings)
      outputs = torch.chunk(siamese_embeddings, self.config.num_graphs, dim=-1)

      # Attention network
      attn_weighted_embedding, attn_W = self.cross_graph_attention(torch.stack(outputs, dim=1))

      return outputs, attn_weighted_embedding, attn_W, graphs




    


      
    

