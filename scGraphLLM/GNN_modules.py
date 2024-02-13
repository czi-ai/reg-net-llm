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
    layers.append(GCNConv(GCNConv(self.hidden_dims[-1], self.conv_dim)))
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


# Hierarchical GNN for processing modular structured graphs
class hierGNN(nn.Module):
  def __init__(self, pooling_ratio, input_dim, hidden_dims, conv_dim, out_dim, 
               activation=nn.ReLU, return_attn = True):
    super().__init__()
    self.pooling_ratio = pooling_ratio
    self.input_dim = input_dim
    self.hidden_dims = hidden_dims
    self.conv_dim = conv_dim
    self.out_dim = out_dim
    self.f = activation
    self.return_attn = return_attn

    self.conv_block = baseGCN(self.input_dim, self.hidden_dims, 
                              self.conv_dim, self.out_dim, self.f)
    self.graph_pooling_block = SAGPooling(self.out_dim, ratio=self.pooling_ratio, nonlinearity='tanh')
    self.edge_conv_block = edgeConv(self.out_dim, self.out_dim)


  def forward(self, X, A, W):
    H = X
    H = self.conv_block(X, A, W)
    H, A_hat, W_hat, _, _, attn = self.graph_pooling_block(H)
    H = self.edge_conv_block(H, A_hat, W_hat)
    return H, attn
 
# Siamese GNN with contrastive learning
class contrastiveGNN(nn.Module):
    def __init__(self, config):
      super(contrastiveGNN, self).__init__()
      self.config = config
      self.GNN = baseGCN(config.input_dim, config.hidden_dims, config.conv_dim, config.out_dim)
      self.siamese_net = nn.Sequential(
        nn.Linear(config.num_graphs * config.out_dim, config.latent_dim),
        nn.LeakyReLU(),
        nn.Linear(config.latent_dim, config.out_dim)
      )
      self.attn_net = MLPAttention(config.out_dim, hidden_size=64)
      
    def forward(self, xs):
      # Obtain embeddings from each graph
      embeddings = []
      for data in xs:
        emb = self.GNN(data.x, data.edge_index, data.edge_atr)
        embeddings.append(emb)
      
      # Get all embeddings
      all_embeddings = torch.cat(embeddings, dim=1)
      siamese_embeddings = self.siamese_net(all_embeddings)
      outputs = torch.chunk(siamese_embeddings, self.config.num_graphs, dim=-1)

      # Attention network
      attn_weighted_embedding, attn_W = self.attn_net(torch.stack(outputs, dim=1))

      return outputs, attn_weighted_embedding, attn_W




    


      
    

