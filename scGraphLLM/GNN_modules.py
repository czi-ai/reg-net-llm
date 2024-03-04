# Graph network modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GCNConv, BatchNorm, global_mean_pool, GATConv
from scGraphLLM.MLP_modules import MLPAttention
from scGraphLLM.config.model_config import GNNConfig
from torch_geometric.utils import to_dense_adj, dense_to_sparse, get_laplacian

class GAT(torch.nn.Module):
    def __init__(self, node_feature_dim, out_dim):
        super().__init__()
        self.gat_conv = GATConv(node_feature_dim, out_dim, concat=False, heads=1, add_self_loops=False)

    def forward(self, graph1, graph2):
        num_nodes = graph1.num_nodes
        combined_x = torch.cat([graph1.x, graph2.x], dim=0)
        combined_edge_index = torch.cat([graph1.edge_index, graph2.edge_index + num_nodes], dim=1)

        # Apply GAT
        combined_x, attn_weights = self.gat_conv(combined_x, combined_edge_index, return_attention_weights=True)
        adj = attn_weights.view(num_nodes, num_nodes)
        return adj
  
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
class GCN_attn(nn.Module):
   def __init__(self, input_dim, hidden_dims, conv_dim, out_dim, num_nodes, activation = nn.LeakyReLU()):
    super().__init__()
    self.input_dim = input_dim
    self.hidden_dims = hidden_dims
    self.conv_dim = conv_dim
    self.out_dim = out_dim
    self.f = activation
    self.num_nodes = num_nodes

    # Attention layers
    attention_networks = [MLPAttention(self.num_nodes, 16)]
    for i in range(1, len(self.hidden_dims)):
      attention_networks.append(GAT(self.hidden_dims[0], self.input_dim))
    attention_networks.append(MLPAttention(self.num_nodes, 16))
    self.attention_networks = nn.ModuleList(attention_networks)

     # GCN layers
    layers = [GCNConv(self.input_dim, self.hidden_dims[0], add_self_loops=False)]
    for i in range(1, len(self.hidden_dims)):
      layers.append(GCNConv(self.hidden_dims[i - 1], self.hidden_dims[i], add_self_loops=False))
    layers.append(GCNConv(self.hidden_dims[-1], self.conv_dim, add_self_loops=False))
    layers.append(nn.Linear(self.conv_dim, self.out_dim))
    self.layers = nn.ModuleList(layers)

    # Batch Norm layers
    bns = [BatchNorm(self.hidden_dims[0])]
    for i in range(1, len(self.hidden_dims)):
      bns.append(BatchNorm(self.hidden_dims[i]))
    bns.append(BatchNorm(self.conv_dim))
    self.bns = nn.ModuleList(bns)

    print("Number of Attention layers: ", len(self.attention_networks))
    print("Number of GNN layers: ", len(self.layers))
    print("Number of Batch Normalization layers: ", len(self.bns))

   def forward(self, X, A, W, batch):
     h = X
     h_conv = X
     A_orig = to_dense_adj(A, edge_attr=W, max_num_nodes=self.num_nodes).squeeze()
     A_merged = A_orig
     for i, layer in enumerate(self.layers):
       if i < len(self.layers) - 1:
        inner = h @ h.T
        A_star = F.normalize(inner, p=2) * (1-torch.eye(A_orig.shape[0])) * A_orig
        A_merged, attn_weights = self.attention_networks[i](torch.stack([A_merged, A_star], dim=1))
        A_in, W_merged = dense_to_sparse(A_merged)
        h = layer(h, A_in, W_merged)
        h = self.bns[i](h)
        h = self.f(h)
        h = F.dropout(h, p=0.5)
       else:
        h_conv = h
        h = global_mean_pool(h, batch)
        h = layer(h)
        h = torch.softmax(h, dim=-1)
     return h, h_conv, A_merged, attn_weights

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

# Graph smoothness reg
def graph_smoothness(x, adj):
  edge_index, weight = dense_to_sparse(adj)
  laplacian_id, laplacian_weight = get_laplacian(edge_index, edge_weight=weight, normalization='sym', num_nodes=x.shape[0])
  L = to_dense_adj(laplacian_id, edge_attr=laplacian_weight)
  prod = x.t() @ L @ x
  smoothness_loss = torch.trace(prod.squeeze())
  return smoothness_loss

def attention_sparsity(attn):
  return attn.norm(p = 1)


    


      
    

