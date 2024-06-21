# Graph network modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GCNConv, BatchNorm, global_mean_pool, GATConv
from MLP_modules import MLPAttention, MLPCrossAttention, CrossAttentionFuse
from torch_geometric.utils import to_dense_adj, dense_to_sparse, get_laplacian, add_self_loops
from torch_geometric.nn.inits import glorot
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import degree
import networkx as nx

class GRNAttention(GATConv):
    def __init__(self, in_channels, out_channels, heads=1, concat=True, 
                 negative_slope=0.2, dropout=0, add_self_loops=True, bias=True):
        super(GRNAttention, self).__init__(in_channels, out_channels, heads=heads, 
                                           concat=concat, negative_slope=negative_slope, 
                                           dropout=dropout, add_self_loops=add_self_loops, bias=bias)
        
        # attention weights R-T edges
        self.att_r_t = torch.nn.Parameter(torch.Tensor(1, heads, out_channels))

        # projection heads for R-R edges
        self.query_r_r = torch.nn.Linear(out_channels, out_channels)
        self.key_r_r = torch.nn.Linear(out_channels, out_channels)
        self.value_r_r = torch.nn.Linear(out_channels, out_channels)

        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.att_r_r)
        torch.nn.init.xavier_uniform_(self.att_r_t)
        torch.nn.init.xavier_uniform_(self.query_r_r.weight)
        torch.nn.init.xavier_uniform_(self.key_r_r.weight)
        torch.nn.init.xavier_uniform_(self.value_r_r.weight)     
        if self.query_r_r.bias is not None:
            torch.nn.init.zeros_(self.query_r_r.bias)
        if self.key_r_r.bias is not None:
            torch.nn.init.zeros_(self.key_r_r.bias)
        if self.value_r_r.bias is not None:
            torch.nn.init.zeros_(self.value_r_r.bias)

    def forward(self, x, edge_index, edge_type):
        if self.add_self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        return self.propagate(edge_index, x=x, edge_type=edge_type)
    
    def propagate(self, edge_index, x, edge_type, **kwargs):
        size = (x.size(0), x.size(0))
        coll_dict = self.__collect__(self.__user_args__, edge_index, size, x=x, 
                                     edge_type=edge_type, **kwargs)
        msg_kwargs = self.inspector.distribute('message', coll_dict)
        out = self.message(**msg_kwargs)

        aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
        out = self.aggregate(out, **aggr_kwargs)

        update_kwargs = self.inspector.distribute('update', coll_dict)
        return self.update(out, **update_kwargs)

    # edge type needs to be binary vector of shape E
    # x_i, x_j has shape (E, out_channels)
    def message(self, x_i, x_j, edge_type):
        # Mask for different edge types
        mask_rr = (edge_type == 0)  # R->R
        mask_rt = (edge_type == 1)  # R->T
        
        # Create an empty tensor for messages, dim (E, out_channels)
        msg = torch.zeros_like(x_j)
        
        if mask_rt.sum() > 0:
            # Multiplicative attention for R->T
            q = self.query_r_r(x_i[mask_rr])
            k = self.key_r_r(x_j[mask_rr])
            v = self.value_r_r(x_j[mask_rr])
            alpha_rr = (q * k).sum(dim=-1) / (self.out_channels ** 0.5)
            alpha_rr = F.softmax(alpha_rr, dim=1)
            msg[mask_rr] = v * alpha_rr.view(-1, self.heads, 1)

        if mask_rr.sum() > 0:
            # Additive attention for R->R
            alpha_rt = (x_i[mask_rt] * self.att_r_t).sum(dim=-1) + (x_j[mask_rt] * self.att_r_t).sum(dim=-1)
            alpha_rt = F.leaky_relu(alpha_rt, negative_slope=self.negative_slope)
            alpha_rt = F.softmax(alpha_rt, dim=1)
            msg[mask_rt] = x_j[mask_rt] * alpha_rt.view(-1, self.heads, 1)

        return msg

    def aggregate(self, inputs, index):
        return torch_geometric.nn.aggr.add(inputs, index)
    
  
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
                activation=nn.LeakyReLU(), dropout_rate=0.5, as_encoder=False):
    super().__init__()
    self.input_dim = input_dim
    self.hidden_dims = hidden_dims
    self.conv_dim = conv_dim
    self.num_heads = num_heads
    self.out_dim = out_dim
    self.f = activation
    self.dropout = nn.Dropout(dropout_rate)
    self.as_encoder=as_encoder

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
    # positional encoding based on shortest path distance
    pos_embedding = shortest_path_embeddings(edge_index, X.shape[0])
    X = torch.cat([X, pos_embedding], dim=1)
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
    if not self.as_encoder:
      h = self.output_layer(h)
    return h
  

# Transformer block with cross attention
class FuseTransformer(nn.Module):
    def __init__(self, embedding_dim, num_heads, forward_expansion, dropout):
        super(FuseTransformer, self).__init__()
        self.local_attn = nn.MultiheadAttention(embedding_dim, num_heads)
        self.global_attn = nn.MultiheadAttention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.norm4 = nn.LayerNorm(embedding_dim)
        self.ffn1 = nn.Sequential(
            nn.Linear(embedding_dim, forward_expansion * embedding_dim),
            nn.ReLU(),
            nn.Linear(forward_expansion * embedding_dim, embedding_dim)
        )
        self.ffn2 = nn.Sequential(
            nn.Linear(embedding_dim, forward_expansion * embedding_dim),
            nn.ReLU(),
            nn.Linear(forward_expansion * embedding_dim, embedding_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_embeddings, global_embedding, mask=None):
        node_embeddings = node_embeddings.unsqueeze(1)
        global_embedding = global_embedding.unsqueeze(0).expand(node_embeddings.size(0), -1, -1)

        # Q=local, K,V=global
        node_out, _ = self.local_attn(node_embeddings, global_embedding, global_embedding, key_padding_mask=mask)
        node_out = self.dropout(self.norm1(node_out + node_embeddings))

        # Q=global, K,V=local
        global_out, _ = self.global_attn(global_embedding, node_embeddings, node_embeddings, key_padding_mask=mask)
        global_out = self.dropout(self.norm2(global_out + global_embedding))

        node_out = node_out.squeeze(1)
        global_out = global_out[0]

        # FFN for local
        node_ffn_out = self.ffn1(node_out)
        node_out = self.dropout(self.norm3(node_ffn_out + node_out))

        # FFN for global
        global_ffn_out = self.ffn2(global_out)
        global_out = self.dropout(self.norm4(global_ffn_out + global_out))

        return node_out, global_out

# GNN to obtain local information. Transformer with cross-attention to attend pooled global information to nodes.
# No mask
class GraphLMEnc(nn.Module):
   def __init__(self, gnn, transformer):
    super().__init__()
    self.gnn = gnn
    self.transformer = transformer # this is a cross attention module instead of a regular transformer
  
   def forward(self,  x, edge_index, edge_weight, batch):
    local_embedding = self.gnn(x, edge_index, edge_weight, batch)
    global_embedding = global_mean_pool(local_embedding, batch)
    local_rep, global_rep = self.transformer(local_embedding, global_embedding)
    return local_rep, global_rep


# Masked node prediction
# Takes in mask
class GraphLMDec(nn.Module):
    def __init__(self, embedding_dim, num_heads, forward_expansion, dropout, output_dim):
        super().__init__()
        self.transformer_block = FuseTransformer(embedding_dim, num_heads, forward_expansion, dropout)
        self.output_layer = nn.Linear(embedding_dim, output_dim)
    
    def forward(self, node_embeddings, global_embedding, edge_index, mask):
        pos_embedding = shortest_path_embeddings(edge_index, node_embeddings.shape[0])
        node_embeddings = torch.cat([node_embeddings, pos_embedding], dim=1)
        # masked attention in decoder
        node_out, _ = self.transformer_block(node_embeddings, global_embedding, mask)
        predictions = self.output_layer(node_out) # used to compute MLM loss
        return predictions


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

# shortest path positional embedding
def shortest_path_embeddings(edge_index, num_nodes):
    G = nx.Graph()
    edge_list = edge_index.t().tolist()
    G.add_edges_from(edge_list)
    shortest_paths = dict(nx.shortest_path_length(G))
    
    sp_matrix = torch.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in shortest_paths[i]:
            sp_matrix[i, j] = shortest_paths[i][j]
    
    sp_embeddings = sp_matrix.mean(dim=1, keepdim=True)  # Example aggregation, mean of shortest paths
    return sp_embeddings   

