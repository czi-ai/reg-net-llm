# Fully connected neural network modules
import torch
import torch.nn as nn

class RobertaLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, output_dim)
        )

    def forward(self, features):
        return self.net(features)


class LinkPredictHead(nn.Module):
    """Head for link prediction"""
    def __init__(self, embed_dim, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(2 * embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, output_dim)
            )

    def forward(self, x_i, x_j):
        x = torch.cat([x_i, x_j], dim=-1)
        h = torch.sigmoid(self.net(x))
        return h


class PerturbEmbedding(nn.Module):
    """
    Perturbational embedding
    Input: graphs, perturbation design mat 
    perturb_one_hot: (cell, gene)
    """
    def __init__(self, max_hop, embed_dim, hidden_dim, output_dim, perturb_one_hot):
        super().__init__()
        self.max_hop = max_hop
        self.embed_dim = embed_dim
        self.net = nn.Sequential(
            nn.Linear(max_hop * self.embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        ) 
        self.emb_p = nn.Embedding(perturb_one_hot.shape[0], embed_dim)
    
    def forward(self, edge_index_list, num_nodes_list, perturb_one_hot):
        init_emb = self.emb_p(perturb_one_hot.T.argmax(dim=1)) # (gene, embed_dim)
        batch_of_emb = []
        device = perturb_one_hot.device
        for i in range(len(edge_index_list)):
            num_nodes = num_nodes_list[i]
            edge_index = edge_index_list[i]
            num_edges = edge_index.shape[1]
            A = torch.sparse_coo_tensor(edge_index, 
                                        torch.ones(num_edges, device=device), 
                                        (num_nodes, num_nodes)).coalesce() # gene x gene
            omega = []
            # start shifting with k hop
            for _ in self.max_hop:
                H = init_emb
                H = torch.sparse.mm(A, H)
                omega.append(H)
                
            omega = torch.stack(omega, dim=-1).reshape(num_nodes, -1) # (gene, embed_dim * k)
            Omega = self.net(torch.cat(omega, dim=-1)) # (gene, d)
            batch_of_emb.append(Omega)
        out = torch.stack(batch_of_emb, dim=0) # (cell, gene, d)
        return out
            
            
            
            
            
        
         
        
        
        
    
