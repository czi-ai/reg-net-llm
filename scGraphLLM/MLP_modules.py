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
    def __init__(self, max_hop, embed_dim, hidden_dim, output_dim, 
                 total_gene_dim, batch_size):
        super().__init__()
        self.max_hop = max_hop
        self.embed_dim = embed_dim

        self.net = nn.Sequential(
            nn.Linear(max_hop * embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.num_genes = total_gene_dim
        self.proj = nn.Linear(batch_size, embed_dim)
        self.bfloat16()

    def forward(self, edge_index_list, num_nodes_list, perturb_one_hot):
        dtype   = self.proj.weight.dtype        
        device  = perturb_one_hot.device

        # (gene, batch) one-hot → bf16 to match weights
        one_hot_mat = (
            torch.nn.functional.one_hot(
                perturb_one_hot, num_classes=self.num_genes
            )
            .to(dtype=dtype, device=device)
            .T                                   # (gene, batch)
        )

        init_emb = self.proj(one_hot_mat)        # (gene, embed_dim)
        assert init_emb.shape == (self.num_genes, self.embed_dim)

        batch_of_emb = []
        for edge_index, num_nodes in zip(edge_index_list, num_nodes_list):
            num_edges = edge_index.shape[1]
            A = torch.sparse_coo_tensor(
                    edge_index,
                    torch.ones(num_edges, device=device, dtype=dtype),
                    (num_nodes, num_nodes),
                ).coalesce()                     # (gene, gene) bf16 sparse

            omega = []
            for _ in range(self.max_hop):
                H = torch.sparse.mm(A.to_dense(), init_emb)  # bf16 matmul
                omega.append(H)

            omega  = torch.stack(omega, dim=-1).reshape(num_nodes, -1)
            Omega  = self.net(omega)
            batch_of_emb.append(Omega)

        out = torch.stack(batch_of_emb, dim=0)   # (cell, gene, d)
        return out                  
            
            
            
            
        
         
        
        
        
    
