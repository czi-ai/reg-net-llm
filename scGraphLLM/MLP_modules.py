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


class PerturbationHead(nn.Module):
    """head for perturbation effect"""
    def __init__(self, embed_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, output_dim)
            )
        
    def forward(self, x):
        perturbed_embs = torch.unbind(x, dim=1)
        outs = []
        for emb in perturbed_embs:
            outs.append(self.net(emb))
        return torch.stack(outs, dim=1).squeeze()
    
