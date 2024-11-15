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
    """Head for linked prediction"""
    def __init__(self, embed_dim, output_dim):
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


# Fully connect self attention network
class AttentionMLP(nn.Module):
    def __init__(self, in_size, hidden_size=16, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        self.attn_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, in_size, bias=False) 
            ) for _ in range(num_heads)
        ])

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, 
                                              gain=nn.init.calculate_gain("leaky_relu"))

        self.attn_heads.apply(init_weights)

    def forward(self, x):
        w = torch.stack([head(x) for head in self.attn_heads], dim=1)
        beta = torch.softmax(w, dim=1) # (G, heads, 2, D)
        attended = (beta * x.unsqueeze(1)).sum(dim=1) # (G, 2, D)
        output = attended.mean(dim=1)
        return output, beta
    
