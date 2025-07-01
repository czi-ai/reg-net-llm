# Graph network modules
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATEncoder(nn.Module):
    """Graph Attention Network (GAT) Encoder"""
    def __init__(self, in_channels, hidden_dim, out_channels, heads=1, layers=1):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.layers = nn.ModuleList()
        self.layers.append(GATConv(in_channels, hidden_dim, heads=heads, concat=False))
        for _ in range(layers - 1):
          self.layers.append(GATConv(hidden_dim, hidden_dim, heads=heads, concat=False))
        self.proj = nn.Linear(hidden_dim, out_channels)

    def forward(self, x, edge_index):
        for layer in self.layers:
           x = layer(x, edge_index)
           x = F.gelu(x)
        x = self.proj(x)
        return x
