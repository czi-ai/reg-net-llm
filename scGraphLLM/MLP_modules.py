# Fully connected neural network modules
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

# Network for learnable attention weight
class MLPAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPAttention, self).__init__()
        self.proj_to_H = nn.Linear(input_dim, hidden_dim)
        self.proj_to_out = nn.Linear(hidden_dim, output_dim)
        self.attn_head = nn.Linear(hidden_dim, 1)
        self.activation = nn.Tanh()
    
    def forward(self, x):
        attn_W = torch.softmax(self.attn_head(self.activation(self.proj_to_H(x))), dim=1)
        weighted_input = torch.sum(x * attn_W, dim=1)
        out = self.proj_to_out(weighted_input)
        return out, attn_W


class MinMaxContrastive(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(MinMaxContrastive, self).__init__()
        self.alpha = alpha  
        self.beta = beta  
        self.cos_sim = nn.CosineSimilarity(dim=2)

    def forward(self, outputs, labels, attention_weights):
        loss_intra = 0
        loss_inter = 0
        for i in range(len(outputs)):
            mask = (labels == labels[i]).unsqueeze(1)
            cos_sim_matrix = self.cos_sim(attention_weights[i].unsqueeze(0), attention_weights[mask].view(-1, attention_weights.shape[1]))
            loss_intra += torch.sum(cos_sim_matrix)

            mask = (labels != labels[i]).unsqueeze(1)
            cos_sim_matrix = self.cos_sim(attention_weights[i].unsqueeze(0), attention_weights[mask].view(-1, attention_weights.shape[1]))
            loss_inter += torch.sum(cos_sim_matrix)

        loss = self.alpha * loss_intra - self.beta * loss_inter
        return loss