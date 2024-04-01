# Fully connected neural network modules
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

class RobertaLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.layer_norm(embed_dim),
            nn.Linear(embed_dim, output_dim)
        )

    def forward(self, features):
        return self.net(features)
        return x


# Fully connect self attention network
class MLPAttention(nn.Module):
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
        beta = torch.softmax(w, dim=1) # (batch, heads, 2, in_dim)
        attended = (beta * x.unsqueeze(1)).sum(dim=1) # (batch, 2, ind_dim)
        output = attended.mean(dim=1)
        return output, beta

# Fully connected cross attention network on edges
class MLPCrossAttention(nn.Module):
    def __init__(self, hidden_size=16, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        self.attn_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, 1, bias=False)
            ) for _ in range(num_heads)
        ])

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, 
                                              gain=nn.init.calculate_gain("leaky_relu"))

        self.attn_heads.apply(init_weights)
    def forward(self, A1, A2):
        num_nodes = A1.shape[0]
        A_pairwise = torch.stack([A1, A2], dim=-1).reshape(-1, 2)  
        attention_weights = torch.cat([
            torch.softmax(head(A_pairwise), dim=0) for head in self.attn_heads
        ], dim=-1)
        attended = attention_weights * A1.reshape(-1, 1)
        output = attended.sum(dim=-1).reshape(num_nodes, num_nodes)
        return output, attention_weights

# Siamese contrastive loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, verbose=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.verbose = verbose

    def forward(self, xs, ys):
        if len(ys) == len(xs):
            ValueError("Inconsistent number of labels and embeddings")

        xs_normalized = [F.normalize(x, p=2) for x in xs]
        x_in = torch.stack(xs_normalized, dim=0)
        S = torch.matmul(x_in, x_in.permute(0, 2, 1))

        pos, neg = [], []
        for i in range(len(ys)):
            for j in range(i + 1, len(ys)):
                if ys[i] == ys[j]:
                    pos.append(S[i, j])
                else:
                    neg.append(S[i, j])
        
        positive_pairs = torch.stack(pos)
        negative_pairs = torch.stack(neg)

        if self.verbose:
            print("Number of matches:", positive_pairs.shape[0])
            print("Number of mismatches:", negative_pairs.shape[0])

        # Siamese contrastive loss
        loss = torch.mean((F.relu(positive_pairs - self.margin) ** 2)) + \
               torch.mean((F.relu(self.margin - negative_pairs) ** 2))
        return loss

        



