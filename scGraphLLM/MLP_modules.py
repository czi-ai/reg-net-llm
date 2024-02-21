# Fully connected neural network modules
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

# Fully connect attention network
class MLPAttention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(MLPAttention, self).__init__()

        self.attn_head = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, x):
        w = self.attn_head(x)
        beta = torch.softmax(w, dim=1)
        return (beta * x).sum(1), beta


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

        



