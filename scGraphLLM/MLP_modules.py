# Fully connected neural network modules
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import get_laplacian, dense_to_sparse, to_dense_adj
from GNN_modules import SPosPE


def laplacian_normalized(A):
    edge_index, weight = dense_to_sparse(A)
    laplacian_ind, laplacian_weight = get_laplacian(edge_index, edge_weight=weight, 
                                                 normalization='sym', num_nodes=A.shape[0])
    L = to_dense_adj(laplacian_ind, edge_attr=laplacian_weight)
    return L


def pseudoinverse(A):
    U, S, V = torch.svd(A)
    S_inv = torch.zeros_like(S)
    non_zero_indices = S > 1e-6
    S_inv[non_zero_indices] = 1.0 / S[non_zero_indices]
    A_pinv = torch.matmul(torch.matmul(V, torch.diag(S_inv)), U.t())

    return A_pinv

class RobertaLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, output_dim)
        )

    def forward(self, features):
        return self.net(features)

class RobertaLMHeadPE(nn.Module):
    def __init__(self, num_nodes, embed_dim, output_dim):
        super().__init__()
        self.PE = SPosPE(num_nodes, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, output_dim)
        )

    def forward(self, edge_index, features):
        pos_embedding = self.PE(edge_index, features.shape[0])
        features = torch.cat([features, pos_embedding], dim=1)
        return self.net(features)


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
    
# Q = virtual node, K,V = actual node embedding
class CrossAttentionMLP(nn.Module):
    def __init__(self, d_model, projection_dim):
        super().__init__()
        self.query_layer = nn.Sequential(
            nn.Linear(d_model, projection_dim),
            nn.GELU(),
        )
        self.key_layer = nn.Sequential(
            nn.Linear(d_model, projection_dim),
            nn.GELU(),
        )
        self.value_layer = nn.Sequential(
            nn.Linear(d_model, projection_dim),
            nn.GELU(),
        )
        self.final_projection = nn.Linear(projection_dim, d_model)
        self.softmax = nn.Softmax(dim=-1)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, virtual_node, node_features):
        query = self.query_layer(virtual_node)
        keys = self.key_layer(node_features)
        values = self.value_layer(node_features)
        
        # Compute attention weights
        attention_scores = torch.matmul(query, keys.transpose(-2, -1))
        attention_weights = self.softmax(attention_scores)
        
        # Compute attended values
        attended_features = torch.matmul(attention_weights, values)
        
        # Project back to the original dimension
        attended_features = self.final_projection(attended_features)
        
        # Layer norm 
        attended_features = self.layer_norm(attended_features + virtual_node)
        
        return attended_features
    
class ContrastiveLossCos(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLossCos, self).__init__()
        self.margin = margin
        self.cosine_similarity = nn.CosineSimilarity(dim=1)

    def forward(self, embedding1, embedding2, label):
        # Cosine similarity
        similarity = self.cosine_similarity(embedding1, embedding2)
        loss = torch.mean((1 - label) * (1 - similarity) +
                          label * torch.clamp(self.margin - similarity, min=0.0))
        return loss
    

class CombinedContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, alpha=0.5, beta=0.5):
        super(CombinedContrastiveLoss, self).__init__()
        self.margin = margin
        self.cosine_similarity = nn.CosineSimilarity(dim=1)
        self.alpha = alpha
        self.beta = beta

    def forward(self, embedding1, embedding2, adj1, adj2, k, label):
        # Node embedding cosine similarity
        embedding_similarity = self.cosine_similarity(embedding1, embedding2)
        embedding_loss = torch.mean((1 - label) * (1 - embedding_similarity) +
                                    label * torch.clamp(self.margin - embedding_similarity, min=0.0))
        
        # Graph Spectral distance: top k eigenvalue of normalized graph laplacians
        L_adj1 = laplacian_normalized(adj1)
        L_adj2 = laplacian_normalized(adj2)
        eigvals1, _ = torch.linalg.eigh(L_adj1)
        eigvals2, _ = torch.linalg.eigh(L_adj2)
        eigvals1 = eigvals1[:k]
        eigvals2 = eigvals2[:k]
        spectral_distance = torch.norm(eigvals1 - eigvals2)
        
        return self.alpha * embedding_loss + self.beta * spectral_distance

    
# Siamese contrastive loss
class ContrastiveLossSiamese(nn.Module):
    def __init__(self, margin=1.0, verbose=False):
        super(ContrastiveLossSiamese, self).__init__()
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

        
class LinkPredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(LinkPredictor, self).__init__()
        self.net = nn.Sequential(
                nn.Linear(2 * in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1)
            )

    def forward(self, x_i, x_j):
        x = torch.cat([x_i, x_j], dim=-1)
        h = torch.sigmoid(self.net(x))
        return h

class Discriminator(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout_rate=0.5):
        super(Discriminator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.network(x)

