# Fully connected neural network modules
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import get_laplacian, dense_to_sparse, to_dense_adj


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
    

# Per edge cross attention with Nystrom approx
class MLPCrossAttention(nn.Module):
    def __init__(self, hidden_size=16, num_heads=1, num_landmarks=50):
        super().__init__()
        self.num_heads = num_heads
        self.num_landmarks = num_landmarks
        self.q_linear = nn.Linear(1, hidden_size * num_heads)
        self.k_linear = nn.Linear(1, hidden_size * num_heads)
        self.v_linear = nn.Linear(1, hidden_size * num_heads)
        self.leaky_relu = nn.LeakyReLU()
        self.out_linear = nn.Linear(hidden_size * num_heads, 1, bias=False)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("leaky_relu"))

        self.apply(init_weights)

    def forward(self, A1, A2):
        num_nodes = A1.shape[0]
        Q = self.leaky_relu(self.q_linear(A1.reshape(-1, 1))).view(num_nodes, -1, self.num_heads)
        K = self.leaky_relu(self.k_linear(A2.reshape(-1, 1))).view(num_nodes, -1, self.num_heads)
        V = self.leaky_relu(self.v_linear(A2.reshape(-1, 1))).view(num_nodes, -1, self.num_heads)

        landmark_indices = torch.randperm(K.size(1))[:self.num_landmarks]
        K_landmarks = K[:, landmark_indices, :]
        QK_landmarks = torch.matmul(Q, K_landmarks.transpose(1, 2))
        K_landmarks_inv = pseudoinverse(K_landmarks)
        attention_scores = torch.matmul(QK_landmarks, torch.matmul(K_landmarks_inv, K.transpose(1, 2)))
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attended = torch.matmul(attention_weights, V)
        output = self.out_linear(attended.view(num_nodes, -1)).reshape(num_nodes, num_nodes)
        return output, attention_weights


# Fully connected local cross attention network on edges. Only attend to top 10 neighbours
# A1 = Q, A2 = K, V
class MLPLocalCrossAttention(nn.Module):
    def __init__(self, hidden_size=16, num_heads=1, k=50):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.k = k

        self.q_proj = nn.Linear(1, hidden_size * num_heads)
        self.k_proj = nn.Linear(1, hidden_size * num_heads)
        self.v_proj = nn.Linear(1, hidden_size * num_heads)

        self.leaky_relu = nn.LeakyReLU()

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("leaky_relu"))

        self.apply(init_weights)

    def forward(self, A1, A2):
        _, topk_indices_A1 = torch.topk(A1, self.k, dim=1)
        _, topk_indices_A2 = torch.topk(A2, self.k, dim=1)

        topk_A1 = torch.gather(A1, 1, topk_indices_A1)
        topk_A2 = torch.gather(A2, 1, topk_indices_A2)

        Q = self.q_proj(topk_A1.view(-1, 1)).view(-1, self.k, self.hidden_size)
        K = self.k_proj(topk_A2.view(-1, 1)).view(-1, self.k, self.hidden_size)
        V = self.v_proj(topk_A2.view(-1, 1)).view(-1, self.k, self.hidden_size)

        attention_scores = torch.matmul(Q, K.transpose(1, 2))
        attention_weights = F.softmax(attention_scores, dim=-1)

        updated_values = torch.matmul(attention_weights, V).mean(dim=2).squeeze()
        updated_A2 = torch.zeros_like(A2)
        for i in range(A2.size(0)):
            updated_A2[i, topk_indices_A2[i]] = updated_values[i]

        return updated_A2, attention_weights

class CrossAttentionFuse(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        self.local_attn = nn.MultiheadAttention(embedding_dim, num_heads)
        self.global_attn = nn.MultiheadAttention(embedding_dim, num_heads)

    def forward(self, node_embeddings, global_embedding):
        node_embeddings = node_embeddings.unsqueeze(1)
        global_embedding = global_embedding.unsqueeze(0).expand(node_embeddings.size(0), -1, -1)
        
        node_out, _ = self.local_attn(node_embeddings, global_embedding, global_embedding)
        global_out, _ = self.global_attn(global_embedding, node_embeddings, node_embeddings)
        
        node_out = node_out.squeeze(1)
        global_out = global_out[0]
        
        return node_out, global_out

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

