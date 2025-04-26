import torch
from torch_geometric.utils import scatter, remove_self_loops

from _globals import *  ## imported global variables are all caps 

def _identity(x):
    return x

def _exp_kernel(x, beta):
    return torch.exp(-beta * (x + 1))

def _cosine_kernel(x):
    PI = torch.acos(torch.Tensor([-1]))
    return torch.cos(PI * x / 2)

def _rescaled_L(edge_index, num_nodes, edge_weight=None):
    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight) 
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32, device=edge_index.device)
    row, col = edge_index[0], edge_index[1]

    if row.shape != edge_weight.shape:
        print(f"Shape mismatch: row={row.shape}, edge_weight={edge_weight.shape}")
    max_index = max_index = row.max().item()
    if max_index >= num_nodes:
        print(f"Row index out of bounds! max index = {max_index} >= num_nodes = {num_nodes}")
    if torch.isnan(edge_weight).any():
        print("NaN detected in edge_weight!")
    if torch.isinf(edge_weight).any():
        print("Inf detected in edge_weight!")

    deg = scatter(edge_weight, row, 0, dim_size=num_nodes, reduce='sum')
    deg = deg.clamp(min=1e-8)
    assert not torch.isnan(edge_weight).any(), "NaN values in edge_weight"
    assert not torch.isnan(deg).any(), "NaN values in degree"
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt.isnan(), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col] # D^(-1/2) * A * D^(-1/2)
    assert not torch.isnan(edge_weight).any(), "NaN values in edge_weight after normalization"
    L_rescaled = torch.sparse_coo_tensor(edge_index, -edge_weight, (num_nodes, num_nodes))
    return L_rescaled

def _chebyshev_coeff(L_rescaled, K, func, N=100):
    # Gauss-Chebyshev quadrature
    ind = torch.arange(0, K+1, dtype=torch.float32, device=L_rescaled.device)
    ratio = torch.pi * (torch.arange(1, N+1, dtype=torch.float32, device=L_rescaled.device) - 0.5) / N
    x = torch.cos(ratio) # quadrature points
    T_kx = torch.cos(ind.view(-1, 1) * ratio) 
    w = torch.ones(N, device=L_rescaled.device) * (torch.pi / N)
    f_x = func(x)
    c_k = (2 / torch.pi) * torch.matmul(T_kx, w * f_x)
    return c_k

@torch.amp.autocast(enabled=False, device_type='cuda')
def _chebyshev_diffusion_per_sample(edge_index, num_nodes, E, k=128, edge_weight=None, beta=0.5):
    """
    E: (S, H, d)
    """
    L_rescaled = _rescaled_L(edge_index, num_nodes, edge_weight)
    c_k = _chebyshev_coeff(L_rescaled, k, lambda x: _exp_kernel(x, beta))
    E = E.to(torch.float32)
    s, h, d = E.size()
    assert s == num_nodes, f"Expect {num_nodes} nodes, Got {s}"
    E_reshaped = E.reshape(num_nodes, h * d)
    c_k = c_k.to(torch.float32)
    T_0 = E_reshaped
    T_1 = torch.sparse.mm(L_rescaled, E_reshaped)
    y = c_k[0] * T_0 + c_k[1] * T_1
    
    # start recursion
    T_k_prev = T_1
    T_k_prev_prev = T_0
    for k in range(2, k + 1):
        T_k = 2 * torch.sparse.mm(L_rescaled, T_k_prev) - T_k_prev_prev
        y += c_k[k] * T_k
        
        # shift index
        T_k_prev_prev = T_k_prev 
        T_k_prev = T_k
        
    final_emb = y.reshape(num_nodes, h, d)
    final_emb = final_emb.bfloat16()
    return final_emb

def _chebyshev_diffusion(edge_index_list, num_nodes_list, E, k=64, beta=0.5):
    """
    edge index list: list of edge index, length B
    E: (B, S, H, d)
    """
    B, S, H, D = E.size()
    final_emb = []
    
    for i in range(B):
        E_i = E[i, :num_nodes_list[i], ...]
        edge_index = edge_index_list[i]
        sample_emb = _chebyshev_diffusion_per_sample(edge_index, num_nodes_list[i], E_i, k=k, beta=beta)
        
        # pad zero at the right end
        pad_size = S - sample_emb.size(0)
        if pad_size > 0:
            zero_pad_right = torch.zeros(pad_size, H, D, device=E.device, dtype=E.dtype)
            sample_emb = torch.cat([sample_emb, zero_pad_right], dim=0)
        final_emb.append(sample_emb)
    fe = torch.stack(final_emb, dim=0)
    assert fe.size() == E.size(), f"Expect {E.size()}, Got {fe.size()}"
    return fe