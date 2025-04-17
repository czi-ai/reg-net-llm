
import numpy as np
import pandas as pd
import torch
import os

REG_VALS = "regulator.values"
TAR_VALS = "target.values"


def mask_values(sparse_mat, mask_prob=0.15, mask_value=0, seed=12345):
    """
    Masks each nonzero value in a sparse matrix with a given probability.

    Parameters:
        sparse_mat (scipy.sparse.spmatrix): Input sparse matrix.
        mask_prob (float): Probability of masking each nonzero value (default: 0.15).
        mask_value (float): Value to use for masking (default: 0, suitable for sparse).

    Returns:
        scipy.sparse.spmatrix: Masked sparse matrix with the same format.
    """
    sparse_mat = sparse_mat.tocoo()
    mask = np.random.default_rng(seed).random(len(sparse_mat.data)) < mask_prob
    masked_indices = (sparse_mat.row[mask], sparse_mat.col[mask])
    sparse_mat.data[mask] = mask_value

    return sparse_mat.tocsr(), masked_indices


def get_locally_indexed_edges(input_genes, src_nodes, dst_nodes):
    network = pd.DataFrame({"src": src_nodes, "dst": dst_nodes})
    edges = {}
    for i, genes_i in enumerate(input_genes):
        local_gene_to_node_index = {gene: i for i,gene in enumerate(genes_i)}
        edges_i = network[
            network["src"].isin(genes_i) & 
            network["dst"].isin(genes_i)
        ].assign(**{
            "src_idx": lambda df: df["src"].map(local_gene_to_node_index),
            "dst_idx": lambda df: df["dst"].map(local_gene_to_node_index),
        })[["src_idx", "dst_idx"]].to_numpy().T
        edges[i] = edges_i
    return edges


def get_locally_indexed_masks_expressions(adata, masked_indices, input_genes):
    masks, masked_expressions = {}, {}
    for i in range(adata.shape[0]):
        # identify masked genes
        row_indices = masked_indices[0]
        col_indices = masked_indices[1]
        row_masked_indices = np.where(row_indices == i)[0]
        masked_genes = adata.var_names[col_indices[row_masked_indices]].tolist()
        # get masked gene input indices
        masked_input_genes = pd.Series(input_genes[i]).pipe(lambda x: x[x.isin(masked_genes)])
        masks[i] = masked_input_genes.index.to_list()
        masked_expressions[i] = adata[i, masked_input_genes].X.toarray().flatten()
    return masks,masked_expressions



def save_embedding(file, x, cache=False, cache_dir=None, **features):
    if not cache:
        np.savez(file=file, allow_pickle=True, x=x, **features)
        return
    
    assert cache_dir is not None, "You must specify a `cache_dir` when `cache=True`"
    os.makedirs(cache_dir, exist_ok=True)

    # detemine num_cells
    num_cells = x.shape[0]
    features["x"] = x
    
    # Check all features are same length along axis 0
    for key, val in features.items():
        if key == "metadata":
            assert isinstance(val, dict), "metadata must be a dictionary of lists, each of length num_cells"
            for sub_key, sub_val in val.items():
                assert len(sub_val) == num_cells, f"metadata feature '{sub_key}' has inconsistent length"
        else:
            assert len(val) == num_cells, f"Feature '{key}' has inconsistent length"

    for i in range(num_cells):
        data = {}
        for k, v in features.items():
            if k == "metadata":
                data[k] = {sub_k: sub_v[i] for sub_k, sub_v in v.items()}
            else:
                data[k] = v[i]

        torch.save(data, os.path.join(cache_dir, f"emb_{i:06d}.pt"))
