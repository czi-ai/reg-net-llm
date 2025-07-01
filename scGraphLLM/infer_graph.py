import numpy as np
import pandas as pd
from typing import Union, Dict, List

from scGraphLLM._globals import *
from scGraphLLM.network import RegulatoryNetwork


def build_class_edge_matrix(
        class_networks: Dict[str, RegulatoryNetwork], 
        classes: List[str], 
        default_alpha: float
    ):
    """
    Build edge × class matrix of p-values. Missing edges use default_alpha.
    """
    # Identify global set of edges
    all_edges = set()
    for network in class_networks.values():
        all_edges.update(network.edges)
    all_edges = sorted(list(all_edges))
    
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    num_edges = len(all_edges)
    num_classes = len(classes)

    # Initialize matrix with default alpha
    E = np.full((num_edges, num_classes), default_alpha, dtype=np.float32)
    W = np.full((num_edges, num_classes), np.nan, dtype=np.float32)

    for j, c in enumerate(classes):
        network = class_networks[c]
        for i, e in enumerate(network.edges):
            idx = edge_to_idx[e]
            edge = network.df.iloc[i]
            E[idx, j] = np.exp(edge[network.lik_name])
            W[idx, j] = edge[network.wt_name]

    return E, W, np.array(all_edges)


def infer_cell_edges_(probs, E, W, alpha=None):
    """
    Fast inference using precomputed class-edge matrix.

    Parameters:
    - probs: array of class probabilities
    - E: [num_edges x num_classes] matrix of per-class p-values
    - all_edges: list of edge tuples (same order as rows in E)
    - alpha: optional p-value threshold

    Returns:
    - List of edge indices (integers into all_edges) passing the threshold
    """
    probs = np.asarray(probs)
    if probs.sum() == 0:
        return np.array([]), np.array([]), np.array([])

    assert np.abs(probs.sum() - 1.0) < 1e-4, "probs must sum to 1"

    expected_pvals = E @ probs
    # if using default W = 0
    # expected_wts = W @ probs
    # if using defaul W = np.nan
    mask = ~np.isnan(W)
    weighted_mis = np.where(mask, W * probs, 0)
    weight_sums = mask @ probs
    expected_wts = np.divide(weighted_mis.sum(axis=1), weight_sums, out=np.zeros_like(weight_sums), where=weight_sums != 0)

    if alpha is not None:
        edge_ids = np.where(expected_pvals <= alpha)[0]
        expected_pvals = expected_pvals[edge_ids]
        expected_wts = expected_wts[edge_ids]
    else:
        edge_ids = np.arange(len(expected_pvals))

    return edge_ids, expected_pvals, expected_wts
