import numpy as np
import pandas as pd

REG_VALS = "regulator.values"
TAR_VALS = "target.values"
MI_VALS = "mi.values"
LOGP_VALS = "log.p.values"
SCC_VALS = "scc.values"


def build_class_edge_matrix(class_networks, classes, default_alpha):
    """
    Build edge × class matrix of p-values. Missing edges use default_alpha.
    """
    
    # Identify global set of edges
    all_edges = set()
    for class_df in class_networks.values():
        all_edges.update(class_df.index)
    all_edges = sorted(list(all_edges))
    
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    num_edges = len(all_edges)
    num_classes = len(classes)

    # Initialize matrix with default alpha
    E = np.full((num_edges, num_classes), default_alpha, dtype=np.float32)
    MI = np.full((num_edges, num_classes), np.nan, dtype=np.float32)

    for j, c in enumerate(classes):
        df = class_networks[c]
        for e in df.index:
            i = edge_to_idx[e]
            E[i, j] = np.exp(df.loc[e, LOGP_VALS])
            MI[i, j] = df.loc[e, MI_VALS]

    return E, MI, all_edges


def infer_cell_edges_(probs, E, MI, alpha=None):
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
    # if using default MI = 0
    # expected_mis = MI @ probs
    # if using defaul MI = np.nan
    mask = ~np.isnan(MI)
    weighted_mis = np.where(mask, MI * probs, 0)
    weight_sums = mask @ probs
    expected_mis = np.divide(weighted_mis.sum(axis=1), weight_sums, out=np.zeros_like(weight_sums), where=weight_sums != 0)

    if alpha is not None:
        edge_ids = np.where(expected_pvals <= alpha)[0]
        expected_pvals = expected_pvals[edge_ids]
        expected_mis = expected_mis[edge_ids]
    else:
        edge_ids = np.arange(len(expected_pvals))

    return edge_ids, expected_pvals, expected_mis


def make_undirected(
    df,
    reg_col=REG_VALS,
    tar_col=TAR_VALS,
    mi_col=MI_VALS,
    logp_col=LOGP_VALS,
    drop_unpaired=False
):
    """
    Makes a graph dataframe undirected by either:
    - Adding reverse edges if missing (default), or
    - Dropping edges without a reverse (if drop_unpaired=True)

    Parameters:
    - df: DataFrame with directed edges
    - drop_unpaired: If True, only keep edges that have reverse counterparts

    Returns:
    - DataFrame with undirected edges (by addition or filtering)
    """
    edge_set = set(zip(df[reg_col], df[tar_col]))
    reverse_set = set((t, r) for r, t in edge_set)

    if drop_unpaired:
        # Keep only edges that have a reverse
        bidirectional_edges = edge_set & reverse_set
        mask = [(r, t) in bidirectional_edges for r, t in zip(df[reg_col], df[tar_col])]
        return df[mask].reset_index(drop=True)
    
    # Add reverse edges if missing
    existing_edges = set(zip(df[reg_col], df[tar_col]))
    reversed_edges = []

    for idx, row in df.iterrows():
        src, tgt = row[reg_col], row[tar_col]
        if (tgt, src) not in existing_edges:
            reversed_edges.append({
                reg_col: tgt,
                tar_col: src,
                mi_col: row[mi_col],
                logp_col: row[logp_col]
            })

    if reversed_edges:
        reversed_df = pd.DataFrame(reversed_edges)
        df = pd.concat([df, reversed_df], ignore_index=True)

    return df

def get_cell_network_df(edge_ids, pvals, mis, all_edges, limit_regulon=None, drop_unpaired=True):
    if len(edge_ids) == 0:
        return pd.DataFrame({REG_VALS: [], TAR_VALS: [], MI_VALS: [], LOGP_VALS: []})
    
    edges = np.array(all_edges)[edge_ids]
    regulators, targets = zip(*edges)
    df = pd.DataFrame({REG_VALS: regulators, TAR_VALS: targets})
    
    if mis is not None:
        df[MI_VALS] = mis

    if pvals is not None:
        df[LOGP_VALS] = np.log(pvals)
    
    if limit_regulon is not None:
        df = df.groupby(REG_VALS, group_keys=False)\
            .apply(lambda regulon: regulon.nlargest(limit_regulon, MI_VALS))\
            .reset_index(drop=True)\
            .pipe(make_undirected, drop_unpaired=drop_unpaired)        

    return df