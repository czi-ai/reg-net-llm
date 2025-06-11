import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data as torchGeomData

from scGraphLLM.vocab import GeneVocab
from scGraphLLM.network import RegulatoryNetwork
from scGraphLLM._globals import *


class GraphTokenizer:
    """
    Converts a single-cell gene expression profile into a a representation suitable
    for input to `GDTransformer`

    This tokenizer bins expression values, filters genes based on expression and 
    presence in the regulatory network, and constructs a graph where nodes are genes 
    and edges represent regulatory interactions.

    Args:
        vocab (GeneVocab): Maps between gene names and node indices.
        network (RegulatoryNetwork, optional): Default regulatory network used to construct edges.
        max_seq_length (int): Maximum number of nodes (genes) to include in the graph.
        only_expressed_genes (bool): If True, exclude genes with zero expression.
        with_edge_weights (bool): If True, include edge weights from the network.
        n_bins (int): Number of bins to discretize gene expression.
        method (str): Method for binning ('quantile' or 'uniform').
    """
    def __init__(
            self, 
            vocab: GeneVocab,
            network: RegulatoryNetwork=None, 
            max_seq_length=2048, 
            only_expressed_genes=True,
            only_network_genes=True, 
            with_edge_weights=False, 
            n_bins=NUM_BINS, 
            method="quantile"
        ):
        self.vocab = vocab
        self.network = network
        self.max_seq_length = max_seq_length
        self.only_expressed_genes = only_expressed_genes
        self.only_network_genes = only_network_genes
        self.with_edge_weights = with_edge_weights
        self.n_bins = n_bins
        self.method = method

    @property
    def gene_to_node(self):
        return self.vocab.gene_to_node

    @property
    def node_to_gene(self):
        return self.vocab.node_to_gene

    def __call__(self, cell_expr: pd.Series, override_network: RegulatoryNetwork = None, from_counts=False, target_sum=1e6):
        """
        Tokenizes a single cell's expression profile into a torch_geometric Data object.

        Args:
            cell_expr (pd.Series): A single-cell gene expression vector (gene names as index).
            override_network (RegulatoryNetwork, optional): Overrides the default network for this cell.
            from_counts (boolean): If True, assume cell_expr is raw UMI counts and normalize + log transform.
            target_sum (float): target sum for normalization if from_counts is True

        Returns:
            torch_geometric.data.Data: Expression features and regulatory edges.
        """
        if from_counts:
            cell_expr = np.log1p(cell_expr / cell_expr.sum() * target_sum)

        # override original network if network is provided
        network = override_network if override_network is not None else self.network
        
        # limit cell to known genes in vocabulary
        cell_expr = cell_expr[cell_expr.index.isin(self.gene_to_node)]

        # tokenize cell by by binning expression values
        cell = tokenize_expr(cell_expr, n_bins=self.n_bins, method=self.method)
        if self.only_expressed_genes:
            cell = cell[cell != ZERO_IDX]

        # limit cell to genes in the the network
        if self.only_network_genes:
            cell = cell[cell.index.isin(network.genes)]
        
        # enforce max sequence length
        if (self.max_seq_length is not None) and (cell.shape[0] > self.max_seq_length):
            cell = cell.nlargest(n=self.max_seq_length, keep="first")

        # create local gene to node mapping
        local_gene_to_node = {gene:i for i, gene in enumerate(cell.index)}

        # Subset network to only include genes in the cell
        network_cell = network.df.copy()
        network_cell = network_cell[
            network_cell[network.reg_name].isin(cell.index) & 
            network_cell[network.tar_name].isin(cell.index)
        ]

        reg_index = network_cell[network.reg_name].map(local_gene_to_node).values
        tar_index = network_cell[network.tar_name].map(local_gene_to_node).values
        edge_index = torch.tensor(np.array([reg_index, tar_index]))

        node_expression = torch.tensor(np.array([
            (self.gene_to_node[gene], cell[gene]) for gene in cell.index
        ]), dtype=torch.long)
            
        if self.with_edge_weights:
            edge_weights = torch.tensor(np.array(network_cell[network.wt_name]))
            data = torchGeomData(
                x=node_expression, 
                edge_index=edge_index, 
                edge_weight=edge_weights
            )
        else:
            data = torchGeomData(
                x=node_expression, 
                edge_index=edge_index
            )

        return data


def tokenize_expr(expr: pd.Series, n_bins: int = 5, method: str = "quantile") -> pd.Series:
    """
    Discretize (bin) a single gene expression profile into categorical bins.

    Parameters
    ----------
    expr : pd.Series
        Gene expression values for a single sample or cell.
        Index should be gene names, values are raw counts or continuous expression levels.
    n_bins : int, optional (default=5)
        Number of bins to categorize expression values into.
    method : str, optional (default="quantile")
        Method to determine bin edges:
        - "quantile": bins are based on quantiles of non-zero expression values.
        - any other value: bins are equally spaced between min and max of non-zero values.

    Returns
    -------
    pd.Series
        A series of the same index as `expr` where each value is the bin number (1 to n_bins)
        representing the discretized expression level.
        Zero-expression genes remain assigned to bin 0.

    Notes
    -----
    - Only non-zero expression values are binned; zero values remain zero.
    - If there is only one unique non-zero value, it is assigned the middle bin (rounded n_bins/2).
    """
    non_zero_idx = expr.to_numpy(dtype=float).nonzero()[0]
    binned = np.zeros_like(expr, dtype=np.int16)

    if len(non_zero_idx) == 0:
        return pd.Series(binned, index=expr.index)
    
    non_zero_expr = expr.iloc[non_zero_idx]
    if np.unique(non_zero_expr).shape[0] > 1:
        if method == "quantile":
            bins = np.quantile(non_zero_expr, np.linspace(0, 1, n_bins))[1:-1]
        else:
            bins = np.linspace(min(non_zero_expr), max(non_zero_expr), n_bins)
        binned[non_zero_idx] = np.digitize(non_zero_expr, bins, right=True) + 1
    else:
        binned[non_zero_idx] = round(n_bins / 2)

    return pd.Series(binned, index=expr.index)
