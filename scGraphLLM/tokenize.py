import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data as torchGeomData

from scGraphLLM.vocab import GeneVocab
from scGraphLLM.network import RegulatoryNetwork
from scGraphLLM.preprocess import tokenize_expr
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
            with_edge_weights=False, 
            n_bins=NUM_BINS, 
            method="quantile"
        ):
        self.vocab = vocab
        self.network = network
        self.max_seq_length = max_seq_length
        self.only_expressed_genes = only_expressed_genes
        self.with_edge_weights = with_edge_weights
        self.n_bins = n_bins
        self.method = method

    @property
    def gene_to_node(self):
        return self.vocab.gene_to_node

    @property
    def node_to_gene(self):
        return self.vocab.node_to_gene

    def __call__(self, cell_expr: pd.Series, override_network: RegulatoryNetwork = None):
        """
        Tokenizes a single cell's expression profile into a torch_geometric Data object.

        Args:
            cell_expr (pd.Series): A single-cell gene expression vector (gene names as index).
            override_network (RegulatoryNetwork, optional): Overrides the default network for this cell.

        Returns:
            torch_geometric.data.Data: Expression features and regulatory edges.
        """
        # override original network if network is provided
        network = override_network if override_network is not None else self.network
        
        # limit cell to genes in the the network
        cell_expr = cell_expr[cell_expr.index.isin(network.genes)]

        cell = tokenize_expr(cell_expr, n_bins=self.n_bins, method=self.method)
        if self.only_expressed_genes:
            cell = cell[cell != ZERO_IDX]
        
        # enforce max sequence length
        if (self.max_seq_length is not None) and (cell.shape[0] > self.max_seq_length):
            cell = cell.nlargest(n=self.max_seq_length)

        # create local gene to node mapping
        local_gene_to_node = {gene:i for i, gene in enumerate(cell.index)}

        # Subset network to only include genes in the cell
        network_cell = network.df.copy()
        network_cell = network_cell[
            network_cell[network.reg_name].isin(cell.index) & 
            network_cell[network.tar_name].isin(cell.index)
        ]

        edge_index = torch.tensor(np.array([
            network_cell[network.reg_name].map(local_gene_to_node).values, 
            network_cell[network.tar_name].map(local_gene_to_node).values
        ]))

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
