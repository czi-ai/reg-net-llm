from tqdm import tqdm
from functools import partial
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from scGraphLLM.data import GraphTransformerDataset, scglm_collate_fn, send_to_gpu
from scGraphLLM.benchmark import send_to_gpu
from scGraphLLM._globals import *
from scGraphLLM.network import RegulatoryNetwork
from scGraphLLM.tokenize import GraphTokenizer


class InferenceDataset(GraphTransformerDataset):
    """
    Dataset class for performing inference with a fixed regulatory network.

    Each cell's expression profile is tokenized using the provided tokenizer.
    This dataset is used to generate embeddings using a pre-trained model.

    Args:
        expression (pd.DataFrame): Gene expression matrix (cells x genes).
        tokenizer (GraphTokenizer): Tokenizer to tokenize each cell into and compute its edge index.
        cache_dir (str, optional): Directory to cache tokenized data.
    """
    def __init__(self, expression: pd.DataFrame, tokenizer: GraphTokenizer, cache_dir=None):
        self.tokenizer = tokenizer
        self.obs_names = expression.index
        self.common_genes = sorted(list(set(self.gene_to_node.keys()) & set(expression.columns)))
        self.expression = expression[self.common_genes]
        super().__init__(cache_dir=cache_dir, mask_fraction=0.0)
    
    @property
    def gene_to_node(self):
        return self.tokenizer.gene_to_node

    @property
    def node_to_gene(self):
        return self.tokenizer.node_to_gene
    
    @property
    def network(self):
        return self.tokenizer.network
    
    def __len__(self):
        return len(self.expression)

    def __getitem__(self, idx):
        """
        Returns a tokenized representation of the cell at the given index.

        Args:
            idx (int): Index of the cell.

        Returns:
            dict: Tokenized representation including gene tokens and graph information.
        """
        cell = self.expression.iloc[idx]
        data = self.tokenizer(cell)
        item = self._item_from_tokenized_data(data)
        item["obs_name"] = self.obs_names[idx]
        return item


class VariableNetworksInferenceDataset(InferenceDataset):
    """
    Dataset class for performing inference using cell-specific regulatory networks.

    Each cell is assigned a custom graph (edges and optional weights),
    allowing dynamic modeling of cell-specific regulatory structures.

    Args:
        edge_ids_list (List[np.ndarray]): List of edge indices (one per cell) into `all_edges`.
        all_edges (np.ndarray): Full array of all possible (regulator, target) edges.
        weights_list (List[np.ndarray], optional): List of edge weights per cell.
        limit_regulon (int, optional): Limit number of regulators per target gene.
        limit_graph (int, optional): Limit the total number of edges in the graph.
        **kwargs: Additional arguments passed to `InferenceDataset`.
    """
    def __init__(
            self, 
            edge_ids_list,
            all_edges,
            weights_list=None,
            limit_regulon=None,
            limit_graph=None,
            **kwargs
        ):
        super().__init__(**kwargs)
        self.edge_ids_list = edge_ids_list
        self.all_edges = np.array(all_edges)
        self.weights_list = weights_list
        self.limit_regulon = limit_regulon
        self.limit_graph = limit_graph
    
    @property
    def prune_graph(self):
        return not(self.limit_regulon is None and self.limit_graph is None)

    def __getitem__(self, idx):
        cell = self.expression.iloc[idx]
        edge_ids = self.edge_ids_list[idx]
        edges = self.all_edges[edge_ids]
        regulators, targets = zip(*edges)
        weights = self.weights_list[idx] if self.weights_list else None
        cell_network = RegulatoryNetwork(regulators=regulators, targets=targets, weights=weights, likelihoods=None)

        if self.prune_graph:
            cell_network.prune(limit_regulon=self.limit_regulon, limit_graph=self.limit_graph, inplace=True)
        
        data = self.tokenizer(cell, cell_network)
        item = self._item_from_tokenized_data(data)
        item["obs_name"] = self.obs_names[idx]

        return item


def get_cell_embeddings(dataset, model):
    """
    Runs the model on the input dataset and computes pooled cell-level embeddings.

    The embeddings are obtained via mean-pooling across the node (gene) dimension.

    Args:
        dataset (Dataset): An instance of `InferenceDataset` or `VariableNetworksInferenceDataset`.
        model (torch.nn.Module): A trained model that returns sequence embeddings.

    Returns:
        pd.DataFrame: DataFrame of pooled embeddings with cell names as index.
    """
    dataloader = DataLoader(
        dataset, 
        batch_size=256, 
        shuffle=False, 
        collate_fn=partial(scglm_collate_fn, inference=True)
    )

    x_list = []
    obs_names_list = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Forward Pass"):
            seq_lengths = torch.tensor(batch["num_nodes"]).to('cuda')
            obs_names = batch["obs_name"]
            x = model(send_to_gpu(batch))[0]
            # apply mean pooling along gene dimesion, respecting sequence length
            mask = torch.arange(x.shape[1], device=x.device).unsqueeze(0) < seq_lengths.unsqueeze(1)
            mask = mask.unsqueeze(-1).float()
            x_masked = x * mask
            x_pooled = x_masked.sum(dim=1) / mask.sum(dim=1)

            x_list.append(x_pooled.cpu().numpy())
            obs_names_list.append(obs_names)

    x = np.vstack(x_list)
    obs_names = np.concatenate(obs_names_list)
    embeddings = pd.DataFrame(x, index=obs_names)

    return embeddings
