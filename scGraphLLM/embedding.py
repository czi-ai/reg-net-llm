import os 
import glob
from collections import defaultdict
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    """
    A PyTorch Dataset for loading graph embeddings with optional expression and metadata support.

    Supports two modes:
    - Cache mode: reads `.pt` files from directories containing precomputed embeddings.
    - NPZ mode: reads `.npz` files containing batched embedding arrays.

    Parameters:
        paths (list of str): Paths to either directories (cache mode) or `.npz` files.
        with_expression (bool): Whether to include expression data in the dataset.
        with_metadata (bool): Whether to include metadata in the dataset.
        target_metadata_key (str or None): Metadata key used for label extraction and classification.
        label_encoder (LabelEncoder or None): Encoder for class labels. If None, a new one is created.
    """
    def __init__(self, paths, with_expression=False, with_metadata=False, target_metadata_key=None, label_encoder=None):
        self.with_expression = with_expression
        self.target_metadata_key = target_metadata_key
        self.with_metadata = True if self.target_metadata_key is not None else with_metadata
        self.label_encoder = label_encoder
        self.cache_mode = False
        self.samples = []

        # Determine if paths are cache directories or .npz files
        if all(os.path.isdir(p) for p in paths):
            self.cache_mode = True
            self._load_cache_dirs(paths)
        elif all(os.path.isfile(p) and p.endswith('.npz') for p in paths):
            self._load_npz_files(paths)
        else:
            raise ValueError("All paths must be either directories (for cache mode) or .npz files.")

    def _load_cache_dirs(self, dirs):
        """
        Load embeddings and metadata from cached `.pt` files in the specified directories.

        Parameters:
            dirs (list of str): List of directory paths containing cached `.pt` embedding files.
        """
        for dir_path in dirs:
            pt_files = sorted(glob.glob(os.path.join(dir_path, 'emb_*.pt')))
            for pt_file in pt_files:
                self.samples.append(pt_file)

        # Load metadata if required
        self.max_seq_length = -1
        if self.target_metadata_key is not None or self.with_metadata:
            self.metadata = defaultdict(list)
            for pt_file in self.samples:
                data = torch.load(pt_file)
                meta = data.get('metadata', {})
                for key, value in meta.items():
                    self.metadata[key].append(value)
                if data["seq_lengths"] > self.max_seq_length:
                    self.max_seq_length = data["seq_lengths"]
                self.metadata["seq_lengths"].append(data["seq_lengths"])

            print(f"Loaded metadata with keys: {self.metadata.keys()}")
            if self.target_metadata_key is not None:
                self.encode_labels()
            
    def _load_npz_files(self, paths):
        """
        Load embeddings from `.npz` files and prepare tensors for usage.

        Parameters:
            paths (list of str): List of `.npz` file paths.
        """
        embeddings = [np.load(p, allow_pickle=True) for p in paths]
        self.seq_lengths = np.concatenate([emb["seq_lengths"] for emb in embeddings], axis=0)
        self.max_seq_length = np.max(self.seq_lengths)
        self.edges = self.aggregate_embedding_dicts(embeddings)
        self.x = np.concatenate([
            np.pad(emb["x"], pad_width=((0, 0), (0, self.max_seq_length - emb["x"].shape[1]), (0, 0)),
                   mode="constant", constant_values=0)
            for emb in embeddings
        ], axis=0)

        if self.with_expression:
            self.expression = np.concatenate([
                np.pad(emb["expression"], pad_width=((0, 0), (0, self.max_seq_length - emb["expression"].shape[1])),
                       mode="constant", constant_values=0)
                for emb in embeddings
            ], axis=0)

        if self.with_metadata:
            metadata = defaultdict(list)
            for emb in embeddings:
                meta = emb["metadata"].item()
                for key, value in meta.items():
                    assert len(value) == emb["x"].shape[0]
                    metadata[key].extend(value)
                del meta
            self.metadata = metadata

            if self.target_metadata_key is not None:
                self.encode_labels()
        
        return embeddings

    def encode_labels(self):
        """
        Encode class labels from metadata using the label encoder.
        Computes class counts and class weights.
        """
        assert self.target_metadata_key in self.metadata, f"Target key {self.target_metadata_key} not in metadata"
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.y = self.label_encoder.fit_transform(self.metadata[self.target_metadata_key])
        elif isinstance(self.label_encoder, LabelEncoder):
            self.y = self.label_encoder.transform(self.metadata[self.target_metadata_key])
        else:
            raise ValueError("label_encoder must be None or a LabelEncoder instance")
        
        self.class_counts = pd.Series(self.metadata[self.target_metadata_key]).value_counts()[self.label_encoder.classes_]
        class_weights = 1.0 / self.class_counts
        self.class_weights = class_weights / class_weights.sum()
      
    def aggregate_embedding_dicts(self, embeddings, key="edges"):
        """
        Combine dictionaries from multiple `.npz` files into one continuous dictionary.

        Parameters:
            embeddings (list): List of loaded `.npz` files.
            key (str): Key in `.npz` files containing a dictionary to merge.

        Returns:
            dict: Aggregated dictionary of embeddings.
        """
        res = {}
        i = 0
        for emb in embeddings:
            d = emb[key].item()
            for j, e in d.items():
                res[i + j] = e
            i += len(d)
        return res

    @property
    def embedding_dim(self):
        """
        Returns:
            int: Dimensionality of the embedding vectors.
        """
        if self.cache_mode:
            sample = torch.load(self.samples[0])
            return sample['x'].shape[1]
        else:
            return self.x.shape[2]

    @property
    def shape(self):
        """
        Returns:
            tuple: Shape of the dataset (num_samples, max_seq_length, embedding_dim).
        """
        return (int(len(self)), int(self.max_seq_length), int(self.embedding_dim))

    @property
    def num_classes(self):
        """
        Returns:
            int or None: Number of classes if label encoder is defined, else None.
        """
        if self.label_encoder is None:
            return None
        return len(self.label_encoder.classes_)

    def __len__(self):
        if self.cache_mode:
            return len(self.samples)
        else:
            return len(self.x)

    def __getitem__(self, idx):
        if self.cache_mode:
            item, data = self._get_cached_item(idx)
            return item
        else:
            return self._get_item(idx)

    def _get_item(self, idx):
        """
        Get a sample from in-memory `.npz`-based dataset.

        Parameters:
            idx (int): Sample index.

        Returns:
            dict: Dictionary with keys: x, seq_lengths, edges, and optionally expression, metadata, y.
        """
        item = {
            "x": torch.tensor(self.x[idx]),
            "seq_lengths": torch.tensor(self.seq_lengths[idx]),
            "edges": torch.tensor(self.edges[idx])
        }

        if self.with_expression:
            item["expression"] = torch.tensor(self.expression[idx])

        if self.with_metadata:
            item["metadata"] = {k: self.metadata[k][idx] for k in self.metadata}

        if self.target_metadata_key is not None:
            item["y"] = torch.tensor(self.y[idx], dtype=torch.long)

        return item

    def _get_cached_item(self, idx):
        """
        Get a sample from `.pt` cache-based dataset.

        Parameters:
            idx (int): Sample index.

        Returns:
            tuple: (item_dict, raw_loaded_data_dict)
        """
        data = torch.load(self.samples[idx])
        item = {
            "x": torch.tensor(data["x"]),
            "seq_lengths": torch.tensor(data["seq_lengths"]),
            "edges": torch.tensor(data["edges"])
        }

        if self.with_expression and "expression" in data:
            item["expression"] = torch.tensor(data["expression"])

        if self.with_metadata and "metadata" in data:
            item["metadata"] = data["metadata"]

        if self.target_metadata_key is not None:
            item["y"] = torch.tensor(self.y[idx], dtype=torch.long)

        return item, data
    

class EmbeddingDatasetWithEdgeMasks(EmbeddingDataset):
    """
    Load `.npz` files and extract precomputed edge masks if not generating masks dynamically.

    Parameters:
        paths (list of str): List of `.npz` file paths.

    Returns:
        list: List of loaded embedding dictionaries.
    """
    def __init__(self, paths, mask_ratio=0.15, generate_edge_masks=False, **kwargs):
        self.mask_ratio = mask_ratio
        self.generate_edge_masks = generate_edge_masks
        self.mask_ratio = mask_ratio
        super().__init__(paths, **kwargs)
    
    def _load_npz_files(self, paths):
        embeddings = super()._load_npz_files(self, paths)
        if not self.generate_edge_masks:
            self.masked_edges = self.aggregate_embedding_dicts(embeddings, key="masked_edges")
            self.non_masked_edges = self.aggregate_embedding_dicts(embeddings, key="non_masked_edges")
            
            mean_perc_masked_edges = np.mean([
                self.masked_edges[k].shape[1] / self.edges[k].shape[1]
                for k in self.edges.keys()
            ])
            print(f"Mean percentage masked edges {mean_perc_masked_edges:.3f}")

            mean_perc_non_masked_edges = np.mean([
                self.non_masked_edges[k].shape[1] / self.edges[k].shape[1]
                for k in self.edges.keys()
            ])
            print(f"Mean percentage non masked edges {mean_perc_non_masked_edges:.3f}") 
        
        return embeddings

    def _get_item(self, idx):
        item = super()._get_item(idx)
        if self.generate_edge_masks:
            non_masked_edges, masked_edges = random_edge_mask(item["edges"], self.mask_ratio)
            item["masked_edges"] = masked_edges
            item["non_masked_edges"] = non_masked_edges
        else:
            item["masked_edges"] = torch.tensor(self.masked_edges[idx])
            item["non_masked_edges"] = torch.tensor(self.non_masked_edges[idx])
        return item

    def _get_cached_item(self, idx):
        item, data = super()._get_cached_item(idx)
        if self.generate_edge_masks:
            non_masked_edges, masked_edges = random_edge_mask(item["edges"], self.mask_ratio)
            item["masked_edges"] = masked_edges
            item["non_masked_edges"] = non_masked_edges
        else:
            item["masked_edges"] = data["masked_edges"]
            item["non_masked_edges"] = data["non_masked_edges"]
        return item, data
    

class EmbeddingDatasetWithGeneMasks(EmbeddingDataset):
    """
    Extension of EmbeddingDataset that includes gene masking for masked gene expression modeling.

    This class loads per-sample gene masks and the corresponding masked expression values,
    typically used in pretraining or inference tasks where some genes are masked out and the
    model must infer or reconstruct their expression.

    Parameters:
        paths (list of str): List of paths to `.npz` files containing embeddings with gene masks.
        **kwargs: Additional arguments passed to the base EmbeddingDataset.
    """
    def __init__(self, paths, **kwargs):
        super().__init__(paths, **kwargs)
        embedding = [np.load(path, allow_pickle=True) for path in self.paths]
        self.masked_genes = self.aggregate_embedding_dicts(embedding, key="masks")
        self.expression = self.aggregate_embedding_dicts(embedding, key="masked_expressions")
    
    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        item["masked_genes"] = torch.tensor(self.masked_genes[idx])
        item["masked_expression"] = torch.tensor(self.expression[idx])
        return item


def embedding_collate_fn(
    batch: List[Dict[str, Any]],
    expression: bool = False,
    metadata: bool = False,
    target_label: bool = False,
    masked_genes: bool = False,
    masked_edges: bool = False
) -> Dict[str, Any]:
    """
    Custom collate function to batch samples from EmbeddingDataset variants.

    Parameters:
        batch (List[Dict]): A list of sample dictionaries returned by the dataset.
        expression (bool): Whether to include 'expression' field in output.
        metadata (bool): Whether to include 'metadata' field in output.
        target_label (bool): Whether to include target labels 'y' in output.
        masked_genes (bool): Whether to include masked gene indices and expression values.
        masked_edges (bool): Whether to include masked and non-masked edges.

    Returns:
        Dict[str, Any]: A dictionary with batched and/or padded tensors and lists,
                        depending on the flags provided.
    """
    collated = {
        "x": pad_sequence([item["x"] for item in batch], batch_first=True, padding_value=0),
        "seq_lengths": torch.tensor([item["seq_lengths"] for item in batch]),
        "edges": [item["edges"] for item in batch]
    }

    if expression:
        collated["expression"] = pad_sequence([item["expression"] for item in batch], batch_first=True, padding_value=0)
    
    if metadata:
        collated["metadata"] = {k: [item["metadata"][k] for item in batch] for k in batch[0]["metadata"].keys()}

    if target_label:
        collated["y"] = torch.stack([item["y"] for item in batch])
    
    if masked_genes:
        collated["masked_genes"] = [item["masked_genes"] for item in batch]
        collated["masked_expression"] = [item["masked_expression"] for item in batch]

    if masked_edges:
        collated["masked_edges"] = [item["masked_edges"] for item in batch]
        collated["non_masked_edges"] = [item["non_masked_edges"] for item in batch]

    return collated


def random_edge_mask(edge_index, mask_ratio=0.15):
    """
    Randomly masks a subset of unique undirected edges in the edge_index tensor.

    Edges are treated as undirected by grouping pairs (u,v) and (v,u) together.
    A fraction `mask_ratio` of these unique edge pairs are randomly selected and masked.

    Parameters
    ----------
    edge_index : torch.LongTensor [2, E]
        Tensor containing edge indices where E is the number of edges.
    mask_ratio : float, default=0.15
        Fraction of unique undirected edges to mask.

    Returns
    -------
    non_masked_edge_index : torch.LongTensor [2, E_non_masked]
        Edge index tensor containing edges not masked.
    masked_edge_index : torch.LongTensor [2, E_masked]
        Edge index tensor containing the masked edges.
    """
    device = edge_index.device
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    E = edge_index.shape[1]
    
    pairs = {}
    for i in range(E):
        u, v = src[i], dst[i]
        mn, mx = (u, v) if u <= v else (v, u)
        if (mn, mx) not in pairs:
            pairs[(mn, mx)] = []
        pairs[(mn, mx)].append(i)
    
    unique_pairs = list(pairs.keys())  
    num_unique = len(unique_pairs)
    num_mask = int(mask_ratio * num_unique)
    
    perm = torch.randperm(num_unique, device=device)
    masked_pairs = [unique_pairs[i.item()] for i in perm[:num_mask]]
    masked_indices = []

    for pair in masked_pairs:
        masked_indices.extend(pairs[pair]) 

    masked_indices = torch.tensor(masked_indices, device=device, dtype=torch.long)
    # get non_masked indices
    all_indices = torch.arange(E, device=device)
    non_masked_indices = all_indices[~torch.isin(all_indices, masked_indices)]

    masked_edge_index = edge_index[:,masked_indices]
    non_masked_edge_index = edge_index[:,non_masked_indices]

    return non_masked_edge_index, masked_edge_index