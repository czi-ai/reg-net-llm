import os
os.environ["TORCH_USE_CUDA_DSA"] = "1"  # Enable device-side assertions
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Make CUDA errors more visible

import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.nn.utils.rnn import pad_sequence
from anndata import AnnData
import torch
from torch.utils.data import DataLoader, SequentialSampler
import os
import sys
import importlib
import json
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Union
from os.path import join, dirname, abspath
import warnings
warnings.filterwarnings("ignore")

from scGraphLLM._globals import * ## these define the indices for the special tokens 
from scGraphLLM.models import GDTransformer
from scGraphLLM.preprocess import quantize_cells
from scGraphLLM.benchmark import send_to_gpu, random_edge_mask
from scGraphLLM.config import *
from scGraphLLM.data import *
from scGraphLLM._globals import NUM_BINS
from utils import mask_values, get_locally_indexed_edges, get_locally_indexed_masks_expressions, save_embedding

scglm_rootdir = dirname(dirname(abspath(importlib.util.find_spec("scGraphLLM").origin)))
gene_names_map = pd.read_csv(join(scglm_rootdir, "data/gene-name-map.csv"), index_col=0)
ensg2hugo = gene_names_map.set_index("ensg.values")["hugo.values"].to_dict()
hugo2ensg = gene_names_map.set_index("hugo.values")["ensg.values"].to_dict()
ensg2hugo_vectorized = np.vectorize(ensg2hugo.get)
hugo2ensg_vectorized = np.vectorize(hugo2ensg.get)

REG_VALS = "regulator.values"
TAR_VALS = "target.values"
MI_VALS = "mi.values"

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--aracne_dir", type=str, required=True)
parser.add_argument("--max_seq_length", type=int, default=2048)
parser.add_argument("--use_masked_edges", action="store_true")
parser.add_argument("--mask_ratio", type=float, default=0.15)
parser.add_argument("--mask_fraction", type=float, default=None)
parser.add_argument("--mask_value", type=float, default=1e-4)
parser.add_argument("--retain_obs_vars", nargs="+", default=[])
parser.add_argument("--gene_index_path", type=str, required=True)
parser.add_argument("--sample_n_cells", type=int, default=None)
parser.add_argument("--cache", action="store_true")
args = parser.parse_args()


def collate_fn(batch):
    data = {
        "orig_gene_id" : [], 
        "orig_rank_indices" : [], 
        "gene_mask" : [], 
        "rank_mask" : [], 
        "both_mask" : [], 
        "edge_index": [], 
        "num_nodes" :[], 
        # "spectral_pe" : [], 
        "dataset_name" : [] 
    }
    
    # Make a dictionary of lists from the list of dictionaries
    for b in batch:
        for key in data.keys():
            data[key].append(b[key])

    # Pad these dictionaries of lists
    for key in data.keys():
        if (key != "dataset_name") & (key != "edge_index") & (key != "num_nodes"):
            data[key] = pad_sequence(data[key], batch_first=True).squeeze()

    return data

def get_edges_dict(edges_list):
    edges = {}
    i = 0
    for lst in edges_list:
        for e in lst:
            edges[i] = e
            i += 1
    return edges

def main(args):
    print("Loading Data...")
    adata = sc.read_h5ad(args.cells_path)
    global_gene_df = pd.read_csv(args.gene_index_path)
    global_gene_to_node = global_gene_df.set_index("gene_name")["idx"].to_dict()
    global_node_to_gene = global_gene_df.set_index("idx")["gene_name"].to_dict()
    network = pd.read_csv(join(args.aracne_dir, "consolidated-net_defaultid.tsv"), sep="\t")

    if args.sample_n_cells is not None and adata.n_obs > args.sample_n_cells:
        sc.pp.subsample(adata, n_obs=args.sample_n_cells, random_state=12345, copy=False)

    adata_original = adata.copy()
    if args.mask_fraction is not None:
        adata_original = adata.copy()
        X_masked, masked_indices = mask_values(adata.X.astype(float), mask_prob=args.mask_fraction, mask_value=args.mask_value)
        adata.X = X_masked
    
    ranks = quantize_cells(adata.to_df(), n_bins=NUM_BINS, method="quantile")
    run_cache(
        network=network, 
        expression=ranks, 
        global_gene_to_node=global_gene_to_node, 
        cache_dir=args.cache_dir,
        overwrite=True, 
        msplit="all", 
        valsg_split_ratio=None, 
        cell_type="cell",
        max_seq_length=args.max_seq_length,
        only_expressed_genes=True,
        min_genes_per_graph=-1, 
        skipped=0, 
        ncells=0
    )

    dataset = GraphTransformerDataset(
        cache_dir=args.all_data_dir,
        dataset_name="cells",
        debug=False,
        mask_fraction=0.0
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Load model
    model: GDTransformer = GDTransformer.load_from_checkpoint(args.model_path, config=graph_kernel_attn_manitou)
    
    # Get embeddings
    print(f"Performing forward pass...")
    model.eval()

    embedding_list = []
    edges_list = []
    masked_edges_list = []
    non_masked_edges_list = []
    seq_lengths = []
    input_gene_ids_list = []
    with torch.no_grad():
        for batch in dataloader:
            if args.use_masked_edges:
                edges_list.append([e.cpu().numpy() for e in batch["edge_index"]])
                # idendify edges to mask for each cell
                # random edge mask returns a tuple (non_masked_edge_index, masked_edge_index)
                random_edge_masks = [random_edge_mask(edge_index, mask_ratio=args.mask_ratio) for edge_index in batch["edge_index"]]
                non_masked_edge_inices = [edge_mask[0] for edge_mask in random_edge_masks]
                masked_edge_indices = [edge_mask[1] for edge_mask in random_edge_masks]
                masked_edges_list.append(masked_edge_indices)
                non_masked_edges_list.append(non_masked_edge_inices)
                batch["edge_index"] = non_masked_edge_inices
                embedding, target_gene_ids, target_rank_ids, mask_locs, edge_index_list, num_nodes_list = model(send_to_gpu(batch))
            else:
                embedding, target_gene_ids, target_rank_ids, mask_locs, edge_index_list, num_nodes_list = model(send_to_gpu(batch))
                edges_list.append([e.cpu().numpy() for e in edge_index_list])
            input_gene_ids_list.append(target_gene_ids.cpu().numpy())
            embedding_list.append(embedding.cpu().numpy())
            seq_lengths.append(batch["num_nodes"])

    seq_lengths = np.concatenate(seq_lengths, axis=0)
    max_seq_length = max(seq_lengths)
    edges = get_edges_dict(edges_list)
    embeddings = np.concatenate([
        np.pad(emb, pad_width=((0, 0), (0, max_seq_length - emb.shape[1]), (0, 0)), 
               mode="constant", constant_values=0)
        for emb in embedding_list
    ], axis=0) 
    input_gene_ids = np.concatenate([
        np.pad(gene_ids, pad_width=((0, 0), (0, max_seq_length - gene_ids.shape[1])), 
               mode="constant", constant_values=global_gene_to_node['<PAD>'])
        for gene_ids in input_gene_ids_list
    ], axis=0)
    input_genes = np.vectorize(global_node_to_gene.get)(input_gene_ids)
    # get original expression
    expression = np.concatenate([
        np.pad(adata_original[i, genes[:seq_lengths[i]]].X.toarray(), 
               pad_width=((0,0), (0, max_seq_length - seq_lengths[i])), 
               mode="constant", constant_values=0)
        for i, genes in enumerate(input_genes)
    ], axis=0)

    # retain requested metadata
    metadata = {}
    for var in args.retain_obs_vars:
        try:
            metadata[var] = adata.obs[var]
        except KeyError:
            print(f"Key {var} not in observational metadata...")

    print("Saving emeddings...")
    # TODO: 1. Write embedding for each cell to embedding cache directory
    if args.use_masked_edges:
        masked_edges = get_edges_dict(masked_edges_list)
        non_masked_edges = get_edges_dict(non_masked_edges_list)
        save_embedding(
            file=args.emb_path,
            cache=args.cache,
            cache_dir=args.emb_cache,
            x=embeddings,
            seq_lengths=seq_lengths,
            edges=edges,
            masked_edges=masked_edges,
            non_masked_edges=non_masked_edges,
            metadata=metadata
        )
        return
    elif args.mask_fraction is None:  
        save_embedding(
            file=args.emb_path,
            cache=args.cache,
            cache_dir=args.emb_cache,
            x=embeddings,
            expression=expression,
            seq_lengths=seq_lengths,
            edges=edges,
            metadata=metadata
        )
        return
    
    masks, masked_expressions = get_locally_indexed_masks_expressions(adata_original, masked_indices, input_genes)
    save_embedding(
        file=args.emb_path,
        cache=args.cache,
        cache_dir=args.emb_cache,
        x=embeddings,
        seq_lengths=seq_lengths,
        edges=edges,
        masks=masks,
        masked_expressions=masked_expressions,
        metadata=metadata
    )


if __name__ == "__main__":
    print("Starting main execution...")
    args.cells_path = join(args.data_dir, "cells.h5ad")
    args.ranks_path = join(args.data_dir, "rank_raw.csv")
    args.emb_path = join(args.out_dir, "embedding.npz")
    args.emb_cache = join(args.out_dir, "cached_embeddings")
    args.cache_dir = join(args.out_dir, "cache")
    args.emb_cache_dir = join(args.out_dir, "emb_cache")
    args.all_data_dir = join(args.cache_dir, "all")
    
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.all_data_dir, exist_ok=True)
    if args.cache:
        os.makedirs(args.emb_cache, exist_ok=True)

    main(args)






    