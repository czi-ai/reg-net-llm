import os
os.environ["TORCH_USE_CUDA_DSA"] = "1"  # Enable device-side assertions
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Make CUDA errors more visible

import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.utils.data import Dataset, DataLoader
from anndata import AnnData
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from scGraphLLM.GNN_modules import *
from scGraphLLM.MLP_modules import *
from scGraphLLM.preprocess import rank
import lightning.pytorch as pl
from scGraphLLM._globals import * ## these define the indices for the special tokens 
from torch_geometric.utils import negative_sampling
from scGraphLLM.models import GDTransformer
from scGraphLLM.config import *
from scGraphLLM.data import *
warnings.filterwarnings("ignore")

from scGraphLLM.benchmark import send_to_gpu



scglm_rootdir = dirname(dirname(abspath(importlib.util.find_spec("scGraphLLM").origin)))
gene_names_map = pd.read_csv(join(scglm_rootdir, "data/gene-name-map.csv"), index_col=0)
ensg2hugo = gene_names_map.set_index("ensg.values")["hugo.values"].to_dict()
hugo2ensg = gene_names_map.set_index("hugo.values")["ensg.values"].to_dict()
ensg2hugo_vectorized = np.vectorize(ensg2hugo.get)
hugo2ensg_vectorized = np.vectorize(hugo2ensg.get)

REG_VALS = "regulator.values"
TAR_VALS = "target.values"

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--aracne_dir", type=str, required=True)
parser.add_argument("--gene_index_path", type=str, required=True)
parser.add_argument("--sample_n_cells", type=int, default=None)
args = parser.parse_args()

from torch.nn.utils.rnn import pad_sequence

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

def run_save_(network, ranks, global_gene_to_node, cache_dir, overwrite, msplit, valsg_split_ratio, cell_type, min_genes_per_graph=MIN_GENES_PER_GRAPH, skipped=0, ncells=0, verbose=False):
    
    os.makedirs(join(cache_dir, msplit), exist_ok=True)

    ranks = ranks + 2 # keep only genes in the network, and offset the ranks by 2 to account for the special tokens, so 2 now corresponds to rank 0(ZERO_IDX)
    network_genes = list(set(network["regulator.values"].to_list() + network["target.values"].to_list()))
    common_genes = list(set(network_genes).intersection(set(ranks.columns)))
    
    ranks = ranks.loc[:, common_genes]
    for i in range(ranks.shape[0]):
        if ncells % 1000 == 0:
            print(f"Processed {ncells} cells", end="\r")
        cell_number = ranks.index[i]
        
        if msplit == "valSG":
            rand = rng.random()
            if rand > valsg_split_ratio:
                split = "train"
            else:
                split = msplit
        else:
            split = msplit
        
        outfile = f"{cache_dir}/{split}/{cell_type}_{cell_number}.pt"
        if (os.path.exists(outfile)) and (not overwrite):
            ncells+=1
            continue
        
        cell = ranks.iloc[i, :]
        cell = cell[cell != ZERO_IDX] + NUM_GENES ## offset the ranks by global number of genes -  this lets the same 
        # VS:
        # keep graph static across batches 
        # cell = cell + NUM_GENES
        # nn.Embedding be used for both gene and rank embeddings
        if cell.shape[0] < min_genes_per_graph: # require a minimum number of genes per cell 
            skipped += 1
            ncells+=1
            continue

        # Subset network to only include genes in the cell
        network_cell = network[
            network["regulator.values"].isin(cell.index) & 
            network["target.values"].isin(cell.index)
        ]

        local_gene_to_node_index = {gene:i for i, gene in enumerate(cell.index)}
        # local_gene_to_node_index = global_gene_to_node
        # each cell graph is disjoint from each other in terms of the relative position of nodes and edges
        # so edge index is local to each graph for each cell.
        # cell.index defines the order of the nodes in the graph
        with warnings.catch_warnings(): # Suppress annoying pandas warnings
            warnings.simplefilter("ignore") 
            edges = network_cell[['regulator.values', 'target.values', 'mi.values']]
            edges['regulator.values'] = edges['regulator.values'].map(local_gene_to_node_index)
            edges['target.values'] = edges['target.values'].map(local_gene_to_node_index)

        edge_list = torch.tensor(np.array(edges[['regulator.values', 'target.values']])).T
        edge_weights = torch.tensor(np.array(edges['mi.values']))
        node_indices = torch.tensor(np.array([(global_gene_to_node[gene], cell[gene]) for gene in cell.index]), dtype=torch.long)
        data = Data(
            x=node_indices, 
            edge_index=edge_list, 
            edge_weight=edge_weights
        )
        
        torch.save(data, outfile)
        ncells += 1
        
        if verbose:
            try:
                torch.load(outfile)
                print(outfile)
            except:
                print(outfile, "-------- Failed")
        
    return (skipped, ncells)



def main(args):
    
    adata = sc.read_h5ad(args.cells_path)
    global_gene_to_node = pd.read_csv(args.gene_index_path).set_index("gene_name")["idx"].to_dict()
    network = pd.read_csv(join(args.aracne_dir, "consolidated-net_defaultid.tsv"), sep="\t")

    if args.sample_n_cells is not None and adata.n_obs > args.sample_n_cells:
        sc.pp.subsample(adata, n_obs=args.sample_n_cells, random_state=12345, copy=False)
 
    # ranks = pd.read_csv(join(args.data_dir, "rank_raw.csv"))
    ranks, _ = rank(adata, n_bins=250, rank_by_z_score=False)
    
    run_save_(
        network=network, 
        ranks=ranks, 
        global_gene_to_node=global_gene_to_node, 
        cache_dir=args.cache_dir,
        overwrite=True, 
        msplit="all", 
        valsg_split_ratio=None, 
        cell_type="cell", 
        min_genes_per_graph=-1, 
        skipped=0, 
        ncells=0
    )

    dataset = GraphTransformerDataset(
        cache_dir=args.all_data_dir,
        dataset_name="cells",
        debug=False,
        mask_fraction=0
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Load model
    model = GDTransformer.load_from_checkpoint(args.model_path, config=graph_kernel_attn_manitou)
    
    # Get embeddings
    model.eval()

    embedding_list = []
    edges_list = []
    seq_lengths = []
    with torch.no_grad():
        for batch in dataloader:
            embedding, target_gene_ids, target_rank_ids, mask_locs, edge_index_list, num_nodes_list = model(send_to_gpu(batch))
            embedding_list.append(embedding.cpu().numpy())
            edges_list.append(edge_index_list.cpu().numpy())
            seq_lengths.append(batch["num_nodes"].cpu().numpy())

    seq_lengths = np.concatenate(seq_lengths, axis=0)
    max_seq_length = max(seq_lengths)
    embeddings = np.concatenate([
        np.pad(emb, pad_width=((0, 0), (0, max_seq_length - emb.shape[1]), (0, 0)), 
                mode="constant", constant_values=0)
        for emb in embedding_list
    ], axis=0)
    
    edges = {}
    i = 0
    for lst in edges_list:
        for e in lst:
            edges[i] = e
            i += 1

    np.savez(
        file=join(args.out_dir, "embedding.npz"), 
        x=embeddings,
        seq_lengths=seq_lengths,
        edges=edges, 
        allow_pickle=True
    )



if __name__ == "__main__":

    args.cells_path = join(args.data_dir, "cells.h5ad")
    args.ranks_path = join(args.data_dir, "rank_raw.csv")
    args.out_dir = join(args.data_dir, "embeddings/scglm")
    args.cache_dir = join(args.out_dir, "cache")
    args.all_data_dir = join(args.cache_dir, "all")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.all_data_dir, exist_ok=True)

    main(args)






    