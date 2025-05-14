print("This is the top of the script...")
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
import gc
import importlib
import shutil
import json
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Union
from os.path import join, dirname, abspath
from functools import partial
import warnings
warnings.filterwarnings("ignore")

from scGraphLLM._globals import * ## these define the indices for the special tokens 
from scGraphLLM.models import GDTransformer
from scGraphLLM.preprocess import quantize_cells
from scGraphLLM.benchmark import send_to_gpu, random_edge_mask
from scGraphLLM.config import *
from scGraphLLM.data import *
from scGraphLLM.eval_config import *
from scGraphLLM._globals import NUM_BINS
from utils import (
    mask_values, 
    get_locally_indexed_edges, 
    get_locally_indexed_masks_expressions, 
    save_embedding,
    collect_metadata
)

scglm_rootdir = dirname(dirname(abspath(importlib.util.find_spec("scGraphLLM").origin)))
gene_names_map = pd.read_csv(join(scglm_rootdir, "data/gene-name-map.csv"), index_col=0)
ensg2hugo = gene_names_map.set_index("ensg.values")["hugo.values"].to_dict()
hugo2ensg = gene_names_map.set_index("hugo.values")["ensg.values"].to_dict()
ensg2hugo_vectorized = np.vectorize(ensg2hugo.get)
hugo2ensg_vectorized = np.vectorize(hugo2ensg.get)

REG_VALS = "regulator.values"
TAR_VALS = "target.values"
MI_VALS = "mi.values"
LOGP_VALS = "log.p.values"
SCC_VALS = "scc.values"

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default=None)
parser.add_argument("--cells_path", type=str, default=None)
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--model_path", type=str, required=True)
# parser.add_argument("--aracne_dir", type=str, required=True)
parser.add_argument("--network_path", type=str)
parser.add_argument("--infer_network", action="store_true")
parser.add_argument("--infer_network_alpha", type=float, default=0.25)
parser.add_argument("--networks", type=str, default=None)
parser.add_argument("--max_seq_length", type=int, default=2048)
parser.add_argument("--use_masked_edges", action="store_true")
parser.add_argument("--mask_ratio", type=float, default=0.15)
parser.add_argument("--mask_fraction", type=float, default=None)
parser.add_argument("--mask_value", type=float, default=1e-4)
parser.add_argument("--retain_obs_vars", nargs="+", default=[])
parser.add_argument("--gene_index_path", type=str, required=True)
parser.add_argument("--sample_n_cells", type=int, default=None)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--skip_preprocess", action="store_true")
parser.add_argument("--cache", action="store_true")

try:
    args = parser.parse_args()
except Exception as e:
    print(f"Error in parsing arguments: {e}")
    sys.exit(1)



def get_edges_dict(edges_list, base_index=0):
    edges = {}
    i = 0
    for lst in edges_list:
        for e in lst:
            edges[base_index + i] = e
            i += 1
    return edges

def run_inference_cache(
        network: pd.DataFrame,
        expression: pd.DataFrame,
        global_gene_to_node, 
        cache_dir, 
        overwrite, 
        msplit, 
        valsg_split_ratio, 
        cell_type,
        networks: dict = None,
        classes: list = None,
        all_edges = None,
        edge_ids_list = None,
        min_genes_per_graph=MIN_GENES_PER_GRAPH, 
        max_seq_length=None, 
        only_expressed_genes=True,
        with_edge_weights=True,
        skipped=0, 
        ncells=0, 
        verbose=False
    ):
    """
    Assign local ARACNe graph to each cell and cache each cell
    """
    os.makedirs(join(cache_dir, msplit), exist_ok=True)
    # remove unknown genes
    expression = expression[expression.columns[expression.columns.isin(global_gene_to_node)]]

    assert (network is None) != (networks is None), "Either network or networks must be provided, not both"
    infer_networks = networks is not None
    expression_genes = set(expression.columns)
    if not infer_networks:
        network = network[
            network[REG_VALS].isin(global_gene_to_node) & 
            network[TAR_VALS].isin(global_gene_to_node)
        ]
        network_genes = set(network[REG_VALS].to_list() + network[TAR_VALS].to_list())
        common_genes = sorted(list(network_genes.intersection(expression_genes)))
    else:
        class_networks = {}
        for name, network in networks.items():
            network = network.reset_index()
            network = network[
                network[REG_VALS].isin(global_gene_to_node) & 
                network[TAR_VALS].isin(global_gene_to_node)
            ]
            network_genes = set(network[REG_VALS].to_list() + network[TAR_VALS].to_list())
            common_genes = sorted(list(network_genes.intersection(expression_genes)))
            class_networks[name] = network, common_genes

    for i in range(expression.shape[0]):
        if ncells % 10 == 0:
            print(f"Processed {ncells} cells", end="\r")

        obs_name = expression.index[i]
        cell_number = i
        
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
        
        if infer_networks:
            # cell_network, common_genes = class_networks[classes[i]]
            edge_ids_i = edge_ids_list[i]
            edges = np.array(all_edges)[edge_ids_i]
            regulators, targets = zip(*edges)
            cell_network = pd.DataFrame({REG_VALS: regulators, TAR_VALS: targets})
            network_genes = set(regulators + targets)
            common_genes = sorted(list(network_genes.intersection(expression_genes)))
        else:   
            cell_network = network

        cell: pd.Series = expression.iloc[i, :][common_genes]

        if cell[cell != ZERO_IDX].shape[0] < min_genes_per_graph: # require a minimum number of expressed genes per cell 
            skipped += 1
            ncells+=1
            continue

        data = get_cell_data(cell_network, global_gene_to_node, max_seq_length, only_expressed_genes, with_edge_weights, cell)
        data.obs_name = obs_name
        
        torch.save(data, outfile)
        ncells += 1
        
        if verbose:
            try:
                torch.load(outfile)
                print(outfile)
            except:
                print(outfile, "-------- Failed")
        
    return (skipped, ncells)


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

    



def main(args):
    print("Loading Data...")
    adata = sc.read_h5ad(args.cells_path)
    global_gene_df = pd.read_csv(args.gene_index_path)
    global_gene_to_node = global_gene_df.set_index("gene_name")["idx"].to_dict()
    global_node_to_gene = global_gene_df.set_index("idx")["gene_name"].to_dict()

    if args.sample_n_cells is not None and adata.n_obs > args.sample_n_cells:
        sc.pp.subsample(adata, n_obs=args.sample_n_cells, random_state=12345, copy=False)

    adata_original = adata.copy()
    if args.mask_fraction is not None:
        adata_original = adata.copy()
        X_masked, masked_indices = mask_values(adata.X.astype(float), mask_prob=args.mask_fraction, mask_value=args.mask_value)
        adata.X = X_masked
    
    ranks = quantize_cells(adata.to_df(), n_bins=NUM_BINS, method="quantile")
    ranks = sc.AnnData(X=ranks.values, obs=adata.obs, var=adata.var, uns=adata.uns, obsm=adata.obsm)
    if args.infer_network:
        print("Inferring cell networks...")
        class_networks = {
            name: pd.read_csv(path, sep="\t").set_index([REG_VALS, TAR_VALS]) 
            for name, path in args.networks.items()
        }
        probs = ranks.obsm["class_probs"]
        classes = ranks.uns["class_probs_names"]
        E, MI, all_edges = build_class_edge_matrix(class_networks, classes, default_alpha=(0.05 + 1)/2)
        edge_ids_list, mis_list = [], []
        for probs_i in probs:
            edge_ids, pvals, mis = infer_cell_edges_(probs_i, E, MI, alpha=args.infer_network_alpha)
            edge_ids_list.append(edge_ids)
            mis_list.append(mis)

        run_inference_cache(
            network=None,
            networks=class_networks,
            classes=None,
            edge_ids_list=edge_ids_list,
            all_edges=all_edges,
            expression=ranks.to_df(),
            global_gene_to_node=global_gene_to_node, 
            cache_dir=args.cache_dir,
            overwrite=True, 
            msplit="all", 
            valsg_split_ratio=None, 
            cell_type="cell",
            max_seq_length=args.max_seq_length,
            only_expressed_genes=True,
            with_edge_weights=False,
            min_genes_per_graph=-1, 
            skipped=0, 
            ncells=0
        )
    else:
        network = pd.read_csv(args.network_path, sep="\t")
        run_inference_cache(
            network=network, 
            expression=ranks.to_df(), 
            global_gene_to_node=global_gene_to_node, 
            cache_dir=args.cache_dir,
            overwrite=True, 
            msplit="all", 
            valsg_split_ratio=None, 
            cell_type="cell",
            max_seq_length=args.max_seq_length,
            only_expressed_genes=True,
            with_edge_weights=False,
            min_genes_per_graph=-1, 
            skipped=0, 
            ncells=0
        )

    # reclaim memory
    del ranks
    gc.collect()

    dataset = GraphTransformerDataset(
        cache_dir=args.all_data_dir,
        dataset_name="cells",
        debug=False,
        inference=True,
        mask_fraction=0.0,
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=partial(scglm_collate_fn, inference=True)
    )
    
    # Load model
    # model: GDTransformer = GDTransformer.load_from_checkpoint(args.model_path, config=graph_kernel_attn_4096_TEST)
    model = GDTransformer.load_from_checkpoint(args.model_path, config=graph_kernel_attn_3L_4096)
    
    # Get embeddings
    print(f"Performing forward pass...")
    model.eval()

    embedding_list = []
    edges_list = []
    masked_edges_list = []
    non_masked_edges_list = []
    seq_lengths = []
    input_gene_ids_list = []
    n_obs = 0
    with torch.no_grad():
        for batch in dataloader:
            if args.use_masked_edges:
                edge_index = [e.cpu().numpy() for e in batch["edge_index"]]
                # idendify edges to mask for each cell
                # random edge mask returns a tuple (non_masked_edge_index, masked_edge_index)
                random_edge_masks = [random_edge_mask(edge_index, mask_ratio=args.mask_ratio) for edge_index in batch["edge_index"]]
                non_masked_edge_inices = [edge_mask[0] for edge_mask in random_edge_masks]
                masked_edge_indices = [edge_mask[1] for edge_mask in random_edge_masks]
                batch["edge_index"] = non_masked_edge_inices
                embedding, target_gene_ids, target_rank_ids, mask_locs, edge_index_list, num_nodes_list = model(send_to_gpu(batch))
            else:
                edge_index = [e.cpu().numpy() for e in batch["edge_index"]]
                embedding, target_gene_ids, target_rank_ids, mask_locs, edge_index_list, num_nodes_list = model(send_to_gpu(batch))
            
            masked_edges_list_ = [masked_edge_indices] if args.use_masked_edges else None
            non_masked_edges_list_ = [non_masked_edge_inices] if args.use_masked_edges else None
            edges_list_ = [edge_index]
            input_gene_ids_list_ = [target_gene_ids.cpu().numpy()]
            embedding_list_ = [embedding.cpu().numpy()]
            seq_lengths_ = [batch["num_nodes"]]

            # cache batch embeddings
            if args.cache:
                seq_lengths, edges, masked_edges, non_masked_edges, x_cls, x, input_genes, expression, metadata = get_scglm_embedding_vars(
                    retain_obs_vars=args.retain_obs_vars, 
                    adata=adata_original[batch["obs_name"],:],
                    global_gene_to_node=global_gene_to_node, 
                    global_node_to_gene=global_node_to_gene,
                    embedding_list=embedding_list_, 
                    edges_list=edges_list_,
                    masked_edges_list=masked_edges_list_,
                    non_masked_edges_list=non_masked_edges_list_,
                    seq_lengths=seq_lengths_, 
                    input_gene_ids_list=input_gene_ids_list_
                )
                save_embedding(
                    file=args.emb_path,
                    cache=args.cache,
                    cache_dir=args.emb_cache,
                    base_index=n_obs,
                    x=x,
                    x_cls=x_cls,
                    expression=expression,
                    seq_lengths=seq_lengths,
                    edges=edges,
                    masked_edges=masked_edges,
                    non_masked_edges=non_masked_edges,
                    metadata=metadata
                )
                gc.collect()

                n_obs += len(batch["num_nodes"])
                print(f"Processed {n_obs:,} observations")
                continue

            if args.use_masked_edges:
                masked_edges_list += masked_edges_list_
                non_masked_edges_list += non_masked_edges_list_
            edges_list += edges_list_
            input_gene_ids_list += input_gene_ids_list_
            embedding_list += embedding_list_
            seq_lengths += seq_lengths_

            n_obs += len(batch["num_nodes"])
            print(f"Processed {n_obs:,} observations")

    # delete temporary cache dir
    if os.path.isdir(args.cache_dir):
        shutil.rmtree(args.cache_dir)

    if args.cache:
        return

    seq_lengths, edges, x_cls, x, input_genes, expression, metadata = get_scglm_embedding_vars(
        retain_obs_vars=args.retain_obs_vars, 
        adata=adata_original,
        global_gene_to_node=global_gene_to_node, 
        global_node_to_gene=global_node_to_gene,  
        embedding_list=embedding_list, 
        edges_list=edges_list,
        masked_edges_list=masked_edges_list,
        non_masked_edges_list=non_masked_edges_list,
        seq_lengths=seq_lengths, 
        input_gene_ids_list=input_gene_ids_list
    )

    print("Saving emeddings...")
    # TODO: 1. Write embedding for each cell to embedding cache directory
    if args.use_masked_edges:
        save_embedding(
            file=args.emb_path,
            cache=args.cache,
            cache_dir=args.emb_cache,
            x=x,
            x_cls=x_cls,
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
            x=x,
            x_cls=x_cls,
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
        x=x,
        x_cls=x_cls,
        seq_lengths=seq_lengths,
        edges=edges,
        masks=masks,
        masked_expressions=masked_expressions,
        metadata=metadata
    )

def get_scglm_embedding_vars(retain_obs_vars, adata, global_gene_to_node, global_node_to_gene, embedding_list, edges_list, masked_edges_list, non_masked_edges_list, seq_lengths, input_gene_ids_list, base_index=0):
    seq_lengths = np.concatenate(seq_lengths, axis=0) # increment for cls token
    max_seq_length = max(seq_lengths)
    edges = get_edges_dict(edges_list, base_index)

    if masked_edges_list is not None and non_masked_edges_list is not None:
        masked_edges = get_edges_dict(masked_edges_list, base_index)
        non_masked_edges = get_edges_dict(non_masked_edges_list, base_index)
    else:
        masked_edges, non_masked_edges = None, None

    embeddings = np.concatenate([
        np.pad(emb, pad_width=((0, 0), (0, max_seq_length + 1 - emb.shape[1]), (0, 0)), # +1 to account for CLS token
               mode="constant", constant_values=0)
        for emb in embedding_list
    ], axis=0)

    # split embedding into cls and gene embeddings
    x_cls, x = embeddings[:,[0],:], embeddings[:,1:,:]

    input_gene_ids = np.concatenate([
        np.pad(gene_ids, pad_width=((0, 0), (0, max_seq_length + 1 - gene_ids.shape[1])), 
               mode="constant", constant_values=global_gene_to_node['<PAD>'])
        for gene_ids in input_gene_ids_list
    ], axis=0)
    input_genes = np.vectorize(global_node_to_gene.get)(input_gene_ids)

    # remove cls token
    input_genes = input_genes[:, 1:]
    
    # get original expression, exluding cls
    expression = np.concatenate([
        np.pad(adata[i, genes[:seq_lengths[i]]].X.toarray(), 
               pad_width=((0,0), (0, max_seq_length - seq_lengths[i])), 
               mode="constant", constant_values=0)
        for i, genes in enumerate(input_genes)
    ], axis=0)

    # retain requested metadata
    metadata = collect_metadata(adata, retain_obs_vars)

    return seq_lengths, edges, masked_edges, non_masked_edges, x_cls, x, input_genes, expression, metadata


if __name__ == "__main__":
    print("Starting main execution...")
    args.cells_path = join(args.data_dir, "cells.h5ad") if args.cells_path is None else args.cells_path
    # args.ranks_path = join(args.data_dir, "rank_raw.csv")
    args.emb_path = join(args.out_dir, "embedding.npz")
    args.emb_cache = join(args.out_dir, "cached_embeddings")
    args.cache_dir = join(args.out_dir, "cache")
    args.emb_cache_dir = join(args.out_dir, "emb_cache")
    args.all_data_dir = join(args.cache_dir, "all")
    
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.all_data_dir, exist_ok=True)
    if args.cache:
        os.makedirs(args.emb_cache, exist_ok=True)
    if args.networks:
        args.networks = NETWORK_SETS[args.networks]

    main(args)






    