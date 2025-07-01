print("This is the top of the script...")
import os
os.environ["TORCH_USE_CUDA_DSA"] = "1"  # Enable device-side assertions
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Make CUDA errors more visible

import argparse
import gc
import shutil
from os.path import join, dirname, abspath
from functools import partial
from typing import Dict
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc
import torch

from scGraphLLM._globals import * ## these define the indices for the special tokens 
from scGraphLLM.models import GDTransformer
from scGraphLLM.infer_graph import infer_cell_edges_, build_class_edge_matrix
from scGraphLLM.benchmark import send_to_gpu, random_edge_mask
from scGraphLLM.config import *
from scGraphLLM.data import *
from scGraphLLM.eval_config import *
from scGraphLLM._globals import *
from scGraphLLM.inference import \
    GeneVocab, GraphTokenizer, InferenceDataset, VariableNetworksInferenceDataset
from scGraphLLM.network import RegulatoryNetwork
from utils import (
    mask_values, 
    get_locally_indexed_edges, 
    get_locally_indexed_masks_expressions, 
    save_embedding,
    collect_metadata
)

def main(args):
    print("Loading Data...")
    adata = sc.read_h5ad(args.cells_path)

    print("Loading Gene Vocabulary")
    vocab = GeneVocab.from_csv(args.gene_index_path)

    if args.sample_n_cells is not None and adata.n_obs > args.sample_n_cells:
        sc.pp.subsample(adata, n_obs=args.sample_n_cells, random_state=12345, copy=False)

    adata_original = adata.copy()
    if args.mask_fraction is not None:
        adata_original = adata.copy()
        X_masked, masked_indices = mask_values(adata.X.astype(float), mask_prob=args.mask_fraction, mask_value=args.mask_value)
        adata.X = X_masked
    
    if args.infer_network:
        print("Inferring cell networks...")
        class_networks = {
            name: RegulatoryNetwork.from_csv(path, sep="\t")
            for name, path in args.networks.items()
        }
        all_edges, edge_ids_list, weights_list = infer_edges(
            probs=adata.obsm["class_probs"],
            classes=adata.uns["class_probs_names"],
            class_networks=class_networks,
            hard_assignment=args.hard_assignment,
            alpha=args.infer_network_alpha,
            default_alpha=(0.05 + 1) / 2
        )
        dataset = VariableNetworksInferenceDataset(
            expression=adata.to_df(), 
            tokenizer=GraphTokenizer(vocab=vocab, n_bins=NUM_BINS),
            edge_ids_list=edge_ids_list,
            weights_list=weights_list,
            all_edges=all_edges,
            limit_regulon=args.limit_regulon, 
            limit_graph=args.limit_graph
        )
    else:
        network = RegulatoryNetwork.from_csv(args.network_path, sep="\t")\
            .prune(limit_regulon=args.limit_regulon, limit_graph=args.limit_graph, inplace=True)
        
        dataset = InferenceDataset(
            expression=adata.to_df(), 
            tokenizer=GraphTokenizer(vocab=vocab, network=network, n_bins=NUM_BINS)
        )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn
    )
    
    # Load model
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
                    global_gene_to_node=vocab.gene_to_node, 
                    global_node_to_gene=vocab.node_to_gene,
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
        global_gene_to_node=vocab.gene_to_node, 
        global_node_to_gene=vocab.node_to_gene,  
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


def infer_edges(probs, classes, class_networks, hard_assignment, alpha, default_alpha):
    E, W, all_edges = build_class_edge_matrix(class_networks, classes, default_alpha)
    all_edges_to_idx = {edge: idx for idx, edge in enumerate(all_edges)}
    edge_ids_list, mis_list = [], []

    for i, probs_i in enumerate(probs):
        if i % 100 == 0:
            print(f"Inferred {i:,} cell networks...")
            
        zero_soft_edges = False
        if not hard_assignment:
            edge_ids, pvals, mis = infer_cell_edges_(probs_i, E, W, alpha=alpha)
            zero_soft_edges = len(edge_ids) == 0

        if hard_assignment or zero_soft_edges:
            class_hat = classes[probs_i.argmax()]
            network = class_networks[class_hat]
            mis = network.weights
            edges = network.edges
            edge_ids = [all_edges_to_idx[edge] for edge in edges]
            
        edge_ids_list.append(edge_ids)
        mis_list.append(mis)

    return all_edges, edge_ids_list, mis_list


def get_edges_dict(edges_list, base_index=0):
    edges = {}
    i = 0
    for lst in edges_list:
        for e in lst:
            edges[base_index + i] = e
            i += 1
    return edges


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--cells_path", type=str, default=None)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--network_path", type=str)
    parser.add_argument("--infer_network", action="store_true")
    parser.add_argument("--hard_assignment", action="store_true")
    parser.add_argument("--limit_regulon", type=int, default=None)
    parser.add_argument("--limit_graph", type=int, default=100)
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
    args = parser.parse_args()

    args.cells_path = join(args.data_dir, "cells.h5ad") if args.cells_path is None else args.cells_path
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
