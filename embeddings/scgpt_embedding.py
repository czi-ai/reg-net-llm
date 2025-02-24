import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.utils.data import Dataset, DataLoader
import os
import sys
import importlib
from os.path import join, dirname, abspath

scglm_rootdir = dirname(dirname(abspath(importlib.util.find_spec("scGraphLLM").origin)))
gene_names_map = pd.read_csv(join(scglm_rootdir, "data/gene-name-map.csv"), index_col=0)
ensg2hugo = gene_names_map.set_index("ensg.values")["hugo.values"].to_dict()
hugo2ensg = gene_names_map.set_index("hugo.values")["ensg.values"].to_dict()
ensg2hugo_vectorized = np.vectorize(ensg2hugo.get)
hugo2ensg_vectorized = np.vectorize(hugo2ensg.get)

REG_VALS = "regulator.values"
TAR_VALS = "target.values"



def main(args):
    sys.path.append(args.scgpt_rootdir)
    from scgpt.tasks.cell_emb import embed_data

    data = sc.read_h5ad(args.cells_path)
    data.var["symbol_id"] = data.var_names.to_series().apply(ensg2hugo.get)
    data = data[:, ~data.var["symbol_id"].isna()]
    data.var.set_index("symbol_id")
    data.var_names = data.var["symbol_id"]

    embeddings, symbol_ids, id_symbol_map = embed_data(
        adata_or_file=data[:10,:],
        model_dir=args.model_dir,
        gene_col="index",
        embedding_mode="raw",
        max_length=1200,
        batch_size=64,
        obs_to_save=None,
        device="cuda",
        use_fast_transformer=True,
        return_new_adata=True,
    ) # C x g x d

    # remove <CLS> token and embeddings
    embeddings = embeddings[:,1:,:]
    symbol_ids = symbol_ids[:,1:]

    # translate gene_ids
    id_gene_map_vectorized = np.vectorize(lambda x: id_symbol_map.get(x))
    genes_symbol = id_gene_map_vectorized(symbol_ids)
    genes_ensg = hugo2ensg_vectorized(genes_symbol)
    
    assert np.sum(genes_symbol == "cls") == 0
    pad_indices = [np.where(seq == '<pad>')[0][0] if np.any(seq == '<pad>') else -1 for seq in genes_symbol]

    # load aracne network
    network = pd.read_csv(join(args.aracne_dir, "consolidated-net_defaultid.tsv"), sep="\t")
    
    # get edges for each cell
    edges = {}
    for i, genes_i in enumerate(genes_ensg):
        local_gene_to_node_index = {gene: i for i,gene in enumerate(genes_i)}
        edges_i = network[
            network[REG_VALS].isin(genes_i) & 
            network[TAR_VALS].isin(genes_i)
        ].assign(**{
            REG_VALS: lambda df: df[REG_VALS].map(local_gene_to_node_index),
            TAR_VALS: lambda df: df[TAR_VALS].map(local_gene_to_node_index),
        })[[REG_VALS, TAR_VALS]].to_numpy().T
        edges[i] = edges_i

    np.savez(
        file=join(args.out_dir, "embedding.npz"), 
        x=embeddings,
        pad_indices=pad_indices,
        edges=edges, 
        allow_pickle=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--aracne_dir", type=str, required=True)
    parser.add_argument("--scgpt_rootdir", type=str, required=True)
    args = parser.parse_args()

    args.cells_path = join(args.data_dir, "cells.h5ad")
    args.out_dir = join(args.data_dir, "embeddings/scgpt")
    args.emb_path = join(args.out_dir, "embedding.h5ad")
    os.makedirs(args.out_dir, exist_ok=True)

    main(args)