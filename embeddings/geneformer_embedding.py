import argparse
import pandas as pd
import numpy as np
import scanpy as sc
from scipy.sparse import csc_matrix
import os
from os.path import join, dirname, abspath
from geneformer import TranscriptomeTokenizer, EmbExtractor

import importlib.util
import os

geneformer_dir = dirname(dirname(abspath(importlib.util.find_spec("geneformer").origin)))

REG_VALS = "regulator.values"
TAR_VALS = "target.values"

def main(args):
    # get counts as annoted data
    adata = sc.read_h5ad(args.cells_path)
    
    if args.sample_n_cells is not None and adata.n_obs > args.sample_n_cells:
        sc.pp.subsample(adata, n_obs=args.sample_n_cells, random_state=12345, copy=False)

    counts = sc.AnnData(
        X=csc_matrix(adata.layers["counts"].astype(int)),
        obs=adata.obs[["n_counts"]],
        var=pd.DataFrame(index=adata.var.index).assign(**{"ensembl_id": lambda df: df.index.to_series()}),
    )
    counts.write_h5ad(args.counts_path)

    tokenizer = TranscriptomeTokenizer(
        model_input_size=2048,
        nproc=1
    )
    tokenizer.tokenize_data(
        data_directory=args.counts_dir, 
        output_directory=args.gf_out_dir,
        output_prefix="", 
        file_format="h5ad"
    )

    embex = EmbExtractor(
        model_type="Pretrained",
        max_ncells=None,
        emb_mode="raw",
        gene_emb_style="mean_pool", # not used
        emb_layer=-1,
        forward_batch_size=32,
        nproc=1,
        token_dictionary_file=join(geneformer_dir, "geneformer/gene_dictionaries_30m/token_dictionary_gc30M.pkl")
    )

    embeddings, token_gene_dict, seq_lengths, input_ids_list = embex.extract_embs(
        model_directory=join(geneformer_dir, "gf-6L-30M-i2048"),
        input_data_file=join(dirname(args.gf_out_dir), "geneformer.dataset"),
        output_directory=args.out_dir, # (not used)  
        output_prefix=""
    )

    id_gene_map_vectorized = np.vectorize(lambda x: token_gene_dict.get(x))
    input_genes = [id_gene_map_vectorized(np.array(ids)) for ids in input_ids_list]

     # load aracne network
    network = pd.read_csv(join(args.aracne_dir, "consolidated-net_defaultid.tsv"), sep="\t")

    # get edges for each cell
    edges = {}
    for i, genes_i in enumerate(input_genes):
        assert len(genes_i) == seq_lengths[i]
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
        seq_lengths=seq_lengths,
        edges=edges, 
        allow_pickle=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--aracne_dir", type=str, required=True)
    parser.add_argument("--sample_n_cells", type=int, default=None)
    args = parser.parse_args()

    args.cells_path = join(args.data_dir, "cells.h5ad")
    args.gf_out_dir = join(args.out_dir, "geneformer")
    args.counts_dir = join(args.data_dir, "counts")
    args.counts_path = join(args.counts_dir, "counts.h5ad")

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.counts_dir, exist_ok=True)

    main(args)
