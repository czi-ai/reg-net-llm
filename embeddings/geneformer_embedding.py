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

def mask_values(sparse_mat, mask_prob=0.15, mask_value=0):
    """
    Masks each nonzero value in a sparse matrix with a given probability.

    Parameters:
        sparse_mat (scipy.sparse.spmatrix): Input sparse matrix.
        mask_prob (float): Probability of masking each nonzero value (default: 0.15).
        mask_value (float): Value to use for masking (default: 0, suitable for sparse).

    Returns:
        scipy.sparse.spmatrix: Masked sparse matrix with the same format.
    """
    sparse_mat = sparse_mat.tocoo()
    mask = np.random.rand(len(sparse_mat.data)) < mask_prob
    masked_indices = (sparse_mat.row[mask], sparse_mat.col[mask])
    sparse_mat.data[mask] = mask_value

    return sparse_mat.tocsr(), masked_indices

def get_masked_gene_expressions(masked_indices, adata):
    """
    Iterates through each row and returns the genes (columns) that were masked in each row.

    Parameters:
        masked_indices (tuple of arrays): Indices of the masked entries (row_indices, col_indices).
        sparse_mat (scipy.sparse.spmatrix): The original sparse matrix.

    Returns:
        masked_genes_by_row (list of lists): List where each element is a list of masked gene indices (columns) for each row.
    """
    row_indices, col_indices = masked_indices
    masked_genes = {}
    masked_gene_expressions = {}
    # Iterate through all rows
    for i in range(adata.shape[0]):
        # Find the indices where the row is involved in masking
        row_masked_indices = np.where(row_indices == i)[0]
        masked_genes_index = col_indices[row_masked_indices].tolist()
        masked_genes[i] = adata.var_names[masked_genes_index].tolist()
        masked_gene_expressions[i] = adata[i, masked_genes[i]].X.toarray().flatten()

    return masked_genes, masked_gene_expressions

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

    if args.mask_fraction is not None:
        X_masked, masked_indices = mask_values(counts.X.astype(float), mask_prob=args.mask_fraction, mask_value=args.mask_value)
        counts.X = X_masked
        masked_genes, masked_gene_expressions = get_masked_gene_expressions(masked_indices, adata)

    counts.write_h5ad(args.counts_path)

    tokenizer = TranscriptomeTokenizer(
        model_input_size=2048,
        nproc=1,
        # token_dictionary_file=join(geneformer_dir, "geneformer/gene_dictionaries_30m/token_dictionary_gc30M.pkl")
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

    embeddings, data = embex.extract_embs(
        model_directory=join(geneformer_dir, "gf-6L-30M-i2048"),
        input_data_file=join(dirname(args.gf_out_dir), "geneformer.dataset"),
        output_directory=args.out_dir, # not used
        output_prefix=""
    )
    seq_lengths = data["length"]
    input_ids_list = data["input_ids"]

    id_gene_map_vectorized = np.vectorize(lambda x: embex.token_gene_dict.get(x))
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

    if args.mask_fraction is None:
        np.savez(
            file=join(args.out_dir, "embedding.npz"), 
            x=embeddings,
            seq_lengths=seq_lengths,
            edges=edges, 
            allow_pickle=True
        )
        return
    
    assert len(masked_genes) == len(input_genes)
    masks = {}
    for i in range(len(masked_genes)):
        input_genes_i = input_genes[i]
        masked_genes_i = masked_genes[i]
        masked_gene_emb_index = pd.Series(input_genes_i).pipe(lambda x: x[x.isin(masked_genes_i)].index.to_list())
        masks[i] = masked_gene_emb_index

    
    np.savez(       
        file=join(args.out_dir, "embedding.npz"), 
        x=embeddings,
        seq_lengths=seq_lengths,
        edges=edges,
        masks=masks,
        expression=masked_gene_expressions,
        allow_pickle=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--aracne_dir", type=str, required=True)
    parser.add_argument("--mask_fraction", type=float, default=None)
    parser.add_argument("--mask_value", type=float, default=1e-4)
    parser.add_argument("--sample_n_cells", type=int, default=None)
    parser.add_argument("--max_n_genes", type=int, default=1200)
    args = parser.parse_args()

    args.cells_path = join(args.data_dir, "cells.h5ad")
    args.gf_out_dir = join(args.out_dir, "geneformer")
    args.counts_dir = join(args.data_dir, "counts")
    args.counts_path = join(args.counts_dir, "counts.h5ad")

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.counts_dir, exist_ok=True)

    main(args)
