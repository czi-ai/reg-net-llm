import argparse
import pandas as pd
import numpy as np
import scanpy as sc
from scipy.sparse import csc_matrix
import os
from os.path import join, dirname, abspath
from geneformer import TranscriptomeTokenizer, EmbExtractor
from utils import mask_values, get_locally_indexed_edges, get_locally_indexed_masks_expressions, save_embedding

import importlib.util
import os

geneformer_dir = dirname(dirname(abspath(importlib.util.find_spec("geneformer").origin)))

REG_VALS = "regulator.values"
TAR_VALS = "target.values"

def main(args):
    # initialize tokenizer
    tokenizer = TranscriptomeTokenizer(
        model_input_size=2048,
        nproc=1,
        special_token=False,
        custom_attr_name_dict={"cell_id": "cell_id"},
        token_dictionary_file=join(geneformer_dir, "geneformer/gene_dictionaries_30m/token_dictionary_gc30M.pkl")
    )

    # get counts as annoted data
    adata = sc.read_h5ad(args.cells_path)
    if args.sample_n_cells is not None and adata.n_obs > args.sample_n_cells:
        sc.pp.subsample(adata, n_obs=args.sample_n_cells, random_state=123456, copy=False)

    adata = adata[adata.obs["n_counts"].sort_values(ascending=False).index]
    adata.obs["cell_id"] = adata.obs_names.values

    # filter out unrecognized genes by the tokenizer
    adata = adata[:, adata.var_names.isin(tokenizer.gene_median_dict.keys())]

    if "counts" in adata.layers:
        counts = sc.AnnData(
            X=csc_matrix(adata.layers["counts"].astype(int)),
            obs=adata.obs[["n_counts", "cell_id"]],
            var=pd.DataFrame(index=adata.var.index).assign(**{"ensembl_id": lambda df: df.index.to_series()}),
        )
    else:
        counts = sc.AnnData(
            X=csc_matrix(np.expm1(adata.X)),
            obs=adata.obs[["cell_id"]],
            var=pd.DataFrame(index=adata.var.index).assign(**{"ensembl_id": lambda df: df.index.to_series()}),
        )
        counts.obs["n_counts"] = np.array(counts.X.sum(axis=1)).flatten()

    if args.mask_fraction is not None:
        X_masked, masked_indices = mask_values(counts.X.astype(float), mask_prob=args.mask_fraction, mask_value=args.mask_value)
        counts.X = X_masked

    counts.write_h5ad(args.counts_path)

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
    
    # reorder data and embeddings according to cell ids in adata
    cell_id_to_index = {cell_id: i for i, cell_id in enumerate(data["cell_id"])}
    reordered_indices = [cell_id_to_index[cell_id] for cell_id in adata.obs["cell_id"]]
    data = data.select(reordered_indices)
    embeddings = embeddings[reordered_indices,:,:]

    input_ids_list = data["input_ids"]
    seq_lengths = [len(input_ids) for input_ids in input_ids_list]
    max_seq_length = max(seq_lengths)

    id_gene_map_vectorized = np.vectorize(lambda x: embex.token_gene_dict.get(x))
    input_genes = [id_gene_map_vectorized(np.array(ids)) for ids in input_ids_list]

    # load aracne network & get edges
    network = pd.read_csv(join(args.aracne_dir, "consolidated-net_defaultid.tsv"), sep="\t")
    edges = get_locally_indexed_edges(input_genes, src_nodes=network[REG_VALS], dst_nodes=network[TAR_VALS])

    # get original expression
    expression = np.concatenate([
        np.pad(adata[i, genes].X.toarray(), 
               pad_width=((0,0), (0, max_seq_length - len(genes))), 
               mode="constant", constant_values=0)
        for i, genes in enumerate(input_genes)
    ], axis=0)

    # retain requested metadata
    metadata = {}
    for var in args.retain_obs_vars:
        try:
            metadata[var] = adata.obs[var].tolist()
        except KeyError:
            print(f"Key {var} not in observational metadata...")

    if args.mask_fraction is None:
        save_embedding(
            file=args.emb_path,
            cache=args.cache,
            cache_dir=args.emb_cache,
            x=embeddings,
            seq_lengths=seq_lengths,
            expression=expression,
            edges=edges,
            metadata=metadata
        )
        return

    masks, masked_expressions = get_locally_indexed_masks_expressions(adata, masked_indices, input_genes)
    save_embedding(
        file=args.emb_path,
        cache=args.cache,
        cache_dir=args.emb_cache,
        x=embeddings,
        seq_lengths=seq_lengths,
        expression=expression,
        edges=edges,
        metadata=metadata,
        masks=masks,
        masked_expressions=masked_expressions
    )
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--aracne_dir", type=str, required=True)
    parser.add_argument("--mask_fraction", type=float, default=None)
    parser.add_argument("--mask_value", type=float, default=1e-4)
    parser.add_argument("--retain_obs_vars", nargs="+", default=[])
    parser.add_argument("--sample_n_cells", type=int, default=None)
    parser.add_argument("--max_n_genes", type=int, default=1200)
    parser.add_argument("--cache", action="store_true")
    args = parser.parse_args()

    args.cells_path = join(args.data_dir, "cells.h5ad")
    args.gf_out_dir = join(args.out_dir, "geneformer")
    args.counts_dir = join(args.data_dir, "counts")
    args.counts_path = join(args.counts_dir, "counts.h5ad")
    args.emb_path = join(args.out_dir, "embedding.npz")
    args.emb_cache = join(args.out_dir, "cached_embeddings")

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.counts_dir, exist_ok=True)
    if args.cache:
        os.makedirs(args.emb_cache, exist_ok=True)

    main(args)
