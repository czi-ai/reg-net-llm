import argparse
import pandas as pd
import scanpy as sc
from scipy.sparse import csc_matrix
import os
from os.path import join, dirname, abspath
from geneformer import TranscriptomeTokenizer, EmbExtractor

import importlib.util
import os

geneformer_dir = dirname(dirname(abspath(importlib.util.find_spec("geneformer").origin)))

def main(args):
    # get counts as annoted data
    adata = sc.read_h5ad(args.cells_path)
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
        output_directory=args.out_dir,
        output_prefix="", 
        file_format="h5ad"
    )

    embex = EmbExtractor(
        model_type="Pretrained",
        max_ncells=None,
        emb_mode="gene",
        gene_emb_style="mean_pool",
        emb_layer=-1,
        forward_batch_size=256,
        nproc=1,
        token_dictionary_file=join(geneformer_dir, "geneformer/gene_dictionaries_30m/token_dictionary_gc30M.pkl")
    )

    embs = embex.extract_embs(
        model_directory=join(geneformer_dir, "gf-6L-30M-i2048"),
        input_data_file=join(dirname(args.out_dir), "geneformer.dataset"),
        output_directory=args.out_dir,  
        output_prefix=""
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()

    args.cells_path = join(args.data_dir, "cells.h5ad")
    args.out_dir = join(args.data_dir, "embeddings/geneformer/geneformer")
    args.counts_dir = join(args.data_dir, "counts")
    args.counts_path = join(args.counts_dir, "counts.h5ad")

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.counts_dir, exist_ok=True)

    main(args)
