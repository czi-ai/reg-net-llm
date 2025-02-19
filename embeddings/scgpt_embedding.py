import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import os
import sys
import importlib
from os.path import join, dirname, abspath

scglm_rootdir = dirname(dirname(abspath(importlib.util.find_spec("scGraphLLM").origin)))
gene_names_map = pd.read_csv(join(scglm_rootdir, "data/gene-name-map.csv"), index_col=0)
ensg2hugo = gene_names_map.set_index("ensg.values")["hugo.values"].to_dict()

def main(args):
    sys.path.append(args.scgpt_rootdir)
    from scgpt.tasks.cell_emb import embed_data

    data = sc.read_h5ad(args.cells_path)
    data.var["symbol_id"] = data.var_names.to_series().apply(ensg2hugo.get)
    data = data[:, ~data.var["symbol_id"].isna()]
    data.var.set_index("symbol_id")
    data.var_names = data.var["symbol_id"]

    embs = embed_data(
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

    np.savez(join(args.out_dir, "embedding.npz"), matrix=embs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--scgpt_rootdir", type=str, required=True)
    args = parser.parse_args()

    args.cells_path = join(args.data_dir, "cells.h5ad")
    args.out_dir = join(args.data_dir, "embeddings/scgpt")
    args.emb_path = join(args.out_dir, "embedding.h5ad")
    os.makedirs(args.out_dir, exist_ok=True)

    main(args)