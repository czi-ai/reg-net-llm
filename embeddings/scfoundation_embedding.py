import argparse
import pandas as pd
import scanpy as sc
import os
from os.path import join, dirname, abspath

import importlib.util
import os

scglm_rootdir = dirname(dirname(abspath(importlib.util.find_spec("scGraphLLM").origin)))
gene_names_map = pd.read_csv(join(scglm_rootdir, "data/gene-name-map.csv"), index_col=0)
ensg2hugo = gene_names_map.set_index("ensg.values")["hugo.values"].to_dict()


def main(args):
    # Load cells & translate to human symbol gene names
    data = sc.read_h5ad(args.cells_path)
    data.var["symbol_id"] = data.var_names.to_series().apply(ensg2hugo.get)
    data = data[:, ~data.var["symbol_id"].isna()]
    data.var.set_index("symbol_id")
    data.var_names = data.var["symbol_id"]

    # save translated data
    data.write_h5ad(args.trans_cells_path)

    # run get embedding script from scFoundation
    os.system(
        f"python {args.scf_embedding_script} "\
        f"--task_name scf_embedding "
        "--input_type singlecell "\
        "--output_type gene "\
        "--pool_type all "\
        "--tgthighres a5 "\
        f"--data_path {args.trans_cells_path} "\
        f"--save_path {args.out_dir} "\
        "--demo "
        "--pre_normalized T "
        "--version rde"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--scf_embedding_script", type=str, required=True)

    args = parser.parse_args()

    args.cells_path = join(args.data_dir, "cells.h5ad")
    args.out_dir = join(args.data_dir, "embeddings/scfoundation")
    args.trans_cells_path = join(args.out_dir, "cells.h5ad")
    args.scf_rootdir = dirname(dirname(args.scf_embedding_script))    

    os.makedirs(args.out_dir, exist_ok=True)

    main(args)
