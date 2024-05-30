"""
Script for preparing the cellxgene dataset for ARACNE inference.

The raw cellxgene data is sorted in partitioants accoring to origin tissue e.g.,
heart
 - partition_0.h5ad
 - partition_1.h5ad
 - ...
pan-cancer
- partition_0.h5ad
- partition_1.h5ad
 - ...

There are two distinct steps:
1. "separate": Normalize cell type names and separate into folders by cell type
2. "preprocess: Preprocess the cells, including quality checks and filtering.

Example: peform both steps 1 and 2:
$ python preprocess_cellxgene.py --steps separate preprocess --parallel

Example: peform both steps 1 and 2, exlusively for heart and lung cells:
$ python preprocess_cellxgene.py --steps separate preprocess --tissues heart lung --parallel 
"""
import json
import pandas as pd 
import numpy as np 
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt

import os
import re
import sys
import warnings
import logging
import multiprocessing as mp
from typing import List
from os.path import join, basename
from functools import partial
from argparse import ArgumentParser

from cell_types import *
from preprocess import list_dirs, list_files, concatenate_partitions

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.DEBUG)

# directories
CELLXGENE_DIR = "/burg/pmg/users/rc3686/data/cellxgene"
DATA_DIR = join(CELLXGENE_DIR, "data")

# var names
FEATURE_NAME = "feature_name"
FEATURE_ID = "feature_id" # ENSG
SOMA_JOINID = "soma_joinid"
FEATURE_LENGTH ="feature_length"
NNZ = "nnz"
N_MEASURED_OBS = "n_measured_obs"

# obs names
CELL_TYPE = "cell_type"
TISSUE = "tissue"

STATIC_VARS = [
    SOMA_JOINID, 
    FEATURE_ID, 
    FEATURE_LENGTH
]

TISSUES = [
    "heart",
    "blood",
    "brain",
    "lung",
    "kidney",
    "intestine",
    "pancreas",
    "others"
]

# protein coding genes
genes_names = pd.read_csv("/burg/pmg/users/rc3686/gene-name-map.csv", index_col=0)
ENSG_PROTEIN_CODING = set(genes_names["ensg.values"])


def clean_cell_type_name(cell_type_name):
    # Replace spaces and backslashes with underscores
    cell_type_name = cell_type_name.replace(" ", "_").replace("/", "_")
    # Remove non-alphanumeric characters, hyphens, and underscores, lower
    cell_type_name = re.sub(r"[^a-zA-Z0-9-_]", "", cell_type_name).lower()
    return cell_type_name


def write_cell_types_for_partition(partition_file, type_dir, write_name, type_limit=None):
    print(f"\nLoading {write_name} partition...")
    adata = sc.read_h5ad(partition_file)
    # filter non-protein-coding genes
    adata = adata[:,adata.var[FEATURE_ID].isin(ENSG_PROTEIN_CODING)]
    
    print(f"Cleaning cell type names for {write_name} partition...")
    # clean cell typenames    
    adata.obs[CELL_TYPE] = adata.obs[CELL_TYPE].map(clean_cell_type_name)
    cell_types = pd.Series(adata.obs[CELL_TYPE].unique().sort_values()) \
        if type_limit is None else \
        adata.obs[CELL_TYPE].value_counts(ascending=False).index[:type_limit].to_series()
    
    # write cell type to separate partitions
    for cell_type in cell_types:
        cell_type_dir = join(type_dir, cell_type, "partitions")
        os.makedirs(cell_type_dir, exist_ok=True)
        subset = adata[adata.obs[CELL_TYPE] == cell_type]
        subset.write_h5ad(join(cell_type_dir, f"{write_name}.h5ad"))
    
    print("="*50)
    print(f"Wrote {len(cell_types)} cell types for {write_name} partition..\n")

    del adata
    return cell_types


def main(args):

    inputs = []
    for tissue in args.tissues:
        partitions_files = list_files(join(args.tissue_dir, tissue))
        for i, file in enumerate(partitions_files):
            inputs.append((file, args.type_dir, f"{tissue}_{i}")) # file_path, type_dir, write_namecd 

    print(f"Writing cell types for {len(inputs)} partitions from {len(args.tissues)} tissues...")
    if args.parallel:
        with mp.Pool(args.n_cores) as pool:
            cell_types = pool.starmap(write_cell_types_for_partition, inputs)
        cell_types = pd.Series(pd.concat(cell_types).unique()).rename("cell_type")
        cell_types.to_csv(args.cell_types_path, index=False)
        print(f"Wrote {len(cell_types)} cell types for {len(inputs)} partitions from {len(args.tissues)} tissues...")
        return
    
    for input in inputs:
        print(f"Writing cell types for {input[2]} partition..\n")
        write_cell_types_for_partition(*input)

    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--tissues", nargs="+", default=TISSUES)
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--out_dir", type=str, default=DATA_DIR)
    parser.add_argument("--n_cores", type=int, default=1)
    parser.add_argument("--suffix", required=True)
    parser.add_argument("--parallel", action="store_true")
    args = parser.parse_args()
    
    # define directories
    args.tissue_dir = join(args.data_dir, "tissue")
    args.type_dir = join(args.out_dir, "cell_type" + "_" + args.suffix)
    args.log_dir = join(args.out_dir, f"log")
    args.cell_types_path = join(args.out_dir, f"{CELL_TYPE}_{args.suffix}.csv")
    
    logging
    os.makedirs(args.log_dir, exist_ok=True)
    logger.addHandler(logging.FileHandler(join(args.log_dir, f"log_{args.suffix}.txt")))
    logger.info(json.dumps(vars(args), sort_keys=True, indent=4))

    try:
        main(args)
    except Exception as e: 
        print(e)
    
