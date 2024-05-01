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

import pandas as pd 
import numpy as np 
import scanpy as sc
import anndata as ad

import os
import re
import warnings
import multiprocessing as mp
from typing import List
from os.path import join, basename
from functools import partial
from argparse import ArgumentParser
warnings.filterwarnings("ignore")

# directories
CELLXGENE_DIR = "/burg/pmg/users/rc3686/data/cellxgene"
DATA_DIR = join(CELLXGENE_DIR, "data")

# var names
FEATURE_NAME = "feature_name"
SOMA_JOINID = "soma_joinid"
FEATURE_ID = "feature_id"
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
    "intestine"
    "pancreas",
    "others"
]

def list_files(dir):
    """List full paths of all files in a directory"""
    root, _, files = list(os.walk(dir))[0]
    return sorted([join(root, file) for file in files])


def list_dirs(dir):
    """List full paths of all subdirectories of a directory"""
    root, dirs, _ = list(os.walk(dir))[0]
    return sorted([join(root, dir) for dir in dirs])


def apply_parallel_list(lst, func, n_cores=None, **kwargs):
    """Parallel apply function to elements of a list"""
    with mp.Pool(n_cores) as pool:
        result = pool.map(partial(func, **kwargs), lst)
    return result


def check_index(df, name):
    if df.index.name != name:
        df.set_index(name, inplace=True)


def read_cxg_h5ad_file(fpath):
    adata = sc.read_h5ad(fpath)
    check_index(adata.var, FEATURE_NAME)
    check_index(adata.obs, SOMA_JOINID)
    # cell_type normalization
    return adata


def concatenate_partitions(partitions):
    """Concatenate the partitions of cellxgene datasets"""
    # concatenate observations
    adata = ad.concat(partitions, axis=0)
    # check consistency in var metadata
    var = partitions[0].var[STATIC_VARS]
    for p in partitions:
        assert p.var[STATIC_VARS].equals(var),\
            "Static vars are not consistent across partitions, cannot be concatenated"
    adata.var = var
    return adata


def get_tissue(tissue, tissue_dir, limit=None):
    tissue_dir = join(tissue_dir, tissue)
    partitions = [read_cxg_h5ad_file(file) for file in list_files(tissue_dir)[:limit]]
    adata = concatenate_partitions(partitions)
    return adata


def clean_cell_type_name(cell_type_name):
    # Replace spaces and backslashes with underscores
    cell_type_name = cell_type_name.replace(" ", "_").replace("/", "_")
    # Remove non-alphanumeric characters, hyphens, and underscores, lower
    cell_type_name = re.sub(r"[^a-zA-Z0-9-_]", "", cell_type_name).lower()
    return cell_type_name


def write_cell_types(tissue, tissue_dir, type_dir, limit=None) -> List[str]:
    """Writes cell-type subsets to specific"""
    print(f"Getting data for tissue {tissue}")
    adata = get_tissue(tissue, tissue_dir=tissue_dir, limit=limit)

    # clean cell typenames
    adata.obs[f"original_{CELL_TYPE}"] = adata.obs[CELL_TYPE].copy()
    adata.obs[CELL_TYPE] = adata.obs[CELL_TYPE].map(clean_cell_type_name)
    cell_types = adata.obs[CELL_TYPE].unique().sort_values().to_series() \
        if limit is None else \
        adata.obs[CELL_TYPE].value_counts(ascending=False).index[:limit].to_series()
    
    # write cell type to separate partitions
    for cell_type in cell_types:
        cell_type_dir = join(type_dir, cell_type)
        os.makedirs(cell_type_dir, exist_ok=True) 
        print(f"Getting {cell_type} cells of {tissue}..." )
        subset = adata[adata.obs[CELL_TYPE] == cell_type]
        subset.write_h5ad(join(cell_type_dir, f"{tissue}.h5ad"))
    
    return cell_types


def load_cell_type(cell_type_dir):
    partitions = [read_cxg_h5ad_file(file) for file in list_files(cell_type_dir)]
    adata = concatenate_partitions(partitions)
    return adata


def preprocess_cell_type(cell_type_dir, mito_thres, umi_min, umi_max, target_sum):
    print(f"Preprocessing cell type: {basename(cell_type_dir)}...")
    # load cell type data
    adata = load_cell_type(cell_type_dir)

    # calculate qc metrics
    adata.var["mt"] = adata.var_names.map(lambda n: str(n).upper().startswith("MT"))
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, inplace=True)

    # apply filters
    adata = adata[adata.obs["pct_counts_mt"] < mito_thres]
    sc.pp.filter_cells(adata, min_counts=umi_min)
    sc.pp.filter_cells(adata, max_counts=umi_max)
    
    # normalize
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)

    # write 
    adata.write_h5ad(join(cell_type_dir, "full.h5ad"))
    return adata


def separate(args):
    kwargs = dict(
        type_dir=args.type_dir,
        tissue_dir=args.tissue_dir,
        limit=args.type_limit
    )
    if args.parallel:
        cell_types = apply_parallel_list(args.tissues, func=partial(write_cell_types, **kwargs))
        cell_types = pd.Series(pd.concat(cell_types).unique()).rename("cell_type")
        cell_types.to_csv(args.cell_types_path, index=False)
        return
    for tissue in args.tissues:
        write_cell_types(tissue, **kwargs)


def preprocess(args):
    kwargs = dict(
        mito_thres=args.mito_thres, 
        umi_min=args.umi_min, 
        umi_max=args.umi_max, 
        target_sum=args.target_sum
    )
    if args.parallel:
        apply_parallel_list(list_dirs(args.type_dir), func=partial(preprocess_cell_type, **kwargs))
        return
    for cell_type_dir in list_dirs(args.type_dir):
        preprocess_cell_type(cell_type_dir, **kwargs)


def main(args):
    if "separate" in args.steps:
        separate(args)
    if "preprocess" in args.steps:
        preprocess(args)
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--steps", nargs="+", default=["separate", "preprocess"])
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--tissues", nargs="+", default=TISSUES)
    parser.add_argument("--type_limit", type=int, default=None)
    parser.add_argument("--cores", default=os.cpu_count())
    parser.add_argument("--mito_thres", type=int, default=20)
    parser.add_argument("--umi_min", type=int, default=1000)
    parser.add_argument("--umi_max", type=int, default=1e6)
    parser.add_argument("--target_sum", type=int, default=1e6)
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--suffix", required=True)
    args = parser.parse_args()
    
    # define directories
    args.tissue_dir = join(args.data_dir, "tissue")
    args.type_dir = join(args.data_dir, "cell_type" + "_" + args.suffix)
    args.cell_types_path = join(args.type_dir, f"{CELL_TYPE}.csv")

    main(args)
