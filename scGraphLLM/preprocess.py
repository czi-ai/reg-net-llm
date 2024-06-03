"""
Script for preprocessing a dataset for use by the Single Cell Graph LLM and ARACNe

Example:
The dataset can be a single file or split into partitions,
python preprocess.py --data_path /path/to/sample.h5ad
python preprocess.py --data_path /path/to/partitions/dir

where /path/to/partitions/dir
heart
|-- partition_0.h5ad
|-- partition_1.h5ad
|-- ...
"""
from argparse import ArgumentParser
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt

import os
import logging
import sys
import multiprocessing as mp
from typing import List
from os.path import join
from functools import partial

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.DEBUG)


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
    n_cores = mp.cpu_count() if n_cores is None else n_cores
    with mp.Pool(n_cores) as pool:
        result = pool.map(partial(func, **kwargs), lst)
    return result


def check_index(df, name):
    if df.index.name != name:
        df.set_index(name, inplace=True)


def concatenate_partitions(partitions, require_matching_metadata=True):
    """Concatenate the partitions of a dataset"""
    # concatenate observations
    adata = ad.concat(partitions, axis=0)
    if require_matching_metadata:
        var = partitions[0].var
        for p in partitions:
            assert p.var.columns.tolist() == (var.columns.tolist()),\
                "Static vars are not consistent across partitions, cannot be concatenated"
        adata.var = var
    return adata


def load_data(data_path):
    if os.path.isdir(data_path):
        partitions = [sc.read_h5ad(file) for file in list_files(data_path)]
        adata = concatenate_partitions(partitions)
    else:
        adata = sc.read_h5ad(data_path)
    return adata


def plot_dim_reduction_figures(adata, title=None):
    plt.style.use("default")
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 16))
    sc.pl.umap(adata, color="tissue", ax=axes[0,0], show=False)
    axes[0,0].set_title("UMAP Plot, tissue")
    axes[0,0].get_legend().remove()

    sc.pl.pca(adata, color="tissue", ax=axes[0,1], show=False)
    axes[0,1].set_title("PCA Plot, Tissue")

    sc.pl.umap(adata, color="original_cell_type", ax=axes[1,0], show=False)
    axes[1,0].set_title("UMAP Plot, Original Cell Type")
    axes[1,0].get_legend().remove()

    sc.pl.pca(adata, color="original_cell_type", ax=axes[1,1], show=False)
    axes[1,1].set_title("PCA Plot, Original Cell Type")

    if title is not None:
        fig.suptitle(title, size=30)
    return fig, axes


def plot_qc_figures(adata, title=None):
    plt.style.use("ggplot")
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 16))
    sc.pl.violin(adata, ['total_counts'], ax=axes[0,0], show=False)
    axes[0,0].set_title("Total Counts")
    
    sc.pl.violin(adata, ['n_genes_by_counts'], ax=axes[1,0], show=False)
    axes[1,0].set_title("N Genes by Counts")
    
    sc.pl.violin(adata, ['pct_counts_mt'], ax=axes[0,1], show=False)
    axes[0,1].set_title("Percentage of Mitochondrial Counts")

    sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts', ax=axes[1,1], show=False)
    axes[1,1].set_title("N Genes by Counts by Total Counts")

    if title is not None:
        fig.suptitle(title, size=30)
    return fig, axes


def write_adata_to_csv_buffered(adata, file, places=4, sep=",", buffer_size=1000):
    df = adata.to_df().T
    X = df.to_numpy().round(places)
    names = df.index.tolist()
    
    def compile_row_string(name, row):
        return sep.join([name] + [str(v) if v != 0 else "0" for v in row])
    
    with open(file, "w") as f:
        header = compile_row_string(df.index.name, range(1, X.shape[1]+1))
        f.write(header + "\n")
        buffer = []
        for i, (name, row) in enumerate(zip(names, X)):
            buffer.append(compile_row_string(name, row))
            if (i + 1) % buffer_size == 0:
                f.write("\n".join(buffer) + "\n")
                buffer = []
        
        # Write remaining rows in buffer
        if buffer:
            f.write("\n".join(buffer) + "\n")


def get_variability(adata):
    variability = sc.pp.highly_variable_genes(adata, inplace=False)
    variability.index = adata.var_names
    variability = variability[lambda df: ~df["dispersions_norm"].isna()].sort_values("dispersions_norm", ascending=False)
    return variability


def preprocess_data(adata, mito_thres, umi_min, umi_max, target_sum, save_path=None):
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

    # re-calculate qc metrics after filtering 
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, inplace=True)

    return adata


def make_aracne_counts(adata, n_sample, filename):
    # excludle invariable genes
    variability = get_variability(adata)
    adata = adata[:,variability.index]
    # subsample
    adata = sc.pp.subsample(adata, n_obs=n_sample, copy=True)
    write_adata_to_csv_buffered(adata, sep="\t", file=filename)
    

def main(args):
    adata = load_data(args.data_path)
    logger.info(f"Loaded dataset: {adata.shape[0]:,} cells, {adata.shape[1]:,} genes")

    adata = preprocess_data(
        adata=adata, 
        mito_thres=args.mito_thres, 
        umi_min=args.umi_min, 
        umi_max=args.umi_max, 
        target_sum=args.target_sum,
        save_path=args.counts_path
    )
    logger.info(f"Preprocessed dataset: {adata.shape[0]:,} cells, {adata.shape[1]:,} genes")

    # # save
    # check_index(adata.var, "feature_id")
    # write_adata_to_csv_buffered(adata, file=args.counts_path)

    make_aracne_counts(
        adata=adata,
        n_sample=args.aracne_n_sample,
        filename=args.aracne_counts_path
    )

    # calculate and stats/figures

    # calculate/save ranks
    


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    # preprocessing 
    parser.add_argument("--mito_thres", type=int, default=20)
    parser.add_argument("--umi_min", type=int, default=1000)
    parser.add_argument("--umi_max", type=int, default=1e5)
    parser.add_argument("--target_sum", type=int, default=1e6)
    parser.add_argument("--n_top_genes", type=int, default=None)
    parser.add_argument("--protein_coding", type=bool, default=True)

    # figures
    parser.add_argument("--produce_figures", action="store_true")
    parser.add_argument("--produce_stats", action="store_true")
    parser.add_argument("--save_h5ad", action="store_true")

    # aracne
    parser.add_argument("--aracne_n_sample", type=int, default=1000)

    # computation
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--n_cores", type=int, default=os.cpu_count())
    
    args = parser.parse_args()
    
    # define paths
    os.makedirs(args.out_dir, exist_ok=True)
    args.counts_path = join(args.out_dir, "counts.csv")
    args.ranks_path = join(args.out_dir, "ranks_raw.csv")
    args.aracne_counts_path = join(args.out_dir, "counts.tsv")
    args.h5ad_path = join(args.out_dir, "counts.h5ad")
    args.log_path = join(args.out_dir, "log.txt")
    logger.addHandler(logging.FileHandler(args.log_path)) 
    
    main(args)

