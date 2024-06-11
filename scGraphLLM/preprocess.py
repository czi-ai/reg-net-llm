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
import pyviper
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt

import pandas as pd
import os
import logging
import sys
import warnings
import multiprocessing as mp
from typing import List
from os.path import join
from functools import partial

warnings.filterwarnings("ignore")
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
        index_name = df.index.name if df.index.name is not None else "index"
        header = compile_row_string(index_name, range(1, X.shape[1]+1))
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
    """Compute the variability metrics for adata, filtering out non-variable columns"""
    variability = sc.pp.highly_variable_genes(adata, inplace=False)
    variability.index = adata.var_names
    variability = variability[lambda df: ~df["dispersions_norm"].isna()].sort_values("dispersions_norm", ascending=False)
    return variability


def calculate_sparsity(counts):
    """Calculate the sparsity of an Counts matrix"""
    if isinstance(counts, sc.AnnData):
        counts = counts.to_df().to_numpy()
    elif isinstance(counts, pd.DataFrame):
        counts = counts.to_numpy()
    return (counts != 0).mean()


def get_samples(adata, index_vars, assign_sample_id=True):
    """Compute samples and sizes for samples, as indexed by tuplets of `index_vars`"""
    if assign_sample_id:
        adata.obs["sample_id"] = adata.obs[index_vars].apply(lambda row: "_".join(row), axis=1)
    samples_groupby = adata.obs.groupby(index_vars)
    samples = samples_groupby.size()[lambda size: size > 0]
    return samples


def preprocess_data(
        adata, 
        mito_thres, 
        umi_min, 
        umi_max, 
        target_sum, 
        max_perc_umi_filtering=0.10):
    
    # calculate qc metrics
    adata.var["mt"] = adata.var_names.map(lambda n: str(n).upper().startswith("MT"))
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, inplace=True)
    
    if max_perc_umi_filtering is not None:
        assert 0 <= max_perc_umi_filtering <= 1, "max_perc_umi_filtering must be within (0 and 1)."
        umi_lb, umi_ub = adata.obs["total_counts"].quantile([max_perc_umi_filtering/2, 1-max_perc_umi_filtering/2])
        umi_max = max(umi_max, umi_ub)
        umi_min = min(umi_min, umi_lb)

    # store raw
    adata.raw = adata

    # apply filters
    adata = adata[adata.obs["pct_counts_mt"] < mito_thres]
    sc.pp.filter_cells(adata, min_counts=umi_min)
    sc.pp.filter_cells(adata, max_counts=umi_max)
    
    # normalize
    adata.raw = adata
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
    if adata.shape[0] > n_sample:
        adata = sc.pp.subsample(adata, n_obs=n_sample, copy=True)

    # write_adata_to_csv_buffered(adata, sep="\t", file=filename)
    

def make_metacells(adata, target_depth, compression, by_cluster, random_state):
    if by_cluster:
        sc.tl.pca(adata, svd_solver='arpack', random_state=random_state)
        sc.pp.neighbors(adata)
        sc.tl.louvain(adata, key_added="cluster", flavor='vtraag')
        clusters = adata.obs["cluster"].value_counts()
        logger.info(f"Detected {len(clusters)} clusters, ranging from {clusters[-1]:,} to {clusters[0]:,} cells")

    metacells = {}
    for name in clusters.index:
        cluster = adata[adata.obs["cluster"] == name]
        sc.pp.scale(cluster)
        pyviper.pp.repr_metacells(
            cluster, 
            counts=None,
            pca_slot="X_pca", 
            dist_slot="corr_dist", 
            size=int(clusters[name] * compression), 
            min_median_depth=target_depth, 
            clusters_slot=None,
            key_added=f"metacells"
        )
        metacells[name] = cluster.uns["metacells"]
        metacells[name].attrs["sparsity"] = calculate_sparsity(metacells[name])
        del cluster
    
    metacells_array = pd.concat([metacells.get(name) for name in sorted(metacells.keys())]).to_numpy()
    metacells_clusters = sum([[name] * metacells.get(name).shape[0] for name in sorted(metacells.keys())], [])
    metacells_obs = pd.DataFrame({"cluster": metacells_clusters})
    metacells_adata = sc.AnnData(metacells_array)
    metacells_adata.obs = metacells_obs
    metacells_adata.var = adata.var

    return metacells_adata


def rank(args):
    logger.info(f"Performing Rank Operation... [{args.aracne_counts_path}]")
    counts_df = pd.read_csv(f"{args.aracne_counts_path}", sep="\t")
    df = counts_df.loc[:, counts_df.columns != "feature_name"]
    rank_df_raw = df.rank(axis=1, method="dense") +1
    rank_df_raw.to_csv(f"{args.ranks_path}")


def main(args):
    adata = load_data(args.data_path)
    adata = adata[adata.obs["is_primary_data"],:]
    adata.obs.reset_index(inplace=True)
    adata.var.reset_index(inplace=True)
    spar = calculate_sparsity(adata)
    logger.info(f"Processed dataset: ({adata.shape[0]:,} cells, {adata.shape[1]:,} genes) with sparsity = {spar:.2f}")

    adata = preprocess_data(
        adata=adata, 
        mito_thres=args.mito_thres, 
        umi_min=args.umi_min, 
        umi_max=args.umi_max,
        max_perc_umi_filtering=args.max_perc_umi_filtering,
        target_sum=args.target_sum
    )
    spar = calculate_sparsity(adata)
    logger.info(f"Loaded dataset: ({adata.shape[0]:,} cells, {adata.shape[1]:,} genes) with sparsity = {spar:.2f}")

    samples = get_samples(adata, index_vars=["dataset_id", "donor_id", "tissue"])
    logger.info(f"Detected {samples.size} samples")

    metacells = make_metacells(
        adata=adata,
        target_depth=args.metacells_target_depth, 
        by_cluster=args.metacells_by_cluster, 
        compression=args.metacells_compression,
        random_state=args.random_state
    )
    logger.info(f"Made {metacells.shape[0]:,} meta cells with sparsity = {calculate_sparsity(metacells):2f}")

    # make_aracne_counts(
    #     adata=metacells,
    #     n_sample=args.aracne_n,
    #     save_dir=args.aracne_dir
    # )

    # calculate and stats/figures

    # calculate/save ranks
    # rank(args)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    # preprocessing 
    parser.add_argument("--mito_thres", type=int, default=20)
    parser.add_argument("--umi_min", type=int, default=1000)
    parser.add_argument("--umi_max", type=int, default=1e5)
    parser.add_argument("--max_perc_umi_filtering", type=float, default=0.10)
    parser.add_argument("--target_sum", type=int, default=1e6)
    parser.add_argument("--n_top_genes", type=int, default=None)
    parser.add_argument("--protein_coding", type=bool, default=True)
    
    parser.add_argument("--index_vars", nargs="+")
    parser.add_argument("--metacells_by_cluster", action="store_true")
    parser.add_argument("--metacells_target_depth", type=float, default=10000)
    parser.add_argument("--metacells_compression", type=float, default=0.2)

    # figures
    parser.add_argument("--produce_figures", action="store_true")
    parser.add_argument("--produce_stats", action="store_true")
    parser.add_argument("--save_h5ad", action="store_true")

    # aracne
    parser.add_argument("--aracne_n", type=int, default=1000)

    # computation
    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--n_cores", type=int, default=os.cpu_count())
    
    args = parser.parse_args()
    
    # define paths 
    args.counts_path = join(args.out_dir, "counts.csv")
    args.ranks_path = join(args.out_dir, "ranks_raw.csv")
    args.aracne_dir = join(args.out_dir, "aracne")
    args.aracne_counts_path = join(args.aracne_dir, "counts.tsv")
    args.h5ad_path = join(args.out_dir, "counts.h5ad")
    args.log_path = join(args.out_dir, "log.txt")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.aracne_dir, exist_ok=True)

    logger.addHandler(logging.FileHandler(args.log_path)) 
    
    main(args)

