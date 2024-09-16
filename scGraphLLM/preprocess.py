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

import numpy as np
import pandas as pd
import os
import logging
import sys
import warnings
import json
from datetime import date
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


def load_data(data_path, var_index_name=None):
    if os.path.isdir(data_path):
        partitions = [sc.read_h5ad(file) for file in list_files(data_path)]
        adata = concatenate_partitions(partitions)
    else:
        adata = sc.read_h5ad(data_path)
    if var_index_name is not None:
        check_index(adata.var, name=var_index_name)
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


def plot_qc_figures(adata, title=None, fig_path=None):
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, inplace=True)
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

    if fig_path is not None:
        plt.savefig(fig_path)
        plt.close()
        
    return fig, axes


def plot_dim_reduction_by_sample_cluster(adata, title=None, level_limit=100, n_limit=20000, fig_path=None):
    """Plot UMAP projections that colored by sample and by cluster """
    if not "X_umap" in adata.obsm:
        sc.tl.umap(adata)
    cluster_levels = adata.obs["cluster"].value_counts(ascending=False)[:level_limit].index
    samples_levels = adata.obs["sample_id"].value_counts(ascending=False)[:level_limit].index
    adata_umap = adata[(
        adata.obs["cluster"].isin(cluster_levels) &
        adata.obs["sample_id"].isin(samples_levels)), :]
    if adata_umap.shape[0] > n_limit:
        adata_umap = sc.pp.subsample(adata, n_obs=n_limit)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
    sc.pl.umap(adata_umap, color="cluster", ax=axes[0], show=False)
    axes[0].set_title("UMAP Plot, Cluster")
    axes[0].get_legend().remove()
    sc.pl.umap(adata_umap, color="sample_id", ax=axes[1], show=False)
    axes[1].set_title("UMAP Plot, Samples")
    axes[1].get_legend().remove()
    if title is not None:
        fig.suptitle(title, size=30)

    if fig_path is not None:
        plt.savefig(fig_path)
        plt.close()

    return fig, axes


def write_adata_to_csv_buffered(adata, file, places=4, sep=",", buffer_size=1000):
    """Write an AnnData object of single cell data to disk for ARACNe inference"""
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


def qc_metrics_dict(adata):
    nnz = adata.X != 0
    return {
        "n_cells": adata.shape[0],
        "n_genes": adata.shape[1],
        "sparsity": nnz.mean().round(4).item(),
        "mean_umi": adata.X.sum(axis=1).mean().round(4).item(),
        "mean_nnz": nnz.sum(axis=1).mean().round(4).item()
    }


def calculate_sparsity(counts):
    """Calculate the sparsity of an Counts matrix"""
    if isinstance(counts, sc.AnnData):
        counts = counts.to_df().to_numpy()
    elif isinstance(counts, pd.DataFrame):
        counts = counts.to_numpy()
    return (counts != 0).mean()


def get_samples(adata, index_vars=None):
    """Compute samples and sizes for samples, as indexed by tuplets of `index_vars`"""
    logger.info(f"Detecting samples indexed by variables {index_vars}...")
    if index_vars is None or len(index_vars) == 0:
        adata.obs["sample_id"] = "0"
    else:
        index_vars = sorted(index_vars)
        adata.obs["sample_id"] = pd.Categorical(adata.obs[index_vars].apply(lambda row: "_".join(row), axis=1))
        
    samples = adata.obs\
        .groupby("sample_id")\
        .size()[lambda size: size > 0]\
        .sort_values(ascending=False)
    
    qc = samples.describe().astype(float).to_dict()
    return samples, qc


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

    return adata, qc_metrics_dict(adata)


def make_aracne_counts(adata, min_n_sample=250, max_n_sample=500, min_perc_nz=0.001, aracne_dir=None):
    """Make counts ARACNe inference for each cluster
    """
    clusters = dict()
    genes = set()
    for key in adata.obs["cluster"].unique():
        cluster = adata[adata.obs["cluster"] == key,:].copy()
        
        # exclude genes not sufficiently represented
        perc_nz = cluster.X.sum(axis=0) / cluster.shape[1]
        cluster = cluster[:, perc_nz > min_perc_nz]

        # enforce sample bounds
        if cluster.shape[0] > max_n_sample:
            sc.pp.subsample(cluster, n_obs=max_n_sample)
        elif cluster.shape[0] < min_n_sample:
            logger.info(f"Cluster {key} has insufficient sample size of {cluster.shape[0]:,} for ARACNe inference")
            continue
        clusters[key] = cluster.shape
        genes = genes.union(cluster.var_names.tolist())
        logger.info(f"Cluster {key} has {cluster.shape[0]:,} cells and {cluster.shape[1]:,} genes for ARACNe inference")
        
        # save
        write_adata_to_csv_buffered(
            cluster, 
            sep="\t", 
            file=join(aracne_dir, f"counts_{key}.tsv"), 
            buffer_size=int(min_n_sample/2))
        del cluster

    # write all 
    aracne_adata = adata[adata.obs["cluster"].isin(clusters), adata.var_names.isin(genes)]
    write_adata_to_csv_buffered(
        aracne_adata, 
        sep="\t", 
        file=join(aracne_dir, f"counts.tsv"), 
        buffer_size=1000)

    return {
        "count": len(clusters),
        "size": clusters
    }


def get_clusters(adata, random_state, qc=True):
    sc.tl.pca(adata, svd_solver='arpack', random_state=random_state)
    sc.pp.neighbors(adata)
    sc.tl.louvain(adata, key_added="cluster", flavor='vtraag')
    return {
        "count": adata.obs["cluster"].nunique(),
        "size": adata.obs["cluster"].value_counts().to_dict()
    }


def make_metacells(adata, target_depth, compression, target_sum, random_state, save_path=None, qc=True):
    """Make meta cells within clusters"""
    assert "cluster" in adata.obs.columns, "Making metacells requires cluster info"
    clusters = adata.obs["cluster"].value_counts()
    metacells = {}
    for name in clusters.index:
        cluster = adata[adata.obs["cluster"] == name].copy()
        try:
            sc.pp.scale(cluster)
            pyviper.pp.repr_metacells(
                cluster, 
                counts=None,
                pca_slot="X_pca", 
                dist_slot="corr_dist", 
                size=int(clusters[name] * compression), 
                min_median_depth=target_depth, 
                clusters_slot=None,
                key_added=f"metacells",
                seed=random_state,
                verbose=False
            )

            metacells[name] = cluster.uns["metacells"]
            metacells[name].attrs["sparsity"] = calculate_sparsity(metacells[name])
        except:
            logger.info(f"(!) ----> Cluster {name} metacell processing FAILED!")
        else:
            logger.info(f"Cluster {name} metacell processing succeeded!")
        finally:
            del cluster
    
    metacells_array = pd.concat([metacells.get(name) for name in sorted(metacells.keys())]).to_numpy()
    metacells_clusters = sum([[name] * metacells.get(name).shape[0] for name in sorted(metacells.keys())], [])
    metacells_obs = pd.DataFrame({"cluster": metacells_clusters})
    metacells_adata = sc.AnnData(metacells_array)
    metacells_adata.obs = metacells_obs
    metacells_adata.var = adata.var
    
    if save_path is not None:
        metacells_adata.write_h5ad(save_path)

    return metacells_adata, qc_metrics_dict(metacells_adata)


def rank(metacells, args, plot=False):
    n_bins = args.n_bins

    df = metacells.to_df()
    rank_bins = np.zeros_like(df, dtype=np.int64)
    df = df.replace(0, np.nan) # Replace zeros with NaN so they are not considered in the ranking

    if plot:
        plot_gene_counts(df, save_to=args.out_dir)
    
    df_ranked = df.rank(axis=1, method="first", ascending=False).replace(np.nan, 0)
    for i in range(df_ranked.shape[0]): # Iterate through rows (single cells)
        row = df_ranked.iloc[i].to_numpy(dtype=int)
        non_zero = row.nonzero()
        if len(row[non_zero]) != 0: # Make sure row is not all zeros/nans (expressionless)
            bins = np.quantile(row[non_zero], np.linspace(0, 1, n_bins-1))
            bindices = np.digitize(row[non_zero], bins)
            rank_bins[i, non_zero] = bindices

    ranks = pd.DataFrame(rank_bins, columns=df.columns)
    ranks.to_csv(f"{args.ranks_path}", index=False, header=True) # Save ranks_raw.csv file to cell-type-specific directory

    # Info
    df_info = df.count(axis=1)
    rank_info = {
        "n_bins": n_bins,
        "min_genes_per_metacell": int(df_info.min()),
        "max_genes_per_metacell": int(df_info.max()),
        "median_genes_per_metacell": df_info.median(),
        "mean_genes_per_bin": round(df.count(axis=0).sum()/(df.dropna(how="all").shape[0] * n_bins), 2),
        "num_unique_highest_expressed": int(np.sum(np.any(np.isin(rank_bins, [99]), axis=0))) # Number of genes that were in the highest bin rank across metacells
    }

    return ranks, rank_info

def plot_gene_counts(df, save_to, upper_range=2500):
    dfmax = df.count(axis=1)
    counts = []
    for i in range(df.shape[1]):
        mask = (dfmax == i)
        indices = df.index[mask]
        s0 = pd.Series(indices)
        counts.append(s0.shape[0])

    # Create the bar plot
    plt.figure(figsize=(10, 6)) 
    plt.bar(range(upper_range), counts[:upper_range])

    # Customize the plot
    cell_type = save_to.split("/")[-1]
    plt.title(f'Gene Expression Distribution ({cell_type})')
    plt.xlabel('# Expressed Genes per Metacell')
    plt.ylabel('# Metacells')

    plt.savefig(f"{save_to}/gene_counts_distribution.png")
    plt.close()


def main(args):
    logger.info(f"Preprocessing cells on {date.today()} with the following configuration:")
    logger.info(json.dumps(args.__dict__, indent=4))
    
    adata = load_data(args.data_path, var_index_name=args.var_index_name)
    qc_initial = qc_metrics_dict(adata)
    logger.info(f"Loaded dataset: ({adata.shape[0]:,} cells, {adata.shape[1]:,} genes) with sparsity = {qc_initial['sparsity']:.2f}")

    # CellXGene specific processing
    adata = adata[adata.obs["is_primary_data"],:]
    if adata.shape[0] == 0:
        logger.info(f"No 'primary data' is contained in this cell-type data: adata has no rows/cells")
    else:
        adata.obs.reset_index(inplace=True)
        adata.var.reset_index(inplace=True)
        check_index(adata.var, "feature_id")

        adata, qc_processed = preprocess_data(
            adata=adata, 
            mito_thres=args.mito_thres, 
            umi_min=args.umi_min, 
            umi_max=args.umi_max,
            max_perc_umi_filtering=args.max_perc_umi_filtering,
            target_sum=args.target_sum)
        logger.info(f"Processed dataset: ({adata.shape[0]:,} cells, {adata.shape[1]:,} genes) with sparsity = {qc_processed['sparsity']:.2f}")

        samples, qc_samples = get_samples(
            adata=adata,
            index_vars=args.sample_index_vars)
        logger.info(f"Detected {qc_samples['count']:,} samples indexed by variables: {args.sample_index_vars}")

        qc_cluster = get_clusters(
            adata, 
            random_state=args.random_state)
        logger.info(f"Detected {qc_cluster['count']} clusters")

        metacells, qc_metacells = make_metacells(
            adata=adata,
            target_depth=args.metacells_target_depth,
            compression=args.metacells_compression,
            target_sum=args.target_sum,
            save_path=(args.meta_path if args.save_metacells else None),
            random_state=args.random_state)
        logger.info(f"Made {metacells.shape[0]:,} meta cells with sparsity = {qc_metacells['sparsity']:2f}")

        qc_aracne = make_aracne_counts(
            adata=metacells,
            min_n_sample=args.aracne_min_n,
            max_n_sample=args.aracne_max_n,
            min_perc_nz=args.aracne_min_perc_nz,
            aracne_dir=args.aracne_dir)
        logger.info(f"{qc_aracne['count']} clusters from {qc_cluster} with sufficient samples (> {args.aracne_min_n}) for ARACNe inference.")

        # calculate/save ranks
        ranks, rank_info = rank(metacells, args, plot=False) # Returns: pandas dataframe with metacell x genes: values are ranking bin number | AND | rank_info JSON element

        # Save Statistics & config
        info = {
            "config": args.__dict__,
            "stats": {
                "initial": qc_initial,
                "preprocessed": qc_processed,
                "samples": qc_samples,
                "clusters": qc_cluster,
                "metacells": qc_metacells,
                "aracne": qc_aracne,
                "ranks": rank_info
            },
        }
        with open(args.info_path, 'w') as file:
            json.dump(info, file, indent=4)

        if args.produce_figures:
            plot_dim_reduction_by_sample_cluster(
                adata=adata,
                title="UMAP Plots by Cluster and Sample",
                fig_path=join(args.fig_dir, "umap.png")
            )
            plot_qc_figures(
                adata=metacells, 
                title="QC Figures for Metacells", 
                fig_path=join(args.fig_dir, "qc_metacells.png")
            )
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--var_index_name", type=str, default=None)
    parser.add_argument("--mito_thres", type=int, default=20)
    parser.add_argument("--umi_min", type=int, default=1000)
    parser.add_argument("--umi_max", type=int, default=1e5)
    parser.add_argument("--max_perc_umi_filtering", type=float, default=0.10)
    parser.add_argument("--target_sum", type=int, default=1e6)
    parser.add_argument("--n_top_genes", type=int, default=None)
    parser.add_argument("--protein_coding", type=bool, default=True)
    parser.add_argument("--sample_index_vars", nargs="+")
    parser.add_argument("--metacells_by_cluster", action="store_true")
    parser.add_argument("--metacells_target_depth", type=float, default=10000)
    parser.add_argument("--metacells_compression", type=float, default=0.2)
    parser.add_argument("--aracne_min_n", type=int, default=250)
    parser.add_argument("--aracne_max_n", type=int, default=1000)
    parser.add_argument("--aracne_min_perc_nz", type=int, default=0.01)
    parser.add_argument("--n_bins", type=int, default=100)
    # figures
    parser.add_argument("--produce_figures", action="store_true")
    parser.add_argument("--produce_stats", action="store_true")
    parser.add_argument("--save_metacells", action="store_true")
    # computation
    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--n_cores", type=int, default=os.cpu_count())
    args = parser.parse_args()
    
    # define paths 
    args.counts_path = join(args.out_dir, "counts.csv")
    args.ranks_path = join(args.out_dir, "rank_raw.csv")
    args.meta_path = join(args.out_dir, "metacells.h5ad")
    args.info_path = join(args.out_dir, "info.json")
    args.log_path = join(args.out_dir, "log.txt")
    args.aracne_dir = join(args.out_dir, "aracne/counts")
    args.fig_dir = join(args.out_dir, "figure")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.aracne_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)
    logger.addHandler(logging.FileHandler(args.log_path)) 
    
    main(args)