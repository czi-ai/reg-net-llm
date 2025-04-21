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
from pyviper._load._load_regulators import load_TFs, load_coTFs, load_sig, load_surf
from pyviper._load._load_translate import load_human2mouse
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import os
import re
import logging
import sys
import warnings
import json
import importlib.util
from datetime import date, datetime
import multiprocessing as mp
from typing import List
from os.path import join, abspath, dirname
from functools import partial

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.DEBUG)

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

genes_names = pd.read_csv("/hpc/projects/group.califano/GLM/data/gene-name-map.csv", index_col=0)
ENSG_PROTEIN_CODING = set(genes_names["ensg.values"])


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


def get_regulators(types=["tf", "cotf"], name="ENSG"):
    """
    Return all regulators of provided types:
        - tf
        - cotf
        - sig
        - surf
    """
    regs = []
    if "tf" in types:
        regs += load_TFs(species="human")
    elif "cotf" in types:
        regs += load_coTFs(species="human")
    elif "sig" in types:
        regs += load_sig(species="human")
    elif "surf" in types:
        regs += load_surf(species="human")

    if name == "ENSG":
        symbol2ensembl = load_human2mouse().set_index("human_symbol")["human_ensembl"].to_dict()
        regs = pd.Series([symbol2ensembl.get(reg) for reg in regs])
        regs = regs.pipe(lambda x: x[~x.isna()])

    return regs


def make_aracne_counts(
        adata, 
        min_n_sample=250, 
        max_n_sample=500, 
        min_perc_nz=0.001,
        top_n_hvg=None,
        regulators=["tf", "cotf"],
        aracne_dir=None
    ):
    """Make counts ARACNe inference for each cluster
    """
    regulators = get_regulators(types=regulators)
    clusters = dict()
    genes = set()
    for key in adata.obs["cluster"].unique():
        cluster = adata[adata.obs["cluster"] == key,:].copy()
        
        # enforce top_n_hvg
        if top_n_hvg is not None:
            hvg = sc.pp.highly_variable_genes(cluster, flavor="seurat", n_top_genes=top_n_hvg, inplace=False, subset=True)
            
            hvg_union_regs = hvg_union_regs = set(hvg.index.to_list()).union(set(regulators))
            cluster = cluster[:, cluster.var_names.isin(hvg_union_regs)]
        else:
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


def make_metacells(
        adata, 
        target_depth,
        compression, 
        target_sum, 
        random_state, 
        save_path=None, 
        size=None,
        groupby_var="cluster",
        qc=True):
    """Make meta cells within clusters"""
    assert groupby_var in adata.obs.columns, f"Making metacells requires group by var {groupby_var}"
    groups = adata.obs[groupby_var].value_counts()
    metacells = {}
    for name in groups.index:
        cluster = adata[adata.obs[groupby_var] == name].copy()
        try:
            sc.pp.scale(cluster)
            pyviper.pp.repr_metacells(
                cluster, 
                counts=None,
                pca_slot="X_pca", 
                dist_slot="corr_dist", 
                size=int(groups[name] * compression) if size is None else size, 
                min_median_depth=target_depth, 
                clusters_slot=None,
                key_added=f"metacells",
                seed=random_state,
                verbose=False
            )
            metacells[name] = cluster.uns["metacells"]
            metacells[name].attrs["sparsity"] = calculate_sparsity(metacells[name])
        except Exception as e:
            logger.info(f"(!) ----> Cluster {name} metacell processing FAILED! {e}")
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
    

def quantize_cells(gex, n_bins, method="quantile"):
    # Create a dataframe of the same size as df with all zeros
    expression_bins = np.zeros(gex.shape, dtype=np.int16)
    
    for i in range(gex.shape[0]): # Iterate through rows (single cells)
        cell = gex.iloc[i].to_numpy(dtype=int) # Get single cell expression
        non_zero = cell.nonzero()[0] # Get indices where expression is non-zero
        
        # "Binnify" the rankings
        if np.unique(cell[non_zero]).shape[0] > 1: # More than one expression value - should be in all cases
            if method == "quantile":
                bins = np.quantile(cell[non_zero], np.linspace(0, 1, n_bins))[1:-1] # Get the bin edges, this gets n_bins-2 bin edges - [1:-1] so anything under/over the bottom/top edge gets added to the same bin
            else: 
                bins = np.linspace(min(cell[non_zero]), max(cell[non_zero]), n_bins)
            expression_bins[i, non_zero] = np.digitize(cell[non_zero], bins, right=True) + 1 # Assign bins, add 1 as the bins are numbered from 0 up and we have already reserved 0 for zero-expression. right=True to differentiate high expression over low expression
        
        elif np.unique(cell[non_zero]).shape[0] == 1: # Edge case: If the cell has two expression levels (zero and x) - this shouldn't really happen
            expression_bins[i, non_zero] = np.zeros(cell[non_zero].shape[0]) + round(n_bins/2) # Set to expressed genes to median bin value
        else: # No update needed if cell is all zeros
            pass
    
    # Convert the binned expressions to a pandas dataframe
    expression_bins = pd.DataFrame(expression_bins, columns=gex.columns, index=gex.index)
    return expression_bins


def quantize(cells, n_bins, save_path=None):    
    # Get the z-score test statistic   
    cells = cells.to_df()
    bins = quantize_cells(cells, n_bins)
    
    # Save bins_raw.csv file to cell-type-specific directory
    if save_path is not None:
        bins.to_csv(save_path, index=False, header=True)

    # Save the bin info
    cells_info = cells.count(axis=1)
    binfo = {
        "n_bins": n_bins,
        "min_genes_per_cell": int(cells_info.min()),
        "max_genes_per_cell": int(cells_info.max()),
        "median_genes_per_cell": cells_info.median(),
        "mean_genes_per_bin": round(cells.count(axis=0).sum()/(cells.dropna(how="all").shape[0] * n_bins), 2),
        "num_unique_highest_expressed": int(np.sum(np.any(np.isin(bins, [99]), axis=0))) # Number of genes that were in the highest bin rank across cells
    }

    return bins, binfo

def main(args):
    logger.info(f"Preprocessing cells on {date.today()} with the following configuration:")
    logger.info(json.dumps(args.__dict__, indent=4))
    info = {"config": args.__dict__}

    if "preprocess" in args.steps:
        adata = load_data(args.data_path, var_index_name=args.var_index_name)
        qc_initial = qc_metrics_dict(adata)
        info["initial"] = qc_initial
        logger.info(f"Loaded dataset: ({adata.shape[0]:,} cells, {adata.shape[1]:,} genes) with sparsity = {qc_initial['sparsity']:.2f}")
        
        # CellXGene specific processing
        if args.dataset == "cell_x_gene":
            adata = adata[adata.obs["is_primary_data"],:]
            adata.obs.reset_index(inplace=True)
            adata.var.reset_index(inplace=True)
            check_index(adata.var, "feature_id")

        adata, qc_processed = preprocess_data(
            adata=adata, 
            mito_thres=args.mito_thres, 
            umi_min=args.umi_min, 
            umi_max=args.umi_max,
            max_perc_umi_filtering=args.max_perc_umi_filtering,
            target_sum=args.target_sum
        ) 
        info["preprocessed"] = qc_processed
        logger.info(f"Processed dataset: ({adata.shape[0]:,} cells, {adata.shape[1]:,} genes) with sparsity = {qc_processed['sparsity']:.2f}")

        samples, qc_samples = get_samples(
            adata=adata,
            index_vars=args.sample_index_vars
        )
        info["samples"] = qc_samples
        logger.info(f"Detected {qc_samples['count']:,} samples indexed by variables: {args.sample_index_vars}")

        qc_cluster = get_clusters(
            adata, 
            random_state=args.random_state
        )
        info["clusters"] = qc_cluster
        logger.info(f"Detected {qc_cluster['count']} clusters")
        
        # Save preprocessed data
        adata.write_h5ad(args.cells_path)

        metacells, qc_metacells = make_metacells(
            adata=adata,
            target_depth=args.metacells_target_depth,
            compression=args.metacells_compression,
            size=args.metacells_size,
            target_sum=args.target_sum,
            groupby_var="sample_id" if args.aracne_by_sample else "cluster",
            save_path=(args.meta_path if args.save_metacells else None),
            random_state=args.random_state
        )
        info["metacells"] = qc_metacells
        logger.info(f"Made {metacells.shape[0]:,} meta cells with sparsity = {qc_metacells['sparsity']:2f}")  
    else:
        adata = sc.read_h5ad(args.cells_path)
        qc_adata = qc_metrics_dict(adata)
        logger.info(f"Loaded {adata.shape[0]:,} processed cells with sparsity = {qc_adata['sparsity']:2f}")  

        metacells = sc.read_h5ad(args.meta_path)
        qc_metacells = qc_metrics_dict(metacells)
        logger.info(f"Loaded {metacells.shape[0]:,} meta cells with sparsity = {qc_metacells['sparsity']:2f}")  

    if "quantize" in args.steps:
        # Returns: pandas dataframe with metacell x genes: values are ranking bin number | AND | rank_info JSON element
        print(f"Quantizing dataset into {args.n_bins} bins...")
        bins, qc_bins = quantize(
            adata,
            n_bins=args.n_bins, 
            save_path=args.bins_path
        )
        info["bins"] = qc_bins
        print("Quantization completed!")
    
    if "aracne" in args.steps:
        # aracne_cells = metacells if args.aracne_metacells else args.
        qc_aracne = make_aracne_counts(
            adata=metacells,
            min_n_sample=args.aracne_min_n,
            max_n_sample=args.aracne_max_n,
            min_perc_nz=args.aracne_min_perc_nz,
            top_n_hvg=args.aracne_top_n_hvg,
            regulators=args.aracne_regulators,
            aracne_dir=args.aracne_dir
        )
        info["aracne"] = qc_aracne
        logger.info(f"{qc_aracne['count']} clusters have sufficient samples (> {args.aracne_min_n}) for ARACNe inference.")

    with open(args.info_path, "w") as file:
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
    parser.add_argument("--dataset", type=str, default="cell_x_gene")
    parser.add_argument("--perturbed", type=str, help="Is this a perturbation dataset? Perturbation information will be stored in caching", default=False)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--steps", nargs="+", default=["preprocess", "quantize", "aracne"])
    parser.add_argument("--var_index_name", type=str, default=None)
    parser.add_argument("--mito_thres", type=int, default=20)
    parser.add_argument("--umi_min", type=int, default=1000)
    parser.add_argument("--umi_max", type=int, default=1e5)
    parser.add_argument("--max_perc_umi_filtering", type=float, default=0.10)
    parser.add_argument("--target_sum", type=int, default=1e6)
    parser.add_argument("--n_top_genes", type=int, default=None)
    parser.add_argument("--protein_coding", type=bool, default=True)
    parser.add_argument("--sample_index_vars", nargs="+")
    parser.add_argument("--metacells_target_depth", type=float, default=10000)
    parser.add_argument("--metacells_compression", type=float, default=0.2)
    parser.add_argument("--metacells_size", type=int, default=None)
    parser.add_argument("--aracne_with_metacells", action="store_true")
    parser.add_argument("--aracne_by_sample", action="store_true")
    parser.add_argument("--aracne_min_n", type=int, default=250)
    parser.add_argument("--aracne_max_n", type=int, default=1000)
    parser.add_argument("--aracne_min_perc_nz", type=float, default=0.01)
    parser.add_argument("--aracne_regulators", nargs="+", default=["tfs", "cotfs"])
    parser.add_argument("--aracne_top_n_hvg", type=int, default=None)
    parser.add_argument("--aracne_dirname", type=str, default="aracne")
    parser.add_argument("--n_bins", type=int, default=250)
    # figures
    parser.add_argument("--produce_figures", action="store_true")
    parser.add_argument("--produce_stats", action="store_true")
    parser.add_argument("--save_metacells", action="store_true")
    # computation
    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--n_cores", type=int, default=os.cpu_count())
    args = parser.parse_args()
    
    # Make sure perturbed argument is in correct form
    args.perturbed = args.perturbed == "true"
    
    if args.sample_index_vars == ["null"]:
        args.sample_index_vars = ["dataset_id", "donor_id", "tissue"]
    
    # define paths 
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    args.bins_path = join(args.out_dir, "binned_expression.csv")
    args.meta_path = join(args.out_dir, "metacells.h5ad")
    args.cells_path = join(args.out_dir, "cells.h5ad")
    args.info_path = join(args.out_dir, f"info_{timestamp}.json")
    args.log_path = join(args.out_dir, f"log_{timestamp}.txt")
    args.aracne_dir = join(args.out_dir, args.aracne_dirname, "counts")
    args.fig_dir = join(args.out_dir, "figure")
    print(f"args.out_dir: {args.out_dir}")
    print(f"args.aracne_dir: {args.aracne_dir}")
    print(f"args.fig_dir: {args.fig_dir}")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.aracne_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)
    logger.addHandler(logging.FileHandler(args.log_path)) 
    
    main(args)