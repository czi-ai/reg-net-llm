import torch
from torch_geometric.data import Data, Dataset
from torch.utils.data import Dataset as torchDataset
from torch.utils.data import DataLoader as torchDataLoader
import numpy as np
import pandas as pd 

import os
from os.path import join
from typing import List
import warnings
from pathlib import Path
from multiprocessing import Pool
import argparse
import lightning.pytorch as pl
from torch_geometric.loader import DataLoader
from numpy.random import default_rng
import pickle

REG_VALS = "regulator.values"
TAR_VALS = "target.values"
MI_VALS = "mi.values"

# from scGraphLLM.graph_op import spectral_PE
from scGraphLLM._globals import * ## imported global variables are all caps 


rng = default_rng(42)
def save(obj, file):
    # The above code is using the `pickle` module in Python to serialize the object `obj` and write it
    # to a file specified by the variable `file` in binary mode. This allows the object to be saved to
    # a file and later deserialized to retrieve the original object.
    with open(file, "wb") as ofl:
        pickle.dump(obj, ofl)
    return 

def load(file):
    with open(file, "rb") as ifl:
        return pickle.load(ifl)

def run_save(
        network, 
        ranks, 
        global_gene_to_node, 
        cache_dir, 
        overwrite, 
        msplit, 
        valsg_split_ratio, 
        cell_type, 
        min_genes_per_graph=MIN_GENES_PER_GRAPH, 
        max_seq_length=None, 
        skipped=0, 
        ncells=0, 
        verbose=False
    ):
    """
    Assign local aracne graph to each cell and cache each cell
    """
    os.makedirs(join(cache_dir, msplit), exist_ok=True)
    # remove unknown genes
    ranks = ranks[ranks.columns[ranks.columns.isin(global_gene_to_node)]]
    # remove edges due to unknown genes
    network = network[
        network[REG_VALS].isin(global_gene_to_node) & 
        network[TAR_VALS].isin(global_gene_to_node)
    ]
    network_genes = list(set(network[REG_VALS].to_list() + network[TAR_VALS].to_list()))
    common_genes = list(set(network_genes).intersection(set(ranks.columns)))
    ranks = ranks.loc[:, common_genes]

    for i in range(ranks.shape[0]):
        if ncells % 1000 == 0:
            print(f"Processed {ncells} cells", end="\r")

        cell_number = ranks.index[i]
        
        if msplit == "valSG":
            rand = rng.random()
            if rand > valsg_split_ratio:
                split = "train"
            else:
                split = msplit
        else:
            split = msplit
        
        outfile = f"{cache_dir}/{split}/{cell_type}_{cell_number}.pt"
        if (os.path.exists(outfile)) and (not overwrite):
            ncells+=1
            continue
        
        cell: pd.Series = ranks.iloc[i, :]
        # filter out genes with 0 expression
        ZERO_EXPRESSION_RANK = 0
        cell = cell[cell != ZERO_EXPRESSION_RANK]

        if cell.shape[0] < min_genes_per_graph: # require a minimum number of expressed genes per cell 
            skipped += 1
            ncells+=1
            continue

        # enforce max sequence length
        if max_seq_length is not None and cell.shape[0] > max_seq_length:
            cell = cell.nlargest(n=max_seq_length)

        # Subset network to only include genes in the cell
        network_cell = network[
            network[REG_VALS].isin(cell.index) & 
            network[TAR_VALS].isin(cell.index)
        ]

        local_gene_to_node_index = {gene:i for i, gene in enumerate(cell.index)}

        with warnings.catch_warnings(): # Suppress annoying pandas warnings
            warnings.simplefilter("ignore") 
            edges = network_cell[[REG_VALS, TAR_VALS, MI_VALS]]
            edges[REG_VALS] = edges[REG_VALS].map(local_gene_to_node_index)
            edges[TAR_VALS] = edges[TAR_VALS].map(local_gene_to_node_index)

        edge_list = torch.tensor(np.array(edges[[REG_VALS, TAR_VALS]])).T
        edge_weights = torch.tensor(np.array(edges[MI_VALS]))
        node_indices = torch.tensor(np.array([(global_gene_to_node[gene], cell[gene]) for gene in cell.index]), dtype=torch.long) # should this be local_gene_to_node?
        data = Data(
            x=node_indices, 
            edge_index=edge_list, 
            edge_weight=edge_weights
        )
        
        torch.save(data, outfile)
        ncells += 1
        
        if verbose:
            try:
                torch.load(outfile)
                print(outfile)
            except:
                print(outfile, "-------- Failed")
        
    return (skipped, ncells)


def transform_and_cache_aracane_graph_ranks(aracne_outdir_info : List[List[str]], gene_to_node_file:str, cache_dir:str, overwrite:bool=False, single=False, valsg_split_ratio = 0.2): # 0.2 makes it closer to equal size btween SG and HOG
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        
        ## this file maps gene names to a index: "global" bc we want the same index for each gene across all experiments 
        global_gene_to_node = pd.read_csv(gene_to_node_file)
        global_gene_to_node = {row.gene_name:row.idx for _,row in global_gene_to_node.iterrows()}
        skipped = 0
        ncells = 0
        
        if single: # Run on a single cell-type
            print("Caching", aracne_outdir_info[0][0].split("/")[-2])
            skipped, ncells = run_save(aracne_outdir_info[0], global_gene_to_node, cache_dir, overwrite, valsg_split_ratio, skipped, ncells)
        # else:
        #     for i in aracne_outdir_info:
        #         skipped, ncells = run_save(i, global_gene_to_node, cache_dir, overwrite, valsg_split_ratio, skipped, ncells)

        print(f"\n**DONE**\nSkipped {skipped} cells")
        print(f"loaded {ncells} cells")


class GraphTransformerDataset(torchDataset):
    def __init__(self, cache_dir:str, dataset_name:str, mask_fraction = 0.1, debug:bool=False, ):
        self.debug = debug
        self.cached_files = [cache_dir+"/" + f for f in os.listdir(cache_dir) if f.endswith(".pt")]
        self.dataset_name = dataset_name
        self.mask_fraction = mask_fraction
        print(f"Cache Directory: {cache_dir}")
        print(f"Observation Count: {len(self):,}")

    def __len__(self):
        if self.debug:
            return 1000
        return len(self.cached_files)

    def __getitem__(self, idx):
        ## mask 5% as a gene only mask; mask 5% as a rank only mask ; mask 5% as both gene and rank mask
        data = torch.load(self.cached_files[idx], weights_only=False)
        node_indices = data.x
        ## for each mask type, create boolean mask of the same shape as node_indices
        if self.mask_fraction == 0:
            gene_mask = torch.zeros(node_indices.shape[0], dtype=torch.bool)
            rank_mask = torch.zeros(node_indices.shape[0], dtype=torch.bool)
            both_mask = torch.zeros(node_indices.shape[0], dtype=torch.bool)
        else:
            gene_mask = torch.rand(node_indices.shape[0]) < self.mask_fraction
            rank_mask = torch.rand(node_indices.shape[0]) < self.mask_fraction
            both_mask = torch.rand(node_indices.shape[0]) < self.mask_fraction
        
        # mask the tensors
        #node_indices[gene_mask, 0] = MASK_IDX
        node_indices[rank_mask, 1] = MASK_IDX + NUM_GENES
        node_indices[both_mask, :] = torch.tensor([MASK_IDX, MASK_IDX + NUM_GENES], 
                                                  dtype=node_indices.dtype)
        
        orig_gene_indices = node_indices[:, 0].clone()
        orig_rank_indices = node_indices[:, 1].clone()
        
        num_nodes = node_indices.shape[0]
        
        # graph positional encoding
        #spectral_pe = spectral_PE(edge_index=data.edge_index, num_nodes=node_indices.shape[0], k=64)
        
        return {
            "orig_gene_id" : orig_gene_indices, 
            "orig_rank_indices" : orig_rank_indices, 
            "gene_mask" : gene_mask, 
            "rank_mask" : rank_mask, 
            "both_mask" : both_mask,
            "edge_index": data.edge_index,
            "num_nodes": num_nodes,
            #"spectral_pe": spectral_pe,
            "dataset_name" : self.dataset_name
        }

class GraphTransformerDataModule(pl.LightningDataModule):
    def __init__(self, data_config, collate_fn=None):
        super().__init__()
        self.data_config = data_config
        self.collate_fn = collate_fn
        self.train_ds = GraphTransformerDataset(**data_config.train)
        self.val_ds = [GraphTransformerDataset(**val) for val in data_config.val]
        if data_config.run_test:
            self.test_ds = [GraphTransformerDataset(**test) for test in data_config.test]
    
    def train_dataloader(self):
        return torchDataLoader(self.train_ds, batch_size = self.data_config.batch_size, 
                               num_workers = self.data_config.num_workers, collate_fn=self.collate_fn)
    def val_dataloader(self):
        return [torchDataLoader(val_ds, batch_size = self.data_config.batch_size, 
                                num_workers = self.data_config.num_workers, collate_fn=self.collate_fn) for val_ds in self.val_ds]
    def test_dataloader(self):
        return [torchDataLoader(test_ds, batch_size = self.data_config.batch_size, 
                                num_workers = self.data_config.num_workers, collate_fn=self.collate_fn) for test_ds in self.test_ds]


if __name__ == "__main__":
    ## This portion lets you generate the cache for the data outside of the model training loop - took about ~1 hour on 5 cores for the pilot data set
    ## python scGraphLLM/data.py --aracane-outdir-md  /hpc/projects/group.califano/GLM/data/aracne_1024_outdir.csv --gene-to-node-file /hpc/projects/group.califano/GLM/data/cellxgene_gene2index.csv --cache-dir /hpc/projects/group.califano/GLM/data/pilotdata_1024 --num-proc 16
    parser = argparse.ArgumentParser()
    parser.add_argument("--aracane-outdir-md", type=str, help="File containing a list of aracne outdirs; `ls path/to/aracaneOutdirs/* > <input>` ")
    parser.add_argument("--gene-to-node-file", type=str, help="File containing gene to node index mapping")
    parser.add_argument("--cache-dir", type=str, help="Directory to store the processed data")
    parser.add_argument("--num-proc", type=int, help="Number of processes to use", default=1)
    parser.add_argument("--single-index", type=int, help="Index in --aracane-outdir-md to path to specific cell-type aracne for single cell-type caching (mainly used with parallelization)", default=0)
    parser.add_argument("--debug", action="store_true", default=False, help="Run in debug mode")
    args = parser.parse_args()
    single_index = args.single_index
    debug = args.debug
    num_proc = int(args.num_proc)
    
    # Read the data directories and their associated dataset label (train, valSG, valHG)
    aracane_metadata = pd.read_csv(args.aracane_outdir_md, names = ["aracne_out", "split"]).values
    
    # Create cache directories for each dataset (train, valSG, valHG)
    unique_subsets = np.unique(aracane_metadata[:,1])
    Path(f"{args.cache_dir}/train").mkdir(parents=True, exist_ok=True) # Train is not in the outdirs csv as (the way the code is written) train data is pulled from the valSG data
    for subset in unique_subsets:
        Path(f"{args.cache_dir}/{subset}").mkdir(parents=True, exist_ok=True)
    sub_lists = np.array_split(aracane_metadata, num_proc)
    args_list = [(sub_list, args.gene_to_node_file, args.cache_dir) for sub_list in sub_lists]
        
    if debug:
        print("DEBUG")
        for arg in args_list:
            transform_and_cache_aracane_graph_ranks(*arg)
    elif single_index: # Single is equal to any non-zero value (the index)
        print("SINGLE CACHE")
        transform_and_cache_aracane_graph_ranks([aracane_metadata[single_index-1]], args.gene_to_node_file, args.cache_dir, single=True)
    else:
        print("RUNNING MULTI-THREADED")
        with Pool(num_proc) as p:
            p.starmap(transform_and_cache_aracane_graph_ranks, args_list)
