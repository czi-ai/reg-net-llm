import torch
from torch_geometric.data import Data as torchGeomData
from torch_geometric.data import Dataset as torchGeomDataset
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
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.loader import DataLoader
from numpy.random import default_rng
import pickle

REG_VALS = "regulator.values"
TAR_VALS = "target.values"
MI_VALS = "mi.values"
SCC_VALS = "scc.values"

# from scGraphLLM.graph_op import spectral_PE
from scGraphLLM._globals import * ## imported global variables are all caps 


rng = default_rng(42)

def send_to_gpu(data):
    if isinstance(data, torch.Tensor):
        return data.to('cuda')  # Send tensor to GPU
    elif isinstance(data, list):
        return [send_to_gpu(item) for item in data]  # Recursively process lists
    elif isinstance(data, dict):
        return {key: send_to_gpu(value) for key, value in data.items()}  # Recursively process dicts
    else:
        return data  # If not a tensor or list/dict, leave unchanged

def scglm_collate_fn(batch, inference=False):
    data = {
        "orig_gene_id": [], 
        "orig_rank_indices": [], 
        "gene_mask": [], 
        "rank_mask": [], 
        "both_mask": [], 
        "edge_index": [], 
        "num_nodes" :[], 
        "dataset_name": []
    }
    if inference:
        data["obs_name"] = []
    
    # Make a dictionary of lists from the list of dictionaries
    for b in batch:
        for key in data.keys():
            data[key].append(b[key])

    # Pad these dictionaries of lists
    for key in data.keys():
        if key in {"dataset_name", "edge_index", "num_nodes", "obs_name"}:
            continue
        elif key == "orig_gene_id":
            pad_value = PAD_GENE_IDX
        elif key == "orig_rank_indices":
            pad_value = PAD_RANK_IDX
        elif (key == "gene_mask") or (key == "rank_mask") or (key == "both_mask"):
            pad_value = False
        data[key] = pad_sequence(data[key], batch_first=True, padding_value=pad_value)

    return data

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

def run_cache(
        network: pd.DataFrame, 
        expression: pd.DataFrame, 
        global_gene_to_node, 
        cache_dir, 
        overwrite, 
        msplit, 
        valsg_split_ratio, 
        cell_type, 
        min_genes_per_graph=MIN_GENES_PER_GRAPH, 
        max_seq_length=None, 
        only_expressed_genes=True,
        with_edge_weights=True,
        skipped=0, 
        ncells=0, 
        verbose=False
    ):
    """
    Assign local ARACNe graph to each cell and cache each cell
    """
    os.makedirs(join(cache_dir, msplit), exist_ok=True)
    # remove unknown genes
    expression = expression[expression.columns[expression.columns.isin(global_gene_to_node)]]
    # remove edges due to unknown genes
    network = network[
        network[REG_VALS].isin(global_gene_to_node) & 
        network[TAR_VALS].isin(global_gene_to_node)
    ]
    network_genes = list(set(network[REG_VALS].to_list() + network[TAR_VALS].to_list()))
    common_genes = list(set(network_genes).intersection(set(expression.columns)))
    expression = expression.loc[:, common_genes]

    for i in range(expression.shape[0]):
        if ncells % 1000 == 0:
            print(f"Processed {ncells} cells", end="\r")

        cell_number = expression.index[i]
        
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
        
        cell: pd.Series = expression.iloc[i, :]

        if cell[cell != ZERO_IDX].shape[0] < min_genes_per_graph: # require a minimum number of expressed genes per cell 
            skipped += 1
            ncells+=1
            continue

        # filter out genes with 0 expression
        data = get_cell_data(network, global_gene_to_node, max_seq_length, only_expressed_genes, with_edge_weights, cell)
        
        torch.save(data, outfile)
        ncells += 1
        
        if verbose:
            try:
                torch.load(outfile)
                print(outfile)
            except:
                print(outfile, "-------- Failed")
        
    return (skipped, ncells)

def get_cell_data(network, global_gene_to_node, max_seq_length, only_expressed_genes, with_edge_weights, cell):
    if only_expressed_genes:
        cell = cell[cell != ZERO_IDX]

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
        edges = network_cell[[REG_VALS, TAR_VALS]]
        edges[REG_VALS] = edges[REG_VALS].map(local_gene_to_node_index)
        edges[TAR_VALS] = edges[TAR_VALS].map(local_gene_to_node_index)

    edge_list = torch.tensor(np.array(edges[[REG_VALS, TAR_VALS]])).T
    node_expression = torch.tensor(np.array([(global_gene_to_node[gene], cell[gene]) for gene in cell.index]), dtype=torch.long) # should this be local_gene_to_node?
        
    if with_edge_weights:
        edge_weights = torch.tensor(np.array(network_cell[MI_VALS]))
        data = torchGeomData(
            x=node_expression, 
            edge_index=edge_list, 
            edge_weight=edge_weights
        )
    else:
        data = torchGeomData(
            x=node_expression, 
            edge_index=edge_list
        )
        
    return data


def cache_aracane_and_bins(
        aracne_outdir_info : List[List[str]], 
        gene_to_node_file:str, 
        cache_dir:str, 
        overwrite:bool=False, 
        single=False, 
        valsg_split_ratio = 0.2 # 0.2 makes it closer to equal size between SG and HOG
    ):
    """ 
    Calls run_cache() function. Transforms and caches the cell-types' ARACNe graphs with their corresponding expression bins.

    Args:
        aracne_outdir_info (List[List[str]]): Array of shape (num cell-types, 2), will be (1, 2) if single == True, where the first element of each pair is the path to that cell-type's aracne directory (i.e. .../endocrine_cell/aracne_4096) and the second element is the dataset (valSG or valHOG)
        gene_to_node_file (str): Path to the file containing all ENSEMBL genes along with their associated token, this file also houses the PAD and MASK token assignments
        cache_dir (str): Path to which the cached files will be stored (in specified sub-directories that are created)
        overwrite (bool, optional): Defaults to False.
        single (bool, optional): Whether the provided information is to process a single cell-type (often used in parallelization where each cell-type is processed individually). Defaults to False.
        valsg_split_ratio (float, optional): Ratio of valSG-marked cells to store in the validation set rather than the training set. Defaults to 0.2.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        
        ## this file maps gene names to a index: "global" bc we want the same index for each gene across all experiments 
        global_gene_to_node = pd.read_csv(gene_to_node_file)
        global_gene_to_node = {row.gene_name:row.idx for _,row in global_gene_to_node.iterrows()}
        skipped = 0
        ncells = 0
        
        # msplit, ####
        # min_genes_per_graph=MIN_GENES_PER_GRAPH, ####
        # max_seq_length=None, ####
        
        for outdir_info in aracne_outdir_info: # If single==True, then this will only run once (for that single cell-type)
            cell_type = outdir_info[0].split('/')[-2] # Get the cell-type's name that is currently being cached
            print(f"Caching: {cell_type}...")
            
            aracne_out = outdir_info[0]
            network = pd.read_csv(aracne_out +"/consolidated-net_defaultid.tsv", sep = "\t") # Get the ARACNe network for this cell-type
            expression = pd.read_csv(str(Path(aracne_out).parents[0]) + "/binned_expression.csv") # Get expression (most likely in binned format)

            skipped, ncells = run_cache(
                network=network, 
                expression=expression, 
                global_gene_to_node=global_gene_to_node, 
                cache_dir=cache_dir, 
                overwrite=overwrite, 
                msplit=outdir_info[1], # Change this to seen_graph
                valsg_split_ratio=valsg_split_ratio, 
                cell_type=cell_type, 
                min_genes_per_graph=MIN_GENES_PER_GRAPH,
                max_seq_length=None,
                skipped=skipped, 
                ncells=ncells
            )

        print(f"\n**DONE**\nSkipped {skipped} cells")
        print(f"loaded {ncells} cells")


class GraphTransformerDataset(torchDataset):
    def __init__(self, cache_dir:str, dataset_name:str=None, mask_fraction=0.15, debug:bool=False, inference=False):
        self.debug = debug
        self.inference = inference
        self.cached_files = sorted([cache_dir+"/" + f for f in os.listdir(cache_dir) if f.endswith(".pt")])
        self.dataset_name = dataset_name
        self.mask_fraction = mask_fraction
        print(f"Cache Directory: {cache_dir}")
        print(f"Observation Count: {len(self):,}")

    def __len__(self):
        return len(self.cached_files)

    def __getitem__(self, idx):
        ## mask 5% as a gene only mask; mask 5% as a rank only mask ; mask 5% as both gene and rank mask
        data = torch.load(self.cached_files[idx], weights_only=False)
        return self._item_from_tokenized_data(data)

    def _item_from_tokenized_data(self, data: torchGeomData):
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
        # node_indices[gene_mask, 0] = MASK_GENE_IDX
        # node_indices[rank_mask, 1] = MASK_RANK_IDX
        # node_indices[both_mask, :] = torch.tensor([MASK_GENE_IDX, MASK_RANK_IDX], dtype=node_indices.dtype)
        
        # add CLS
        cls_token = torch.tensor([[CLS_GENE_IDX, CLS_TOKEN]], dtype=node_indices.dtype)
        node_indices = torch.cat([cls_token, node_indices], dim=0) # CLS can never be masked

        # add False to masks
        gene_mask = torch.cat([torch.tensor([False]), gene_mask])
        rank_mask = torch.cat([torch.tensor([False]), rank_mask])
        both_mask = torch.cat([torch.tensor([False]), both_mask])

        orig_gene_indices = node_indices[:, 0].clone()
        orig_rank_indices = node_indices[:, 1].clone()
        num_nodes = node_indices.shape[0] - 1 # discount the cls node
        edge_index = data.edge_index # edges index assumes the first gene token's index is 0

        # graph positional encoding
        #spectral_pe = spectral_PE(edge_index=data.edge_index, num_nodes=node_indices.shape[0], k=64)
        
        item = {
            "orig_gene_id" : orig_gene_indices, 
            "orig_rank_indices" : orig_rank_indices, 
            "gene_mask" : gene_mask, 
            "rank_mask" : rank_mask, 
            "both_mask" : both_mask,
            "edge_index": edge_index, # edges index 
            "num_nodes": num_nodes,
            #"spectral_pe": spectral_pe,
            "dataset_name" : self.dataset_name
        }
        
        if self.inference:
            item["obs_name"] = getattr(data, "obs_name", None)

        return item


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
            cache_aracane_and_bins(*arg)
    elif (single_index) or (single_index == 0): # An index value has been given - corresponding to a specific cell-type to be process individually
        print("SINGLE CACHE")
        cache_aracane_and_bins(aracane_metadata[single_index].reshape(1, 2), args.gene_to_node_file, args.cache_dir, single=True)
    else:
        print("RUNNING MULTI-THREADED")
        with Pool(num_proc) as p:
            p.starmap(cache_aracane_and_bins, args_list)
