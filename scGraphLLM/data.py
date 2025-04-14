import torch
from torch_geometric.data import Data, Dataset
from torch.utils.data import Dataset as torchDataset
from torch.utils.data import DataLoader as torchDataLoader
import numpy as np
import pandas as pd 
import anndata as ad
import random

import os
from typing import List
import warnings
from pathlib import Path
from multiprocessing import Pool
import argparse
import lightning.pytorch as pl
from torch_geometric.loader import DataLoader
from numpy.random import default_rng
import pickle

from graph_op import spectral_PE
from _globals import * ## imported global variables are all caps 


gene_to_node = pd.read_csv("/hpc/projects/group.califano/GLM/data/cellxgene_gene2index.csv", index_col=0)
INCLUDED_ENSG = gene_to_node.index[2:]

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

def run_save(i, global_gene_to_node, cache_dir, overwrite, valsg_split_ratio, skipped, ncells):
    aracne_out = i[0]
    msplit = i[1]
    cell_type = aracne_out.split("/")[-2]
    sample = aracne_out.split("/")[-1]
    assert aracne_out[-1] != "/", "aracne_out should not end with a /"
    
    network = pd.read_csv(aracne_out +"/consolidated-net_defaultid.tsv", sep = "\t")
    network_genes = list(set(network["regulator.values"].to_list() + network["target.values"].to_list()))
    
    bins = pd.read_csv(str(Path(aracne_out).parents[0]) + "/binned_expression.csv") + 2 # keep only genes in the network, and offset the bins by 2 to account for the special tokens, so 2 now corresponds to bin 0 (ZERO_IDX)
    common_genes = list(set(network_genes).intersection(set(bins.columns)))
    bins = bins.loc[:, common_genes] # Keep only genes that are also in network (no change expected - they should be the same)

    for i in range(bins.shape[0]):
        if ncells % 1000 == 0:
            print(f"Processed {ncells} cells", end="\r")
        cell_number = bins.index[i]
        
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
        
        cell = bins.iloc[i, :]
        # cell = cell[cell != ZERO_IDX] + NUM_GENES ## offset the bins by global number of genes -  this lets the same 
        # VS:
        # keep graph static across batches 
        cell = cell + NUM_GENES
        # nn.Embedding be used for both gene and bin embeddings
        if cell.shape[0] < MIN_GENES_PER_GRAPH: # require a minimum number of genes per cell 
            skipped += 1
            ncells+=1
            continue

        # Subset network to only include genes in the cell
        network_cell = network[
            network["regulator.values"].isin(cell.index) & network["target.values"].isin(cell.index)
        ]

        #local_gene_to_node_index = {gene:i for i, gene in enumerate(cell.index)}
        local_gene_to_node_index = global_gene_to_node
        # each cell graph is disjoint from each other in terms of the relative position of nodes and edges
        # so edge index is local to each graph for each cell. 
        # cell.index defines the order of the nodes in the graph
        with warnings.catch_warnings(): # Suppress annoying pandas warnings
            warnings.simplefilter("ignore") 
            edges = network_cell[['regulator.values', 'target.values', 'mi.values']]
            edges['regulator.values'] = edges['regulator.values'].map(local_gene_to_node_index)
            edges['target.values'] = edges['target.values'].map(local_gene_to_node_index)

        edge_list = torch.tensor(np.array(edges[['regulator.values', 'target.values']])).T
        edge_weights = torch.tensor(np.array(edges['mi.values']))
        node_indices = torch.tensor(np.array([(global_gene_to_node[gene], cell[gene]) for gene in cell.index]), dtype=torch.long)
        
        data = Data(
            x=node_indices, 
            edge_index=edge_list, 
            edge_weight=edge_weights
        )
        
        torch.save(data, outfile)
        ncells += 1
            
        try:
            torch.load(outfile)
            print(outfile)
        except:
            print(outfile, "-------- Failed")
        
    return (skipped, ncells)

def run_save_perturbed(i, global_gene_to_node, cache_dir, overwrite, valsg_split_ratio, skipped, ncells, perturbed=False, gene_id="gene_id"):
    aracne_out = i[0]
    cell_type = aracne_out.split("/")[-2]
    sample = aracne_out.split("/")[-1]
    assert aracne_out[-1] != "/", "aracne_out should not end with a /"
    
    network = pd.read_csv(aracne_out +"/consolidated-net_defaultid.tsv", sep = "\t")
    network_genes = list(set(network["regulator.values"].to_list() + network["target.values"].to_list()))
    
    # ----------------------------------------------------------------------------------------------------
    # Perturbation specific below 
    prt_dataset = ad.read(str(Path(aracne_out).parents[0]) + "/cells.h5ad", backed="r") # Get perturbation dataset - only raw/un-binned expression data is used for perturbation
    assert all(prt_dataset.var.index.isin(INCLUDED_ENSG)) # Make sure ALL the genes (columns) in the dataset are genes we are able to tokenize

    # Below, any cells where gene_id is 'nan' are discarded
    gene_ids = prt_dataset.obs[gene_id] # Get the perturbed gene ids
    control_mask = (gene_ids == "non-targeting") # Mask accounting for cells with non-targeting intervention
    perturbed_mask = gene_ids.isin(prt_dataset.var.index).to_numpy() # Mask accounting for cells with perturbed genes that are included in our dataset

    control_cells = prt_dataset[control_mask] # Get the control cells from the perturbation dataset
    perturbed_cells = prt_dataset[perturbed_mask] # Get the perturbed cells from the perturbation dataset

    print("Number of control cells:", control_cells.shape[0])
    print("Number of perturbed cells:", perturbed_cells.shape[0])
    
    # Create one-hot vectors for each perturbed cell -> '1' corresponds to perturbed gene
    perturbations = perturbed_cells.obs["gene_id"].reset_index(drop=True) # Series object of perturbation for each cell (after filtering cells)
    one_hot_perturbations = np.zeros((perturbed_cells.shape), dtype=int)  # Initialize one-hot matrix with all zero
    for i, gene in enumerate(perturbations): # Iterate through each cell in the perturbed_cell dataset
        one_hot_perturbations[i, perturbed_cells.var.index.get_loc(gene)] = 1 # Set "1" in appropriate positions
        
    # Convert to pandas DataFrames
    control_df = pd.DataFrame(control_cells.X, columns=control_cells.var.index).reset_index(drop=True)
    perturbed_df = pd.DataFrame(perturbed_cells.X, columns=perturbed_cells.var.index).reset_index(drop=True)
    
    # 1. Gene ordering must be the same across 'control_cells', 'perturbed_cells', and 'one_hot_perturbations'
    # 2. Cell ordering must ONLY be the same across 'perturbed_cells' and 'one_hot_perturbations'
    # Point '2' is verified in that 'one_hot_perturbations' explicitly follows the format of 'perturbed_cells' in the way it is generated (this also means gene order is respected)
    # So we only need to check for point '1' between 'control_cells' and 'perturbed_cells', below
    assert control_cells.var.index.equals(perturbed_cells.var.index) # Check that genes are in the same order for both control and perturbed datasets

    # split = "control"
    # split = "perturbed"
    for split in ["control", "perturbed"]:
        # Pick dataset
        if split == "control":
            dataset = control_df
        else:
            dataset = perturbed_df
            
        # Run caching for this dataset
        for i in range(dataset.shape[0]):
            if ncells % 1000 == 0:
                print(f"Processed {ncells} cells", end="\r")
            
            outfile = f"{cache_dir}/{split}/{cell_type}_{i}.pt"
            
            if (os.path.exists(outfile)) and (not overwrite):
                ncells+=1
                continue
            
            cell = dataset.iloc[i, :] # Get the i-th cell in the dataset
            cell = cell + NUM_GENES # Shift 
            # nn.Embedding be used for both gene and bin embeddings
            if cell.shape[0] < MIN_GENES_PER_GRAPH: # require a minimum number of genes per cell 
                skipped += 1
                ncells+=1
                continue

            # Subset network to only include genes in the cell
            network_cell = network[
                network["regulator.values"].isin(cell.index) & network["target.values"].isin(cell.index)
            ]

            local_gene_to_node_index = {gene:i for i, gene in enumerate(cell.index)}
            # local_gene_to_node_index = global_gene_to_node
            # each cell graph is disjoint from each other in terms of the relative position of nodes and edges
            # so edge index is local to each graph for each cell. 
            # cell.index defines the order of the nodes in the graph
            with warnings.catch_warnings(): # Suppress annoying pandas warnings
                warnings.simplefilter("ignore") 
                edges = network_cell[['regulator.values', 'target.values', 'mi.values']]
                edges['regulator.values'] = edges['regulator.values'].map(local_gene_to_node_index)
                edges['target.values'] = edges['target.values'].map(local_gene_to_node_index)

            edge_list = torch.tensor(np.array(edges[['regulator.values', 'target.values']])).T
            edge_weights = torch.tensor(np.array(edges['mi.values']))
            node_indices = torch.tensor(np.array([(global_gene_to_node[gene], cell[gene]) for gene in cell.index]), dtype=torch.long)
            
            if perturbed:
                cell_perturbation = torch.tensor(one_hot_perturbations[i], dtype=torch.int) # Get the perturbation corresponding to this cell
                data = Data(
                    x=node_indices, 
                    edge_index=edge_list, 
                    edge_weight=edge_weights,
                    cell_perturbation=cell_perturbation
                )
            else:
                data = Data(
                    x=node_indices, 
                    edge_index=edge_list, 
                    edge_weight=edge_weights
                )
            
            torch.save(data, outfile)
            ncells += 1
                
            try:
                torch.load(outfile)
                print(outfile)
            except:
                print(outfile, "-------- Failed")
        
    return (skipped, ncells)


def transform_and_cache_aracane_graph_ranks(aracne_outdir_info : List[List[str]], gene_to_node_file:str, cache_dir:str, overwrite:bool=False, single=False, valsg_split_ratio = 0.2, perturbed=False, gene_id="gene_id"): # 0.2 makes it closer to equal size btween SG and HOG
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        
        ## this file maps gene names to a index: "global" bc we want the same index for each gene across all experiments 
        global_gene_to_node = pd.read_csv(gene_to_node_file)
        global_gene_to_node = {row.gene_name:row.idx for _,row in global_gene_to_node.iterrows()}
        skipped = 0
        ncells = 0
        
        if single: # Run on a single cell-type
            print("Caching", aracne_outdir_info[0][0].split("/")[-2])
            if perturbed:
                print("Caching perturbation data...")
                skipped, ncells = run_save_perturbed(aracne_outdir_info[0], global_gene_to_node, cache_dir, overwrite, valsg_split_ratio, skipped, ncells, perturbed=perturbed, gene_id=gene_id)
            else:
                print("Caching data...")
                skipped, ncells = run_save(aracne_outdir_info[0], global_gene_to_node, cache_dir, overwrite, valsg_split_ratio, skipped, ncells)

        print(f"\n**DONE**\nSkipped {skipped} cells")
        print(f"loaded {ncells} cells")


class AracneGraphWithRanksDataset(Dataset):
    def __init__(self, cache_dir:str, dataset_name:str, debug:bool=False):
        """
        Args:
            aracne_outdirs (List[str]): list of aracne outdirs. Must be a fullpath 
            global_gene_to_node_file (str): path to file that maps gene name to integer index 
            cache_dir (str): path to directory where the processed data will be stored
        """   
        print(cache_dir)     
        self.debug = debug
        self.cached_files = [cache_dir+"/" + f for f in os.listdir(cache_dir) if f.endswith(".pt")]
        self.dataset_name = dataset_name
        super().__init__(None, None, None)
    
    def len(self):
        if self.debug:
            return 1000
        print(len(self.cached_files))
        return len(self.cached_files)
    
    def get(self, idx, mask_fraction = 0.05):
        ## mask 5% as a gene only mask; mask 5% as a rank only mask ; mask 5% as both gene and rank mask
        data = torch.load(self.cached_files[idx], weights_only=False)
        #data = Data(**data_dict)

        node_indices = data.x
        # need the clones otherwise the original indices will be modified
        orig_gene_indices = node_indices[:, 0].clone()
        orig_rank_indices = node_indices[:, 1].clone()
        ## for each mask type, create boolean mask of the same shape as node_indices
        gene_mask = torch.rand(node_indices.shape[0]) < mask_fraction
        rank_mask = torch.rand(node_indices.shape[0]) < mask_fraction
        both_mask = torch.rand(node_indices.shape[0]) < mask_fraction
        node_indices[gene_mask, 0] = MASK_IDX
        node_indices[rank_mask, 1] = MASK_IDX
        node_indices[both_mask, :] = torch.tensor([MASK_IDX, MASK_IDX], dtype=node_indices.dtype)
        
        return Data(
            x=node_indices, 
            edge_index=data.edge_index, 
            edge_weight=data.edge_weight, 
            gene_mask=gene_mask, 
            rank_mask=rank_mask, 
            both_mask=both_mask, 
            orig_gene_id=orig_gene_indices, 
            orig_rank_indices=orig_rank_indices, 
            dataset_name=self.dataset_name,
        )

class LitDataModule(pl.LightningDataModule):
    def __init__(self, data_config):
        super().__init__()
        self.data_config = data_config
        self.train_ds = AracneGraphWithRanksDataset(**data_config.train)
        self.val_ds = [AracneGraphWithRanksDataset(**val) for val in data_config.val]
        if data_config.run_test:
            self.test_ds = [AracneGraphWithRanksDataset(**test) for test in data_config.test]
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size = self.data_config.batch_size, num_workers = self.data_config.num_workers)
    def val_dataloader(self):
        return [DataLoader(val_ds, batch_size = self.data_config.batch_size, num_workers = self.data_config.num_workers) for val_ds in self.val_ds]
    def test_dataloader(self):
        return [DataLoader(test_ds, batch_size = self.data_config.batch_size, num_workers = self.data_config.num_workers) for test_ds in self.test_ds]

class GraphTransformerDataset(torchDataset):
    def __init__(self, cache_dir:str, dataset_name:str, debug:bool=False):
        """
        Args:
            aracne_outdirs (List[str]): list of aracne outdirs. Must be a fullpath 
            global_gene_to_node_file (str): path to file that maps gene name to integer index 
            cache_dir (str): path to directory where the processed data will be stored
        """   
        print(cache_dir)     
        self.debug = debug
        self.cached_files = [cache_dir+"/" + f for f in os.listdir(cache_dir) if f.endswith(".pt")]
        self.dataset_name = dataset_name

    def __len__(self):
        if self.debug:
            return 1000
        print(len(self.cached_files))
        return len(self.cached_files)

    def __getitem__(self, idx, mask_fraction = 0.1):
        ## mask 5% as a gene only mask; mask 5% as a rank only mask ; mask 5% as both gene and rank mask
        data = torch.load(self.cached_files[idx], weights_only=False)
        node_indices = data.x
        ## for each mask type, create boolean mask of the same shape as node_indices
        gene_mask = torch.rand(node_indices.shape[0]) < mask_fraction
        rank_mask = torch.rand(node_indices.shape[0]) < mask_fraction
        both_mask = torch.rand(node_indices.shape[0]) < mask_fraction
        
        
        # mask the tensors
        node_indices[gene_mask, 0] = MASK_IDX
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
        

################################################################################################
##################################### PERTURBATION DATASET #####################################
################################################################################################    

# PerturbationDataset provides a control cell and perturbed cell pair from the same cell-line
# The control cell is sampled randomly from the control population, whereas the perturbed cell is fetched as per usual using indexing
# The perturbed cell information consists of both the expected cell information (node indices, rank indices, etc.) as well as a one-hot
# vector representing the perturbed gene. The position of the one-hot corresponds to the position of that gene in the expression matrix columns
class PerturbationDataset(torchDataset):
    def __init__(self, cache_dir:str, dataset_name:str, debug:bool=False):
        """
        Args:
            aracne_outdirs (List[str]): list of aracne outdirs. Must be a fullpath 
            global_gene_to_node_file (str): path to file that maps gene name to integer index 
            cache_dir (str): path to directory where the processed data will be stored
        """ 
        print(cache_dir)     
        self.debug = debug
        self.cached_files_perturbed = [cache_dir+"/" + f for f in os.listdir(f"{cache_dir}/perturbed") if f.endswith(".pt")]
        self.cached_files_control = [cache_dir+"/" + f for f in os.listdir(f"{cache_dir}/control") if f.endswith(".pt")]
        self.num_control = len(self.cached_files_control) # Record number of control cells for random sampling in __getitem__()
        self.dataset_name = dataset_name

    def __len__(self):
        if self.debug:
            return 1000
        print(len(self.cached_files))
        return len(self.cached_files)

    def __getitem__(self, idx, mask_fraction = 0.1):        
        ######## CONTROL CELL ########
        # Get cell from the control population
        control_idx = random.randrange(0, self.num_control) # Get a random index
        control_cell = torch.load(self.cached_files_control[idx], weights_only=False) # Select random control cell
        control_node_indices = control_cell.x
        
        control_gene_indices = control_node_indices[:, 0].clone()
        control_rank_indices = control_node_indices[:, 1].clone()
        control_num_nodes = control_node_indices.shape[0]
        
        control_dict = {
                        "orig_gene_id" : control_gene_indices, 
                        "orig_rank_indices" : control_rank_indices, 
                        "edge_index": control_cell.edge_index,
                        "num_nodes": control_num_nodes,
                        }
        
        
        ######## PERTURBED CELL ########
        # Get perturbed cell & one-hot perturbation vector
        perturbed_cell = torch.load(self.cached_files_perturbed[idx], weights_only=False) # Select perturbed cell
        perturbed_node_indices = perturbed_cell.x
        perturbed_one_hot = perturbed_cell.cell_perturbation # One-hot vector representing which gene was perturbed
        
        perturbed_gene_indices = perturbed_node_indices[:, 0].clone()
        perturbed_rank_indices = perturbed_node_indices[:, 1].clone()
        perturbed_num_nodes = perturbed_node_indices.shape[0]
        
        perturbed_dict = {
                            "perturbation": perturbed_cell.cell_perturbation, # Perturbation one-hot vector
                            "orig_gene_id" : perturbed_gene_indices, 
                            "orig_rank_indices" : perturbed_rank_indices, 
                            "edge_index": perturbed_cell.edge_index,
                            "num_nodes": num_nodes,
                         }


        # graph positional encoding
        # spectral_pe = spectral_PE(edge_index=data.edge_index, num_nodes=node_indices.shape[0], k=64)
        
        return {
                "control" : control_dict,
                "perturbed": perturbed_dict,
                "dataset_name" : self.dataset_name
                }


class PerturbationDataModule(pl.LightningDataModule):
    def __init__(self, data_config, collate_fn=None):
        super().__init__()
        self.data_config = data_config
        self.train_ds = PerturbationDataset(**data_config.train)
        self.val_ds = [PerturbationDataset(**val) for val in data_config.val]
        
        if collate_fn: # If a collate function is specified
            self.collate_fn = collate_fn
        else: # Otherwise use default
            self.collate_fn = self.perturbation_collate_fn
        
        if data_config.run_test:
            self.test_ds = [PerturbationDataset(**test) for test in data_config.test]
    
    def perturbation_collate_fn(self, batch):
        control = { 
            "orig_gene_id" : [],
            "orig_rank_indices" : [],
            "edge_index": [],
            "num_nodes" :[],
        }

        perturbed = { 
            "perturbation": [],
            "orig_gene_id" : [],
            "orig_rank_indices" : [],
            "edge_index": [],
            "num_nodes" :[],
        }
        
        # Create lists of values (for each sample in the batch) for each key (orig_gene_id, orig_rank_indices, etc.)
        for sample in batch:
            for key in control.keys():
                control[key].append(sample["control"][key])
                perturbed[key].append(sample["perturbed"][key])
            perturbed["perturbation"].append(sample["perturbed"]["perturbation"]) # manually add perturbation information as this is not a key in control.keys()
            
        # Pad these lists
        for key in control.keys():
            if (key != "edge_index") and (key != "num_nodes"):
                control[key] = pad_sequence(control[key], batch_first=True)
                perturbed[key] = pad_sequence(perturbed[key], batch_first=True)
            # perturbed["perturbation"] = pad_sequence(perturbed["perturbation"], batch_first=True) # Not needed as these should all already be the same size 
        
        one_hot_dim = perturbed["perturbation"][0].shape
        assert all([prt.shape == one_hot_dim for prt in perturbed["perturbation"]]) # Check all perturbations one-hot vectors are the same shape

        data = {
            "control": control,
            "perturbed": perturbed,
            "dataset_name" : batch[0]["dataset_name"]
        }
        return data    
    
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
    # This portion lets you generate the cache for the data outside of the model training loop - took about ~1 hour on 5 cores for the pilot data set
    # python scGraphLLM/data.py --aracane-outdir-md  /hpc/projects/group.califano/GLM/data/aracne_1024_outdir.csv --gene-to-node-file /hpc/projects/group.califano/GLM/data/cellxgene_gene2index.csv --cache-dir /hpc/projects/group.califano/GLM/data/pilotdata_1024 --num-proc 16
    parser = argparse.ArgumentParser()
    parser.add_argument("--aracane-outdir-md", type=str, help="File containing a list of aracne outdirs; `ls path/to/aracaneOutdirs/* > <input>` ")
    parser.add_argument("--gene-to-node-file", type=str, help="File containing gene to node index mapping")
    parser.add_argument("--cache-dir", type=str, help="Directory to store the processed data")
    parser.add_argument("--num-proc", type=int, help="Number of processes to use", default=1)
    parser.add_argument("--perturbed", type=str, help="Is this a perturbation dataset? Perturbation information will be stored in caching", default=False)
    parser.add_argument("--gene_id", type=str, help="For perturbation ONLY: column in dataset.obs corresponding to the perturbed gene_id (ENSEMBL symbol notation)", default=None)
    parser.add_argument("--single-index", type=int, help="Index in --aracane-outdir-md to path to specific cell-type aracne for single cell-type caching (mainly used with parallelization)")
    parser.add_argument("--debug", action="store_true", default=False, help="Run in debug mode")
    args = parser.parse_args()
    single_index = args.single_index-1 # SLURM starts counting at 1 not 0
    debug = args.debug
    num_proc = int(args.num_proc)
    
    if args.perturbed == "true":
        args.perturbed = True
    else:
        args.perturbed = False
    
    # Read the data directories and their associated dataset label (train, valSG, valHG)
    aracane_metadata = pd.read_csv(args.aracane_outdir_md, names = ["aracne_out", "split"]).values
    
    if args.perturbed:
        # Create perturbation directories
        Path(f"{args.cache_dir}/control").mkdir(parents=True, exist_ok=True)
        Path(f"{args.cache_dir}/perturbed").mkdir(parents=True, exist_ok=True)
    else:
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
    elif (single_index) or (single_index == 0): # An index value has been given - corresponding to a specific cell-type to be process individually
        print("SINGLE CACHE")
        transform_and_cache_aracane_graph_ranks([aracane_metadata[single_index]], args.gene_to_node_file, args.cache_dir, single=True, perturbed=args.perturbed, gene_id=args.gene_id)
    else:
        print("RUNNING MULTI-THREADED")
        with Pool(num_proc) as p:
            p.starmap(transform_and_cache_aracane_graph_ranks, args_list)
