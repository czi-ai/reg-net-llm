import torch
from torch_geometric.data import Data, Dataset
from torch.utils.data import Dataset as torchDataset
from torch.utils.data import DataLoader as torchDataLoader
import numpy as np
import pandas as pd 
import anndata as ad

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
from scGraphLLM._globals import * ## imported global variables are all caps 

REG_VALS = "regulator.values"
TAR_VALS = "target.values"
MI_VALS = "mi.values"

gene_to_node = pd.read_csv("/hpc/projects/group.califano/GLM/data/cellxgene_gene2index.csv", index_col=0)
INCLUDED_ENSG = gene_to_node.index[2:]

rng = default_rng(42)

def scglm_collate_fn(batch):
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
    
    # Make a dictionary of lists from the list of dictionaries
    for b in batch:
        for key in data.keys():
            data[key].append(b[key])

    # Pad these dictionaries of lists
    for key in data.keys():
        if (key == "dataset_name") or (key == "edge_index") or (key == "num_nodes"):
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
        network, 
        expression, 
        global_gene_to_node, 
        cache_dir, 
        overwrite, 
        msplit, 
        valsg_split_ratio, 
        cell_type, 
        min_genes_per_graph=MIN_GENES_PER_GRAPH, 
        max_seq_length=None, 
        only_expressed_genes=True,
        skipped=0, 
        ncells=0, 
        verbose=False
    ):
    """
    Assign local ARACNe graph to each cell and cache each cell
    """
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
        
        if msplit == "valSG":
            rand = rng.random()
            if rand > valsg_split_ratio:
                split = "train"
            else:
                split = msplit
        else:
            split = msplit
        
        # Get this cell/metacell's index number
        cell_number = expression.index[i]
        
        # Path to which the file will be cached
        outfile = f"{cache_dir}/{split}/{cell_type}_{cell_number}.pt"
        if (os.path.exists(outfile)) and (not overwrite):
            ncells+=1
            continue
        
        # Get this cell/metacell's expression
        cell: pd.Series = expression.iloc[i, :]
        
        # filter out genes with 0 expression
        if only_expressed_genes:
            cell = cell[cell != ZERO_IDX]

        if cell[cell != ZERO_IDX].shape[0] < min_genes_per_graph: # require a minimum number of expressed genes per cell 
            skipped += 1
            ncells+=1
            continue

        # enforce max sequence length
        if max_seq_length is not None and cell.shape[0] > max_seq_length:
            cell = cell.nlargest(n=max_seq_length)

        # Subset network to only include genes expressed in the cell
        if not only_expressed_genes:
            print("IMPLEMENT THE ONLY EXPRESSED GENES CAPABILITY HERE! NOT CURRENTLY USED!")
            print("I SUSPECT IT'S JUST: network_cell = network")
            exit()
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

        edge_list = torch.tensor(np.array(edges[[REG_VALS, TAR_VALS]]), dtype=torch.int16).T
        edge_weights = torch.tensor(np.array(edges[MI_VALS]), dtype=torch.float16)
        node_expression = torch.tensor(np.array([(global_gene_to_node[gene], cell[gene]) for gene in cell.index]), dtype=torch.int16) # should this be local_gene_to_node?
        data = Data(
            x=node_expression, 
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


def run_cache_perturbation(
        network, 
        expression, 
        global_gene_to_node, 
        cache_dir, 
        overwrite, 
        msplit, 
        valsg_split_ratio, 
        cell_type, 
        perturbation_var="gene_id",
        min_genes_per_graph=MIN_GENES_PER_GRAPH, 
        max_seq_length=None, 
        only_expressed_genes=True,
        skipped=0, 
        ncells=0,
        partition=0,
        verbose=False,
        test_mode=False
    ):
    """
    Assign local ARACNe graph to each cell and cache each cell
    """
    
    # remove edges due to unknown genes
    network = network[
        network[REG_VALS].isin(global_gene_to_node) & 
        network[TAR_VALS].isin(global_gene_to_node)
    ]

    # Below, any cells where perturbation_var is 'nan' - or the perturbed gene is not in the column genes - are discarded
    gene_ids = expression.obs[perturbation_var] # Get the perturbed gene ids
    control_mask = (gene_ids == "non-targeting") # Mask accounting for cells with non-targeting intervention
    perturbed_mask = gene_ids.isin(expression.var.index).to_numpy() # Mask accounting for cells with perturbed genes that are included in our dataset

    control_cells = expression[control_mask] # Get the control cells from the perturbation dataset
    perturbed_cells = expression[perturbed_mask] # Get the perturbed cells from the perturbation dataset

    print("Number of control cells:", control_cells.shape[0])
    print("Number of perturbed cells:", perturbed_cells.shape[0])
    
    # Convert to pandas DataFrames
    perturbation_ENSGs = perturbed_cells.obs[perturbation_var].reset_index(drop=True) # Series object of perturbation for each cell (after filtering cells)
    perturbation_ids = [perturbed_cells.var.index.get_loc(gene) for gene in perturbation_ENSGs]
    
    control_df = pd.DataFrame(control_cells.X.toarray(), columns=control_cells.var.index).reset_index(drop=True)
    perturbed_df = pd.DataFrame(perturbed_cells.X.toarray(), columns=perturbed_cells.var.index).reset_index(drop=True)

    # Gene ordering must be the same across 'control_cells' and 'perturbed_cells'
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
                
            # Randomly assign this cell to train, val, or test
            dataset_type_list = ['train', 'val', 'test']
            dataset_type_probabilities = [0.7, 0.2, 0.1] # train, val, test
            dataset_type = np.random.choice(dataset_type_list, p=dataset_type_probabilities)
            
            # Path to which the file will be cached
            outfile = f"{cache_dir}/{split}/{dataset_type}/{cell_type}_{partition}_{i}.pt"

            if test_mode: # Only for unit-testing
                outfile = f"{cache_dir}/{msplit}/{split}/{cell_type}_{partition}_{i}.pt"
            else:
                # Randomly assign this cell to train, val, or test
                dataset_type_list = ['train', 'val', 'test']
                dataset_type_probabilities = [0.7, 0.2, 0.1] # train, val, test
                dataset_type = np.random.choice(dataset_type_list, p=dataset_type_probabilities)
                
                # Path to which the file will be cached
                outfile = f"{cache_dir}/{split}/{dataset_type}/{cell_type}_{partition}_{i}.pt"
                
            if (os.path.exists(outfile)) and (not overwrite):
                ncells+=1
                continue
            
            # Get this cell's expression
            cell: pd.Series = dataset.iloc[i, :]
            
            if cell[cell != 0].shape[0] < min_genes_per_graph: # require a minimum number of expressed genes per cell 
                skipped += 1
                ncells+=1
                continue

            # Subset network to only include genes expressed in the cell
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

            edge_list = torch.tensor(np.array(edges[[REG_VALS, TAR_VALS]]), dtype=torch.int16).T
            edge_weights = torch.tensor(np.array(edges[MI_VALS]), dtype=torch.float16)
            node_expression = torch.tensor(np.array([(global_gene_to_node[gene], cell[gene]) for gene in cell.index]), dtype=torch.float16)
            
            if split == "control": # Control cell doesn't have the perturbation attribute
                data = Data(
                    x=node_expression, 
                    edge_index=edge_list, 
                    edge_weight=edge_weights
                )
            else: # Perturbed cell
                perturbation = perturbation_ids[i]
                data = Data(
                    x=node_expression, 
                    edge_index=edge_list, 
                    edge_weight=edge_weights,
                    perturbation=perturbation
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

def cache_aracane_and_bins(
        aracne_outdir_info : List[List[str]], 
        gene_to_node_file:str, 
        cache_dir:str, 
        perturbation_var="gene_id",
        overwrite:bool=False, 
        single=False, 
        valsg_split_ratio=0.2, # 0.2 makes it closer to equal size between SG and HOG 
        num_partitions=1, # Optional, if parallelizing the perturbed dataset, determines how many partitions/parallel processes
        partition=0 # Optional, if parallelizing the perturbed dataset, determines which partition to attend to
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
         
        for outdir_info in aracne_outdir_info: # If single == True, then this will only run once (for that single cell-type)
            cell_type = outdir_info[0].split('/')[-2] # Get the cell-type's name that is currently being cached
            print(f"Caching: {cell_type}...")
            
            aracne_out = outdir_info[0]
            msplit = outdir_info[1]
            network = pd.read_csv(aracne_out +"/consolidated-net_defaultid.tsv", sep = "\t") # Get the ARACNe network for this cell-type
            if msplit == "perturbed": # If this is a perturbation dataset
                expression = ad.read_h5ad(str(Path(aracne_out).parents[0]) + "/cells.h5ad", backed="r") # Get perturbation dataset - only raw/un-binned expression data is used for perturbation
                
                # Add parallelization functionality - specify which range of the expression samples are to be covered by this run
                partition_size = (expression.shape[0] // num_partitions) + 1 
                
                start = partition * partition_size # Get the starting index for this partition
                end = min((partition + 1) * partition_size, expression.shape[0]) # Get the end index for this partition

                # Create a new AnnData object for the slice
                expression_subset = expression[start:end, :].to_memory()
                
                # Ensure only the right genes and perturbations are included in the dataset
                gene_mask = expression_subset.var_names.isin(global_gene_to_node)
                pert_mask = expression_subset.obs[perturbation_var].isin(global_gene_to_node) | (expression_subset.obs[perturbation_var] == "non-targeting")
                
                expression_subset = expression_subset[pert_mask, gene_mask]
                assert all(expression_subset.var.index.isin(INCLUDED_ENSG)) # Make sure ALL the genes (columns) in the dataset are genes we are able to tokenize
                
                skipped, ncells = run_cache_perturbation(
                    network=network, 
                    expression=expression_subset, 
                    global_gene_to_node=global_gene_to_node, 
                    cache_dir=cache_dir, 
                    overwrite=overwrite, 
                    msplit=msplit, # Change this to seen_graph 
                    perturbation_var=perturbation_var, 
                    valsg_split_ratio=valsg_split_ratio, 
                    cell_type=cell_type, 
                    min_genes_per_graph=MIN_GENES_PER_GRAPH, 
                    max_seq_length=None, 
                    partition=partition,
                    skipped=skipped, 
                    ncells=ncells 
                )
                
            else: # Normal binned expression dataset
                expression = pd.read_csv(str(Path(aracne_out).parents[0]) + "/binned_expression.csv") # Get expression (most likely in binned format)
                skipped, ncells = run_cache(
                    network=network, 
                    expression=expression, 
                    global_gene_to_node=global_gene_to_node, 
                    cache_dir=cache_dir, 
                    overwrite=overwrite, 
                    msplit=msplit, # Change this to seen_graph
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
    def __init__(self, cache_dir:str, dataset_name:str, mask_fraction=0.15, debug:bool=False):
        self.debug = debug
        self.cached_files = [cache_dir+"/" + f for f in os.listdir(cache_dir) if f.endswith(".pt")]
        self.dataset_name = dataset_name
        self.mask_fraction = mask_fraction
        print(f"Cache Directory: {cache_dir}")
        print(f"Observation Count: {len(self):,}")

    def __len__(self):
        return len(self.cached_files)

    def __getitem__(self, idx):
        data = torch.load(self.cached_files[idx], weights_only=False)
        node_indices = data.x
        # For each mask type, create boolean mask of the same shape as node_indices
        if self.mask_fraction == 0:
            expression_mask = torch.zeros(node_indices.shape[0], dtype=torch.bool)
        else:
            expression_mask = torch.rand(node_indices.shape[0]) < self.mask_fraction
        
        # Add CLS
        cls = torch.tensor([[CLS_GENE_IDX, CLS_TOKEN]], dtype=node_indices.dtype)
        node_indices = torch.cat([cls, node_indices], dim=0) # CLS can never be masked

        # Add False to mask accounting for CLS
        expression_mask = torch.cat([torch.tensor([False]), expression_mask])

        orig_gene_indices = node_indices[:, 0].clone()
        orig_expression_indices = node_indices[:, 1].clone()
        num_nodes = node_indices.shape[0] - 1 # discount the cls node
        edge_index = data.edge_index # edges index assumes the first gene token's index is 0

        # graph positional encoding
        # spectral_pe = spectral_PE(edge_index=data.edge_index, num_nodes=node_indices.shape[0], k=64)
        
        return {
            "orig_gene_id" : orig_gene_indices, 
            "orig_expression_id" : orig_expression_indices, 
            "expression_mask" : expression_mask, 
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
                               num_workers = self.data_config.num_workers, collate_fn=self.collate_fn, shuffle=True)
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
# The perturbed cell information consists of both the expected cell information (node indices, rank indices, etc.) as well as a single index value
# representing the perturbed gene's index in the expression matrix columns
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
        perturbation = perturbed_cell.perturbation # One-hot vector representing which gene was perturbed
        
        perturbed_gene_indices = perturbed_node_indices[:, 0].clone()
        perturbed_rank_indices = perturbed_node_indices[:, 1].clone()
        perturbed_num_nodes = perturbed_node_indices.shape[0]
        
        perturbed_dict = {
                            "perturbation": perturbation, # Perturbation one-hot vector
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
        return torchDataLoader(self.train_ds, batch_size=self.data_config.batch_size, 
                               num_workers=self.data_config.num_workers, collate_fn=self.collate_fn)
    def val_dataloader(self):
        return [torchDataLoader(val_ds, batch_size = self.data_config.batch_size, 
                                num_workers=self.data_config.num_workers, collate_fn=self.collate_fn) for val_ds in self.val_ds]
    def test_dataloader(self):
        return [torchDataLoader(test_ds, batch_size=self.data_config.batch_size, 
                                num_workers=self.data_config.num_workers, collate_fn=self.collate_fn) for test_ds in self.test_ds]


if __name__ == "__main__":
    ## This portion lets you generate the cache for the data outside of the model training loop - took about ~1 hour on 5 cores for the pilot data set
    ## python scGraphLLM/data.py --aracane-outdir-md  /hpc/projects/group.califano/GLM/data/aracne_1024_outdir.csv --gene-to-node-file /hpc/projects/group.califano/GLM/data/cellxgene_gene2index.csv --cache-dir /hpc/projects/group.califano/GLM/data/pilotdata_1024 --num-proc 16
    parser = argparse.ArgumentParser()
    parser.add_argument("--aracane-outdir-md", type=str, help="File containing a list of aracne outdirs; `ls path/to/aracaneOutdirs/* > <input>` ")
    parser.add_argument("--gene-to-node-file", type=str, help="File containing gene to node index mapping")
    parser.add_argument("--cache-dir", type=str, help="Directory to store the processed data")
    parser.add_argument("--perturbation-var", type=str, default=None)
    parser.add_argument("--num-proc", type=int, help="Number of processes to use", default=1)
    parser.add_argument("--partition", type=int, help="Which process index, if parallelizing", default=0)
    parser.add_argument("--single-index", type=int, help="Index in --aracane-outdir-md to path to specific cell-type aracne for single cell-type caching (mainly used with parallelization)", default=0)
    parser.add_argument("--debug", action="store_true", default=False, help="Run in debug mode")
    args = parser.parse_args()
    single_index = args.single_index
    debug = args.debug
    num_proc = int(args.num_proc)
    
    # Read the data directories and their associated dataset label (train, valSG, valHG)
    aracane_metadata = pd.read_csv(args.aracane_outdir_md, names = ["aracne_out", "split"]).values
    
    if aracane_metadata[0][1] == "perturbed": # Check if this is a perturbed dataset
        # Create perturbation directories
        Path(f"{args.cache_dir}/control").mkdir(parents=True, exist_ok=True)
        Path(f"{args.cache_dir}/control/train").mkdir(parents=True, exist_ok=True)
        Path(f"{args.cache_dir}/control/val").mkdir(parents=True, exist_ok=True)
        Path(f"{args.cache_dir}/control/test").mkdir(parents=True, exist_ok=True)
        
        Path(f"{args.cache_dir}/perturbed").mkdir(parents=True, exist_ok=True)
        Path(f"{args.cache_dir}/perturbed/train").mkdir(parents=True, exist_ok=True)
        Path(f"{args.cache_dir}/perturbed/val").mkdir(parents=True, exist_ok=True)
        Path(f"{args.cache_dir}/perturbed/test").mkdir(parents=True, exist_ok=True)
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
            cache_aracane_and_bins(*arg)
    elif (single_index) or (single_index == 0): # An index value has been given - corresponding to a specific cell-type to be process individually
        print("SINGLE CACHE")
        cache_aracane_and_bins(
            aracane_metadata[single_index].reshape(1, 2), 
            args.gene_to_node_file, 
            args.cache_dir, 
            perturbation_var=args.perturbation_var, 
            single=True,
            num_partitions=num_proc, # Optional, if parallelizing the perturbed dataset, determines how many partitions/parallel processes
            partition=args.partition # Optional, if parallelizing the perturbed dataset, determines which partition to attend to
        )
    else:
        print("RUNNING MULTI-THREADED")
        with Pool(num_proc) as p:
            p.starmap(cache_aracane_and_bins, args_list)
