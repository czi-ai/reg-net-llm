#%%
import torch
from torch_geometric.data import Data, Dataset
from torch.utils.data import Dataset as torchDataset
from torch.utils.data import DataLoader as torchDataLoader
import numpy as np
import pandas as pd 
from _globals import * ## imported global variables are all caps 
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
import warnings

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

def transform_and_cache_aracane_graph_ranks(aracne_outdir_info : List[List[str]], gene_to_node_file:str, cache_dir:str, overwrite:bool=False, valsg_split_ratio = 0.3):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        
        global_gene_to_node = pd.read_csv(gene_to_node_file)
        global_gene_to_node = {row.gene_name:row.idx for _,row in global_gene_to_node.iterrows()}
        skipped = 0
        ncells = 0
        
        for i in aracne_outdir_info:
            aracne_out = i[0]
            msplit = i[1]
            sample = aracne_out.split("/")[-1]
            assert aracne_out[-1] != "/", "aracne_out should not end with a /"
            network = pd.read_csv(aracne_out +"/consolidated-net_defaultid.tsv", sep = "\t")
            network_genes = list(set(network["regulator.values"].to_list() + network["target.values"].to_list()))
            #ranks = pd.read_csv(aracne_out + "/rank_raw.csv") + 2 # keep only genes in the network, and offset the ranks by 2 to account for the special tokens, so 2 now corresponds to rank 0(ZERO_IDX)
            ranks = pd.read_csv(str(Path(aracne_out).parents[0]) + "/rank_raw.csv") + 2 
            common_genes = list(set(network_genes).intersection(set(ranks.columns)))

            ranks = ranks.loc[:,common_genes]
            for i in range(ranks.shape[0]):
                if ncells % 1000 ==0:
                    print(f"Processed {ncells} cells", end="\r")
                cell_name = ranks.index[i]
                if msplit == "valSG":
                    rand = rng.random()
                    if rand > valsg_split_ratio:
                        split = "train"
                    else:
                        split = msplit
                else:
                    split = msplit
                outfile = f"{cache_dir}/{split}/{sample}_{cell_name}.pt"
                if os.path.exists(outfile)&(not overwrite):
                    ncells+=1
                    continue
                cell = ranks.iloc[i, :]
                #cell = cell[cell != ZERO_IDX] + NUM_GENES ## offset the ranks by global number of genes -  this lets the same 
                #VS:
                # keep graph static across batches 
                cell = cell + NUM_GENES
                ## nn.Embedding be used for both gene and rank embeddings
                if cell.shape[0] < MIN_GENES_PER_GRAPH: ## reauire a minimum number of genes per cell 
                    skipped += 1
                    ncells+=1
                    continue

                ## subset network to only include genes in the cell
                network_cell = network[
                    network["regulator.values"].isin(cell.index) & network["target.values"].isin(cell.index)
                ]

                local_gene_to_node_index = {gene:i for i, gene in enumerate(cell.index)}
                ## each cell graph is disjoint from each other interms of the relative position of nodes and edges
                ## so edge index is local to each graph for each cell. 
                ## cell.index defines the order of the nodes in the graph
                with warnings.catch_warnings() : ## suppress annoying pandas warnings
                    warnings.simplefilter("ignore") 
                    edges = network_cell[['regulator.values', 'target.values', 'mi.values']]
                    edges['regulator.values'] = edges['regulator.values'].map(local_gene_to_node_index)
                    edges['target.values'] = edges['target.values'].map(local_gene_to_node_index)

                edge_list = torch.tensor(np.array(edges[['regulator.values', 'target.values']])).T
                edge_weights = torch.tensor(np.array(edges['mi.values']))
                node_indices = torch.tensor(np.array([(global_gene_to_node[gene], cell[gene])for gene in cell.index]), dtype=torch.long)
                
                # od = {"x":node_indices, "edge_index":edge_list, "edge_weight":edge_weights}
                # save(od, outfile)
                
                if outfile == "aracne_1024_13827.pt":
                    print(aracne_outdir_info, i)
                
                data = Data(x = node_indices, edge_index = edge_list, edge_weight = edge_weights)
                torch.save(data, outfile)
                ncells += 1
            exit()

        print(f"\n**DONE**\nSkipped {skipped} cells")
        print(f"loaded {ncells} cells")
        return 


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
        data_dict = load(self.cached_files[idx])
        data = Data(**data_dict)

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
        return Data(x = node_indices, edge_index = data.edge_index, edge_weight = data.edge_weight, gene_mask = gene_mask, rank_mask = rank_mask, both_mask = both_mask, orig_gene_id = orig_gene_indices, orig_rank_indices = orig_rank_indices, dataset_name = self.dataset_name)

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

class TransformerDataset(torchDataset):
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

    def __getitem__(self, idx, mask_fraction = 0.05):
        ## mask 5% as a gene only mask; mask 5% as a rank only mask ; mask 5% as both gene and rank mask
        data = torch.load(self.cached_files[idx])
        node_indices = data.x
        orig_gene_indices = node_indices[:, 0].clone()
        orig_rank_indices = node_indices[:, 1].clone()
        ## for each mask type, create boolean mask of the same shape as node_indices
        gene_mask = torch.rand(node_indices.shape[0]) < mask_fraction
        rank_mask = torch.rand(node_indices.shape[0]) < mask_fraction
        both_mask = torch.rand(node_indices.shape[0]) < mask_fraction

        return {
                "orig_gene_id" : orig_gene_indices, 
                "orig_rank_indices" : orig_rank_indices, 
                "gene_mask" : gene_mask, 
                "rank_mask" : rank_mask, 
                "both_mask" : both_mask, 
                "dataset_name" : self.dataset_name
                }

class TransformerDataModule(pl.LightningDataModule):
    def __init__(self, data_config, collate_fn=None):
        super().__init__()
        self.data_config = data_config
        self.collate_fn = collate_fn
        self.train_ds = TransformerDataset(**data_config.train)
        self.val_ds = [TransformerDataset(**val) for val in data_config.val]
        if data_config.run_test:
            self.test_ds = [TransformerDataset(**test) for test in data_config.test]
    
    def train_dataloader(self):
        return torchDataLoader(self.train_ds, batch_size = self.data_config.batch_size, num_workers = self.data_config.num_workers, collate_fn=self.collate_fn)
    def val_dataloader(self):
        return [torchDataLoader(val_ds, batch_size = self.data_config.batch_size, num_workers = self.data_config.num_workers, collate_fn=self.collate_fn) for val_ds in self.val_ds]
    def test_dataloader(self):
        return [torchDataLoader(test_ds, batch_size = self.data_config.batch_size, num_workers = self.data_config.num_workers, collate_fn=self.collate_fn) for test_ds in self.test_ds]


if __name__ == "__main__":
    ## This portion lets you generate the cache for the data outside of the model training loop - took about ~1 hour on 5 cores for the pilot data set
    ## python scGraphLLM/data.py --aracane-outdir-md  /hpc/projects/group.califano/GLM/data/aracne_1024_outdir.csv --gene-to-node-file /hpc/projects/group.califano/GLM/data/cellxgene_gene2index.csv --cache-dir /hpc/projects/group.califano/GLM/data/pilotdata_1024 --num-proc 16
    parser = argparse.ArgumentParser()
    parser.add_argument("--aracane-outdir-md", type=str, help="File containing a list of aracne outdirs; `ls path/to/aracaneOutdirs/* > <input>` ")
    parser.add_argument("--gene-to-node-file", type=str, help="File containing gene to node index mapping")
    parser.add_argument("--cache-dir", type=str, help="Directory to store the processed data")
    parser.add_argument("--num-proc", type=int, help="Number of processes to use", default=1)
    parser.add_argument("--debug", action="store_true", default=False, help="Run in debug mode")
    args = parser.parse_args()
    debug = args.debug
    num_proc = int(args.num_proc)
    aracane_metadata = pd.read_csv(args.aracane_outdir_md, names = ["aracne_out", "split"]).values

    unique_susbets = np.unique(aracane_metadata[:,1])
    for subset in unique_susbets:
        Path(f"{args.cache_dir}/{subset}").mkdir(parents=True, exist_ok=True)
    sub_lists = np.array_split(aracane_metadata, num_proc)
    args = [(sub_list, args.gene_to_node_file, args.cache_dir) for sub_list in sub_lists]
    
    if debug:
        print("DEBUG")
        for arg in args:
            transform_and_cache_aracane_graph_ranks(*arg)
    else:
        print("RUNNING MULTI-THREADED")
        with Pool(num_proc) as p:
            p.starmap(transform_and_cache_aracane_graph_ranks, args)

#%%
# ds = AracneGraphWithRanksDataset("/burg/pmg/collab/scGraphLLM/data/pilotdata_cache/")
# # %%
# print(ds[10])
# # %%
