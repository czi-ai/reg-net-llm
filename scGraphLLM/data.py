#%%
import torch
from torch_geometric.data import Data, Dataset
import numpy as np
import pandas as pd 
from _globals import * ## imported global variables are all caps 
import os
from typing import List
import warnings
from pathlib import Path

class AracneGraphWithRanksDataset(Dataset):
    def __init__(self, aracne_outdirs : List[str], gene_to_node_file:str, cache_dir:str, debug:bool=False):
        """

        Args:
            aracne_outdirs (List[str]): list of aracne outdirs. Must be a fullpath 
            global_gene_to_node_file (str): path to file that maps gene name to integer index 
            cache_dir (str): path to directory where the processed data will be stored
        """        
        global_gene_to_node = pd.read_csv(gene_to_node_file)
        global_gene_to_node = {row.gene_name:row.idx for _,row in global_gene_to_node.iterrows()}
        self.aracne_outdirs = aracne_outdirs
        self.cache_dir = cache_dir
        self.debug = debug
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        ## just do the processing here on init 
        self.cached_files = []
        skipped = 0
        ncells = 0
        for aracne_out in aracne_outdirs:
            sample = aracne_out.split("/")[-1]
            assert aracne_out[-1] != "/", "aracne_out should not end with a /"
            network = pd.read_csv(aracne_out +"/aracne/consolidated-net_defaultid.tsv", sep = "\t")
            network_genes = list(set(network["regulator.values"].to_list() + network["target.values"].to_list()))
            ranks = pd.read_csv(aracne_out + "/rank_raw.csv", index_col=0)[network_genes] # keep only genes in the network

            for i in range(ranks.shape[0]):
                if ncells % 25 ==0:
                    print(f"Processed {ncells} cells", end="\r")
                cell_name = ranks.index[i]
                outfile = f"{self.cache_dir}/{sample}_{cell_name}.pt"
                if os.path.exists(outfile):
                    self.cached_files.append(outfile)
                    ncells+=1
                    continue
                cell = ranks.iloc[i, :]
                cell = cell[cell != ZERO_IDX] + NUM_GENES ## offset the ranks by global number of genes -  this lets the same 
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
                with warnings.catch_warnings(action="ignore") : ## suppress annoying pandas warnings
                    edges = network_cell[['regulator.values', 'target.values', 'mi.values']]
                    edges['regulator.values'] = edges['regulator.values'].map(local_gene_to_node_index)
                    edges['target.values'] = edges['target.values'].map(local_gene_to_node_index)

                edge_list = torch.tensor(np.array(edges[['regulator.values', 'target.values']])).T
                edge_weights = torch.tensor(np.array(edges['mi.values']))
                node_indices = torch.tensor(np.array([(global_gene_to_node[gene], cell[gene])for gene in cell.index]), dtype=torch.long)
                data = Data(x = node_indices, edge_index = edge_list, edge_weight = edge_weights)
                torch.save(data, outfile)
                self.cached_files.append(outfile)
                ncells += 1
        print(f"\n**DONE**\n\nSkipped {skipped} cells")
        print(f"loaded {len(self.cached_files)} cells")
        super().__init__(None, None, None)#
    def len(self):
        if self.debug:
            return 100
        return len(self.cached_files)
    def get(self, idx):
        ## generate random number between 1 and 100
        return torch.load(self.cached_files[idx])
#%%
# ds = AracneGraphWithRanksDataset([
#                                 '/burg/pmg/collab/scGraphLLM//data/samples/geneset_hvg/C164',
#                                 '/burg/pmg/collab/scGraphLLM//data/samples/geneset_hvg/T_cac7',
#                                 '/burg/pmg/collab/scGraphLLM//data/samples/geneset_hvg/SMC19',
#                                 '/burg/pmg/collab/scGraphLLM//data/samples/geneset_hvg/C124',
#                                 '/burg/pmg/collab/scGraphLLM//data/samples/geneset_hvg/T_cac15',
#                                 '/burg/pmg/collab/scGraphLLM//data/samples/geneset_hvg/KUL19',
#                                 '/burg/pmg/collab/scGraphLLM//data/samples/geneset_hvg/SMC17',
#                                 '/burg/pmg/collab/scGraphLLM//data/samples/geneset_hvg/SMC15',
#                                 '/burg/pmg/collab/scGraphLLM//data/samples/geneset_hvg/SMC07',
#                                 '/burg/pmg/collab/scGraphLLM//data/samples/geneset_hvg/C130',
#                                 '/burg/pmg/collab/scGraphLLM//data/samples/geneset_hvg/SMC22',
#                                 '/burg/pmg/collab/scGraphLLM//data/samples/geneset_hvg/SMC20',
#                                 '/burg/pmg/collab/scGraphLLM//data/samples/geneset_hvg/T_cac10',
#                                 '/burg/pmg/collab/scGraphLLM//data/samples/geneset_hvg/KUL01'
#                             ], "/burg/pmg/collab/scGraphLLM/data/example_gene2index.csv", "/burg/pmg/collab/scGraphLLM/data/modeldata/newgraphdata/")
# # %%
# ds[10]
# %%
