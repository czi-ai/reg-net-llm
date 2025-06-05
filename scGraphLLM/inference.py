import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

from argparse import ArgumentParser
from tqdm import tqdm
import os
from os.path import join
from functools import partial

import torch
from torch.utils.data import DataLoader
from scGraphLLM.data import GraphTransformerDataset, torchGeomData, scglm_collate_fn, send_to_gpu
from scGraphLLM.benchmark import send_to_gpu
from scGraphLLM.preprocess import tokenize_expr
from scGraphLLM._globals import *
from scGraphLLM.models import GDTransformer
from scGraphLLM.config import graph_kernel_attn_3L_4096
from scGraphLLM.network import RegulatoryNetwork


class GeneVocab(object):
    def __init__(self, genes, nodes):
        self.genes = genes
        self.nodes = nodes
        self.gene_to_node = dict(zip(genes, nodes))
        self.node_to_gene = dict(zip(nodes, genes))
        if len(self.gene_to_node) != len(genes) or len(self.node_to_gene) != len(nodes):
            raise ValueError("Relationship between genes and nodes is not one-to-one.")
    
    @classmethod
    def from_csv(cls, path, gene_col="gene_name", node_col="idx", **kwargs):
        df = pd.read_csv(path, **kwargs)
        if gene_col not in df.columns or node_col not in df.columns:
            raise ValueError(f"Expected columns '{gene_col}' and '{node_col}' not found in CSV.")
        genes = df[gene_col].tolist()
        nodes = df[node_col].tolist()
        return cls(genes, nodes)


class GraphTokenizer:
    def __init__(
            self, 
            vocab: GeneVocab,
            network: RegulatoryNetwork=None, 
            max_seq_length=2048, 
            only_expressed_genes=True, 
            limit_regulon=None, 
            drop_unpaired=False,
            with_edge_weights=False, 
            n_bins=100, 
            method="quantile"
        ):
        self.vocab = vocab
        self.network = network
        self.max_seq_length = max_seq_length
        self.only_expressed_genes = only_expressed_genes
        self.limit_regulon = limit_regulon
        self.drop_unpaired = drop_unpaired
        self.with_edge_weights = with_edge_weights
        self.n_bins = n_bins
        self.method = method

    @property
    def gene_to_node(self):
        return self.vocab.gene_to_node

    @property
    def node_to_gene(self):
        return self.vocab.node_to_gene

    def __call__(self, cell_expr: pd.Series, override_network: RegulatoryNetwork = None):
        """
        Tokenize a single cell expression vector into a PyG Data object.
        """
        # override original network if network is provided
        network = override_network if override_network is not None else self.network

        cell = tokenize_expr(cell_expr, n_bins=self.n_bins, method=self.method)

        if self.only_expressed_genes:
            cell = cell[cell != ZERO_IDX]
        
        # enforce max sequence length
        if (self.max_seq_length is not None) and (cell.shape[0] > self.max_seq_length):
            cell = cell.nlargest(n=self.max_seq_length)

        # create local gene to node mapping
        local_gene_to_node = {gene:i for i, gene in enumerate(cell.index)}

        # Subset network to only include genes in the cell
        network_cell = network.df.copy()
        network_cell = network_cell[
            network_cell[network.reg_name].isin(cell.index) & 
            network_cell[network.tar_name].isin(cell.index)
        ]

        edge_index = torch.tensor(np.array([
            network_cell[network.reg_name].map(local_gene_to_node).values, 
            network_cell[network.tar_name].map(local_gene_to_node).values
        ]))

        node_expression = torch.tensor(np.array([
            (self.gene_to_node[gene], cell[gene]) for gene in cell.index
        ]), dtype=torch.long)
            
        if self.with_edge_weights:
            edge_weights = torch.tensor(np.array(network_cell[network.wt_name]))
            data = torchGeomData(
                x=node_expression, 
                edge_index=edge_index, 
                edge_weight=edge_weights
            )
        else:
            data = torchGeomData(
                x=node_expression, 
                edge_index=edge_index
            )

        return data
    

class GraphTransformerInferenceDataset(GraphTransformerDataset):
    def __init__(self, expression: pd.DataFrame, tokenizer: GraphTokenizer, cache_dir=None):
        self.tokenizer = tokenizer
        self.obs_names = expression.index
        self.common_genes = sorted(list(
            set(self.gene_to_node.keys()) & set(expression.columns) & set(self.network.genes)
        ))
        self.expression = expression[self.common_genes]
        super().__init__(cache_dir=cache_dir, mask_fraction=0.0)
    
    @property
    def gene_to_node(self):
        return self.tokenizer.gene_to_node

    @property
    def node_to_gene(self):
        return self.tokenizer.node_to_gene
    
    @property
    def network(self):
        return self.tokenizer.network
    
    def __len__(self):
        return len(self.expression)

    def __getitem__(self, idx):
        cell = self.expression.iloc[idx]
        data = self.tokenizer(cell)
        item = self._item_from_tokenized_data(data)
        item["obs_name"] = self.obs_names[idx]
        return item


def get_cell_embeddings(dataset, model):
    # Initialize Dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=64, 
        shuffle=False, 
        collate_fn=partial(scglm_collate_fn, inference=True)
    )

    x_list = []
    obs_names_list = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Forward Pass"):
            seq_lengths = torch.tensor(batch["num_nodes"]).to('cuda')
            obs_names = batch["obs_name"]
            x = model(send_to_gpu(batch))[0]
            
            # apply mean pooling
            B, N, D = x.shape
            mask = torch.arange(N, device=x.device).unsqueeze(0) < seq_lengths.unsqueeze(1)
            mask = mask.unsqueeze(-1).float()
            x_masked = x * mask
            x_pooled = x_masked.sum(dim=1) / mask.sum(dim=1)

            x_list.append(x_pooled.cpu().numpy())
            obs_names_list.append(obs_names)

    x = np.vstack(x_list)
    obs_names = np.concatenate(obs_names_list)
    embeddings = pd.DataFrame(x, index=obs_names)

    return embeddings


def main(args):
    # Load data
    adata = sc.read_h5ad(args.data_path)

    # Load vocab
    vocab = GeneVocab.from_csv(args.vocab_path, gene_col="gene_name", node_col="idx")

    # Load network
    network = RegulatoryNetwork.from_csv(args.network_path, sep="\t")

    # Load model
    model = GDTransformer.load_from_checkpoint(args.model_path, config=graph_kernel_attn_3L_4096)

    # Initialize dataset for inference
    dataset = GraphTransformerInferenceDataset(
        expression=adata.to_df(), 
        tokenizer=GraphTokenizer(vocab=vocab, network=network, n_bins=NUM_BINS)
    )

    # get embeddings
    x = get_cell_embeddings(dataset, model)

    # save with original metadata
    embeddings = ad.AnnData(x.values, obs=adata.obs.loc[x.index])
    embeddings.write_h5ad(args.emb_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--vocab_path", type=str, required=True)
    parser.add_argument("--network_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)

    args.emb_path = join(args.out_dir, "embedding.h5ad")

    main(args)