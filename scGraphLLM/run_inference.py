import os
from os.path import join
from argparse import ArgumentParser
import scanpy as sc
import anndata as ad
import pandas as pd

from scGraphLLM.config import graph_kernel_attn_3L_4096
from scGraphLLM.inference import get_cell_embeddings, get_gene_embeddings
from scGraphLLM.models import GDTransformer
from scGraphLLM import RegulatoryNetwork, GeneVocab, GraphTokenizer, InferenceDataset


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
    dataset = InferenceDataset(
        expression=adata.to_df(), 
        tokenizer=GraphTokenizer(vocab=vocab, network=network)
    )

    # get cell embeddings 
    x_cell = get_cell_embeddings(dataset, model, vocab, cls_policy="exclude")

    # get cell embeddings including cls token embedding
    x_cell_with_cls = get_cell_embeddings(dataset, model, vocab, cls_policy="include")

    # get cls token embedding as cell embedding
    x_cell_only_cls = get_cell_embeddings(dataset, model, vocab, cls_policy="only")

    # get gene embeddings
    x_gene = get_gene_embeddings(dataset, model, vocab)

    # save with original metadata
    save_with_metadata(x_gene, metadata=adata.var, path=join(args.out_dir, "emb_gene.h5ad"))
    save_with_metadata(x_cell, metadata=adata.obs, path=join(args.out_dir, "emb_cell.h5ad"))
    save_with_metadata(x_cell_with_cls, metadata=adata.obs, path=join(args.out_dir, "emb_cell_with_cls.h5ad"))
    save_with_metadata(x_cell_only_cls, metadata=adata.obs, path=join(args.out_dir, "emb_cell_only_cls.h5ad"))
    

def save_with_metadata(x: pd.DataFrame, metadata: pd.DataFrame, path):
    adata = ad.AnnData(x.values, obs=metadata.loc[x.index])
    adata.write_h5ad(path)
    return adata


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--vocab_path", type=str, required=True)
    parser.add_argument("--network_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    main(args)