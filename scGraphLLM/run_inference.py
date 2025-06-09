import os
from os.path import join
from argparse import ArgumentParser
import scanpy as sc
import anndata as ad

from scGraphLLM.config import graph_kernel_attn_3L_4096
from scGraphLLM.inference import get_cell_embeddings
from scGraphLLM import GDTransformer, RegulatoryNetwork, GeneVocab, GraphTokenizer, InferenceDataset


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