import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.utils.data import Dataset, DataLoader
from anndata import AnnData
from torch.utils.data import DataLoader, SequentialSampler
import os
import sys
import importlib
import json
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Union
from os.path import join, dirname, abspath
import warnings
warnings.filterwarnings("ignore")

scglm_rootdir = dirname(dirname(abspath(importlib.util.find_spec("scGraphLLM").origin)))
gene_names_map = pd.read_csv(join(scglm_rootdir, "data/gene-name-map.csv"), index_col=0)
ensg2hugo = gene_names_map.set_index("ensg.values")["hugo.values"].to_dict()
hugo2ensg = gene_names_map.set_index("hugo.values")["ensg.values"].to_dict()
ensg2hugo_vectorized = np.vectorize(ensg2hugo.get)
hugo2ensg_vectorized = np.vectorize(hugo2ensg.get)

REG_VALS = "regulator.values"
TAR_VALS = "target.values"

# Parsing arguments outside main clause in order to import scgpt code
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--model_dir", type=str, required=True)
parser.add_argument("--aracne_dir", type=str, required=True)
parser.add_argument("--scgpt_rootdir", type=str, required=True)
parser.add_argument("--sample_n_cells", type=int, default=None)
args = parser.parse_args()

sys.path.append(args.scgpt_rootdir)
from scgpt.tasks.cell_emb import embed_data
from scgpt.data_collator import DataCollator
from scgpt.model import TransformerModel
from scgpt.tokenizer import GeneVocab
from scgpt.utils import load_pretrained

PathLike = Union[str, os.PathLike]


class scGPTDataset(torch.utils.data.Dataset):
    def __init__(self, count_matrix, gene_ids, cls_token, pad_value, batch_ids=None):
        self.count_matrix = count_matrix
        self.gene_ids = gene_ids
        self.batch_ids = batch_ids
        self.cls_token = cls_token
        self.pad_value = pad_value

    def __len__(self):
        return len(self.count_matrix)

    def __getitem__(self, idx):
        row = self.count_matrix[idx]
        nonzero_idx = np.nonzero(row)[0]
        values = row[nonzero_idx]
        genes = self.gene_ids[nonzero_idx]
        
        genes = np.insert(genes, 0, self.cls_token)
        values = np.insert(values, 0, self.pad_value)
        genes = torch.from_numpy(genes).long()
        values = torch.from_numpy(values).float()
        output = {
            "id": idx,
            "genes": genes,
            "expressions": values,
        }
        if self.batch_ids is not None:
            output["batch_labels"] = self.batch_ids[idx]
        return output

def get_batch_embeddings(
    adata,
    embedding_mode: str = "cls",
    model=None,
    vocab=None,
    max_length=1200,
    batch_size=64,
    model_configs=None,
    gene_ids=None,
    use_batch_labels=False,
) -> np.ndarray:
    """
    Get the cell embeddings for a batch of cells.

    Args:
        adata (AnnData): The AnnData object.
        cell_embedding_mode (str): The mode to get the cell embeddings. Defaults to "cls".
        model (TransformerModel, optional): The model. Defaults to None.
        vocab (GeneVocab, optional): The vocabulary. Defaults to None.
        max_length (int): The maximum length of the input sequence. Defaults to 1200.
        batch_size (int): The batch size for inference. Defaults to 64.
        model_configs (dict, optional): The model configurations. Defaults to None.
        gene_ids (np.ndarray, optional): The gene vocabulary ids. Defaults to None.
        use_batch_labels (bool): Whether to use batch labels. Defaults to False.

    Returns:
        np.ndarray: The cell embeddings.
    """

    count_matrix = adata.X
    count_matrix = (
        count_matrix if isinstance(count_matrix, np.ndarray) else count_matrix.A
    )

    # gene vocabulary ids
    if gene_ids is None:
        gene_ids = np.array(adata.var["id_in_vocab"])
        assert np.all(gene_ids >= 0)

    if use_batch_labels:
        batch_ids = np.array(adata.obs["batch_id"].tolist())

    dataset = scGPTDataset(
        count_matrix=count_matrix, 
        gene_ids=gene_ids,
        cls_token=vocab["<cls>"],
        pad_value=model_configs["pad_value"],
        batch_ids=batch_ids if use_batch_labels else None
    )
    collator = DataCollator(
        do_padding=True,
        pad_token_id=vocab[model_configs["pad_token"]],
        pad_value=model_configs["pad_value"],
        do_mlm=False,
        do_binning=True,
        max_length=max_length,
        sampling=True,
        keep_first_n_tokens=1,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset),
        collate_fn=collator,
        drop_last=False,
        num_workers=min(len(os.sched_getaffinity(0)), batch_size),
        pin_memory=True,
    )

    device = next(model.parameters()).device
    embeddings_list = []
    genes_list = []
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        count = 0
        for data_dict in tqdm(data_loader, desc="Embedding cells"):
            input_gene_ids = data_dict["gene"].to(device)
            src_key_padding_mask = input_gene_ids.eq(
                vocab[model_configs["pad_token"]]
            )
            embeddings = model._encode(
                input_gene_ids,
                data_dict["expr"].to(device),
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=data_dict["batch_labels"].to(device)
                if use_batch_labels
                else None,
            )
            embeddings_list += [embeddings.cpu().numpy()]
            genes_list += [input_gene_ids.cpu().numpy()]
            count += len(embeddings)
            
            # Free up memory
            del input_gene_ids, embeddings
            torch.cuda.empty_cache()

        # concatenate embeddings and genes lists
        max_seq_length = max(emb.shape[1] for emb in embeddings_list)
        embeddings = np.concatenate([
            np.pad(emb, pad_width=((0, 0), (0, max_seq_length - emb.shape[1]), (0, 0)), 
                   mode="constant", constant_values=0)
            for emb in embeddings_list
        ], axis=0)
        genes = np.concatenate([
            np.pad(gene_ids, pad_width=((0, 0), (0, max_seq_length - gene_ids.shape[1])), 
                   mode="constant", constant_values=vocab[model_configs["pad_token"]])
            for gene_ids in genes_list
        ], axis=0)
    

    if embedding_mode == "cls":
        cell_embeddings = embeddings[:, 0, :] # get the <cls> position embedding
        cell_embeddings = cell_embeddings / np.linalg.norm(cell_embeddings, axis=1, keepdims=True)
        return cell_embeddings
    elif embedding_mode == "raw":
        return embeddings, genes
    else:
        raise ValueError(f"Unknown cell embedding mode: {embedding_mode}")


def embed_data(
    adata_or_file: Union[AnnData, PathLike],
    model_dir: PathLike,
    gene_col: str = "feature_name",
    embedding_mode="cls",
    max_length=1200,
    batch_size=64,
    obs_to_save: Optional[list] = None,
    device: Union[str, torch.device] = "cuda",
    use_fast_transformer: bool = True,
    return_new_adata: bool = False,
) -> AnnData:
    """
    Preprocess anndata and embed the data using the model.

    Args:
        adata_or_file (Union[AnnData, PathLike]): The AnnData object or the path to the
            AnnData object.
        model_dir (PathLike): The path to the model directory.
        gene_col (str): The column in adata.var that contains the gene names.
        max_length (int): The maximum length of the input sequence. Defaults to 1200.
        batch_size (int): The batch size for inference. Defaults to 64.
        obs_to_save (Optional[list]): The list of obs columns to save in the output adata.
            Useful for retaining meta data to output. Defaults to None.
        device (Union[str, torch.device]): The device to use. Defaults to "cuda".
        use_fast_transformer (bool): Whether to use flash-attn. Defaults to True.
        return_new_adata (bool): Whether to return a new AnnData object. If False, will
            add the cell embeddings to a new :attr:`adata.obsm` with key "X_scGPT".

    Returns:
        AnnData: The AnnData object with the cell embeddings.
    """
    if isinstance(adata_or_file, AnnData):
        adata = adata_or_file
    else:
        adata = sc.read_h5ad(adata_or_file)

    if isinstance(obs_to_save, str):
        assert obs_to_save in adata.obs, f"obs_to_save {obs_to_save} not in adata.obs"
        obs_to_save = [obs_to_save]

    # verify gene col
    if gene_col == "index":
        adata.var["index"] = adata.var.index
    else:
        assert gene_col in adata.var

    if device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            print("WARNING: CUDA is not available. Using CPU instead.")

    # LOAD MODEL
    model_dir = Path(model_dir)
    vocab_file = model_dir / "vocab.json"
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]

    # vocabulary
    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)
    adata.var["id_in_vocab"] = [
        vocab[gene] if gene in vocab else -1 for gene in adata.var[gene_col]
    ]
    gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
    print(
        f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
        f"in vocabulary of size {len(vocab)}."
    )
    adata = adata[:, adata.var["id_in_vocab"] >= 0]

    with open(model_config_file, "r") as f:
        model_configs = json.load(f)

    # Binning will be applied after tokenization. A possible way to do is to use the unified way of binning in the data collator.

    vocab.set_default_index(vocab["<pad>"])
    genes = adata.var[gene_col].tolist()
    gene_ids = np.array(vocab(genes), dtype=int)

    # all_counts = adata.layers["counts"]
    # num_of_non_zero_genes = [
    #     np.count_nonzero(all_counts[i]) for i in range(all_counts.shape[0])
    # ]
    # max_length = min(max_length, np.max(num_of_non_zero_genes) + 1)

    model = TransformerModel(
        ntoken=len(vocab),
        d_model=model_configs["embsize"],
        nhead=model_configs["nheads"],
        d_hid=model_configs["d_hid"],
        nlayers=model_configs["nlayers"],
        nlayers_cls=model_configs["n_layers_cls"],
        n_cls=1,
        vocab=vocab,
        dropout=model_configs["dropout"],
        pad_token=model_configs["pad_token"],
        pad_value=model_configs["pad_value"],
        do_mvc=True,
        do_dab=False,
        use_batch_labels=False,
        domain_spec_batchnorm=False,
        explicit_zero_prob=False,
        use_fast_transformer=use_fast_transformer,
        fast_transformer_backend="flash",
        pre_norm=False,
    )
    load_pretrained(model, torch.load(model_file, map_location=device), verbose=False)
    model.to(device)
    model.eval()

    # get cell embeddings
    result = get_batch_embeddings(
        adata,
        embedding_mode=embedding_mode,
        model=model,
        vocab=vocab,
        max_length=max_length,
        batch_size=batch_size,
        model_configs=model_configs,
        gene_ids=gene_ids,
        use_batch_labels=False,
    )

    if embedding_mode == "raw":
        id_gene_map = {idx: token for idx, token in enumerate(vocab.get_itos())}
        embeddings, genes = result
        return embeddings, genes, id_gene_map

    if return_new_adata:
        obs_df = adata.obs[obs_to_save] if obs_to_save is not None else None
        return sc.AnnData(X=result, obs=obs_df, dtype="float32")

    adata.obsm["X_scGPT"] = result
    return adata


def main(args):
    data: sc.AnnData = sc.read_h5ad(args.cells_path)
    data.var["symbol_id"] = data.var_names.to_series().apply(ensg2hugo.get)
    data = data[:, ~data.var["symbol_id"].isna()]
    data.var.set_index("symbol_id")
    data.var_names = data.var["symbol_id"]

    if args.sample_n_cells is not None and data.n_obs > args.sample_n_cells:
        sc.pp.subsample(data, n_obs=args.sample_n_cells, random_state=12345, copy=False)

    embeddings, symbol_ids, id_symbol_map = embed_data(
        adata_or_file=data,
        model_dir=args.model_dir,
        gene_col="index",
        embedding_mode="raw",
        max_length=1200,
        batch_size=32,
        obs_to_save=None,
        device="cuda",
        use_fast_transformer=True,
        return_new_adata=True,
    ) # cells x max_seq_length x embedding_dim

    # remove <CLS> token and associated embeddings
    embeddings = embeddings[:,1:,:] #.astype(np.float16)
    symbol_ids = symbol_ids[:,1:]

    # translate gene_ids
    id_gene_map_vectorized = np.vectorize(lambda x: id_symbol_map.get(x))
    genes_symbol = id_gene_map_vectorized(symbol_ids)
    genes_ensg = hugo2ensg_vectorized(genes_symbol)
    
    assert np.sum(genes_symbol == "cls") == 0
    max_seq_length = embeddings.shape[1]
    seq_lengths = [np.where(seq == '<pad>')[0][0] if np.any(seq == '<pad>') else max_seq_length for seq in genes_symbol]

    # load aracne network
    network = pd.read_csv(join(args.aracne_dir, "consolidated-net_defaultid.tsv"), sep="\t")
    
    # get edges for each cell
    edges = {}
    for i, genes_i in enumerate(genes_ensg):
        local_gene_to_node_index = {gene: i for i,gene in enumerate(genes_i)}
        edges_i = network[
            network[REG_VALS].isin(genes_i) & 
            network[TAR_VALS].isin(genes_i)
        ].assign(**{
            REG_VALS: lambda df: df[REG_VALS].map(local_gene_to_node_index),
            TAR_VALS: lambda df: df[TAR_VALS].map(local_gene_to_node_index),
        })[[REG_VALS, TAR_VALS]].to_numpy().T
        edges[i] = edges_i

    np.savez(
        file=join(args.out_dir, "embedding.npz"), 
        x=embeddings,
        seq_lengths=seq_lengths,
        edges=edges, 
        allow_pickle=True
    )


if __name__ == "__main__":

    args.cells_path = join(args.data_dir, "cells.h5ad")
    args.out_dir = join(args.data_dir, "embeddings/scgpt")
    args.emb_path = join(args.out_dir, "embedding.h5ad")
    os.makedirs(args.out_dir, exist_ok=True)

    main(args)