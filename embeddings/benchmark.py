import os 
from os.path import join, abspath, dirname
import sys
from argparse import ArgumentParser

from torch_geometric.utils import negative_sampling
import lightning.pytorch as pl

from scGraphLLM.data import *
from scGraphLLM.GNN_modules import *
from scGraphLLM.MLP_modules import *
from scGraphLLM._globals import *
from scGraphLLM.flash_transformer import GDTransformer
from scGraphLLM.config import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import tqdm
import matplotlib.pyplot as plt
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def send_to_gpu(data):
    if isinstance(data, torch.Tensor):
        return data.to('cuda')  # Send tensor to GPU
    elif isinstance(data, list):
        return [send_to_gpu(item) for item in data]  # Recursively process lists
    elif isinstance(data, dict):
        return {key: send_to_gpu(value) for key, value in data.items()}  # Recursively process dicts
    else:
        return data  # If not a tensor or list/dict, leave unchanged

class GeneEmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, paths):
        self.paths = paths
        
    @property
    def paths(self):
        return self._paths

    @property
    def embedding_dim(self):
        return self.x.shape[2]

    @paths.setter
    def paths(self, paths):
        self._paths = paths     
        if isinstance(paths, list):
            for path in paths:
                assert os.path.exists(path), f"File not found: {path} in provided paths list."
            embedding = [np.load(path, allow_pickle=True) for path in self.paths]
            
            self.seq_lengths = np.concatenate([emb["seq_lengths"] for emb in embedding], axis=0)
            self.max_seq_length = np.max(self.seq_lengths)
            self.x = concatenate_embeddings([emb["x"] for emb in embedding])

            self.edges = {}
            i = 0
            for emb in embedding:
                edges = emb["edges"].item()
                for j in range(len(edges)):
                    self.edges[i] = edges[j]
                    i += 1
        else:
            assert os.path.exists(self.paths), f"File not found: {self.paths}"
            embedding = np.load(self.paths, allow_pickle=True)
            self.x = embedding["x"]
            self.seq_lengths = embedding["seq_lengths"]
            self.max_seq_length = np.max(self.seq_lengths)
            self.edges = embedding["edges"].item()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {
            "x": torch.tensor(self.x[idx]), 
            "seq_lengths": torch.tensor(self.seq_lengths[idx]),
            "edges": torch.tensor(self.edges[idx])
        }

def concatenate_embeddings(x_list):
    max_seq_length = max(x.shape[1] for x in x_list)
    x_list_padded = [
        np.pad(x, pad_width=((0, 0), (0, max_seq_length - x.shape[1]), (0, 0)), mode="constant", constant_values=0)
        for x in x_list
    ]
    return np.concatenate(x_list_padded, axis=0)


def embedding_collate_fn(batch):
    return {
        "x": torch.stack([item["x"] for item in batch]),
        "seq_lengths": torch.tensor([item["seq_lengths"] for item in batch]),
        "edges": [item["edges"] for item in batch]
    }

def link_pred_loss(predictor, node_embedding, edge_index_list, mask_locs=None, seq_lengths=None):
    pos_out = []
    neg_out = []
    pos_labels = []
    neg_labels = []

    batch_size, max_seq_length, embed_dim = node_embedding.shape
    device = node_embedding.device

    for i in range(batch_size):
        edge_index = edge_index_list[i].to(device)

        if mask_locs is not None:
            masked_nodes = torch.where(mask_locs[i])[0]
            if masked_nodes.numel() == 0:
                continue
            masked_nodes = masked_nodes.to(device)

            masked_nodes_bool = torch.zeros(max_seq_length, dtype=torch.bool, device=device)
            masked_nodes_bool[masked_nodes] = True # boolean-valued 1 dimensional array indicating True for masked, False for not
            src_nodes = edge_index[0]
            dst_nodes = edge_index[1]
            edge_mask = (~masked_nodes_bool[src_nodes]) & (~masked_nodes_bool[dst_nodes])
            pos_edge_index = edge_index[:, edge_mask]
        else:
            pos_edge_index = edge_index
            
        if pos_edge_index.size(1) == 0:
            continue

        num_nodes = max_seq_length if seq_lengths is None else seq_lengths[i].item()
        num_neg_samples = pos_edge_index.size(1)
        neg_edge_index = negative_sampling(
            edge_index=edge_index,
            num_nodes=num_nodes,
            num_neg_samples=num_neg_samples,
            method='sparse'
        ).to(device)

        # Positive scores
        src_emb_pos = node_embedding[i, pos_edge_index[0]]
        dst_emb_pos = node_embedding[i, pos_edge_index[1]]
        pos_scores = predictor(src_emb_pos, dst_emb_pos)
        pos_out.append(pos_scores)
        pos_labels.append(torch.ones_like(pos_scores, device=device))  # Positive labels (1)

        # Negative scores
        src_emb_neg = node_embedding[i, neg_edge_index[0]]
        dst_emb_neg = node_embedding[i, neg_edge_index[1]]
        neg_scores = predictor(src_emb_neg, dst_emb_neg)
        neg_out.append(neg_scores)
        neg_labels.append(torch.zeros_like(neg_scores, device=device))  # Negative labels (0)

    if pos_out:
        pos_out = torch.cat(pos_out, dim=0)
        neg_out = torch.cat(neg_out, dim=0)
        pos_labels = torch.cat(pos_labels, dim=0)
        neg_labels = torch.cat(neg_labels, dim=0)

        # Loss calculation
        pos_loss = -torch.log(pos_out + 1e-10).mean()
        neg_loss = -torch.log(1 - neg_out + 1e-10).mean()

        # Concatenate outputs and labels
        all_outputs = torch.cat([pos_out, neg_out], dim=0)
        all_labels = torch.cat([pos_labels, neg_labels], dim=0)

        return pos_loss + neg_loss, all_outputs, all_labels
    else:
        return torch.tensor(0.0, device=device), torch.tensor([], device=device), torch.tensor([], device=device)


def fine_tune(ft_model: torch.nn.Module, train_dataloader, lr=1e-3, num_epochs=100, max_num_batches=200):
    train_losses = []
    opt = torch.optim.Adam(ft_model.parameters(), lr=lr, weight_decay=1e-4)
    ft_model.train()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Training phase
        train_loss_epoch = 0
        train_batches = len(train_dataloader)
        num_batches = 0
        for batch in  tqdm.tqdm(train_dataloader, desc="Training", leave=False):
            loss, _, _ = link_pred_loss(
                predictor=ft_model,
                node_embedding=batch["x"].to(device), 
                mask_locs=None,
                seq_lengths=batch["seq_lengths"],
                edge_index_list=batch["edges"]
            )
            loss.backward()
            opt.step()
    
            train_loss_epoch += loss.item()
            num_batches += 1
            if num_batches >= max_num_batches:
                break
        train_loss_epoch /= train_batches
        train_losses.append(train_loss_epoch)
        print(f"Train loss: {train_loss_epoch:.4f}")
    return train_losses


def main(args):
    dataset = GeneEmbeddingDataset(args.embedding_path)
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=embedding_collate_fn
    )

    link_predictor = LinkPredictHead(dataset.embedding_dim, 1).to(device)

    fine_tune(
        ft_model=link_predictor,
        train_dataloader=dataloader,
        max_num_batches=20
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--embedding_path", type=str, required=True)
    # parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    main(args)
