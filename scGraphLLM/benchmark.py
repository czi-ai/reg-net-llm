import json
import os 
from os.path import join, abspath, dirname
from typing import Union
import sys
import gc
from argparse import ArgumentParser
from datetime import datetime
from functools import partial
from typing import Dict

from torch_geometric.utils import negative_sampling
import lightning.pytorch as pl

from scGraphLLM.data import *
from scGraphLLM.GNN_modules import GATEncoder 
from scGraphLLM.MLP_modules import LinkPredictHead, RobertaLMHead
from scGraphLLM._globals import *
# from scGraphLLM.flash_transformer import GDTransformer
from scGraphLLM.config import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.metrics import (
    roc_curve, 
    auc,
    precision_recall_curve,
    average_precision_score,
    mean_absolute_error, 
    mean_squared_error, 
    mean_absolute_percentage_error, 
    r2_score
)
from scipy.stats import pearsonr, spearmanr
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


class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, paths):
        self.paths = paths 
        for path in self.paths:
            assert os.path.exists(path), f"File not found: {path} in provided paths list."
        
        # Load data based on paths   
        for path in paths:
            assert os.path.exists(path), f"File not found: {path} in provided paths list."
        embedding = [np.load(path, allow_pickle=True) for path in self.paths]
        self.seq_lengths = np.concatenate([emb["seq_lengths"] for emb in embedding], axis=0)
        self.max_seq_length = np.max(self.seq_lengths)
        self.edges = self.aggregate_embedding_dicts(embedding)
        self.x = np.concatenate([
            np.pad(emb["x"], pad_width=((0, 0), (0, self.max_seq_length - emb["x"].shape[1]), (0, 0)), 
                   mode="constant", constant_values=0)
            for emb in embedding
        ], axis=0)
        self.expression = np.concatenate([
            np.pad(emb["expression"], pad_width=((0,0), (0, self.max_seq_length - emb["expression"].shape[1])),
                   mode="constant", constant_values=0)
            for emb in embedding
        ], axis=0)


    def aggregate_embedding_dicts(self, embedding, key="edges"):
        res = {}
        i = 0
        for emb in embedding:
            d = emb[key].item()
            for j,e in d.items():
                res[i+j] = e
            i += len(d)
        return res

    @property
    def embedding_dim(self):
        return self.x.shape[2]
       
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {
            "x": torch.tensor(self.x[idx]), 
            "expression": torch.tensor(self.expression[idx]),
            "seq_lengths": torch.tensor(self.seq_lengths[idx]),
            "edges": torch.tensor(self.edges[idx])
        }

class EmbeddingDatasetWithExpression(EmbeddingDataset):
    def __init__(self, paths):
        super().__init__(paths)
        embeddings = [np.load(path, allow_pickle=True) for path in self.paths]
        

class EmbeddingDatasetWithGeneMasks(EmbeddingDataset):
    def __init__(self, paths):
        super().__init__(paths)
        embedding = [np.load(path, allow_pickle=True) for path in self.paths]
        self.masked_genes = self.aggregate_embedding_dicts(embedding, key="masks")
        self.expression = self.aggregate_embedding_dicts(embedding, key="masked_expressions")
    
    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        item["masked_genes"] = torch.tensor(self.masked_genes[idx])
        item["masked_expression"] = torch.tensor(self.expression[idx])
        return item
    

class EmbeddingDatasetWithEdgeMasks(EmbeddingDataset):
    def __init__(self, paths, mask_ratio=0.15, generate_edge_masks=False):
        super().__init__(paths)
        self.mask_ratio = mask_ratio
        self.generate_edge_masks = generate_edge_masks
        self.mask_ratio = mask_ratio

        if not generate_edge_masks:
            embedding = [np.load(path, allow_pickle=True) for path in self.paths]
            self.masked_edges = self.aggregate_embedding_dicts(embedding, key="masked_edges")
            self.non_masked_edges = self.aggregate_embedding_dicts(embedding, key="non_masked_edges")
            
            mean_perc_masked_edges = np.mean([
                self.masked_edges[k].shape[1] / self.edges[k].shape[1]
                for k in self.edges.keys()
            ])
            print(f"Mean percentage masked edges {mean_perc_masked_edges:.3f}")

            mean_perc_non_masked_edges = np.mean([
                self.non_masked_edges[k].shape[1] / self.edges[k].shape[1]
                for k in self.edges.keys()
            ])
            print(f"Mean percentage non masked edges {mean_perc_non_masked_edges:.3f}") 
    
    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        if self.generate_edge_masks:
            non_masked_edges, masked_edges = random_edge_mask(item["edges"], self.mask_ratio)
            item["masked_edges"] = masked_edges
            item["non_masked_edges"] = non_masked_edges
        else:
            item["masked_edges"] = torch.tensor(self.masked_edges[idx])
            item["non_masked_edges"] = torch.tensor(self.non_masked_edges[idx])
        return item


def concatenate_embeddings(x_list):
    max_seq_length = max(x.shape[1] for x in x_list)
    x_list_padded = [
        np.pad(x, pad_width=((0, 0), (0, max_seq_length - x.shape[1]), (0, 0)), mode="constant", constant_values=0)
        for x in x_list
    ]
    return np.concatenate(x_list_padded, axis=0)


def embedding_collate_fn(batch, masked_genes=False, masked_edges=False):
    collated = {
        "x": torch.stack([item["x"] for item in batch]),
        "expression": torch.stack([item["expression"] for item in batch]),
        "seq_lengths": torch.tensor([item["seq_lengths"] for item in batch]),
        "edges": [item["edges"] for item in batch]
    }
    
    if masked_genes:
        collated["masked_genes"] = [item["masked_genes"] for item in batch]
        collated["masked_expression"] = [item["masked_expression"] for item in batch]

    if masked_edges:
        collated["masked_edges"] = [item["masked_edges"] for item in batch]
        collated["non_masked_edges"] = [item["non_masked_edges"] for item in batch]

    return collated


class LinkPredictor(nn.Module):
    """Full model: GAT Encoder + Link Prediction Head"""
    def __init__(self, in_channels, hidden_dim, embed_dim, use_gat=False):
        super().__init__()
        self.use_gat = use_gat
        self.encoder = GATEncoder(in_channels, hidden_dim, embed_dim) if self.use_gat else None
        self.link_predictor = LinkPredictHead(embed_dim, output_dim=1)

    def forward(self, x, edge_index, edge_pairs):
        if self.use_gat:
            node_embeddings = self.encoder(x, edge_index)
        else:
            node_embeddings = x
        x_i = node_embeddings[edge_pairs[0]]
        x_j = node_embeddings[edge_pairs[1]]
        return self.link_predictor(x_i, x_j)


class MaskedGeneExpressionPredictor(nn.Module):
    def __init__(self, embed_dim, output_dim=1):
        super().__init__()
        self.roberta_head = RobertaLMHead(embed_dim, output_dim)
        self.mask_embedding = nn.Embedding(2, embed_dim)  # 0 = unmasked, 1 = masked

    def forward(self, node_embedding: torch.Tensor, mask: torch.Tensor):
        """
        node_embedding: (B, G, D) - B=batch, G=genes, D=embed_dim
        mask_status: (B, G) - LongTensor with values 0 (unmasked) or 1 (masked)
        """
        mask_emb = self.mask_embedding(mask)  # (B, G, D)
        combined = node_embedding + mask_emb
        return self.roberta_head(combined)


def generalized_link_pred_loss(
    predictor: LinkPredictor,
    node_embedding,
    edge_index_list,
    masked_edge_index_list=None,
    non_masked_edge_index_list=None,
    mask_locs=None,
    seq_lengths=None,
    device="cuda",
    use_bce_loss=True
):
    pos_preds, neg_preds = [], []
    pos_labels, neg_labels = [], []
    
    batch_size, max_seq_length, _ = node_embedding.shape
    
    for i in range(batch_size):
        edge_index = edge_index_list[i].to(device)
        if masked_edge_index_list is not None:
            assert non_masked_edge_index_list is not None
            masked_edge_index = masked_edge_index_list[i].to(device)
            non_masked_edge_index = non_masked_edge_index_list[i].to(device)
            if masked_edge_index.size(1) == 0:
                continue
            # set positive edges as witheld (masked) edges
            pos_edge_index = masked_edge_index
            operative_edge_index = non_masked_edge_index
        elif mask_locs is not None:
            masked_nodes = torch.where(mask_locs[i])[0].to(device)
            if masked_nodes.numel() == 0:
                continue
            masked_nodes_bool = torch.zeros(max_seq_length, dtype=torch.bool, device=device)
            masked_nodes_bool[masked_nodes] = True
            src_nodes, dst_nodes = edge_index[0], edge_index[1]
            edge_mask = (~masked_nodes_bool[src_nodes]) & (~masked_nodes_bool[dst_nodes])
            pos_edge_index = edge_index[:, edge_mask]
            operative_edge_index = pos_edge_index
        else:
            pos_edge_index = edge_index
            operative_edge_index = edge_index
        
        if pos_edge_index.size(1) == 0:
            continue
        
        # set negative edges as edges not in edge-index
        num_nodes = max_seq_length if seq_lengths is None else seq_lengths[i].item() 
        neg_edge_index = negative_sampling(
            edge_index=edge_index,
            num_nodes=num_nodes,
            num_neg_samples=pos_edge_index.size(1),
            method="sparse"
        ).to(device)
        
        # Positive scores
        # provide the operative edge index to use in the graph convolution
        # for mgm, this is the non-masked portion of the graph, for link prediction, this is the entire graph
        pos_scores = predictor(node_embedding[i], edge_index=operative_edge_index, edge_pairs=pos_edge_index)
        pos_preds.append(pos_scores)
        pos_labels.append(torch.ones_like(pos_scores, device=device))
        
        # Negative scores
        neg_scores = predictor(node_embedding[i], edge_index=operative_edge_index, edge_pairs=neg_edge_index)
        neg_preds.append(neg_scores)
        neg_labels.append(torch.zeros_like(neg_scores, device=device))
        
    if not pos_preds:
        return torch.tensor(0.0, device=device), torch.tensor([], device=device), torch.tensor([], device=device)
    
    # Concatenate results
    pos_preds, neg_preds = torch.cat(pos_preds, dim=0), torch.cat(neg_preds, dim=0)
    pos_labels, neg_labels = torch.cat(pos_labels, dim=0), torch.cat(neg_labels, dim=0)
    
    if use_bce_loss:
        pos_loss = F.binary_cross_entropy(pos_preds, pos_labels)
        neg_loss = F.binary_cross_entropy(neg_preds, neg_labels)
    else:
        pos_loss = -torch.log(pos_preds + 1e-10).mean()
        neg_loss = -torch.log(1 - neg_preds + 1e-10).mean()
    
    loss = pos_loss + neg_loss
    all_outputs, all_labels = torch.cat([pos_preds, neg_preds], dim=0), torch.cat([pos_labels, neg_labels], dim=0)
    
    return loss, all_outputs, all_labels


def masked_gene_expression_pred_loss(
        predictor: nn.Module, 
        node_embedding: torch.Tensor,
        masked_genes: Dict,
        masked_expression: torch.Tensor,
        device="cuda"
    ):
    mask_index = torch.tensor([(i, idx.item()) for i, indices in enumerate(masked_genes) for idx in indices])
    mask = torch.zeros(len(masked_genes), node_embedding.shape[1], dtype=torch.int, device=device)
    mask[mask_index[:,0], mask_index[:,1]] = 1
    
    yhat = predictor(node_embedding, mask).squeeze()
    yhat_masked = yhat[mask_index[:,0], mask_index[:,1]]
    y_masked = torch.cat(masked_expression)
    loss = nn.MSELoss()(yhat_masked, y_masked)
    return loss, yhat_masked, y_masked

def gene_expression_pred_loss(
        predictor: nn.Module, 
        node_embedding: torch.Tensor,
        expression: torch.Tensor,
        seq_lengths: torch.Tensor
    ):
    device = node_embedding.device
    mask = torch.arange(node_embedding.shape[1], device=device).unsqueeze(0) < seq_lengths.unsqueeze(1)
    pred = predictor(node_embedding).squeeze() # B, L
    yhat = pred[mask]
    y = expression[mask]
    loss = nn.MSELoss()(yhat, y)
    return loss, yhat, y


class FineTuneModule(pl.LightningModule):
    def __init__(self, model, task, lr=1e-3, weight_decay=1e-4):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.model.to(self.device)
        self.task = task
        assert self.task in {"link", "mgm", "expr", "mlm"}
    
    def training_step(self, batch, batch_idx):
        loss, _, _ = self._step(batch)
        self.log("train_loss", loss, batch_size=batch["x"].size(0), prog_bar=True, on_epoch=True, on_step=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, _, _ = self._step(batch)
        self.log("val_loss", loss, batch_size=batch["x"].size(0), prog_bar=True, on_epoch=True, on_step=False)

    def predict_step(self, batch, batch_idx):
        """Handles predictions during inference."""
        _, yhat, y = self._step(batch)
        return {"yhat": yhat, "y": y}

    def _step(self, batch):
        if self.task == "link":
            return generalized_link_pred_loss(
                predictor=self.model,
                node_embedding=batch["x"].to(self.device),
                seq_lengths=batch["seq_lengths"],
                edge_index_list=batch["edges"],
                use_bce_loss=False
            )
        elif self.task == "mgm":
            return generalized_link_pred_loss(
                predictor=self.model,
                node_embedding=batch["x"].to(self.device), 
                seq_lengths=batch["seq_lengths"],
                edge_index_list=batch["edges"],
                masked_edge_index_list=batch["masked_edges"],
                non_masked_edge_index_list=batch["non_masked_edges"],
                use_bce_loss=True
            )
        elif self.task == "expr":
            return gene_expression_pred_loss(
                self.model,
                node_embedding=batch["x"],
                expression=batch["expression"],
                seq_lengths=batch["seq_lengths"]
            )
        elif self.task == "mlm":
            return masked_gene_expression_pred_loss(
                self.model,
                node_embedding=batch["x"],
                masked_genes=batch["masked_genes"],
                masked_expression=batch["masked_expression"]
            )

    def configure_optimizers(self):
        return {
            "optimizer": torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay),
            "frequency": 1
        }

def fine_tune_pl(
        ft_model, 
        train_dataloader, 
        task, 
        save_dir, 
        lr,
        weight_decay,
        num_epochs, 
        max_num_batches,
        val_check_interval,
        patience,
        val_dataloader=None
    ):
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss", 
        mode="min",
        verbose=True,
        strict=True,
        patience=patience
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss", 
        mode="min",
        save_top_k=3,
        verbose=True,
        every_n_epochs=1, # FIXME
        save_last=True,
        filename="{epoch}-{step}-{val_loss:.4f}",
        dirpath=save_dir
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = "last-{epoch}-{step}-{val_loss:.4f}"

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        limit_train_batches=max_num_batches,
        val_check_interval=val_check_interval,
        callbacks=[
            early_stop_callback,
            checkpoint_callback
        ],
        accumulate_grad_batches=1,
        deterministic=True
    )
    model = FineTuneModule(
        model=ft_model, 
        task=task,
        lr=lr,
        weight_decay=weight_decay
    )
    trainer.fit(
        model=model, 
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )

    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        print(f"Loading best model from: {best_model_path}")
        best_model = FineTuneModule.load_from_checkpoint(best_model_path, model=ft_model, lr=lr, task=task)
    else:
        print("No best model checkpoint found. Returning the original model.")
        best_model = model

    return best_model


def random_edge_mask(edge_index, mask_ratio=0.15):
    device = edge_index.device
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    E = edge_index.shape[1]
    
    pairs = {}
    for i in range(E):
        u, v = src[i], dst[i]
        mn, mx = (u, v) if u <= v else (v, u)
        if (mn, mx) not in pairs:
            pairs[(mn, mx)] = []
        pairs[(mn, mx)].append(i)
    
    unique_pairs = list(pairs.keys())  
    num_unique = len(unique_pairs)
    num_mask = int(mask_ratio * num_unique)
    
    perm = torch.randperm(num_unique, device=device)
    masked_pairs = [unique_pairs[i.item()] for i in perm[:num_mask]]
    masked_indices = []

    for pair in masked_pairs:
        masked_indices.extend(pairs[pair]) 

    masked_indices = torch.tensor(masked_indices, device=device, dtype=torch.long)
    # get non_masked indices
    all_indices = torch.arange(E, device=device)
    non_masked_indices = all_indices[~torch.isin(all_indices, masked_indices)]

    masked_edge_index = edge_index[:,masked_indices]
    non_masked_edge_index = edge_index[:,non_masked_indices]

    return non_masked_edge_index, masked_edge_index


def predict(model: FineTuneModule, dataloader, task, max_num_batches=None):    
    model.eval().to("cuda")
    yhat = []
    y = []

    n_b = 0
    for batch in tqdm.tqdm(dataloader, leave=False):
        batch = send_to_gpu(batch)
        result = model.predict_step(batch, batch_idx=None)
        yhat_batch = result["yhat"]
        y_batch = result["y"]
        yhat.extend(yhat_batch.cpu().detach().numpy())
        y.extend(y_batch.cpu().detach().numpy())
        n_b += 1
        if max_num_batches is not None and n_b >= max_num_batches:
            break

    if task in {"link", "mgm"}:
        # AUROC
        fpr, tpr, _ = roc_curve(y, yhat)
        auc_score = auc(fpr, tpr)
        
        # PR
        p, r, _ = precision_recall_curve(y, yhat)
        apr = average_precision_score(y, yhat)

        # Sample sizes
        n_pos = np.sum(np.array(y) == 1)
        n_neg = np.sum(np.array(y) == 0)
        
        return fpr, tpr, auc_score, p, r, apr, n_pos, n_neg
    
    elif task in {"expr", "mlm"}: 
        mae = mean_absolute_error(y, yhat)
        mse = mean_squared_error(y, yhat)
        mape = mean_absolute_percentage_error(y, yhat)
        r2 = r2_score(y, yhat)
        pear = pearsonr(y, yhat)[0]
        spear = spearmanr(y, yhat)[0]

        return y, yhat, mae, mse, mape, r2, pear, spear


def plot_auc_roc_pr(fpr_train, tpr_train, auc_score_train, precision_train, recall_train, apr_train,
                    fpr_test, tpr_test, auc_score_test, precision_test, recall_test, apr_test, 
                    save_path=None):

    NEUTRAL = (.1, .1, .1)
    BLUE = (.0, .4, .8)
    RED = (.8, 0, .1)
    PURPLE = (.3, .3, 0.5)
    GREEN = (.2, .5, 0.3)
    plt.style.use("ggplot")
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))

    # ROC Curve
    ax[0].plot(fpr_train, tpr_train, label=f"Train ROC curve (AUC = {auc_score_train:.3f})", color=(*BLUE, 0.6))
    ax[0].plot(fpr_test, tpr_test, label=f"Test ROC curve (AUC = {auc_score_test:.3f})", color=(*RED, 0.6))
    ax[0].plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.5)")
    ax[0].set_xlabel("False Positive Rate")
    ax[0].set_ylabel("True Positive Rate")
    ax[0].set_title("ROC Curve")
    ax[0].legend()

    # Precision-Recall Curve
    ax[1].plot(recall_train, precision_train, label=f"Train PR curve (AP = {apr_train:.3f})", color=(*BLUE, 0.6))
    ax[1].plot(recall_test, precision_test, label=f"Test PR curve (AP = {apr_test:.3f})", color=(*RED, 0.6))
    ax[1].set_xlabel("Recall")
    ax[1].set_ylabel("Precision")
    ax[1].set_title("Precision-Recall Curve")
    ax[1].legend()

    if save_path is not None:
        fig.savefig(save_path)
        
    return fig, ax


def plot_expression_prediction(y, yhat, r2, save_path=None):
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.scatter(y, yhat, alpha=0.6, edgecolor='k', label='Predictions')

        # Reference line: y = x
    min_val = min(np.min(y), np.min(yhat))
    max_val = max(np.max(y), np.max(yhat))
    ax.plot([min_val, max_val], [min_val, max_val], '--', label='Perfect prediction (y = x)')

    ax.text(0.95, 0.05, f"R² = {r2:.3f}", transform=ax.transAxes,
                fontsize=20, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))


    # Labels and formatting
    ax.set_xlabel("Actual Expression", fontsize=20)
    ax.set_ylabel("Predicted Expression", fontsize=20)
    ax.set_title("Predicted vs Actual Expression", fontsize=25)
    ax.legend(loc='upper left', fontsize=15)

    if save_path is not None:
        fig.savefig(save_path)
        
    return fig, ax


def print_dataset_info(name, dataset):
    num_cells, seq_length, embedding_size = dataset.x.shape
    print(f"{name} Dataset: Number of cells: {num_cells:,}, Sequence length: {seq_length:,}, Embedding size: {embedding_size:,}")


def main(args):
    print("Running train.py with args:")
    print(json.dumps(args.__dict__, indent=4))
    info = dict(args=args.__dict__)

    # pytorch lightning seed
    pl.seed_everything(args.random_seed, workers=True)

    print("Loading dataset...")
    if args.model == "scglm":
        if args.task == "link":
            embedding_path_format = "/hpc/mydata/rowan.cassius/data/scGPT/human_immune/cell_type/{}/embeddings/scglm/embedding.npz"
        elif args.task == "mgm":
            embedding_path_format = "/hpc/mydata/rowan.cassius/data/scGPT/human_immune/cell_type/{}/embeddings/scglm/aracne_4096_masked_0.45/embedding.npz"
    elif args.model == "scgpt":
        embedding_path_format = "/hpc/mydata/rowan.cassius/data/scGPT/human_immune/cell_type/{}/embeddings/scgpt/embedding.npz"
    elif args.model == "scf":
        embedding_path_format = "/hpc/mydata/rowan.cassius/data/scGPT/human_immune/cell_type/{}/embeddings/scfoundation/aracne_4096/embedding.npz"
    elif args.model == "gf":
        embedding_path_format = "/hpc/mydata/rowan.cassius/data/scGPT/human_immune/cell_type/{}/embeddings/geneformer/aracne_4096/embedding.npz"

    train_cell_types = [
        "cd14_monocytes",
        "cd20_b_cells",
        "cd8_t_cells",
        "nkt_cells"
    ]
    val_cell_types = [
        "erythrocytes",
        "cd16_monocytes"
    ]
    test_cell_types = [
        "cd4_t_cells",
        "monocyte-derived_dendritic_cells",
        "nk_cells"
    ]
    train_paths = [embedding_path_format.format(cell_type) for cell_type in train_cell_types]
    val_paths = [embedding_path_format.format(cell_type) for cell_type in val_cell_types]
    test_paths = [embedding_path_format.format(cell_type) for cell_type in test_cell_types]

    print("Loading Train and Validation Data...")
    if args.task in {"link", "expr"}:
        train_dataset = EmbeddingDataset(train_paths)
        val_dataset = EmbeddingDataset(val_paths)
        collate_fn = embedding_collate_fn
    elif args.task == "mgm":
        train_dataset = EmbeddingDatasetWithEdgeMasks(
            paths=train_paths, 
            generate_edge_masks=args.generate_edge_masks,
            mask_ratio=args.mask_ratio
        )
        val_dataset = EmbeddingDatasetWithEdgeMasks(
            paths=val_paths, 
            generate_edge_masks=args.generate_edge_masks,
            mask_ratio=args.mask_ratio
        )
        collate_fn = partial(embedding_collate_fn, masked_edges=True)
    elif args.task == "mlm":
        train_dataset = EmbeddingDatasetWithGeneMasks(train_paths)
        val_dataset = EmbeddingDatasetWithGeneMasks(val_paths)
        collate_fn = partial(embedding_collate_fn, masked_genes=True)

    print_dataset_info("Train", train_dataset)
    print_dataset_info("Validation", val_dataset)
    info["train_embedding_tensor"] = train_dataset.x.shape
    info["val_embedding_tensor"] = val_dataset.x.shape

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )   

    if args.task in {"link", "mgm"}:
        predictor = LinkPredictor(
            in_channels=train_dataset.embedding_dim,
            hidden_dim=train_dataset.embedding_dim,
            embed_dim=train_dataset.embedding_dim,
            use_gat=args.use_gat
        )
    elif args.task == "expr":
        predictor = RobertaLMHead(
            embed_dim=train_dataset.embedding_dim,
            output_dim=1
        )
    elif args.task == "mlm":
        predictor = MaskedGeneExpressionPredictor(
            embed_dim=train_dataset.embedding_dim,
            output_dim=1
        )
        
    print(f"Fine tuning link predictor...\n {predictor}")
    best_model: FineTuneModule = fine_tune_pl(
        ft_model=predictor,
        task=args.task,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_num_batches=args.max_num_batches,
        val_check_interval=args.val_check_interval,
        num_epochs=args.num_epochs,
        patience=args.patience,
        save_dir=args.model_save_dir
    )

    print("Making Inference with predictor...")
    result_train = predict(
        best_model, train_dataloader, max_num_batches=None, task=args.task
    )
    
    # Free up memory by deleting datasets and dataloaders
    del train_dataset, val_dataset, train_dataloader, val_dataloader
    gc.collect()  # Run garbage collection to release Python memory

    # If using GPU, empty CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Loading Test Data...")
    if args.task in {"link", "expr"}:
        test_dataset = EmbeddingDataset(test_paths)
    elif args.task == "mgm":
        test_dataset = EmbeddingDatasetWithEdgeMasks(
            paths=test_paths, 
            generate_edge_masks=args.generate_edge_masks,
            mask_ratio=args.mask_ratio
        )
    elif args.task == "mlm":
        test_dataset = EmbeddingDatasetWithGeneMasks(test_paths)

    print_dataset_info("Test", test_dataset)
    info["test_embedding_tensor"] = test_dataset.x.shape

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    result_test = predict(
        best_model, test_dataloader, max_num_batches=None, task=args.task
    )

    if args.task in {"link", "mgm"}:
        fpr_train, tpr_train, auc_score_train, p_train, r_train, apr_train, n_pos_train, n_neg_train = result_train
        fpr_test, tpr_test, auc_score_test, p_test, r_test, apr_test, n_pos_test, n_neg_test = result_test

        save_path = join(args.res_dir, "roc_prc.png")
        print(f"Saving plot to: {save_path}")
        plot_auc_roc_pr(
            fpr_train, tpr_train, auc_score_train, p_train, r_train, apr_train,
            fpr_test, tpr_test, auc_score_test, p_test, r_test, apr_test,
            save_path=save_path
        )

        save_data_path = join(args.res_dir, "roc_prc_data.npz")
        print(f"Saving roc prc data to: {save_data_path}")
        np.savez(save_data_path,
            # Test metrics
            fpr_test=fpr_test,
            tpr_test=tpr_test,
            auc_score_test=auc_score_test,
            p_test=p_test,
            r_test=r_test,
            apr_test=apr_test,
            n_pos_test=n_pos_test,
            n_neg_test=n_neg_test,
            # Train metrics
            fpr_train=fpr_train,
            tpr_train=tpr_train,
            auc_score_train=auc_score_train,
            p_train=p_train,
            r_train=r_train,
            apr_train=apr_train,
            n_pos_train=n_pos_train,
            n_neg_train=n_neg_train
        )

    elif args.task in {"mlm", "expr"}:
        y_train, yhat_train, mae_train, mse_train, mape_train, r2_train, pear_train, spear_train = result_train
        y_test, yhat_test, mae_test, mse_test, mape_test, r2_test, pear_test, spear_test = result_test
        
        save_path = join(args.res_dir, "scatter_test_pred.png")
        print(f"Saving plot to: {save_path}")
        plot_expression_prediction(y_test, yhat_test, r2=r2_test, save_path=save_path)

        save_data_path = join(args.res_dir, "mlm_data.npz")
        print(f"Saving performance data to: {save_data_path}")
        np.savez(save_data_path,
            # Test metrics
            y_test=y_test,
            yhat_test=yhat_test,
            mse_test=mse_test,
            mae_test=mae_test,
            mape_test=mape_test,
            r2_test=r2_test,
            pear_test=pear_test,
            spear_test=spear_test,
            # Train metrics
            y_train=y_train,
            yhat_train=yhat_train,
            mse_train=mse_train,
            mae_train=mae_train,
            mape_train=mape_train,
            r2_train=r2_train,
            pear_train=pear_train,
            spear_train=spear_train
        )

    with open(args.info_path, "w") as file:
            json.dump(info, file, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--suffix", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=["scglm", "scgpt", "scf", "gf"])
    parser.add_argument("--task", type=str, required=True, choices=["link", "mgm", "expr", "mlm"])
    parser.add_argument("--generate_edge_masks", action="store_true")
    parser.add_argument("--mask_ratio", type=float, default=0.15)
    parser.add_argument("--use_gat", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_num_batches", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--val_check_interval", type=Union[int, float], default=1.0)
    parser.add_argument("--random_seed", default=0)
    # effectively, early stopping will kick in if the model has not improved after
    # seeing (batch_size * val_check_interval * patience) cells
    # e.g, trainer will stop if validation score hasn't improved after
    # training on batch_size=8 * val_check_interval=16 * patience=8 = 1024 cells
    args = parser.parse_args()
    
    args.res_dir = join(args.out_dir, f"{args.suffix}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}")
    args.model_save_dir = join(args.res_dir, "model")
    args.info_path = join(args.res_dir, f"info.json")
    os.makedirs(args.res_dir, exist_ok=False)
    main(args)
