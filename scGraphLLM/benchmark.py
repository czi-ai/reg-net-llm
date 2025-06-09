import json
import os 
import gc
import tqdm
from os.path import join
from argparse import ArgumentParser
from datetime import datetime
from functools import partial
from typing import Optional, Tuple, Literal
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_curve, 
    auc,
    precision_recall_curve,
    average_precision_score,
    mean_absolute_error, 
    mean_squared_error, 
    mean_absolute_percentage_error, 
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torch.utils.data import Subset, random_split
from torch_geometric.utils import negative_sampling
from torch.utils.data import Subset, DataLoader, random_split

from scGraphLLM.data import *
from scGraphLLM.GNN_modules import GATEncoder 
from scGraphLLM.MLP_modules import LinkPredictHead, RobertaLMHead
from scGraphLLM._globals import *
from scGraphLLM.config import *
from scGraphLLM.eval_config import EMBEDDING_DATASETS, SPLIT_CONFIGS
from scGraphLLM.embedding import (
    EmbeddingDataset, 
    EmbeddingDatasetWithEdgeMasks, 
    EmbeddingDatasetWithGeneMasks,
    embedding_collate_fn
)

# colors
NEUTRAL = (.1, .1, .1)
BLUE = (.0, .4, .8)
RED = (.8, 0, .1)
PURPLE = (.3, .3, 0.5)
GREEN = (.2, .5, 0.3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LinkPredictor(nn.Module):
    """
    A full link prediction model consisting of an optional GAT encoder and a link prediction head.

    This model takes node features and a set of node pairs (edges) and predicts a scalar score 
    for each pair, indicating the likelihood of a link.

    Args:
        in_channels (int): Dimension of input node features.
        hidden_dim (int): Hidden dimension used in the GAT encoder (if enabled).
        embed_dim (int): Output embedding dimension from the GAT encoder.
        use_gat (bool): If True, applies a GATEncoder to input features before prediction.

    Components:
        - encoder: A GAT-based encoder (optional).
        - link_predictor: A pairwise link prediction head operating on node embeddings.
    """
    def __init__(self, in_channels, hidden_dim, embed_dim, use_gat=False):
        super().__init__()
        self.use_gat = use_gat
        self.encoder = GATEncoder(in_channels, hidden_dim, embed_dim) if self.use_gat else None
        self.link_predictor = LinkPredictHead(embed_dim, output_dim=1)

    def forward(self, x, edge_index, edge_pairs):
        """
        Forward pass for the link predictor.

        Args:
            x (Tensor): Node feature matrix of shape (B, N, D) or (B, D).
            edge_index (Tensor): Edge index tensor of shape (2, E), representing the graph.
            edge_pairs (Tuple[Tensor, Tensor]): Two tensors of indices representing pairs of nodes
                                                to evaluate for link prediction.

        Returns:
            Tensor: Predicted link scores of shape (num_pairs, 1).
        """
        if self.use_gat:
            node_embeddings = self.encoder(x, edge_index)
        else:
            node_embeddings = x
        x_i = node_embeddings[edge_pairs[0]]
        x_j = node_embeddings[edge_pairs[1]]
        return self.link_predictor(x_i, x_j)
    

def generalized_link_pred_loss(
        predictor,
        node_embedding: torch.Tensor,
        edge_index_list: list[torch.Tensor],
        masked_edge_index_list: list[torch.Tensor] = None,
        non_masked_edge_index_list: list[torch.Tensor] = None,
        mask_locs: torch.Tensor = None,
        seq_lengths: torch.Tensor = None,
        use_bce_loss: bool = True,
        device: str = "cuda"
    ):
    """
    Computes the generalized link prediction loss over a batch of graphs with optional masking.

    Args:
        predictor: A model that takes (node_embedding, edge_index, edge_pairs) and returns
                   edge scores in [0,1].
        node_embedding (torch.Tensor): Node embeddings with shape (batch_size, max_seq_length, embed_dim).
        edge_index_list (list[torch.Tensor]): List of edge_index tensors (2 x E) for each graph in batch.
        masked_edge_index_list (list[torch.Tensor], optional): List of masked (positive) edges per graph.
        non_masked_edge_index_list (list[torch.Tensor], optional): List of non-masked edges per graph.
        mask_locs (torch.Tensor, optional): Boolean mask tensor (batch_size x max_seq_length) indicating masked nodes.
        seq_lengths (torch.Tensor, optional): Lengths of sequences per batch element, shape (batch_size,).
        device (str, optional): Device string, e.g., "cuda" or "cpu".
        use_bce_loss (bool, optional): Whether to use binary cross-entropy loss (True) or log loss (False).

    Returns:
        loss (torch.Tensor): Scalar loss tensor.
        all_outputs (torch.Tensor): Concatenated predicted scores for positive and negative edges.
        all_labels (torch.Tensor): Concatenated ground truth labels (1 for positive, 0 for negative).
    
    Notes:
        - If both masked_edge_index_list and non_masked_edge_index_list are provided, the positive
          edges are considered to be the masked edges, and the operative graph used for message passing
          is the non-masked edges.
        - If mask_locs is provided (and masked_edge_index_list is None), edges connected to masked nodes
          are excluded when selecting positive edges.
        - If no masking information is provided, all edges are considered positive.
        - Negative edges are sampled dynamically per batch element to balance positive examples.
    """
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


class CellClassifier(nn.Module):
    """
    A neural network model for classifying cells based on node embeddings.

    Optionally uses a GAT encoder followed by a customizable MLP. Supports
    various pooling strategies (mean, max, or both) over the sequence of nodes.

    Args:
        input_dim (int): Dimension of input node features.
        num_classes (int): Number of output classes.
        class_weights (Optional[List[float]]): Class weights for loss computation (for imbalanced data).
        num_layers (int): Number of layers in the classification head MLP.
        use_gat (bool): Whether to use a GAT encoder before classification.
        lr (float): Learning rate (stored for convenience).
        hidden_dim (Optional[int]): Hidden layer size; defaults to `input_dim` if not provided.
        pooling (str): Pooling strategy to aggregate node features. Options: "mean", "max", "both".
    """
    def __init__(
            self, 
            input_dim: int, 
            num_classes: int, 
            class_weights: Optional[List[float]] = None, 
            num_layers: int = 1, 
            use_gat: bool = False, 
            lr: float = 1e-3, 
            hidden_dim: Optional[int] = None, 
            pooling: Literal["mean", "max", "both"] = "mean"
        ):
        super().__init__()
        self.lr = lr
        self.hidden_dim = hidden_dim or input_dim
        self.use_gat = use_gat
        self.pooling = pooling.lower()
        self.encoder = GATEncoder(input_dim, self.hidden_dim, self.hidden_dim) if self.use_gat else None
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device) if class_weights is not None else None

        layers = []
        in_dim = self.encoder.out_channels if self.use_gat else input_dim
        hidden_dim = self.hidden_dim

        if self.pooling == "both":
            in_dim *= 2

        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x, edge_index_list=None, seq_lengths=None):
        """
        Forward pass through the model.

        Args:
            x (Tensor): Input node features of shape (B, N, D).
            edge_index_list (Optional[List[Tensor]]): List of edge indices per sample.
            seq_lengths (Optional[Tensor]): Sequence lengths for each sample in the batch.

        Returns:
            Tensor: Logits of shape (B, num_classes).
        """
        B, N, D = x.shape
        if self.use_gat:
            x = x.view(B * N, D)

            # build batched edge_index
            edge_indices = []
            for i, ei in enumerate(edge_index_list):
                edge_indices.append(ei + i * N)
            edge_index = torch.cat(edge_indices, dim=1)

            x = self.encoder(x, edge_index)
            x = x.view(B, N, -1)

        mask = torch.arange(N, device=x.device).unsqueeze(0) < seq_lengths.unsqueeze(1)
        mask_float = mask.unsqueeze(-1).float()
        x_masked = x * mask_float

        if self.pooling == "mean":
            x_pooled = x_masked.sum(dim=1) / mask_float.sum(dim=1)
        elif self.pooling == "max":
            x_masked[~mask] = float('-inf')
            x_pooled, _ = x_masked.max(dim=1)
        elif self.pooling == "both":
            x_mean = x_masked.sum(dim=1) / mask_float.sum(dim=1)
            x_masked[~mask] = float('-inf')
            x_max, _ = x_masked.max(dim=1)
            x_pooled = torch.cat([x_mean, x_max], dim=1)
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling}")

        return self.net(x_pooled)


def cell_classification_loss(
        predictor,
        node_embedding: torch.Tensor,
        edge_index_list: List[torch.Tensor],
        labels: torch.Tensor,
        seq_lengths: Optional[torch.Tensor] = None,
        class_weights: Optional[torch.Tensor] = None,
        device: str = "cuda"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the cross-entropy loss for cell classification and return predictions.

    Args:
        predictor: A model that outputs class logits from node embeddings and edges.
        node_embedding (Tensor): Node embeddings of shape (N, F).
        edge_index_list (List[Tensor]): List of edge index tensors per sample or batch.
        labels (Tensor): Ground-truth class labels of shape (N,).
        seq_lengths (Optional[Tensor]): Optional sequence lengths, if required by the predictor.
        class_weights (Optional[Tensor]): Class weights tensor for imbalanced classification.
        device (str): Device to compute the loss on (e.g., "cuda" or "cpu").

    Returns:
        Tuple[Tensor, Tensor, Tensor]:
            - loss: Cross-entropy loss tensor (scalar).
            - preds: Predicted class indices.
            - labels: Ground truth labels (moved to target device).
    """
    logits = predictor(node_embedding, edge_index_list, seq_lengths)
    labels = labels.to(device)
    loss = F.cross_entropy(logits, labels, weight=class_weights)
    preds = torch.argmax(logits, dim=1)
    return loss, preds, labels
    

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


def masked_gene_expression_pred_loss(
        predictor: nn.Module, 
        node_embedding: torch.Tensor,
        masked_genes: dict,
        masked_expression: torch.Tensor,
        device="cuda"
    ):
    """
    Computes mean squared error loss for predicting masked gene expression values.

    Args:
        predictor (nn.Module): Model that predicts gene expression from node embeddings and a mask.
        node_embedding (torch.Tensor): Node embeddings tensor of shape (batch_size, num_genes, embed_dim).
        masked_genes (dict): Dictionary mapping batch indices to lists or tensors of masked gene indices.
        masked_expression (torch.Tensor): Concatenated true expression values corresponding to masked genes.
        device (str, optional): Device on which tensors are allocated (default: "cuda").

    Returns:
        loss (torch.Tensor): Mean squared error loss computed only on masked gene predictions.
        yhat_masked (torch.Tensor): Predicted expression values for masked genes.
        y_masked (torch.Tensor): Ground truth expression values for masked genes.
    
    Notes:
        - The mask is constructed as a binary tensor with 1 indicating masked gene locations.
        - Only predictions at masked gene positions are used to compute the loss.
    """
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
    """
    Computes mean squared error loss between predicted and true gene expression values
    for valid (non-padded) sequence positions.

    Args:
        predictor (nn.Module): Model that predicts gene expression from node embeddings.
        node_embedding (torch.Tensor): Tensor of node embeddings with shape [batch_size, seq_length, embed_dim].
        expression (torch.Tensor): Ground truth gene expression tensor with shape [batch_size, seq_length].
        seq_lengths (torch.Tensor): 1D tensor containing valid sequence lengths per batch element.

    Returns:
        loss (torch.Tensor): Mean squared error loss over valid gene positions.
        yhat (torch.Tensor): Predicted gene expression values corresponding to valid positions.
        y (torch.Tensor): True gene expression values corresponding to valid positions.

    Notes:
        - Padding positions beyond the valid sequence length are masked out and excluded from the loss.
    """
    device = node_embedding.device
    mask = torch.arange(node_embedding.shape[1], device=device).unsqueeze(0) < seq_lengths.unsqueeze(1)
    pred = predictor(node_embedding).squeeze(-1) # From [B, L, 1] → [B, L]
    yhat = pred[mask]
    y = expression[mask]
    loss = nn.MSELoss()(yhat, y)
    return loss, yhat, y


class FineTuneModule(pl.LightningModule):
    """
    PyTorch Lightning module for fine-tuning models on various biological tasks.

    Supports tasks:
      - "link": Link prediction using generalized_link_pred_loss.
      - "mgm": Masked graph modeling (graph edge prediction) using generalized_link_pred_loss.
      - "cls": Cell classification using cross-entropy loss.
      - "expr": Gene expression prediction using mean squared error loss.
      - "mlm": Masked gene expression prediction using mean squared error loss.

    Args:
        model (nn.Module): The model to be fine-tuned.
        task (str): Task identifier; one of {"link", "mgm", "cls", "expr", "mlm"}.
        lr (float, optional): Learning rate for the optimizer. Default: 1e-3.
        weight_decay (float, optional): Weight decay (L2 regularization) for optimizer. Default: 1e-4.

    Methods:
        training_step(batch, batch_idx):
            Executes one training step and logs training loss.

        validation_step(batch, batch_idx):
            Executes one validation step and logs validation loss.

        predict_step(batch, batch_idx):
            Performs prediction during inference, returns predictions and targets.

        _step(batch):
            Internal method to compute loss and predictions for the given batch
            depending on the task type.

        configure_optimizers():
            Sets up Adam optimizer with specified learning rate and weight decay.
    """
    def __init__(self, model, task, lr=1e-3, weight_decay=1e-4):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.model.to(self.device)
        self.task = task
        assert self.task in {"link", "mgm", "cls", "expr", "mlm"}
    
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
                node_embedding=batch["x"],
                seq_lengths=batch["seq_lengths"],
                edge_index_list=batch["edges"],
                use_bce_loss=False
            )
        elif self.task == "mgm":
            return generalized_link_pred_loss(
                predictor=self.model,
                node_embedding=batch["x"], 
                seq_lengths=batch["seq_lengths"],
                edge_index_list=batch["edges"],
                masked_edge_index_list=batch["masked_edges"],
                non_masked_edge_index_list=batch["non_masked_edges"],
                use_bce_loss=True
            )
        elif self.task == "cls":
            return cell_classification_loss(
                predictor=self.model,
                node_embedding=batch["x"],
                edge_index_list=batch["edges"],
                seq_lengths=batch["seq_lengths"],
                labels=batch["y"],
                class_weights=self.model.class_weights
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
    """
    Fine-tunes a given PyTorch model using PyTorch Lightning with early stopping and checkpointing.

    Args:
        ft_model (nn.Module): The model to be fine-tuned.
        train_dataloader (DataLoader): DataLoader for training data.
        task (str): Task type (e.g., "link", "mgm", "cls", "expr", "mlm") passed to FineTuneModule.
        save_dir (str): Directory path to save model checkpoints.
        lr (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay (L2 regularization) for the optimizer.
        num_epochs (int): Maximum number of epochs for training.
        max_num_batches (int or float): Maximum number or fraction of batches per epoch to train on (for debugging or partial training).
        val_check_interval (float or int): Interval (in fraction or batches) to check validation performance.
        patience (int): Number of validation checks with no improvement before early stopping triggers.
        val_dataloader (DataLoader, optional): DataLoader for validation data. If None, no validation will be performed.

    Returns:
        FineTuneModule: The trained FineTuneModule instance loaded with the best checkpoint if available, otherwise the last trained model.
    """
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


def predict(model: FineTuneModule, dataloader, task, max_num_batches=None):
    """
    Generates predictions from a fine-tuned model on a given dataset and computes evaluation metrics based on the task.

    Args:
        model (FineTuneModule): The fine-tuned PyTorch Lightning module to use for prediction.
        dataloader (DataLoader): DataLoader providing batches of data for prediction.
        task (str): The prediction task type. Supported values:
            - "link" or "mgm": link prediction tasks (binary classification on edges).
            - "expr" or "mlm": regression tasks predicting gene expression.
            - "cls": cell classification task (multi-class classification).
        max_num_batches (int or None): Optional maximum number of batches to predict on. If None, predict on all batches.
    """
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
    
    if task == "cls":
        pre_w = precision_score(y, yhat, average="weighted")
        rec_w = recall_score(y, yhat, average="weighted")
        f1_w = f1_score(y, yhat, average="weighted")
        acc = accuracy_score(y, yhat)

        pre_mic = precision_score(y, yhat, average="micro")
        rec_mic = recall_score(y, yhat, average="micro")
        f1_mic = f1_score(y, yhat, average="micro")

        pre_mac = precision_score(y, yhat, average="macro")
        rec_mac = recall_score(y, yhat, average="macro")
        f1_mac = f1_score(y, yhat, average="macro")

        # convert labels
        ds = dataloader.dataset
        while isinstance(ds, Subset):
            ds = ds.dataset
        label_encoder: LabelEncoder = ds.label_encoder
        y = label_encoder.inverse_transform(y)
        yhat = label_encoder.inverse_transform(yhat)

        return y, yhat, acc, pre_mac, rec_mac, f1_mac
    

def plot_auc_roc_pr(fpr_train, tpr_train, auc_score_train, precision_train, recall_train, apr_train,
                    fpr_test, tpr_test, auc_score_test, precision_test, recall_test, apr_test, 
                    save_path=None):
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


def plot_confusion_matrix(y, yhat, normalize=True, title="Confusion Matrix", base_color=GREEN, save_path=None):
    light_color = tuple(0.5 * g + 0.5 for g in base_color)
    custom_cmap = LinearSegmentedColormap.from_list(name="light_color", colors=[(1, 1, 1), light_color])

    # Compute confusion matrix
    labels = sorted(np.unique(np.concatenate([y, yhat])))
    cm = confusion_matrix(y, yhat, labels=labels)
    
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, where=row_sums != 0)  # avoid division by zero

    # Calculate metrics
    accuracy = accuracy_score(y, yhat)

    # Plot using matplotlib
    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(cm, cmap=custom_cmap)

    # Add text annotations
    for i in range(len(cm)):
        for j in range(len(cm)):
            value = cm[i, j]
            display_val = f"{value:.2f}" if normalize else f"{int(value):,}"
            ax.text(j, i, display_val, ha='center', va='center', color='black', fontsize=15)

    # Set labels
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=15)
    ax.set_yticklabels(labels, fontsize=15)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xlabel('Predicted label', fontsize=20)
    ax.set_ylabel('True label', fontsize=20)
    ax.set_title(f"{title} (Accuracy = {accuracy:.3f})", fontsize=25)

    if save_path is not None:
        fig.savefig(save_path)

    return fig, ax


def print_dataset_info(name, dataset):
    print(f"{name} Dataset: Number of cells: {len(dataset):,}, Max sequence length: {dataset.max_seq_length:,}, Embedding size: {dataset.embedding_dim:,}")


def split_dataset(
        dataset, 
        ratio_config=(None, None, None), 
        metadata_config=None,
        filter_config=None,
        seed=42):
    """
    Splits a dataset into training, validation, and test subsets based on metadata criteria and/or specified ratios.

    This function supports:
    - Filtering the dataset by metadata keys and values (inclusion or exclusion).
    - Splitting by metadata values for each subset (train/val/test).
    - Random splitting by specified ratios for remaining data after metadata splits.
    - Ensuring no overlap in metadata-based splits.
    - Handling cases where some splits use metadata and others use ratios.
    
    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        The full dataset to be split. Should have a `metadata` attribute (dict or DataFrame-compatible).
    ratio_config : tuple of float or None, default=(None, None, None)
        Ratios for random splitting into (train, val, test) subsets. Values should sum to 1.0 if specified.
        Use None for splits that will be defined by metadata_config.
    metadata_config : tuple (str, list of lists) or None, default=None
        Metadata key and list of metadata values defining each subset.
        For example, ('batch', [['batch1', 'batch2'], None, ['batch3']]) assigns train to batches 1 & 2, test to batch3, and val by ratio.
    filter_config : dict or None, default=None
        Dictionary specifying filters on metadata before splitting.
        Format: {key: {"values": [...], "mode": "include" or "exclude"}}.
        Only samples matching filters are kept.
    seed : int, default=42
        Random seed for reproducible splitting when using ratios.

    Returns
    -------
    train_dataset, val_dataset, test_dataset : torch.utils.data.Subset or None
        Subsets of the dataset corresponding to train, val, and test splits.
        Some may be None if not defined.

    Raises
    ------
    ValueError
        If ratios do not sum to 1, or both ratio and metadata are specified for the same split,
        or if metadata split values overlap between splits, or invalid filter mode is specified.
    """
    # get metadata as dataframe
    if (metadata_config is not None) or (filter_config is not None):
        metadata = pd.DataFrame(dataset.metadata)

    # Filtering
    if filter_config:
        mask = pd.Series(True, index=metadata.index)
        for key, config in filter_config.items():
            values = config["values"]
            mode = config.get("mode", "include")
            if mode == "include":
                mask &= metadata[key].isin(values)
            elif mode == "exclude":
                mask &= ~metadata[key].isin(values)
            else:
                raise ValueError(f"Unknown mode '{mode}' for filter_config[{key}]. Use 'include' or 'exclude'.")

        filtered_indices = metadata[mask].index.tolist()
        dataset = Subset(dataset, filtered_indices)

        metadata = metadata.loc[filtered_indices].reset_index(drop=True)

    # Splitting
    if ratio_config != (None, None, None):
        specified_ratios = [r for r in ratio_config if r is not None]
        if specified_ratios and not abs(sum(specified_ratios) - 1.0) < 1e-6:
            raise ValueError("Specified ratios must sum to 1.0.")

    # Validate that each split is defined by either metadata or ratio, but not both
    if metadata_config:
        key, split_values = metadata_config
        for i, (values, ratio) in enumerate(zip(split_values, ratio_config)):
            split = ["train", "val", "test"][i]
            if (values is None) and (ratio is None):
                raise ValueError(f"Split {split} has neither metadata values nor ratio specified. Please specify only one.")
            elif (values is None) and (ratio is not None):
                pass
            elif (values is not None) and (ratio is None):
                pass
            elif (values is not None) and (ratio is not None):
                raise ValueError(f"Split {split} has both metadata values and a ratio specified. Please specify only one.")
        
        # Check for overlapping metadata values across splits
        all_values = [value for values in split_values if values is not None for value in values]
        if len(all_values) != len(set(all_values)):
            raise ValueError("Overlapping metadata values detected across splits.")
    
    meta_split_datasets = [None, None, None]
    if metadata_config:
        key, split_values = metadata_config 
        meta_split_datasets = [
            Subset(dataset, indices=metadata[metadata[key].isin(values)].index.tolist()) 
                if values is not None else None
            for values in split_values
        ]
        if all(dataset is not None for dataset in meta_split_datasets):
            return meta_split_datasets

    remaining_indices = set(range(len(dataset))) - {
        idx for subset in meta_split_datasets if subset is not None for idx in subset.indices
    }
    remaining_dataset = Subset(dataset, list(remaining_indices))
    remaining_total = len(remaining_dataset)
    remaining_sizes = [
        int(ratio * remaining_total) if ratio is not None else 0
        for ratio in ratio_config
    ]
    discrepancy = remaining_total - sum(remaining_sizes)
    # Find the last split that was defined by a ratio, and add the leftover
    for i in reversed(range(len(remaining_sizes))):
        if ratio_config[i] is not None:
            remaining_sizes[i] += discrepancy
            break

    remaining_split_datasets = random_split(remaining_dataset, lengths=remaining_sizes, generator=torch.Generator().manual_seed(seed))

    train_dataset, val_dataset, test_dataset = (
        meta_split_dataset if meta_split_dataset is not None else rem_split_dataset
        for meta_split_dataset, rem_split_dataset in zip(meta_split_datasets, remaining_split_datasets)
    )

    return train_dataset, val_dataset, test_dataset


def main(args):
    print("Running train.py with args:")
    print(json.dumps(args.__dict__, indent=4))
    info = dict(args=args.__dict__)

    # pytorch lightning seed
    pl.seed_everything(args.random_seed, workers=True)

    print("Loading Data...")
    if args.task == "link":
        dataset = EmbeddingDataset(
            path=args.data_paths,
            with_metadata=True
        )
        collate_fn = embedding_collate_fn
    elif args.task == "expr":
        dataset = EmbeddingDataset(
            paths=args.data_paths,
            with_expression=True,
            with_metadata=True
        )
        collate_fn = partial(embedding_collate_fn, expression=True)
    elif args.task == "cls":
        dataset = EmbeddingDataset(
            paths=args.data_paths,
            target_metadata_key=args.target
        )
        collate_fn=partial(embedding_collate_fn, target_label=True)
    elif args.task == "mgm":
        dataset = EmbeddingDatasetWithEdgeMasks(
            paths=args.data_paths,
            generate_edge_masks=args.generate_edge_masks,
            mask_ratio=args.mask_ratio,
            with_metadata=True
        )
        collate_fn = partial(embedding_collate_fn, masked_edges=True)
    elif args.task == "mlm":
        dataset = EmbeddingDatasetWithGeneMasks(args.data_paths)
        collate_fn = partial(embedding_collate_fn, masked_genes=True)

    # Split Dataset according to split config
    train_dataset, val_dataset, test_dataset = split_dataset(
        dataset=dataset,
        metadata_config=args.split_config.metadata_config,
        ratio_config=args.split_config.ratio_config,
        filter_config=getattr(args.split_config, "filter_config", None)
    )
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    info["embedding_tensor"] = dataset.shape
    info["train_size"] = len(train_dataset)
    info["val_size"] = len(val_dataset)
    info["test_size"] = len(test_dataset)

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

    # Initialize model to train
    if args.task in {"link", "mgm"}:
        predictor = LinkPredictor(
            in_channels=dataset.embedding_dim,
            hidden_dim=dataset.embedding_dim,
            embed_dim=dataset.embedding_dim,
            use_gat=args.use_gat
        )
    elif args.task == "cls":
        predictor = CellClassifier(
            input_dim=dataset.embedding_dim,
            num_classes=dataset.num_classes,
            class_weights=dataset.class_weights if args.use_weighted_ce else None,
            use_gat=args.use_gat,
            num_layers=args.cls_layers,
            pooling="mean"
        )
    elif args.task == "expr":
        predictor = RobertaLMHead(
            embed_dim=dataset.embedding_dim,
            output_dim=1
        )
    elif args.task == "mlm":
        predictor = MaskedGeneExpressionPredictor(
            embed_dim=dataset.embedding_dim,
            output_dim=1
        )

    if args.prediction:
        print(f"Loading the following model for prediction... {args.model_path}")
        best_model = FineTuneModule.load_from_checkpoint(args.model_path, model=predictor, lr=args.lr, task=args.task)    
    else:   
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

    print("Making Inference with predictor for train set...")
    result_train = predict(
        best_model, train_dataloader, max_num_batches=None, task=args.task
    )

    print("Making Inference with predictor for validation set...")
    result_val = predict(
        best_model, val_dataloader, max_num_batches=None, task=args.task
    )
    
    # Free up memory by deleting datasets and dataloaders
    del train_dataset, val_dataset, train_dataloader, val_dataloader
    gc.collect()

    # If using GPU, empty CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Run inference on test set
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
    elif args.task == "cls":
        y_train, yhat_train, acc_train, pre_train, rec_train, f1_train = result_train
        y_test, yhat_test, acc_test, pre_test, rec_test, f1_test = result_test
        y_val, yhat_val, acc_val, pre_val, rec_val, f1_val = result_val

        save_path_test = join(args.res_dir, "cm_test.png")
        print(f"Saving test confusion matrix to: {save_path_test}")
        plot_confusion_matrix(y_test, yhat_test, normalize=True, save_path=save_path_test)

        save_path_train = join(args.res_dir, "cm_train.png")
        print(f"Saving train confusion matrix to: {save_path_train}")
        plot_confusion_matrix(y_train, yhat_train, normalize=True, save_path=save_path_train)

        save_path_val = join(args.res_dir, "cm_val.png")
        print(f"Saving val confusion matrix to: {save_path_val}")
        plot_confusion_matrix(y_val, yhat_val, normalize=True, save_path=save_path_val)
        
        save_data_path = join(args.res_dir, "cls_data.npz")
        print(f"Saving classification performance data to: {save_data_path}")
        np.savez(save_data_path,
            # Train metrics
            y_train=y_train,
            yhat_train=yhat_train,
            acc_train=acc_train,
            pre_train=pre_train,
            rec_train=rec_train,
            f1_train=f1_train,
            # Test Metrics
            y_test=y_test,
            yhat_test=yhat_test,
            acc_test=acc_test,
            pre_test=pre_test,
            rec_test=rec_test,
            f1_test=f1_test,
            # Val Metrics
            y_val=y_val,
            yhat_val=yhat_val,
            acc_val=acc_val,
            pre_val=pre_val,
            rec_val=rec_val,
            f1_val=f1_val
        )

    with open(args.info_path, "w") as file:
        json.dump(info, file, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split_config", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--prediction", action="store_true")
    parser.add_argument("--suffix", type=str, required=True)
    parser.add_argument("--task", type=str, required=True, choices=["link", "mgm", "expr", "mlm", "cls"])
    parser.add_argument("--cls_layers", type=int, default=1)
    parser.add_argument("--use_weighted_ce", action="store_true")
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--generate_edge_masks", action="store_true")
    parser.add_argument("--mask_ratio", type=float, default=0.15)
    parser.add_argument("--use_gat", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_num_batches", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--val_check_interval", type=float, default=1.0)
    parser.add_argument("--random_seed", default=0)
    args = parser.parse_args()
    # effectively, early stopping will kick in if the model has not improved after
    # seeing (batch_size * val_check_interval * patience) cells
    # e.g, trainer will stop if validation score hasn't improved after
    # training on batch_size=8 * val_check_interval=16 * patience=8 = 1024 cells
    
    args.res_dir = join(args.out_dir, f"{args.suffix}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}")
    args.model_save_dir = join(args.res_dir, "model")
    args.info_path = join(args.res_dir, f"info.json")
    os.makedirs(args.res_dir, exist_ok=False)

    # get paths of dataset
    args.data_paths = EMBEDDING_DATASETS[args.dataset]
    args.split_config = SPLIT_CONFIGS[args.split_config]

    main(args)
