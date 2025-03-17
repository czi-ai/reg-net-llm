import os 
from os.path import join, abspath, dirname
import sys
import gc
from argparse import ArgumentParser
from datetime import datetime
from functools import partial

from torch_geometric.utils import negative_sampling
import lightning.pytorch as pl

from scGraphLLM.data import *
from scGraphLLM.GNN_modules import GATEncoder 
from scGraphLLM.MLP_modules import LinkPredictHead
from scGraphLLM._globals import *
# from scGraphLLM.flash_transformer import GDTransformer
from scGraphLLM.config import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, random_split

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
        for path in self.paths:
            assert os.path.exists(path), f"File not found: {path} in provided paths list."
        
        # Load data based on paths   
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
            n_samples = len(edges)
            for j,e in edges.items():
                self.edges[i+j] = e
            i += n_samples
        
    @property
    def embedding_dim(self):
        return self.x.shape[2]
       
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {
            "x": torch.tensor(self.x[idx]), 
            "seq_lengths": torch.tensor(self.seq_lengths[idx]),
            "edges": torch.tensor(self.edges[idx])
        }

class GeneEmbeddingDatasetWithMasks(GeneEmbeddingDataset):
    def __init__(self, paths):
        super().__init__(paths)
        embedding = [np.load(path, allow_pickle=True) for path in self.paths]
        self.masked_edges = {}
        i = 0
        for emb in embedding:
            n_samples = len(emb["x"])
            masked_edges = emb["masked_edges"].item()
            for j,e in masked_edges.items():
                self.masked_edges[i+j] = e
            i += n_samples
    
    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        item["masked_edges"] = torch.tensor(self.masked_edges[idx])
        return item

def concatenate_embeddings(x_list):
    max_seq_length = max(x.shape[1] for x in x_list)
    x_list_padded = [
        np.pad(x, pad_width=((0, 0), (0, max_seq_length - x.shape[1]), (0, 0)), mode="constant", constant_values=0)
        for x in x_list
    ]
    return np.concatenate(x_list_padded, axis=0)


def embedding_collate_fn(batch, masked_edges=False):
    collated = {
        "x": torch.stack([item["x"] for item in batch]),
        "seq_lengths": torch.tensor([item["seq_lengths"] for item in batch]),
        "edges": [item["edges"] for item in batch]
    }
    if masked_edges:
        collated["masked_edges"] = [item["masked_edges"] for item in batch]
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


def generalized_link_pred_loss(
    predictor: LinkPredictor,
    node_embedding,
    edge_index_list,
    masked_edge_index_list=None,
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
            masked_edge_index = masked_edge_index_list[i].to(device)
            if masked_edge_index.size(1) == 0:
                continue
            # set positive edges as witheld (masked) edges
            pos_edge_index = masked_edge_index
        elif mask_locs is not None:
            masked_nodes = torch.where(mask_locs[i])[0].to(device)
            if masked_nodes.numel() == 0:
                continue
            masked_nodes_bool = torch.zeros(max_seq_length, dtype=torch.bool, device=device)
            masked_nodes_bool[masked_nodes] = True
            src_nodes, dst_nodes = edge_index[0], edge_index[1]
            edge_mask = (~masked_nodes_bool[src_nodes]) & (~masked_nodes_bool[dst_nodes])
            pos_edge_index = edge_index[:, edge_mask]
        else:
            pos_edge_index = edge_index
        
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
        # src_emb_pos, dst_emb_pos = node_embedding[i, pos_edge_index[0]], node_embedding[i, pos_edge_index[1]]
        # pos_scores = predictor(src_emb_pos, dst_emb_pos)
        # FIXME: must provide the operative edge used in embedding as edge_idnex when doing MGM
        pos_scores = predictor(node_embedding[i], edge_index=edge_index, edge_pairs=pos_edge_index)
        pos_preds.append(pos_scores)
        pos_labels.append(torch.ones_like(pos_scores, device=device))
        
        # Negative scores
        # src_emb_neg, dst_emb_neg = node_embedding[i, neg_edge_index[0]], node_embedding[i, neg_edge_index[1]]
        # neg_scores = predictor(src_emb_neg, dst_emb_neg)
        neg_scores = predictor(node_embedding[i], edge_index=edge_index, edge_pairs=neg_edge_index)
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


class FineTuneModule(pl.LightningModule):
    def __init__(self, model, task, lr=1e-3, weight_decay=1e-4):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.model.to(self.device)
        self.task = task
    
    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("train_loss", loss, batch_size=batch["x"].size(0), prog_bar=True, on_epoch=True, on_step=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("val_loss", loss, batch_size=batch["x"].size(0), prog_bar=True, on_epoch=True, on_step=False)

    def _step(self, batch):
        if self.task == "link":
            loss, _, _ = generalized_link_pred_loss(
                predictor=self.model,
                node_embedding=batch["x"].to(self.device),
                seq_lengths=batch["seq_lengths"],
                edge_index_list=batch["edges"],
                masked_edge_index_list=None,
                use_bce_loss=False
            )
        elif self.task == "mgm":
            loss, _, _ = generalized_link_pred_loss(
                predictor=self.model,
                node_embedding=batch["x"].to(self.device), 
                seq_lengths=batch["seq_lengths"],
                edge_index_list=batch["edges"],
                masked_edge_index_list=batch["masked_edges"],
                use_bce_loss=True
            )
            pass
        return loss

    def configure_optimizers(self):
        return {
            "optimizer": torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay),
            "frequency": 1
        }

def fine_tune_pl(ft_model, train_dataloader, task, save_dir, val_dataloader=None, lr=1e-3, num_epochs=100, max_num_batches=200):
    
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss", 
        mode="min",
        verbose=True,
        strict=True,
        patience=10
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss", 
        mode="min",
        save_top_k=3,
        verbose=True,
        every_n_epochs=20, # FIXME
        save_last=True,
        filename="{epoch}-{step}-{val_loss:.4f}",
        dirpath=save_dir
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = "last-{epoch}-{step}-{val_loss:.4f}"

    trainer = pl.Trainer(
        max_epochs=num_epochs, 
        limit_train_batches=max_num_batches,
        callbacks=[
            early_stop_callback,
            checkpoint_callback
        ],
        check_val_every_n_epoch=1,
        accumulate_grad_batches=1
    )
    model = FineTuneModule(
        model=ft_model, 
        task=task,
        lr=lr
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
    # perm = np.random.permutation(num_unique)
    masked_pairs = [unique_pairs[i.item()] for i in perm[:num_mask]]
    masked_indices = []
    for pair in masked_pairs:
        masked_indices.extend(pairs[pair]) 
    masked_indices = torch.tensor(masked_indices, device=device, dtype=torch.long)
    
    masked_edge_index = edge_index[:, masked_indices]

    return masked_edge_index


def predict_links(model, dataloader, task, max_num_batches=100):
    model.eval().to("cuda")
    
    all_preds = []
    all_labels = []
    n_b = 0
    for batch in tqdm.tqdm(dataloader, leave=False):
        batch = send_to_gpu(batch)
        if task == "link":
            loss, preds, labels = generalized_link_pred_loss(
                predictor=model,
                node_embedding=batch["x"].to(device), 
                mask_locs=None,
                seq_lengths=batch["seq_lengths"],
                edge_index_list=batch["edges"],
                use_bce_loss=False
            )
        elif task == "mgm":
            loss, preds, labels = generalized_link_pred_loss(
                predictor=model,
                node_embedding=batch["x"].to(device), 
                seq_lengths=batch["seq_lengths"],
                edge_index_list=batch["edges"],
                masked_edge_index_list=batch["masked_edges"],
                use_bce_loss=True
            )
        all_preds.extend(preds.cpu().detach().numpy())
        all_labels.extend(labels.cpu().detach().numpy())
        
        n_b += 1
        if n_b >= max_num_batches:
            break
    
    # AUROC
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    auc_score = auc(fpr, tpr)
    
    # PR
    p, r, _ = precision_recall_curve(all_labels, all_preds)
    apr = average_precision_score(all_labels, all_preds)
    
    return fpr, tpr, auc_score, p, r, apr



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


def main(args):
    print("Loading dataset...")
    if args.model == "scglm":
        embedding_path_format = "/hpc/mydata/rowan.cassius/data/scGPT/human_immune/cell_type/{}/embeddings/scglm/embedding.npz"
        # embedding_path_format = "/hpc/mydata/rowan.cassius/data/scGPT/human_immune/cell_type/{}/embeddings/scglm/aracne_4096_masked/embedding.npz"
    elif args.model == "scgpt":
        embedding_path_format = "/hpc/mydata/rowan.cassius/data/scGPT/human_immune/cell_type/{}/embeddings/scgpt/embedding.npz"
    elif args.model == "scf":
        embedding_path_format = "/hpc/mydata/rowan.cassius/data/scGPT/human_immune/cell_type/{}/embeddings/scfoundation/aracne_4096/embedding.npz"

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
    if args.task == "link":
        train_dataset = GeneEmbeddingDataset(paths=train_paths)
        val_dataset = GeneEmbeddingDataset(paths=val_paths)
        collate_fn = embedding_collate_fn
    elif args.task == "mgm":
        train_dataset = GeneEmbeddingDatasetWithMasks(paths=train_paths)
        val_dataset = GeneEmbeddingDatasetWithMasks(paths=val_paths)
        collate_fn = partial(embedding_collate_fn, masked_edges=True)
    
    batch_size=8

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=collate_fn
    )   

    predictor = LinkPredictor(
        in_channels=train_dataset.embedding_dim,
        hidden_dim=train_dataset.embedding_dim,
        embed_dim=train_dataset.embedding_dim,
        use_gat=args.use_gat
    )

    print("Fine tuning link predictor...")
    best_model: FineTuneModule = fine_tune_pl(
        ft_model=predictor,
        task=args.task,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        lr=1e-5,
        max_num_batches=8,
        num_epochs=200, #FIXME
        save_dir=args.model_save_dir
    )
    best_predictor = best_model.model

    print("Making Inference with link predictor...")

    fpr_train, tpr_train, auc_score_train, p_train, r_train, apr_train = predict_links(
        best_predictor, train_dataloader, task=args.task, max_num_batches=200
    )

    # Free up memory by deleting datasets and dataloaders
    del train_dataset, val_dataset, train_dataloader, val_dataloader
    gc.collect()  # Run garbage collection to release Python memory

    # If using GPU, empty CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    

    print("Loading Test Data")
    if args.task == "link":
        test_dataset = GeneEmbeddingDataset(paths=test_paths)
    elif args.task == "mgm":
        test_dataset = GeneEmbeddingDatasetWithMasks(paths=test_paths)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    fpr_test, tpr_test, auc_score_test, p_test, r_test, apr_test = predict_links(
        best_predictor, test_dataloader,task=args.task, max_num_batches=200
    )

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
        fpr_test=fpr_test,
        tpr_test=tpr_test,
        auc_score_test=auc_score_test,
        p_test=p_test,
        r_test=r_test,
        apr_test=apr_test,
        fpr_train=fpr_train,
        tpr_train=tpr_train,
        auc_score_train=auc_score_train,
        p_train=p_train,
        r_train=r_train,
        apr_train=apr_train
    )

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--suffix", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=["scglm", "scgpt", "scf"])
    parser.add_argument("--task", type=str, required=True, choices=["link", "mgm"])
    parser.add_argument("--use_gat", action="store_true")
    args = parser.parse_args()
    
    args.res_dir = join(args.out_dir, f"{args.suffix}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}")
    args.model_save_dir = join(args.res_dir, "model")
    os.makedirs(args.res_dir, exist_ok=False)
    main(args)
