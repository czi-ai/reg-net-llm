import os 
from os.path import join, abspath, dirname
import sys
import gc
from argparse import ArgumentParser
from datetime import datetime

from torch_geometric.utils import negative_sampling
import lightning.pytorch as pl

from scGraphLLM.data import *
from scGraphLLM.GNN_modules import *
from scGraphLLM.MLP_modules import *
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


class FineTuneModule(pl.LightningModule):
    def __init__(self, model, lr=1e-3, weight_decay=1e-4):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.model.to(self.device)

    def training_step(self, batch, batch_idx):
        loss, _, _ = link_pred_loss(
            predictor=self.model,
            node_embedding=batch["x"].to(self.device), 
            mask_locs=None,
            seq_lengths=batch["seq_lengths"],
            edge_index_list=batch["edges"]
        )
        self.log("train_loss", loss, batch_size=batch["x"].size(0), prog_bar=True, on_epoch=True, on_step=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, _, _ = link_pred_loss(
            predictor=self.model,
            node_embedding=batch["x"].to(self.device), 
            mask_locs=None,
            seq_lengths=batch["seq_lengths"],
            edge_index_list=batch["edges"]
        )
        self.log("val_loss", loss, batch_size=batch["x"].size(0), prog_bar=True, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        return {
            "optimizer": torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay),
            "frequency": 1
        }

def fine_tune_pl(ft_model, train_dataloader, save_dir, val_dataloader=None, lr=1e-3, num_epochs=100, max_num_batches=200):
    
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
        every_n_epochs=10,
        save_last=True,
        filename="{epoch}-{step}-{val_loss_stage_1:.4f}",
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
        best_model = FineTuneModule.load_from_checkpoint(best_model_path, model=ft_model, lr=lr)
    else:
        print("No best model checkpoint found. Returning the original model.")
        best_model = ft_model

    return best_model

def predict_links(model, dataloader, max_num_batches=100):
    model.eval().to("cuda")
    
    all_preds = []
    all_labels = []
    n_b = 0
    for batch in tqdm.tqdm(dataloader, leave=False):
        batch = send_to_gpu(batch)
        loss, preds, labels = link_pred_loss(
            predictor=model,
            node_embedding=batch["x"].to(device), 
            mask_locs=None,
            seq_lengths=batch["seq_lengths"],
            edge_index_list=batch["edges"]
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
    train_dataset = GeneEmbeddingDataset( 
        paths=[embedding_path_format.format(cell_type) for cell_type in train_cell_types]
    )
    val_dataset = GeneEmbeddingDataset( 
        paths=[embedding_path_format.format(cell_type) for cell_type in val_cell_types]
    )
    
    batch_size=8

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=embedding_collate_fn
    )  

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=embedding_collate_fn
    )   

    link_predictor = LinkPredictHead(
        embed_dim=train_dataset.embedding_dim, 
        output_dim=1
    ).to(device)

    print("Fine tuning link predictor...")
    best_model: FineTuneModule = fine_tune_pl(
        ft_model=link_predictor,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        lr=1e-5,
        max_num_batches=8,
        num_epochs=200,
        save_dir=args.model_save_dir
    )
    best_link_predictor = best_model.model

    print("Making Inference with link predictor...")

    fpr_train, tpr_train, auc_score_train, p_train, r_train, apr_train = predict_links(
        best_link_predictor, train_dataloader, max_num_batches=200
    )


    # Free up memory by deleting datasets and dataloaders
    del train_dataset, val_dataset, train_dataloader, val_dataloader
    gc.collect()  # Run garbage collection to release Python memory

    # If using GPU, empty CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    

    print("Loading Test Data")
    test_dataset = GeneEmbeddingDataset( 
        paths=[embedding_path_format.format(cell_type) for cell_type in test_cell_types]
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=embedding_collate_fn
    )
    
    fpr_test, tpr_test, auc_score_test, p_test, r_test, apr_test = predict_links(
        best_link_predictor, test_dataloader, max_num_batches=200
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
    args = parser.parse_args()
    
    args.res_dir = join(args.out_dir, f"{args.suffix}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}")
    args.model_save_dir = join(args.res_dir, "model")
    os.makedirs(args.res_dir, exist_ok=False)
    main(args)
