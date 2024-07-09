import models as models
import torch 
from _globals import *

class Config(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")
    def __setattr__(self, key, value):
        self[key] = value
    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")




gnn_config = Config({
    "input_dim": 128,
    "hidden_dims": [128, 128, 128],
    "conv_dim": 256, 
    "num_heads": [2, 2, 2],
    "out_dim": 128
})

base_model_config = Config({
    "gnn_config" : gnn_config,
    "num_ranks":NUM_RANKS, ## arbitary rn, but theoretically should be the cell with most genes 
    "num_genes": NUM_GENES, 
    "node_embedding_dim": 64
})


full_run_config = Config({
    "model":models.LitScGraphLLM,
    "model_config": base_model_config,
    "data_config":Config({
        "train": Config({"cache_dir":"/burg/pmg/collab/scGraphLLM/data/pilotdata_train_cache/", "dataset_name": "train"}),  # NOTE: bc we are reading from disk each time, we need to cache in /pmglocal
        "val": [
            Config({"cache_dir":"/burg/pmg/collab/scGraphLLM/data/pilotdata_valSG_cache/", "dataset_name":"valSG"}),
            Config({"cache_dir":"/burg/pmg/collab/scGraphLLM/data/pilotdata_valHOG_cache/", "dataset_name":"valHOG"})
            ],
        "test": [
            ],
        "run_test":False, 
        "num_workers": 4,
        "batch_size": 16
    }),
    "trainer_config":Config({
        "max_epochs" : 100,
        "accelerator" : "gpu",
        "devices" : 1,
        "precision":"bf16",
        "num_sanity_val_steps" : 0,
        "log_every_n_steps" : 1
    }),
    "loss_config":Config({
        "loss":"mlm"
    }),
    "optim_config":Config({
        "optimizer": torch.optim.Adam,
        "args":{
            "lr": 0.0001,
            "betas": [0.9, 0.999]
         }
    }),
    "repo_name":"scGraphLLM",
    "wandb_user":"aqlab",
    "wandb_project":"scGraphLLM",
})

        