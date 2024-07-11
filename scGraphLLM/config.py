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




base_gnn_config = Config({
    "input_dim":  256,
    "hidden_dims": [256, 256, 256],
    "conv_dim": 256, 
    "num_heads": [2, 2, 2],
    "out_dim": 256
})

base_model_config = Config({
    "gnn_config" : base_gnn_config,
    "num_ranks":NUM_RANKS, ## arbitary rn, but theoretically should be the cell with most genes 
    "num_genes": NUM_GENES, 
    "node_embedding_dim": 128
})


pilot_run_config = Config({
    "model":models.LitScGraphLLM,
    "model_config": base_model_config,
    "data_config":Config({
        "train": Config({"cache_dir":"/pmglocal/scGraphLLM/modeldata/pilotdata_cache/pilotdata_train_cache", "dataset_name": "train"}),  # NOTE: bc we are reading from disk each time, we need to cache in /pmglocal
        "val": [
            Config({"cache_dir":"/pmglocal/scGraphLLM/modeldata/pilotdata_cache/pilotdata_train_cache", "dataset_name":"valSG"}),
            Config({"cache_dir":"/pmglocal/scGraphLLM/modeldata/pilotdata_cache/pilotdata_train_cache", "dataset_name":"valHOG"})
            ],
        "test": [
            ],
        "run_test":False, 
        "num_workers": 8,
        "batch_size": 40
    }),
    "trainer_config":Config({
        "max_epochs" : 100,
        "accelerator" : "gpu",
        "devices" : 1,
        "precision":"bf16",
        "num_sanity_val_steps" : 0,
        "log_every_n_steps" : 1,
        "val_check_interval":0.25

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

        