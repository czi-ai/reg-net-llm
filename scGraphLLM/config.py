import models as models
import flash_transformer as flash_transformer
import torch
import torch.nn as nn
from _globals import *
from copy import deepcopy

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

base_transformer_config = Config({
    "input_dim":  2*base_model_config.node_embedding_dim,
    "feed_dim": 256,
    "hidden_dims": [256, 256, 256],
    "conv_dim": 256, 
    "num_heads": 2,
    "out_dim": base_gnn_config.out_dim,
    "activation": "gelu",
    "dropout": 0.1,
    "batch_first": True,
    "use_pe": False,
    "use_attn_mask": False,
    "use_flash_attn": True
})


graph_transformer_config = Config({
    "input_dim":  2*base_model_config.node_embedding_dim,
    "feed_dim": 256,
    "hidden_dims": [256, 256, 256],
    "conv_dim": 256, 
    "num_heads": 2,
    "out_dim": base_gnn_config.out_dim,
    "activation": "gelu",
    "dropout": 0.1,
    "batch_first": True,
    "use_pe": True,
    "use_attn_mask": False,
    "use_flash_attn": True
})

pilot_run_config = Config({
    "model": flash_transformer.FlashTRAN, # models.LitScGraphLLM,#
    "model_config": base_model_config,
    "transformer_config": base_transformer_config,
    "data_config":Config({
        "train": Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/pilot_manitou/train", "dataset_name": "train"}),  # NOTE: bc we are reading from disk each time, we need to cache in /pmglocal
        "val": [
            Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/pilot_manitou/valSG", "dataset_name":"valSG"}),
            Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/pilot_manitou/valHOG", "dataset_name":"valHOG"})
            ],
        "test": [
            ],
        "run_test":False, 
        "num_workers": 1,
        "batch_size": 32
    }),
    "trainer_config":Config({
        "max_epochs" : 100,
        "accelerator" : "gpu",
        "max_time": "01:00:00:00",
        "devices" : 1,
        "precision":"bf16",
        "num_sanity_val_steps" : 0,
        "log_every_n_steps" : 1,
        "val_check_interval":0.1

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
    "wandb_user":"mingxuan-zhang",
    "wandb_project":"scGraphLLM",
})

pilot_graph_pe_run_config = Config({
    "model": flash_transformer.GraphTransformer, # models.LitScGraphLLM,#
    "model_config": base_model_config,
    "transformer_config": graph_transformer_config,
    "data_config":Config({
        "train": Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/pilot_manitou/train", "dataset_name": "train"}),  # NOTE: bc we are reading from disk each time, we need to cache in /pmglocal
        "val": [
            Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/pilot_manitou/valSG", "dataset_name":"valSG"}),
            Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/pilot_manitou/valHOG", "dataset_name":"valHOG"})
            ],
        "test": [
            ],
        "run_test":False, 
        "num_workers": 1,
        "batch_size": 32
    }),
    "trainer_config":Config({
        "max_epochs" : 100,
        "accelerator" : "gpu",
        "max_time": "01:00:00:00",
        "devices" : 1,
        "precision":"bf16",
        "num_sanity_val_steps" : 0,
        "log_every_n_steps" : 1,
        "val_check_interval":0.1

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
    "wandb_user":"mingxuan-zhang",
    "wandb_project":"scGraphLLM",
})







dataconfig_1024 =  Config({
        "train": Config({"cache_dir":"/pmglocal/vss2134/scGraphLLM/modeldata/pilotdata_cache/pilotdata_1024/train", "dataset_name": "train"}),  # NOTE: bc we are reading from disk each time, we need to cache in /pmglocal
        "val": [
            Config({"cache_dir":"/pmglocal/vss2134/scGraphLLM/modeldata/pilotdata_cache/pilotdata_1024/valSG", "dataset_name":"valSG"}),
            Config({"cache_dir":"/pmglocal/vss2134/scGraphLLM/modeldata/pilotdata_cache/pilotdata_1024/valHG", "dataset_name":"valHOG"})
            ],
        "test": [
            ],
        "run_test":False, 
        "num_workers": 8,
        "batch_size": 64
    })

dataconfig_2048 =  Config({
        "train": Config({"cache_dir":"/pmglocal/vss2134/scGraphLLM/modeldata/pilotdata_cache/pilotdata_2048/train", "dataset_name": "train"}),  # NOTE: bc we are reading from disk each time, we need to cache in /pmglocal
        "val": [
            Config({"cache_dir":"/pmglocal/vss2134/scGraphLLM/modeldata/pilotdata_cache/pilotdata_2048/valSG", "dataset_name":"valSG"}),
            Config({"cache_dir":"/pmglocal/vss2134/scGraphLLM/modeldata/pilotdata_cache/pilotdata_2048/valSG", "dataset_name":"valHOG"})
            ],
        "test": [
            ],
        "run_test":False, 
        "num_workers": 8,
        "batch_size": 64
    })

dataconfig_4096 =  Config({
        "train": Config({"cache_dir":"/pmglocal/vss2134/scGraphLLM/modeldata/pilotdata_cache/pilotdata_4096/train", "dataset_name": "train"}),  # NOTE: bc we are reading from disk each time, we need to cache in /pmglocal
        "val": [
            Config({"cache_dir":"/pmglocal/vss2134/scGraphLLM/modeldata/pilotdata_cache/pilotdata_4096/valSG", "dataset_name":"valSG"}),
            Config({"cache_dir":"/pmglocal/vss2134/scGraphLLM/modeldata/pilotdata_cache/pilotdata_4096/valSG", "dataset_name":"valHOG"})
            ],
        "test": [
            ],
        "run_test":False, 
        "num_workers": 8,
        "batch_size": 64
    })

prc_gs_1024 = deepcopy(pilot_run_config)
prc_gs_1024["data_config"] = dataconfig_1024

prc_gs_2048 = deepcopy(pilot_run_config)
prc_gs_2048["data_config"] = dataconfig_2048

prc_gs_4096 = deepcopy(pilot_run_config)
prc_gs_4096["data_config"] = dataconfig_4096