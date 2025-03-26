import torch
import torch.nn as nn
from _globals import *
from copy import deepcopy
import models as models
#import linear_model as linear_model

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


node_hyperparams = Config({
    "num_ranks": NUM_RANKS, 
    "num_genes": NUM_GENES, 
    "node_embedding_dim": 128,
    "num_hvgs": 1024,
    "freeze_encoder": False
})

transformer_dim = Config({
    "input_dim": 2*node_hyperparams.node_embedding_dim,
    "feed_dim": 512
})

vanilla_config = Config({
    "transformer_dim": transformer_dim,
    "num_heads": 8,
    "num_encoder_layers": 1,
    "activation": "gelu",
    "dropout": 0.1,
    "batch_first": True,
    "use_pe": False,
    "use_attn_mask": False,
    "use_flash_attn": True
})

vanilla_3L_config = Config({
    "transformer_dim": transformer_dim,
    "num_heads": 8,
    "num_encoder_layers": 3,
    "activation": "gelu",
    "dropout": 0.1,
    "batch_first": True,
    "use_pe": False,
    "use_attn_mask": False,
    "use_flash_attn": True
})

vanilla_6L_config = Config({
    "transformer_dim": transformer_dim,
    "num_heads": 8,
    "num_encoder_layers": 6,
    "activation": "gelu",
    "dropout": 0.1,
    "batch_first": True,
    "use_pe": False,
    "use_attn_mask": False,
    "use_flash_attn": True
})

graph_kernel_attn_config = Config({
    "transformer_dim": transformer_dim,
    "num_heads": 8,
    "num_encoder_layers": 12,
    "activation": "gelu",
    "dropout": 0.1,
    "batch_first": True,
    "use_pe": False,
    "use_attn_mask": True,
    "use_flash_attn": True,
    "fine_tuning": False
})

graph_kernel_attn_3L_config = Config({
    "transformer_dim": transformer_dim,
    "num_heads": 8,
    "num_encoder_layers": 3,
    "activation": "gelu",
    "dropout": 0.5,
    "batch_first": True,
    "use_pe": False,
    "use_attn_mask": True,
    "use_flash_attn": True
})

graph_kernel_attn_6L_config = Config({
    "transformer_dim": transformer_dim,
    "num_heads": 8,
    "num_encoder_layers": 6,
    "activation": "gelu",
    "dropout": 0.1,
    "batch_first": True,
    "use_pe": False,
    "use_attn_mask": True,
    "use_flash_attn": True
})

vanilla_manitou = Config({
    "model": models.GDTransformer, # models.LitScGraphLLM
    "model_config": node_hyperparams, # prediction_head_config
    "transformer_config": vanilla_config,
    "data_config":Config({
        "train": Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/single_cell/pilot_cache_4096/train", "dataset_name": "train"}),  # NOTE: bc we are reading from disk each time, we need to cache in /pmglocal
        "val": [
            Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/single_cell/pilot_cache_4096/valSG", "dataset_name":"valSG"}),
            Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/single_cell/pilot_cache_4096/valHOG", "dataset_name":"valHOG"})
            ],
        "test": [
            ],
        "run_test":False, 
        "num_workers": 1,
        "batch_size": 16
    }),
    "trainer_config":Config({
        "max_epochs" : 100,
        "accelerator" : "gpu",
        "max_time": "03:00:00:00",
        "devices" : 1,
        "precision":"bf16",
        "num_sanity_val_steps" : 0,
        "log_every_n_steps" : 1,
        "val_check_interval":0.1,
        "checkpoint_config": {
                                "save_top_k": -1, # Save all checkpoints
                                "every_n_train_steps" : 5000 # Save every 5000 training steps
                            }
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

vanilla_1024_base = Config({
    "model": models.GDTransformer, # models.LitScGraphLLM
    "model_config": node_hyperparams, # prediction_head_config
    "transformer_config": vanilla_config,
    "data_config":Config({
        "train": Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/cxg_cache_1024_base/train", "dataset_name": "train"}),  # NOTE: bc we are reading from disk each time, we need to cache in /pmglocal
        "val": [
            Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/cxg_cache_1024_base/valSG", "dataset_name":"valSG"}),
            Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/cxg_cache_1024_base/valHOG", "dataset_name":"valHOG"})
            ],
        "test": [
            ],
        "run_test":False, 
        "num_workers": 1,
        "batch_size": 16
    }),
    "trainer_config":Config({
        "max_epochs" : 100,
        "accelerator" : "gpu",
        "max_time": "01:00:00:00",
        "devices" : 1,
        "precision":"bf16",
        "num_sanity_val_steps" : 0,
        "log_every_n_steps" : 1,
        "val_check_interval":0.05,
        "checkpoint_config": {
                                "save_top_k": -1, # Save all checkpoints
                                "every_n_train_steps" : 5000 # Save every 5000 training steps
                            }
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

vanilla_1024 = Config({
    "model": models.GDTransformer, # models.LitScGraphLLM
    "model_config": node_hyperparams, # prediction_head_config
    "transformer_config": vanilla_config,
    "data_config":Config({
        "train": Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/cxg_cache_1024/train", "dataset_name": "train"}),  # NOTE: bc we are reading from disk each time, we need to cache in /pmglocal
        "val": [
            Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/cxg_cache_1024/valSG", "dataset_name":"valSG"}),
            Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/cxg_cache_1024/valHOG", "dataset_name":"valHOG"})
            ],
        "test": [
            ],
        "run_test":False, 
        "num_workers": 1,
        "batch_size": 16
    }),
    "trainer_config":Config({
        "max_epochs" : 100,
        "accelerator" : "gpu",
        "max_time": "01:00:00:00",
        "devices" : 1,
        "precision":"bf16",
        "num_sanity_val_steps" : 0,
        "log_every_n_steps" : 1,
        "val_check_interval":0.05,
        "checkpoint_config": {
                                "save_top_k": -1, # Save all checkpoints
                                "every_n_train_steps" : 5000 # Save every 5000 training steps
                            }
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

vanilla_2048 = Config({
    "model": models.GDTransformer, # models.LitScGraphLLM
    "model_config": node_hyperparams, # prediction_head_config
    "transformer_config": vanilla_config,
    "data_config":Config({
        "train": Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/cxg_cache_2048/train", "dataset_name": "train"}),  # NOTE: bc we are reading from disk each time, we need to cache in /pmglocal
        "val": [
            Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/cxg_cache_2048/valSG", "dataset_name":"valSG"}),
            Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/cxg_cache_2048/valHOG", "dataset_name":"valHOG"})
            ],
        "test": [
            ],
        "run_test":False, 
        "num_workers": 1,
        "batch_size": 16
    }),
    "trainer_config":Config({
        "max_epochs" : 100,
        "accelerator" : "gpu",
        "max_time": "01:00:00:00",
        "devices" : 1,
        "precision":"bf16",
        "num_sanity_val_steps" : 0,
        "log_every_n_steps" : 1,
        "val_check_interval":0.05,
        "checkpoint_config": {
                                "save_top_k": -1, # Save all checkpoints
                                "every_n_train_steps" : 5000 # Save every 5000 training steps
                            }
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

vanilla_4096 = Config({
    "model": models.GDTransformer, # models.LitScGraphLLM
    "model_config": node_hyperparams, # prediction_head_config
    "transformer_config": vanilla_config,
    "data_config":Config({
        "train": Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/cxg_cache_4096/train", "dataset_name": "train"}),  # NOTE: bc we are reading from disk each time, we need to cache in /pmglocal
        "val": [
            Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/cxg_cache_4096/valSG", "dataset_name":"valSG"}),
            Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/cxg_cache_4096/valHOG", "dataset_name":"valHOG"})
            ],
        "test": [
            ],
        "run_test":False, 
        "num_workers": 1,
        "batch_size": 16
    }),
    "trainer_config":Config({
        "max_epochs" : 100,
        "accelerator" : "gpu",
        "max_time": "01:00:00:00",
        "devices" : 1,
        "precision":"bf16",
        "num_sanity_val_steps" : 0,
        "log_every_n_steps" : 1,
        "val_check_interval":0.05,
        "checkpoint_config": {
                                "save_top_k": -1, # Save all checkpoints
                                "every_n_train_steps" : 5000 # Save every 5000 training steps
                            }
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

vanilla_3L_4096 = Config({
    "model": models.GDTransformer, # models.LitScGraphLLM
    "model_config": node_hyperparams, # prediction_head_config
    "transformer_config": vanilla_3L_config,
    "data_config":Config({
        "train": Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/cxg_cache_4096/train", "dataset_name": "train"}),  # NOTE: bc we are reading from disk each time, we need to cache in /pmglocal
        "val": [
            Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/cxg_cache_4096/valSG", "dataset_name":"valSG"}),
            Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/cxg_cache_4096/valHOG", "dataset_name":"valHOG"})
            ],
        "test": [
            ],
        "run_test":False, 
        "num_workers": 1,
        "batch_size": 8
    }),
    "trainer_config":Config({
        "max_epochs" : 100,
        "accelerator" : "gpu",
        "max_time": "01:00:00:00",
        "devices" : 1,
        "precision":"bf16",
        "num_sanity_val_steps" : 0,
        "log_every_n_steps" : 1,
        "val_check_interval":0.05,
        "checkpoint_config": {
                                "save_top_k": -1, # Save all checkpoints
                                "every_n_train_steps" : 5000 # Save every 5000 training steps
                            }
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

vanilla_6L_4096 = Config({
    "model": models.GDTransformer, # models.LitScGraphLLM
    "model_config": node_hyperparams, # prediction_head_config
    "transformer_config": vanilla_6L_config,
    "data_config":Config({
        "train": Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/cxg_cache_4096/train", "dataset_name": "train"}),  # NOTE: bc we are reading from disk each time, we need to cache in /pmglocal
        "val": [
            Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/cxg_cache_4096/valSG", "dataset_name":"valSG"}),
            Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/cxg_cache_4096/valHOG", "dataset_name":"valHOG"})
            ],
        "test": [
            ],
        "run_test":False, 
        "num_workers": 1,
        "batch_size": 8
    }),
    "trainer_config":Config({
        "max_epochs" : 100,
        "accelerator" : "gpu",
        "max_time": "01:00:00:00",
        "devices" : 1,
        "precision":"bf16",
        "num_sanity_val_steps" : 0,
        "log_every_n_steps" : 1,
        "val_check_interval":0.05,
        "checkpoint_config": {
                                "save_top_k": -1, # Save all checkpoints
                                "every_n_train_steps" : 5000 # Save every 5000 training steps
                            }
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

graph_kernel_attn_manitou = Config({
    "model": models.GDTransformer, # models.LitScGraphLLM,#
    "model_config": node_hyperparams,
    "transformer_config": graph_kernel_attn_config,
    "data_config":Config({
        "train": Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/pilot_manitou/train", "dataset_name": "train"}),  # NOTE: bc we are reading from disk each time, we need to cache in /pmglocal
        "val": [
            Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/pilot_manitou/valSG", "dataset_name":"valSG"}),
            Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/pilot_manitou/valHOG", "dataset_name":"valHOG"})
            ],
        "test": [
            ],
        "run_test":False, 
        "num_workers": 8,
        "batch_size": 8
    }),
    "trainer_config":Config({
        "max_epochs" : 10,
        "accelerator" : "gpu",
        "max_time": "05:00:00:00",
        "devices" : 8,
        "precision":"bf16",
        "num_sanity_val_steps" : 0,
        "log_every_n_steps" : 1,
        "val_check_interval":0.1,
        "checkpoint_config": {
                                "save_top_k": -1, # Save all checkpoints
                                "every_n_train_steps" : 5000 # Save every 5000 training steps
                            }
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

graph_kernel_attn_1024_base = Config({
    "model": models.GDTransformer, # models.LitScGraphLLM,#
    "model_config": node_hyperparams,
    "transformer_config": graph_kernel_attn_config,
    "data_config":Config({
        "train": Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/cxg_cache_1024_base/train", "dataset_name": "train"}),  # NOTE: bc we are reading from disk each time, we need to cache in /pmglocal
        "val": [
            Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/cxg_cache_1024_base/valSG", "dataset_name":"valSG"}),
            Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/cxg_cache_1024_base/valHOG", "dataset_name":"valHOG"})
            ],
        "test": [
            ],
        "run_test":False, 
        "num_workers": 1,
        "batch_size": 16
    }),
    "trainer_config":Config({
        "max_epochs" : 100,
        "accelerator" : "gpu",
        "max_time": "01:00:00:00",
        "devices" : 1,
        "precision":"bf16",
        "num_sanity_val_steps" : 0,
        "log_every_n_steps" : 1,
        "val_check_interval":0.05,
        "checkpoint_config": {
                                "save_top_k": -1, # Save all checkpoints
                                "every_n_train_steps" : 5000 # Save every 5000 training steps
                            }
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

graph_kernel_attn_1024 = Config({
    "model": models.GDTransformer, # models.LitScGraphLLM,#
    "model_config": node_hyperparams,
    "transformer_config": graph_kernel_attn_config,
    "data_config":Config({
        "train": Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/cxg_cache_1024/train", "dataset_name": "train"}),  # NOTE: bc we are reading from disk each time, we need to cache in /pmglocal
        "val": [
            Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/cxg_cache_1024/valSG", "dataset_name":"valSG"}),
            Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/cxg_cache_1024/valHOG", "dataset_name":"valHOG"})
            ],
        "test": [
            ],
        "run_test":False, 
        "num_workers": 1,
        "batch_size": 16
    }),
    "trainer_config":Config({
        "max_epochs" : 100,
        "accelerator" : "gpu",
        "max_time": "01:00:00:00",
        "devices" : 1,
        "precision":"bf16",
        "num_sanity_val_steps" : 0,
        "log_every_n_steps" : 1,
        "val_check_interval":0.05,
        "checkpoint_config": {
                                "save_top_k": -1, # Save all checkpoints
                                "every_n_train_steps" : 5000 # Save every 5000 training steps
                            }
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

graph_kernel_attn_2048 = Config({
    "model": models.GDTransformer, # models.LitScGraphLLM,#
    "model_config": node_hyperparams,
    "transformer_config": graph_kernel_attn_config,
    "data_config":Config({
        "train": Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/cxg_cache_2048/train", "dataset_name": "train"}),  # NOTE: bc we are reading from disk each time, we need to cache in /pmglocal
        "val": [
            Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/cxg_cache_2048/valSG", "dataset_name":"valSG"}),
            Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/cxg_cache_2048/valHOG", "dataset_name":"valHOG"})
            ],
        "test": [
            ],
        "run_test":False, 
        "num_workers": 1,
        "batch_size": 16
    }),
    "trainer_config":Config({
        "max_epochs" : 100,
        "accelerator" : "gpu",
        "max_time": "01:00:00:00",
        "devices" : 1,
        "precision":"bf16",
        "num_sanity_val_steps" : 0,
        "log_every_n_steps" : 1,
        "val_check_interval":0.05,
        "checkpoint_config": {
                                "save_top_k": -1, # Save all checkpoints
                                "every_n_train_steps" : 5000 # Save every 5000 training steps
                            }
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

graph_kernel_attn_4096 = Config({
    "model": models.GDTransformer, # models.LitScGraphLLM,#
    "model_config": node_hyperparams,
    "transformer_config": graph_kernel_attn_config,
    "data_config":Config({
        "train": Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/cxg_cache_4096_META/train", "dataset_name": "train"}),  # NOTE: bc we are reading from disk each time, we need to cache in /pmglocal
        "val": [
            Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/cxg_cache_4096_META/valSG", "dataset_name":"valSG"}),
            Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/cxg_cache_4096_META/valHOG", "dataset_name":"valHOG"})
            ],
        "test": [
            ],
        "run_test":False, 
        "num_workers": 8,
        "batch_size": 8
    }),
    "trainer_config":Config({
        "max_epochs" : 10,
        "accelerator" : "gpu",
        "max_time": "05:00:00:00",
        "devices" : 8,
        "precision":"bf16",
        "num_sanity_val_steps" : 0,
        "log_every_n_steps" : 1,
        "val_check_interval":0.05,
        "checkpoint_config": {
                                "save_top_k": -1, # Save all checkpoints
                                "every_n_train_steps" : 1000 # Save every 5000 training steps
                            }
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

graph_kernel_attn_3L_4096 = Config({
    "model": models.GDTransformer, # models.LitScGraphLLM,#
    "model_config": node_hyperparams,
    "transformer_config": graph_kernel_attn_3L_config,
    "data_config":Config({
        "train": Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/cxg_cache_4096/train", "dataset_name": "train"}),  # NOTE: bc we are reading from disk each time, we need to cache in /pmglocal
        "val": [
            Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/cxg_cache_4096/valSG", "dataset_name":"valSG"}),
            Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/cxg_cache_4096/valHOG", "dataset_name":"valHOG"})
            ],
        "test": [
            ],
        "run_test":False, 
        "num_workers": 1,
        "batch_size": 8
    }),
    "trainer_config":Config({
        "max_epochs" : 100,
        "accelerator" : "gpu",
        "max_time": "05:00:00:00",
        "devices" : 1,
        "precision":"bf16",
        "num_sanity_val_steps" : 0,
        "log_every_n_steps" : 1,
        "val_check_interval":0.01,
        "checkpoint_config": {
                                "save_top_k": -1, # Save all checkpoints
                                "every_n_train_steps" : 5000 # Save every 5000 training steps
                            }
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

graph_kernel_attn_6L_4096 = Config({
    "model": models.GDTransformer, # models.LitScGraphLLM,#
    "model_config": node_hyperparams,
    "transformer_config": graph_kernel_attn_6L_config,
    "data_config":Config({
        "train": Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/cxg_cache_4096/train", "dataset_name": "train"}),  # NOTE: bc we are reading from disk each time, we need to cache in /pmglocal
        "val": [
            Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/cxg_cache_4096/valSG", "dataset_name":"valSG"}),
            Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/cxg_cache_4096/valHOG", "dataset_name":"valHOG"})
            ],
        "test": [
            ],
        "run_test":False, 
        "num_workers": 1,
        "batch_size": 8
    }),
    "trainer_config":Config({
        "max_epochs" : 100,
        "accelerator" : "gpu",
        "max_time": "01:00:00:00",
        "devices" : 1,
        "precision":"bf16",
        "num_sanity_val_steps" : 0,
        "log_every_n_steps" : 1,
        "val_check_interval":0.05,
        "checkpoint_config": {
                                "save_top_k": -1, # Save all checkpoints
                                "every_n_train_steps" : 5000 # Save every 5000 training steps
                            }
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

# # prc_gs_1024 = deepcopy(vanilla)
# # prc_gs_1024["data_config"] = dataconfig_1024

# #prc_gs_2048 = deepcopy(vanilla)
# #prc_gs_2048["data_config"] = dataconfig_2048

# #prc_gs_4096 = deepcopy(vanilla)
# #prc_gs_4096["data_config"] = dataconfig_4096