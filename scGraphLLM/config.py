import torch
import torch.nn as nn
from copy import deepcopy

#import linear_model as linear_model

import scGraphLLM.models as models
from scGraphLLM._globals import *

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

      
"""
-----------------------------
Node-specific hyperparameters
-----------------------------
"""
node_hyperparams = Config({
    "num_expression_bins": NUM_EXPRESSION_BINS, 
    "num_genes": NUM_GENES, 
    "node_embedding_dim": 256,
    "num_hvgs": 4096,
    "freeze_encoder": False
})


"""
------------------------------------
Transformer-specific hyperparameters
------------------------------------
"""
transformer_dim = Config({
    "input_dim": 2*node_hyperparams.node_embedding_dim,
    "feed_dim": 512
})


"""
--------------------------
Transformer configurations
--------------------------
"""
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

graph_kernel_attn_1DIFF_config = Config({
    "transformer_dim": transformer_dim,
    "num_heads": 8,
    "num_encoder_layers": 12,
    "activation": "gelu",
    "dropout": 0.1,
    "batch_first": True,
    "use_pe": False,
    "use_attn_mask": True,
    "use_flash_attn": [0],
    "fine_tuning": False
})

graph_kernel_attn_2DIFF_A_config = Config({
    "transformer_dim": transformer_dim,
    "num_heads": 8,
    "num_encoder_layers": 12,
    "activation": "gelu",
    "dropout": 0.1,
    "batch_first": True,
    "use_pe": False,
    "use_attn_mask": True,
    "use_flash_attn": [0, 1],
    "fine_tuning": False
})

graph_kernel_attn_2DIFF_B_config = Config({
    "transformer_dim": transformer_dim,
    "num_heads": 8,
    "num_encoder_layers": 12,
    "activation": "gelu",
    "dropout": 0.1,
    "batch_first": True,
    "use_pe": False,
    "use_attn_mask": True,
    "use_flash_attn": [0, 6],
    "fine_tuning": False
})

graph_kernel_attn_3L_config = Config({
    "transformer_dim": transformer_dim,
    "num_heads": 8,
    "num_encoder_layers": 3,
    "activation": "gelu",
    "dropout": 0.1,
    "batch_first": True,
    "use_pe": False,
    "use_attn_mask": True,
    "use_flash_attn": True,
    "fine_tuning": False
})

graph_kernel_attn_3L_1DIFF_config = Config({
    "transformer_dim": transformer_dim,
    "num_heads": 8,
    "num_encoder_layers": 3,
    "activation": "gelu",
    "dropout": 0.1,
    "batch_first": True,
    "use_pe": False,
    "use_attn_mask": True,
    "use_flash_attn": [0],
    "fine_tuning": False
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
    "use_flash_attn": True,
    "fine_tuning": False
})

graph_kernel_attn_6L_3DIFF_config = Config({
    "transformer_dim": transformer_dim,
    "num_heads": 8,
    "num_encoder_layers": 6,
    "activation": "gelu",
    "dropout": 0.1,
    "batch_first": True,
    "use_pe": False,
    "use_attn_mask": True,
    "use_flash_attn": [0, 1, 2],
    "fine_tuning": False
})


"""
--------------------------------
General model-run configurations
--------------------------------
"""
graph_kernel_attn_4096_TEST = Config({
    "model": models.GDTransformer,
    "model_config": node_hyperparams,
    "transformer_config": graph_kernel_attn_config,
    "data_config":Config({
        "train": Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/meta_cell/cxg_TEST/train", "dataset_name": "train"}),
        "val": [
            Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/meta_cell/cxg_TEST/valSG", "dataset_name":"valSG"}),
            Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/meta_cell/cxg_TEST/valHOG", "dataset_name":"valHOG"})
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
        "max_time": "07:00:00:00",
        "devices" : 1,
        "precision":"bf16",
        "num_sanity_val_steps" : 0,
        "log_every_n_steps" : 50,
        "val_check_interval": 0.05,
        "num_nodes":  1,
        "strategy" :"ddp", # "ddp_find_unused_parameters_true",
        "checkpoint_config": {
                                "save_top_k": -1, # Save all checkpoints
                                "every_n_train_steps" : 1000 # Save every 1000 training steps
                            }
    }),
    "loss_config":Config({
        "loss":"mlm"
    }),
    "optim_config":Config({
        "optimizer": torch.optim.AdamW,
        "args":{
            "lr": 0.00005,
            "betas": [0.9, 0.999],
            "weight_decay": 0.0001,
         }
    }),
    "repo_name":"scGraphLLM",
    "wandb_user":"aqlab",
    "wandb_project":"scGraphLLM",
})


graph_kernel_attn_4096 = Config({
    "model": models.GDTransformer,
    "model_config": node_hyperparams,
    "transformer_config": graph_kernel_attn_config,
    "data_config":Config({
        "train": Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/meta_cell/cxg_4096/train", "dataset_name": "train"}),
        "val": [
            Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/meta_cell/cxg_4096/valSG", "dataset_name":"valSG"}),
            Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/meta_cell/cxg_4096/valHOG", "dataset_name":"valHOG"})
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
        "max_time": "07:00:00:00",
        "devices" : 2,
        "precision":"bf16",
        "num_sanity_val_steps" : 0,
        "log_every_n_steps" : 50,
        "val_check_interval": 0.05,
        "num_nodes":  1,
        "strategy" :"ddp", # "ddp_find_unused_parameters_true",
        "checkpoint_config": {
                                "save_top_k": -1, # Save all checkpoints
                                "every_n_train_steps" : 1000 # Save every 1000 training steps
                            }
    }),
    "loss_config":Config({
        "loss":"mlm"
    }),
    "optim_config":Config({
        "optimizer": torch.optim.AdamW,
        "args":{
            "lr": 0.00005,
            "betas": [0.9, 0.999],
            "weight_decay": 0.0001,
         }
    }),
    "repo_name":"scGraphLLM",
    "wandb_user":"aqlab",
    "wandb_project":"scGraphLLM",
})


graph_kernel_attn_3L_4096_sc = Config({
    "model": models.GDTransformer,
    "model_config": node_hyperparams,
    "transformer_config": graph_kernel_attn_config,
    "data_config":Config({
        "train": Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/single_cell/cxg_4096_NEW/train", "dataset_name": "train"}),
        "val": [
            Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/single_cell/cxg_4096_NEW/valSG", "dataset_name":"valSG"}),
            Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/single_cell/cxg_4096_NEW/valHOG", "dataset_name":"valHOG"})
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
        "max_time": "07:00:00:00",
        "devices" : 2,
        "precision":"bf16",
        "num_sanity_val_steps" : 0,
        "log_every_n_steps" : 50,
        "val_check_interval": 0.05,
        "num_nodes":  1,
        "strategy" :"ddp", # "ddp_find_unused_parameters_true",
        "checkpoint_config": {
                                "save_top_k": -1, # Save all checkpoints
                                "every_n_train_steps" : 1000 # Save every 1000 training steps
                            }
    }),
    "loss_config":Config({
        "loss":"mlm"
    }),
    "optim_config":Config({
        "optimizer": torch.optim.AdamW,
        "args":{
            "lr": 0.00005,
            "betas": [0.9, 0.999],
            "weight_decay": 0.0001,
         }
    }),
    "repo_name":"scGraphLLM",
    "wandb_user":"aqlab",
    "wandb_project":"scGraphLLM",
})


graph_kernel_attn_1DIFF_4096 = Config({
    "model": models.GDTransformer,
    "model_config": node_hyperparams,
    "transformer_config": graph_kernel_attn_1DIFF_config,
    "data_config":Config({
        "train": Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/meta_cell/cxg_4096/train", "dataset_name": "train"}),
        "val": [
            Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/meta_cell/cxg_4096/valSG", "dataset_name":"valSG"}),
            Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/meta_cell/cxg_4096/valHOG", "dataset_name":"valHOG"})
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
        "max_time": "07:00:00:00",
        "devices" : 2,
        "precision":"bf16",
        "num_sanity_val_steps" : 0,
        "log_every_n_steps" : 50,
        "val_check_interval": 0.05,
        "num_nodes":  1,
        "strategy" :"ddp", # "ddp_find_unused_parameters_true",
        "checkpoint_config": {
                                "save_top_k": -1, # Save all checkpoints
                                "every_n_train_steps" : 1000 # Save every 1000 training steps
                            }
    }),
    "loss_config":Config({
        "loss":"mlm"
    }),
    "optim_config":Config({
        "optimizer": torch.optim.AdamW,
        "args":{
            "lr": 0.00005,
            "betas": [0.9, 0.999],
            "weight_decay": 0.0001,
         }
    }),
    "repo_name":"scGraphLLM",
    "wandb_user":"aqlab",
    "wandb_project":"scGraphLLM",
})


graph_kernel_attn_2DIFF_A_4096 = Config({
    "model": models.GDTransformer,
    "model_config": node_hyperparams,
    "transformer_config": graph_kernel_attn_2DIFF_A_config,
    "data_config":Config({
        "train": Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/meta_cell/cxg_4096/train", "dataset_name": "train"}),
        "val": [
            Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/meta_cell/cxg_4096/valSG", "dataset_name":"valSG"}),
            Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/meta_cell/cxg_4096/valHOG", "dataset_name":"valHOG"})
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
        "max_time": "07:00:00:00",
        "devices" : 2,
        "precision":"bf16",
        "num_sanity_val_steps" : 0,
        "log_every_n_steps" : 50,
        "val_check_interval": 0.05,
        "num_nodes":  1,
        "strategy" :"ddp", # "ddp_find_unused_parameters_true",
        "checkpoint_config": {
                                "save_top_k": -1, # Save all checkpoints
                                "every_n_train_steps" : 1000 # Save every 1000 training steps
                            }
    }),
    "loss_config":Config({
        "loss":"mlm"
    }),
    "optim_config":Config({
        "optimizer": torch.optim.AdamW,
        "args":{
            "lr": 0.00005,
            "betas": [0.9, 0.999],
            "weight_decay": 0.0001,
         }
    }),
    "repo_name":"scGraphLLM",
    "wandb_user":"aqlab",
    "wandb_project":"scGraphLLM",
})


graph_kernel_attn_2DIFF_B_4096 = Config({
    "model": models.GDTransformer,
    "model_config": node_hyperparams,
    "transformer_config": graph_kernel_attn_2DIFF_B_config,
    "data_config":Config({
        "train": Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/meta_cell/cxg_4096/train", "dataset_name": "train"}),
        "val": [
            Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/meta_cell/cxg_4096/valSG", "dataset_name":"valSG"}),
            Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/meta_cell/cxg_4096/valHOG", "dataset_name":"valHOG"})
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
        "max_time": "07:00:00:00",
        "devices" : 2,
        "precision":"bf16",
        "num_sanity_val_steps" : 0,
        "log_every_n_steps" : 50,
        "val_check_interval": 0.05,
        "num_nodes":  1,
        "strategy" :"ddp", # "ddp_find_unused_parameters_true",
        "checkpoint_config": {
                                "save_top_k": -1, # Save all checkpoints
                                "every_n_train_steps" : 1000 # Save every 1000 training steps
                            }
    }),
    "loss_config":Config({
        "loss":"mlm"
    }),
    "optim_config":Config({
        "optimizer": torch.optim.AdamW,
        "args":{
            "lr": 0.00005,
            "betas": [0.9, 0.999],
            "weight_decay": 0.0001,
         }
    }),
    "repo_name":"scGraphLLM",
    "wandb_user":"aqlab",
    "wandb_project":"scGraphLLM",
})


graph_kernel_attn_3L_4096 = Config({
    "model": models.GDTransformer,
    "model_config": node_hyperparams,
    "transformer_config": graph_kernel_attn_3L_config,
    "data_config":Config({
        "train": Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/meta_cell/cxg_4096/train", "dataset_name": "train"}),
        "val": [
            Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/meta_cell/cxg_4096/valSG", "dataset_name":"valSG"}),
            Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/meta_cell/cxg_4096/valHOG", "dataset_name":"valHOG"})
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
        "max_time": "07:00:00:00",
        "devices" : 8,
        "precision":"bf16",
        "num_sanity_val_steps" : 0,
        "log_every_n_steps" : 50,
        "val_check_interval": 0.05,
        "num_nodes":  1,
        "strategy" :"ddp", # "ddp_find_unused_parameters_true",
        "checkpoint_config": {
                                "save_top_k": -1, # Save all checkpoints
                                "every_n_train_steps" : 1000 # Save every 1000 training steps
                            }
    }),
    "loss_config":Config({
        "loss":"mlm"
    }),
    "optim_config":Config({
        "optimizer": torch.optim.AdamW,
        "args":{
            "lr": 0.00005,
            "betas": [0.9, 0.999],
            "weight_decay": 0.0001,
         }
    }),
    "repo_name":"scGraphLLM",
    "wandb_user":"aqlab",
    "wandb_project":"scGraphLLM",
})

graph_kernel_attn_3L_1DIFF_4096 = Config({
    "model": models.GDTransformer,
    "model_config": node_hyperparams,
    "transformer_config": graph_kernel_attn_3L_1DIFF_config,
    "data_config":Config({
        "train": Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/meta_cell/cxg_4096/train", "dataset_name": "train"}),
        "val": [
            Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/meta_cell/cxg_4096/valSG", "dataset_name":"valSG"}),
            Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/meta_cell/cxg_4096/valHOG", "dataset_name":"valHOG"})
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
        "max_time": "07:00:00:00",
        "devices" : 8,
        "precision":"bf16",
        "num_sanity_val_steps" : 0,
        "log_every_n_steps" : 50,
        "val_check_interval": 0.05,
        "num_nodes":  1,
        "strategy" :"ddp", # "ddp_find_unused_parameters_true",
        "checkpoint_config": {
                                "save_top_k": -1, # Save all checkpoints
                                "every_n_train_steps" : 1000 # Save every 1000 training steps
                            }
    }),
    "loss_config":Config({
        "loss":"mlm"
    }),
    "optim_config":Config({
        "optimizer": torch.optim.AdamW,
        "args":{
            "lr": 0.00005,
            "betas": [0.9, 0.999],
            "weight_decay": 0.0001,
         }
    }),
    "repo_name":"scGraphLLM",
    "wandb_user":"aqlab",
    "wandb_project":"scGraphLLM",
})


graph_kernel_attn_6L_4096 = Config({
    "model": models.GDTransformer,
    "model_config": node_hyperparams,
    "transformer_config": graph_kernel_attn_6L_config,
    "data_config":Config({
        "train": Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/meta_cell/cxg_4096/train", "dataset_name": "train"}),
        "val": [
            Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/meta_cell/cxg_4096/valSG", "dataset_name":"valSG"}),
            Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/meta_cell/cxg_4096/valHOG", "dataset_name":"valHOG"})
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
        "max_time": "07:00:00:00",
        "devices" : 2,
        "precision":"bf16",
        "num_sanity_val_steps" : 0,
        "log_every_n_steps" : 50,
        "val_check_interval": 0.05,
        "num_nodes":  1,
        "strategy" :"ddp", # "ddp_find_unused_parameters_true",
        "checkpoint_config": {
                                "save_top_k": -1, # Save all checkpoints
                                "every_n_train_steps" : 1000 # Save every 1000 training steps
                            }
    }),
    "loss_config":Config({
        "loss":"mlm"
    }),
    "optim_config":Config({
        "optimizer": torch.optim.AdamW,
        "args":{
            "lr": 0.00005,
            "betas": [0.9, 0.999],
            "weight_decay": 0.0001,
         }
    }),
    "repo_name":"scGraphLLM",
    "wandb_user":"aqlab",
    "wandb_project":"scGraphLLM",
})


graph_kernel_attn_6L_3DIFF_4096 = Config({
    "model": models.GDTransformer,
    "model_config": node_hyperparams,
    "transformer_config": graph_kernel_attn_6L_3DIFF_config,
    "data_config":Config({
        "train": Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/meta_cell/cxg_4096/train", "dataset_name": "train"}),
        "val": [
            Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/meta_cell/cxg_4096/valSG", "dataset_name":"valSG"}),
            Config({"cache_dir":"/hpc/projects/group.califano/GLM/data/meta_cell/cxg_4096/valHOG", "dataset_name":"valHOG"})
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
        "max_time": "07:00:00:00",
        "devices" : 2,
        "precision":"bf16",
        "num_sanity_val_steps" : 0,
        "log_every_n_steps" : 50,
        "val_check_interval": 0.05,
        "num_nodes":  1,
        "strategy" :"ddp", # "ddp_find_unused_parameters_true",
        "checkpoint_config": {
                                "save_top_k": -1, # Save all checkpoints
                                "every_n_train_steps" : 1000 # Save every 1000 training steps
                            }
    }),
    "loss_config":Config({
        "loss":"mlm"
    }),
    "optim_config":Config({
        "optimizer": torch.optim.AdamW,
        "args":{
            "lr": 0.00005,
            "betas": [0.9, 0.999],
            "weight_decay": 0.0001,
         }
    }),
    "repo_name":"scGraphLLM",
    "wandb_user":"aqlab",
    "wandb_project":"scGraphLLM",
})
