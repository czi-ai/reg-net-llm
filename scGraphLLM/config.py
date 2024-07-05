import models as models
import torch 

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
    "conv_dim": 256, # this needs to be 2 * last layer dim
    "num_heads": [2, 2, 2, 1], #number of head per graph attention layer. Length = num_hidden_layers + 1
    "out_dim": 128
})

base_model_config = Config({
    "gnn_config" : gnn_config,
    "num_ranks":5002, ## arbitary rn, but theoretically should be the cell with most genes 
    "num_genes": 5003, 
    "node_embedding_dim": 64
})


full_run_config = Config({
    "model":models.LitScGraphLLM,
    "model_config": base_model_config,
    "data_config":Config({
        "train": Config({
            "aracne_outdirs":[
                                '/burg/pmg/users/rc3686/data/cellxgene/data/cell_type_005/dendritic_cell',
                                '/burg/pmg/users/rc3686/data/cellxgene/data/cell_type_005/cd4-positive_alpha-beta_t_cell',
                                '/burg/pmg/users/rc3686/data/cellxgene/data/cell_type_005/b_cell',
                                '/burg/pmg/users/rc3686/data/cellxgene/data/cell_type_005/glial_cell',
                                '/burg/pmg/users/rc3686/data/cellxgene/data/cell_type_005/mast_cell',
                                # '/burg/pmg/collab/scGraphLLM//data/samples/geneset_hvg/C164',
                                # '/burg/pmg/collab/scGraphLLM//data/samples/geneset_hvg/T_cac7',
                                # '/burg/pmg/collab/scGraphLLM//data/samples/geneset_hvg/SMC19',
                                # '/burg/pmg/collab/scGraphLLM//data/samples/geneset_hvg/C124',
                                # '/burg/pmg/collab/scGraphLLM//data/samples/geneset_hvg/T_cac15',
                                # '/burg/pmg/collab/scGraphLLM//data/samples/geneset_hvg/KUL19',
                                # '/burg/pmg/collab/scGraphLLM//data/samples/geneset_hvg/SMC17',
                                # '/burg/pmg/collab/scGraphLLM//data/samples/geneset_hvg/SMC15',
                                # '/burg/pmg/collab/scGraphLLM//data/samples/geneset_hvg/SMC07',
                                # '/burg/pmg/collab/scGraphLLM//data/samples/geneset_hvg/C130',
                                # '/burg/pmg/collab/scGraphLLM//data/samples/geneset_hvg/SMC22',
                                # '/burg/pmg/collab/scGraphLLM//data/samples/geneset_hvg/SMC20',
                                # '/burg/pmg/collab/scGraphLLM//data/samples/geneset_hvg/T_cac10',
                                # '/burg/pmg/collab/scGraphLLM//data/samples/geneset_hvg/KUL01'
                            ],
            "gene_to_node_file":"/burg/pmg/collab/scGraphLLM/data/cellxgene_gene2index.csv",
            "cache_dir":"/pmglocal/vss2134/scGraphLLM/data/modeldata/newgraphdata/", # NOTE: bc we are reading from disk each time, we need to cache in /pmglocal
        }),
        "val": Config({
            "aracne_outdirs":[
                                '/burg/pmg/collab/scGraphLLM//data/samples/geneset_hvg/SMC21',
                                '/burg/pmg/collab/scGraphLLM//data/samples/geneset_hvg/SMC09',
                                '/burg/pmg/collab/scGraphLLM//data/samples/geneset_hvg/KUL30',
                                '/burg/pmg/collab/scGraphLLM//data/samples/geneset_hvg/T_cac12'
                            ],
            "gene_to_node_file":"/burg/pmg/collab/scGraphLLM/data/example_gene2index.csv", 
            "cache_dir":"/pmglocal/vss2134/scGraphLLM/data/modeldata/newgraphdata/", 
        }),
        "test": Config({
            "aracne_outdirs":[
                                '/burg/pmg/collab/scGraphLLM//data/samples/geneset_hvg/T_cac6',
                                '/burg/pmg/collab/scGraphLLM//data/samples/geneset_hvg/SMC10'
                            ],
            "gene_to_node_file":"/burg/pmg/collab/scGraphLLM/data/example_gene2index.csv", 
            "cache_dir":"/pmglocal/vss2134/scGraphLLM/data/modeldata/newgraphdata/", 
        }),
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


# class GNNConfig:
#     num_nodes: int = 5000
#     input_dim: int = 2000
#     hidden_dims: tuple = (128,)
#     conv_dim: int = 50
#     out_dim: int = 20
#     latent_dim: int = 64
#     num_graphs: int = 6
        