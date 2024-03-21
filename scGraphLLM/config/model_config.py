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



mlm_config = Config({
    "d_model": 128, 
    "nhead" :4, 
    "dim_feedforward" : 512, 
    "dropout":0.1 ,
    "activation":"gelu",
    "batch_first":True
})

basegcn_config = Config({ 
    "input_dim": 64,
    "hidden_dims": (64,),
    "conv_dim": 64,
    "out_dim": 64,
})

base_model_config = Config({
    "gnn_config" : basegcn_config,
    "mlm_config" : mlm_config,
    "rank_embedding_size":500, ## arbitary rn 
    "rank_embedding_dim": 64,
    "node_embedding_size": 5000, ## arbitary rn
    "node_embedding_dim": 64
})

class GNNConfig:
    num_nodes: int = 5000
    input_dim: int = 2000
    hidden_dims: tuple = (128,)
    conv_dim: int = 50
    out_dim: int = 20
    latent_dim: int = 64
    num_graphs: int = 6
        