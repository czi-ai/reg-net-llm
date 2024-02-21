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

base_model_config = Config({
    "num_attention_heads": 4,
    "attention_probs_dropout_prob": 0.1,
    "hidden_act": "gelu_fast",
    "mlp_intermediate_size": 2048,
    "dim_encoder": 512,
    "num_layers_encoder": 4
})

class GNNConfig:
    num_nodes: int = 5000
    input_dim: int = 2000
    hidden_dims: tuple = (128,)
    conv_dim: int = 50
    out_dim: int = 20
    latent_dim: int = 64
    num_graphs: int = 6
        