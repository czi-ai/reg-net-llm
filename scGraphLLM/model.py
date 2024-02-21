# For the model class
import torch 
class LitSCGraphLLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.graph_encoder = ...
        self.transformer_encoder = ...
    def graph_encoder_forward(self, x):
        return self.graph_encoder(x)
    def transformer_encoder_forward(self, x):
        return self.transformer_encoder(x)
