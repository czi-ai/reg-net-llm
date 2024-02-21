# Main location of training
import torch

"""
Multi-GPU training will be tricky for this model, since we'll need to maintain the output of the graph encoder. So for now, lets just get something 
running on a single GPU, in vanilla pytorch, and then worry about multi-GPU training/ lightning later on
"""


def extract_gene_embeddings():

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def train_loop(self, model, graph_dataset, transformer_dataset):
        # Initialize the model
        model = model.to(self.device)
        # Initialize the optimizer
        optimizer = self.config.optim.optimizer(model.parameters(), lr=self.config.optim.lr)
        # Initialize the loss function
        loss_fn = self.config.training.loss_fn
        # Initialize the data loader
        graph_loader = torch.utils.data.DataLoader(graph_dataset, batch_size=self.config.batch_size, shuffle=True)
        transformer_loader = torch.utils.data.DataLoader(transformer_dataset, batch_size=self.config.batch_size, shuffle=True)
        # Initialize the training loop
        for epoch in range(self.config.epochs):
            graph_nodes = []
            graph_edges = []
            for batch in graph_loader:
                batch = batch.to(self.device)
                updated_nodes, updated_edges = model.graph_encoder_forward(batch)
                
