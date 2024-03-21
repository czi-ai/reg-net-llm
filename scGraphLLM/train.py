# Main location of training
import torch
from config.model_config import GNNConfig
from GNN_modules import GCN_attn, attention_sparsity, graph_smoothness


"""
Multi-GPU training will be tricky for this model, since we'll need to maintain the output of the graph encoder. So for now, lets just get something 
running on a single GPU, in vanilla pytorch, and then worry about multi-GPU training/ lightning later on
"""
class TrainerConfig:
    lr: float = 0.001
    weight_decay: float = 1e-4
    max_grad: float = 20.0
    num_epochs: int = 100
    alpha: float = 0.001 # graph smoothness regularization weight
    beta: float = 0.001 # attention weight sparsity regularization weight
    lambda_l1: float = 0.001
    lambda_l2: float = 0.001
    verbose: bool = True
    train_loss: list = []
    val_loss: list = []


class GNN_Trainer:
    def __init__(self, train_loader, val_loader, model_config=GNNConfig(), train_config=TrainerConfig()):
        self.model_config = model_config
        self.train_config = train_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GCN_attn(self.model_config.input_dim, self.model_config.hidden_dims, 
                              self.model_config.conv_dim, self.model_config.out_dim, self.model_config.num_nodes).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.train_config.lr, 
                                          weight_decay=self.train_config.weight_decay)
        self.train_data = train_loader
        self.val_data = val_loader
    
    def train_one_batch(self, train_batch):
        self.model.train()
        train_batch = train_batch.to(self.device)
        h,h_conv, graphs, w = self.model(train_batch.x, train_batch.edge_index, train_batch.edge_attr, train_batch.batch)
        graph_labels = train_batch.y
        gs_reg = self.train_config.alpha * graph_smoothness(h_conv, graphs)
        loss = torch.nn.CrossEntropyLoss()(h, graph_labels)
        l1_reg = sum(torch.norm(param, p=1) for param in self.model.parameters())
        loss += self.train_config.lambda_l1 * l1_reg
        l2_reg = sum(torch.norm(param, p=2) ** 2 for param in self.model.parameters())
        loss += self.train_config.lambda_l2 * l2_reg
        loss += gs_reg
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                       max_norm=self.train_config.max_grad)
        self.optimizer.step()
        return loss
    
    def val_one_batch(self, val_batch):
        self.model.eval()
        val_batch = val_batch.to(self.device)
        h, h_conv, graphs, w = self.model(val_batch.x, val_batch.edge_index, val_batch.edge_attr, val_batch.batch)
        graph_labels = val_batch.y
        gs_reg = self.train_config.alpha * graph_smoothness(h_conv, graphs)
        loss = torch.nn.CrossEntropyLoss()(h, graph_labels)
        l1_reg = sum(torch.norm(param, p=1) for param in self.model.parameters())
        loss += self.train_config.lambda_l1 * l1_reg
        l2_reg = sum(torch.norm(param, p=2) ** 2 for param in self.model.parameters())
        loss += self.train_config.lambda_l2 * l2_reg
        loss += gs_reg
        return loss, graphs, h_conv, w
    
    def train_loop(self):
        all_graphs, all_h, all_attn = [], [], []
        train_loss, val_loss = [], []
        for epoch in range(self.train_config.num_epochs):
            batch_loss, batch_val = [], []
            for train_batch in self.train_data:
                train_batch_loss = self.train_one_batch(train_batch)
                batch_loss.append(train_batch_loss / len(self.train_data))

            for val_batch in self.val_data:
                val_batch_loss, graphs, hs, ws = self.val_one_batch(val_batch)
                batch_val.append(val_batch_loss / len(self.val_data))
                all_graphs.append(graphs)
                all_h.append(hs)
                all_attn.append(ws)

            train_loss.append(torch.mean(torch.tensor(batch_loss)).item())
            val_loss.append(torch.mean(torch.tensor(batch_val)).item())

            if self.train_config.verbose:
                print("[Epoch %04d]  Overall Loss: %.5f" % (epoch + 1, torch.mean(torch.tensor(batch_loss)).item()))
                print("[Epoch %04d]  Overall val: %.5f" % (epoch + 1, torch.mean(torch.tensor(batch_val)).item()))
        return self.model, all_graphs, all_h, all_attn, train_loss, val_loss


def extract_gene_embeddings():
    pass

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
                
