import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

from .model_utils import build_mlp

class MessagePassingLayer(MessagePassing):
    def __init__(self, hidden_dim: int, dropout: float = 0.0, activation: str = 'silu'):
        super().__init__(aggr='add') # "Aggregation: sum"
        
        self.message_mlp = build_mlp(
            input_dim=hidden_dim * 2 + hidden_dim, # h_i, h_j, e_ij
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            n_layers=2,
            activation=activation,
            dropout=dropout
        )
        
        self.update_mlp = build_mlp(
            input_dim=hidden_dim * 2, # h_i, agg_messages
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            n_layers=2,
            activation=activation,
            dropout=dropout
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        # Propagate messages
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        # Update node features
        update_input = torch.cat([x, out], dim=-1)
        update = self.update_mlp(update_input)
        
        # Residual connection and layer norm
        x_new = x + update
        x_new = self.layer_norm(x_new)
        
        return x_new, edge_attr

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor):
        # x_i: target node features, x_j: source node features
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.message_mlp(msg_input)


class BaselineGNN(nn.Module):
    def __init__(
        self,
        node_input_dim: int = 4,      # position(3) + bead_index(1)
        edge_input_dim: int = 5,      # displacement(3) + distance(1) + bond_type(1)
        hidden_dim: int = 128,
        n_layers: int = 4,
        dropout: float = 0.0,
        activation: str = 'silu',
    ):
        super().__init__()
        
        self.node_encoder = build_mlp(
            input_dim=node_input_dim, 
            hidden_dim=hidden_dim, 
            output_dim=hidden_dim, 
            n_layers=2, 
            activation=activation, 
            dropout=dropout
        )
        
        self.edge_encoder = build_mlp(
            input_dim=edge_input_dim, 
            hidden_dim=hidden_dim, 
            output_dim=hidden_dim, 
            n_layers=2, 
            activation=activation, 
            dropout=dropout
        )
        
        self.mp_layers = nn.ModuleList([
            MessagePassingLayer(hidden_dim=hidden_dim, dropout=dropout, activation=activation)
            for _ in range(n_layers)
        ])
        
        self.decoder = build_mlp(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim // 2,
            output_dim=3,
            n_layers=2,
            activation=activation,
            dropout=0.0
        )
        
    def forward(self, data) -> torch.Tensor:
        """Forward pass. data is a PyG Data/Batch object.
        Returns displacement predictions, shape (N_total, 3)."""
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        
        for layer in self.mp_layers:
            x, edge_attr = layer(x, edge_index, edge_attr)
            
        out = self.decoder(x)
        return out
