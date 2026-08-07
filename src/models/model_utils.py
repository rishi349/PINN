import torch
import torch.nn as nn

def get_activation(name: str) -> nn.Module:
    """Get activation function by name. Supports: silu, relu, gelu, tanh."""
    name = name.lower()
    if name == 'silu':
        return nn.SiLU()
    elif name == 'relu':
        return nn.ReLU()
    elif name == 'gelu':
        return nn.GELU()
    elif name == 'tanh':
        return nn.Tanh()
    else:
        raise ValueError(f"Unsupported activation function: {name}")

def build_mlp(
    input_dim: int, 
    hidden_dim: int, 
    output_dim: int, 
    n_layers: int = 2, 
    activation: str = 'silu', 
    dropout: float = 0.0
) -> nn.Sequential:
    """Build a multi-layer perceptron."""
    layers = []
    
    if n_layers == 1:
        layers.append(nn.Linear(input_dim, output_dim))
        return nn.Sequential(*layers)
        
    act_fn = get_activation(activation)
    
    # First layer
    layers.append(nn.Linear(input_dim, hidden_dim))
    layers.append(act_fn)
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
        
    # Hidden layers
    for _ in range(n_layers - 2):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(act_fn)
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
            
    # Final layer
    layers.append(nn.Linear(hidden_dim, output_dim))
    
    return nn.Sequential(*layers)

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
