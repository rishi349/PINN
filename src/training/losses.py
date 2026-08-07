import torch
import torch.nn.functional as F

def displacement_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE loss on predicted vs actual displacement.
    pred: (N, 3), target: (N, 3)
    Returns scalar tensor."""
    return F.mse_loss(pred, target)

def bond_length_penalty(
    positions: torch.Tensor,
    predicted_displacement: torch.Tensor, 
    bonds: torch.Tensor,
    r0: float = 1.0,
    k_penalty: float = 1.0,
) -> torch.Tensor:
    """Placeholder for physics-informed bond loss (Month 4+).
    Penalizes predicted positions that violate bond length constraints."""
    return torch.tensor(0.0, device=positions.device, requires_grad=True)

def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    physics_loss: torch.Tensor = None,
    physics_weight: float = 0.0,
) -> torch.Tensor:
    """Combined data + physics loss."""
    mse = displacement_mse_loss(pred, target)
    if physics_loss is not None and physics_weight > 0.0:
        return mse + physics_weight * physics_loss
    return mse
