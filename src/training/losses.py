"""Loss functions for polymer GNN training (§8.5).

data loss: displacement_mse_loss
physics loss: bond_length_penalty (under construction)
combined_loss for physics-informed training
"""
import torch
import torch.nn.functional as F


def displacement_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE loss on predicted vs actual per-bead displacement.

    Parameters
    ----------
    pred   : (N, dim) — predicted displacement
    target : same shape — ground-truth displacement

    Returns
    -------
    Scalar tensor.
    """
    return F.mse_loss(pred, target)


def bond_length_penalty(
    positions: torch.Tensor,
    predicted_displacement: torch.Tensor,
    bonds: torch.Tensor,
    r0: float = 1.0,
    k_penalty: float = 1.0,
) -> torch.Tensor:
    """Bond-length penalty for physics-informed training (§8.5).

    L_bond = (1/(N-1)) * Σ (|r_i_new - r_j_new| - r0)²
    """
    new_positions = positions + predicted_displacement
    if bonds.shape[0] == 2:
        bonds = bonds.t()
    i_idx = bonds[:, 0]
    j_idx = bonds[:, 1]
    r_i = new_positions[i_idx]
    r_j = new_positions[j_idx]
    bond_lengths = torch.norm(r_i - r_j, dim=-1)
    deviations = (bond_lengths - r0) ** 2
    return k_penalty * deviations.mean()


def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    physics_loss: torch.Tensor = None,
    physics_weight: float = 0.0,
) -> torch.Tensor:
    """Combined data + physics loss (§8.5)."""
    mse = displacement_mse_loss(pred, target)
    if physics_loss is not None and physics_weight > 0.0:
        return mse + physics_weight * physics_loss
    return mse
