"""Loss functions for polymer GNN training (§8.5, §10.2).

Three loss components:
  1. displacement_mse_loss  — data loss, the primary objective
  2. bond_length_penalty    — physics loss: penalise bond-length violations
  3. excluded_volume_penalty — physics loss: penalise bead-bead overlaps
  4. combined_loss          — weighted sum for physics-informed training

For the baseline model (Month 5-6):  physics_weight = 0.0  → pure MSE
For the physics-informed model (Month 7): physics_weight > 0.0

The bond loss and EV loss are OFF by default (weight=0) so the baseline
and physics-informed models are trained under identical conditions except
for the weight — satisfying the "fair comparison" requirement from §7.3.
"""

import torch
import torch.nn.functional as F


def displacement_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE loss on predicted vs actual per-bead displacement.

    Parameters
    ----------
    pred   : (N, dim) or (batch*N, dim) — predicted displacement
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

    Penalises predicted positions whose bond lengths deviate from r0.

    L_bond = (1/(N-1)) * Σ_{bonds (i,j)} (|r_i_new - r_j_new| - r0)²

    where r_new = positions + predicted_displacement.

    Parameters
    ----------
    positions : torch.Tensor, shape (N, dim)
        Current bead positions.
    predicted_displacement : torch.Tensor, shape (N, dim)
        Model-predicted displacement.
    bonds : torch.Tensor, shape (2, n_bonds) or (n_bonds, 2)
        Bond edge list (bead index pairs).
    r0 : float
        Equilibrium bond length (default 1.0 σ; FENE+WCA gives ~0.965σ
        but we penalise deviations from the nominal r0 used in the contract).
    k_penalty : float
        Scale factor for the penalty (tuned alongside physics_weight).

    Returns
    -------
    Scalar tensor — bond-length penalty loss.
    """
    # Predicted next positions
    new_positions = positions + predicted_displacement  # (N, dim)

    # Normalise bond tensor to shape (n_bonds, 2)
    if bonds.shape[0] == 2:
        bonds = bonds.t()  # (n_bonds, 2)

    i_idx = bonds[:, 0]
    j_idx = bonds[:, 1]

    r_i = new_positions[i_idx]  # (n_bonds, dim)
    r_j = new_positions[j_idx]  # (n_bonds, dim)

    bond_lengths = torch.norm(r_i - r_j, dim=-1)         # (n_bonds,)
    deviations = (bond_lengths - r0) ** 2                 # (n_bonds,)
    return k_penalty * deviations.mean()


def excluded_volume_penalty(
    positions: torch.Tensor,
    predicted_displacement: torch.Tensor,
    sigma: float = 1.0,
    k_penalty: float = 1.0,
) -> torch.Tensor:
    """Excluded-volume penalty: penalise non-bonded bead overlaps (§8.5).

    Penalises pairs |i-j| > 1 that are closer than 2^(1/6)*sigma
    (the WCA cutoff). Uses a soft repulsion: penalty ∝ (r_cut - r)²
    for r < r_cut.

    Parameters
    ----------
    positions : torch.Tensor, shape (N, dim)
    predicted_displacement : torch.Tensor, shape (N, dim)
    sigma : float
        WCA length scale (default 1.0 σ).
    k_penalty : float
        Scale factor.

    Returns
    -------
    Scalar tensor.
    """
    new_positions = positions + predicted_displacement  # (N, dim)
    N = new_positions.shape[0]

    if N < 3:
        return torch.tensor(0.0, device=positions.device, requires_grad=True)

    r_cut = sigma * (2.0 ** (1.0 / 6.0))

    # All pairs i < j with |i-j| > 1  (non-bonded)
    # Build indices
    idx_i, idx_j = torch.triu_indices(N, N, offset=2, device=positions.device)

    if idx_i.numel() == 0:
        return torch.tensor(0.0, device=positions.device, requires_grad=True)

    r_ij = new_positions[idx_j] - new_positions[idx_i]   # (n_pairs, dim)
    dist = torch.norm(r_ij, dim=-1)                        # (n_pairs,)

    # Penalty only for pairs inside the cutoff
    overlap = torch.clamp(r_cut - dist, min=0.0)          # (n_pairs,)
    return k_penalty * (overlap ** 2).mean()


def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    physics_loss: torch.Tensor = None,
    physics_weight: float = 0.0,
) -> torch.Tensor:
    """Combined data + physics loss (§8.5).

    For baseline (Month 5–6):         physics_weight = 0.0
    For physics-informed (Month 7):   physics_weight > 0.0

    Parameters
    ----------
    pred           : predicted displacement
    target         : ground-truth displacement
    physics_loss   : pre-computed physics penalty (bond + EV or combined)
    physics_weight : scalar weight λ on the physics term

    Returns
    -------
    Scalar tensor: MSE + λ * physics_loss
    """
    mse = displacement_mse_loss(pred, target)
    if physics_loss is not None and physics_weight > 0.0:
        return mse + physics_weight * physics_loss
    return mse

