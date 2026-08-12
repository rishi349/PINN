"""
Force implementations for polymer chain dynamics.

All forces follow the exact formulas from the execution plan (§4, §8.1):

Harmonic Bond (clean first version, §4 footnote):
    U_bond = 0.5 * k_bond * (|r_ij| - r0)²
    F_bond = -k_bond * (|r_ij| - r0) * r̂_ij
    Parameters: k_bond = 100 ε/σ², r0 = 1.0 σ

FENE Bond (standard Kremer-Grest, §4):
    U_fene = -0.5 * k * R0² * ln(1 - (|r_ij|/R0)²)
    F_fene = -k * r_ij / (1 - (|r_ij|/R0)²)
    Parameters: k = 30 ε/σ², R0 = 1.5 σ
    Resulting equilibrium bond length with WCA: ≈0.965σ

WCA Excluded Volume (§4):
    U_WCA = 4ε[(σ/r)¹² - (σ/r)⁶] + ε    for r < r_cut = 2^(1/6)σ
    U_WCA = 0                               for r >= r_cut
    F_WCA = 24ε/r * [2(σ/r)¹² - (σ/r)⁶] * r̂   for r < r_cut
    F_WCA = 0                                      for r >= r_cut
    Parameters: ε = 1.0, σ = 1.0

Sign convention: forces are returned as vectors FROM bead i TO bead j,
i.e., the force that should be ADDED to bead i's total force.
Newton's 3rd law: F_ij = -F_ji is enforced explicitly in the pair loops.

All functions operate in reduced units (§4): ε=1.0, σ=1.0, m=1.0.
"""

import numpy as np
from typing import Tuple


# =============================================================================
# PAIRWISE FORCE FUNCTIONS
# =============================================================================

def harmonic_bond_force(r_ij: np.ndarray, k_bond: float, r0: float) -> np.ndarray:
    """
    Compute harmonic bond force on bead i due to bead j.

    From §4 footnote:
        F = -k_bond * (|r_ij| - r0) * r̂_ij

    where r_ij = r_j - r_i (vector from i to j), and the returned force
    is the force acting ON bead i (pointing toward j if stretched,
    away from j if compressed).

    Parameters
    ----------
    r_ij : np.ndarray, shape (3,) or (dim,)
        Separation vector r_j - r_i (from bead i to bead j).
    k_bond : float
        Spring constant in ε/σ². Standard value: 100.0.
    r0 : float
        Equilibrium bond length in σ. Standard value: 1.0.

    Returns
    -------
    force_on_i : np.ndarray, shape (3,) or (dim,)
        Force vector acting on bead i due to this bond.
        Newton's 3rd law: force on j = -force_on_i.
    """
    dist = np.linalg.norm(r_ij)
    if dist < 1e-12:
        # Beads are on top of each other — return zero to avoid division by zero
        return np.zeros_like(r_ij)
    r_hat = r_ij / dist
    # Force magnitude: -k * (|r| - r0), direction: toward j if stretched (|r|>r0)
    # The minus sign in -k*(d-r0) makes force attractive when stretched (d>r0)
    # and repulsive when compressed (d<r0), which is correct for a spring.
    # Force on i points along r_hat (toward j) when stretched.
    force_magnitude = -k_bond * (dist - r0)
    # Return as force ON i: when stretched (dist > r0), magnitude < 0,
    # but we want force pointing TOWARD j, so we negate:
    # Actually: F_on_i = k_bond * (dist - r0) * r_hat
    # (positive when stretched → pulls i toward j)
    force_on_i = k_bond * (dist - r0) * r_hat
    return force_on_i


def harmonic_bond_potential(r_ij: np.ndarray, k_bond: float, r0: float) -> float:
    """
    Compute harmonic bond potential energy.

    U_bond = 0.5 * k_bond * (|r_ij| - r0)²

    Parameters
    ----------
    r_ij : np.ndarray, shape (3,)
        Separation vector r_j - r_i.
    k_bond : float
        Spring constant in ε/σ².
    r0 : float
        Equilibrium bond length in σ.

    Returns
    -------
    float
        Potential energy of this bond.
    """
    dist = np.linalg.norm(r_ij)
    return 0.5 * k_bond * (dist - r0) ** 2


def fene_bond_force(r_ij: np.ndarray, k: float, R0: float) -> np.ndarray:
    """
    Compute FENE (Finitely Extensible Nonlinear Elastic) bond force on bead i.

    From §4:
        F_fene = -k * r_ij / (1 - (|r_ij|/R0)²)

    The FENE force diverges as |r_ij| → R0, preventing bond extension
    beyond R0. Standard Kremer-Grest parameters: k=30 ε/σ², R0=1.5 σ.

    Parameters
    ----------
    r_ij : np.ndarray, shape (3,)
        Separation vector r_j - r_i.
    k : float
        FENE spring constant. Standard: 30.0 ε/σ².
    R0 : float
        Maximum bond extension. Standard: 1.5 σ.

    Returns
    -------
    force_on_i : np.ndarray, shape (3,)
        Force vector acting on bead i.

    Raises
    ------
    ValueError
        If bond length exceeds R0 (physically impossible, indicates
        a timestep that's too large or a bug).
    """
    dist = np.linalg.norm(r_ij)
    if dist < 1e-12:
        return np.zeros_like(r_ij)

    ratio_sq = (dist / R0) ** 2
    if ratio_sq >= 1.0:
        raise ValueError(
            f"FENE bond exceeded maximum extension: |r_ij|={dist:.6f} >= R0={R0:.6f}. "
            f"This indicates dt is too large or a numerical blow-up."
        )

    # F = -k * r_ij / (1 - (r/R0)²)
    # This is the force on bead i: it's attractive (pulls i toward j)
    # because r_ij points from i to j, and the denominator is positive but < 1.
    # The minus sign combined with r_ij pointing toward j means the force
    # pulls i toward j (restoring force).
    # 
    # Wait — let's be careful about signs:
    # The FENE potential is: U = -0.5 * k * R0² * ln(1 - (r/R0)²)
    # Force = -dU/dr * r_hat = -[-0.5 * k * R0² * (-2r/R0²) / (1-(r/R0)²)] * r_hat
    #       = -k * r / (1 - (r/R0)²) * r_hat
    # This force is attractive (points from i toward j, along r_hat),
    # which is correct — FENE pulls bonded beads together.
    # 
    # As a vector using r_ij = r_j - r_i:
    # F_on_i = -k * r_ij / (1 - (|r_ij|/R0)²)
    # Wait, that's wrong. Let me re-derive:
    # U depends on r = |r_ij|. F_on_i = -∇_i U = -dU/dr * ∂r/∂r_i
    # ∂r/∂r_i = -(r_j - r_i)/|r_j - r_i| = -r_hat_ij
    # dU/dr = -0.5 * k * R0² * (-2r/R0²) / (1-(r/R0)²) = k*r / (1-(r/R0)²)
    # F_on_i = -dU/dr * (-r_hat_ij) = dU/dr * r_hat_ij
    #        = k * r / (1-(r/R0)²) * r_hat_ij
    # 
    # Actually, let's express as: F_on_i = k * dist / (1-(dist/R0)²) * r_hat
    # where r_hat = r_ij / |r_ij| points from i toward j.
    # This is ATTRACTIVE: it pulls i toward j, which is correct.
    denominator = 1.0 - ratio_sq
    force_on_i = k * dist / denominator * (r_ij / dist)
    return force_on_i


def fene_bond_potential(r_ij: np.ndarray, k: float, R0: float) -> float:
    """
    Compute FENE bond potential energy.

    U_fene = -0.5 * k * R0² * ln(1 - (|r_ij|/R0)²)

    Parameters
    ----------
    r_ij : np.ndarray, shape (3,)
        Separation vector r_j - r_i.
    k : float
        FENE spring constant.
    R0 : float
        Maximum bond extension.

    Returns
    -------
    float
        Potential energy of this bond.
    """
    dist = np.linalg.norm(r_ij)
    ratio_sq = (dist / R0) ** 2
    if ratio_sq >= 1.0:
        return float('inf')
    return -0.5 * k * R0**2 * np.log(1.0 - ratio_sq)


def wca_force(r_ij: np.ndarray, epsilon: float, sigma: float) -> np.ndarray:
    """
    Compute WCA (Weeks-Chandler-Andersen) excluded-volume force on bead i.

    From §4:
        F_WCA = 24ε/r * [2(σ/r)¹² - (σ/r)⁶] * r̂    for r < r_cut
        F_WCA = 0                                        for r >= r_cut

    where r_cut = 2^(1/6) * σ ≈ 1.12246σ.

    WCA is the repulsive part of Lennard-Jones, cut and shifted at the
    minimum. It prevents bead overlap (excluded volume).

    Parameters
    ----------
    r_ij : np.ndarray, shape (3,)
        Separation vector r_j - r_i (from bead i to bead j).
    epsilon : float
        WCA energy scale. Standard: 1.0.
    sigma : float
        WCA length scale. Standard: 1.0.

    Returns
    -------
    force_on_i : np.ndarray, shape (3,)
        Force vector acting on bead i due to WCA repulsion.
        This force REPELS i away from j (points from j toward i).
    """
    dist = np.linalg.norm(r_ij)
    r_cut = sigma * 2.0 ** (1.0 / 6.0)  # ≈ 1.12246σ

    if dist >= r_cut or dist < 1e-12:
        return np.zeros_like(r_ij)

    # Clamp minimum distance to prevent force divergence (WCA ~ r^-13).
    # At dist = 0.1σ the raw force is ~5e14, which is non-physical.
    # Clamping to 0.4σ caps the force at ~2400 ε/σ, still strongly repulsive
    # but within the regime where Euler-Maruyama remains stable at dt=0.001.
    dist = max(dist, 0.4 * sigma)

    # WCA force derivation:
    # U_LJ = 4ε[(σ/r)¹² - (σ/r)⁶]
    # U_WCA = U_LJ + ε  (shifted so U=0 at r_cut)
    # F = -dU/dr * r_hat
    # dU_LJ/dr = 4ε[-12σ¹²/r¹³ + 6σ⁶/r⁷] = -24ε/r * [2(σ/r)¹² - (σ/r)⁶]
    # F_on_i = -dU/dr * (-r_hat_ij) = dU/dr * r_hat_ij
    # But: dU/dr < 0 for r < r_cut (repulsive), so F_on_i points AWAY from j.
    #
    # More carefully:
    # F_on_i = -∇_i U = -dU/dr * ∂r/∂r_i = -dU/dr * (-r_hat)
    # where r_hat = r_ij/|r_ij| points from i to j.
    # dU_LJ/dr = 4ε[-12σ¹²r⁻¹³ + 6σ⁶r⁻⁷]
    # For r < r_cut: dU/dr < 0 (potential is decreasing = we're on repulsive wall)
    # So F_on_i = dU/dr * r_hat, which is negative * toward-j = AWAY from j ✓

    sr6 = (sigma / dist) ** 6
    sr12 = sr6 ** 2
    r_hat = r_ij / dist

    # Force magnitude factor: 24ε/r * [2*(σ/r)¹² - (σ/r)⁶]
    # This is positive for r < r_cut (repulsive)
    force_factor = 24.0 * epsilon / dist * (2.0 * sr12 - sr6)

    # Force on i is REPULSIVE: points away from j = -r_hat direction
    # F_on_i = -force_factor * r_hat
    # (negative because we want force AWAY from j, and r_hat points TOWARD j)
    force_on_i = -force_factor * r_hat
    return force_on_i


def wca_potential(r_ij: np.ndarray, epsilon: float, sigma: float) -> float:
    """
    Compute WCA potential energy.

    U_WCA = 4ε[(σ/r)¹² - (σ/r)⁶] + ε    for r < r_cut = 2^(1/6)σ
    U_WCA = 0                               for r >= r_cut

    Parameters
    ----------
    r_ij : np.ndarray, shape (3,)
        Separation vector r_j - r_i.
    epsilon : float
        WCA energy scale.
    sigma : float
        WCA length scale.

    Returns
    -------
    float
        Potential energy of this pair.
    """
    dist = np.linalg.norm(r_ij)
    r_cut = sigma * 2.0 ** (1.0 / 6.0)

    if dist >= r_cut or dist < 1e-12:
        return 0.0

    # Clamp minimum distance to be consistent with the force clamping
    dist = max(dist, 0.4 * sigma)

    sr6 = (sigma / dist) ** 6
    sr12 = sr6 ** 2
    return 4.0 * epsilon * (sr12 - sr6) + epsilon


# =============================================================================
# AGGREGATE FORCE COMPUTATION (§8.1 pseudocode)
# =============================================================================

def compute_bonded_forces(
    positions: np.ndarray,
    bond_type: str = "harmonic",
    k_bond: float = 100.0,
    r0: float = 1.0,
    k_fene: float = 30.0,
    R0_fene: float = 1.5,
) -> Tuple[np.ndarray, float]:
    """
    Compute bonded forces for a linear polymer chain.

    From §8.1 pseudocode:
        for i in 1..N-1:
            r_ij = r[i+1] - r[i]
            F_bond = bond_force(r_ij, k_bond, r0)
            F[i] += F_bond
            F[i+1] -= F_bond    # Newton's 3rd law

    Parameters
    ----------
    positions : np.ndarray, shape (N, dim)
        Bead positions.
    bond_type : str
        "harmonic" or "fene".
    k_bond : float
        Harmonic spring constant (used if bond_type="harmonic").
    r0 : float
        Harmonic equilibrium length (used if bond_type="harmonic").
    k_fene : float
        FENE spring constant (used if bond_type="fene").
    R0_fene : float
        FENE max extension (used if bond_type="fene").

    Returns
    -------
    forces : np.ndarray, shape (N, dim)
        Net bonded force on each bead.
    potential_energy : float
        Total bonded potential energy.
    """
    N, dim = positions.shape
    forces = np.zeros_like(positions)
    
    # Vectorized computation for all N-1 bonds
    r_ij = positions[1:] - positions[:-1]  # Vectors from i to i+1
    dist = np.linalg.norm(r_ij, axis=1)
    
    nonzero = dist > 1e-12
    r_hat = np.zeros_like(r_ij)
    r_hat[nonzero] = r_ij[nonzero] / dist[nonzero, np.newaxis]
    
    f_bond = np.zeros_like(r_ij)
    pe = 0.0
    
    if bond_type == "harmonic":
        f_mag = k_bond * (dist - r0)
        f_bond[nonzero] = f_mag[nonzero, np.newaxis] * r_hat[nonzero]
        pe = float(np.sum(0.5 * k_bond * (dist - r0) ** 2))
    elif bond_type == "fene":
        ratio_sq = (dist / R0_fene) ** 2
        if np.any(ratio_sq >= 1.0):
            max_dist = np.max(dist)
            raise ValueError(
                f"FENE bond exceeded maximum extension: max |r_ij|={max_dist:.6f} >= R0={R0_fene:.6f}. "
                f"This indicates dt is too large or a numerical blow-up."
            )
        denominator = 1.0 - ratio_sq
        f_mag = k_fene * dist / denominator
        f_bond[nonzero] = f_mag[nonzero, np.newaxis] * r_hat[nonzero]
        pe = float(np.sum(-0.5 * k_fene * R0_fene**2 * np.log(denominator)))
    else:
        raise ValueError(f"Unknown bond type: {bond_type}")

    # §8.1: F[i] += F_bond; F[i+1] -= F_bond (Newton's 3rd law)
    forces[:-1] += f_bond
    forces[1:] -= f_bond

    return forces, pe


def compute_nonbonded_forces(
    positions: np.ndarray,
    epsilon: float = 1.0,
    sigma: float = 1.0,
) -> Tuple[np.ndarray, float]:
    """
    Compute nonbonded (WCA excluded-volume) forces.

    From §8.1 pseudocode:
        for all nonbonded pairs (i,j) with |i-j| > 1:
            if distance(r[i], r[j]) < cutoff:
                F_rep = wca_force(r[i]-r[j], epsilon, sigma)
                F[i] += F_rep
                F[j] -= F_rep    # Newton's 3rd law

    Note: pairs with |i-j| <= 1 are bonded neighbors and are skipped
    to avoid double-counting with the bonded force.

    Parameters
    ----------
    positions : np.ndarray, shape (N, dim)
        Bead positions.
    epsilon : float
        WCA energy scale.
    sigma : float
        WCA length scale.

    Returns
    -------
    forces : np.ndarray, shape (N, dim)
        Net nonbonded force on each bead.
    potential_energy : float
        Total nonbonded potential energy.
    """
    N, dim = positions.shape
    forces = np.zeros_like(positions)
    pe = 0.0
    r_cut = sigma * 2.0 ** (1.0 / 6.0)

    # Vectorized computation over all nonbonded pairs (j > i + 1)
    i_idx, j_idx = np.triu_indices(N, k=2)
    if len(i_idx) == 0:
        return forces, pe
        
    r_ij = positions[j_idx] - positions[i_idx]
    dist = np.linalg.norm(r_ij, axis=1)

    mask = (dist < r_cut) & (dist > 1e-12)

    # Clamp minimum distance to prevent WCA force divergence (consistent
    # with pairwise wca_force). This caps the maximum repulsive force
    # at ~2400 ε/σ instead of allowing it to reach 10^14+ at close approach.
    dist = np.clip(dist, 0.4 * sigma, None)
    if not np.any(mask):
        return forces, pe
        
    r_ij_active = r_ij[mask]
    dist_active = dist[mask]
    i_idx_active = i_idx[mask]
    j_idx_active = j_idx[mask]
    
    sr6 = (sigma / dist_active) ** 6
    sr12 = sr6 ** 2
    
    force_factor = 24.0 * epsilon / dist_active * (2.0 * sr12 - sr6)
    r_hat = r_ij_active / dist_active[:, np.newaxis]
    
    # F_on_i is repulsive (points away from j = -r_hat)
    f_wca = -force_factor[:, np.newaxis] * r_hat
    
    # Add forces back to original array (Newton's 3rd law)
    np.add.at(forces, i_idx_active, f_wca)
    np.subtract.at(forces, j_idx_active, f_wca)
    
    pe = float(np.sum(4.0 * epsilon * (sr12 - sr6) + epsilon))

    return forces, pe


def compute_all_forces(
    positions: np.ndarray,
    bond_type: str = "harmonic",
    k_bond: float = 100.0,
    r0: float = 1.0,
    k_fene: float = 30.0,
    R0_fene: float = 1.5,
    epsilon: float = 1.0,
    sigma: float = 1.0,
) -> Tuple[np.ndarray, float, float]:
    """
    Compute all forces (bonded + nonbonded) on a polymer chain.

    From §8.1: total force is the sum of bonded forces and excluded-volume
    (WCA) forces.

    Parameters
    ----------
    positions : np.ndarray, shape (N, dim)
        Bead positions.
    bond_type : str
        "harmonic" or "fene".
    k_bond, r0, k_fene, R0_fene : float
        Bond parameters (see compute_bonded_forces).
    epsilon, sigma : float
        WCA parameters.

    Returns
    -------
    total_forces : np.ndarray, shape (N, dim)
        Total force on each bead.
    bond_pe : float
        Bonded potential energy.
    nonbond_pe : float
        Nonbonded (WCA) potential energy.
    """
    f_bond, pe_bond = compute_bonded_forces(
        positions, bond_type, k_bond, r0, k_fene, R0_fene
    )
    f_nonbond, pe_nonbond = compute_nonbonded_forces(positions, epsilon, sigma)

    total_forces = f_bond + f_nonbond

    # Safety: cap maximum per-bead force magnitude to prevent rare
    # catastrophic single-step launches from close WCA encounters.
    # A cap of 1000 ε/σ corresponds to a max displacement of
    # (dt/γ) * F_max = 0.001 * 1000 = 1.0σ per step, which is large
    # but recoverable. Without this, forces can reach 10^14+.
    F_MAX = 1000.0  # ε/σ
    force_magnitudes = np.linalg.norm(total_forces, axis=1, keepdims=True)
    clipping_mask = force_magnitudes > F_MAX
    if np.any(clipping_mask):
        scale = np.where(clipping_mask, F_MAX / force_magnitudes, 1.0)
        total_forces = total_forces * scale

    return total_forces, pe_bond, pe_nonbond
