"""Fluctuation-Dissipation Theorem (FDT) temperature check for the simulator.

For overdamped Langevin dynamics (Brownian / Rouse dynamics), the
fluctuation-dissipation theorem (FDT) relates the mean-squared displacement
per step to the temperature:

    ⟨|Δr_i|²⟩ = 2 * dim * (k_B·T / γ) * dt
                = 2 * dim * D_monomer * dt

where:
    Δr_i  = r_i(t + dt) - r_i(t)   (single-step displacement)
    dim   = spatial dimension (3)
    k_B·T = thermal energy (= 1.0 in our reduced units)
    γ     = friction coefficient (= 1.0 in our Euler-Maruyama integrator)
    dt    = time step
    D_monomer = k_B·T / γ = 1.0 (in reduced units)

So the *expected* mean squared single-step displacement is:
    ⟨|Δr|²⟩_expected = 2 * 3 * 1.0 * dt = 6 * dt

At dt = 0.001, this gives ⟨|Δr|²⟩ ≈ 0.006 σ².

A measured temperature outside ±5% of the target T is a red flag —
it means the integrator is incorrectly scaled, the friction is wrong,
or the force capping is substantially biasing the dynamics.

References
----------
- de Gennes (1979) Scaling Concepts in Polymer Physics, Ch. VI.
- Doi & Edwards (1986) Theory of Polymer Dynamics, §3.1.
- Kremer & Grest (1990) J. Chem. Phys. 92, 5057.
- Project contract §2 (Euler-Maruyama integrator, overdamped Langevin).
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple


def estimate_temperature_from_displacements(
    positions_traj: np.ndarray,
    dt: float = 0.001,
    gamma: float = 1.0,
    dim: int = 3,
) -> Tuple[float, float]:
    """Estimate effective temperature from per-step displacement statistics.

    Uses the FDT relation:
        T_measured = ⟨|Δr_i|²⟩ * γ / (2 * dim * dt)

    where ⟨·⟩ is averaged over all beads and all time steps.

    Parameters
    ----------
    positions_traj : np.ndarray, shape (T, N, dim)
        Trajectory of bead positions.
    dt : float
        Simulation time step (in reduced units).
    gamma : float
        Friction coefficient (= 1.0 in standard KG reduced units).
    dim : int
        Spatial dimension (3).

    Returns
    -------
    T_measured : float
        Effective temperature inferred from displacements.
    T_std : float
        Standard deviation across beads of the per-bead estimated temperature
        (a measure of spatial inhomogeneity — should be small).
    """
    T_frames, N, d = positions_traj.shape

    # Single-step displacements: shape (T-1, N, dim)
    displacements = positions_traj[1:] - positions_traj[:-1]

    # Squared displacement per bead per step: (T-1, N)
    sq_disp = np.sum(displacements ** 2, axis=-1)

    # Per-bead mean squared displacement
    msd_per_bead = sq_disp.mean(axis=0)  # (N,)

    # T_i = msd_i * gamma / (2 * dim * dt)
    T_per_bead = msd_per_bead * gamma / (2 * dim * dt)

    T_measured = float(T_per_bead.mean())
    T_std = float(T_per_bead.std())

    return T_measured, T_std
