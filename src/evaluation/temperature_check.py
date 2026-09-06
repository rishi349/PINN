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


def check_fdt(
    positions_traj: np.ndarray,
    T_target: float = 1.0,
    dt: float = 0.001,
    gamma: float = 1.0,
    dim: int = 3,
    rtol: float = 0.05,
) -> dict:
    """Run the FDT temperature check on a trajectory.

    Passes if |T_measured - T_target| / T_target < rtol (default 5%).

    Parameters
    ----------
    positions_traj : np.ndarray, shape (T, N, dim)
        Trajectory of bead positions.
    T_target : float
        Target temperature in reduced units (k_B·T = 1.0 by default).
    dt : float
        Simulation time step.
    gamma : float
        Friction coefficient.
    dim : int
        Spatial dimension.
    rtol : float
        Relative tolerance. Default 0.05 → 5%.

    Returns
    -------
    dict with keys:
        'passes'       : bool — True if within rtol of target
        'T_measured'   : float — measured effective temperature
        'T_target'     : float — expected temperature
        'T_std'        : float — spatial std of per-bead temperatures
        'rel_error'    : float — |T_measured - T_target| / T_target
        'rtol_used'    : float — tolerance used
        'msd_expected' : float — expected ⟨|Δr|²⟩ at T_target
        'msd_measured' : float — actual ⟨|Δr|²⟩ from trajectory
    """
    T_measured, T_std = estimate_temperature_from_displacements(
        positions_traj, dt=dt, gamma=gamma, dim=dim
    )

    rel_error = abs(T_measured - T_target) / T_target

    # Also compute raw MSD for transparency
    displacements = positions_traj[1:] - positions_traj[:-1]
    msd_measured = float(np.mean(np.sum(displacements ** 2, axis=-1)))
    msd_expected = 2.0 * dim * (T_target / gamma) * dt

    return {
        'passes': bool(rel_error < rtol),
        'T_measured': T_measured,
        'T_target': T_target,
        'T_std': T_std,
        'rel_error': rel_error,
        'rtol_used': rtol,
        'msd_expected': msd_expected,
        'msd_measured': msd_measured,
    }


def check_fdt_rollout(
    predicted_positions: np.ndarray,
    T_target: float = 1.0,
    dt: float = 0.001,
    gamma: float = 1.0,
    rtol: float = 0.10,
) -> dict:
    """Run the FDT temperature check on a GNN rollout trajectory.

    Uses a relaxed tolerance (default 10% vs 5% for the simulator) because
    the GNN rollout integrates errors over many steps.

    Parameters
    ----------
    predicted_positions : np.ndarray, shape (T, N, dim)
        GNN rollout trajectory (positions at each step).
    T_target : float
        Target temperature in reduced units.
    dt : float
        Time step (same as used in training data generation).
    gamma : float
        Friction coefficient.
    rtol : float
        Relative tolerance (default 10% for rollout).

    Returns
    -------
    dict
        Same structure as check_fdt() output.
    """
    return check_fdt(
        predicted_positions,
        T_target=T_target,
        dt=dt,
        gamma=gamma,
        rtol=rtol,
    )
