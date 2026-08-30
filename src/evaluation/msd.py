"""Mean Squared Displacement (MSD) analysis for polymer chain dynamics.

Implements the three MSD functions (g1, g2, g3) defined in de Gennes
Ch. VI and standard polymer simulation literature:

    g1(t) = ⟨ |r_i(t) - r_i(0)|² ⟩         — individual bead MSD
    g2(t) = ⟨ |(r_i(t)-R_cm(t)) - (r_i(0)-R_cm(0))|² ⟩  — bead MSD relative to COM
    g3(t) = ⟨ |R_cm(t) - R_cm(0)|² ⟩        — centre-of-mass MSD

Rouse model predictions (§13 coding_agent_brief Task 2):
    g1(t) ~ t^{1/2}   for τ_b << t << τ_R   (sub-diffusive, Rouse regime)
    g2(t) ~ t^{1/2}   for τ_b << t << τ_R
    g3(t) ~ t^{1.0}   for t >> τ_R           (Fickian diffusion of COM)

References
----------
- Kremer & Grest (1990) J. Chem. Phys. 92, 5057.
- de Gennes (1979) Scaling Concepts in Polymer Physics, Ch. VI.
- Doi & Edwards (1986) Theory of Polymer Dynamics, §4.1.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# MSD functions
# ---------------------------------------------------------------------------

def compute_msd_g1(
    positions_traj: np.ndarray,
    max_lag: Optional[int] = None,
    average_over_beads: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute g1(t): individual bead MSD ⟨|r_i(t) - r_i(0)|²⟩.

    Uses a time-origin-average: for each lag τ, average over all available
    (t0, t0+τ) pairs for better statistics.

    Parameters
    ----------
    positions_traj : np.ndarray, shape (T, N, dim)
        Trajectory of bead positions.
    max_lag : int, optional
        Maximum lag to compute. Defaults to T//2.
    average_over_beads : bool
        If True (default), average over all beads. If False, return per-bead MSD
        with shape (max_lag, N).

    Returns
    -------
    lags : np.ndarray, shape (max_lag,)
        Lag indices (multiply by dt to get time).
    msd : np.ndarray, shape (max_lag,) or (max_lag, N)
        g1(t) values.
    """
    T, N, dim = positions_traj.shape
    if max_lag is None:
        max_lag = T // 2

    msd = np.zeros((max_lag, N))
    counts = np.zeros(max_lag, dtype=int)

    for lag in range(1, max_lag + 1):
        disp = positions_traj[lag:] - positions_traj[:T - lag]  # (T-lag, N, dim)
        sq_disp = np.sum(disp ** 2, axis=-1)                     # (T-lag, N)
        msd[lag - 1] = sq_disp.mean(axis=0)                      # (N,)
        counts[lag - 1] = T - lag

    lags = np.arange(1, max_lag + 1)
    if average_over_beads:
        return lags, msd.mean(axis=1)
    return lags, msd


def compute_msd_g2(
    positions_traj: np.ndarray,
    max_lag: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute g2(t): bead MSD relative to chain centre-of-mass.

    g2(t) = ⟨ |(r_i(t) - R_cm(t)) - (r_i(0) - R_cm(0))|² ⟩

    This isolates internal chain motion from COM diffusion. In the Rouse
    model, g2(t) ~ t^{1/2} for τ_b << t << τ_R.

    Parameters
    ----------
    positions_traj : np.ndarray, shape (T, N, dim)
        Trajectory of bead positions.
    max_lag : int, optional
        Maximum lag to compute. Defaults to T//2.

    Returns
    -------
    lags : np.ndarray, shape (max_lag,)
    msd_g2 : np.ndarray, shape (max_lag,)
    """
    T, N, dim = positions_traj.shape
    if max_lag is None:
        max_lag = T // 2

    # Subtract COM at each frame: shape (T, N, dim)
    com = positions_traj.mean(axis=1, keepdims=True)  # (T, 1, dim)
    pos_rel = positions_traj - com                     # (T, N, dim)

    msd_g2 = np.zeros(max_lag)
    for lag in range(1, max_lag + 1):
        disp = pos_rel[lag:] - pos_rel[:T - lag]       # (T-lag, N, dim)
        sq_disp = np.sum(disp ** 2, axis=-1)            # (T-lag, N)
        msd_g2[lag - 1] = sq_disp.mean()

    lags = np.arange(1, max_lag + 1)
    return lags, msd_g2


def compute_msd_g3(
    positions_traj: np.ndarray,
    max_lag: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute g3(t): chain centre-of-mass MSD.

    g3(t) = ⟨ |R_cm(t) - R_cm(0)|² ⟩

    In the Rouse model, g3(t) ~ t for t >> τ_R (Fickian diffusion).
    The slope gives D_chain = g3(t)/(6t).

    Parameters
    ----------
    positions_traj : np.ndarray, shape (T, N, dim)
        Trajectory of bead positions.
    max_lag : int, optional
        Maximum lag to compute. Defaults to T//2.

    Returns
    -------
    lags : np.ndarray, shape (max_lag,)
    msd_g3 : np.ndarray, shape (max_lag,)
    """
    T, N, dim = positions_traj.shape
    if max_lag is None:
        max_lag = T // 2

    com = positions_traj.mean(axis=1)  # (T, dim) — chain COM trajectory

    msd_g3 = np.zeros(max_lag)
    for lag in range(1, max_lag + 1):
        disp = com[lag:] - com[:T - lag]          # (T-lag, dim)
        sq_disp = np.sum(disp ** 2, axis=-1)       # (T-lag,)
        msd_g3[lag - 1] = sq_disp.mean()

    lags = np.arange(1, max_lag + 1)
    return lags, msd_g3


# ---------------------------------------------------------------------------
# Exponent fitting
# ---------------------------------------------------------------------------
