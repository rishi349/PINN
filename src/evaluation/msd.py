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

def fit_msd_exponent(
    msd: np.ndarray,
    lags: np.ndarray,
    fit_start: float = 0.1,
    fit_end: float = 0.9,
) -> Tuple[float, float]:
    """Fit MSD ~ A·t^α in log-log space to extract the scaling exponent.

    Parameters
    ----------
    msd : np.ndarray, shape (max_lag,)
        MSD values.
    lags : np.ndarray, shape (max_lag,)
        Corresponding lag indices.
    fit_start : float
        Fraction of the lag range at which to start fitting (avoids ballistic
        regime at short times). Default 0.1.
    fit_end : float
        Fraction of the lag range at which to stop fitting (avoids noise
        saturation at long times). Default 0.9.

    Returns
    -------
    alpha : float
        Scaling exponent (0.5 expected for Rouse subdiffusion of g1/g2,
        1.0 expected for Fickian diffusion of g3).
    prefactor : float
        Prefactor A in MSD = A·t^α.
    """
    n = len(lags)
    i_start = max(1, int(n * fit_start))
    i_end = min(n - 1, int(n * fit_end))

    log_t = np.log(lags[i_start:i_end].astype(float))
    log_msd = np.log(np.clip(msd[i_start:i_end], 1e-30, None))

    valid = np.isfinite(log_t) & np.isfinite(log_msd)
    if valid.sum() < 3:
        return np.nan, np.nan

    coeffs = np.polyfit(log_t[valid], log_msd[valid], 1)
    alpha = float(coeffs[0])
    prefactor = float(np.exp(coeffs[1]))
    return alpha, prefactor


# ---------------------------------------------------------------------------
# Diffusion coefficient
# ---------------------------------------------------------------------------

def estimate_diffusion_coefficient(
    msd_g3: np.ndarray,
    lags: np.ndarray,
    dt: float = 0.001,
    dim: int = 3,
    fit_start: float = 0.5,
    fit_end: float = 0.9,
) -> float:
    """Estimate chain diffusion coefficient from g3(t) = 2*dim*D*t.

    D = slope(g3 vs t) / (2 * dim)

    Parameters
    ----------
    msd_g3 : np.ndarray, shape (max_lag,)
        Centre-of-mass MSD.
    lags : np.ndarray, shape (max_lag,)
        Lag indices (multiply by dt for time).
    dt : float
        Time step in simulation units.
    dim : int
        Spatial dimension.
    fit_start, fit_end : float
        Fraction of lag range to use for linear fit.

    Returns
    -------
    float
        Diffusion coefficient D in units σ²/τ.
    """
    times = lags * dt
    n = len(times)
    i_start = int(n * fit_start)
    i_end = int(n * fit_end)

    if i_end <= i_start + 2:
        return np.nan

    coeffs = np.polyfit(times[i_start:i_end], msd_g3[i_start:i_end], 1)
    D = coeffs[0] / (2 * dim)
    return float(D)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def full_msd_analysis(
    positions_traj: np.ndarray,
    dt: float = 0.001,
    max_lag: Optional[int] = None,
) -> dict:
    """Run the complete MSD analysis: g1, g2, g3, exponent fits, D.

    Parameters
    ----------
    positions_traj : np.ndarray, shape (T, N, dim)
        Trajectory of bead positions.
    dt : float
        Simulation time step.
    max_lag : int, optional
        Maximum lag. Defaults to T//2.

    Returns
    -------
    dict with keys:
        'lags_time'     : times in simulation units
        'g1', 'g2', 'g3' : MSD arrays
        'alpha_g1', 'alpha_g2', 'alpha_g3' : scaling exponents
        'prefactor_g1', 'prefactor_g2', 'prefactor_g3'
        'D_chain'       : diffusion coefficient from g3
        'rouse_g1_ok'   : bool — g1 exponent within 20% of 0.5
        'rouse_g2_ok'   : bool — g2 exponent within 20% of 0.5
    """
    lags, g1 = compute_msd_g1(positions_traj, max_lag=max_lag)
    _, g2 = compute_msd_g2(positions_traj, max_lag=max_lag)
    _, g3 = compute_msd_g3(positions_traj, max_lag=max_lag)

    times = lags * dt

    alpha_g1, pre_g1 = fit_msd_exponent(g1, lags)
    alpha_g2, pre_g2 = fit_msd_exponent(g2, lags)
    alpha_g3, pre_g3 = fit_msd_exponent(g3, lags)
    D = estimate_diffusion_coefficient(g3, lags, dt=dt)

    return {
        'lags_time': times,
        'g1': g1,
        'g2': g2,
        'g3': g3,
        'alpha_g1': alpha_g1,
        'alpha_g2': alpha_g2,
        'alpha_g3': alpha_g3,
        'prefactor_g1': pre_g1,
        'prefactor_g2': pre_g2,
        'prefactor_g3': pre_g3,
        'D_chain': D,
        # Rouse check: exponents should be ~0.5 for g1/g2
        'rouse_g1_ok': bool(np.isfinite(alpha_g1) and abs(alpha_g1 - 0.5) / 0.5 < 0.20),
        'rouse_g2_ok': bool(np.isfinite(alpha_g2) and abs(alpha_g2 - 0.5) / 0.5 < 0.20),
    }
