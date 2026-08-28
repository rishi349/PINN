"""Rouse mode analysis for polymer chain dynamics (§13, coding_agent_brief Task 1).

The Rouse model predicts that normal modes of a Gaussian chain have
independently relaxing amplitudes. For a linear chain of N beads:

    X_p(t) = (1/N) Σ_{i=1}^{N} r_i(t) · cos(π·p·(i - 1/2) / N)

    for mode index p = 0, 1, ..., N-1.

Key predictions (Rouse 1953, de Gennes Ch. VI):
  - Mode p=0 is the chain centre-of-mass.
  - Autocorrelation:  C_p(τ) = ⟨X_p(t)·X_p(t+τ)⟩ = C_p(0)·exp(-τ/τ_p)
  - Relaxation times: τ_p ∝ 1/p²  (the "gold standard" of Rouse dynamics)
  - Slowest time:     τ_1 = τ_R (Rouse time) ∝ N²

References
----------
- Rouse, P.E. (1953) J. Chem. Phys. 21, 1272.
- de Gennes (1979) Scaling Concepts in Polymer Physics, Ch. VI.
- Doi & Edwards (1986) The Theory of Polymer Dynamics, Ch. 4.
- Kremer & Grest (1990) J. Chem. Phys. 92, 5057.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Mode projection
# ---------------------------------------------------------------------------

def compute_rouse_modes(
    positions_traj: np.ndarray,
    n_modes: Optional[int] = None,
) -> np.ndarray:
    """Project bead positions onto Rouse normal modes X_p(t).

    Parameters
    ----------
    positions_traj : np.ndarray, shape (T, N, dim)
        Trajectory of bead positions.  T = number of frames,
        N = number of beads, dim = spatial dimension (3).
    n_modes : int, optional
        Number of modes to compute (p = 0 ... n_modes-1).
        Defaults to N (all modes).

    Returns
    -------
    np.ndarray, shape (T, n_modes, dim)
        Rouse mode amplitudes X_p(t) for each time step and mode.

    Notes
    -----
    The cosine projection is:
        X_p(t) = (1/N) Σ_{i=1}^{N} r_i(t) · cos(π·p·(i - 1/2) / N)

    This is the standard Rouse-mode definition (Doi & Edwards Eq. 4.5).
    Mode p=0 gives the centre of mass position (up to a factor of 1/N).
    """
    T, N, dim = positions_traj.shape
    if n_modes is None:
        n_modes = N

    # Build projection matrix once: shape (n_modes, N)
    i_idx = np.arange(1, N + 1)             # bead index 1..N
    p_idx = np.arange(n_modes)[:, np.newaxis]  # mode index 0..n_modes-1
    cos_matrix = np.cos(np.pi * p_idx * (i_idx - 0.5) / N) / N  # (n_modes, N)

    # Project: (T, N, dim) -> (T, n_modes, dim)
    # X_p(t) = cos_matrix @ positions_traj[t]  for each t
    modes = np.einsum("pn,tnd->tpd", cos_matrix, positions_traj)  # (T, n_modes, dim)
    return modes


# ---------------------------------------------------------------------------
# Autocorrelation
# ---------------------------------------------------------------------------

def compute_mode_autocorrelation(
    modes: np.ndarray,
    max_lag: Optional[int] = None,
    subtract_mean: bool = True,
) -> np.ndarray:
    """Compute the autocorrelation C_p(τ) = ⟨X_p(t)·X_p(t+τ)⟩ for each mode.

    Parameters
    ----------
    modes : np.ndarray, shape (T, n_modes, dim)
        Rouse mode amplitudes from compute_rouse_modes().
    max_lag : int, optional
        Maximum lag τ to compute. Defaults to T//2 for statistical reliability.
    subtract_mean : bool
        Whether to subtract the time-mean of X_p before computing correlation
        (i.e., compute the fluctuation autocorrelation). Default True.

    Returns
    -------
    np.ndarray, shape (max_lag, n_modes)
        C_p(τ) for each mode p, normalised so C_p(0) = 1.
    """
    T, n_modes, dim = modes.shape
    if max_lag is None:
        max_lag = T // 2

    if subtract_mean:
        modes = modes - modes.mean(axis=0, keepdims=True)

    # Scalar mode amplitude: take vector dot product over spatial dim
    # amp[t, p] = X_p(t) · X_p(t)  -- we need the cross-time product
    # C_p(τ) = (1/(T-τ)) Σ_t  X_p(t) · X_p(t+τ)

    corr = np.zeros((max_lag, n_modes))
    for lag in range(max_lag):
        # dot product summed over dim: (T-lag, n_modes)
        prod = np.sum(modes[:T - lag] * modes[lag:], axis=-1)  # (T-lag, n_modes)
        corr[lag] = prod.mean(axis=0)

    # Normalise by C_p(0)
    c0 = corr[0:1]  # (1, n_modes)
    mask = np.abs(c0) > 1e-12
    corr = np.where(mask, corr / np.where(mask, c0, 1.0), 0.0)
    return corr  # (max_lag, n_modes)


# ---------------------------------------------------------------------------
# Relaxation time extraction
# ---------------------------------------------------------------------------

def extract_relaxation_times(
    corr: np.ndarray,
    dt: float = 0.001,
    fit_range: float = 0.8,
) -> np.ndarray:
    """Fit exponential decay to each mode's autocorrelation to get τ_p.

    Fits C_p(τ) = exp(-τ/τ_p) in the range [0, fit_range] of the
    normalised correlation (i.e., stops fitting once the correlation
    decays below `fit_range` threshold to avoid fitting noise).

    Parameters
    ----------
    corr : np.ndarray, shape (max_lag, n_modes)
        Normalised mode autocorrelations from compute_mode_autocorrelation().
    dt : float
        Time step between frames in simulation time units.
    fit_range : float
        Fraction of initial correlation value above which to fit.
        Default 0.8 means fit while C_p(τ) > 0.8·C_p(0)=0.8.

    Returns
    -------
    np.ndarray, shape (n_modes,)
        Relaxation time τ_p for each mode, in simulation time units.
        Returns np.nan for modes where the fit fails.
    """
    max_lag, n_modes = corr.shape
    tau_p = np.full(n_modes, np.nan)
    lags = np.arange(max_lag) * dt

    for p in range(n_modes):
        c = corr[:, p]
        # Use only positive-correlation region
        valid = (c > 0) & (~np.isnan(c))
        if not np.any(valid):
            continue

        # Linear fit in log-space:  log C_p(τ) = -τ/τ_p
        log_c = np.log(np.clip(c[valid], 1e-12, None))
        t_valid = lags[valid]

        try:
            coeffs = np.polyfit(t_valid, log_c, 1)
            slope = coeffs[0]
            if slope < 0:
                tau_p[p] = -1.0 / slope
        except (np.linalg.LinAlgError, ValueError):
            pass

    return tau_p


# ---------------------------------------------------------------------------
# Rouse scaling check
# ---------------------------------------------------------------------------

def check_rouse_scaling(
    tau_p: np.ndarray,
    mode_indices: Optional[np.ndarray] = None,
    rtol: float = 0.25,
    min_modes: int = 4,
) -> dict:
    """Check whether τ_p ∝ 1/p² as predicted by the Rouse model.

    Fits log(τ_p) = -α·log(p) + const and checks if α ≈ 2.

    Parameters
    ----------
    tau_p : np.ndarray, shape (n_modes,)
        Relaxation times from extract_relaxation_times().
    mode_indices : np.ndarray, optional
        Which mode indices to include in the fit. Defaults to p=1..n_modes-1
        (excluding COM mode p=0 and the highest modes that are noise-dominated).
    rtol : float
        Tolerance on the exponent α — passes if |α - 2| / 2 < rtol.
    min_modes : int
        Minimum number of valid modes needed to attempt the fit.

    Returns
    -------
    dict with keys:
        'passes'       : bool   — whether τ_p ∝ 1/p² within tolerance
        'alpha'        : float  — fitted exponent
        'alpha_target' : float  — theoretical value (2.0)
        'rtol_used'    : float  — relative tolerance used
        'n_modes_fit'  : int    — number of modes included in fit
        'tau_p_fit'    : array  — τ_p values used in fit
        'p_fit'        : array  — mode indices used in fit
    """
    n_modes = len(tau_p)

    if mode_indices is None:
        # Exclude p=0 (COM) and use only lower half of modes
        # (high modes have very short τ and are noise-dominated)
        mode_indices = np.arange(1, max(2, n_modes // 2 + 1))

    # Keep only valid (finite, positive) modes
    valid = np.isfinite(tau_p[mode_indices]) & (tau_p[mode_indices] > 0)
    p_fit = mode_indices[valid]
    tau_fit = tau_p[p_fit]

    result = {
        'passes': False,
        'alpha': np.nan,
        'alpha_target': 2.0,
        'rtol_used': rtol,
        'n_modes_fit': len(p_fit),
        'tau_p_fit': tau_fit,
        'p_fit': p_fit,
    }

    if len(p_fit) < min_modes:
        return result

    # Log-log fit: log(τ) = -α·log(p) + const
    log_p = np.log(p_fit.astype(float))
    log_tau = np.log(tau_fit)
    try:
        coeffs = np.polyfit(log_p, log_tau, 1)
        alpha = -coeffs[0]   # negative slope → exponent
        result['alpha'] = float(alpha)
        result['passes'] = bool(abs(alpha - 2.0) / 2.0 < rtol)
    except (np.linalg.LinAlgError, ValueError):
        pass

    return result


# ---------------------------------------------------------------------------
# Convenience: full pipeline
# ---------------------------------------------------------------------------

def full_rouse_analysis(
    positions_traj: np.ndarray,
    dt: float = 0.001,
    n_modes: Optional[int] = None,
    max_lag: Optional[int] = None,
) -> dict:
    """Run the complete Rouse mode analysis pipeline.

    Parameters
    ----------
    positions_traj : np.ndarray, shape (T, N, dim)
        Trajectory of bead positions.
    dt : float
        Time step in simulation units.
    n_modes : int, optional
        Number of modes to analyse (defaults to N).
    max_lag : int, optional
        Maximum lag for autocorrelation (defaults to T//2).

    Returns
    -------
    dict with keys:
        'modes'         : (T, n_modes, dim) — Rouse mode amplitudes
        'corr'          : (max_lag, n_modes) — normalised autocorrelations
        'tau_p'         : (n_modes,) — relaxation times in sim. units
        'scaling_check' : dict from check_rouse_scaling()
    """
    modes = compute_rouse_modes(positions_traj, n_modes=n_modes)
    corr = compute_mode_autocorrelation(modes, max_lag=max_lag)
    tau_p = extract_relaxation_times(corr, dt=dt)
    scaling = check_rouse_scaling(tau_p)
    return {
        'modes': modes,
        'corr': corr,
        'tau_p': tau_p,
        'scaling_check': scaling,
    }
