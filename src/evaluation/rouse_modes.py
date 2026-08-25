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
