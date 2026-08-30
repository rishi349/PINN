"""Tests for Rouse mode analysis (src/evaluation/rouse_modes.py)."""

import numpy as np
import pytest
from src.evaluation.rouse_modes import (
    compute_rouse_modes,
    compute_mode_autocorrelation,
    extract_relaxation_times,
    check_rouse_scaling,
    full_rouse_analysis,
)


def make_ideal_rouse_trajectory(
    T: int = 2000,
    N: int = 20,
    dim: int = 3,
    dt: float = 0.001,
    seed: int = 0,
) -> np.ndarray:
    """Generate a synthetic Rouse-like trajectory for testing.

    Uses independent OU processes for each mode with τ_p = τ_1/p².
    """
    rng = np.random.default_rng(seed)
    tau_1 = 0.5  # slowest mode relaxation time

    # Generate mode amplitudes as OU processes
    n_modes = N
    modes = np.zeros((T, n_modes, dim))

    for p in range(1, n_modes):
        tau_p = tau_1 / p ** 2
        amp = np.sqrt(tau_p * dt)   # noise amplitude
        for t in range(1, T):
            modes[t, p] = modes[t-1, p] * (1 - dt / tau_p) + rng.normal(0, amp, dim)

    # Add COM drift (p=0)
    modes[:, 0] = rng.normal(0, 0.1, (T, dim)).cumsum(axis=0) * dt

    # Inverse Rouse transform to get bead positions
    i_idx = np.arange(1, N + 1)
    p_idx = np.arange(n_modes)[:, np.newaxis]
    cos_mat = np.cos(np.pi * p_idx * (i_idx - 0.5) / N)  # (n_modes, N)

    # positions[t] = Σ_p X_p(t) * cos_mat[p, :] * 2  (for p>0), X_0 for p=0
    positions = np.einsum("tpd,pn->tnd", modes, cos_mat)
    return positions


class TestComputeRouseModes:

    def test_output_shape(self):
        traj = np.random.randn(100, 20, 3)
        modes = compute_rouse_modes(traj)
        assert modes.shape == (100, 20, 3)

    def test_n_modes_argument(self):
        traj = np.random.randn(100, 20, 3)
        modes = compute_rouse_modes(traj, n_modes=5)
        assert modes.shape == (100, 5, 3)

    def test_mode_0_is_com(self):
        """Mode p=0 should equal (1/N) * Σ r_i = COM / 1 (up to factor of N)."""
        N = 10
        positions = np.random.randn(50, N, 3)
        modes = compute_rouse_modes(positions)
        com = positions.mean(axis=1)  # (T, 3)
        # X_0 = (1/N) Σ r_i · cos(0) = (1/N) Σ r_i = COM
        np.testing.assert_allclose(modes[:, 0, :], com, atol=1e-10)

    def test_linearity(self):
        """Projection must be linear."""
        traj_a = np.random.randn(50, 10, 3)
        traj_b = np.random.randn(50, 10, 3)
        modes_sum = compute_rouse_modes(traj_a + traj_b)
        modes_a_plus_b = compute_rouse_modes(traj_a) + compute_rouse_modes(traj_b)
        np.testing.assert_allclose(modes_sum, modes_a_plus_b, atol=1e-10)


class TestModeAutocorrelation:

    def test_output_shape(self):
        modes = np.random.randn(200, 10, 3)
        corr = compute_mode_autocorrelation(modes, max_lag=50)
        assert corr.shape == (50, 10)

    def test_zero_lag_is_one(self):
        """Normalised autocorrelation at lag 0 must be 1 for all modes."""
        modes = np.random.randn(200, 5, 3)
        corr = compute_mode_autocorrelation(modes)
        np.testing.assert_allclose(corr[0], 1.0, atol=1e-10)

    def test_decays_with_lag(self):
        """Autocorrelation should generally decrease with lag for a random signal."""
        rng = np.random.default_rng(42)
        modes = rng.standard_normal((500, 3, 3))  # white noise
        corr = compute_mode_autocorrelation(modes, max_lag=100)
        # White noise: |C(τ)| << 1 for τ > 0
        assert corr[1:].max() < 0.3
