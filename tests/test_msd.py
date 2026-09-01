"""Tests for MSD analysis (src/evaluation/msd.py)."""

import numpy as np
import pytest
from src.evaluation.msd import (
    compute_msd_g1,
    compute_msd_g2,
    compute_msd_g3,
    fit_msd_exponent,
    estimate_diffusion_coefficient,
    full_msd_analysis,
)


def make_brownian_chain(T: int = 1000, N: int = 10, D: float = 0.1, dt: float = 0.01, seed: int = 0) -> np.ndarray:
    """Independent Brownian walkers (no spring coupling) for known-answer tests.
    Each bead: Δr ~ N(0, sqrt(2*D*dt)).
    """
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, np.sqrt(2 * D * dt), size=(T - 1, N, 3))
    positions = np.zeros((T, N, 3))
    positions[1:] = steps.cumsum(axis=0)
    return positions


class TestComputeMsdG1:

    def test_output_shape_averaged(self):
        traj = np.random.randn(200, 10, 3)
        lags, g1 = compute_msd_g1(traj, max_lag=50)
        assert lags.shape == (50,)
        assert g1.shape == (50,)

    def test_output_shape_per_bead(self):
        traj = np.random.randn(200, 10, 3)
        lags, g1 = compute_msd_g1(traj, max_lag=50, average_over_beads=False)
        assert g1.shape == (50, 10)

    def test_monotonically_increasing_for_diffusion(self):
        """For a pure random walk, g1 should be monotonically increasing on average."""
        traj = make_brownian_chain(T=500, N=5, D=0.1, dt=0.01)
        lags, g1 = compute_msd_g1(traj, max_lag=100)
        # Check it's generally increasing (allow small noise fluctuations)
        assert g1[-1] > g1[0]

    def test_linear_scaling_for_diffusion(self):
        """For Brownian motion, g1 ~ 2*dim*D*t, so alpha ≈ 1.0."""
        D = 0.05
        dt = 0.01
        traj = make_brownian_chain(T=2000, N=8, D=D, dt=dt, seed=42)
        lags, g1 = compute_msd_g1(traj, max_lag=500)
        alpha, _ = fit_msd_exponent(g1, lags)
        assert np.isfinite(alpha)
        assert abs(alpha - 1.0) < 0.25, f"Expected alpha~1.0 for diffusion, got {alpha:.3f}"

    def test_zero_for_stationary_trajectory(self):
        """If the chain doesn't move, g1 should be 0."""
        traj = np.ones((100, 10, 3)) * 5.0   # constant positions
        lags, g1 = compute_msd_g1(traj, max_lag=20)
        np.testing.assert_allclose(g1, 0.0, atol=1e-12)


class TestComputeMsdG2:

    def test_output_shape(self):
        traj = np.random.randn(200, 10, 3)
        lags, g2 = compute_msd_g2(traj, max_lag=50)
        assert lags.shape == (50,)
        assert g2.shape == (50,)

    def test_zero_for_rigid_translation(self):
        """If the chain translates rigidly (all beads move the same), g2=0."""
        N, T = 10, 100
        com_drift = np.cumsum(np.random.randn(T, 1, 3), axis=0)  # COM random walk
        positions = np.tile(com_drift, (1, N, 1))  # All beads at COM
        lags, g2 = compute_msd_g2(positions, max_lag=20)
        np.testing.assert_allclose(g2, 0.0, atol=1e-10)

    def test_nonnegative(self):
        traj = np.random.randn(200, 10, 3)
        _, g2 = compute_msd_g2(traj, max_lag=50)
        assert np.all(g2 >= 0)


class TestComputeMsdG3:

    def test_output_shape(self):
        traj = np.random.randn(200, 10, 3)
        lags, g3 = compute_msd_g3(traj, max_lag=50)
        assert lags.shape == (50,)
        assert g3.shape == (50,)

    def test_linear_for_diffusing_com(self):
        """g3 of pure Brownian chain COM should scale ~linearly."""
        D = 0.1 / 10   # COM diffusion = D_monomer / N
        dt = 0.01
        traj = make_brownian_chain(T=2000, N=10, D=0.1, dt=dt, seed=7)
        lags, g3 = compute_msd_g3(traj, max_lag=500)
        alpha, _ = fit_msd_exponent(g3, lags)
        assert np.isfinite(alpha)
        # COM of 10 independent walkers should still scale ~linearly
        assert abs(alpha - 1.0) < 0.3, f"g3 exponent {alpha:.3f} far from 1.0"

    def test_zero_for_fixed_com(self):
        """If COM doesn't move (zero mean displacement), g3≈0."""
        N, T = 10, 100
        # Symmetric random walk: beads move equally in + and - directions
        positions = np.zeros((T, N, 3))
        _, g3 = compute_msd_g3(positions, max_lag=20)
        np.testing.assert_allclose(g3, 0.0, atol=1e-10)
