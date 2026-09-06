"""Tests for FDT temperature check (src/evaluation/temperature_check.py)."""

import numpy as np
import pytest
from src.evaluation.temperature_check import (
    estimate_temperature_from_displacements,
    check_fdt,
    check_fdt_rollout,
)


def make_langevin_trajectory(
    T: int = 2000,
    N: int = 10,
    T_target: float = 1.0,
    dt: float = 0.001,
    gamma: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """Ideal overdamped Langevin (pure noise, no force) for FDT tests.
    Δr_i = sqrt(2 * T * dt / gamma) * N(0, 1)
    """
    rng = np.random.default_rng(seed)
    noise_std = np.sqrt(2.0 * T_target * dt / gamma)
    steps = rng.normal(0, noise_std, size=(T - 1, N, 3))
    positions = np.zeros((T, N, 3))
    positions[1:] = steps.cumsum(axis=0)
    return positions


class TestEstimateTemperatureFromDisplacements:

    def test_recovers_target_temperature(self):
        """Ideal Langevin trajectory should give T≈1.0."""
        T_target = 1.0
        dt = 0.001
        traj = make_langevin_trajectory(T=5000, N=20, T_target=T_target, dt=dt)
        T_meas, T_std = estimate_temperature_from_displacements(traj, dt=dt)
        assert abs(T_meas - T_target) / T_target < 0.05, \
            f"T_measured={T_meas:.4f}, expected ~{T_target}"

    def test_recovers_different_temperature(self):
        """Should work for T != 1.0."""
        T_target = 2.0
        dt = 0.001
        traj = make_langevin_trajectory(T=5000, N=20, T_target=T_target, dt=dt, seed=1)
        T_meas, _ = estimate_temperature_from_displacements(traj, dt=dt)
        assert abs(T_meas - T_target) / T_target < 0.10

    def test_returns_tuple(self):
        traj = np.random.randn(100, 10, 3)
        result = estimate_temperature_from_displacements(traj)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_std_small_for_homogeneous_chain(self):
        """For i.i.d. beads, spatial std of T should be small."""
        traj = make_langevin_trajectory(T=3000, N=30)
        T_meas, T_std = estimate_temperature_from_displacements(traj)
        # T_std should be much smaller than T_meas
        assert T_std < T_meas * 0.15

    def test_stationary_trajectory_gives_zero_temperature(self):
        """A trajectory with no motion has T=0."""
        traj = np.ones((100, 10, 3)) * 5.0
        T_meas, _ = estimate_temperature_from_displacements(traj)
        assert T_meas == pytest.approx(0.0, abs=1e-12)


class TestCheckFdt:

    def test_passes_for_correct_temperature(self):
        """Ideal Langevin trajectory at T=1.0 should pass the FDT check."""
        traj = make_langevin_trajectory(T=5000, N=20, T_target=1.0, dt=0.001, seed=42)
        result = check_fdt(traj, T_target=1.0, dt=0.001, rtol=0.05)
        assert result['passes'], (
            f"FDT check failed: T_measured={result['T_measured']:.4f}, "
            f"rel_error={result['rel_error']:.4f}"
        )

    def test_fails_for_wrong_temperature(self):
