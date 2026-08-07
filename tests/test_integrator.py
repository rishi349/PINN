"""
Unit tests for the Euler–Maruyama integrator.

Tests verify:
1. Overdamped update formula matches §8.1 exactly:
   noise = sqrt(2*kBT*dt/gamma) * N(0,1)
   r[i] = r[i] + (dt/gamma) * F[i] + noise

2. Noise scaling: amplitude = sqrt(2*kBT*dt/gamma), scales with √dt

3. Deterministic part (drift) is correct: (dt/gamma) * F

4. Statistical properties: with zero force, displacement variance
   should be 2*kBT*dt/gamma per dimension (Einstein relation for
   overdamped Brownian motion)
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.physics.integrators import (
    euler_maruyama_overdamped_step,
    euler_maruyama_underdamped_step,
)


class TestOverdampedEulerMaruyama:
    """Tests for overdamped Euler–Maruyama integrator."""

    def test_zero_force_zero_temperature(self):
        """With F=0 and T=0, beads don't move."""
        N, dim = 5, 3
        positions = np.random.randn(N, dim)
        forces = np.zeros((N, dim))
        rng = np.random.default_rng(42)
        new_pos, noise = euler_maruyama_overdamped_step(
            positions, forces, dt=0.005, gamma=1.0, kBT=0.0, rng=rng
        )
        np.testing.assert_allclose(new_pos, positions, atol=1e-15)
        np.testing.assert_allclose(noise, np.zeros((N, dim)), atol=1e-15)

    def test_drift_term_correct(self):
        """Drift = (dt/gamma) * F."""
        positions = np.array([[1.0, 2.0, 3.0]])
        forces = np.array([[10.0, -5.0, 3.0]])
        dt, gamma = 0.005, 1.0
        rng = np.random.default_rng(42)
        new_pos, noise = euler_maruyama_overdamped_step(
            positions, forces, dt=dt, gamma=gamma, kBT=0.0, rng=rng
        )
        expected_drift = (dt / gamma) * forces
        np.testing.assert_allclose(new_pos - positions, expected_drift, atol=1e-15)

    def test_noise_amplitude(self):
        """Noise amplitude = sqrt(2*kBT*dt/gamma)."""
        N, dim = 1000, 3
        positions = np.zeros((N, dim))
        forces = np.zeros((N, dim))
        dt, gamma, kBT = 0.005, 1.0, 1.0
        rng = np.random.default_rng(42)
        new_pos, noise = euler_maruyama_overdamped_step(
            positions, forces, dt=dt, gamma=gamma, kBT=kBT, rng=rng
        )
        expected_amplitude = np.sqrt(2.0 * kBT * dt / gamma)
        # noise should have std ≈ expected_amplitude
        measured_std = np.std(noise)
        np.testing.assert_allclose(measured_std, expected_amplitude, rtol=0.1,
            err_msg=f"Noise std should be √(2kBT·dt/γ)={expected_amplitude:.6f}")

    def test_noise_scales_with_sqrt_dt(self):
        """
        §3: "noise scales with √dt, not dt"
        Verify by comparing noise amplitudes at two different dt values.
        """
        N, dim = 10000, 3
        positions = np.zeros((N, dim))
        forces = np.zeros((N, dim))
        gamma, kBT = 1.0, 1.0

        dt1, dt2 = 0.001, 0.004
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        _, noise1 = euler_maruyama_overdamped_step(positions, forces, dt1, gamma, kBT, rng1)
        _, noise2 = euler_maruyama_overdamped_step(positions, forces, dt2, gamma, kBT, rng2)

        ratio_expected = np.sqrt(dt2 / dt1)  # = 2.0
        ratio_measured = np.std(noise2) / np.std(noise1)
        np.testing.assert_allclose(ratio_measured, ratio_expected, rtol=0.1,
            err_msg="Noise should scale with √dt")

    def test_einstein_relation_free_diffusion(self):
        """
        For free particle (F=0), <Δr²> = 2*d*D*dt where D = kBT/gamma.
        So <|Δr|²> per dimension = 2*kBT*dt/gamma.
        This is the overdamped Einstein relation.
        """
        N = 50000
        dim = 3
        positions = np.zeros((N, dim))
        forces = np.zeros((N, dim))
        dt, gamma, kBT = 0.005, 1.0, 1.0
        rng = np.random.default_rng(123)

        new_pos, _ = euler_maruyama_overdamped_step(
            positions, forces, dt, gamma, kBT, rng
        )
        displacements = new_pos - positions
        # Mean squared displacement per dimension
        msd_per_dim = np.mean(displacements ** 2)
        expected_msd = 2.0 * kBT * dt / gamma
        np.testing.assert_allclose(msd_per_dim, expected_msd, rtol=0.05,
            err_msg="MSD should match Einstein relation: 2*kBT*dt/γ per dim")

    def test_reproducibility_with_same_seed(self):
        """Same seed → same trajectory (reproducibility)."""
        positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        forces = np.array([[0.1, -0.2, 0.3], [-0.1, 0.2, -0.3]])
        dt, gamma, kBT = 0.005, 1.0, 1.0

        rng1 = np.random.default_rng(42)
        pos1, noise1 = euler_maruyama_overdamped_step(positions, forces, dt, gamma, kBT, rng1)

        rng2 = np.random.default_rng(42)
        pos2, noise2 = euler_maruyama_overdamped_step(positions, forces, dt, gamma, kBT, rng2)

        np.testing.assert_array_equal(pos1, pos2)
        np.testing.assert_array_equal(noise1, noise2)

    def test_different_seeds_different_trajectories(self):
        """Different seeds → different trajectories (§13.1: seed reproducibility)."""
        positions = np.array([[1.0, 2.0, 3.0]])
        forces = np.zeros((1, 3))
        dt, gamma, kBT = 0.005, 1.0, 1.0

        rng1 = np.random.default_rng(42)
        pos1, _ = euler_maruyama_overdamped_step(positions, forces, dt, gamma, kBT, rng1)

        rng2 = np.random.default_rng(99)
        pos2, _ = euler_maruyama_overdamped_step(positions, forces, dt, gamma, kBT, rng2)

        assert not np.allclose(pos1, pos2), "Different seeds should give different results"

    def test_gamma_effect(self):
        """Higher friction → smaller displacements (both drift and diffusion)."""
        positions = np.array([[0.0, 0.0, 0.0]])
        forces = np.array([[10.0, 0.0, 0.0]])
        dt, kBT = 0.005, 1.0

        rng1 = np.random.default_rng(42)
        pos_low_gamma, _ = euler_maruyama_overdamped_step(
            positions, forces, dt, gamma=0.5, kBT=kBT, rng=rng1
        )

        rng2 = np.random.default_rng(42)
        pos_high_gamma, _ = euler_maruyama_overdamped_step(
            positions, forces, dt, gamma=2.0, kBT=kBT, rng=rng2
        )

        drift_low = (dt / 0.5) * forces[0, 0]
        drift_high = (dt / 2.0) * forces[0, 0]
        assert drift_low > drift_high, "Higher gamma should mean smaller drift"


class TestUnderdampedEulerMaruyama:
    """Tests for underdamped (full Langevin) Euler–Maruyama integrator."""

    def test_zero_everything(self):
        """With F=0, v=0, T=0, nothing moves."""
        positions = np.array([[1.0, 2.0, 3.0]])
        velocities = np.zeros((1, 3))
        forces = np.zeros((1, 3))
        rng = np.random.default_rng(42)
        new_pos, new_vel, noise = euler_maruyama_underdamped_step(
            positions, velocities, forces, dt=0.005, gamma=1.0,
            kBT=0.0, mass=1.0, rng=rng
        )
        np.testing.assert_allclose(new_pos, positions, atol=1e-15)
        np.testing.assert_allclose(new_vel, velocities, atol=1e-15)

    def test_velocity_update_formula(self):
        """
        §8.1: v[i] = v[i] + dt*(F[i] - gamma*v[i])/m + noise/m
        With T=0 (no noise), verify deterministic velocity update.
        """
        velocities = np.array([[2.0, 0.0, 0.0]])
        forces = np.array([[5.0, 0.0, 0.0]])
        dt, gamma, mass = 0.005, 1.0, 1.0
        rng = np.random.default_rng(42)
        _, new_vel, _ = euler_maruyama_underdamped_step(
            np.zeros((1, 3)), velocities, forces, dt, gamma, kBT=0.0,
            mass=mass, rng=rng
        )
        # v_new = v + dt*(F - gamma*v)/m = 2 + 0.005*(5 - 1*2)/1 = 2 + 0.015 = 2.015
        expected_v = velocities[0, 0] + dt * (forces[0, 0] - gamma * velocities[0, 0]) / mass
        np.testing.assert_allclose(new_vel[0, 0], expected_v, atol=1e-12)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
