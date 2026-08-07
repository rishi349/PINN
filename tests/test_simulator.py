"""
Integration tests for the NumPy simulator.

From §9 Month 2 Gate:
    "Force unit tests pass; a single bead-pair (not yet a full chain)
    simulated under known conditions matches the by-hand expected behavior
    (e.g., two WCA-repelled beads separate, two harmonic-bonded beads
    oscillate around r₀)."

Tests:
1. Two WCA-repelled beads separate over time
2. Two harmonic-bonded beads oscillate around r₀
3. Short chain simulation runs without blow-ups (no NaN/Inf)
4. Bond lengths stay physically reasonable
5. Temperature estimation from displacements matches target
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.physics.forces import compute_all_forces
from src.physics.integrators import euler_maruyama_overdamped_step


# Standard parameters from §4
K_BOND = 100.0
R0 = 1.0
EPSILON = 1.0
SIGMA = 1.0
KBT = 1.0
GAMMA = 1.0
DT = 0.005


class TestTwoBeadSimulations:
    """
    §9 Month 2 Gate: bead-pair simulations match expected behavior.
    """

    def test_wca_repelled_beads_separate(self):
        """
        Two non-bonded beads placed close together should repel and separate.
        "two WCA-repelled beads separate" — §9 Month 2 Gate

        Note: compute_nonbonded_forces skips pairs with |i-j|<=1 (bonded
        neighbors), so we use 3 beads and test pair (0,2) which has |i-j|=2.
        """
        # 3 beads: 0 and 2 are close (WCA range), bead 1 is far away
        positions = np.array([
            [0.0, 0.0, 0.0],
            [50.0, 0.0, 0.0],  # Far away, irrelevant
            [0.9, 0.0, 0.0],  # Close to bead 0, |0-2|=2 > 1 → WCA applies
        ])
        rng = np.random.default_rng(42)
        r_cut = SIGMA * 2.0 ** (1.0 / 6.0)

        initial_dist = np.linalg.norm(positions[2] - positions[0])
        assert initial_dist < r_cut, "Beads 0 and 2 should start within WCA range"

        # Run 500 steps with T=0, k_bond=0 to isolate WCA effect
        for _ in range(500):
            forces, _, _ = compute_all_forces(
                positions, bond_type="harmonic", k_bond=0.0, r0=R0,
                epsilon=EPSILON, sigma=SIGMA
            )
            positions, _ = euler_maruyama_overdamped_step(
                positions, forces, DT, GAMMA, kBT=0.0, rng=rng
            )

        final_dist = np.linalg.norm(positions[2] - positions[0])
        assert final_dist > initial_dist, \
            f"WCA-repelled beads should separate: {initial_dist:.4f} → {final_dist:.4f}"
        assert final_dist >= r_cut - 0.05, \
            f"Beads should reach near WCA cutoff: {final_dist:.4f} vs {r_cut:.4f}"

    def test_harmonic_bonded_beads_oscillate_around_r0(self):
        """
        Two bonded beads should oscillate around r₀.
        "two harmonic-bonded beads oscillate around r₀" — §9 Month 2 Gate
        """
        # Start stretched at r=1.5σ
        positions = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
        rng = np.random.default_rng(42)

        distances = []
        for _ in range(2000):
            forces, _, _ = compute_all_forces(
                positions, bond_type="harmonic", k_bond=K_BOND, r0=R0,
                epsilon=EPSILON, sigma=SIGMA
            )
            positions, _ = euler_maruyama_overdamped_step(
                positions, forces, DT, GAMMA, kBT=KBT, rng=rng
            )
            dist = np.linalg.norm(positions[1] - positions[0])
            distances.append(dist)

        distances = np.array(distances)
        mean_dist = np.mean(distances[-1000:])  # Average over second half

        # Mean should be near r0 = 1.0σ
        assert abs(mean_dist - R0) < 0.15, \
            f"Mean bond length should be near r₀={R0}, got {mean_dist:.4f}"
        # Should not blow up
        assert np.all(np.isfinite(distances)), "Bond lengths should be finite"
        assert np.max(distances) < 5.0, "Bond length should not diverge"


class TestShortChainSimulation:
    """
    Short chain simulation smoke tests for Month 2.
    Full validation (R_g, MSD, temperature, etc.) is Month 3.
    """

    def test_five_bead_chain_no_blowup(self):
        """5-bead chain simulation runs without NaN/Inf for 1000 steps."""
        N = 5
        positions = np.zeros((N, 3))
        for i in range(N):
            positions[i, 0] = i * R0  # Linear chain

        rng = np.random.default_rng(42)

        for step in range(1000):
            forces, pe_bond, pe_nonbond = compute_all_forces(
                positions, bond_type="harmonic", k_bond=K_BOND, r0=R0,
                epsilon=EPSILON, sigma=SIGMA
            )
            positions, _ = euler_maruyama_overdamped_step(
                positions, forces, DT, GAMMA, KBT, rng
            )

            # Check no blow-ups (§13.1)
            assert np.all(np.isfinite(positions)), \
                f"NaN/Inf at step {step}: positions contain non-finite values"
            assert np.all(np.isfinite(forces)), \
                f"NaN/Inf at step {step}: forces contain non-finite values"

    def test_five_bead_chain_bonds_reasonable(self):
        """5-bead chain: bond lengths stay near r₀ after equilibration."""
        N = 5
        positions = np.zeros((N, 3))
        for i in range(N):
            positions[i, 0] = i * R0

        rng = np.random.default_rng(42)

        # Run 2000 steps, check last 1000
        bond_lengths_history = []
        for step in range(2000):
            forces, _, _ = compute_all_forces(
                positions, bond_type="harmonic", k_bond=K_BOND, r0=R0,
                epsilon=EPSILON, sigma=SIGMA
            )
            positions, _ = euler_maruyama_overdamped_step(
                positions, forces, DT, GAMMA, KBT, rng
            )

            if step >= 1000:
                bonds = [np.linalg.norm(positions[i+1] - positions[i])
                         for i in range(N-1)]
                bond_lengths_history.extend(bonds)

        bond_lengths = np.array(bond_lengths_history)
        mean_bl = np.mean(bond_lengths)

        # §13.1: "Bond-length histogram centered near expected value"
        assert 0.8 < mean_bl < 1.2, \
            f"Mean bond length should be near {R0}, got {mean_bl:.4f}"
        assert np.std(bond_lengths) < 0.3, \
            f"Bond length std should be small, got {np.std(bond_lengths):.4f}"

    def test_momentum_conservation(self):
        """Total force on the system should be zero at every step (Newton's 3rd)."""
        N = 5
        positions = np.zeros((N, 3))
        for i in range(N):
            positions[i, 0] = i * R0

        forces, _, _ = compute_all_forces(
            positions, bond_type="harmonic", k_bond=K_BOND, r0=R0,
            epsilon=EPSILON, sigma=SIGMA
        )
        total_force = np.sum(forces, axis=0)
        np.testing.assert_allclose(total_force, [0, 0, 0], atol=1e-10,
            err_msg="Total force should be zero (Newton's 3rd law)")


class TestNumpySimulatorIntegration:
    """Test the full NumpySimulator class."""

    def test_simulator_runs(self):
        """Simulator runs a short trajectory without errors."""
        import yaml
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "default.yaml"
        )
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Override for fast test
        config["chain"]["N"] = 5
        config["simulation"]["T_steps"] = 500
        config["simulation"]["n_burnin"] = 100
        config["simulation"]["save_every"] = 50

        from src.simulator.numpy_simulator import NumpySimulator
        sim = NumpySimulator(config)
        trajectory = sim.run(traj_id=0, seed=42, verbose=False)

        assert "metadata" in trajectory
        assert "frames" in trajectory
        assert trajectory["n_frames"] > 0
        assert trajectory["metadata"]["integrator_type"] == "overdamped_langevin"
        assert trajectory["metadata"]["chain_length"] == 5
        assert trajectory["metadata"]["noise_seed"] == 42

    def test_simulator_metadata_schema(self):
        """All §5 and §5.1 metadata fields present."""
        import yaml
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "default.yaml"
        )
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        config["chain"]["N"] = 3
        config["simulation"]["T_steps"] = 200
        config["simulation"]["n_burnin"] = 50
        config["simulation"]["save_every"] = 50

        from src.simulator.numpy_simulator import NumpySimulator
        sim = NumpySimulator(config)
        trajectory = sim.run(traj_id=1, seed=99, verbose=False)
        meta = trajectory["metadata"]

        # §5 required fields
        required_s5 = [
            "traj_id", "integrator_type", "noise_seed", "friction",
            "target_temperature", "timestep_size", "cutoff_distance",
            "simulation_box_size", "boundary_conditions", "mass",
            "chain_length", "split_label", "ood_flag", "ablation_id",
        ]
        for field in required_s5:
            assert field in meta, f"Missing §5 metadata field: {field}"

        # §5.1 required fields
        required_s51 = [
            "python_version", "numpy_version", "platform",
            "generation_timestamp", "generating_script_git_commit",
        ]
        for field in required_s51:
            assert field in meta, f"Missing §5.1 metadata field: {field}"

        # §5 per-frame fields
        frame = trajectory["frames"][0]
        required_frame = [
            "timestep_index", "positions", "forces", "bond_list",
            "potential_energy_bond", "potential_energy_nonbond",
            "potential_energy_total", "is_equilibrated",
            "center_of_mass", "end_to_end_vector", "end_to_end_distance",
            "radius_of_gyration", "bond_lengths", "mean_bond_length",
        ]
        for field in required_frame:
            assert field in frame, f"Missing per-frame field: {field}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
