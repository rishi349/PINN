"""
Pure NumPy Euler–Maruyama simulator for polymer chain dynamics.

This implements the full §8.1 pseudocode as an independent cross-check
of HOOMD-blue's output (§9 Month 3). On a tiny system, both should
produce statistically similar results.

Main simulation loop (§8.1):
    Input: N, T_steps, dt, gamma, T, k_bond, r0, epsilon, sigma,
           n_burnin, save_every, seed

    Initialize random seed
    Initialize polymer coordinates r[1..N] as stretched or random walk chain

    for t in 1..T_steps:
        F = zeros_like(r)
        # bonded forces
        for i in 1..N-1:
            r_ij = r[i+1] - r[i]
            F_bond = bond_force(r_ij, k_bond, r0)
            F[i] += F_bond
            F[i+1] -= F_bond
        # excluded-volume forces
        for all nonbonded pairs (i,j) with |i-j| > 1:
            if distance(r[i], r[j]) < cutoff:
                F_rep = wca_force(r[i]-r[j], epsilon, sigma)
                F[i] += F_rep
                F[j] -= F_rep
        # overdamped stochastic dynamics update
        noise = sqrt(2*kBT*dt/gamma) * Normal(0,1)
        r[i] = r[i] + (dt/gamma) * F[i] + noise

        if t > n_burnin and t mod save_every == 0:
            save(r, F, potential_energy, t, seed, ...)
"""

import numpy as np
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

import yaml

from src.physics.forces import compute_all_forces
from src.physics.integrators import euler_maruyama_overdamped_step


class NumpySimulator:
    """
    Pure NumPy simulator for overdamped Langevin dynamics of a bead-spring
    polymer chain.

    Implements §8.1 pseudocode exactly, with all §5 and §5.1 metadata.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize simulator from a config dictionary (loaded from YAML).

        Parameters
        ----------
        config : dict
            Configuration dictionary matching configs/default.yaml schema.
        """
        # === Chain parameters ===
        self.N = config["chain"]["N"]
        self.dim = config["chain"]["dimension"]

        # === Bond parameters ===
        self.bond_type = config["bond"]["type"]
        self.k_bond = config["bond"]["k_bond"]
        self.r0 = config["bond"]["r0"]
        self.k_fene = config["fene"]["k"]
        self.R0_fene = config["fene"]["R0"]

        # === WCA parameters ===
        self.epsilon = config["wca"]["epsilon"]
        self.sigma = config["wca"]["sigma"]

        # === Thermostat ===
        self.kBT = config["thermostat"]["kBT"]
        self.gamma = config["thermostat"]["gamma"]
        self.mass = config["thermostat"]["mass"]

        # === Integration ===
        self.integrator_type = config["integrator"]["type"]
        self.dt = config["integrator"]["dt"]

        # === Simulation length ===
        self.T_steps = config["simulation"]["T_steps"]
        self.n_burnin = config["simulation"]["n_burnin"]
        self.save_every = config["simulation"]["save_every"]

        # === Initialization ===
        self.init_method = config["initialization"]["method"]
        self.init_spacing = config["initialization"]["spacing"]

        # === Box ===
        self.box_L = config["box"]["L"]
        self.periodic = config["box"]["periodic"]

        # === Base seed ===
        self.base_seed = config["seed"]

    def initialize_chain(self, rng: np.random.Generator) -> np.ndarray:
        """
        Initialize polymer chain coordinates.

        From §8.1: "Initialize polymer coordinates r[1..N] as a stretched
        or random walk chain"

        Parameters
        ----------
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        positions : np.ndarray, shape (N, dim)
            Initial bead positions.
        """
        positions = np.zeros((self.N, self.dim))

        if self.init_method == "linear":
            # Stretched linear chain along x-axis
            for i in range(self.N):
                positions[i, 0] = i * self.init_spacing
        elif self.init_method == "random_walk":
            # Random walk with step size = init_spacing
            for i in range(1, self.N):
                step = rng.standard_normal(self.dim)
                step = step / np.linalg.norm(step) * self.init_spacing
                positions[i] = positions[i - 1] + step
        else:
            raise ValueError(f"Unknown initialization method: {self.init_method}")

        # Center chain in the box
        center = np.mean(positions, axis=0)
        positions -= center

        return positions

    def compute_observables(self, positions: np.ndarray) -> Dict[str, Any]:
        """
        Compute physical observables for a given configuration.

        From §5 per-frame fields and §12 metrics checklist.

        Parameters
        ----------
        positions : np.ndarray, shape (N, dim)
            Bead positions.

        Returns
        -------
        dict
            Observable values.
        """
        # Center of mass (§5)
        com = np.mean(positions, axis=0)

        # End-to-end vector (§5)
        end_to_end_vec = positions[-1] - positions[0]
        end_to_end_dist = np.linalg.norm(end_to_end_vec)

        # Radius of gyration (§12: R_g distribution)
        # R_g² = (1/N) * Σ|r_i - r_com|²
        displacements = positions - com
        Rg_sq = np.mean(np.sum(displacements**2, axis=1))
        Rg = np.sqrt(Rg_sq)

        # Bond lengths (§13.1: bond-length histogram)
        bond_lengths = np.array([
            np.linalg.norm(positions[i + 1] - positions[i])
            for i in range(self.N - 1)
        ])

        return {
            "center_of_mass": com.tolist(),
            "end_to_end_vector": end_to_end_vec.tolist(),
            "end_to_end_distance": float(end_to_end_dist),
            "radius_of_gyration": float(Rg),
            "Rg_squared": float(Rg_sq),
            "bond_lengths": bond_lengths.tolist(),
            "mean_bond_length": float(np.mean(bond_lengths)),
            "max_bond_length": float(np.max(bond_lengths)),
            "min_bond_length": float(np.min(bond_lengths)),
        }

    def _get_git_commit(self) -> str:
        """Get current git commit hash for §5.1 provenance tracking."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5
            )
            return result.stdout.strip() if result.returncode == 0 else "unknown"
        except Exception:
            return "unknown"

    def _build_trajectory_metadata(
        self, traj_id: int, seed: int
    ) -> Dict[str, Any]:
        """
        Build per-trajectory metadata following §5 and §5.1 schemas.

        §5 Per-trajectory metadata:
            traj_id, integrator_type, noise_seed, friction, target_temperature,
            timestep_size, cutoff_distance, simulation_box_size, boundary_conditions,
            mass, chain_length, split_label, ood_flag, ablation_id

        §5.1 Reproducibility fields:
            hoomd_version, python_version, numpy_version, platform,
            sha256_checksum, generation_timestamp, generating_script_git_commit
        """
        r_cut = self.sigma * 2.0 ** (1.0 / 6.0)

        return {
            # §5 fields
            "traj_id": traj_id,
            "integrator_type": "overdamped_langevin",
            "noise_seed": seed,
            "friction": self.gamma,
            "target_temperature": self.kBT,
            "timestep_size": self.dt,
            "cutoff_distance": float(r_cut),
            "simulation_box_size": self.box_L,
            "boundary_conditions": "periodic" if self.periodic else "none",
            "mass": self.mass,
            "chain_length": self.N,
            "bond_type": self.bond_type,
            "k_bond": self.k_bond,
            "r0": self.r0,
            "wca_epsilon": self.epsilon,
            "wca_sigma": self.sigma,
            "split_label": "",  # Assigned later during dataset engineering
            "ood_flag": False,
            "ablation_id": "",
            # §5.1 software environment
            "simulator": "numpy",
            "hoomd_version": "N/A",
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            # §5.1 data provenance
            "generation_timestamp": datetime.now(timezone.utc).isoformat(),
            "generating_script_git_commit": self._get_git_commit(),
            # Checksum added after file is written
        }

    def run(
        self,
        traj_id: int = 0,
        seed: Optional[int] = None,
        output_dir: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run a full simulation trajectory.

        Implements the complete §8.1 pseudocode main loop.

        Parameters
        ----------
        traj_id : int
            Trajectory identifier.
        seed : int, optional
            Random seed. If None, uses base_seed + traj_id.
        output_dir : str, optional
            Directory to save trajectory data.
        verbose : bool
            Print progress messages.

        Returns
        -------
        trajectory : dict
            Dictionary containing all trajectory data and metadata per §5/§5.1.
        """
        # === Initialize random seed (§8.1: "Initialize random seed") ===
        if seed is None:
            seed = self.base_seed + traj_id
        rng = np.random.default_rng(seed)

        # === Initialize chain (§8.1: "Initialize polymer coordinates") ===
        positions = self.initialize_chain(rng)

        # === Build metadata (§5, §5.1) ===
        metadata = self._build_trajectory_metadata(traj_id, seed)

        # === Storage for saved frames (§5 per-frame fields) ===
        frames = []
        bond_list = [(i, i + 1) for i in range(self.N - 1)]

        if verbose:
            print(f"[Traj {traj_id}] Starting simulation: N={self.N}, "
                  f"T_steps={self.T_steps}, dt={self.dt}, seed={seed}")

        # === Main simulation loop (§8.1) ===
        for t in range(1, self.T_steps + 1):
            # Compute all forces (§8.1: bonded + excluded-volume)
            forces, pe_bond, pe_nonbond = compute_all_forces(
                positions,
                bond_type=self.bond_type,
                k_bond=self.k_bond,
                r0=self.r0,
                k_fene=self.k_fene,
                R0_fene=self.R0_fene,
                epsilon=self.epsilon,
                sigma=self.sigma,
            )

            # Overdamped dynamics update (§8.1)
            new_positions, noise = euler_maruyama_overdamped_step(
                positions, forces, self.dt, self.gamma, self.kBT, rng
            )

            positions = new_positions

            # §8.1: "if t > n_burnin and t mod save_every == 0: save(...)"
            if t > self.n_burnin and t % self.save_every == 0:
                observables = self.compute_observables(positions)

                frame = {
                    # §5 per-frame fields
                    "timestep_index": t,
                    "positions": positions.copy().tolist(),
                    "forces": forces.copy().tolist(),
                    "bond_list": bond_list,
                    "potential_energy_bond": float(pe_bond),
                    "potential_energy_nonbond": float(pe_nonbond),
                    "potential_energy_total": float(pe_bond + pe_nonbond),
                    "is_equilibrated": True,  # Post burn-in
                    # Observables
                    **observables,
                }
                frames.append(frame)

            # Progress reporting
            if verbose and t % (self.T_steps // 10) == 0:
                obs = self.compute_observables(positions)
                print(f"  Step {t}/{self.T_steps}: "
                      f"R_g={obs['radius_of_gyration']:.4f}, "
                      f"mean_bond={obs['mean_bond_length']:.4f}, "
                      f"PE={pe_bond + pe_nonbond:.4f}")

        # === Build trajectory output ===
        trajectory = {
            "metadata": metadata,
            "frames": frames,
            "n_frames": len(frames),
        }

        # === Save to disk if output_dir specified ===
        if output_dir is not None:
            self._save_trajectory(trajectory, output_dir)

        if verbose:
            print(f"[Traj {traj_id}] Done. Saved {len(frames)} frames.")

        return trajectory

    def _save_trajectory(
        self, trajectory: Dict[str, Any], output_dir: str
    ) -> str:
        """
        Save trajectory to disk and compute SHA256 checksum (§5.1).

        Checksum is stored in a sidecar file (.sha256) to avoid the
        circular dependency of embedding a hash inside the file it hashes.

        Parameters
        ----------
        trajectory : dict
            Trajectory data.
        output_dir : str
            Output directory.
            
        Returns
        -------
        filepath : str
            Path to saved file.
        """
        os.makedirs(output_dir, exist_ok=True)
        traj_id = trajectory["metadata"]["traj_id"]
        filepath = os.path.join(output_dir, f"trajectory_{traj_id:04d}.json")

        # Single write — final file contents
        with open(filepath, "w") as f:
            json.dump(trajectory, f)

        # §5.1: checksum of the final file, stored in sidecar
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        checksum = sha256.hexdigest()

        sidecar_path = filepath + ".sha256"
        with open(sidecar_path, "w") as f:
            f.write(checksum)

        trajectory["metadata"]["sha256_checksum"] = checksum

        return filepath


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
