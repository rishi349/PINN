"""
HOOMD-blue simulator wrapper for polymer chain dynamics.

Wraps HOOMD-blue's Python API to implement §8.1 using the `hoomd.md.methods.Brownian`
integrator for overdamped Langevin dynamics (§2).

Same config interface and output schema as the NumPy simulator for cross-checking (§9 Month 3).

NOTE: This module requires HOOMD-blue to be installed via conda-forge.
If HOOMD is not available, the NumPy simulator can be used as a standalone alternative.
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

try:
    import hoomd
    import hoomd.md
    HOOMD_AVAILABLE = True
    HOOMD_VERSION = hoomd.version.version
except ImportError:
    HOOMD_AVAILABLE = False
    HOOMD_VERSION = "not_installed"


class HoomdSimulator:
    """
    HOOMD-blue based simulator for overdamped Langevin dynamics.

    Uses `hoomd.md.methods.Brownian` for overdamped (Brownian) dynamics (§2).
    Implements the same interface and output schema as NumpySimulator.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize from config dictionary.

        Parameters
        ----------
        config : dict
            Configuration matching configs/default.yaml schema.
        """
        if not HOOMD_AVAILABLE:
            raise ImportError(
                "HOOMD-blue is not installed. Install via conda-forge:\n"
                "  conda install -c conda-forge hoomd\n"
                "Or use NumpySimulator as a standalone alternative."
            )

        # Store all parameters (same as NumpySimulator)
        self.N = config["chain"]["N"]
        self.dim = config["chain"]["dimension"]
        self.bond_type = config["bond"]["type"]
        self.k_bond = config["bond"]["k_bond"]
        self.r0 = config["bond"]["r0"]
        self.k_fene = config["fene"]["k"]
        self.R0_fene = config["fene"]["R0"]
        self.epsilon = config["wca"]["epsilon"]
        self.sigma = config["wca"]["sigma"]
        self.kBT = config["thermostat"]["kBT"]
        self.gamma = config["thermostat"]["gamma"]
        self.mass = config["thermostat"]["mass"]
        self.dt = config["integrator"]["dt"]
        self.T_steps = config["simulation"]["T_steps"]
        self.n_burnin = config["simulation"]["n_burnin"]
        self.save_every = config["simulation"]["save_every"]
        self.init_method = config["initialization"]["method"]
        self.init_spacing = config["initialization"]["spacing"]
        self.box_L = config["box"]["L"]
        self.base_seed = config["seed"]
        self.git_commit = self._get_git_commit()

    def _create_initial_snapshot(self, seed: int) -> "hoomd.Snapshot":
        """
        Create HOOMD snapshot with initial polymer chain configuration.

        Implements §8.1: "Initialize polymer coordinates r[1..N]
        as a stretched or random walk chain"
        """
        rng = np.random.default_rng(seed)

        # Generate initial positions
        positions = np.zeros((self.N, 3))
        if self.init_method == "linear":
            positions[:, 0] = np.arange(self.N) * self.init_spacing
        elif self.init_method == "random_walk":
            steps = rng.standard_normal((self.N - 1, 3))
            steps = (steps / np.linalg.norm(steps, axis=1, keepdims=True)) * self.init_spacing
            positions[1:] = np.cumsum(steps, axis=0)

        # Center in box
        positions -= np.mean(positions, axis=0)

        # Create HOOMD snapshot
        snapshot = hoomd.Snapshot()
        snapshot.particles.N = self.N
        snapshot.particles.types = ["A"]
        snapshot.particles.position[:] = positions
        snapshot.particles.typeid[:] = 0
        snapshot.particles.mass[:] = self.mass
        snapshot.configuration.box = [self.box_L, self.box_L, self.box_L, 0, 0, 0]

        # Define bonds (linear chain: i--(i+1))
        snapshot.bonds.N = self.N - 1
        snapshot.bonds.types = ["polymer"]
        snapshot.bonds.typeid[:] = 0
        snapshot.bonds.group[:] = [[i, i + 1] for i in range(self.N - 1)]

        return snapshot

    def _setup_simulation(self, snapshot: "hoomd.Snapshot", seed: int) -> "hoomd.Simulation":
        """
        Set up HOOMD simulation with forces and integrator.

        §2: Uses hoomd.md.methods.Brownian for overdamped dynamics.
        §4: Harmonic bond + WCA pair potential.
        """
        try:
            device = hoomd.device.GPU()
        except Exception:
            device = hoomd.device.CPU()
            
        sim = hoomd.Simulation(device=device, seed=seed)
        sim.create_state_from_snapshot(snapshot)

        # === Bond potential ===
        if self.bond_type == "harmonic":
            # §4 footnote: k_bond=100, r0=1.0
            harmonic = hoomd.md.bond.Harmonic()
            harmonic.params["polymer"] = dict(k=self.k_bond, r0=self.r0)
        elif self.bond_type == "fene":
            # §4: k=30, R0=1.5
            fene = hoomd.md.bond.FENEWCA()
            fene.params["polymer"] = dict(
                k=self.k_fene, r0=self.R0_fene,
                epsilon=self.epsilon, sigma=self.sigma, delta=0.0
            )

        # === WCA pair potential (excluded volume) ===
        # §4: Repulsive LJ cut at 2^(1/6)σ
        r_cut = self.sigma * 2.0 ** (1.0 / 6.0)
        try:
            nlist = hoomd.md.nlist.Tree(buffer=0.4)
        except AttributeError:
            nlist = hoomd.md.nlist.Cell(buffer=0.4)
        lj = hoomd.md.pair.LJ(nlist=nlist, default_r_cut=r_cut)
        lj.params[("A", "A")] = dict(epsilon=self.epsilon, sigma=self.sigma)
        lj.mode = "shift"  # Shift to zero at cutoff (= WCA)

        # === Brownian integrator (§2: overdamped Langevin) ===
        # §2: "hoomd.md.methods.Brownian(filter=..., kT=...)"
        # §4: gamma directly set (not alpha)
        brownian = hoomd.md.methods.Brownian(
            filter=hoomd.filter.All(),
            kT=self.kBT,
        )
        # Set gamma per particle type
        brownian.gamma["A"] = self.gamma

        # Assemble integrator
        integrator = hoomd.md.Integrator(
            dt=self.dt,
            methods=[brownian],
            forces=[harmonic if self.bond_type == "harmonic" else fene, lj],
        )
        sim.operations.integrator = integrator

        return sim

    def _compute_observables(self, positions: np.ndarray) -> Dict[str, Any]:
        """Compute physical observables (same as NumpySimulator)."""
        com = np.mean(positions, axis=0)
        end_to_end_vec = positions[-1] - positions[0]
        end_to_end_dist = np.linalg.norm(end_to_end_vec)
        displacements = positions - com
        Rg_sq = np.mean(np.sum(displacements**2, axis=1))
        Rg = np.sqrt(Rg_sq)
        bond_vectors = positions[1:] - positions[:-1]
        bond_lengths = np.linalg.norm(bond_vectors, axis=1)
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
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5
            )
            return result.stdout.strip() if result.returncode == 0 else "unknown"
        except Exception:
            return "unknown"

    def run(
        self,
        traj_id: int = 0,
        seed: Optional[int] = None,
        output_dir: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run a full simulation trajectory using HOOMD-blue.

        Parameters match NumpySimulator.run() for interchangeability.
        """
        if seed is None:
            seed = self.base_seed + traj_id

        # Create initial state
        snapshot = self._create_initial_snapshot(seed)
        sim = self._setup_simulation(snapshot, seed)

        # Metadata (§5, §5.1)
        r_cut = self.sigma * 2.0 ** (1.0 / 6.0)
        metadata = {
            "traj_id": traj_id,
            "integrator_type": "overdamped_langevin",
            "noise_seed": seed,
            "friction": self.gamma,
            "target_temperature": self.kBT,
            "timestep_size": self.dt,
            "cutoff_distance": float(r_cut),
            "simulation_box_size": self.box_L,
            "boundary_conditions": "periodic",
            "mass": self.mass,
            "chain_length": self.N,
            "bond_type": self.bond_type,
            "k_bond": self.k_bond,
            "r0": self.r0,
            "wca_epsilon": self.epsilon,
            "wca_sigma": self.sigma,
            "split_label": "",
            "ood_flag": False,
            "ablation_id": "",
            # §5.1
            "simulator": "hoomd",
            "hoomd_version": HOOMD_VERSION,
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "generation_timestamp": datetime.now(timezone.utc).isoformat(),
            "generating_script_git_commit": self.git_commit,
        }

        # Storage
        frames = []
        bond_list = [(i, i + 1) for i in range(self.N - 1)]

        if verbose:
            print(f"[Traj {traj_id}] HOOMD simulation: N={self.N}, "
                  f"T_steps={self.T_steps}, dt={self.dt}, seed={seed}")

        # 1. Run burn-in phase in one fast C++/GPU call
        if self.n_burnin > 0:
            sim.run(self.n_burnin)

        # 2. Advance simulation in blocks of save_every
        remaining_steps = self.T_steps - self.n_burnin
        n_frames_to_save = max(0, remaining_steps // self.save_every)
        progress_interval = max(1, n_frames_to_save // 10)

        for frame_idx in range(1, n_frames_to_save + 1):
            sim.run(self.save_every)
            current_t = self.n_burnin + frame_idx * self.save_every

            # Fetch snapshot and compute observables once per frame
            snap = sim.state.get_snapshot()
            positions = np.array(snap.particles.position[:self.N])
            observables = self._compute_observables(positions)

            frame = {
                "timestep_index": current_t,
                "positions": positions.tolist(),
                "forces": [],  # HOOMD doesn't easily expose per-particle forces
                "bond_list": bond_list,
                "potential_energy_bond": 0.0,  # Would need logger
                "potential_energy_nonbond": 0.0,
                "potential_energy_total": 0.0,
                "is_equilibrated": True,
                **observables,
            }
            frames.append(frame)

            # Reuse computed observables for verbose logging without extra GPU sync
            if verbose and (frame_idx % progress_interval == 0 or frame_idx == n_frames_to_save):
                print(f"  Step {current_t}/{self.T_steps}: "
                      f"R_g={observables['radius_of_gyration']:.4f}, "
                      f"mean_bond={observables['mean_bond_length']:.4f}")

        trajectory = {
            "metadata": metadata,
            "frames": frames,
            "n_frames": len(frames),
        }

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, f"trajectory_{traj_id:04d}.json")
            raw_json = json.dumps(trajectory)
            trajectory["metadata"]["sha256_checksum"] = hashlib.sha256(
                raw_json.encode("utf-8")
            ).hexdigest()
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(trajectory, f)

        if verbose:
            print(f"[Traj {traj_id}] Done. Saved {len(frames)} frames.")

        return trajectory
