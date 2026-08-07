#!/usr/bin/env python
"""
CLI entry point for running polymer chain simulations.

Usage:
    python scripts/run_simulation.py --config configs/default.yaml
    python scripts/run_simulation.py --config configs/default.yaml --n-trajectories 5 --output data/raw
    python scripts/run_simulation.py --config configs/default.yaml --simulator numpy --traj-id 0

One YAML config file per experiment from day one — don't hardcode
parameters in scripts (§9 Month 1).
"""

import argparse
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml


def main():
    parser = argparse.ArgumentParser(
        description="Run polymer chain Brownian dynamics simulation"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file (e.g., configs/default.yaml)"
    )
    parser.add_argument(
        "--simulator", type=str, default="numpy", choices=["numpy", "hoomd"],
        help="Simulator backend: 'numpy' (pure Python) or 'hoomd' (HOOMD-blue)"
    )
    parser.add_argument(
        "--n-trajectories", type=int, default=1,
        help="Number of trajectories to generate"
    )
    parser.add_argument(
        "--traj-id-start", type=int, default=0,
        help="Starting trajectory ID"
    )
    parser.add_argument(
        "--output", type=str, default="data/raw",
        help="Output directory for trajectory files"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override base random seed (default: from config)"
    )
    parser.add_argument(
        "--t-steps", type=int, default=None,
        help="Override total simulation timesteps (T_steps)"
    )
    parser.add_argument(
        "--n-burnin", type=int, default=None,
        help="Override burn-in timesteps (n_burnin)"
    )
    parser.add_argument(
        "--save-every", type=int, default=None,
        help="Override frame saving frequency (save_every)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output"
    )
    args = parser.parse_args()

    # Load config (§9 Month 1: "One YAML config file per experiment")
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.seed is not None:
        config["seed"] = args.seed
    if args.t_steps is not None:
        config["simulation"]["T_steps"] = args.t_steps
    if args.n_burnin is not None:
        config["simulation"]["n_burnin"] = args.n_burnin
    if args.save_every is not None:
        config["simulation"]["save_every"] = args.save_every

    # Select simulator
    if args.simulator == "numpy":
        from src.simulator.numpy_simulator import NumpySimulator
        sim = NumpySimulator(config)
    elif args.simulator == "hoomd":
        from src.simulator.hoomd_simulator import HoomdSimulator
        sim = HoomdSimulator(config)

    # Run trajectories
    print(f"Running {args.n_trajectories} trajectories with {args.simulator} simulator")
    print(f"Config: {args.config}")
    print(f"Output: {args.output}")
    print(f"Chain length: N={config['chain']['N']}")
    print(f"Bond type: {config['bond']['type']}")
    print(f"dt={config['integrator']['dt']}, gamma={config['thermostat']['gamma']}, "
          f"kBT={config['thermostat']['kBT']}")
    print("=" * 60)

    total_start = time.time()
    for i in range(args.n_trajectories):
        traj_id = args.traj_id_start + i
        traj_start = time.time()
        trajectory = sim.run(
            traj_id=traj_id,
            output_dir=args.output,
            verbose=not args.quiet,
        )
        elapsed = time.time() - traj_start
        print(f"[Traj {traj_id}] Completed in {elapsed:.1f}s, "
              f"{trajectory['n_frames']} frames saved")

    total_elapsed = time.time() - total_start
    print(f"\nAll {args.n_trajectories} trajectories completed in {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
