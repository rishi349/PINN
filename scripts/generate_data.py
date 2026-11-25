#!/usr/bin/env python3
"""
Generate multiple simulation trajectories for GNN training.

Usage:
    python scripts/generate_data.py --config configs/default.yaml
    python scripts/generate_data.py --config configs/default.yaml --n-trajectories 5 --output data/raw
    python scripts/generate_data.py --config configs/default.yaml --chain-length 50 --n-trajectories 2

Generates trajectories with different random seeds for reproducible training.
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.simulator.numpy_simulator import NumpySimulator, load_config
from src.data.trajectory import save_trajectory_npz


def main():
    parser = argparse.ArgumentParser(
        description="Generate multiple simulation trajectories for GNN training"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--n-trajectories",
        type=int,
        default=10,
        help="Number of trajectories to generate (default: 10)",
    )
    parser.add_argument(
        "--chain-length",
        type=int,
        default=None,
        help="Override chain length N (default: from config)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw",
        help="Output directory (default: data/raw)",
    )
    parser.add_argument(
        "--traj-id-start",
        type=int,
        default=0,
        help="Starting trajectory ID (default: 0)",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=None,
        help="Override total simulation steps",
    )
    parser.add_argument(
        "--burnin",
        type=int,
        default=None,
        help="Override burn-in steps (default: auto-adjusted if n_steps < default burn-in)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=None,
        help="Override save frequency (default: from config)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "npz"],
        default="npz",
        help="Output format (default: npz for compact storage)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-step output",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override chain length if specified
    if args.chain_length is not None:
        config["chain"]["N"] = args.chain_length

    # Override simulation steps if specified
    if args.n_steps is not None:
        config["simulation"]["T_steps"] = args.n_steps

    # Override burn-in steps if specified, or auto-adjust if n_steps < default burn-in
    if args.burnin is not None:
        config["simulation"]["n_burnin"] = args.burnin
    elif args.n_steps is not None and args.n_steps <= config["simulation"]["n_burnin"]:
        # Auto-scale burn-in to 20% of steps so post-burn-in production frames are saved
        config["simulation"]["n_burnin"] = max(0, int(0.2 * args.n_steps))

    # Override save_every if specified
    if args.save_every is not None:
        config["simulation"]["save_every"] = args.save_every
    elif args.n_steps is not None and (config["simulation"]["T_steps"] - config["simulation"]["n_burnin"]) < config["simulation"]["save_every"] * 5:
        prod_steps = config["simulation"]["T_steps"] - config["simulation"]["n_burnin"]
        config["simulation"]["save_every"] = max(1, prod_steps // 20)

    N = config["chain"]["N"]
    T_steps = config["simulation"]["T_steps"]
    n_burnin = config["simulation"]["n_burnin"]
    save_every = config["simulation"]["save_every"]
    output_dir = Path(args.output) / f"N{N}"

    print("=" * 60)
    print("TRAJECTORY GENERATION")
    print("=" * 60)
    print(f"Chain length:    N = {N}")
    print(f"Sim steps:       {T_steps} (Burn-in: {n_burnin}, Save every: {save_every})")
    print(f"Trajectories:    {args.n_trajectories}")
    print(f"Output dir:      {output_dir}")
    print(f"Format:          {args.format}")
    print(f"ID range:        {args.traj_id_start} .. "
          f"{args.traj_id_start + args.n_trajectories - 1}")
    print("=" * 60)

    # Create simulator
    sim = NumpySimulator(config)

    total_start = time.time()

    for i in range(args.n_trajectories):
        traj_id = args.traj_id_start + i
        seed = config["seed"] + traj_id

        print(f"\n--- Trajectory {traj_id} (seed={seed}) ---")

        traj_start = time.time()

        # Run simulation
        trajectory = sim.run(
            traj_id=traj_id,
            seed=seed,
            verbose=not args.quiet,
        )

        traj_elapsed = time.time() - traj_start

        # Save trajectory
        if args.format == "npz":
            filepath = save_trajectory_npz(trajectory, str(output_dir))
        else:
            from src.data.trajectory import save_trajectory_json
            filepath = save_trajectory_json(trajectory, str(output_dir))

        n_frames = trajectory["n_frames"]
        print(f"  Saved: {filepath}")
        print(f"  Frames: {n_frames}, Time: {traj_elapsed:.1f}s")

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"All {args.n_trajectories} trajectories generated in {total_elapsed:.1f}s")
    print(f"Output: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
