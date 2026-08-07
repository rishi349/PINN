#!/usr/bin/env python3
"""
Evaluate a trained GNN model via autoregressive rollout.

Usage:
    python scripts/evaluate.py --checkpoint models/saved/best_model.pt \\
                               --config configs/default.yaml \\
                               --data-dir data/raw/N30

Runs rollout evaluation at multiple horizons and reports physics metrics.
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch

from src.models.baseline_gnn import BaselineGNN
from src.evaluation.rollout import RolloutEvaluator
from src.evaluation.metrics import (
    rollout_metrics,
    bond_length_deviation,
    radius_of_gyration,
)
from src.simulator.numpy_simulator import load_config


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained GNN model via rollout"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory with test trajectories (for ground truth comparison)",
    )
    parser.add_argument(
        "--rollout-lengths",
        type=int,
        nargs="+",
        default=[100, 500, 1000],
        help="Rollout lengths to evaluate (default: 100 500 1000)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device (default: cpu)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    N = config["chain"]["N"]
    neighbor_cutoff = config.get("graph", {}).get("neighbor_cutoff", 2.5)

    print("=" * 60)
    print("GNN MODEL EVALUATION")
    print("=" * 60)
    print(f"Checkpoint:      {args.checkpoint}")
    print(f"Chain length:    N = {N}")
    print(f"Rollout lengths: {args.rollout_lengths}")
    print(f"Device:          {args.device}")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)

    model = BaselineGNN(
        node_input_dim=4,
        edge_input_dim=5,
        hidden_dim=checkpoint.get("hidden_dim", 128),
        n_layers=checkpoint.get("n_layers", 4),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"  Loaded from epoch {checkpoint.get('epoch', '?')}")
    print(f"  Val loss: {checkpoint.get('val_loss', '?')}")

    # Create rollout evaluator
    evaluator = RolloutEvaluator(
        model=model,
        n_beads=N,
        neighbor_cutoff=neighbor_cutoff,
        device=args.device,
    )

    # Load test data for initial positions and ground truth
    if args.data_dir:
        data_dir = Path(args.data_dir)
        npz_files = sorted(data_dir.glob("*.npz"))
        if not npz_files:
            print(f"  No .npz files found in {data_dir}")
            sys.exit(1)

        # Use first test trajectory
        from src.data.trajectory import load_trajectory_npz
        test_traj = load_trajectory_npz(str(npz_files[-1]))  # last traj as test
        initial_positions = test_traj["positions"][0]  # First frame
        gt_trajectory = test_traj["positions"]
        print(f"  Test trajectory: {npz_files[-1].name}")
        print(f"  Ground truth frames: {len(gt_trajectory)}")
    else:
        # Generate initial positions from simulator
        from src.simulator.numpy_simulator import NumpySimulator
        sim = NumpySimulator(config)
        rng = np.random.default_rng(999)
        initial_positions = sim.initialize_chain(rng)
        gt_trajectory = None
        print("  No test data dir — using random initial positions")

    # Run rollout evaluations
    results = {}
    for length in args.rollout_lengths:
        print(f"\n--- Rollout length: {length} ---")
        start_time = time.time()

        rollout_result = evaluator.rollout(
            initial_positions=initial_positions,
            n_steps=length,
        )
        elapsed = time.time() - start_time

        trajectory = rollout_result["trajectory"]
        blowup = rollout_result["blowup_step"]

        # Final frame metrics
        final_pos = trajectory[-1]
        bl_stats = bond_length_deviation(final_pos, r0=config["bond"]["r0"])
        rg = radius_of_gyration(final_pos)

        print(f"  Time: {elapsed:.2f}s")
        print(f"  Blowup: {'step ' + str(blowup) if blowup else 'None'}")
        print(f"  Final R_g: {rg:.4f}")
        print(f"  Bond length: mean={bl_stats['mean']:.4f}, "
              f"std={bl_stats['std']:.4f}, "
              f"max_dev={bl_stats['max_deviation']:.4f}")

        if gt_trajectory is not None and length <= len(gt_trajectory) - 1:
            gt_slice = gt_trajectory[:length + 1]
            metrics = rollout_metrics(trajectory[:len(gt_slice)], gt_slice,
                                      r0=config["bond"]["r0"])
            print(f"  Position MSE (final): "
                  f"{metrics['position_mse_per_step'][-1]:.6f}")

        results[str(length)] = {
            "blowup_step": blowup,
            "final_rg": float(rg),
            "bond_stats": bl_stats,
            "elapsed_seconds": elapsed,
        }

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")

    print(f"\n{'=' * 60}")
    print("EVALUATION COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
