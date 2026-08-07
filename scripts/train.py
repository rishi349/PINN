#!/usr/bin/env python3
"""
Train a GNN model on polymer dynamics data.

Usage:
    python scripts/train.py --config configs/default.yaml --data-dir data/raw/N30
    python scripts/train.py --config configs/default.yaml --data-dir data/raw/N30 --epochs 50
    python scripts/train.py --config configs/default.yaml --data-dir data/raw/N30 --device cuda

Trains the baseline message-passing GNN (Model 1) to predict next-step
displacements for polymer beads.
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch

from src.data.dataset import create_dataloaders
from src.models.baseline_gnn import BaselineGNN
from src.models.model_utils import count_parameters
from src.training.trainer import Trainer
from src.simulator.numpy_simulator import load_config


def main():
    parser = argparse.ArgumentParser(
        description="Train GNN model on polymer dynamics data"
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
        required=True,
        help="Directory containing trajectory files (NPZ format)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="models/saved",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size (default: 32)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="GNN hidden dimension (default: 128)",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=4,
        help="Number of message passing layers (default: 4)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: cpu, cuda, mps (default: auto-detect)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    # Load config
    config = load_config(args.config)

    # Build training config
    training_config = {
        "epochs": args.epochs or 200,
        "lr": args.lr or 1e-3,
        "weight_decay": 1e-5,
        "scheduler": "cosine",
        "patience": 20,
        "physics_loss_weight": 0.0,  # Baseline: pure data-driven
    }

    # Set seeds for reproducibility
    torch.manual_seed(args.seed)

    print("=" * 60)
    print("GNN TRAINING")
    print("=" * 60)
    print(f"Data dir:        {args.data_dir}")
    print(f"Device:          {device}")
    print(f"Hidden dim:      {args.hidden_dim}")
    print(f"Layers:          {args.n_layers}")
    print(f"Epochs:          {training_config['epochs']}")
    print(f"Learning rate:   {training_config['lr']}")
    print(f"Batch size:      {args.batch_size}")
    print(f"Checkpoint dir:  {args.checkpoint_dir}")
    print("=" * 60)

    # Create data loaders
    print("\nLoading and processing data...")
    neighbor_cutoff = config.get("graph", {}).get("neighbor_cutoff", 2.5)

    train_loader, val_loader, test_loader = create_dataloaders(
        root="data/processed",
        trajectories_dir=args.data_dir,
        batch_size=args.batch_size,
        neighbor_cutoff=neighbor_cutoff,
        seed=args.seed,
    )

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")

    # Create model
    model = BaselineGNN(
        node_input_dim=4,   # position(3) + bead_index(1)
        edge_input_dim=5,   # displacement(3) + distance(1) + bond_type(1)
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        dropout=0.0,
        activation="silu",
    )

    n_params = count_parameters(model)
    print(f"\nModel: BaselineGNN")
    print(f"  Parameters: {n_params:,}")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Train
    print(f"\nStarting training...")
    start_time = time.time()
    history = trainer.train()
    elapsed = time.time() - start_time

    print(f"\n{'=' * 60}")
    print(f"Training complete in {elapsed:.1f}s")
    print(f"Best val loss: {history['best_val_loss']:.6f} "
          f"(epoch {history['best_epoch']})")
    print(f"Checkpoint: {args.checkpoint_dir}/best_model.pt")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
