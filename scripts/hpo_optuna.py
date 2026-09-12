#!/usr/bin/env python3
"""Hyperparameter optimization with Optuna + W&B (Month 6).

Searches over:
  - Learning rate (log-uniform)
  - Hidden dimension
  - Number of GNN layers
  - Weight decay (log-uniform)
  - Physics loss weight (for Model 3, Month 7)

Usage
-----
    # Default 50-trial search:
    python scripts/hpo_optuna.py --config configs/default.yaml --n-trials 50

    # Quick test with 5 trials:
    python scripts/hpo_optuna.py --config configs/default.yaml --n-trials 5 --epochs 10

    # Resume a previous study:
    python scripts/hpo_optuna.py --study-name polymer-gnn-hpo --storage sqlite:///hpo.db

Notes
-----
- Each trial trains a model for `--epochs` epochs (default from config).
- The objective is validation loss (lower = better).
- Results are logged to W&B if --wandb is passed.
- The best hyperparameters are printed and saved to `hpo_best_params.yaml`.
"""

from __future__ import annotations

import argparse
import os
import sys
import yaml
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import optuna
from optuna.trial import Trial

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def create_objective(base_config: dict, epochs_override: int = None, wandb_enabled: bool = False):
    """Factory that returns an Optuna objective function."""

    def objective(trial: Trial) -> float:
        import torch
        from torch.utils.data import DataLoader
        from src.models.baseline_gnn import BaselineGNN
        from src.training.trainer import Trainer

        # --- Sample hyperparameters ---
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        hidden_dim = trial.suggest_categorical('hidden_dim', [32, 64, 128, 256])
        n_layers = trial.suggest_int('n_layers', 2, 5)
        weight_decay = trial.suggest_float('weight_decay', 1e-7, 1e-3, log=True)
        dropout = trial.suggest_float('dropout', 0.0, 0.3)

        # Physics loss weight (0.0 for baseline, > 0 for physics-informed)
        physics_weight = trial.suggest_float('physics_loss_weight', 0.0, 0.1)

        # Build config for this trial
        config = base_config.copy()
        config.update({
            'lr': lr,
            'hidden_dim': hidden_dim,
            'n_layers': n_layers,
            'weight_decay': weight_decay,
            'dropout': dropout,
            'physics_loss_weight': physics_weight,
        })

        if epochs_override is not None:
            config['epochs'] = epochs_override

        # W&B run name
        run_name = f"trial-{trial.number}_lr{lr:.1e}_h{hidden_dim}_L{n_layers}"

        # --- Build model ---
        input_dim = config.get('input_dim', 3)
        output_dim = config.get('output_dim', 3)

        model = BaselineGNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=n_layers,
            dropout=dropout,
        )

        # --- Build data loaders ---
        # For HPO, use a subset of the training data for speed
        # In production, these would be the full dataset loaders
        try:
            from src.data.dataset import PolymerDataset
            dataset = PolymerDataset(root='data')
            train_data = dataset[0]  # will be replaced with proper split
            val_data = dataset[1]
            train_loader = DataLoader(train_data, batch_size=config.get('batch_size', 32), shuffle=True)
            val_loader = DataLoader(val_data, batch_size=config.get('batch_size', 32))
        except Exception:
            # If dataset isn't available yet, create dummy data for testing the HPO pipeline
            from torch.utils.data import TensorDataset
            N = config.get('chain_length', 30)
            n_train = 100
            n_val = 20
            X_train = torch.randn(n_train, N * 3)
            y_train = torch.randn(n_train, N * 3)
            X_val = torch.randn(n_val, N * 3)
            y_val = torch.randn(n_val, N * 3)
            train_loader = DataLoader(
                TensorDataset(X_train, y_train),
                batch_size=config.get('batch_size', 32), shuffle=True
            )
            val_loader = DataLoader(
                TensorDataset(X_val, y_val),
                batch_size=config.get('batch_size', 32)
            )

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # --- Train ---
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            checkpoint_dir=f'models/hpo/trial_{trial.number}',
            wandb_enabled=wandb_enabled,
            wandb_project='polymer-gnn-hpo',
            wandb_run_name=run_name,
        )

        history = trainer.train()

        # Report intermediate values for pruning
        val_loss = history['best_val_loss']

        return val_loss

    return objective


def main():
    parser = argparse.ArgumentParser(description="Optuna HPO for Polymer GNN")
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to base config YAML')
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Number of Optuna trials')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override epochs per trial (default: from config)')
    parser.add_argument('--study-name', type=str, default='polymer-gnn-hpo',
                        help='Optuna study name')
    parser.add_argument('--storage', type=str, default=None,
                        help='Optuna storage URI (e.g. sqlite:///hpo.db). If None, in-memory.')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable W&B logging for each trial')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for Optuna sampler')
    args = parser.parse_args()

    # Load base config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            base_config = yaml.safe_load(f) or {}
    else:
        print(f"Warning: config file {config_path} not found, using defaults")
        base_config = {}

    # Create Optuna study
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction='minimize',  # minimize val_loss
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    objective = create_objective(
        base_config,
        epochs_override=args.epochs,
        wandb_enabled=args.wandb,
    )

    print("=" * 65)
    print(f"OPTUNA HPO — {args.n_trials} trials")
    print(f"Study: {args.study_name}")
    print(f"Storage: {args.storage or 'in-memory'}")
    print(f"W&B: {'enabled' if args.wandb else 'disabled'}")
    print("=" * 65)

    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    # Print results
    print("\n" + "=" * 65)
    print("BEST TRIAL")
    print("=" * 65)
    best = study.best_trial
    print(f"  Trial #{best.number}")
    print(f"  Val Loss: {best.value:.6f}")
    print(f"  Params:")
    for key, value in best.params.items():
        print(f"    {key}: {value}")

    # Save best params to YAML
    output_path = Path('configs/hpo_best_params.yaml')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(dict(best.params), f, default_flow_style=False)
    print(f"\n  Saved to: {output_path}")

    # Print top-5 trials
    print(f"\nTOP 5 TRIALS:")
    for t in sorted(study.trials, key=lambda t: t.value if t.value else float('inf'))[:5]:
        print(f"  #{t.number:3d} | val_loss={t.value:.6f} | lr={t.params.get('lr', '?'):.1e} h={t.params.get('hidden_dim', '?')} L={t.params.get('n_layers', '?')}")

    print("=" * 65)


if __name__ == "__main__":
    main()
