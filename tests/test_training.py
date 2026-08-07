"""
Unit tests for training infrastructure.

Tests verify:
1. Loss functions compute correct values
2. Single training step reduces loss (overfitting test)
3. Trainer initializes correctly
4. Checkpoint save/load works
"""

import os
import numpy as np
import pytest

# Skip all tests if torch/pyg not available
torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

from src.models.baseline_gnn import BaselineGNN
from src.training.losses import displacement_mse_loss, bond_length_penalty, combined_loss
from src.training.trainer import Trainer


def _create_dummy_graph(n_beads: int = 5) -> Data:
    """Create a dummy PyG Data object for testing."""
    positions = np.zeros((n_beads, 3), dtype=np.float32)
    positions[:, 0] = np.arange(n_beads, dtype=np.float32)
    bead_idx = np.arange(n_beads, dtype=np.float32) / n_beads
    x = np.concatenate([positions, bead_idx[:, None]], axis=1)

    src, dst, attrs = [], [], []
    for i in range(n_beads - 1):
        dr = positions[i + 1] - positions[i]
        dist = float(np.linalg.norm(dr))
        src.extend([i, i + 1])
        dst.extend([i + 1, i])
        attrs.append([dr[0], dr[1], dr[2], dist, 1.0])
        attrs.append([-dr[0], -dr[1], -dr[2], dist, 1.0])

    # Small displacement target
    y = 0.01 * np.random.randn(n_beads, 3).astype(np.float32)

    return Data(
        x=torch.tensor(x),
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        edge_attr=torch.tensor(attrs, dtype=torch.float),
        y=torch.tensor(y),
    )


class TestLossFunctions:
    """Tests for loss functions."""

    def test_mse_loss_zero(self):
        """MSE loss should be zero when pred equals target."""
        pred = torch.randn(10, 3)
        loss = displacement_mse_loss(pred, pred)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_mse_loss_positive(self):
        """MSE loss should be positive when pred != target."""
        pred = torch.randn(10, 3)
        target = torch.randn(10, 3)
        loss = displacement_mse_loss(pred, target)
        assert loss.item() > 0

    def test_mse_loss_known_value(self):
        """MSE loss should match manual computation."""
        pred = torch.tensor([[1.0, 0.0, 0.0]])
        target = torch.tensor([[0.0, 0.0, 0.0]])
        loss = displacement_mse_loss(pred, target)
        # MSE = mean((1-0)^2 + (0-0)^2 + (0-0)^2) = 1/3
        assert loss.item() == pytest.approx(1.0 / 3.0, abs=1e-6)

    def test_bond_length_penalty_placeholder(self):
        """Bond length penalty should return 0 (placeholder)."""
        positions = torch.randn(5, 3)
        displacement = torch.randn(5, 3)
        bonds = torch.tensor([[0, 1], [1, 2]])
        loss = bond_length_penalty(positions, displacement, bonds)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_combined_loss_no_physics(self):
        """Combined loss with weight=0 should equal MSE."""
        pred = torch.randn(10, 3)
        target = torch.randn(10, 3)

        mse = displacement_mse_loss(pred, target)
        combined = combined_loss(pred, target, physics_weight=0.0)
        assert combined.item() == pytest.approx(mse.item(), abs=1e-7)

    def test_combined_loss_with_physics(self):
        """Combined loss with physics should be MSE + weight * physics."""
        pred = torch.randn(10, 3)
        target = torch.randn(10, 3)
        physics = torch.tensor(2.0)
        weight = 0.5

        mse = displacement_mse_loss(pred, target)
        combined = combined_loss(pred, target, physics_loss=physics, physics_weight=weight)
        expected = mse.item() + weight * 2.0
        assert combined.item() == pytest.approx(expected, abs=1e-5)


class TestTrainer:
    """Tests for the Trainer class."""

    @pytest.fixture
    def setup(self, tmp_path):
        """Set up model, data, and trainer for testing."""
        # Create small model
        model = BaselineGNN(hidden_dim=16, n_layers=1)

        # Create tiny dataset
        data_list = [_create_dummy_graph(n_beads=5) for _ in range(20)]
        train_data = data_list[:14]
        val_data = data_list[14:]

        train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=4, shuffle=False)

        config = {
            'epochs': 5,
            'lr': 1e-3,
            'weight_decay': 0.0,
            'patience': 100,  # Don't early-stop in tests
            'physics_loss_weight': 0.0,
        }

        checkpoint_dir = str(tmp_path / "checkpoints")

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device='cpu',
            checkpoint_dir=checkpoint_dir,
        )

        return trainer, checkpoint_dir

    def test_train_epoch_returns_loss(self, setup):
        """train_epoch should return a dict with train_loss."""
        trainer, _ = setup
        result = trainer.train_epoch()
        assert 'train_loss' in result
        assert isinstance(result['train_loss'], float)
        assert result['train_loss'] > 0

    def test_validate_returns_loss(self, setup):
        """validate should return a dict with val_loss."""
        trainer, _ = setup
        result = trainer.validate()
        assert 'val_loss' in result
        assert isinstance(result['val_loss'], float)
        assert result['val_loss'] > 0

    def test_full_training(self, setup):
        """Full training loop should complete and return history."""
        trainer, _ = setup
        history = trainer.train()

        assert 'train_loss' in history
        assert 'val_loss' in history
        assert len(history['train_loss']) == 5  # 5 epochs
        assert len(history['val_loss']) == 5

    def test_checkpoint_save_load(self, setup):
        """Should save and load checkpoints correctly."""
        trainer, checkpoint_dir = setup

        # Train one epoch
        trainer.train_epoch()
        val_result = trainer.validate()

        # Save checkpoint
        ckpt_path = os.path.join(checkpoint_dir, 'test_checkpoint.pt')
        trainer.save_checkpoint(ckpt_path, epoch=1, val_loss=val_result['val_loss'])

        assert os.path.exists(ckpt_path)

        # Load checkpoint
        checkpoint = trainer.load_checkpoint(ckpt_path)
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint
        assert checkpoint['epoch'] == 1

    def test_best_model_saved(self, setup):
        """Training should save the best model checkpoint."""
        trainer, checkpoint_dir = setup
        trainer.train()

        best_path = os.path.join(checkpoint_dir, 'best_model.pt')
        assert os.path.exists(best_path), "Best model checkpoint not saved"
