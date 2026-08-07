"""
Unit tests for the PolymerDataset and data splitting.

Tests verify:
1. Dataset creates correct splits
2. Trajectory-based splitting (no frame leakage)
3. Split ratios are approximately correct
4. DataLoader creation works
"""

import os
import json
import tempfile
import shutil
import numpy as np
import pytest

# Skip all tests if torch/pyg not available
torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import PolymerDataset, create_dataloaders


def _create_dummy_trajectories(output_dir: str, n_trajectories: int = 10,
                                n_frames: int = 20, n_beads: int = 5):
    """Create dummy trajectory JSON files for testing."""
    os.makedirs(output_dir, exist_ok=True)

    for traj_id in range(n_trajectories):
        positions = np.random.randn(n_frames, n_beads, 3).tolist()
        forces = np.random.randn(n_frames, n_beads, 3).tolist()

        frames = []
        for i in range(n_frames):
            frames.append({
                "positions": positions[i],
                "forces": forces[i],
                "timestep_index": i * 100,
            })

        trajectory = {
            "metadata": {"traj_id": traj_id, "chain_length": n_beads},
            "frames": frames,
            "n_frames": n_frames,
        }

        filepath = os.path.join(output_dir, f"trajectory_{traj_id:04d}.json")
        with open(filepath, "w") as f:
            json.dump(trajectory, f)


class TestPolymerDataset:
    """Tests for PolymerDataset."""

    @pytest.fixture
    def temp_dirs(self, tmp_path):
        """Create temporary directories for test data."""
        traj_dir = str(tmp_path / "trajectories")
        root_dir = str(tmp_path / "processed")
        _create_dummy_trajectories(traj_dir, n_trajectories=10, n_frames=6, n_beads=5)
        return traj_dir, root_dir

    def test_train_split_created(self, temp_dirs):
        """Train split should be created successfully."""
        traj_dir, root_dir = temp_dirs
        dataset = PolymerDataset(
            root=root_dir, trajectories_dir=traj_dir,
            split='train', seed=42,
        )
        assert len(dataset) > 0

    def test_val_split_created(self, temp_dirs):
        """Val split should be created successfully."""
        traj_dir, root_dir = temp_dirs
        dataset = PolymerDataset(
            root=root_dir, trajectories_dir=traj_dir,
            split='val', seed=42,
        )
        assert len(dataset) > 0

    def test_test_split_created(self, temp_dirs):
        """Test split should be created successfully."""
        traj_dir, root_dir = temp_dirs
        dataset = PolymerDataset(
            root=root_dir, trajectories_dir=traj_dir,
            split='test', seed=42,
        )
        assert len(dataset) > 0

    def test_no_frame_leakage(self, temp_dirs):
        """Train + val + test should cover all frames without overlap."""
        traj_dir, root_dir = temp_dirs

        train = PolymerDataset(root=root_dir, trajectories_dir=traj_dir,
                               split='train', seed=42)
        val = PolymerDataset(root=root_dir, trajectories_dir=traj_dir,
                             split='val', seed=42)
        test = PolymerDataset(root=root_dir, trajectories_dir=traj_dir,
                              split='test', seed=42)

        # Total should equal all frames
        # 10 trajectories * (6-1) frames per trajectory = 50 total data points
        total = len(train) + len(val) + len(test)
        expected = 10 * (6 - 1)  # n_trajs * (n_frames - 1)
        assert total == expected, (
            f"Total samples {total} != expected {expected}. "
            f"Train={len(train)}, Val={len(val)}, Test={len(test)}"
        )

    def test_approximate_split_ratios(self, temp_dirs):
        """Split ratios should approximately match 70/15/15."""
        traj_dir, root_dir = temp_dirs

        train = PolymerDataset(root=root_dir, trajectories_dir=traj_dir,
                               split='train', seed=42)
        val = PolymerDataset(root=root_dir, trajectories_dir=traj_dir,
                             split='val', seed=42)
        test = PolymerDataset(root=root_dir, trajectories_dir=traj_dir,
                              split='test', seed=42)

        total = len(train) + len(val) + len(test)
        train_ratio = len(train) / total
        # With 10 trajectories: 7 train, 1 val, 2 test
        # Ratios won't be exact but train should be majority
        assert train_ratio > 0.5, f"Train ratio {train_ratio} too low"

    def test_data_has_correct_fields(self, temp_dirs):
        """Each sample should have x, edge_index, edge_attr, y."""
        traj_dir, root_dir = temp_dirs
        dataset = PolymerDataset(root=root_dir, trajectories_dir=traj_dir,
                                 split='train', seed=42)

        sample = dataset[0]
        assert hasattr(sample, 'x'), "Missing node features x"
        assert hasattr(sample, 'edge_index'), "Missing edge_index"
        assert hasattr(sample, 'edge_attr'), "Missing edge_attr"
        assert hasattr(sample, 'y'), "Missing target y"

    def test_invalid_split_raises(self, temp_dirs):
        """Invalid split name should raise ValueError."""
        traj_dir, root_dir = temp_dirs
        with pytest.raises(ValueError):
            PolymerDataset(root=root_dir, trajectories_dir=traj_dir,
                          split='invalid', seed=42)


class TestCreateDataloaders:
    """Tests for create_dataloaders helper."""

    @pytest.fixture
    def temp_dirs(self, tmp_path):
        traj_dir = str(tmp_path / "trajectories")
        root_dir = str(tmp_path / "processed")
        _create_dummy_trajectories(traj_dir, n_trajectories=10, n_frames=6, n_beads=5)
        return traj_dir, root_dir

    def test_returns_three_loaders(self, temp_dirs):
        """Should return exactly 3 DataLoaders."""
        traj_dir, root_dir = temp_dirs
        train_loader, val_loader, test_loader = create_dataloaders(
            root=root_dir, trajectories_dir=traj_dir, batch_size=4, seed=42,
        )
        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None

    def test_batch_iteration(self, temp_dirs):
        """Should be able to iterate over a batch."""
        traj_dir, root_dir = temp_dirs
        train_loader, _, _ = create_dataloaders(
            root=root_dir, trajectories_dir=traj_dir, batch_size=4, seed=42,
        )
        batch = next(iter(train_loader))
        assert batch.x.ndim == 2
        assert batch.y.ndim == 2
