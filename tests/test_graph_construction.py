"""
Unit tests for graph construction.

Tests verify:
1. Correct node feature dimensions
2. Correct edge construction (bond + nonbonded)
3. Correct target (displacement) computation
4. Edge attribute dimensions
5. Bidirectional edges
"""

import numpy as np
import pytest

# Skip all tests if torch/pyg not available
torch = pytest.importorskip("torch")
pyg_data = pytest.importorskip("torch_geometric.data")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.graph_construction import frame_to_pyg_data, trajectory_to_pyg_data_list


class TestFrameToPygData:
    """Tests for frame_to_pyg_data conversion."""

    def test_output_type(self):
        """Output should be a PyG Data object."""
        N = 5
        pos_t = np.zeros((N, 3))
        pos_t[:, 0] = np.arange(N)
        pos_t1 = pos_t + 0.01 * np.random.randn(N, 3)

        data = frame_to_pyg_data(pos_t, pos_t1, n_beads=N)
        assert isinstance(data, pyg_data.Data)

    def test_node_feature_shape(self):
        """Node features should be (N, 4): position(3) + bead_index(1)."""
        N = 10
        pos_t = np.zeros((N, 3))
        pos_t[:, 0] = np.arange(N)
        pos_t1 = pos_t.copy()

        data = frame_to_pyg_data(pos_t, pos_t1, n_beads=N)
        assert data.x.shape == (N, 4)

    def test_target_shape(self):
        """Target y should be (N, 3) displacement."""
        N = 10
        pos_t = np.zeros((N, 3))
        pos_t[:, 0] = np.arange(N)
        pos_t1 = pos_t + 0.1

        data = frame_to_pyg_data(pos_t, pos_t1, n_beads=N)
        assert data.y.shape == (N, 3)

    def test_target_value(self):
        """Target should equal pos_t1 - pos_t."""
        N = 5
        pos_t = np.random.randn(N, 3).astype(np.float32)
        pos_t1 = np.random.randn(N, 3).astype(np.float32)

        data = frame_to_pyg_data(pos_t, pos_t1, n_beads=N)
        expected = pos_t1 - pos_t
        np.testing.assert_allclose(data.y.numpy(), expected, atol=1e-6)

    def test_bond_edges_present(self):
        """Bond edges (i, i+1) should always be present."""
        N = 5
        pos_t = np.zeros((N, 3))
        pos_t[:, 0] = np.arange(N) * 1.0  # Well-separated
        pos_t1 = pos_t.copy()

        data = frame_to_pyg_data(pos_t, pos_t1, n_beads=N, neighbor_cutoff=1.5)

        edge_set = set()
        for i in range(data.edge_index.shape[1]):
            src = data.edge_index[0, i].item()
            dst = data.edge_index[1, i].item()
            edge_set.add((src, dst))

        # All bond edges should be present (both directions)
        for i in range(N - 1):
            assert (i, i + 1) in edge_set, f"Bond edge ({i}, {i+1}) missing"
            assert (i + 1, i) in edge_set, f"Bond edge ({i+1}, {i}) missing"

    def test_edge_attr_shape(self):
        """Edge attributes should be (E, 5): displacement(3) + distance(1) + bond_type(1)."""
        N = 5
        pos_t = np.zeros((N, 3))
        pos_t[:, 0] = np.arange(N) * 1.0
        pos_t1 = pos_t.copy()

        data = frame_to_pyg_data(pos_t, pos_t1, n_beads=N)
        assert data.edge_attr.shape[1] == 5

    def test_bond_type_values(self):
        """Bond edges should have bond_type=1, nonbonded should have bond_type=0."""
        N = 5
        pos_t = np.zeros((N, 3))
        pos_t[:, 0] = np.arange(N) * 1.0
        pos_t1 = pos_t.copy()

        data = frame_to_pyg_data(pos_t, pos_t1, n_beads=N, neighbor_cutoff=2.5)

        for i in range(data.edge_index.shape[1]):
            src = data.edge_index[0, i].item()
            dst = data.edge_index[1, i].item()
            bond_type = data.edge_attr[i, 4].item()

            if abs(src - dst) == 1:
                assert bond_type == 1.0, f"Bond edge ({src},{dst}) should have bond_type=1"
            else:
                assert bond_type == 0.0, f"Nonbonded edge ({src},{dst}) should have bond_type=0"

    def test_nonbonded_edges_within_cutoff(self):
        """Non-bonded edges should only exist for pairs within cutoff."""
        N = 10
        pos_t = np.zeros((N, 3))
        pos_t[:, 0] = np.arange(N) * 1.0  # spacing = 1.0
        pos_t1 = pos_t.copy()
        cutoff = 2.5

        data = frame_to_pyg_data(pos_t, pos_t1, n_beads=N, neighbor_cutoff=cutoff)

        for i in range(data.edge_index.shape[1]):
            src = data.edge_index[0, i].item()
            dst = data.edge_index[1, i].item()
            if abs(src - dst) > 1:
                # Nonbonded edge — verify distance < cutoff
                dist = np.linalg.norm(pos_t[dst] - pos_t[src])
                assert dist < cutoff, (
                    f"Nonbonded edge ({src},{dst}) has distance {dist} >= cutoff {cutoff}"
                )

    def test_normalized_bead_index(self):
        """Bead index feature should be i/N (normalized)."""
        N = 10
        pos_t = np.zeros((N, 3))
        pos_t[:, 0] = np.arange(N) * 1.0
        pos_t1 = pos_t.copy()

        data = frame_to_pyg_data(pos_t, pos_t1, n_beads=N)

        # Last feature column is bead_index
        bead_indices = data.x[:, 3].numpy()
        expected = np.arange(N, dtype=np.float32) / float(N)
        np.testing.assert_allclose(bead_indices, expected, atol=1e-6)


class TestTrajectoryToPygDataList:
    """Tests for trajectory_to_pyg_data_list."""

    def test_correct_number_of_graphs(self):
        """Should produce n_frames - 1 graph pairs."""
        n_frames = 10
        N = 5
        positions = [np.random.randn(N, 3) for _ in range(n_frames)]
        trajectory = {"frames": [{"positions": p.tolist()} for p in positions]}

        data_list = trajectory_to_pyg_data_list(trajectory)
        assert len(data_list) == n_frames - 1

    def test_npz_format(self):
        """Should handle NPZ-style trajectory with positions array."""
        n_frames = 8
        N = 5
        positions = np.random.randn(n_frames, N, 3)
        trajectory = {"positions": positions}

        data_list = trajectory_to_pyg_data_list(trajectory)
        assert len(data_list) == n_frames - 1

    def test_empty_trajectory(self):
        """Empty trajectory should return empty list."""
        trajectory = {"frames": []}
        data_list = trajectory_to_pyg_data_list(trajectory)
        assert len(data_list) == 0

    def test_single_frame(self):
        """Single frame trajectory should return empty list (no pairs)."""
        N = 5
        trajectory = {"frames": [{"positions": np.zeros((N, 3)).tolist()}]}
        data_list = trajectory_to_pyg_data_list(trajectory)
        assert len(data_list) == 0
