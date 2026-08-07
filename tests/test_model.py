"""
Unit tests for the BaselineGNN model.

Tests verify:
1. Forward pass produces correct output shape
2. Gradient flow through all parameters
3. Model handles different numbers of beads
4. Parameter counting works
"""

import numpy as np
import pytest

# Skip all tests if torch/pyg not available
torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torch_geometric.data import Data, Batch
from src.models.baseline_gnn import BaselineGNN
from src.models.model_utils import build_mlp, get_activation, count_parameters


def _create_dummy_graph(n_beads: int = 10) -> Data:
    """Create a dummy PyG Data object for testing."""
    # Node features: position(3) + bead_index(1) = 4
    positions = np.random.randn(n_beads, 3).astype(np.float32)
    bead_idx = np.arange(n_beads, dtype=np.float32) / n_beads
    x = np.concatenate([positions, bead_idx[:, None]], axis=1)

    # Bond edges (bidirectional) + a few nonbonded
    src, dst, attrs = [], [], []
    for i in range(n_beads - 1):
        # Forward bond
        dr = positions[i + 1] - positions[i]
        dist = np.linalg.norm(dr)
        src.append(i); dst.append(i + 1)
        attrs.append([dr[0], dr[1], dr[2], dist, 1.0])
        # Reverse bond
        src.append(i + 1); dst.append(i)
        attrs.append([-dr[0], -dr[1], -dr[2], dist, 1.0])

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(attrs, dtype=torch.float)

    # Target displacement
    y = torch.randn(n_beads, 3)

    data = Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
    )
    return data


class TestBaselineGNN:
    """Tests for the BaselineGNN model."""

    def test_forward_output_shape(self):
        """Forward pass should output (N, 3) displacement predictions."""
        model = BaselineGNN(hidden_dim=32, n_layers=2)
        data = _create_dummy_graph(n_beads=10)

        out = model(data)
        assert out.shape == (10, 3), f"Expected (10, 3), got {out.shape}"

    def test_forward_batched(self):
        """Forward pass should work with batched graphs."""
        model = BaselineGNN(hidden_dim=32, n_layers=2)

        graphs = [_create_dummy_graph(n_beads=10) for _ in range(4)]
        batch = Batch.from_data_list(graphs)

        out = model(batch)
        assert out.shape == (40, 3), f"Expected (40, 3), got {out.shape}"

    def test_gradient_flow(self):
        """All parameters should receive gradients."""
        model = BaselineGNN(hidden_dim=32, n_layers=2)
        data = _create_dummy_graph(n_beads=5)

        out = model(data)
        loss = out.sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.all(param.grad == 0), (
                    f"Zero gradient for {name}"
                )

    def test_different_bead_counts(self):
        """Model should handle different chain lengths."""
        model = BaselineGNN(hidden_dim=32, n_layers=2)

        for n_beads in [5, 10, 30]:
            data = _create_dummy_graph(n_beads=n_beads)
            out = model(data)
            assert out.shape == (n_beads, 3)

    def test_deterministic_output(self):
        """Same input should give same output (no dropout in eval mode)."""
        model = BaselineGNN(hidden_dim=32, n_layers=2, dropout=0.0)
        model.eval()

        data = _create_dummy_graph(n_beads=5)
        out1 = model(data)
        out2 = model(data)

        torch.testing.assert_close(out1, out2)


class TestModelUtils:
    """Tests for model utility functions."""

    def test_build_mlp_shapes(self):
        """MLP should have correct input/output dimensions."""
        mlp = build_mlp(input_dim=10, hidden_dim=32, output_dim=3, n_layers=3)
        x = torch.randn(5, 10)
        out = mlp(x)
        assert out.shape == (5, 3)

    def test_build_mlp_single_layer(self):
        """Single-layer MLP should work."""
        mlp = build_mlp(input_dim=10, hidden_dim=32, output_dim=3, n_layers=1)
        x = torch.randn(5, 10)
        out = mlp(x)
        assert out.shape == (5, 3)

    def test_get_activation(self):
        """Should return correct activation modules."""
        for name in ['silu', 'relu', 'gelu', 'tanh']:
            act = get_activation(name)
            assert isinstance(act, torch.nn.Module)

    def test_get_activation_invalid(self):
        """Should raise for invalid activation name."""
        with pytest.raises(ValueError):
            get_activation('invalid_activation')

    def test_count_parameters(self):
        """Parameter count should be positive."""
        model = BaselineGNN(hidden_dim=32, n_layers=2)
        n_params = count_parameters(model)
        assert n_params > 0
        assert isinstance(n_params, int)
