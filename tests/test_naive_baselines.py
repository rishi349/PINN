"""Tests for naive baseline models (0a and 0b)."""

import numpy as np
import pytest
from src.models.naive_baselines import ZeroDisplacementBaseline, GlobalStatsBaseline


# ---------------------------------------------------------------------------
# ZeroDisplacementBaseline
# ---------------------------------------------------------------------------

class TestZeroDisplacementBaseline:

    def test_predict_returns_zeros(self):
        bl = ZeroDisplacementBaseline(n_beads=30, dim=3)
        positions = np.random.randn(30, 3)
        pred = bl.predict(positions)
        assert pred.shape == (30, 3)
        assert np.allclose(pred, 0.0)

    def test_predict_ignores_input(self):
        bl = ZeroDisplacementBaseline(n_beads=10, dim=3)
        pos_a = np.random.randn(10, 3)
        pos_b = np.random.randn(10, 3) * 100
        assert np.allclose(bl.predict(pos_a), bl.predict(pos_b))

    def test_one_step_mse_equals_mean_sq_displacement(self):
        bl = ZeroDisplacementBaseline()
        # If true displacements are all 1.0, MSE should be 1.0
        true_disp = np.ones((100, 30, 3))
        mse = bl.evaluate_one_step_mse(true_disp)
        assert abs(mse - 1.0) < 1e-9

    def test_one_step_mse_zero_for_zero_target(self):
        bl = ZeroDisplacementBaseline()
        true_disp = np.zeros((50, 30, 3))
        assert bl.evaluate_one_step_mse(true_disp) == 0.0

    def test_repr(self):
        bl = ZeroDisplacementBaseline(n_beads=5, dim=2)
        assert "ZeroDisplacementBaseline" in repr(bl)
        assert "n_beads=5" in repr(bl)


# ---------------------------------------------------------------------------
# GlobalStatsBaseline
# ---------------------------------------------------------------------------
