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

class TestGlobalStatsBaseline:

    def _make_train_data(self, n_frames=500, N=30, dim=3, mean=0.01, std=0.05, seed=0):
        rng = np.random.default_rng(seed)
        return rng.normal(loc=mean, scale=std, size=(n_frames, N, dim))

    def test_fit_sets_stats(self):
        bl = GlobalStatsBaseline(n_beads=30, fit_global=True)
        data = self._make_train_data()
        bl.fit(data)
        assert bl._fitted
        assert bl._mean is not None
        assert bl._std is not None
        assert bl._mean.shape == (3,)  # global pooled → shape (dim,)

    def test_fit_per_bead(self):
        bl = GlobalStatsBaseline(n_beads=30, fit_global=False)
        data = self._make_train_data()
        bl.fit(data)
        assert bl._mean.shape == (30, 3)

    def test_predict_raises_if_unfitted(self):
        bl = GlobalStatsBaseline(n_beads=30)
        with pytest.raises(RuntimeError, match="fitted"):
            bl.predict(np.zeros((30, 3)))

    def test_predict_shape(self):
        bl = GlobalStatsBaseline(n_beads=30, fit_global=True)
        bl.fit(self._make_train_data())
        pred = bl.predict(np.zeros((30, 3)))
        assert pred.shape == (30, 3)

    def test_predict_mean_close_to_true_mean(self):
        # With many predictions, the sample mean should converge to the fitted mean
        true_mean = 0.01
        bl = GlobalStatsBaseline(n_beads=5, fit_global=True, seed=42)
        data = np.random.default_rng(0).normal(true_mean, 0.001, (1000, 5, 3))
        bl.fit(data)
        preds = np.stack([bl.predict(np.zeros((5, 3))) for _ in range(2000)])
        assert abs(preds.mean() - true_mean) < 0.002

    def test_from_stats_classmethod(self):
        mean = np.array([0.01, 0.0, -0.01])
        std = np.array([0.05, 0.05, 0.05])
        bl = GlobalStatsBaseline.from_stats(mean, std, n_beads=10)
        assert bl._fitted
        pred = bl.predict(np.zeros((10, 3)))
        assert pred.shape == (10, 3)

    def test_evaluate_one_step_mse(self):
        bl = GlobalStatsBaseline(n_beads=5, fit_global=True, seed=0)
        data = np.random.default_rng(0).normal(0.0, 0.05, (500, 5, 3))
        bl.fit(data)
        true_disp = np.zeros((50, 5, 3))
        mse = bl.evaluate_one_step_mse(true_disp, n_samples=20)
        # MSE against zeros ≈ var(pred) ≈ 0.05² = 0.0025, times dim etc.
        assert mse > 0.0
        assert np.isfinite(mse)

    def test_repr(self):
        bl = GlobalStatsBaseline(n_beads=10)
        assert "GlobalStatsBaseline" in repr(bl)
        assert "unfitted" in repr(bl)
        bl.fit(self._make_train_data(N=10))
        assert "fitted" in repr(bl)
