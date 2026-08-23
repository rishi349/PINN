"""Naive baselines for polymer GNN evaluation (§10.2, Month 4).

Two baselines required by the execution plan before any GNN work:
  0a. ZeroDisplacementBaseline  — predicts zero motion for all beads
  0b. GlobalStatsBaseline       — samples displacement from the training
                                  set's marginal displacement distribution

Both implement the same .predict(positions) interface as the trained GNN
models, so they slot directly into RolloutEvaluator and scripts/evaluate.py
without any special-casing.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


class ZeroDisplacementBaseline:
    """Baseline 0a: predict zero displacement for every bead at every step.

    This is the most conservative possible predictor. A GNN that cannot beat
    this baseline has learned nothing — it is worse than assuming the chain
    does not move.

    Useful as a lower-bound MSE reference: any model whose one-step MSE
    exceeds the zero-displacement MSE on the test set has negative value.

    Parameters
    ----------
    n_beads : int
        Number of beads in the chain.
    dim : int
        Spatial dimension (default 3).
    """

    def __init__(self, n_beads: int = 30, dim: int = 3):
        self.n_beads = n_beads
        self.dim = dim

    def predict(self, positions: np.ndarray) -> np.ndarray:
        """Return zero displacement for all beads.

        Parameters
        ----------
        positions : np.ndarray, shape (N, dim)
            Current bead positions (not used — baseline ignores state).

        Returns
        -------
        np.ndarray, shape (N, dim)
            Zero array.
        """
        return np.zeros_like(positions)

    def evaluate_one_step_mse(
        self,
        true_displacements: np.ndarray,
    ) -> float:
        """MSE of zero predictor against true displacements.

        MSE = mean(||Δr_true||²) — equals the mean squared displacement
        of the ground-truth trajectory.

        Parameters
        ----------
        true_displacements : np.ndarray, shape (n_frames, N, dim)
            True displacements from the test set.

        Returns
        -------
        float
            One-step MSE of the zero-displacement baseline.
        """
        return float(np.mean(true_displacements ** 2))

    def __repr__(self) -> str:
        return f"ZeroDisplacementBaseline(n_beads={self.n_beads}, dim={self.dim})"


class GlobalStatsBaseline:
    """Baseline 0b: sample displacement from the training set marginal.

    Fits a per-bead Gaussian to the training displacement distribution
    (mean and std per spatial dimension), then samples from it at inference
    time. This tests whether the GNN has learned anything beyond the
    marginal distribution of displacements — a model beaten by this
    baseline has no useful structural awareness.

    Parameters
    ----------
    n_beads : int
        Number of beads in the chain.
    dim : int
        Spatial dimension (default 3).
    seed : int
        Random seed for reproducible sampling.
    fit_global : bool
        If True (default), fit a single Gaussian over all beads and frames
        (pooled stats). If False, fit per-bead statistics independently.
        Pooled is safer with limited data.
    """

    def __init__(
        self,
        n_beads: int = 30,
        dim: int = 3,
        seed: int = 42,
        fit_global: bool = True,
    ):
        self.n_beads = n_beads
        self.dim = dim
        self.rng = np.random.default_rng(seed)
        self.fit_global = fit_global

        # Filled by .fit()
        self._mean: Optional[np.ndarray] = None  # shape (N, dim) or (dim,)
        self._std: Optional[np.ndarray] = None   # shape (N, dim) or (dim,)
        self._fitted = False

    def fit(self, train_displacements: np.ndarray) -> "GlobalStatsBaseline":
        """Fit marginal statistics from training displacements.

        Parameters
        ----------
        train_displacements : np.ndarray, shape (n_frames, N, dim)
            True displacements from the training split.

        Returns
        -------
        self
        """
        if train_displacements.ndim == 2:
            # (N, dim) — single frame, reshape
            train_displacements = train_displacements[np.newaxis]

        if self.fit_global:
            # Pool over all beads and frames → shape (dim,)
            flat = train_displacements.reshape(-1, self.dim)
            self._mean = flat.mean(axis=0)         # (dim,)
            self._std = flat.std(axis=0) + 1e-8    # (dim,)
        else:
            # Per-bead stats → shape (N, dim)
            self._mean = train_displacements.mean(axis=0)           # (N, dim)
            self._std = train_displacements.std(axis=0) + 1e-8      # (N, dim)

        self._fitted = True
        return self

    @classmethod
    def from_stats(
        cls,
        mean: np.ndarray,
        std: np.ndarray,
        n_beads: int = 30,
        dim: int = 3,
        seed: int = 42,
    ) -> "GlobalStatsBaseline":
        """Construct baseline from pre-computed statistics.

        Useful when normalization stats are already saved to disk.

        Parameters
        ----------
        mean : np.ndarray
            Displacement mean, shape (dim,) or (N, dim).
        std : np.ndarray
            Displacement std, shape (dim,) or (N, dim).
        """
        obj = cls(n_beads=n_beads, dim=dim, seed=seed)
        obj._mean = mean
        obj._std = std
        obj._fitted = True
        return obj

    def predict(self, positions: np.ndarray) -> np.ndarray:
        """Sample a displacement from the fitted marginal distribution.

        Parameters
        ----------
        positions : np.ndarray, shape (N, dim)
            Current bead positions (not used — baseline is state-unaware).

        Returns
        -------
        np.ndarray, shape (N, dim)
            Sampled displacement.
        """
        if not self._fitted:
            raise RuntimeError(
                "GlobalStatsBaseline must be fitted before calling predict(). "
                "Call .fit(train_displacements) first."
            )
        N = positions.shape[0]
        noise = self.rng.standard_normal((N, self.dim))
        return self._mean + self._std * noise

    def evaluate_one_step_mse(
        self,
        true_displacements: np.ndarray,
        n_samples: int = 100,
    ) -> float:
        """Estimate MSE of this baseline via Monte Carlo sampling.

        Because this baseline is stochastic, we average over multiple
        independent prediction sets to get a stable MSE estimate.

        Parameters
        ----------
        true_displacements : np.ndarray, shape (n_frames, N, dim)
            True displacements from the test set.
        n_samples : int
            Number of independent prediction draws to average over.

        Returns
        -------
        float
            Expected one-step MSE of the global-stats baseline.
        """
        if not self._fitted:
            raise RuntimeError("Must call .fit() before .evaluate_one_step_mse().")

        mse_sum = 0.0
        n_frames, N, dim = true_displacements.shape
        for _ in range(n_samples):
            # Sample one displacement per frame (state-unaware, so positions irrelevant)
            dummy_pos = np.zeros((N, dim))
            preds = np.stack([self.predict(dummy_pos) for _ in range(n_frames)])
            mse_sum += float(np.mean((preds - true_displacements) ** 2))
        return mse_sum / n_samples

    def __repr__(self) -> str:
        fitted_str = "fitted" if self._fitted else "unfitted"
        return (
            f"GlobalStatsBaseline(n_beads={self.n_beads}, dim={self.dim}, "
            f"fit_global={self.fit_global}, {fitted_str})"
        )
