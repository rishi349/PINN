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
