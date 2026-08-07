"""Evaluation module: metrics, rollout evaluation, and plotting."""

from .metrics import (
    per_step_mse,
    bond_length_deviation,
    radius_of_gyration,
    energy_drift,
    rollout_metrics
)
from .rollout import RolloutEvaluator

__all__ = [
    'per_step_mse',
    'bond_length_deviation',
    'radius_of_gyration',
    'energy_drift',
    'rollout_metrics',
    'RolloutEvaluator'
]
