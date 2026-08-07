# src/training/__init__.py
"""Training module: training loops, loss functions, and experiment tracking."""

from .trainer import Trainer
from .losses import displacement_mse_loss, bond_length_penalty, combined_loss
