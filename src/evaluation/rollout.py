"""Autoregressive rollout evaluator for polymer GNN models (§8.7)."""

import numpy as np
import torch
import torch.nn as nn
from src.data.graph_construction import frame_to_pyg_data
from src.data.normalization import DisplacementNormalizer

class RolloutEvaluator:
    def __init__(
        self,
        model: nn.Module,
        n_beads: int = 30,
        neighbor_cutoff: float = 2.5,
        device: str = 'cpu',
        normalizer: DisplacementNormalizer = None,
    ):
        self.model = model
        self.n_beads = n_beads
        self.neighbor_cutoff = neighbor_cutoff
        self.device = device
        self.normalizer = normalizer
        self.model.to(self.device)
        self.model.eval()

    def rollout(
        self,
        initial_positions: np.ndarray,
        n_steps: int = 100,
    ) -> dict:
        """Perform autoregressive rollout.
        
        At each step:
        1. Construct graph from current positions
        2. Predict displacement with model
        3. Update positions: new_pos = old_pos + predicted_displacement
        4. Check for blowups (NaN, extreme positions, broken bonds)
        
        Returns dict with:
        - 'trajectory': np.ndarray (n_steps+1, N, 3)
        - 'blowup_step': int or None
        - 'bond_lengths': list of arrays
        - 'rg': list of floats
        """
        trajectory = [initial_positions]
        bond_lengths = []
        rg = []
        
        current_positions = initial_positions.copy()
        blowup_step = None
        
        for step in range(n_steps):
            # Record metrics for the current step
            if current_positions.shape[0] > 1:
                diffs = current_positions[1:] - current_positions[:-1]
                lengths = np.linalg.norm(diffs, axis=1)
                bond_lengths.append(lengths)
            else:
                lengths = np.array([])
                bond_lengths.append(lengths)
                
            com = np.mean(current_positions, axis=0)
            rg_val = np.sqrt(np.mean(np.sum((current_positions - com) ** 2, axis=1)))
            rg.append(float(rg_val))
            
            # Check blowup conditions
            if np.any(np.isnan(current_positions)) or np.any(np.isinf(current_positions)):
                blowup_step = step
                break
            if np.max(np.abs(current_positions)) > 1000.0: # 1000σ
                blowup_step = step
                break
            if len(lengths) > 0 and (np.any(lengths > 5.0) or np.any(lengths < 0.1)):
                blowup_step = step
                break
                
            # Dummy next frame
            dummy_next = np.zeros_like(current_positions)
            
            # Construct graph
            data = frame_to_pyg_data(
                positions_t=current_positions,
                positions_t1=dummy_next,
                neighbor_cutoff=self.neighbor_cutoff,
                n_beads=self.n_beads,
            ).to(self.device)
            
            with torch.no_grad():
                pred_tensor = self.model(data)
                if self.normalizer is not None:
                    pred_tensor = self.normalizer.denormalize(pred_tensor)
                pred_displacement = pred_tensor.cpu().numpy()
                
            current_positions = current_positions + pred_displacement
            trajectory.append(current_positions)
            
        # Process the final frame if it didn't blow up (or record it anyway)
        if blowup_step is None and current_positions.shape[0] > 1:
            diffs = current_positions[1:] - current_positions[:-1]
            lengths = np.linalg.norm(diffs, axis=1)
            bond_lengths.append(lengths)
            com = np.mean(current_positions, axis=0)
            rg_val = np.sqrt(np.mean(np.sum((current_positions - com) ** 2, axis=1)))
            rg.append(float(rg_val))
            
        return {
            'trajectory': np.stack(trajectory),
            'blowup_step': blowup_step,
            'bond_lengths': bond_lengths,
            'rg': rg
        }

    def evaluate_rollout_lengths(
        self,
        initial_positions: np.ndarray,
        ground_truth_trajectory: np.ndarray,
        rollout_lengths: list = [100, 500, 1000],
    ) -> dict:
        """Evaluate at multiple rollout lengths."""
        results = {}
        for length in rollout_lengths:
            n_steps = min(length, ground_truth_trajectory.shape[0] - 1)
            if n_steps <= 0:
                continue
                
            results[f'length_{length}'] = self.rollout(
                initial_positions=initial_positions,
                n_steps=n_steps
            )
            
        return results
