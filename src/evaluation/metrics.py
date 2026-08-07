"""Metrics for evaluating PINN models."""

import numpy as np

def per_step_mse(
    pred_displacement: np.ndarray,
    true_displacement: np.ndarray,
) -> float:
    """Mean squared error of predicted vs true displacement."""
    return float(np.mean((pred_displacement - true_displacement) ** 2))

def bond_length_deviation(
    positions: np.ndarray,
    r0: float = 1.0,
) -> dict:
    """Compute bond length statistics for a linear chain.
    Returns: {'mean': float, 'std': float, 'max_deviation': float, 'violations': int}"""
    if positions.shape[0] < 2:
        return {'mean': 0.0, 'std': 0.0, 'max_deviation': 0.0, 'violations': 0}
    
    diffs = positions[1:] - positions[:-1]
    lengths = np.linalg.norm(diffs, axis=1)
    
    deviations = np.abs(lengths - r0)
    return {
        'mean': float(np.mean(lengths)),
        'std': float(np.std(lengths)),
        'max_deviation': float(np.max(deviations)),
        'violations': int(np.sum(deviations > 0.1 * r0))
    }

def radius_of_gyration(positions: np.ndarray) -> float:
    """Compute R_g for a configuration."""
    com = np.mean(positions, axis=0)
    rg_sq = np.mean(np.sum((positions - com) ** 2, axis=1))
    return float(np.sqrt(rg_sq))

def energy_drift(
    energy_series: np.ndarray,
) -> dict:
    """Compute energy drift metrics.
    Returns: {'mean': float, 'std': float, 'drift_rate': float, 'max_deviation': float}"""
    mean_energy = float(np.mean(energy_series)) if len(energy_series) > 0 else 0.0
    std_energy = float(np.std(energy_series)) if len(energy_series) > 0 else 0.0
    drift_rate = 0.0
    
    if len(energy_series) > 1:
        x = np.arange(len(energy_series))
        drift_rate = float(np.polyfit(x, energy_series, 1)[0])
        
    initial_energy = energy_series[0] if len(energy_series) > 0 else 0.0
    max_deviation = float(np.max(np.abs(energy_series - initial_energy))) if len(energy_series) > 0 else 0.0
    
    return {
        'mean': mean_energy,
        'std': std_energy,
        'drift_rate': drift_rate,
        'max_deviation': max_deviation
    }

def rollout_metrics(
    predicted_trajectory: np.ndarray,
    ground_truth_trajectory: np.ndarray,
    r0: float = 1.0,
) -> dict:
    """Comprehensive metrics for a rollout.
    Returns dict with: position_mse_per_step, bond_stats, rg_predicted, rg_true, etc."""
    T = predicted_trajectory.shape[0]
    
    position_mses = []
    bond_stats = []
    rg_predicted = []
    rg_true = []
    
    for t in range(T):
        pred_pos = predicted_trajectory[t]
        true_pos = ground_truth_trajectory[t] if t < ground_truth_trajectory.shape[0] else predicted_trajectory[t]
        
        position_mses.append(float(np.mean((pred_pos - true_pos) ** 2)))
        bond_stats.append(bond_length_deviation(pred_pos, r0))
        rg_predicted.append(radius_of_gyration(pred_pos))
        rg_true.append(radius_of_gyration(true_pos))
        
    return {
        'position_mse_per_step': position_mses,
        'bond_stats': bond_stats,
        'rg_predicted': rg_predicted,
        'rg_true': rg_true
    }
