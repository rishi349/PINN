"""Metrics for evaluating polymer GNN models (§12 metrics checklist).

One-step MSE · bond-length error · excluded-volume violation rate ·
long-horizon rollout stability · steps-until-divergence · R_g distribution ·
end-to-end distance · MSD · energy drift.
"""

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

def end_to_end_distance(positions: np.ndarray) -> float:
    """Compute end-to-end distance |r_N - r_1| for a linear chain (§12)."""
    return float(np.linalg.norm(positions[-1] - positions[0]))

def excluded_volume_violation_rate(
    positions: np.ndarray,
    sigma: float = 1.0,
) -> dict:
    """Compute excluded-volume violation rate (§12, §13.1).

    Counts the fraction of nonbonded pairs (|i-j| > 1) with distance
    below the WCA cutoff (2^(1/6) * sigma). After equilibration this
    should be approximately 0.

    Parameters
    ----------
    positions : np.ndarray, shape (N, dim)
        Bead positions.
    sigma : float
        WCA length scale. Standard: 1.0.

    Returns
    -------
    dict
        'violation_count': number of nonbonded pairs below cutoff
        'total_pairs': total nonbonded pairs checked
        'violation_rate': fraction of pairs violating excluded volume
    """
    N = positions.shape[0]
    r_cut = sigma * 2.0 ** (1.0 / 6.0)

    i_idx, j_idx = np.triu_indices(N, k=2)  # |i-j| > 1
    if len(i_idx) == 0:
        return {'violation_count': 0, 'total_pairs': 0, 'violation_rate': 0.0}

    r_ij = positions[j_idx] - positions[i_idx]
    dist = np.linalg.norm(r_ij, axis=1)
    violations = int(np.sum(dist < r_cut))

    return {
        'violation_count': violations,
        'total_pairs': len(i_idx),
        'violation_rate': float(violations / len(i_idx)),
    }

def steps_until_divergence(
    rollout_trajectory: np.ndarray,
    max_position: float = 1000.0,
    max_bond_length: float = 5.0,
    min_bond_length: float = 0.1,
) -> int:
    """Compute steps-until-divergence for a rollout (§12).

    Returns the first step index at which the rollout is considered
    diverged (NaN, extreme positions, or broken bonds). Returns -1
    if the rollout never diverges (i.e., fully stable).

    This is the "clean, cheap pass/fail number" the execution plan §12
    calls out as a key metric.

    Parameters
    ----------
    rollout_trajectory : np.ndarray, shape (n_steps, N, dim)
        Positions at each rollout step.
    max_position : float
        Maximum absolute coordinate before declaring blowup.
    max_bond_length : float
        Maximum allowed bond length.
    min_bond_length : float
        Minimum allowed bond length.

    Returns
    -------
    int
        Step index of first divergence, or -1 if stable throughout.
    """
    for step in range(rollout_trajectory.shape[0]):
        pos = rollout_trajectory[step]

        # NaN / Inf check
        if np.any(np.isnan(pos)) or np.any(np.isinf(pos)):
            return step

        # Extreme position check
        if np.max(np.abs(pos)) > max_position:
            return step

        # Bond integrity check
        if pos.shape[0] > 1:
            diffs = pos[1:] - pos[:-1]
            lengths = np.linalg.norm(diffs, axis=1)
            if np.any(lengths > max_bond_length) or np.any(lengths < min_bond_length):
                return step

    return -1  # Never diverged

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
        'rg_true': rg_true,
        'steps_until_div': steps_until_divergence(predicted_trajectory),
    }

