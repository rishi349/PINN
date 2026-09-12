"""Publication-quality plotting utilities for the Polymer GNN project.

Integrates Matplotlib + Seaborn (static) and Plotly (interactive).
All functions produce both static (PDF/PNG) and interactive (HTML) outputs.

Usage
-----
    from src.evaluation.plotting import (
        plot_training_curves,
        plot_rollout_msd,
        plot_rouse_spectrum,
        plot_bond_length_histogram,
        plot_hpo_importance,
    )
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # non-interactive backend for server/headless
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set_theme(style='whitegrid', context='paper', font_scale=1.2)
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Color palette (consistent across all plots)
# ---------------------------------------------------------------------------
COLORS = {
    'baseline': '#2196F3',       # blue
    'physics': '#FF5722',        # orange-red
    'equivariant': '#4CAF50',    # green
    'zero_baseline': '#9E9E9E',  # grey
    'random_baseline': '#795548',# brown
    'ground_truth': '#000000',   # black
    'train': '#2196F3',          # blue
    'val': '#FF9800',            # orange
}


def _save_fig(fig, path: str, dpi: int = 300):
    """Save matplotlib figure as both PNG and PDF."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(p), dpi=dpi, bbox_inches='tight')
    fig.savefig(str(p.with_suffix('.pdf')), bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. Training Curves
# ---------------------------------------------------------------------------

def plot_training_curves(
    history: Dict[str, list],
    title: str = 'Training Progress',
    save_path: str = 'figures/training_curves.png',
) -> None:
    """Plot train/val loss curves with early stopping marker.

    Parameters
    ----------
    history : dict with keys 'train_loss', 'val_loss', optional 'best_epoch'
    title : str
    save_path : str
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], color=COLORS['train'],
            label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], color=COLORS['val'],
            label='Val Loss', linewidth=2)

    if 'best_epoch' in history:
        best = history['best_epoch']
        ax.axvline(x=best, color='red', linestyle='--', alpha=0.5,
                   label=f'Best epoch ({best})')
        ax.scatter([best], [history['val_loss'][best - 1]],
                   color='red', s=100, zorder=5)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title(title)
    ax.legend()
    ax.set_yscale('log')
    _save_fig(fig, save_path)

    # Interactive Plotly version
    if PLOTLY_AVAILABLE:
        fig_plotly = go.Figure()
        fig_plotly.add_trace(go.Scatter(
            x=list(epochs), y=history['train_loss'],
            name='Train Loss', line=dict(color=COLORS['train'], width=2)
        ))
        fig_plotly.add_trace(go.Scatter(
            x=list(epochs), y=history['val_loss'],
            name='Val Loss', line=dict(color=COLORS['val'], width=2)
        ))
        fig_plotly.update_layout(
            title=title, xaxis_title='Epoch', yaxis_title='Loss (MSE)',
            yaxis_type='log', template='plotly_white'
        )
        fig_plotly.write_html(str(Path(save_path).with_suffix('.html')))


# ---------------------------------------------------------------------------
# 2. MSD Subdiffusion Plot
# ---------------------------------------------------------------------------

def plot_rollout_msd(
    lags_time: np.ndarray,
    g1: np.ndarray,
    g2: Optional[np.ndarray] = None,
    g3: Optional[np.ndarray] = None,
    g1_ref: Optional[np.ndarray] = None,
    alpha_g1: Optional[float] = None,
    title: str = 'MSD Subdiffusion Analysis',
    save_path: str = 'figures/msd_rollout.png',
) -> None:
    """Log-log MSD plot with Rouse scaling reference lines."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.loglog(lags_time, g1, color=COLORS['baseline'], linewidth=2,
              label=f'g₁ (monomer)' + (f' α={alpha_g1:.2f}' if alpha_g1 else ''))

    if g2 is not None:
        ax.loglog(lags_time, g2, color=COLORS['physics'], linewidth=2,
                  label='g₂ (monomer − COM)')
    if g3 is not None:
        ax.loglog(lags_time, g3, color=COLORS['equivariant'], linewidth=2,
                  label='g₃ (COM)')
    if g1_ref is not None:
        ax.loglog(lags_time, g1_ref, color=COLORS['ground_truth'],
                  linewidth=1.5, linestyle='--', label='Ground truth g₁')

    # Rouse reference line: t^0.5
    t_ref = lags_time[1:]
    ax.loglog(t_ref, 0.1 * t_ref**0.5, 'k:', alpha=0.3, label='t^0.5 (Rouse)')
    ax.loglog(t_ref, 0.01 * t_ref**1.0, 'k--', alpha=0.3, label='t^1.0 (diffusion)')

    ax.set_xlabel('Time (τ)')
    ax.set_ylabel('MSD (σ²)')
    ax.set_title(title)
    ax.legend(fontsize=9)
    _save_fig(fig, save_path)


# ---------------------------------------------------------------------------
# 3. Rouse Mode Spectrum
# ---------------------------------------------------------------------------

def plot_rouse_spectrum(
    tau_p: np.ndarray,
    tau_p_ref: Optional[np.ndarray] = None,
    title: str = 'Rouse Mode Relaxation Times',
    save_path: str = 'figures/rouse_spectrum.png',
) -> None:
    """Log-log plot of τ_p vs p with 1/p² reference line."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    valid = np.isfinite(tau_p) & (tau_p > 0)
    p = np.arange(len(tau_p))

    ax.scatter(p[valid], tau_p[valid], color=COLORS['baseline'], s=50,
               zorder=5, label='GNN rollout')

    if tau_p_ref is not None:
        valid_ref = np.isfinite(tau_p_ref) & (tau_p_ref > 0)
        ax.scatter(p[valid_ref], tau_p_ref[valid_ref], color=COLORS['ground_truth'],
                   s=30, marker='x', zorder=4, label='Simulation (ground truth)')

    # Reference line: τ_p = τ_1 / p²
    p_ref = np.arange(1, len(tau_p))
    if np.any(valid[1:]):
        tau_1 = tau_p[1] if np.isfinite(tau_p[1]) else 1.0
        ax.loglog(p_ref, tau_1 / p_ref**2, 'k--', alpha=0.4,
                  label='τ₁/p² (Rouse prediction)')

    ax.set_xlabel('Mode index p')
    ax.set_ylabel('Relaxation time τ_p')
    ax.set_title(title)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    _save_fig(fig, save_path)


# ---------------------------------------------------------------------------
# 4. Bond Length Histogram
# ---------------------------------------------------------------------------

def plot_bond_length_histogram(
    bond_lengths: np.ndarray,
    r0: float = 0.965,
    title: str = 'Bond Length Distribution',
    save_path: str = 'figures/bond_histogram.png',
) -> None:
    """Histogram of bond lengths with equilibrium reference line."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))

    if SEABORN_AVAILABLE:
        sns.histplot(bond_lengths, bins=60, kde=True, ax=ax,
                     color=COLORS['baseline'], alpha=0.6)
    else:
        ax.hist(bond_lengths, bins=60, color=COLORS['baseline'],
                alpha=0.6, density=True)

    ax.axvline(x=r0, color='red', linestyle='--', linewidth=2,
               label=f'r₀ = {r0:.3f}σ')
    ax.set_xlabel('Bond length (σ)')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()
    _save_fig(fig, save_path)


# ---------------------------------------------------------------------------
# 5. Model Comparison Bar Plot
# ---------------------------------------------------------------------------

def plot_model_comparison(
    model_names: List[str],
    metrics: Dict[str, List[float]],
    title: str = 'Model Comparison',
    save_path: str = 'figures/model_comparison.png',
) -> None:
    """Grouped bar plot comparing multiple models across metrics."""
    n_models = len(model_names)
    n_metrics = len(metrics)
    x = np.arange(n_models)
    width = 0.8 / n_metrics

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    colors_list = list(COLORS.values())
    for i, (metric_name, values) in enumerate(metrics.items()):
        offset = (i - n_metrics / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=metric_name,
               color=colors_list[i % len(colors_list)], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.set_ylabel('Metric Value')
    ax.set_title(title)
    ax.legend()
    _save_fig(fig, save_path)


# ---------------------------------------------------------------------------
# 6. HPO Importance Plot (Optuna)
# ---------------------------------------------------------------------------

def plot_hpo_importance(
    study,
    save_path: str = 'figures/hpo_importance.png',
) -> None:
    """Plot hyperparameter importance from an Optuna study.

    Parameters
    ----------
    study : optuna.Study
        A completed Optuna study.
    save_path : str
    """
    try:
        import optuna
        from optuna.importance import get_param_importances
        importances = get_param_importances(study)

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        params = list(importances.keys())
        values = list(importances.values())

        if SEABORN_AVAILABLE:
            sns.barplot(x=values, y=params, ax=ax, orient='h',
                        palette='viridis')
        else:
            ax.barh(params, values, color=COLORS['baseline'])

        ax.set_xlabel('Importance')
        ax.set_title('Hyperparameter Importance (Optuna)')
        _save_fig(fig, save_path)
    except Exception as e:
        print(f"Warning: could not plot HPO importance: {e}")


# ---------------------------------------------------------------------------
# 7. Rollout Stability Plot (drift over time)
# ---------------------------------------------------------------------------

def plot_rollout_stability(
    steps: np.ndarray,
    metric_values: Dict[str, np.ndarray],
    title: str = 'Rollout Stability',
    save_path: str = 'figures/rollout_stability.png',
) -> None:
    """Plot metric evolution during rollout (e.g. Rg, bond length, energy)."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for name, values in metric_values.items():
        ax.plot(steps, values, linewidth=1.5, label=name)

    ax.set_xlabel('Rollout Step')
    ax.set_ylabel('Metric Value')
    ax.set_title(title)
    ax.legend()
    _save_fig(fig, save_path)
