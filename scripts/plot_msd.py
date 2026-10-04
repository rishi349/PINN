#!/usr/bin/env python3
"""
Plot MSD vs time (mean-squared displacement) from an existing trajectory.

Extracts Rouse dynamics fingerprint directly from saved frames — no new
simulation needed.

Physics
-------
For overdamped Langevin / Rouse dynamics of a bead-spring chain:

  g1(t) = <|r_i(t0+t) - r_i(t0)|²>  (monomer MSD, averaged over beads & origins)

  Short time   (t << τ_bond):   g1 ~ t     (free Brownian diffusion)
  Intermediate (τ_bond << t << τ_R):  g1 ~ t^0.5  (Rouse subdiffusion)
  Long time    (t >> τ_R):      g1 ~ t     (whole-chain diffusion)

  g3(t) = <|R_cm(t0+t) - R_cm(t0)|²>  (center-of-mass MSD)
  g3 ~ 6·D_cm·t  at all times, D_cm = kBT / (N·γ)

Usage (from PINN/ root):
    python scripts/plot_msd.py
    python scripts/plot_msd.py --trajectory data/raw/trajectory_0000.json \\
        --output plots/msd_vs_time.png --max_frames 5000
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def compute_msd(positions, max_lag_frac=0.25, n_lag_points=80):
    """
    Compute monomer MSD g1(Δt) and center-of-mass MSD g3(Δt).

    Parameters
    ----------
    positions : np.ndarray, shape (n_frames, N, 3)
        Bead positions over time.
    max_lag_frac : float
        Maximum lag as fraction of total frames (avoid poor statistics at long lags).
    n_lag_points : int
        Number of logarithmically spaced lag values.

    Returns
    -------
    lags : np.ndarray, shape (n_lags,)
        Lag indices (in frames).
    g1 : np.ndarray, shape (n_lags,)
        Monomer MSD averaged over beads and time origins.
    g1_err : np.ndarray, shape (n_lags,)
        Standard error of g1.
    g3 : np.ndarray, shape (n_lags,)
        Center-of-mass MSD averaged over time origins.
    g3_err : np.ndarray, shape (n_lags,)
        Standard error of g3.
    """
    n_frames, N, dim = positions.shape
    max_lag = int(max_lag_frac * n_frames)

    # Logarithmically spaced lags (more resolution at short times)
    lags = np.unique(
        np.geomspace(1, max_lag, n_lag_points).astype(int)
    )
    lags = lags[lags < max_lag]

    # Center of mass trajectory
    com = np.mean(positions, axis=1)  # (n_frames, 3)

    g1_list = []
    g1_err_list = []
    g3_list = []
    g3_err_list = []

    for lag in lags:
        # Displacements for all time origins
        # monomer: shape (n_origins, N, 3)
        dr_monomer = positions[lag:] - positions[:-lag]
        # squared displacement per bead per origin
        sd_monomer = np.sum(dr_monomer ** 2, axis=2)  # (n_origins, N)
        # mean over beads for each origin
        msd_per_origin = np.mean(sd_monomer, axis=1)  # (n_origins,)

        g1_list.append(float(np.mean(msd_per_origin)))
        g1_err_list.append(float(np.std(msd_per_origin) / np.sqrt(len(msd_per_origin))))

        # COM
        dr_com = com[lag:] - com[:-lag]
        sd_com = np.sum(dr_com ** 2, axis=1)  # (n_origins,)

        g3_list.append(float(np.mean(sd_com)))
        g3_err_list.append(float(np.std(sd_com) / np.sqrt(len(sd_com))))

    return (
        lags,
        np.array(g1_list),
        np.array(g1_err_list),
        np.array(g3_list),
        np.array(g3_err_list),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Plot MSD vs time from existing trajectory data"
    )
    parser.add_argument(
        "--trajectory", type=str,
        default="data/raw/trajectory_0000.json",
        help="Path to trajectory JSON file"
    )
    parser.add_argument(
        "--output", type=str, default="plots/msd_vs_time.png",
        help="Output PNG path"
    )
    parser.add_argument(
        "--max_frames", type=int, default=None,
        help="Limit number of frames loaded (for memory). None = use all."
    )
    parser.add_argument(
        "--max_lag_frac", type=float, default=0.25,
        help="Maximum lag as fraction of total frames"
    )
    args = parser.parse_args()

    # ── load trajectory ──────────────────────────────────────────────────────
    traj_path = os.path.join(project_root, args.trajectory)
    print(f"Loading trajectory from {traj_path}...")
    with open(traj_path, "r") as f:
        data = json.load(f)

    meta = data.get("metadata", {})
    frames = data.get("frames", [])
    n_total = len(frames)
    N = meta.get("chain_length", 30)
    dt = meta.get("timestep_size", 0.001)
    save_every = 100  # from config
    gamma = meta.get("friction", 1.0)
    kBT = meta.get("target_temperature", 1.0)

    print(f"  Total frames : {n_total}")
    print(f"  Chain length : N={N}")
    print(f"  dt           : {dt}")
    print(f"  save_every   : {save_every}")
    print(f"  Frame dt     : {save_every * dt} tau")

    # Subsample if requested
    if args.max_frames is not None and n_total > args.max_frames:
        step = max(1, n_total // args.max_frames)
        frames = frames[::step]
        save_every *= step  # adjust time between frames
        print(f"  Subsampled to {len(frames)} frames (every {step}th)")

    # Extract positions: (n_frames, N, 3)
    print("Extracting positions...")
    positions = np.array([f["positions"] for f in frames])
    n_frames = len(positions)
    print(f"  Shape: {positions.shape}")

    # ── compute MSD ──────────────────────────────────────────────────────────
    print("Computing MSD...")
    frame_dt = save_every * dt  # physical time between consecutive loaded frames

    lags, g1, g1_err, g3, g3_err = compute_msd(
        positions,
        max_lag_frac=args.max_lag_frac,
        n_lag_points=80,
    )

    # Convert lag (in frames) to physical time (tau)
    t_lag = lags * frame_dt

    # ── power-law fits ───────────────────────────────────────────────────────
    # Fit g1 in the intermediate regime (exclude first 5 and last 5 points)
    n_fit = len(t_lag)
    if n_fit > 15:
        fit_start = 5
        fit_end = n_fit - 5
    elif n_fit > 6:
        fit_start = 2
        fit_end = n_fit - 2
    else:
        fit_start = 0
        fit_end = n_fit

    valid = (g1[fit_start:fit_end] > 0) & (t_lag[fit_start:fit_end] > 0)
    log_t = np.log10(t_lag[fit_start:fit_end][valid])
    log_g1 = np.log10(g1[fit_start:fit_end][valid])

    slope_g1, intercept_g1, r_g1, _, se_g1 = stats.linregress(log_t, log_g1)

    # Fit g3 (should be slope ~1.0)
    log_g3 = np.log10(g3[fit_start:fit_end][valid])
    slope_g3, intercept_g3, r_g3, _, se_g3 = stats.linregress(log_t, log_g3)

    # Theory
    tau_R = N ** 2 * frame_dt  # Rouse time in physical time (rough)
    tau_R_tau = N ** 2  # in tau units (with gamma=kBT=1)
    D_cm = kBT / (N * gamma)
    D_bead = kBT / gamma

    print("\n" + "=" * 65)
    print("  MSD FIT RESULTS")
    print("=" * 65)
    print(f"  g1 (monomer MSD) exponent : {slope_g1:.4f} +/- {se_g1:.4f}")
    print(f"    Expected: 0.5 (Rouse subdiffusion) or 1.0 (free/long-time)")
    print(f"    R2 = {r_g1**2:.4f}")
    print(f"  g3 (COM MSD) exponent     : {slope_g3:.4f} +/- {se_g3:.4f}")
    print(f"    Expected: 1.0 (diffusive)")
    print(f"    R2 = {r_g3**2:.4f}")
    print(f"  D_bead (theory)           : {D_bead:.4f}")
    print(f"  D_cm (theory)             : {D_cm:.6f}")
    print(f"  Rouse time tau_R          : {tau_R_tau} tau")
    print("=" * 65)

    # ── theory reference lines ───────────────────────────────────────────────
    t_ref = np.logspace(np.log10(t_lag.min()), np.log10(t_lag.max()), 200)
    g1_fit_line = 10 ** intercept_g1 * t_ref ** slope_g1
    g3_fit_line = 10 ** intercept_g3 * t_ref ** slope_g3
    # Pure diffusion reference: 6·D_bead·t (single free bead)
    g1_diffusive = 6 * D_bead * t_ref
    # Rouse subdiffusion reference: A * t^0.5
    # Normalize to pass through midpoint of data
    mid_idx = len(t_lag) // 2
    A_rouse = g1[mid_idx] / np.sqrt(t_lag[mid_idx])
    g1_rouse = A_rouse * np.sqrt(t_ref)
    # COM diffusion: 6·D_cm·t
    g3_diffusive = 6 * D_cm * t_ref

    # ── figure ───────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
    })

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor("#0f1117")

    for ax in axes:
        ax.set_facecolor("#1a1d27")
        for spine in ax.spines.values():
            spine.set_color("#3a3d52")
        ax.tick_params(colors="#c8cad8")
        ax.xaxis.label.set_color("#c8cad8")
        ax.yaxis.label.set_color("#c8cad8")
        ax.title.set_color("#e8eaf6")
        ax.grid(True, color="#2e3247", linewidth=0.6, zorder=0)

    # ── left panel: g1 (monomer MSD) ─────────────────────────────────────────
    ax = axes[0]
    ax.loglog(t_lag, g1, "o", color="#7c83fd", markersize=5, alpha=0.8,
              zorder=5, label="g1 (monomer MSD)")
    ax.fill_between(t_lag, g1 - 2 * g1_err, g1 + 2 * g1_err,
                     color="#7c83fd", alpha=0.15, zorder=3)
    ax.loglog(t_ref, g1_fit_line, "--", color="#ff6b6b", linewidth=2,
              label=f"Fit: t^{slope_g1:.3f} (R2={r_g1**2:.3f})", zorder=4)
    ax.loglog(t_ref, g1_rouse, ":", color="#ffd166", linewidth=1.8,
              label="Rouse: t^0.5", zorder=3)
    ax.loglog(t_ref, g1_diffusive, "-.", color="#06d6a0", linewidth=1.8,
              label="Free diffusion: t^1.0", zorder=3)
    # Mark Rouse time
    if tau_R_tau * dt < t_lag.max():
        rouse_t = tau_R_tau * dt
    else:
        rouse_t = None
    if rouse_t and rouse_t > t_lag.min():
        ax.axvline(rouse_t, color="#c77dff", linestyle="--", alpha=0.6,
                   linewidth=1.5, label=f"tau_R = {tau_R_tau} tau")

    ax.set_xlabel("Time lag (tau)", fontsize=13)
    ax.set_ylabel("g1(t) = <|r(t0+t) - r(t0)|^2>", fontsize=13)
    ax.set_title("Monomer MSD", fontsize=14)
    ax.legend(fontsize=9, framealpha=0.2, labelcolor="#e8eaf6",
              facecolor="#1a1d27", edgecolor="#3a3d52", loc="upper left")

    ax.text(
        0.95, 0.05,
        f"N={N}, gamma={gamma}, kBT={kBT}\n"
        f"Frames: {n_frames}\n"
        f"Exponent: {slope_g1:.3f} +/- {se_g1:.3f}",
        transform=ax.transAxes, fontsize=9,
        verticalalignment="bottom", horizontalalignment="right",
        color="#e8eaf6",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#2a2d3e",
                  alpha=0.8, edgecolor="#3a3d52")
    )

    # ── right panel: g3 (COM MSD) ────────────────────────────────────────────
    ax = axes[1]
    ax.loglog(t_lag, g3, "s", color="#06d6a0", markersize=5, alpha=0.8,
              zorder=5, label="g3 (COM MSD)")
    ax.fill_between(t_lag, g3 - 2 * g3_err, g3 + 2 * g3_err,
                     color="#06d6a0", alpha=0.15, zorder=3)
    ax.loglog(t_ref, g3_fit_line, "--", color="#ff6b6b", linewidth=2,
              label=f"Fit: t^{slope_g3:.3f} (R2={r_g3**2:.3f})", zorder=4)
    ax.loglog(t_ref, g3_diffusive, ":", color="#ffd166", linewidth=1.8,
              label=f"Theory: 6*D_cm*t (D_cm={D_cm:.4f})", zorder=3)

    ax.set_xlabel("Time lag (tau)", fontsize=13)
    ax.set_ylabel("g3(t) = <|R_cm(t0+t) - R_cm(t0)|^2>", fontsize=13)
    ax.set_title("Center-of-Mass MSD", fontsize=14)
    ax.legend(fontsize=9, framealpha=0.2, labelcolor="#e8eaf6",
              facecolor="#1a1d27", edgecolor="#3a3d52", loc="upper left")

    ax.text(
        0.95, 0.05,
        f"D_cm (theory) = {D_cm:.5f}\n"
        f"Exponent: {slope_g3:.3f} +/- {se_g3:.3f}",
        transform=ax.transAxes, fontsize=9,
        verticalalignment="bottom", horizontalalignment="right",
        color="#e8eaf6",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#2a2d3e",
                  alpha=0.8, edgecolor="#3a3d52")
    )

    fig.suptitle(
        f"Mean-Squared Displacement | N={N}, kBT={kBT}, gamma={gamma}",
        fontsize=15, color="#e8eaf6", y=1.01
    )
    fig.tight_layout(pad=2.0)

    out_path = os.path.join(project_root, args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\n  Saved plot -> {out_path}")


if __name__ == "__main__":
    main()
