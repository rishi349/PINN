#!/usr/bin/env python3
"""
Plot ⟨Rg²⟩ vs N (radius of gyration squared vs chain length).

For a 3D polymer with excluded volume (self-avoiding walk, SAW):
    ⟨Rg²⟩ ∝ N^(2ν)   with ν ≈ 0.588  (Flory exponent, SAW / good solvent)

For an ideal Gaussian chain (no excluded volume):
    ⟨Rg²⟩ = b²N / 6  →  exponent 2ν = 1.0

This script:
  1. Runs short equilibrium simulations for each N in N_VALUES.
  2. Time-averages ⟨Rg²⟩ over production frames.
  3. Fits log(⟨Rg²⟩) = 2ν·log(N) + const via linear regression.
  4. Plots the data, the fit, and the ideal (Gaussian) reference line.

Usage (from the PINN/ root):
    python scripts/plot_rg2_vs_N.py
    python scripts/plot_rg2_vs_N.py --config configs/short.yaml --output plots/rg2_vs_N.png
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats

# ── project root on path ────────────────────────────────────────────────────
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.simulator.numpy_simulator import NumpySimulator, load_config
from src.physics.forces import compute_all_forces
from src.physics.integrators import euler_maruyama_overdamped_step


# ── default chain lengths to sweep ──────────────────────────────────────────
DEFAULT_N_VALUES = [5, 10, 15, 20, 30, 40, 50, 70, 100]

# ── steps per N  (short but enough to sample equilibrium) ───────────────────
# Rouse time τ_R ≈ N²·τ (in reduced units with γ=kBT=1).
# We use T_total = 5·τ_R, burn-in = 3·τ_R, save every 100 steps.
DT = 0.001
ROUSE_PREFACTOR_BURNIN = 3   # multiples of τ_R for burn-in
ROUSE_PREFACTOR_PROD   = 2   # multiples of τ_R for production
SAVE_EVERY = 100


def rg2_for_chain(N, config_template, seed, verbose):
    """
    Run an equilibrium simulation for a chain of length N and return
    (mean_Rg2, sem_Rg2) averaged over production frames.
    """
    import copy
    cfg = copy.deepcopy(config_template)

    cfg["chain"]["N"] = N
    tau_R = N * N
    n_burnin = max(int(ROUSE_PREFACTOR_BURNIN * tau_R / DT), 2000)
    n_prod   = max(int(ROUSE_PREFACTOR_PROD   * tau_R / DT), 1000)
    cfg["simulation"]["T_steps"]    = n_burnin + n_prod
    cfg["simulation"]["n_burnin"]   = n_burnin
    cfg["simulation"]["save_every"] = SAVE_EVERY
    cfg["integrator"]["dt"]         = DT
    cfg["box"]["L"] = max(10 * N, 100.0)

    sim = NumpySimulator(cfg)
    rng = np.random.default_rng(seed)
    pos = sim.initialize_chain(rng)

    Rg2_samples = []

    for t in range(1, sim.T_steps + 1):
        forces, _, _ = compute_all_forces(
            pos,
            bond_type=sim.bond_type,
            k_bond=sim.k_bond,
            r0=sim.r0,
            k_fene=sim.k_fene,
            R0_fene=sim.R0_fene,
            epsilon=sim.epsilon,
            sigma=sim.sigma,
        )
        pos, _ = euler_maruyama_overdamped_step(pos, forces, DT, sim.gamma, sim.kBT, rng)

        if t > sim.n_burnin and t % sim.save_every == 0:
            obs = sim.compute_observables(pos)
            Rg2_samples.append(obs["Rg_squared"])

    Rg2 = np.array(Rg2_samples)
    if len(Rg2) == 0:
        return float("nan"), float("nan")

    mean_Rg2 = float(np.mean(Rg2))
    sem_Rg2  = float(np.std(Rg2) / np.sqrt(len(Rg2)))

    if verbose:
        print(f"  N={N:4d}  frames={len(Rg2):5d}  <Rg2>={mean_Rg2:.4f}  +/-{sem_Rg2:.4f}")

    return mean_Rg2, sem_Rg2


def main():
    parser = argparse.ArgumentParser(
        description="Plot <Rg2> vs N with Flory-exponent fit"
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Base YAML config (chain length and sim steps are overridden per N)"
    )
    parser.add_argument(
        "--output", type=str, default="plots/rg2_vs_N.png",
        help="Output PNG path"
    )
    parser.add_argument(
        "--N_values", type=int, nargs="+", default=DEFAULT_N_VALUES,
        help="Chain lengths to sweep (e.g. --N_values 5 10 20 50 100)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed (each N gets seed + N)"
    )
    parser.add_argument(
        "--max_steps_per_N", type=int, default=None,
        help="Hard cap on total steps per N (burn-in=60%%, prod=40%%). "
             "Use e.g. 50000 for a quick sanity check."
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-step output"
    )
    args = parser.parse_args()

    config_path = os.path.join(project_root, args.config)
    config = load_config(config_path)

    N_values = sorted(args.N_values)
    verbose  = not args.quiet

    print("=" * 65)
    print("  Rg2 vs N -- Flory scaling analysis")
    print("=" * 65)
    print(f"  Config          : {args.config}")
    print(f"  Chain lengths N : {N_values}")
    print(f"  Bond type       : {config['bond']['type']}")
    print(f"  kBT             : {config['thermostat']['kBT']}")
    print(f"  Output          : {args.output}")
    print("=" * 65)

    mean_Rg2_list = []
    sem_Rg2_list  = []

    for N in N_values:
        tau_R = N * N
        n_burnin_steps = max(int(ROUSE_PREFACTOR_BURNIN * tau_R / DT), 2000)
        n_prod_steps   = max(int(ROUSE_PREFACTOR_PROD   * tau_R / DT), 1000)
        # Hard cap for quick runs
        if args.max_steps_per_N is not None:
            cap = args.max_steps_per_N
            n_burnin_steps = min(n_burnin_steps, int(cap * 0.6))
            n_prod_steps   = min(n_prod_steps,   max(int(cap * 0.4), 500))
        n_frames       = n_prod_steps // SAVE_EVERY
        print(f"\nN={N}: tau_R~{tau_R}tau  burn-in={n_burnin_steps} steps  "
              f"prod={n_prod_steps} steps  expected frames~{n_frames}")

        mean_Rg2, sem_Rg2 = rg2_for_chain(
            N, config, seed=args.seed + N, verbose=verbose
        )
        if np.isnan(mean_Rg2):
            print(f"  WARNING: N={N} produced no frames — increase steps or lower N.")
        mean_Rg2_list.append(mean_Rg2)
        sem_Rg2_list.append(sem_Rg2)

    N_arr   = np.array(N_values,      dtype=float)
    Rg2_arr = np.array(mean_Rg2_list, dtype=float)
    sem_arr = np.array(sem_Rg2_list,  dtype=float)

    # ── power-law fit: log(Rg2) = 2nu * log(N) + C ─────────────────────────
    valid     = np.isfinite(Rg2_arr) & (Rg2_arr > 0)
    log_N     = np.log(N_arr[valid])
    log_Rg2   = np.log(Rg2_arr[valid])
    slope, intercept, r_val, p_val, se_slope = stats.linregress(log_N, log_Rg2)
    nu_fit    = slope / 2.0
    prefactor = np.exp(intercept)

    print("\n" + "=" * 65)
    print("  FIT RESULTS")
    print("=" * 65)
    print(f"  log(<Rg2>) = {slope:.4f}*log(N) + {intercept:.4f}")
    print(f"  => 2nu (measured) = {slope:.4f}  (expected SAW ~ 1.176)")
    print(f"  => nu  (measured) = {nu_fit:.4f}  (expected SAW ~ 0.588)")
    print(f"  R2              = {r_val**2:.4f}")
    print(f"  slope SE        = +/-{se_slope:.4f}  (+/-{se_slope/2:.4f} in nu)")
    print("=" * 65)

    # ── reference curves ─────────────────────────────────────────────────────
    N_ref       = np.logspace(np.log10(N_arr.min()), np.log10(N_arr.max()), 200)
    nu_SAW      = 0.588
    Rg2_SAW     = prefactor * N_ref ** (2 * nu_SAW)
    b2          = config["bond"]["r0"] ** 2 + config["thermostat"]["kBT"] / config["bond"]["k_bond"]
    Rg2_ideal   = (b2 / 6.0) * N_ref
    Rg2_fit_line = prefactor * N_ref ** slope

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

    # ── left panel: log-log ──────────────────────────────────────────────────
    ax = axes[0]
    ax.errorbar(
        N_arr[valid], Rg2_arr[valid], yerr=2 * sem_arr[valid],
        fmt="o", color="#7c83fd", markersize=8, linewidth=1.5,
        ecolor="#a0a4ff", capsize=4, capthick=1.5,
        zorder=5, label="Simulation (+/-2 SEM)"
    )
    ax.plot(N_ref, Rg2_fit_line, "--",
            color="#ff6b6b", linewidth=2,
            label=f"Fit: N^{slope:.3f}  (nu={nu_fit:.3f})",
            zorder=4)
    ax.plot(N_ref, Rg2_SAW, ":",
            color="#ffd166", linewidth=1.8,
            label="SAW theory: nu=0.588",
            zorder=3)
    ax.plot(N_ref, Rg2_ideal, "-.",
            color="#06d6a0", linewidth=1.8,
            label="Ideal chain: nu=0.5",
            zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Chain length N", fontsize=13)
    ax.set_ylabel("<Rg2> (sigma^2)", fontsize=13)
    ax.set_title("<Rg2> vs N -- log-log", fontsize=14)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.legend(fontsize=9.5, framealpha=0.2, labelcolor="#e8eaf6",
              facecolor="#1a1d27", edgecolor="#3a3d52")
    ax.text(
        0.05, 0.95,
        f"2*nu_fit = {slope:.3f} +/- {se_slope:.3f}\nR2 = {r_val**2:.4f}",
        transform=ax.transAxes, fontsize=10, verticalalignment="top",
        color="#e8eaf6",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#2a2d3e",
                  alpha=0.8, edgecolor="#3a3d52")
    )

    # ── right panel: linear scale ─────────────────────────────────────────────
    ax = axes[1]
    ax.fill_between(
        N_ref,
        Rg2_fit_line * np.exp(-se_slope * np.log(N_ref)),
        Rg2_fit_line * np.exp(+se_slope * np.log(N_ref)),
        color="#ff6b6b", alpha=0.15, label="Fit +/-1 sigma band"
    )
    ax.plot(N_ref, Rg2_fit_line, "--",
            color="#ff6b6b", linewidth=2,
            label=f"Fit: N^{slope:.3f}")
    ax.plot(N_ref, Rg2_SAW, ":",
            color="#ffd166", linewidth=1.8, label="SAW: nu=0.588")
    ax.plot(N_ref, Rg2_ideal, "-.",
            color="#06d6a0", linewidth=1.8, label="Ideal: nu=0.5")
    ax.errorbar(
        N_arr[valid], Rg2_arr[valid], yerr=2 * sem_arr[valid],
        fmt="o", color="#7c83fd", markersize=8, linewidth=1.5,
        ecolor="#a0a4ff", capsize=4, capthick=1.5,
        zorder=5, label="Simulation (+/-2 SEM)"
    )
    ax.set_xlabel("Chain length N", fontsize=13)
    ax.set_ylabel("<Rg2> (sigma^2)", fontsize=13)
    ax.set_title("<Rg2> vs N -- linear scale", fontsize=14)
    ax.legend(fontsize=9.5, framealpha=0.2, labelcolor="#e8eaf6",
              facecolor="#1a1d27", edgecolor="#3a3d52")

    fig.suptitle(
        f"Flory Scaling: <Rg2> ~ N^(2*nu)   "
        f"(bond={config['bond']['type']}, kBT={config['thermostat']['kBT']})",
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
