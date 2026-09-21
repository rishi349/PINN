#!/usr/bin/env python3
"""
Plot average force vs end-to-end extension for a polymer chain.

Physics background
------------------
A freely-jointed chain (FJC) of N bonds of length b stretched to end-to-end
distance z obeys:
    z/L = L(F*b/kBT) = coth(x) - 1/x     x = F*b/kBT,  L = contour length N*b
    => F = (kBT/b) * L^{-1}(z/L)  [inverse Langevin]

The worm-like chain (WLC) interpolation formula (Marko & Siggia 1995):
    F * Lp / kBT = 1/4 * (1 - z/L)^{-2} - 1/4 + z/L

The harmonic / linear-elastic regime (small extensions):
    F = 3*kBT/(N*b²) * z   (entropic spring)

This script
-----------
1. For a range of fixed end-to-end extensions z, clamps the two end beads
   at ±z/2 along the x-axis and runs a short constrained MD.
2. Measures the mean force on the end beads along x (restoring force F).
3. Plots F vs z together with FJC, WLC, and linear-elastic theory curves.

Usage (from the PINN/ root):
    python scripts/plot_force_extension.py
    python scripts/plot_force_extension.py --config configs/default.yaml \\
        --N 30 --n_ext 20 --output plots/force_extension.png
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ── project root on path ────────────────────────────────────────────────────
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.simulator.numpy_simulator import NumpySimulator, load_config
from src.physics.forces import compute_all_forces
from src.physics.integrators import euler_maruyama_overdamped_step


# ── FJC / WLC analytical curves ─────────────────────────────────────────────

def _langevin(x):
    """Langevin function L(x) = coth(x) - 1/x."""
    if np.isscalar(x):
        if abs(x) < 1e-6:
            return x / 3.0
        return 1.0 / np.tanh(x) - 1.0 / x
    out = np.zeros_like(x, dtype=float)
    small = np.abs(x) < 1e-6
    out[small] = x[small] / 3.0
    big = ~small
    out[big] = 1.0 / np.tanh(x[big]) - 1.0 / x[big]
    return out


def _inverse_langevin(y, n_iter=50):
    """
    Numerical inverse Langevin: given y = L(x), find x.
    Uses Newton-Raphson on f(x) = L(x) - y.
    """
    y = np.asarray(y, dtype=float)
    # Initial guess: Cohen (1991) approximation
    x = y * (3.0 - y ** 2) / (1.0 - y ** 2)
    for _ in range(n_iter):
        Lx = _langevin(x)
        # Derivative of L: dL/dx = 1/x^2 - 1/sinh(x)^2  = -d/dx[L]
        # More stable: use 1 - L(x)^2 - L(x)/x ??? use numerical diff
        # dL/dx = -(coth^2(x)-1) + 1/x^2  = 1/x^2 - csch^2(x)
        with np.errstate(over="ignore", invalid="ignore"):
            csch2 = 1.0 / np.sinh(x) ** 2
            dLdx  = 1.0 / x ** 2 - csch2
            small = np.abs(x) < 1e-6
            dLdx[small] = 1.0 / 3.0 - x[small] ** 2 / 15.0  # series
        dx = (Lx - y) / dLdx
        x -= dx
    return x


def fjc_force(z_over_L, kBT, b, N):
    """FJC force: F = (kBT/b) * L^{-1}(z/L)."""
    # clip to avoid singularity at z/L -> 1
    y = np.clip(z_over_L, 0.0, 0.999)
    x = _inverse_langevin(y)
    return (kBT / b) * x


def wlc_force(z, L_contour, Lp, kBT):
    """WLC (Marko-Siggia): F = kBT/Lp * [1/(4*(1-z/L)^2) - 1/4 + z/L]."""
    x = np.clip(z / L_contour, 0.0, 0.999)
    return (kBT / Lp) * (0.25 / (1 - x) ** 2 - 0.25 + x)


def entropic_spring_force(z, N, b, kBT):
    """Linear (harmonic entropic spring): F = 3*kBT/(N*b^2) * z."""
    return 3.0 * kBT / (N * b ** 2) * z


# ── constrained simulation ───────────────────────────────────────────────────

def measure_force_at_extension(
    z_target: float,
    N: int,
    config: dict,
    n_equil: int,
    n_sample: int,
    sample_every: int,
    seed: int,
    verbose: bool,
) -> tuple:
    """
    Clamp bead 0 at x=-z/2 and bead N-1 at x=+z/2 along the x-axis.
    Run overdamped MD for the interior beads (SETTLE-like).
    Return (mean_force, sem_force) = time-average of |F_x on end bead|.

    Clamping is implemented by zeroing the force and displacement of
    the two end beads at every step (Lagrangian constraint: fixed position).
    """
    import copy
    cfg = copy.deepcopy(config)
    cfg["chain"]["N"] = N
    cfg["integrator"]["dt"] = 0.001
    cfg["box"]["L"] = max(10 * N, 200.0)

    sim = NumpySimulator(cfg)
    rng = np.random.default_rng(seed)

    # ── initial configuration: straight chain along x ────────────────────────
    # Always init at spacing >= sigma to avoid WCA core overlap.
    # After equil the interior beads will relax to their constrained equilibrium.
    sigma_wca = cfg["wca"]["sigma"]
    safe_spacing = max(sigma_wca * 1.05, z_target / max(N - 1, 1))
    init_span = safe_spacing * (N - 1)
    pos = np.zeros((N, 3))
    pos[:, 0] = np.linspace(-init_span / 2.0, init_span / 2.0, N)

    dt = sim.dt

    force_samples = []

    total_steps = n_equil + n_sample
    for t in range(1, total_steps + 1):
        # Enforce endpoint positions (clamp BEFORE force computation)
        pos[0,  0] = -z_target / 2.0
        pos[0,  1] = 0.0
        pos[0,  2] = 0.0
        pos[-1, 0] = +z_target / 2.0
        pos[-1, 1] = 0.0
        pos[-1, 2] = 0.0

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

        # ── sample restoring tension BEFORE updating ───────────────────────
        # We want the SIGNED tension along the chain axis (+x).
        # Bead 0 at -z/2: forces[0,0] > 0 means chain pulls it in +x (tensile).
        # Bead N-1 at +z/2: forces[-1,0] < 0 means chain pulls it in -x (tensile).
        # Signed tension = 0.5 * (forces[0,0] - forces[-1,0]):
        #   positive  => chain is under tension (stretched)
        #   negative  => chain is under compression (shorter than eq)
        if t > n_equil and (t - n_equil) % sample_every == 0:
            tension = 0.5 * (forces[0, 0] - forces[-1, 0])
            force_samples.append(float(tension))

        # ── Euler-Maruyama step for ALL beads ──────────────────────────────
        pos, _ = euler_maruyama_overdamped_step(
            pos, forces, dt, sim.gamma, sim.kBT, rng
        )

        # Re-clamp endpoints AFTER integration
        pos[0,  0] = -z_target / 2.0
        pos[0,  1] = 0.0
        pos[0,  2] = 0.0
        pos[-1, 0] = +z_target / 2.0
        pos[-1, 1] = 0.0
        pos[-1, 2] = 0.0

    fs = np.array(force_samples)
    if len(fs) == 0:
        return float("nan"), float("nan"), 0

    mean_F = float(np.mean(fs))
    sem_F  = float(np.std(fs) / np.sqrt(len(fs))) if len(fs) > 1 else 0.0

    if verbose:
        print(f"  z={z_target:7.3f}  frames={len(fs):4d}  <F>={mean_F:.5f}  +/-{sem_F:.5f}")

    return mean_F, sem_F, len(fs)


def main():
    parser = argparse.ArgumentParser(
        description="Average force vs end-to-end extension for a polymer chain"
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml"
    )
    parser.add_argument(
        "--output", type=str, default="plots/force_extension.png"
    )
    parser.add_argument(
        "--N", type=int, default=30,
        help="Chain length (number of beads)"
    )
    parser.add_argument(
        "--n_ext", type=int, default=20,
        help="Number of extension points to sample"
    )
    parser.add_argument(
        "--z_max_frac", type=float, default=0.95,
        help="Maximum extension as fraction of contour length (0<z_max_frac<1)"
    )
    parser.add_argument(
        "--n_equil", type=int, default=10000,
        help="Equilibration steps at each extension (default 10k for reliable relaxation)"
    )
    parser.add_argument(
        "--n_sample", type=int, default=20000,
        help="Production sampling steps at each extension (default 20k)"
    )
    parser.add_argument(
        "--sample_every", type=int, default=50,
        help="Sample force every N steps during production"
    )
    parser.add_argument(
        "--seed", type=int, default=42
    )
    parser.add_argument(
        "--quiet", action="store_true"
    )
    args = parser.parse_args()

    config_path = os.path.join(project_root, args.config)
    config = load_config(config_path)

    N       = args.N
    b       = config["bond"]["r0"]          # equilibrium bond length (sigma)
    kBT     = config["thermostat"]["kBT"]
    L_c     = (N - 1) * b                  # contour length
    # Persistence length: for harmonic bond, Lp ~ k_bond * b / kBT (stiff rod limit)
    # But our chain is flexible; use Lp ~ b / 2 as an effective worm-like parameter
    Lp_eff  = b / 2.0

    verbose = not args.quiet

    # Extensions: sweep from just above equilibrium Ree to near full stretch.
    # Equilibrium Ree for SAW chain: ~ b * N^nu * sqrt(1/3) ... use Rg-based estimate.
    # Simpler: Ree_eq ~ sqrt(N) * b for ideal, ~ N^0.6 * b for SAW.
    # We start from ~30% of L_c (safely above equilibrium) up to z_max_frac * L_c.
    # This ensures we're always in the tensile (stretched) regime.
    sigma_cfg = config["wca"]["sigma"]
    # Hard minimum: beads must have at least sigma spacing
    z_physical_min = (N - 1) * sigma_cfg * 1.05   # fully extended at WCA contact
    # Start from whichever is larger: 30% of L_c or the N*sigma minimum
    z_min = max(0.30 * L_c, z_physical_min * 0.5)
    z_max = args.z_max_frac * L_c
    if z_min >= z_max:
        z_min = 0.5 * z_max
    z_values   = np.linspace(z_min, z_max, args.n_ext)

    print("=" * 65)
    print("  Force vs Extension -- polymer chain")
    print("=" * 65)
    print(f"  Config     : {args.config}")
    print(f"  N          : {N}")
    print(f"  b (r0)     : {b}")
    print(f"  kBT        : {kBT}")
    print(f"  Contour L  : {L_c:.3f} sigma")
    print(f"  z range    : {z_min:.2f} -- {z_max:.2f} sigma")
    print(f"  Extensions : {args.n_ext}")
    print(f"  Equil steps: {args.n_equil}")
    print(f"  Sample steps:{args.n_sample}")
    print(f"  Output     : {args.output}")
    print("=" * 65)

    mean_F_list = []
    sem_F_list  = []

    for i, z in enumerate(z_values):
        print(f"\n[{i+1}/{args.n_ext}] z = {z:.3f} ({100*z/L_c:.1f}% of L_c)")
        mean_F, sem_F, n_samples = measure_force_at_extension(
            z_target=z,
            N=N,
            config=config,
            n_equil=args.n_equil,
            n_sample=args.n_sample,
            sample_every=args.sample_every,
            seed=args.seed + i,
            verbose=verbose,
        )
        mean_F_list.append(mean_F)
        sem_F_list.append(sem_F)

    z_arr = z_values
    F_arr = np.array(mean_F_list)
    S_arr = np.array(sem_F_list)

    # ── theory curves ────────────────────────────────────────────────────────
    z_theory  = np.linspace(0.01 * L_c, 0.999 * L_c, 500)
    F_fjc     = fjc_force(z_theory / L_c, kBT, b, N - 1)
    F_wlc     = wlc_force(z_theory, L_c, Lp_eff, kBT)
    F_spring  = entropic_spring_force(z_theory, N - 1, b, kBT)

    # ── linear fit to simulation data (small z) ──────────────────────────────
    n_lin = max(3, args.n_ext // 4)
    valid = np.isfinite(F_arr)
    if valid.sum() > 2:
        slope_lin, intercept_lin, r_lin, _, _ = stats.linregress(
            z_arr[valid][:n_lin], F_arr[valid][:n_lin]
        )
        F_lin_fit = slope_lin * z_theory + intercept_lin
        spring_const = slope_lin
    else:
        F_lin_fit    = None
        spring_const = float("nan")

    print("\n" + "=" * 65)
    print("  RESULTS")
    print("=" * 65)
    print(f"  Theoretical entropic spring const: "
          f"{3*kBT/((N-1)*b**2):.5f} kBT/sigma^2")
    print(f"  Measured (linear fit, first {n_lin} pts): {spring_const:.5f}")
    print("=" * 65)

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

    # ── left panel: F vs z (absolute units) ──────────────────────────────────
    ax = axes[0]
    ax.plot(z_theory, F_fjc,    "-",  color="#ffd166", linewidth=1.8,
            label=f"FJC (N-1={N-1} segments)")
    ax.plot(z_theory, F_wlc,    "--", color="#06d6a0", linewidth=1.8,
            label=f"WLC (Lp={Lp_eff:.2f} sigma)")
    ax.plot(z_theory, F_spring, "-.", color="#ef9d3a", linewidth=1.8,
            label="Entropic spring (linear)")
    if F_lin_fit is not None:
        ax.plot(z_theory, F_lin_fit, ":", color="#c77dff", linewidth=1.6,
                label=f"Sim linear fit (k={spring_const:.4f})")
    ax.errorbar(
        z_arr[valid], F_arr[valid], yerr=2 * S_arr[valid],
        fmt="o", color="#7c83fd", markersize=7, linewidth=1.5,
        ecolor="#a0a4ff", capsize=4, capthick=1.5,
        zorder=5, label="Simulation (+/-2 SEM)"
    )
    ax.set_xlabel("Extension z (sigma)", fontsize=13)
    ax.set_ylabel("Force F (epsilon/sigma)", fontsize=13)
    ax.set_title(f"Force vs Extension  (N={N})", fontsize=14)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9, framealpha=0.2, labelcolor="#e8eaf6",
              facecolor="#1a1d27", edgecolor="#3a3d52")

    # inset annotation
    ax.text(
        0.05, 0.95,
        f"N={N}, b={b}, kBT={kBT}\n"
        f"L_c = {L_c:.2f} sigma\n"
        f"k_spring (sim) = {spring_const:.4f}",
        transform=ax.transAxes, fontsize=9, verticalalignment="top",
        color="#e8eaf6",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#2a2d3e",
                  alpha=0.8, edgecolor="#3a3d52")
    )

    # ── right panel: normalized (F * L_c/kBT vs z/L_c) ───────────────────────
    ax = axes[1]
    x_th  = z_theory / L_c
    ax.plot(x_th, F_fjc   * L_c / kBT, "-",  color="#ffd166", linewidth=1.8,
            label="FJC")
    ax.plot(x_th, F_wlc   * L_c / kBT, "--", color="#06d6a0", linewidth=1.8,
            label="WLC")
    ax.plot(x_th, F_spring * L_c / kBT, "-.", color="#ef9d3a", linewidth=1.8,
            label="Entropic spring")
    ax.errorbar(
        z_arr[valid] / L_c,
        F_arr[valid] * L_c / kBT,
        yerr=2 * S_arr[valid] * L_c / kBT,
        fmt="o", color="#7c83fd", markersize=7, linewidth=1.5,
        ecolor="#a0a4ff", capsize=4, capthick=1.5,
        zorder=5, label="Simulation (+/-2 SEM)"
    )
    ax.set_xlabel("Reduced extension z / L_c", fontsize=13)
    ax.set_ylabel("Reduced force F*L_c / kBT", fontsize=13)
    ax.set_title("Normalized Force-Extension", fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9.5, framealpha=0.2, labelcolor="#e8eaf6",
              facecolor="#1a1d27", edgecolor="#3a3d52")

    fig.suptitle(
        f"Average Force vs Extension  |  N={N}, bond={config['bond']['type']}, "
        f"kBT={kBT}",
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
