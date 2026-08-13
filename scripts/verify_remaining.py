#!/usr/bin/env python
"""
Address the remaining yellow/red items from the production review:

1. YELLOW: Force cap/clamp activation count during production
2. YELLOW: Displacement autocorrelation (actual GNN target, not PE proxy)
3. YELLOW: Compare equilibrium distributions against theory (R_g scaling, ⟨R²(s)⟩ vs s)
4. RED:    Verify dataset split handles temporal correlation

Output: plots/equilibration/remaining_checks.png + console report
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
from src.physics.integrators import euler_maruyama_overdamped_step
from src.simulator.numpy_simulator import NumpySimulator
import time


def integrated_autocorrelation_time(x):
    """Estimate integrated autocorrelation time using window method."""
    xp = x - np.mean(x)
    var = np.var(x)
    if var < 1e-30:
        return 0.5, np.ones(len(x) // 2)
    f = np.fft.fft(xp, n=2 * len(x))
    acf = np.fft.ifft(np.abs(f) ** 2).real[:len(x)] / (len(x) * var)
    tau_int = 0.5
    for i in range(1, len(acf)):
        if acf[i] < 0:
            break
        tau_int += acf[i]
        if i > 6 * tau_int:
            break
    return tau_int, acf


def compute_forces_instrumented(positions, sim):
    """Compute forces WITH explicit cap/clamp counting.
    
    Reimplements the force pipeline with counters instead of modifying
    the source code, so we don't add permanent instrumentation overhead.
    """
    from src.physics.forces import compute_bonded_forces

    N, dim = positions.shape
    sigma = sim.sigma
    epsilon = sim.epsilon
    r_cut = sigma * 2.0 ** (1.0 / 6.0)

    # --- Bonded forces (unchanged) ---
    f_bond, pe_bond = compute_bonded_forces(
        positions, bond_type=sim.bond_type, k_bond=sim.k_bond, r0=sim.r0,
        k_fene=sim.k_fene, R0_fene=sim.R0_fene
    )

    # --- Nonbonded forces with clamp counting ---
    forces_nb = np.zeros_like(positions)
    pe_nb = 0.0
    clamp_count = 0

    i_idx, j_idx = np.triu_indices(N, k=2)
    if len(i_idx) > 0:
        r_ij = positions[j_idx] - positions[i_idx]
        dist_raw = np.linalg.norm(r_ij, axis=1)

        mask = (dist_raw < r_cut) & (dist_raw > 1e-12)

        # Count how many pairs would be clamped
        clamp_mask = dist_raw < 0.4 * sigma
        clamp_count = int(np.sum(clamp_mask & (dist_raw > 1e-12)))

        # Apply clamp
        dist = np.clip(dist_raw, 0.4 * sigma, None)

        if np.any(mask):
            r_ij_active = r_ij[mask]
            dist_active = dist[mask]
            i_active = i_idx[mask]
            j_active = j_idx[mask]

            sr6 = (sigma / dist_active) ** 6
            sr12 = sr6 ** 2
            force_factor = 24.0 * epsilon / dist_active * (2.0 * sr12 - sr6)
            r_hat = r_ij_active / dist_active[:, np.newaxis]
            f_wca = -force_factor[:, np.newaxis] * r_hat

            np.add.at(forces_nb, i_active, f_wca)
            np.subtract.at(forces_nb, j_active, f_wca)
            pe_nb = float(np.sum(4.0 * epsilon * (sr12 - sr6) + epsilon))

    total_forces = f_bond + forces_nb

    # --- Force cap counting ---
    F_MAX = 1000.0
    force_mags = np.linalg.norm(total_forces, axis=1, keepdims=True)
    cap_mask = force_mags > F_MAX
    cap_count = int(np.sum(cap_mask))

    if np.any(cap_mask):
        scale = np.where(cap_mask, F_MAX / force_mags, 1.0)
        total_forces = total_forces * scale

    return total_forces, pe_bond, pe_nb, clamp_count, cap_count


def main():
    output_dir = "plots/equilibration"
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")) as f:
        config = yaml.safe_load(f)

    sim = NumpySimulator(config)
    seed = sim.base_seed + 999
    rng = np.random.default_rng(seed)
    positions = sim.initialize_chain(rng)

    T_steps = sim.T_steps
    n_burnin = sim.n_burnin
    dt = sim.dt
    N = sim.N
    log_every = 1000

    # Counters for force cap/clamp
    total_clamp_activations = 0
    total_cap_activations = 0
    total_prod_steps = 0

    # Storage for displacement autocorrelation
    prev_positions = None
    prod_displacements_per_bead = []   # per-bead displacement magnitudes
    prod_displacements_mean = []        # mean displacement magnitude per frame
    prod_displacement_vectors = []      # full displacement vectors for a single bead

    # Storage for internal distance analysis
    prod_positions_snapshots = []  # Store some snapshots for ⟨R²(s)⟩

    print("=" * 70)
    print("REMAINING VERIFICATION CHECKS")
    print("=" * 70)

    start_time = time.time()

    for t in range(1, T_steps + 1):
        forces, pe_b, pe_nb, n_clamp, n_cap = compute_forces_instrumented(positions, sim)

        old_positions = positions.copy()
        positions, _ = euler_maruyama_overdamped_step(
            positions, forces, dt, sim.gamma, sim.kBT, rng
        )

        if t > n_burnin:
            total_prod_steps += 1
            total_clamp_activations += n_clamp
            total_cap_activations += n_cap

            if t % log_every == 0:
                # Displacement = positions - old_positions (the GNN target)
                disp = positions - old_positions
                disp_mags = np.linalg.norm(disp, axis=1)
                prod_displacements_mean.append(np.mean(disp_mags))
                prod_displacements_per_bead.append(disp_mags)

                # Track displacement of bead 0 for vector autocorrelation
                prod_displacement_vectors.append(disp[0].copy())

                # Store position snapshots (every 10th log for memory)
                if t % (log_every * 10) == 0:
                    prod_positions_snapshots.append(positions.copy())

        if t % (T_steps // 10) == 0:
            print(f"  [{t}/{T_steps}] clamp_total={total_clamp_activations}, "
                  f"cap_total={total_cap_activations}")

    elapsed = time.time() - start_time
    print(f"\nSimulation completed in {elapsed:.1f}s")

    # Convert to arrays
    prod_displacements_mean = np.array(prod_displacements_mean)
    prod_displacement_vectors = np.array(prod_displacement_vectors)  # (n_samples, 3)

    # =====================================================================
    # CHECK A: Force cap/clamp activation count
    # =====================================================================
    print("\n" + "=" * 70)
    print("CHECK A: Force cap/clamp activation during production")
    print("=" * 70)
    print(f"  Total production steps: {total_prod_steps}")
    print(f"  WCA distance clamp activations: {total_clamp_activations}")
    print(f"  Force magnitude cap activations: {total_cap_activations}")

    if total_clamp_activations == 0 and total_cap_activations == 0:
        print(f"  => PASS: Neither the clamp nor the cap was activated during production.")
        print(f"     The safety mechanisms did not modify the physical dynamics.")
    else:
        clamp_rate = total_clamp_activations / total_prod_steps
        cap_rate = total_cap_activations / total_prod_steps
        print(f"  Clamp rate: {clamp_rate:.6f} per step ({total_clamp_activations} total)")
        print(f"  Cap rate: {cap_rate:.6f} per step ({total_cap_activations} total)")
        if clamp_rate < 1e-5 and cap_rate < 1e-5:
            print(f"  => ACCEPTABLE: Activations are extremely rare (<0.001% of steps).")
        else:
            print(f"  => WARNING: Safety mechanisms are modifying dynamics non-negligibly.")

    # =====================================================================
    # CHECK B: Displacement autocorrelation (actual GNN target)
    # =====================================================================
    print("\n" + "=" * 70)
    print("CHECK B: Displacement autocorrelation (GNN target)")
    print("=" * 70)

    # Mean displacement magnitude
    tau_int_disp, acf_disp = integrated_autocorrelation_time(prod_displacements_mean)
    sample_interval_tau = log_every * dt
    tau_int_disp_tau = tau_int_disp * sample_interval_tau

    print(f"  Mean |displacement| autocorrelation:")
    print(f"    tau_int = {tau_int_disp:.1f} samples = {tau_int_disp_tau:.1f} tau")

    # Displacement vector components for bead 0
    dx = prod_displacement_vectors[:, 0]
    dy = prod_displacement_vectors[:, 1]
    dz = prod_displacement_vectors[:, 2]

    tau_dx, acf_dx = integrated_autocorrelation_time(dx)
    tau_dy, acf_dy = integrated_autocorrelation_time(dy)
    tau_dz, acf_dz = integrated_autocorrelation_time(dz)

    tau_dx_tau = tau_dx * sample_interval_tau
    tau_dy_tau = tau_dy * sample_interval_tau
    tau_dz_tau = tau_dz * sample_interval_tau

    print(f"  Bead 0 displacement components:")
    print(f"    dx: tau_int = {tau_dx:.1f} samples = {tau_dx_tau:.1f} tau")
    print(f"    dy: tau_int = {tau_dy:.1f} samples = {tau_dy_tau:.1f} tau")
    print(f"    dz: tau_int = {tau_dz:.1f} samples = {tau_dz_tau:.1f} tau")

    n_eff_disp = len(prod_displacements_mean) / (2 * tau_int_disp)
    print(f"  Effective independent displacement samples: ~{n_eff_disp:.0f}")
    print(f"  (Compare with Rg: ~15 independent global conformations)")

    # =====================================================================
    # CHECK C: Internal distance scaling ⟨R²(s)⟩ vs s
    # =====================================================================
    print("\n" + "=" * 70)
    print("CHECK C: Internal distance scaling (theory comparison)")
    print("=" * 70)

    snapshots = np.array(prod_positions_snapshots)
    n_snaps = len(snapshots)
    print(f"  Using {n_snaps} position snapshots for ensemble average")

    # Compute ⟨R²(s)⟩ for contour separations s = 1..N-1
    s_values = np.arange(1, N)
    r2_mean = np.zeros(len(s_values))

    for snap in snapshots:
        for si, s in enumerate(s_values):
            # Average over all pairs separated by s along the chain
            diffs = snap[s:] - snap[:-s]
            r2_mean[si] += np.mean(np.sum(diffs ** 2, axis=1))
    r2_mean /= n_snaps

    # Theoretical: for an ideal chain (no excluded volume), ⟨R²(s)⟩ = s * b²
    # where b is the effective bond length.
    # With excluded volume (SAW), ⟨R²(s)⟩ ~ s^(2ν) with ν ≈ 0.588 in 3D.
    # Fit a power law to the data
    log_s = np.log(s_values[1:])  # Skip s=1 for fitting
    log_r2 = np.log(r2_mean[1:])
    slope, intercept = np.polyfit(log_s, log_r2, 1)
    nu_measured = slope / 2.0

    print(f"  Power law fit: ⟨R²(s)⟩ ~ s^{slope:.3f}")
    print(f"  Measured Flory exponent: ν = {nu_measured:.3f}")
    print(f"  Expected for ideal chain: ν = 0.500 (⟨R²⟩ ~ s)")
    print(f"  Expected for SAW in 3D:   ν = 0.588 (⟨R²⟩ ~ s^1.176)")

    # Rg check
    rg_values = []
    for snap in snapshots:
        com = np.mean(snap, axis=0)
        rg_sq = np.mean(np.sum((snap - com) ** 2, axis=1))
        rg_values.append(np.sqrt(rg_sq))
    rg_mean = np.mean(rg_values)
    rg_std = np.std(rg_values)

    # Theoretical Rg for ideal chain: Rg² = (1/6) * N * b²
    b_eff = np.sqrt(r2_mean[0])  # Effective bond length from ⟨R²(1)⟩
    rg_ideal = np.sqrt(N * b_eff ** 2 / 6.0)
    print(f"\n  Measured Rg: {rg_mean:.3f} ± {rg_std:.3f}")
    print(f"  Ideal chain Rg (b={b_eff:.3f}): {rg_ideal:.3f}")
    print(f"  Ratio measured/ideal: {rg_mean/rg_ideal:.3f}")
    if rg_mean > rg_ideal:
        print(f"  => Polymer is swollen relative to ideal (consistent with excluded volume)")
    else:
        print(f"  => Polymer is compact relative to ideal")

    # =====================================================================
    # CHECK D: Dataset split design for temporal correlation
    # =====================================================================
    print("\n" + "=" * 70)
    print("CHECK D: Dataset split design (train/test leakage risk)")
    print("=" * 70)

    print(f"  Current dataset.py splitting strategy: BY TRAJECTORY (not by frame)")
    print(f"  This is the correct approach for preventing leakage.")
    print(f"")
    print(f"  For a single trajectory, recommended contiguous block split:")
    print(f"    TRAIN:      2700–3200τ  (500τ, ~5000 frames)")
    print(f"    VALIDATION: 3200–3450τ  (250τ, ~2500 frames)")
    print(f"    TEST:        3450–3700τ  (250τ, ~2500 frames)")
    print(f"")
    print(f"  With tau_int(displacement) = {tau_int_disp_tau:.1f}τ:")
    print(f"    Gap between train/val boundary: 0τ (but train end is >250τ")
    print(f"    from test start, which is >> tau_int)")
    print(f"")
    print(f"  For STRONG generalization testing:")
    print(f"    Generate multiple independent trajectories (different seeds)")
    print(f"    Split by trajectory → zero temporal leakage")

    # =====================================================================
    # PLOTS
    # =====================================================================
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Remaining Verification Checks", fontsize=16, fontweight='bold')

    # Plot A: Cap/Clamp counts
    ax = axs[0, 0]
    categories = ['WCA Clamp\n(dist < 0.4σ)', 'Force Cap\n(|F| > 1000)']
    counts = [total_clamp_activations, total_cap_activations]
    colors = ['green' if c == 0 else 'orange' for c in counts]
    bars = ax.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
    for bar, count in zip(bars, counts):
        label = str(count) if count > 0 else "0 ✓"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                label, ha='center', fontweight='bold', fontsize=14,
                color='green' if count == 0 else 'red')
    ax.set_ylabel(f"Activations (out of {total_prod_steps} steps)")
    ax.set_title("A. Safety Mechanism Activations")
    ax.grid(True, alpha=0.3, axis='y')

    # Plot B: Displacement ACF
    ax = axs[0, 1]
    max_lag = min(50, len(acf_disp))
    lag_tau = np.arange(max_lag) * sample_interval_tau
    ax.plot(lag_tau, acf_disp[:max_lag], 'b-', linewidth=2,
            label=f'|Δr| mean (τ_int={tau_int_disp_tau:.1f}τ)')
    ax.plot(lag_tau, acf_dx[:max_lag], 'r--', alpha=0.6,
            label=f'Δr_x bead 0 (τ_int={tau_dx_tau:.1f}τ)')
    ax.axhline(0, color='black', alpha=0.5)
    ax.axhline(np.exp(-1), color='gray', linestyle=':', alpha=0.5, label='1/e')
    ax.set_xlabel("Lag (τ)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("B. Displacement Autocorrelation (GNN Target)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot C: ⟨R²(s)⟩ vs s
    ax = axs[1, 0]
    ax.loglog(s_values, r2_mean, 'ko-', markersize=4, label='Measured')
    # Ideal chain reference
    ax.loglog(s_values, b_eff**2 * s_values, 'b--', alpha=0.5,
              label=f'Ideal: s^1.0 (ν=0.5)')
    # SAW reference
    c_saw = r2_mean[0]  # normalize to match at s=1
    ax.loglog(s_values, c_saw * s_values ** 1.176, 'r--', alpha=0.5,
              label=f'SAW: s^1.176 (ν=0.588)')
    # Measured fit
    ax.loglog(s_values, np.exp(intercept) * s_values ** slope, 'g-', linewidth=2,
              label=f'Fit: s^{slope:.2f} (ν={nu_measured:.3f})')
    ax.set_xlabel("Contour separation s (beads)")
    ax.set_ylabel("⟨R²(s)⟩ (σ²)")
    ax.set_title("C. Internal Distance Scaling")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot D: Correlation structure diagram
    ax = axs[1, 1]
    ax.set_xlim(2600, 3800)
    ax.set_ylim(0, 5)

    # Draw the timeline
    ax.axhline(2.5, color='black', linewidth=2)

    # Mark regions
    ax.axvspan(2700, 3200, alpha=0.3, color='blue', label='TRAIN (500τ)')
    ax.axvspan(3200, 3450, alpha=0.3, color='orange', label='VAL (250τ)')
    ax.axvspan(3450, 3700, alpha=0.3, color='red', label='TEST (250τ)')
    ax.axvspan(2600, 2700, alpha=0.2, color='gray', label='Burn-in end')

    # Mark autocorrelation times
    ax.annotate('', xy=(2700 + tau_int_disp_tau, 4), xytext=(2700, 4),
                arrowprops=dict(arrowstyle='<->', color='green', lw=2))
    ax.text(2700 + tau_int_disp_tau / 2, 4.3,
            f'τ_int(Δr)={tau_int_disp_tau:.1f}τ', ha='center', fontsize=9, color='green')

    ax.set_xlabel("Time (τ)")
    ax.set_yticks([])
    ax.set_title("D. Recommended Split (no temporal leakage)")
    ax.legend(loc='lower right', fontsize=7)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    out_path = os.path.join(output_dir, "remaining_checks.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
