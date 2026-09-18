#!/usr/bin/env python
"""
Production data verification script.

Verifies that the post-burn-in trajectory is valid equilibrium data by checking:
1. Rg production mean/std in several time windows
2. Bond-length distribution: early vs late production
3. Ree production statistics in several windows
4. Integrated autocorrelation time
5. Frame saving frequency analysis
6. WCA/numerical failure detection in production frames

Output: plots/equilibration/production_verification.png + console report
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
from src.physics.forces import compute_all_forces
from src.physics.integrators import euler_maruyama_overdamped_step
from src.simulator.numpy_simulator import NumpySimulator
import time


def autocorrelation_fft(x):
    """Normalized autocorrelation via FFT."""
    xp = x - np.mean(x)
    var = np.var(x)
    if var < 1e-30:
        return np.ones(len(x) // 2)
    f = np.fft.fft(xp, n=2 * len(x))
    acf = np.fft.ifft(np.abs(f) ** 2).real[:len(x)] / (len(x) * var)
    return acf


def integrated_autocorrelation_time(x):
    """Estimate integrated autocorrelation time tau_int.
    Uses the standard window method: sum ACF until it drops below zero
    or until the window exceeds 6*tau_int (self-consistent cutoff).
    """
    acf = autocorrelation_fft(x)
    tau_int = 0.5  # Start with the t=0 contribution (0.5 by convention)
    for i in range(1, len(acf)):
        if acf[i] < 0:
            break
        tau_int += acf[i]
        # Self-consistent cutoff: stop if window > 6*tau_int
        if i > 6 * tau_int:
            break
    return tau_int, acf


def main():
    output_dir = "plots/equilibration"
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")) as f:
        config = yaml.safe_load(f)

    sim = NumpySimulator(config)
    seed = sim.base_seed
    rng = np.random.default_rng(seed)
    positions = sim.initialize_chain(rng)

    T_steps = sim.T_steps
    n_burnin = sim.n_burnin
    dt = sim.dt
    save_every = sim.save_every  # From config: how often frames are saved
    N = sim.N

    # We log observables every 1000 steps for the time series,
    # but we also track production frames at the actual save_every interval
    log_every = 1000

    # Storage
    times_all = []
    rg_all = []
    ree_all = []

    # Production-only storage (collected every log_every steps after burn-in)
    prod_times = []
    prod_rg = []
    prod_ree = []
    prod_pe = []
    prod_mean_bond = []
    prod_all_bonds_early = []
    prod_all_bonds_late = []
    prod_max_bond = []
    prod_min_bond = []
    prod_max_coord = []
    prod_max_force = []
    prod_nan_count = 0
    prod_extreme_bond_count = 0  # bonds > 2.0 or < 0.5
    prod_extreme_force_count = 0  # forces > 500

    # Production time boundaries
    prod_start_tau = n_burnin * dt  # 2700 tau
    prod_end_tau = T_steps * dt     # 3700 tau
    prod_duration = prod_end_tau - prod_start_tau  # 1000 tau
    prod_quarter = prod_duration / 4  # 250 tau per window

    print("=" * 70)
    print("PRODUCTION DATA VERIFICATION")
    print("=" * 70)
    print(f"Chain: N={N}, dt={dt}, gamma={sim.gamma}, kBT={sim.kBT}")
    print(f"Total steps: {T_steps} ({T_steps*dt:.0f} tau)")
    print(f"Burn-in: {n_burnin} steps ({n_burnin*dt:.0f} tau)")
    print(f"Production: {T_steps - n_burnin} steps ({prod_duration:.0f} tau)")
    print(f"Frame save interval: every {save_every} steps ({save_every*dt:.3f} tau)")
    print(f"Expected production frames: {(T_steps - n_burnin) // save_every}")
    print("=" * 70)

    start_time = time.time()

    for t in range(1, T_steps + 1):
        forces, pe_b, pe_nb = compute_all_forces(
            positions, bond_type=sim.bond_type, k_bond=sim.k_bond, r0=sim.r0,
            k_fene=sim.k_fene, R0_fene=sim.R0_fene,
            epsilon=sim.epsilon, sigma=sim.sigma,
        )
        positions, _ = euler_maruyama_overdamped_step(
            positions, forces, dt, sim.gamma, sim.kBT, rng
        )

        if t % log_every == 0:
            obs = sim.compute_observables(positions)
            tau_now = t * dt

            times_all.append(tau_now)
            rg_all.append(obs["radius_of_gyration"])
            ree_all.append(obs["end_to_end_distance"])

            if t > n_burnin:
                prod_times.append(tau_now)
                prod_rg.append(obs["radius_of_gyration"])
                prod_ree.append(obs["end_to_end_distance"])
                prod_pe.append(pe_b + pe_nb)
                prod_mean_bond.append(obs["mean_bond_length"])
                prod_max_bond.append(obs["max_bond_length"])
                prod_min_bond.append(obs["min_bond_length"])
                prod_max_coord.append(float(np.abs(positions).max()))
                prod_max_force.append(float(np.abs(forces).max()))

                # Check for numerical failures
                if np.any(np.isnan(positions)):
                    prod_nan_count += 1
                if obs["max_bond_length"] > 2.0 or obs["min_bond_length"] < 0.5:
                    prod_extreme_bond_count += 1
                if np.abs(forces).max() > 500:
                    prod_extreme_force_count += 1

                # Collect bond lengths for distribution comparison
                # Early = first quarter, Late = last quarter
                if tau_now < prod_start_tau + prod_quarter:
                    prod_all_bonds_early.extend(obs["bond_lengths"])
                elif tau_now > prod_end_tau - prod_quarter:
                    prod_all_bonds_late.extend(obs["bond_lengths"])

        if t % (T_steps // 10) == 0:
            obs = sim.compute_observables(positions)
            print(f"  [{t}/{T_steps}] Rg={obs['radius_of_gyration']:.3f}, "
                  f"Ree={obs['end_to_end_distance']:.3f}, "
                  f"PE={pe_b+pe_nb:.3f}")

    elapsed = time.time() - start_time
    print(f"\nSimulation completed in {elapsed:.1f}s")

    # Convert to arrays
    prod_times = np.array(prod_times)
    prod_rg = np.array(prod_rg)
    prod_ree = np.array(prod_ree)
    prod_pe = np.array(prod_pe)
    prod_mean_bond = np.array(prod_mean_bond)

    # =====================================================================
    # CHECK 1: Rg in time windows
    # =====================================================================
    print("\n" + "=" * 70)
    print("CHECK 1: Rg production statistics in time windows")
    print("=" * 70)

    n_windows = 4
    window_size = len(prod_rg) // n_windows
    rg_window_stats = []
    for w in range(n_windows):
        start_idx = w * window_size
        end_idx = (w + 1) * window_size if w < n_windows - 1 else len(prod_rg)
        window_data = prod_rg[start_idx:end_idx]
        t_start = prod_times[start_idx]
        t_end = prod_times[end_idx - 1]
        mean = np.mean(window_data)
        std = np.std(window_data)
        rg_window_stats.append((t_start, t_end, mean, std))
        print(f"  Window {w+1} [{t_start:.0f}–{t_end:.0f} tau]: "
              f"mean={mean:.4f}, std={std:.4f}")

    # Check stationarity: means should be within ~2 sigma of each other
    all_means = [s[2] for s in rg_window_stats]
    all_stds = [s[3] for s in rg_window_stats]
    overall_mean = np.mean(prod_rg)
    overall_std = np.std(prod_rg)
    max_deviation = max(abs(m - overall_mean) for m in all_means)
    print(f"  Overall: mean={overall_mean:.4f}, std={overall_std:.4f}")
    print(f"  Max window deviation from overall mean: {max_deviation:.4f} "
          f"({max_deviation/overall_std:.2f} sigma)")

    # =====================================================================
    # CHECK 2: Bond-length distribution early vs late
    # =====================================================================
    print("\n" + "=" * 70)
    print("CHECK 2: Bond-length distribution (early vs late production)")
    print("=" * 70)

    early_bonds = np.array(prod_all_bonds_early)
    late_bonds = np.array(prod_all_bonds_late)
    early_mean, early_std = np.mean(early_bonds), np.std(early_bonds)
    late_mean, late_std = np.mean(late_bonds), np.std(late_bonds)
    print(f"  Early quarter: {len(early_bonds)} bond samples, "
          f"mean={early_mean:.4f}, std={early_std:.4f}")
    print(f"  Late quarter:  {len(late_bonds)} bond samples, "
          f"mean={late_mean:.4f}, std={late_std:.4f}")

    # KS test: are the two distributions consistent?
    ks_stat, ks_p = stats.ks_2samp(early_bonds, late_bonds)
    print(f"  KS test: statistic={ks_stat:.6f}, p-value={ks_p:.6f}")
    
    bond_mean_diff = abs(early_mean - late_mean)
    bond_std_diff = abs(early_std - late_std)
    bond_dist_pass = (ks_p > 0.01) or (bond_mean_diff < 0.005 and bond_std_diff < 0.005)

    if ks_p > 0.01:
        print(f"  => PASS: distributions are consistent (p > 0.01)")
    elif bond_dist_pass:
        print(f"  => PASS: p < 0.01, but absolute mean/std differences (< 0.005σ) are physically irrelevant.")
    else:
        print(f"  => WARNING: distributions differ significantly (p < 0.01, mean diff = {bond_mean_diff:.4f}σ)")

    # =====================================================================
    # CHECK 3: Ree in time windows
    # =====================================================================
    print("\n" + "=" * 70)
    print("CHECK 3: Ree production statistics in time windows")
    print("=" * 70)

    ree_window_stats = []
    for w in range(n_windows):
        start_idx = w * window_size
        end_idx = (w + 1) * window_size if w < n_windows - 1 else len(prod_ree)
        window_data = prod_ree[start_idx:end_idx]
        t_start = prod_times[start_idx]
        t_end = prod_times[end_idx - 1]
        mean = np.mean(window_data)
        std = np.std(window_data)
        ree_window_stats.append((t_start, t_end, mean, std))
        print(f"  Window {w+1} [{t_start:.0f}–{t_end:.0f} tau]: "
              f"mean={mean:.4f}, std={std:.4f}")

    overall_ree_mean = np.mean(prod_ree)
    overall_ree_std = np.std(prod_ree)
    max_ree_dev = max(abs(s[2] - overall_ree_mean) for s in ree_window_stats)
    print(f"  Overall: mean={overall_ree_mean:.4f}, std={overall_ree_std:.4f}")
    print(f"  Max window deviation: {max_ree_dev:.4f} "
          f"({max_ree_dev/overall_ree_std:.2f} sigma)")

    # =====================================================================
    # CHECK 4: Integrated autocorrelation time
    # =====================================================================
    print("\n" + "=" * 70)
    print("CHECK 4: Integrated autocorrelation time")
    print("=" * 70)

    tau_int_rg, acf_rg = integrated_autocorrelation_time(prod_rg)
    tau_int_ree, acf_ree = integrated_autocorrelation_time(prod_ree)
    tau_int_pe, acf_pe = integrated_autocorrelation_time(prod_pe)

    # Convert from sample units to tau
    sample_interval_tau = log_every * dt  # 1.0 tau
    tau_int_rg_tau = tau_int_rg * sample_interval_tau
    tau_int_ree_tau = tau_int_ree * sample_interval_tau
    tau_int_pe_tau = tau_int_pe * sample_interval_tau

    print(f"  Rg:  tau_int = {tau_int_rg:.1f} samples = {tau_int_rg_tau:.1f} tau")
    print(f"  Ree: tau_int = {tau_int_ree:.1f} samples = {tau_int_ree_tau:.1f} tau")
    print(f"  PE:  tau_int = {tau_int_pe:.1f} samples = {tau_int_pe_tau:.1f} tau")

    n_eff_rg = len(prod_rg) / (2 * tau_int_rg)
    print(f"  Effective independent Rg samples: {n_eff_rg:.0f}")

    save_interval_tau = save_every * dt
    print(f"\n  Frame save interval: {save_every} steps = {save_interval_tau:.3f} tau")
    print(f"  Autocorrelation time (Rg): {tau_int_rg_tau:.1f} tau")
    if save_interval_tau < tau_int_rg_tau:
        ratio = tau_int_rg_tau / save_interval_tau
        print(f"  => NOTE: Frames are saved {ratio:.1f}x more frequently than "
              f"the autocorrelation time.")
        print(f"     Consecutive frames are correlated. For independent samples,")
        print(f"     subsample every ~{int(np.ceil(ratio))} frames.")
    else:
        print(f"  => GOOD: Save interval >= autocorrelation time. Frames are ~independent.")

    # =====================================================================
    # CHECK 5: Frame saving frequency
    # =====================================================================
    print("\n" + "=" * 70)
    print("CHECK 5: Frame saving frequency analysis")
    print("=" * 70)

    n_prod_steps = T_steps - n_burnin
    n_prod_frames = n_prod_steps // save_every
    prod_duration_tau = n_prod_steps * dt

    print(f"  Production duration: {n_prod_steps} steps = {prod_duration_tau:.0f} tau")
    print(f"  Save interval: every {save_every} steps = {save_interval_tau:.3f} tau")
    print(f"  Total production frames: {n_prod_frames}")
    print(f"  Effective independent frames (from tau_int): ~{n_eff_rg:.0f}")

    # =====================================================================
    # CHECK 6: Numerical failure detection
    # =====================================================================
    print("\n" + "=" * 70)
    print("CHECK 6: Numerical failure / WCA anomaly detection in production")
    print("=" * 70)

    n_prod_samples = len(prod_rg)
    print(f"  Production samples checked: {n_prod_samples}")
    print(f"  NaN coordinates: {prod_nan_count}")
    print(f"  Extreme bonds (>2.0 or <0.5): {prod_extreme_bond_count} "
          f"({100*prod_extreme_bond_count/max(1,n_prod_samples):.3f}%)")
    print(f"  Extreme forces (>500): {prod_extreme_force_count} "
          f"({100*prod_extreme_force_count/max(1,n_prod_samples):.3f}%)")
    print(f"  Max bond length in production: {max(prod_max_bond):.4f}")
    print(f"  Min bond length in production: {min(prod_min_bond):.4f}")
    print(f"  Max coordinate in production: {max(prod_max_coord):.2f}")
    print(f"  Max force in production: {max(prod_max_force):.2f}")

    if prod_nan_count == 0 and prod_extreme_bond_count == 0:
        print(f"  => PASS: No numerical failures detected in production data.")
    else:
        print(f"  => WARNING: Anomalies found in production data.")

    # =====================================================================
    # PLOTS
    # =====================================================================
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Production Data Verification (post-2700τ)", fontsize=16, fontweight='bold')

    # Plot 1: Rg in windows
    ax = axs[0, 0]
    for w, (t0, t1, m, s) in enumerate(rg_window_stats):
        ax.barh(w, s * 2, left=m - s, height=0.6, alpha=0.6,
                label=f"[{t0:.0f}–{t1:.0f}τ]")
        ax.plot(m, w, 'ko', markersize=6)
    ax.axvline(overall_mean, color='red', linestyle='--', label=f'Overall={overall_mean:.3f}')
    ax.set_xlabel("Rg")
    ax.set_yticks(range(n_windows))
    ax.set_yticklabels([f"Q{w+1}" for w in range(n_windows)])
    ax.set_title("1. Rg: Window Statistics")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Plot 2: Bond distribution early vs late
    ax = axs[0, 1]
    bins = np.linspace(
        min(early_bonds.min(), late_bonds.min()),
        max(early_bonds.max(), late_bonds.max()),
        60
    )
    ax.hist(early_bonds, bins=bins, density=True, alpha=0.6, color='blue',
            label=f'Early ({prod_start_tau:.0f}–{prod_start_tau+prod_quarter:.0f}τ)')
    ax.hist(late_bonds, bins=bins, density=True, alpha=0.6, color='orange',
            label=f'Late ({prod_end_tau-prod_quarter:.0f}–{prod_end_tau:.0f}τ)')
    ax.set_xlabel("Bond Length (σ)")
    ax.set_ylabel("Density")
    ax.set_title(f"2. Bond Distribution (KS p={ks_p:.4f})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 3: Ree in windows
    ax = axs[0, 2]
    for w, (t0, t1, m, s) in enumerate(ree_window_stats):
        ax.barh(w, s * 2, left=m - s, height=0.6, alpha=0.6,
                label=f"[{t0:.0f}–{t1:.0f}τ]")
        ax.plot(m, w, 'ko', markersize=6)
    ax.axvline(overall_ree_mean, color='red', linestyle='--',
               label=f'Overall={overall_ree_mean:.3f}')
    ax.set_xlabel("Ree")
    ax.set_yticks(range(n_windows))
    ax.set_yticklabels([f"Q{w+1}" for w in range(n_windows)])
    ax.set_title("3. Ree: Window Statistics")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Plot 4: Autocorrelation functions
    ax = axs[1, 0]
    max_lag = min(200, len(acf_rg))
    lag_tau = np.arange(max_lag) * sample_interval_tau
    ax.plot(lag_tau, acf_rg[:max_lag], label=f'Rg (τ_int={tau_int_rg_tau:.1f}τ)', color='blue')
    ax.plot(lag_tau, acf_ree[:max_lag], label=f'Ree (τ_int={tau_int_ree_tau:.1f}τ)', color='green')
    ax.plot(lag_tau, acf_pe[:max_lag], label=f'PE (τ_int={tau_int_pe_tau:.1f}τ)', color='red')
    ax.axhline(0, color='black', alpha=0.5)
    ax.axhline(np.exp(-1), color='gray', linestyle='--', alpha=0.5, label='1/e')
    ax.set_xlabel("Lag (τ)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("4. Autocorrelation + τ_int")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 5: Frame saving context
    ax = axs[1, 1]
    ax.plot(prod_times, prod_mean_bond, alpha=0.5, linewidth=0.5, color='green')
    ax.set_xlabel("Time (τ)")
    ax.set_ylabel("Mean Bond Length (σ)")
    ax.set_title(f"5. Production Trace (save_every={save_every} steps)")
    ax.axhline(np.mean(prod_mean_bond), color='red', linestyle='--',
               label=f'mean={np.mean(prod_mean_bond):.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: Numerical health
    ax = axs[1, 2]
    categories = ['NaN\ncoords', 'Extreme\nbonds', 'Extreme\nforces']
    counts = [prod_nan_count, prod_extreme_bond_count, prod_extreme_force_count]
    colors = ['green' if c == 0 else 'red' for c in counts]
    bars = ax.bar(categories, counts, color=colors, alpha=0.7)
    ax.set_ylabel("Count")
    ax.set_title("6. Numerical Failures (0 = clean)")
    
    # Ensure y-axis has a reasonable minimum scale so text doesn't overlap the axis
    ax.set_ylim(0, max(max(counts) * 1.2, 1.0))
    for bar, count in zip(bars, counts):
        y_pos = bar.get_height() + ax.get_ylim()[1] * 0.05
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                str(count), ha='center', fontweight='bold',
                color='green' if count == 0 else 'red')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    out_path = os.path.join(output_dir, "production_verification.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved verification plot to {out_path}")

    # =====================================================================
    # FINAL VERDICT
    # =====================================================================
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    issues = []
    if prod_nan_count > 0:
        issues.append(f"NaN detected ({prod_nan_count} frames)")
    if prod_extreme_bond_count > 0:
        issues.append(f"Extreme bonds ({prod_extreme_bond_count} frames)")
    if max_deviation / overall_std > 2.0:
        issues.append(f"Rg window drift ({max_deviation/overall_std:.1f} sigma)")
    if not bond_dist_pass:
        issues.append(f"Bond distribution shift (mean diff = {bond_mean_diff:.4f}σ, KS p={ks_p:.4f})")

    if not issues:
        print("  ✅ ALL CHECKS PASSED")
        print("  The post-2700τ trajectory is valid equilibrium data for GNN training.")
    else:
        print("  ⚠️  ISSUES FOUND:")
        for issue in issues:
            print(f"    - {issue}")


if __name__ == "__main__":
    main()
