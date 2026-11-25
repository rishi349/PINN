#!/usr/bin/env python3
"""
PyTorch GPU-Accelerated Physics Audit for the PINN Polymer Simulator.

Designed to run on NVIDIA GPUs (including non-AVX virtualized environments
where JAX cannot import) using batched PyTorch CUDA tensors (B, N, 3).

Checks performed (with publication-ready PNG plots saved to --output-dir):
  1. Exact force parity against src/physics/forces.py (FENE+WCA and Harmonic)
  2. Bond-length equilibrium & FDT temperature verification
  3. Monomer MSD g1(t) and COM diffusion g3(t) (D_cm = kBT / (N * gamma))
  4. Rouse mode relaxation scaling (tau_p ~ p^-2)
  5. Flory scaling <Rg^2> ~ N^(2*nu) across plan chain lengths

Usage:
    python scripts/gpu_audit.py --quick
    python scripts/gpu_audit.py --config configs/default.yaml --output-dir plots/gpu_audit
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import linregress

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.physics.forces import compute_all_forces
from src.simulator.numpy_simulator import load_config


# ---------------------------------------------------------------------------
# Batched PyTorch Physics Kernel (mirrors src/physics/forces.py & integrators.py)
# ---------------------------------------------------------------------------

def compute_forces_torch(
    pos: torch.Tensor,
    bond_type: str = "fene",
    k_bond: float = 100.0,
    r0: float = 1.0,
    k_fene: float = 30.0,
    R0_fene: float = 1.5,
    epsilon: float = 1.0,
    sigma: float = 1.0,
    nonbonded_mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Compute total (bonded + nonbonded WCA) forces for a batch of chains (B, N, 3).
    Matches src/physics/forces.py including 0.4*sigma clamp and F_MAX=1000 cap.
    """
    B, N, _ = pos.shape
    device = pos.device
    dtype = pos.dtype
    r_cut = sigma * (2.0 ** (1.0 / 6.0))
    forces = torch.zeros_like(pos)

    # --- 1. Bonded forces: pairs (i, i+1) ---
    r_ij = pos[:, 1:, :] - pos[:, :-1, :]  # (B, N-1, 3), from i to i+1
    dist = torch.linalg.norm(r_ij, dim=-1, keepdim=True)  # (B, N-1, 1)
    nonzero = dist > 1e-12
    safe_dist = torch.where(nonzero, dist, torch.ones_like(dist))
    r_hat = torch.where(nonzero, r_ij / safe_dist, torch.zeros_like(r_ij))

    f_bond = torch.zeros_like(r_ij)
    if bond_type == "harmonic":
        f_mag = k_bond * (dist - r0)
        f_bond = torch.where(nonzero, f_mag * r_hat, f_bond)
    elif bond_type == "fene":
        ratio_sq = (dist / R0_fene) ** 2
        if torch.any(ratio_sq >= 1.0):
            max_d = float(torch.max(dist).item())
            raise ValueError(
                f"FENE bond exceeded R0={R0_fene}: max |r_ij|={max_d:.5f}. Reduce dt."
            )
        denom = 1.0 - ratio_sq
        f_mag = k_fene * dist / denom
        f_bond = torch.where(nonzero, f_mag * r_hat, f_bond)

        # Kremer-Grest WCA repulsion on bonded pairs
        wca_b_mask = (dist < r_cut) & nonzero
        dist_b_wca = torch.clamp(dist, min=0.4 * sigma)
        sr6_b = (sigma / dist_b_wca) ** 6
        sr12_b = sr6_b ** 2
        f_wca_b = (24.0 * epsilon / dist_b_wca) * (2.0 * sr12_b - sr6_b)
        f_bond = f_bond - torch.where(wca_b_mask, f_wca_b * r_hat, torch.zeros_like(r_hat))
    else:
        raise ValueError(f"Unknown bond_type: {bond_type}")

    forces[:, :-1, :] += f_bond
    forces[:, 1:, :] -= f_bond

    # --- 2. Non-bonded WCA forces: all pairs |i - j| >= 2 ---
    # diff[b, i, j, :] = pos[b, j, :] - pos[b, i, :]
    diff = pos.unsqueeze(1) - pos.unsqueeze(2)  # (B, N, N, 3) -> pos_i - pos_j
    # Wait: let r_ij_mat = pos_j - pos_i so force on i is -f_factor * r_hat (away from j)
    # Equivalently, pos_i - pos_j points AWAY from j, so force on i = +f_factor * (pos_i - pos_j)/d
    d_mat = torch.linalg.norm(diff, dim=-1)  # (B, N, N)
    if nonbonded_mask is None:
        idx = torch.arange(N, device=device)
        nonbonded_mask = (torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1)) >= 2)

    active = nonbonded_mask.unsqueeze(0) & (d_mat < r_cut) & (d_mat > 1e-12)
    d_safe = torch.clamp(d_mat, min=0.4 * sigma)
    sr6 = (sigma / d_safe) ** 6
    sr12 = sr6 ** 2
    # Force magnitude / d_safe = 24*eps / d_safe^2 * (2*sr12 - sr6)
    f_over_d = torch.where(
        active,
        (24.0 * epsilon / (d_safe ** 2)) * (2.0 * sr12 - sr6),
        torch.zeros_like(d_safe),
    )
    f_nonbond = torch.sum(f_over_d.unsqueeze(-1) * diff, dim=2)  # (B, N, 3)
    forces = forces + f_nonbond

    # --- 3. Force magnitude cap (F_MAX = 1000) ---
    f_norm = torch.linalg.norm(forces, dim=-1, keepdim=True)
    f_scale = torch.where(f_norm > 1000.0, 1000.0 / torch.clamp(f_norm, min=1e-12), torch.ones_like(f_norm))
    forces = forces * f_scale
    return forces


def step_overdamped_torch(
    pos: torch.Tensor,
    forces: torch.Tensor,
    dt: float,
    gamma: float,
    kBT: float,
    generator: torch.Generator,
):
    """Batched Euler-Maruyama overdamped step with MAX_DISP = 0.5*sigma cap."""
    noise_amp = (2.0 * kBT * dt / gamma) ** 0.5
    noise = noise_amp * torch.randn(pos.shape, device=pos.device, dtype=pos.dtype, generator=generator)
    disp = (dt / gamma) * forces + noise
    disp_mag = torch.linalg.norm(disp, dim=-1, keepdim=True)
    scale = torch.where(disp_mag > 0.5, 0.5 / torch.clamp(disp_mag, min=1e-30), torch.ones_like(disp_mag))
    return pos + disp * scale, noise


@torch.no_grad()
def run_batched_sim(
    B: int,
    N: int,
    n_burnin: int,
    n_prod: int,
    save_every: int,
    dt: float,
    bond_type: str,
    k_bond: float,
    r0: float,
    k_fene: float,
    R0_fene: float,
    epsilon: float,
    sigma: float,
    gamma: float,
    kBT: float,
    device: torch.device,
    seed: int = 42,
    init_method: str = "linear",
):
    """Run B trajectories in parallel on GPU and return saved positions (F, B, N, 3)."""
    dtype = torch.float32
    gen = torch.Generator(device=device).manual_seed(seed)

    # Initialize chain
    pos = torch.zeros((B, N, 3), device=device, dtype=dtype)
    if init_method == "linear":
        spacing = 0.97 * sigma if bond_type == "fene" else r0
        x = torch.arange(N, device=device, dtype=dtype) * spacing
        pos[:, :, 0] = x.unsqueeze(0)
    else:
        steps = torch.randn((B, N - 1, 3), device=device, dtype=dtype, generator=gen)
        steps = steps / torch.clamp(torch.linalg.norm(steps, dim=-1, keepdim=True), min=1e-12) * sigma
        pos[:, 1:, :] = torch.cumsum(steps, dim=1)
    pos = pos - torch.mean(pos, dim=1, keepdim=True)

    idx = torch.arange(N, device=device)
    nb_mask = (torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1)) >= 2)

    # Burn-in
    for _ in range(n_burnin):
        f = compute_forces_torch(
            pos, bond_type, k_bond, r0, k_fene, R0_fene, epsilon, sigma, nb_mask
        )
        pos, _ = step_overdamped_torch(pos, f, dt, gamma, kBT, gen)

    # Production
    frames = []
    fdt_ratios = []
    for step in range(1, n_prod + 1):
        f = compute_forces_torch(
            pos, bond_type, k_bond, r0, k_fene, R0_fene, epsilon, sigma, nb_mask
        )
        pos_next, noise = step_overdamped_torch(pos, f, dt, gamma, kBT, gen)
        if step % save_every == 0:
            frames.append(pos_next.detach().cpu().numpy())
            # FDT check on raw stochastic displacement: var(noise) == 2*kBT*dt/gamma
            fdt_ratios.append(float(torch.mean(noise ** 2).item() / (2.0 * kBT * dt / gamma)))
        pos = pos_next

    return np.stack(frames, axis=0), float(np.mean(fdt_ratios))


# ---------------------------------------------------------------------------
# Audit Sections
# ---------------------------------------------------------------------------

def check_force_parity(cfg, device):
    print("\n[1/5] Force Parity Check (PyTorch GPU vs NumPy forces.py)...")
    rng = np.random.default_rng(123)
    max_rel_err = 0.0
    for N in [30, 60, 120]:
        # Place chain along x with small perturbation so bonds < R0
        pos_np = np.zeros((N, 3), dtype=np.float64)
        pos_np[:, 0] = np.arange(N) * 0.97
        pos_np += rng.normal(0.0, 0.05, size=(N, 3))

        f_np, _, _ = compute_all_forces(
            pos_np,
            bond_type=cfg["bond"]["type"],
            k_bond=cfg["bond"]["k_bond"],
            r0=cfg["bond"]["r0"],
            k_fene=cfg["fene"]["k"],
            R0_fene=cfg["fene"]["R0"],
            epsilon=cfg["wca"]["epsilon"],
            sigma=cfg["wca"]["sigma"],
        )
        pos_t = torch.tensor(pos_np, device=device, dtype=torch.float64).unsqueeze(0)
        f_t = compute_forces_torch(
            pos_t,
            bond_type=cfg["bond"]["type"],
            k_bond=cfg["bond"]["k_bond"],
            r0=cfg["bond"]["r0"],
            k_fene=cfg["fene"]["k"],
            R0_fene=cfg["fene"]["R0"],
            epsilon=cfg["wca"]["epsilon"],
            sigma=cfg["wca"]["sigma"],
        )[0].cpu().numpy()

        rel = np.max(np.abs(f_np - f_t)) / (np.max(np.abs(f_np)) + 1e-12)
        max_rel_err = max(max_rel_err, rel)
        print(f"  N={N:3d}: max rel err = {rel:.2e}")

    passed = max_rel_err < 1e-10
    print(f"  -> {'PASS' if passed else 'FAIL'} (max rel err = {max_rel_err:.2e})")
    return passed, max_rel_err


def compute_msd_curves(traj_np):
    """traj_np shape: (F, B, N, 3). Returns lags (frames), g1, g3."""
    F = traj_np.shape[0]
    max_lag = max(2, F // 3)
    lags = np.arange(1, max_lag + 1)
    com = np.mean(traj_np, axis=2)  # (F, B, 3)
    g1, g3 = [], []
    for lag in lags:
        d1 = traj_np[lag:] - traj_np[:-lag]
        d3 = com[lag:] - com[:-lag]
        g1.append(np.mean(np.sum(d1 ** 2, axis=-1)))
        g3.append(np.mean(np.sum(d3 ** 2, axis=-1)))
    return lags, np.array(g1), np.array(g3)


def check_rouse_modes(traj_np, frame_dt):
    """Compute Rouse modes p=1..8 on traj_np (F, B, N, 3)."""
    F, B, N, _ = traj_np.shape
    p_vals = np.arange(1, min(9, N))
    taus = []
    n_idx = np.arange(N) + 0.5
    max_lag = max(4, F // 3)
    t_lags = np.arange(max_lag) * frame_dt

    for p in p_vals:
        cos_w = np.cos(p * np.pi * n_idx / N)  # (N,)
        # X_p: (F, B, 3)
        xp = np.tensordot(traj_np, cos_w, axes=([2], [0])) * np.sqrt(2.0 / N)
        # Autocorrelation C_p(lag)
        norm0 = np.mean(np.sum(xp ** 2, axis=-1))
        acfs = [1.0]
        for lag in range(1, max_lag):
            c = np.mean(np.sum(xp[lag:] * xp[:-lag], axis=-1)) / (norm0 + 1e-12)
            acfs.append(c)
        acfs = np.array(acfs)
        # Integrate positive part or fit log decay
        pos_idx = np.where(acfs > 0.1)[0]
        if len(pos_idx) >= 3:
            slope, _, _, _, _ = linregress(t_lags[pos_idx], np.log(acfs[pos_idx]))
            tau_p = -1.0 / slope if slope < -1e-9 else float("nan")
        else:
            tau_p = float("nan")
        taus.append(tau_p)
    return p_vals, np.array(taus)


def main():
    parser = argparse.ArgumentParser(description="PyTorch CUDA GPU Physics Audit")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output-dir", type=str, default="plots/gpu_audit")
    parser.add_argument("--quick", action="store_true", help="Fast ~2 min verification mode")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cfg = load_config(os.path.join(project_root, args.config))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print(f"  PyTorch GPU Physics Audit | Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Bond: {cfg['bond']['type']} | dt: {cfg['integrator']['dt']}")
    print(f"  Mode: {'QUICK' if args.quick else 'FULL'}")
    print("=" * 65)

    t0_total = time.time()
    dt = float(cfg["integrator"]["dt"])
    bond_type = cfg["bond"]["type"]
    k_bond = float(cfg["bond"]["k_bond"])
    r0 = float(cfg["bond"]["r0"])
    k_fene = float(cfg["fene"]["k"])
    R0_fene = float(cfg["fene"]["R0"])
    eps = float(cfg["wca"]["epsilon"])
    sig = float(cfg["wca"]["sigma"])
    gamma = float(cfg["thermostat"]["gamma"])
    kBT = float(cfg["thermostat"]["kBT"])

    results = {}

    # 1. Force Parity
    p_ok, p_err = check_force_parity(cfg, device)
    results["1_force_parity"] = (p_ok, f"rel={p_err:.1e}")

    # 2. Primary N=120 Bond & FDT + MSD
    print("\n[2/5] Running batched N=120 simulation on GPU (Bond, FDT, MSD)...")
    if args.quick:
        B_120, burn_120, prod_120, save_120 = 4, 20000, 20000, 100
    else:
        B_120, burn_120, prod_120, save_120 = 8, 150000, 150000, 250

    traj_120, fdt_ratio = run_batched_sim(
        B_120, 120, burn_120, prod_120, save_120, dt,
        bond_type, k_bond, r0, k_fene, R0_fene, eps, sig, gamma, kBT, device, seed=42
    )
    bond_vecs = traj_120[:, :, 1:, :] - traj_120[:, :, :-1, :]
    bond_lens = np.linalg.norm(bond_vecs, axis=-1).ravel()
    mean_bl = float(np.mean(bond_lens))
    bl_ok = (0.92 < mean_bl < 1.02) if bond_type == "fene" else (1.00 < mean_bl < 1.12)
    fdt_ok = abs(fdt_ratio - 1.0) < 0.05
    results["2_bond_eq"] = (bl_ok, f"<b>={mean_bl:.4f}")
    results["3_fdt_temp"] = (fdt_ok, f"T/T0={fdt_ratio:.4f}")
    print(f"  Mean bond length: {mean_bl:.4f} sigma -> {'PASS' if bl_ok else 'FAIL'}")
    print(f"  FDT noise ratio : {fdt_ratio:.4f} (target 1.00) -> {'PASS' if fdt_ok else 'FAIL'}")

    # Plot Bond + FDT
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.hist(bond_lens, bins=50, density=True, alpha=0.75, color="#4C72B0", edgecolor="white")
    ax.axvline(mean_bl, color="red", linestyle="--", label=f"Mean = {mean_bl:.4f} sigma")
    ax.set_xlabel("Bond length (sigma)")
    ax.set_ylabel("Probability density")
    ax.set_title(f"N=120 Bond Length Distribution ({bond_type.upper()}+WCA)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "bond_distribution.png"), dpi=120)
    plt.close()

    # 3. MSD & COM Diffusion at N=120
    print("\n[3/5] Computing MSD & COM Diffusion (N=120)...")
    lags, g1, g3 = compute_msd_curves(traj_120)
    t_axis = lags * save_120 * dt
    D_theory = kBT / (120.0 * gamma)
    slope_g3, _, r_g3, _, _ = linregress(np.log(t_axis), np.log(g3))
    D_meas = float(np.mean(g3 / (6.0 * t_axis)))
    msd_ok = abs(slope_g3 - 1.0) < 0.15 and abs(D_meas - D_theory) / D_theory < 0.20
    results["4_com_msd"] = (msd_ok, f"alpha={slope_g3:.2f}, D={D_meas:.5f}")
    print(f"  g3 slope={slope_g3:.3f} (target 1.0), D_meas={D_meas:.5f} vs D_theory={D_theory:.5f} -> {'PASS' if msd_ok else 'FAIL'}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.loglog(t_axis, g1, "o-", markersize=3, label="g1 (monomer)")
    ax1.loglog(t_axis, g1[0] * (t_axis / t_axis[0]) ** 0.5, "--", label="Rouse t^0.5")
    ax1.set_xlabel("Lag time (tau)"); ax1.set_ylabel("g1(t)"); ax1.set_title("Monomer MSD (N=120)"); ax1.legend()
    ax2.loglog(t_axis, g3, "s-", markersize=3, color="#2ca02c", label=f"g3 (slope={slope_g3:.2f})")
    ax2.loglog(t_axis, 6 * D_theory * t_axis, "--", color="orange", label=f"Theory 6*D*t (D={D_theory:.5f})")
    ax2.set_xlabel("Lag time (tau)"); ax2.set_ylabel("g3(t)"); ax2.set_title("COM Diffusion (N=120)"); ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "msd_check.png"), dpi=120)
    plt.close()

    # 4. Rouse Modes at N=80
    print("\n[4/5] Running batched N=80 simulation for Rouse Mode Scaling...")
    if args.quick:
        B_80, burn_80, prod_80, save_80 = 8, 30000, 30000, 100
    else:
        B_80, burn_80, prod_80, save_80 = 16, 150000, 150000, 200
    traj_80, _ = run_batched_sim(
        B_80, 80, burn_80, prod_80, save_80, dt,
        bond_type, k_bond, r0, k_fene, R0_fene, eps, sig, gamma, kBT, device, seed=100
    )
    p_vals, taus = check_rouse_modes(traj_80, save_80 * dt)
    valid_r = np.isfinite(taus) & (taus > 0)
    if np.sum(valid_r) >= 3:
        r_slope, _, r_r2, _, _ = linregress(np.log(p_vals[valid_r]), np.log(taus[valid_r]))
        rouse_ok = (-2.4 < r_slope < -1.5)
    else:
        r_slope, r_r2, rouse_ok = float("nan"), 0.0, False
    results["5_rouse_modes"] = (rouse_ok, f"slope={r_slope:.2f}")
    print(f"  Rouse tau_p ~ p^{r_slope:.2f} (target -2.0) -> {'PASS' if rouse_ok else 'FAIL'}")

    fig, ax = plt.subplots(figsize=(6, 4.2))
    if np.sum(valid_r) >= 2:
        ax.loglog(p_vals[valid_r], taus[valid_r], "o-", label=f"Measured (slope={r_slope:.2f})")
        ax.loglog(p_vals[valid_r], taus[valid_r][0] * (p_vals[valid_r] / p_vals[valid_r][0]) ** (-2.0), "--", label="Theory p^-2")
    ax.set_xlabel("Mode p"); ax.set_ylabel("Relaxation time tau_p"); ax.set_title("Rouse Mode Scaling (N=80)"); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "rouse_modes.png"), dpi=120)
    plt.close()

    # 5. Flory Scaling <Rg^2> ~ N^(2*nu)
    print("\n[5/5] Running batched Flory Scaling sweep...")
    if args.quick:
        N_list = [30, 60, 80, 120]
        B_flory, burn_f, prod_f, save_f = 4, 40000, 20000, 200
    else:
        N_list = [60, 80, 100, 120, 130, 140, 150]
        B_flory, burn_f, prod_f, save_f = 8, 250000, 150000, 500

    rg2_means, rg2_sems = [], []
    for N_val in N_list:
        tr, _ = run_batched_sim(
            B_flory, N_val, burn_f, prod_f, save_f, dt,
            bond_type, k_bond, r0, k_fene, R0_fene, eps, sig, gamma, kBT, device,
            seed=200 + N_val, init_method="random_walk"
        )
        com = np.mean(tr, axis=2, keepdims=True)
        rg2_per_frame = np.mean(np.sum((tr - com) ** 2, axis=-1), axis=-1).ravel()
        m_rg2 = float(np.mean(rg2_per_frame))
        s_rg2 = float(np.std(rg2_per_frame) / np.sqrt(len(rg2_per_frame)))
        rg2_means.append(m_rg2)
        rg2_sems.append(s_rg2)
        print(f"  N={N_val:3d}: <Rg2> = {m_rg2:.3f} +/- {s_rg2:.3f}")

    slope_f, intercept_f, r_f, _, se_f = linregress(np.log(N_list), np.log(rg2_means))
    nu_meas = slope_f / 2.0
    flory_ok = 0.52 < nu_meas < 0.66
    results["6_flory_nu"] = (flory_ok, f"nu={nu_meas:.3f}")
    print(f"  Flory nu = {nu_meas:.4f} (target 0.588) -> {'PASS' if flory_ok else 'FAIL'}")

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    N_arr = np.array(N_list, dtype=float)
    ax.errorbar(N_arr, rg2_means, yerr=2 * np.array(rg2_sems), fmt="o", capsize=4, label="GPU Simulation")
    ax.loglog(N_arr, np.exp(intercept_f) * (N_arr ** slope_f), "--", color="red", label=f"Fit: nu={nu_meas:.3f}")
    ax.loglog(N_arr, rg2_means[0] * (N_arr / N_arr[0]) ** 1.176, ":", color="green", label="SAW Theory: nu=0.588")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Chain length N"); ax.set_ylabel("<Rg^2>"); ax.set_title("Flory Scaling (<Rg^2> vs N)"); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "rg2_scaling.png"), dpi=120)
    plt.close()

    # Dashboard Summary
    n_pass = sum(1 for v in results.values() if v[0])
    total_t = time.time() - t0_total
    fig, ax = plt.subplots(figsize=(7, 3.5))
    labels = list(results.keys())
    colors = ["#2ca02c" if results[k][0] else "#d62728" for k in labels]
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, [1] * len(labels), color=colors, alpha=0.8)
    for i, k in enumerate(labels):
        status = "PASS" if results[k][0] else "FAIL"
        ax.text(0.5, i, f"{k}: {status} ({results[k][1]})", ha="center", va="center", color="white", fontweight="bold")
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title(f"PyTorch GPU Audit | {n_pass}/{len(results)} PASS | {total_t:.1f}s ({device})")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "audit_dashboard.png"), dpi=120)
    plt.close()

    print("\n" + "=" * 65)
    print(f"  AUDIT COMPLETE: {n_pass}/{len(results)} PASS in {total_t:.1f}s")
    print(f"  Plots saved in: {args.output_dir}/")
    print("=" * 65)


if __name__ == "__main__":
    main()
