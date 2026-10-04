#!/usr/bin/env python
"""
Diagnostic script to check polymer chain equilibration.

Generates the 6 priority checks:
1. Radius of gyration (Rg) over time
2. End-to-end distance (Ree) over time
3. Bond length and distribution
4. Potential energy (PE) over time
5. Shape/asphericity (Relative Shape Anisotropy) over time
6. Autocorrelation of Rg during production

Output is saved to `plots/equilibration/`.
"""

import os
import sys
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
from src.simulator.numpy_simulator import NumpySimulator
from src.physics.forces import compute_all_forces
from src.physics.integrators import euler_maruyama_overdamped_step

def compute_shape_anisotropy(positions: np.ndarray) -> float:
    """Compute Relative Shape Anisotropy (kappa^2).
    kappa^2 = 1 - 3 * (l1*l2 + l2*l3 + l3*l1) / (l1+l2+l3)^2
    where l_i are the eigenvalues of the gyration tensor.
    kappa^2 = 0 for a sphere, 1 for a line.
    """
    com = np.mean(positions, axis=0)
    shifted = positions - com
    S = np.dot(shifted.T, shifted) / len(positions)
    
    # Avoid numerical issues for perfectly degenerate cases
    try:
        eigenvalues = np.linalg.eigvalsh(S)
        # Ensure eigenvalues are positive
        eigenvalues = np.clip(eigenvalues, 1e-12, None)
        l1, l2, l3 = eigenvalues
        trace = l1 + l2 + l3
        kappa_sq = 1 - 3 * (l1*l2 + l2*l3 + l3*l1) / (trace**2)
        return float(np.clip(kappa_sq, 0.0, 1.0))
    except np.linalg.LinAlgError:
        return 0.0

def autocorrelation(x: np.ndarray) -> np.ndarray:
    """Compute autocorrelation function of a 1D array using FFT."""
    if len(x) == 0:
        return np.array([])
    xp = x - np.mean(x)
    var = np.var(x)
    if var == 0:
        return np.ones(len(x)//2)
    f = np.fft.fft(xp, n=2*len(x))
    p = np.real(f)**2 + np.imag(f)**2
    pi = np.fft.ifft(p)
    ac = np.real(pi)[:len(x)//2] / (len(x) * var)
    return ac

def main():
    parser = argparse.ArgumentParser(description="Run equilibration diagnostics.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output-dir", type=str, default="plots/equilibration")
    parser.add_argument("--log-every", type=int, default=1000, 
                        help="Log observables every N steps during the entire run")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # We will use NumpySimulator just to initialize the parameters and chain
    sim = NumpySimulator(config)
    seed = sim.base_seed  # Match the trajectory seed
    rng = np.random.default_rng(seed)
    
    positions = sim.initialize_chain(rng)
    
    T_steps = sim.T_steps
    n_burnin = sim.n_burnin
    dt = sim.dt
    log_every = args.log_every
    
    # Data storage
    times = []
    rg_list = []
    ree_list = []
    pe_list = []
    mean_bond_list = []
    shape_list = []
    all_bonds_prod = []  # To store bond lengths from the production phase
    
    print(f"Starting diagnostic run: {T_steps} steps (dt={dt})")
    print(f"Burn-in finishes at step {n_burnin}")
    print(f"Logging every {log_every} steps...")
    
    start_time = time.time()
    
    for t in range(1, T_steps + 1):
        forces, pe_bond, pe_nonbond = compute_all_forces(
            positions,
            bond_type=sim.bond_type,
            k_bond=sim.k_bond,
            r0=sim.r0,
            k_fene=sim.k_fene,
            R0_fene=sim.R0_fene,
            epsilon=sim.epsilon,
            sigma=sim.sigma,
        )

        positions, _ = euler_maruyama_overdamped_step(
            positions, forces, sim.dt, sim.gamma, sim.kBT, rng
        )
        
        if t % log_every == 0:
            obs = sim.compute_observables(positions)
            
            times.append(t * dt)  # Physical time tau
            rg_list.append(obs["radius_of_gyration"])
            ree_list.append(obs["end_to_end_distance"])
            pe_list.append(pe_bond + pe_nonbond)
            mean_bond_list.append(obs["mean_bond_length"])
            shape_list.append(compute_shape_anisotropy(positions))
            
            if t > n_burnin:
                all_bonds_prod.extend(obs["bond_lengths"])
                
            if t % (T_steps // 10) == 0:
                print(f"  [{t}/{T_steps}] Rg={rg_list[-1]:.2f}, Ree={ree_list[-1]:.2f}")

    print(f"Simulation completed in {time.time() - start_time:.1f}s")
    
    # Convert to numpy arrays for plotting
    times = np.array(times)
    rg_list = np.array(rg_list)
    ree_list = np.array(ree_list)
    pe_list = np.array(pe_list)
    mean_bond_list = np.array(mean_bond_list)
    shape_list = np.array(shape_list)
    
    burn_in_time = n_burnin * dt
    
    # Create the 6 plots
    fig, axs = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Polymer Equilibration Diagnostics", fontsize=16)
    
    # Helper to plot a line with burn-in marked
    def plot_trace(ax, y, ylabel, title):
        ax.plot(times, y, color='blue', alpha=0.7)
        ax.axvline(x=burn_in_time, color='red', linestyle='--', label='Burn-in Ends')
        ax.set_xlabel("Time (tau)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 1. Rg(t)
    plot_trace(axs[0, 0], rg_list, "R_g", "1. Radius of Gyration")
    
    # 2. Ree(t)
    plot_trace(axs[0, 1], ree_list, "R_{ee}", "2. End-to-End Distance")
    
    # 3. Bond distribution
    ax_bond = axs[0, 2]
    ax_bond.plot(times, mean_bond_list, color='green', label='Mean Bond Length')
    ax_bond.axvline(x=burn_in_time, color='red', linestyle='--')
    ax_bond.set_xlabel("Time (tau)")
    ax_bond.set_ylabel("Length")
    ax_bond.set_title("3. Mean Bond Length & Distribution")
    ax_bond.legend()
    ax_bond.grid(True, alpha=0.3)
    
    # Insert an inset for the histogram of production bonds
    if len(all_bonds_prod) > 0:
        ins_ax = ax_bond.inset_axes([0.5, 0.1, 0.45, 0.35])
        ins_ax.hist(all_bonds_prod, bins=50, color='purple', alpha=0.7, density=True)
        ins_ax.set_title("Prod. Dist.", fontsize=8)
        ins_ax.set_yticks([])
        
    # 4. Potential Energy
    plot_trace(axs[1, 0], pe_list, "U", "4. Potential Energy")
    
    # 5. Shape Asphericity
    plot_trace(axs[1, 1], shape_list, "kappa^2", "5. Relative Shape Anisotropy")
    
    # 6. Autocorrelation of Rg in production phase
    ax_ac = axs[1, 2]
    prod_indices = times > burn_in_time
    if np.any(prod_indices):
        rg_prod = rg_list[prod_indices]
        ac = autocorrelation(rg_prod)
        lag_times = np.arange(len(ac)) * (log_every * dt)
        ax_ac.plot(lag_times, ac, color='orange')
        ax_ac.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax_ac.axhline(y=np.exp(-1), color='gray', linestyle='--', label='1/e')
        ax_ac.set_xlabel("Lag Time (tau)")
        ax_ac.set_ylabel("Autocorrelation")
        ax_ac.set_title("6. R_g Autocorrelation (Production)")
        ax_ac.set_xlim(0, max(1, min(lag_times[-1], 200))) # Zoom in on the decay
        ax_ac.legend()
        ax_ac.grid(True, alpha=0.3)
    else:
        ax_ac.text(0.5, 0.5, "No production data", ha='center', va='center')
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    output_file = os.path.join(args.output_dir, "equilibration_diagnostics.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved diagnostic plot to {output_file}")

if __name__ == "__main__":
    main()
