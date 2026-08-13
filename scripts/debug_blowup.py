"""
Longer probe: run 500K steps with every-step anomaly checking.
Also check what happens when two beads get very close (WCA edge case).
"""
import sys, os, numpy as np, yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.physics.forces import compute_all_forces
from src.physics.integrators import euler_maruyama_overdamped_step

with open(os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")) as f:
    config = yaml.safe_load(f)

N = config["chain"]["N"]

# Test 1: Check WCA force at very small distances
print("=== WCA Force at small distances ===")
from src.physics.forces import wca_force
for dist in [0.5, 0.3, 0.1, 0.05, 0.01, 1e-6, 1e-10, 1e-13]:
    r_ij = np.array([dist, 0.0, 0.0])
    f = wca_force(r_ij, 1.0, 1.0)
    print(f"  dist={dist:.1e}  ->  F_wca magnitude = {np.linalg.norm(f):.4e}")

print("\n=== Harmonic force at large distances ===")
from src.physics.forces import harmonic_bond_force
for dist in [1.0, 2.0, 5.0, 10.0, 100.0, 1000.0]:
    r_ij = np.array([dist, 0.0, 0.0])
    f = harmonic_bond_force(r_ij, 100.0, 1.0)
    print(f"  dist={dist:.1e}  ->  F_bond magnitude = {np.linalg.norm(f):.4e}")

# Test 2: 500K step run with same seed as diagnostic (base_seed + 999)
print("\n=== Long run with diagnostic seed (base_seed + 999 = 1041) ===")
rng = np.random.default_rng(1041)
positions = np.zeros((N, 3))
for i in range(N):
    positions[i, 0] = i * 1.0
positions -= np.mean(positions, axis=0)

dt = config["integrator"]["dt"]
gamma = config["thermostat"]["gamma"]
kBT = config["thermostat"]["kBT"]

anomaly_count = 0
max_rg_seen = 0
max_bond_seen = 0
max_coord_seen = 0

for t in range(1, 500001):
    forces, pe_b, pe_nb = compute_all_forces(
        positions, bond_type="harmonic", k_bond=100.0, r0=1.0,
        k_fene=30.0, R0_fene=1.5, epsilon=1.0, sigma=1.0
    )
    
    # Check for extreme forces BEFORE stepping
    max_f = np.abs(forces).max()
    if max_f > 1e4:
        print(f"  EXTREME FORCE at step {t}: max_force={max_f:.4e}")
        # Find which bead
        bead_idx = np.unravel_index(np.argmax(np.abs(forces)), forces.shape)[0]
        print(f"    Bead {bead_idx}: pos={positions[bead_idx]}, force={forces[bead_idx]}")
        if bead_idx > 0:
            d = np.linalg.norm(positions[bead_idx] - positions[bead_idx-1])
            print(f"    Bond to {bead_idx-1}: dist={d:.6f}")
        if bead_idx < N-1:
            d = np.linalg.norm(positions[bead_idx+1] - positions[bead_idx])
            print(f"    Bond to {bead_idx+1}: dist={d:.6f}")
    
    positions, _ = euler_maruyama_overdamped_step(positions, forces, dt, gamma, kBT, rng)
    
    # Check for NaN
    if np.any(np.isnan(positions)):
        print(f"  NaN at step {t}!")
        break
    
    bond_vecs = positions[1:] - positions[:-1]
    bond_lengths = np.linalg.norm(bond_vecs, axis=1)
    com = np.mean(positions, axis=0)
    rg = np.sqrt(np.mean(np.sum((positions - com)**2, axis=1)))
    
    max_rg_seen = max(max_rg_seen, rg)
    max_bond_seen = max(max_bond_seen, bond_lengths.max())
    max_coord_seen = max(max_coord_seen, np.abs(positions).max())
    
    if rg > 100 or bond_lengths.max() > 50:
        anomaly_count += 1
        if anomaly_count <= 5:
            print(f"  SPIKE step {t}: Rg={rg:.2f} max_bond={bond_lengths.max():.2f} max_coord={np.abs(positions).max():.2f}")
    
    if t % 100000 == 0:
        print(f"  Step {t}: Rg={rg:.4f} max_bond={bond_lengths.max():.4f} max_coord={np.abs(positions).max():.2f}")

print(f"\n=== Summary ===")
print(f"Total anomalies (Rg>100 or bond>50): {anomaly_count}")
print(f"Max Rg seen: {max_rg_seen:.4f}")
print(f"Max bond length seen: {max_bond_seen:.4f}")
print(f"Max coordinate seen: {max_coord_seen:.4f}")
