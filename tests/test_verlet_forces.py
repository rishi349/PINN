"""Correctness test: all-pairs (original path, neighbor_pairs=None) vs Verlet list."""
import numpy as np
from src.physics.forces import build_verlet_neighbor_list, compute_all_forces, compute_nonbonded_forces

SIGMA, EPS = 1.0, 1.0
SKIN = 0.3 * SIGMA
R_CUT = SIGMA * 2 ** (1 / 6)


def random_chain(rng, N, spacing=0.97, compress=1.0):
    steps = rng.standard_normal((N, 3))
    steps /= np.linalg.norm(steps, axis=1, keepdims=True)
    pos = np.cumsum(steps * spacing, axis=0) * compress
    return pos - pos.mean(axis=0)


def check_equal(pos, pairs, bond_type):
    f0, b0, n0 = compute_all_forces(pos, bond_type=bond_type, epsilon=EPS, sigma=SIGMA)
    f1, b1, n1 = compute_all_forces(pos, bond_type=bond_type, epsilon=EPS, sigma=SIGMA,
                                    neighbor_pairs=pairs)
    assert np.array_equal(f0, f1), np.abs(f0 - f1).max()
    assert b0 == b1 and n0 == n1, (b0, b1, n0, n1)


def test_static():
    rng = np.random.default_rng(0)
    for N in (2, 3, 4, 10, 50, 200):
        for compress in (1.0, 0.5, 0.3):   # compress -> overlaps, exercises 0.4σ clamp + force clip
            pos = random_chain(rng, N, compress=compress)
            pairs = build_verlet_neighbor_list(pos, sigma=SIGMA, skin=SKIN)
            check_equal(pos, pairs, "harmonic")
    print("static equality (bitwise): OK")


def test_dynamic_no_missed_pairs():
    """Random-walk the beads; rebuild only when max displacement >= skin/2."""
    rng = np.random.default_rng(1)
    N = 100
    pos = random_chain(rng, N, compress=0.8)
    pairs = build_verlet_neighbor_list(pos, SIGMA, SKIN)
    ref = pos.copy()
    rebuilds = 0
    for step in range(3000):
        pos = pos + 0.01 * rng.standard_normal(pos.shape)
        if np.max(np.linalg.norm(pos - ref, axis=1)) >= 0.5 * SKIN:
            pairs = build_verlet_neighbor_list(pos, SIGMA, SKIN)
            ref = pos.copy()
            rebuilds += 1
        # nonbonded only (bond forces are irrelevant to list validity, and the
        # random walk would blow up FENE)
        f0, p0 = compute_nonbonded_forces(pos, EPS, SIGMA)
        f1, p1 = compute_nonbonded_forces(pos, EPS, SIGMA, neighbor_pairs=pairs)
        assert np.array_equal(f0, f1) and p0 == p1, f"mismatch at step {step}"
        # direct check: every pair within the physical cutoff is in the list
        i, j = np.triu_indices(N, k=2)
        d = np.linalg.norm(pos[j] - pos[i], axis=1)
        inside = set(zip(i[d < R_CUT], j[d < R_CUT]))
        assert inside <= set(zip(*pairs)), f"missed pair at step {step}"
    print(f"dynamic test OK over 3000 steps, {rebuilds} rebuilds")


if __name__ == "__main__":
    test_static()
    test_dynamic_no_missed_pairs()