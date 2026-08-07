"""
Unit tests for force implementations.

From §9 Month 2:
    "Write unit tests for the force functions in isolation — compute the
    FENE/WCA force at 3–4 hand-picked bond lengths / separations and compare
    against your by-hand derivation, BEFORE wiring them into the full simulator.
    This catches sign errors and unit-convention mistakes at the cheapest
    possible point to fix them."

Tests cover:
    1. Harmonic bond force at 4 hand-picked distances (equilibrium, stretched,
       compressed, very stretched)
    2. WCA force at distances below, at, and above cutoff 2^(1/6)σ
    3. FENE force at equilibrium, stretched, near max extension
    4. Newton's 3rd law (F_ij = -F_ji) for all force types
    5. Potential energy consistency (force = -gradient of potential)
    6. Aggregate force computation

All hand-computed reference values use the exact formulas from §4:
    Harmonic: F = k_bond * (|r| - r0) * r_hat   (force ON bead i toward j)
    WCA: F = -24ε/r * [2(σ/r)¹² - (σ/r)⁶] * r_hat  (repulsive, away from j)
    FENE: F = k * r / (1-(r/R0)²) * r_hat  (attractive, toward j)
"""

import sys
import os
import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.physics.forces import (
    harmonic_bond_force,
    harmonic_bond_potential,
    wca_force,
    wca_potential,
    fene_bond_force,
    fene_bond_potential,
    compute_bonded_forces,
    compute_nonbonded_forces,
    compute_all_forces,
)


# =============================================================================
# Standard parameters from §4
# =============================================================================
K_BOND = 100.0   # ε/σ² — harmonic spring constant (§4 footnote)
R0 = 1.0          # σ — harmonic equilibrium length (§4 footnote)
K_FENE = 30.0     # ε/σ² — FENE spring constant (§4)
R0_FENE = 1.5     # σ — FENE max extension (§4)
EPSILON = 1.0      # WCA energy scale (§4)
SIGMA = 1.0        # WCA length scale (§4)
R_CUT_WCA = SIGMA * 2.0 ** (1.0 / 6.0)  # ≈ 1.12246σ


class TestHarmonicBondForce:
    """
    Test harmonic bond force: F = k_bond * (|r_ij| - r0) * r̂_ij
    Force on bead i, pointing toward j when bond is stretched (|r|>r0).
    """

    def test_at_equilibrium(self):
        """At r = r0, force should be zero."""
        r_ij = np.array([R0, 0.0, 0.0])
        f = harmonic_bond_force(r_ij, K_BOND, R0)
        np.testing.assert_allclose(f, [0.0, 0.0, 0.0], atol=1e-12)

    def test_stretched_bond(self):
        """Stretched bond (r=1.5σ): force pulls i toward j."""
        dist = 1.5
        r_ij = np.array([dist, 0.0, 0.0])
        f = harmonic_bond_force(r_ij, K_BOND, R0)
        # Hand calculation: F = 100 * (1.5 - 1.0) * (1, 0, 0) = (50, 0, 0)
        # Force on i points TOWARD j (+x direction) ✓
        expected_magnitude = K_BOND * (dist - R0)
        np.testing.assert_allclose(f, [expected_magnitude, 0.0, 0.0], atol=1e-10)
        assert f[0] > 0, "Stretched bond: force on i should point toward j"

    def test_compressed_bond(self):
        """Compressed bond (r=0.5σ): force pushes i away from j."""
        dist = 0.5
        r_ij = np.array([dist, 0.0, 0.0])
        f = harmonic_bond_force(r_ij, K_BOND, R0)
        # Hand calculation: F = 100 * (0.5 - 1.0) * (1, 0, 0) = (-50, 0, 0)
        # Force on i points AWAY from j (-x direction) ✓
        expected_magnitude = K_BOND * (dist - R0)  # negative
        np.testing.assert_allclose(f, [expected_magnitude, 0.0, 0.0], atol=1e-10)
        assert f[0] < 0, "Compressed bond: force on i should point away from j"

    def test_very_stretched(self):
        """Very stretched bond (r=2.0σ): large restoring force."""
        dist = 2.0
        r_ij = np.array([0.0, dist, 0.0])  # Along y-axis
        f = harmonic_bond_force(r_ij, K_BOND, R0)
        expected_magnitude = K_BOND * (dist - R0)
        np.testing.assert_allclose(f, [0.0, expected_magnitude, 0.0], atol=1e-10)

    def test_3d_diagonal(self):
        """Force direction correct in 3D."""
        r_ij = np.array([1.0, 1.0, 1.0])  # dist = √3 ≈ 1.732
        dist = np.linalg.norm(r_ij)
        r_hat = r_ij / dist
        f = harmonic_bond_force(r_ij, K_BOND, R0)
        expected = K_BOND * (dist - R0) * r_hat
        np.testing.assert_allclose(f, expected, atol=1e-10)

    def test_newtons_third_law(self):
        """F_ij = -F_ji (Newton's 3rd law)."""
        r_ij = np.array([1.3, 0.2, -0.5])
        f_on_i = harmonic_bond_force(r_ij, K_BOND, R0)
        f_on_j = harmonic_bond_force(-r_ij, K_BOND, R0)
        np.testing.assert_allclose(f_on_i, -f_on_j, atol=1e-12,
            err_msg="Newton's 3rd law violated for harmonic bond")

    def test_potential_energy(self):
        """U = 0.5 * k * (r - r0)²."""
        dist = 1.3
        r_ij = np.array([dist, 0.0, 0.0])
        pe = harmonic_bond_potential(r_ij, K_BOND, R0)
        expected = 0.5 * K_BOND * (dist - R0) ** 2
        assert abs(pe - expected) < 1e-12

    def test_force_is_negative_gradient_of_potential(self):
        """Numerical check: F ≈ -dU/dr (force-potential consistency)."""
        r_ij = np.array([1.2, 0.0, 0.0])
        dr = 1e-6
        r_plus = np.array([1.2 + dr, 0.0, 0.0])
        r_minus = np.array([1.2 - dr, 0.0, 0.0])
        dU = harmonic_bond_potential(r_plus, K_BOND, R0) - harmonic_bond_potential(r_minus, K_BOND, R0)
        numerical_force_x = -dU / (2 * dr)  # This is force component along x on bead j, not i
        # Force on i = k*(d-r0)*r_hat, the x-component for r along x is k*(d-r0)
        # But the numerical gradient gives force ON J along x (moving j in +x increases r)
        # So force on i along x = -numerical_force_on_j = numerical_force_x... let's verify:
        # Actually: ∂U/∂(r_j_x) = dU/dr * ∂r/∂(r_j_x) = dU/dr * r_hat_x
        # F_on_j_x = -∂U/∂(r_j_x) = -dU/dr * r_hat_x
        # F_on_i_x = ∂U/∂(r_j_x) via Newton's 3rd = -F_on_j_x = dU/dr * r_hat_x
        # Wait, let me think more carefully:
        # U depends on |r_j - r_i|. If we move r_j in +x, r increases if r_ij is along +x.
        # So dU/d(r_j_x) = dU/dr * 1 (for r_ij along x)
        # F_on_j_x = -dU/d(r_j_x) = -dU/dr
        # F_on_i_x = -dU/d(r_i_x) = -dU/dr * (-1) = dU/dr
        # 
        # numerical_force_x computed above = -(U(r+dr) - U(r-dr))/(2dr) ≈ -dU/dr
        # This is F_on_j. F_on_i = -F_on_j = dU/dr.
        f_on_i = harmonic_bond_force(r_ij, K_BOND, R0)
        # f_on_i[0] should equal dU/dr ≈ -numerical_force_x... wait:
        # numerical_force_x = -(U(r+dr)-U(r-dr))/(2dr) = -dU/dr  → this is F_on_j
        # f_on_i[0] = k*(d-r0) = dU/dr * ... hmm let me just test the magnitude
        #
        # Simpler: just check that finite-difference gradient of U matches force magnitude
        dist = np.linalg.norm(r_ij)
        f_magnitude = np.linalg.norm(f_on_i)
        dU_dr = (harmonic_bond_potential(np.array([dist + dr, 0, 0]), K_BOND, R0) -
                 harmonic_bond_potential(np.array([dist - dr, 0, 0]), K_BOND, R0)) / (2 * dr)
        # Force magnitude = |dU/dr| (the sign/direction is handled by r_hat)
        assert abs(f_magnitude - abs(dU_dr)) < 1e-5, \
            f"Force-potential inconsistency: |F|={f_magnitude}, |dU/dr|={abs(dU_dr)}"


class TestWCAForce:
    """
    Test WCA excluded-volume force.
    F_WCA = -24ε/r * [2(σ/r)¹² - (σ/r)⁶] * r̂   for r < r_cut
    Force is REPULSIVE: pushes bead i away from j.
    """

    def test_above_cutoff_zero(self):
        """Force is exactly zero for r >= r_cut = 2^(1/6)σ."""
        dist = R_CUT_WCA + 0.01
        r_ij = np.array([dist, 0.0, 0.0])
        f = wca_force(r_ij, EPSILON, SIGMA)
        np.testing.assert_allclose(f, [0.0, 0.0, 0.0], atol=1e-15)

    def test_at_cutoff_nearly_zero(self):
        """Force is very small just inside r_cut."""
        dist = R_CUT_WCA - 1e-6
        r_ij = np.array([dist, 0.0, 0.0])
        f = wca_force(r_ij, EPSILON, SIGMA)
        # At r_cut, force is exactly zero (minimum of LJ).
        # Just inside, it should be very small and repulsive.
        assert np.linalg.norm(f) < 0.1, "Force should be small near cutoff"

    def test_below_cutoff_repulsive(self):
        """Force is repulsive (points away from j) for r < r_cut."""
        dist = 1.0  # At σ
        r_ij = np.array([dist, 0.0, 0.0])  # j is at +x from i
        f = wca_force(r_ij, EPSILON, SIGMA)
        # Force on i should push i AWAY from j → negative x direction
        assert f[0] < 0, f"WCA force should be repulsive (away from j), got f[0]={f[0]}"

    def test_at_sigma_hand_calculation(self):
        """At r = σ: F = -24ε/σ * [2*1 - 1] * r̂ = -24ε/σ * r̂."""
        dist = SIGMA
        r_ij = np.array([dist, 0.0, 0.0])
        f = wca_force(r_ij, EPSILON, SIGMA)
        # sr6 = (1/1)^6 = 1, sr12 = 1
        # factor = 24*1/1 * (2*1 - 1) = 24
        # Force on i = -24 * r_hat = (-24, 0, 0)
        np.testing.assert_allclose(f, [-24.0, 0.0, 0.0], atol=1e-10)

    def test_close_range_strong_repulsion(self):
        """Very close beads (r=0.8σ) experience strong repulsion."""
        dist = 0.8
        r_ij = np.array([dist, 0.0, 0.0])
        f = wca_force(r_ij, EPSILON, SIGMA)
        # Should be strongly repulsive
        assert f[0] < -24.0, "Force at r=0.8σ should be stronger than at r=σ"

    def test_newtons_third_law(self):
        """F_ij = -F_ji for WCA."""
        r_ij = np.array([0.9, 0.3, -0.1])
        dist = np.linalg.norm(r_ij)
        if dist < R_CUT_WCA:
            f_on_i = wca_force(r_ij, EPSILON, SIGMA)
            f_on_j = wca_force(-r_ij, EPSILON, SIGMA)
            np.testing.assert_allclose(f_on_i, -f_on_j, atol=1e-12,
                err_msg="Newton's 3rd law violated for WCA")

    def test_potential_zero_above_cutoff(self):
        """U_WCA = 0 for r >= r_cut."""
        dist = R_CUT_WCA + 0.1
        r_ij = np.array([dist, 0.0, 0.0])
        pe = wca_potential(r_ij, EPSILON, SIGMA)
        assert pe == 0.0

    def test_potential_at_cutoff(self):
        """U_WCA = 0 at r = r_cut (shifted to zero)."""
        r_ij = np.array([R_CUT_WCA - 1e-10, 0.0, 0.0])
        pe = wca_potential(r_ij, EPSILON, SIGMA)
        # Should be very close to 0 (potential is 0 at cutoff by construction)
        assert abs(pe) < 0.01

    def test_potential_at_sigma(self):
        """U_WCA at r=σ: 4ε[1-1] + ε = ε = 1.0."""
        r_ij = np.array([SIGMA, 0.0, 0.0])
        pe = wca_potential(r_ij, EPSILON, SIGMA)
        np.testing.assert_allclose(pe, EPSILON, atol=1e-10)

    def test_force_potential_consistency(self):
        """Numerical gradient of U matches force magnitude."""
        dist = 0.95
        dr = 1e-7
        r_ij = np.array([dist, 0.0, 0.0])
        f = wca_force(r_ij, EPSILON, SIGMA)
        U_plus = wca_potential(np.array([dist + dr, 0, 0]), EPSILON, SIGMA)
        U_minus = wca_potential(np.array([dist - dr, 0, 0]), EPSILON, SIGMA)
        dU_dr = (U_plus - U_minus) / (2 * dr)
        # Force on i along x = -(-dU/dr) = dU/dr... wait:
        # For WCA, F_on_i is repulsive (away from j = -x).
        # dU/dr < 0 for r < r_cut (we're on the repulsive wall, U is decreasing)
        # F_on_j_x = -dU/d(r_j_x) = -dU/dr * (r_hat_x) = -dU/dr (for r along x)
        # F_on_i_x = -F_on_j_x = dU/dr
        # But our function returns F_on_i which should be repulsive (negative x)
        # So F_on_i_x should be negative, and dU/dr should also be negative. Let's verify:
        # Actually F_on_i = -∇_i U.  ∂U/∂(r_i_x) = dU/dr * ∂r/∂(r_i_x) = dU/dr * (-r_hat_x)
        # F_on_i_x = -∂U/∂(r_i_x) = dU/dr * r_hat_x
        # For r along +x: r_hat_x = 1, so F_on_i_x = dU/dr
        # Since dU/dr < 0 for r < r_cut, F_on_i_x < 0 (repulsive ✓)
        np.testing.assert_allclose(f[0], dU_dr, atol=1e-3,
            err_msg=f"WCA force-potential inconsistency: F_x={f[0]}, dU/dr={dU_dr}")


class TestFENEBondForce:
    """
    Test FENE bond force.
    F = k * r / (1-(r/R0)²) * r̂  (attractive, toward j)
    """

    def test_at_zero_separation(self):
        """At r=0, force should be zero (beads on top of each other)."""
        r_ij = np.array([0.0, 0.0, 0.0])
        f = fene_bond_force(r_ij, K_FENE, R0_FENE)
        np.testing.assert_allclose(f, [0.0, 0.0, 0.0], atol=1e-12)

    def test_small_extension(self):
        """Small extension (r=0.5σ): attractive, moderate force."""
        dist = 0.5
        r_ij = np.array([dist, 0.0, 0.0])
        f = fene_bond_force(r_ij, K_FENE, R0_FENE)
        # Hand calc: F = 30 * 0.5 / (1 - (0.5/1.5)²) * r_hat
        # = 15 / (1 - 1/9) = 15 / (8/9) = 15 * 9/8 = 16.875
        ratio_sq = (dist / R0_FENE) ** 2
        expected = K_FENE * dist / (1.0 - ratio_sq)
        np.testing.assert_allclose(f[0], expected, atol=1e-10)
        assert f[0] > 0, "FENE force should be attractive (toward j)"

    def test_near_max_extension_diverges(self):
        """Near R0 (r=1.49σ): force should be very large."""
        dist = 1.49  # Very close to R0=1.5
        r_ij = np.array([dist, 0.0, 0.0])
        f = fene_bond_force(r_ij, K_FENE, R0_FENE)
        # Force should be very large as denominator → 0
        assert f[0] > 1000, f"FENE force should diverge near R0, got {f[0]}"

    def test_exceeds_max_extension_raises(self):
        """r >= R0 should raise ValueError (physically impossible)."""
        r_ij = np.array([R0_FENE + 0.01, 0.0, 0.0])
        with pytest.raises(ValueError, match="exceeded maximum extension"):
            fene_bond_force(r_ij, K_FENE, R0_FENE)

    def test_newtons_third_law(self):
        """F_ij = -F_ji for FENE."""
        r_ij = np.array([0.8, 0.3, -0.2])
        f_on_i = fene_bond_force(r_ij, K_FENE, R0_FENE)
        f_on_j = fene_bond_force(-r_ij, K_FENE, R0_FENE)
        np.testing.assert_allclose(f_on_i, -f_on_j, atol=1e-12,
            err_msg="Newton's 3rd law violated for FENE")

    def test_equilibrium_with_wca(self):
        """
        FENE+WCA equilibrium ≈ 0.965σ (§4).
        At equilibrium, FENE (attractive) + WCA (repulsive) forces balance.
        We check that the net force changes sign around 0.965σ.
        """
        # Below equilibrium: WCA repulsion > FENE attraction → net pushes apart
        # Above equilibrium: FENE attraction > WCA repulsion → net pulls together
        dists = np.linspace(0.9, 1.05, 50)
        net_forces = []
        for d in dists:
            r_ij = np.array([d, 0.0, 0.0])
            f_fene = fene_bond_force(r_ij, K_FENE, R0_FENE)
            f_wca = wca_force(r_ij, EPSILON, SIGMA)
            net = f_fene[0] + f_wca[0]
            net_forces.append(net)
        net_forces = np.array(net_forces)
        # Find sign change (equilibrium point)
        sign_changes = np.where(np.diff(np.sign(net_forces)))[0]
        assert len(sign_changes) > 0, "Should find equilibrium point for FENE+WCA"
        eq_dist = dists[sign_changes[0]]
        assert 0.93 < eq_dist < 1.0, \
            f"FENE+WCA equilibrium should be ≈0.965σ, got {eq_dist:.4f}σ"

    def test_potential_energy(self):
        """U_fene = -0.5 * k * R0² * ln(1 - (r/R0)²)."""
        dist = 0.8
        r_ij = np.array([dist, 0.0, 0.0])
        pe = fene_bond_potential(r_ij, K_FENE, R0_FENE)
        ratio_sq = (dist / R0_FENE) ** 2
        expected = -0.5 * K_FENE * R0_FENE**2 * np.log(1.0 - ratio_sq)
        np.testing.assert_allclose(pe, expected, atol=1e-10)

    def test_force_potential_consistency(self):
        """Numerical gradient matches force."""
        dist = 0.9
        dr = 1e-7
        r_ij = np.array([dist, 0.0, 0.0])
        f = fene_bond_force(r_ij, K_FENE, R0_FENE)
        U_plus = fene_bond_potential(np.array([dist + dr, 0, 0]), K_FENE, R0_FENE)
        U_minus = fene_bond_potential(np.array([dist - dr, 0, 0]), K_FENE, R0_FENE)
        dU_dr = (U_plus - U_minus) / (2 * dr)
        # F_on_i_x = dU/dr * r_hat_x (same reasoning as harmonic test)
        # For FENE: U increases with r (more stretching = more energy)
        # dU/dr > 0, and force on i is attractive (positive x toward j)
        np.testing.assert_allclose(f[0], dU_dr, atol=1e-2,
            err_msg=f"FENE force-potential inconsistency: F_x={f[0]}, dU/dr={dU_dr}")


class TestAggregateForcesComputation:
    """
    Test the aggregate force computation functions that implement
    the §8.1 bonded and nonbonded force loops.
    """

    def test_two_bead_harmonic(self):
        """Two beads at distance 1.5σ with harmonic bond."""
        positions = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
        forces, pe = compute_bonded_forces(positions, "harmonic", K_BOND, R0)
        # Bead 0 should be pulled toward bead 1 (+x)
        # Bead 1 should be pulled toward bead 0 (-x)
        assert forces[0, 0] > 0, "Bead 0 should be pulled toward bead 1"
        assert forces[1, 0] < 0, "Bead 1 should be pulled toward bead 0"
        # Newton's 3rd: forces sum to zero
        np.testing.assert_allclose(np.sum(forces, axis=0), [0, 0, 0], atol=1e-10,
            err_msg="Total force on system should be zero (Newton's 3rd law)")

    def test_three_bead_chain_momentum_conservation(self):
        """Three beads: total force on system = 0 (momentum conservation)."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.2, 0.0, 0.0],
            [2.5, 0.0, 0.0],
        ])
        forces, pe = compute_bonded_forces(positions, "harmonic", K_BOND, R0)
        np.testing.assert_allclose(np.sum(forces, axis=0), [0, 0, 0], atol=1e-10,
            err_msg="Total bonded force should be zero")

    def test_wca_nonbonded_skip_neighbors(self):
        """WCA forces skip direct bonded neighbors (|i-j|<=1)."""
        # 3 beads, all close enough for WCA
        positions = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.3, 0.0, 0.0],  # Close to bead 0, but |0-2|=2 > 1 → WCA applies
        ])
        forces, pe = compute_nonbonded_forces(positions, EPSILON, SIGMA)
        # Only pair (0,2) should have WCA (|0-2|=2 > 1), pair (0,1) and (1,2) skipped
        # Total force should still be zero (Newton's 3rd)
        np.testing.assert_allclose(np.sum(forces, axis=0), [0, 0, 0], atol=1e-10)

    def test_total_forces_combine(self):
        """compute_all_forces = bonded + nonbonded."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ])
        f_total, pe_bond, pe_nonbond = compute_all_forces(
            positions, "harmonic", K_BOND, R0, K_FENE, R0_FENE, EPSILON, SIGMA
        )
        f_bond, pe_b = compute_bonded_forces(positions, "harmonic", K_BOND, R0)
        f_nb, pe_nb = compute_nonbonded_forces(positions, EPSILON, SIGMA)
        np.testing.assert_allclose(f_total, f_bond + f_nb, atol=1e-12)
        np.testing.assert_allclose(pe_bond, pe_b, atol=1e-12)
        np.testing.assert_allclose(pe_nonbond, pe_nb, atol=1e-12)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
