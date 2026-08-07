# Project Contract — Locked Decisions

**Date:** 2026-08-07
**Status:** LOCKED — do not change without explicit team consensus + supervisor approval

---

## 1. Dynamics Regime
- **Overdamped Langevin (Brownian dynamics)**: LOCKED
- Primary equation: `γ dr/dt = F(r) + ξ(t)`
- Inertial term `m d²r/dt²` is dropped
- Valid for coarse-grained polymer beads in implicit solvent
- Would break down for very light/fast beads or short-timescale ballistic motion

## 2. Integrator
- **Euler–Maruyama**: LOCKED
- Strong convergence order **1.0** (not 0.5) because noise is additive (position-independent)
- Update rule: `r_new = r + (dt/γ) * F + √(2*k_B*T*dt/γ) * N(0,1)`
- Noise scales with `√dt`, not `dt`

## 3. Bond Potential (First Version)
- **Harmonic bond**: LOCKED for first version
- `U_bond = 0.5 * k_bond * (|r_ij| - r₀)²`
- `F_bond = -k_bond * (|r_ij| - r₀) * r̂_ij`
- Parameters: `k_bond = 100 ε/σ²`, `r₀ = 1.0 σ`
- Will transition to FENE for production (Month 3+)

## 4. Excluded Volume
- **WCA (Weeks-Chandler-Andersen)**: LOCKED
- Repulsive-only Lennard-Jones, cut and shifted at `r_cut = 2^(1/6) * σ ≈ 1.122σ`
- `U_WCA = 4ε[(σ/r)¹² - (σ/r)⁶] + ε` for r < r_cut, 0 otherwise
- `F_WCA = 24ε/r * [2(σ/r)¹² - (σ/r)⁶] * r̂` for r < r_cut, 0 otherwise
- Parameters: `ε = 1.0`, `σ = 1.0`

## 5. Prediction Target
- **Next-step displacement**: LOCKED
- `target = r(t+dt) - r(t)`
- Not absolute positions (displacement is translation-invariant)

## 6. Data Split
- **70% train / 15% val / 15% test**: LOCKED
- Split by **trajectory**, never by frame
- Zero leakage verified by automated assertion

## 7. Chain Lengths
- N=30 (main training/evaluation)
- N=50 (OOD test)
- N=100, 200 (scaling study, Months 9-10)

## 8. Reduced Units
All simulations use reduced (dimensionless) units:
- Energy: ε = 1.0
- Length: σ = 1.0
- Mass: m = 1.0
- Time: τ = σ√(m/ε) = 1.0
- Temperature: k_BT = 1.0 ε

## 9. What "Physically Reasonable" Looks Like (Validation Targets)
- **Harmonic bond length:** mean ≈ r₀ = 1.0σ, narrow distribution
- **FENE+WCA bond length (future):** mean ≈ 0.965σ
- **Temperature:** stable near k_BT = 1.0 (estimated from displacement statistics, NOT raw kinetic energy in overdamped limit)
- **R_g scaling:** R_g ~ N^ν with ν ≈ 0.588 (good solvent / self-avoiding walk)
- **No blow-ups:** no NaN/Inf positions, no bond lengths diverging
- **No persistent overlaps:** fraction of frames with bead-bead distance < WCA cutoff ≈ 0 after equilibration
