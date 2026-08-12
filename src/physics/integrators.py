"""
Euler–Maruyama integrator for overdamped Langevin dynamics.

From the execution plan §3 and §8.1:

Primary equation: γ dr/dt = F(r) + ξ(t)

Euler–Maruyama update for overdamped (Brownian) dynamics:
    noise = sqrt(2 * kBT * dt / γ) * N(0, 1)
    r[i] = r[i] + (dt / γ) * F[i] + noise

Key properties (§3):
- Noise is ADDITIVE (does not depend on position), so strong convergence
  order is 1.0 (better than the generic 0.5).
- Noise scales with √dt, NOT dt — this is fundamental to stochastic
  calculus and must not be "fixed" even though it looks like a bug.
- Temperature in the overdamped limit must be estimated from displacement
  statistics, NOT from kinetic energy (velocities are undefined).

Underdamped (full Langevin) variant also provided (§8.1):
    noise = sqrt(2 * γ * kBT * dt) * N(0, 1)
    v[i] = v[i] + dt * (F[i] - γ * v[i]) / m + noise / m
    r[i] = r[i] + dt * v[i]
"""

import numpy as np
from typing import Tuple, Optional


def euler_maruyama_overdamped_step(
    positions: np.ndarray,
    forces: np.ndarray,
    dt: float,
    gamma: float,
    kBT: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform one Euler–Maruyama step for overdamped Langevin dynamics.

    From §8.1 pseudocode:
        noise = sqrt(2*kB*T*dt/gamma) * Normal(0,1)
        r[i] = r[i] + (dt/gamma) * F[i] + noise

    Parameters
    ----------
    positions : np.ndarray, shape (N, dim)
        Current bead positions.
    forces : np.ndarray, shape (N, dim)
        Current forces on each bead.
    dt : float
        Timestep size in τ.
    gamma : float
        Friction coefficient in m/τ.
    kBT : float
        Thermal energy in ε.
    rng : np.random.Generator
        NumPy random number generator for reproducibility.

    Returns
    -------
    new_positions : np.ndarray, shape (N, dim)
        Updated bead positions.
    noise_realization : np.ndarray, shape (N, dim)
        The noise vector that was applied (stored for diagnostics / temperature estimation).
    """
    # §8.1: noise = sqrt(2*kBT*dt/gamma) * N(0,1)
    noise_amplitude = np.sqrt(2.0 * kBT * dt / gamma)
    noise = noise_amplitude * rng.standard_normal(positions.shape)

    # §8.1: r[i] = r[i] + (dt/gamma) * F[i] + noise
    drift = (dt / gamma) * forces
    displacement = drift + noise

    # Safety: cap maximum per-bead displacement to prevent rare
    # catastrophic single-step launches. Without this, WCA close
    # encounters can produce forces ~10^14, yielding displacements
    # of ~10^11 σ in a single step. A cap of 0.5σ is conservative
    # and preserves correct dynamics for all normal configurations.
    MAX_DISP = 0.5  # σ
    disp_magnitudes = np.linalg.norm(displacement, axis=1, keepdims=True)
    clipping_mask = disp_magnitudes > MAX_DISP
    if np.any(clipping_mask):
        safe_magnitudes = np.maximum(disp_magnitudes, 1e-30)
        scale = np.where(clipping_mask, MAX_DISP / safe_magnitudes, 1.0)
        displacement = displacement * scale

    new_positions = positions + displacement

    return new_positions, noise


def euler_maruyama_underdamped_step(
    positions: np.ndarray,
    velocities: np.ndarray,
    forces: np.ndarray,
    dt: float,
    gamma: float,
    kBT: float,
    mass: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform one Euler–Maruyama step for underdamped (full) Langevin dynamics.

    From §8.1 pseudocode:
        noise = sqrt(2*gamma*kBT*dt) * Normal(0,1)
        v[i] = v[i] + dt*(F[i] - gamma*v[i])/m + noise/m
        r[i] = r[i] + dt*v[i]

    Parameters
    ----------
    positions : np.ndarray, shape (N, dim)
        Current bead positions.
    velocities : np.ndarray, shape (N, dim)
        Current bead velocities.
    forces : np.ndarray, shape (N, dim)
        Current forces on each bead.
    dt : float
        Timestep size.
    gamma : float
        Friction coefficient.
    kBT : float
        Thermal energy.
    mass : float
        Bead mass.
    rng : np.random.Generator
        NumPy random number generator.

    Returns
    -------
    new_positions : np.ndarray, shape (N, dim)
        Updated bead positions.
    new_velocities : np.ndarray, shape (N, dim)
        Updated bead velocities.
    noise_realization : np.ndarray, shape (N, dim)
        The noise vector that was applied.
    """
    # §8.1: noise = sqrt(2*gamma*kBT*dt) * N(0,1)
    noise_amplitude = np.sqrt(2.0 * gamma * kBT * dt)
    noise = noise_amplitude * rng.standard_normal(positions.shape)

    # §8.1: v[i] = v[i] + dt*(F[i] - gamma*v[i])/m + noise/m
    new_velocities = velocities + dt * (forces - gamma * velocities) / mass + noise / mass

    # §8.1: r[i] = r[i] + dt*v[i]
    new_positions = positions + dt * new_velocities

    return new_positions, new_velocities, noise
