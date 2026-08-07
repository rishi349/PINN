"""
Trajectory I/O and metadata management.

Handles saving/loading trajectory data with the full §5 and §5.1 schema,
including SHA256 checksums for data provenance tracking.

Supports both JSON (human-readable, for pilot data) and NumPy npz
(compact, for production data) formats.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def save_trajectory_json(
    trajectory: Dict[str, Any], output_dir: str
) -> str:
    """
    Save a trajectory as JSON with SHA256 checksum (§5.1).

    Parameters
    ----------
    trajectory : dict
        Trajectory data with "metadata" and "frames" keys.
    output_dir : str
        Output directory.

    Returns
    -------
    filepath : str
        Path to saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    traj_id = trajectory["metadata"]["traj_id"]
    filepath = os.path.join(output_dir, f"trajectory_{traj_id:04d}.json")

    with open(filepath, "w") as f:
        json.dump(trajectory, f, indent=2)

    # §5.1: SHA256 checksum
    checksum = compute_file_checksum(filepath)
    trajectory["metadata"]["sha256_checksum"] = checksum

    # Re-save with checksum
    with open(filepath, "w") as f:
        json.dump(trajectory, f, indent=2)

    return filepath


def load_trajectory_json(filepath: str) -> Dict[str, Any]:
    """Load a trajectory from JSON."""
    with open(filepath, "r") as f:
        return json.load(f)


def save_trajectory_npz(
    trajectory: Dict[str, Any], output_dir: str
) -> str:
    """
    Save a trajectory as compressed NumPy npz (for production data).

    Parameters
    ----------
    trajectory : dict
        Trajectory data.
    output_dir : str
        Output directory.

    Returns
    -------
    filepath : str
        Path to saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    traj_id = trajectory["metadata"]["traj_id"]
    filepath = os.path.join(output_dir, f"trajectory_{traj_id:04d}.npz")

    frames = trajectory["frames"]
    n_frames = len(frames)

    if n_frames == 0:
        return filepath

    # Extract arrays from frames
    N = len(frames[0]["positions"])
    dim = len(frames[0]["positions"][0])

    positions = np.array([f["positions"] for f in frames])  # (n_frames, N, dim)
    forces = np.array([f["forces"] for f in frames])        # (n_frames, N, dim)
    timesteps = np.array([f["timestep_index"] for f in frames])
    pe_bond = np.array([f["potential_energy_bond"] for f in frames])
    pe_nonbond = np.array([f["potential_energy_nonbond"] for f in frames])
    rg = np.array([f["radius_of_gyration"] for f in frames])
    ree = np.array([f["end_to_end_distance"] for f in frames])
    mean_bl = np.array([f["mean_bond_length"] for f in frames])

    np.savez_compressed(
        filepath,
        positions=positions,
        forces=forces,
        timesteps=timesteps,
        pe_bond=pe_bond,
        pe_nonbond=pe_nonbond,
        radius_of_gyration=rg,
        end_to_end_distance=ree,
        mean_bond_length=mean_bl,
        metadata=json.dumps(trajectory["metadata"]),
    )

    # §5.1: checksum
    checksum = compute_file_checksum(filepath)
    trajectory["metadata"]["sha256_checksum"] = checksum

    return filepath


def load_trajectory_npz(filepath: str) -> Dict[str, Any]:
    """Load a trajectory from NumPy npz format."""
    data = np.load(filepath, allow_pickle=True)
    metadata = json.loads(str(data["metadata"]))
    return {
        "metadata": metadata,
        "positions": data["positions"],
        "forces": data["forces"],
        "timesteps": data["timesteps"],
        "pe_bond": data["pe_bond"],
        "pe_nonbond": data["pe_nonbond"],
        "radius_of_gyration": data["radius_of_gyration"],
        "end_to_end_distance": data["end_to_end_distance"],
        "mean_bond_length": data["mean_bond_length"],
    }


def compute_file_checksum(filepath: str) -> str:
    """
    Compute SHA256 checksum of a file (§5.1 data provenance).

    Parameters
    ----------
    filepath : str
        Path to the file.

    Returns
    -------
    str
        Hex-encoded SHA256 checksum.
    """
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def verify_checksum(filepath: str, expected_checksum: str) -> bool:
    """
    Verify file integrity using SHA256 checksum (§13.2).

    §13.2: "Confirm the SHA256 checksums recorded in §5.1 match the
    actual files on disk before starting any expensive training run"
    """
    actual = compute_file_checksum(filepath)
    return actual == expected_checksum
