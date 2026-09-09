#!/usr/bin/env python3
"""Generate simulation trajectories for all chain-length arms (Month 4).

Runs generate_data.py for each arm in the OOD / scaling study:
  - N=30  : 150 trajectories  (main training arm)
  - N=50  :  30 trajectories  (OOD arm, Month 9)
  - N=100 :  20 trajectories  (scaling arm, Month 10)
  - N=200 :  10 trajectories  (scaling arm, Month 10)

Usage
-----
    # Generate all arms (safe to interrupt and re-run — skips existing files):
    python scripts/generate_all_arms.py --config configs/default.yaml

    # Dry run (print what would be generated, don't run):
    python scripts/generate_all_arms.py --dry-run

    # Generate only specific arms:
    python scripts/generate_all_arms.py --arms 30 50

    # Override trajectory counts:
    python scripts/generate_all_arms.py --n-trajs-30 10 --n-trajs-50 5

Notes
-----
- Each trajectory at N=30 takes ~15–20 minutes on a modern CPU.
- This script is designed to run unattended overnight / in the background.
- It checks existing files and skips completed trajectories automatically.
- Output is saved to data/raw/N{N}/ directories.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


# ---------------------------------------------------------------------------
# Arm configuration
# ---------------------------------------------------------------------------

DEFAULT_ARMS = {
    30:  {"n_trajs": 150, "label": "main training arm"},
    50:  {"n_trajs": 30,  "label": "OOD arm (Month 9)"},
    100: {"n_trajs": 20,  "label": "scaling arm (Month 10)"},
    200: {"n_trajs": 10,  "label": "scaling arm (Month 10)"},
}

# Rough wall-time estimate per trajectory at N=30 on a modern CPU (seconds)
SECONDS_PER_TRAJ_N30 = 900   # ~15 min; scales roughly as N
TRAJ_TIME_SCALE = 1.2        # N scales slightly super-linearly due to NB pairs


def estimate_wall_time(N: int, n_trajs: int) -> str:
    """Return human-readable wall time estimate for an arm."""
    t_per = SECONDS_PER_TRAJ_N30 * (N / 30) ** TRAJ_TIME_SCALE
    total_sec = t_per * n_trajs
    hours = int(total_sec // 3600)
    minutes = int((total_sec % 3600) // 60)
    return f"~{hours}h {minutes}m"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_existing_trajs(output_dir: Path) -> int:
    """Count trajectory files already present in the directory."""
    npz = list(output_dir.glob("*.npz"))
    jsn = list(output_dir.glob("*.json"))
    return len(npz) + len(jsn)
