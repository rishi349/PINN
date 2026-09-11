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


def run_arm(
    N: int,
    n_trajs: int,
    config_path: str,
    output_root: str = "data/raw",
    traj_id_start: int = 0,
    fmt: str = "npz",
    quiet: bool = False,
) -> bool:
    """Run generate_data.py for one arm. Returns True if successful."""
    cmd = [
        sys.executable,
        str(project_root / "scripts" / "generate_data.py"),
        "--config", config_path,
        "--chain-length", str(N),
        "--n-trajectories", str(n_trajs),
        "--output", output_root,
        "--traj-id-start", str(traj_id_start),
        "--format", fmt,
    ]
    if quiet:
        cmd.append("--quiet")

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate all chain-length arms for the scaling / OOD study"
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--arms", type=int, nargs="+", default=None,
        help="Which chain lengths to generate (default: all: 30 50 100 200)",
    )
    parser.add_argument(
        "--output", type=str, default="data/raw",
        help="Root output directory (default: data/raw)",
    )
    parser.add_argument(
        "--format", type=str, choices=["npz", "json"], default="npz",
        help="Output format (default: npz)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print plan without running anything",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-step simulator output",
    )
    # Per-arm count overrides
    parser.add_argument("--n-trajs-30",  type=int, default=None)
    parser.add_argument("--n-trajs-50",  type=int, default=None)
    parser.add_argument("--n-trajs-100", type=int, default=None)
    parser.add_argument("--n-trajs-200", type=int, default=None)

    args = parser.parse_args()

    # Build arm list
    arms = DEFAULT_ARMS.copy()
    if args.n_trajs_30  is not None: arms[30]["n_trajs"]  = args.n_trajs_30
    if args.n_trajs_50  is not None: arms[50]["n_trajs"]  = args.n_trajs_50
    if args.n_trajs_100 is not None: arms[100]["n_trajs"] = args.n_trajs_100
    if args.n_trajs_200 is not None: arms[200]["n_trajs"] = args.n_trajs_200

    selected_Ns = args.arms if args.arms is not None else sorted(arms.keys())

    # Print plan
    print("=" * 65)
    print("MULTI-ARM DATA GENERATION PLAN")
    print("=" * 65)
    total_trajs = 0
    for N in selected_Ns:
        cfg = arms[N]
        out_dir = Path(args.output) / f"N{N}"
        existing = count_existing_trajs(out_dir) if out_dir.exists() else 0
        remaining = max(0, cfg["n_trajs"] - existing)
        wall = estimate_wall_time(N, remaining) if remaining > 0 else "DONE"
        print(
            f"  N={N:3d} ({cfg['label']}):  "
            f"{cfg['n_trajs']} total, {existing} done, "
            f"{remaining} remaining  [{wall}]"
        )
        total_trajs += remaining
    print(f"\nTotal new trajectories to generate: {total_trajs}")
    print("=" * 65)

    if args.dry_run:
        print("\nDry run — exiting without generating.")
        return

    if total_trajs == 0:
        print("\nAll trajectories already exist. Nothing to do.")
        return

    # Generate each arm
    overall_start = time.time()
    failed_arms = []

    for N in selected_Ns:
        cfg = arms[N]
        out_dir = Path(args.output) / f"N{N}"
        out_dir.mkdir(parents=True, exist_ok=True)

        existing = count_existing_trajs(out_dir)
        remaining = max(0, cfg["n_trajs"] - existing)

        if remaining == 0:
            print(f"\n[N={N}] All {cfg['n_trajs']} trajectories already exist. Skipping.")
            continue

        print(f"\n{'=' * 65}")
        print(f"[N={N}] Generating {remaining} trajectories (starting from id={existing})")
        print(f"       Est. wall time: {estimate_wall_time(N, remaining)}")
        print(f"{'=' * 65}")

        arm_start = time.time()
        success = run_arm(
            N=N,
            n_trajs=remaining,
            config_path=args.config,
            output_root=args.output,
            traj_id_start=existing,
            fmt=args.format,
            quiet=args.quiet,
        )
        arm_elapsed = time.time() - arm_start

        if success:
            print(f"[N={N}] ✅ Done in {arm_elapsed/60:.1f} min")
        else:
            print(f"[N={N}] ❌ FAILED after {arm_elapsed/60:.1f} min")
            failed_arms.append(N)

    total_elapsed = time.time() - overall_start
    print(f"\n{'=' * 65}")
    print(f"All arms complete. Total time: {total_elapsed/3600:.2f}h")
    if failed_arms:
        print(f"FAILED arms: N = {failed_arms}")
        sys.exit(1)
    else:
        print("✅ All arms succeeded.")
    print("=" * 65)


if __name__ == "__main__":
    main()
