#!/usr/bin/env python
"""
3D trajectory visualization script.

Plots 3D snapshots of polymer chain configurations (Initial, Middle, Final)
from trajectory JSON files.
"""

import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np


def visualize_snapshots(json_path: str, output_path: str):
    print(f"Loading {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames = data["frames"]
    n_frames = len(frames)

    if n_frames == 0:
        print("No frames found in the trajectory.")
        return

    # Select first, middle, and last frames
    idx_first = 0
    idx_mid = n_frames // 2
    idx_last = n_frames - 1

    selected_indices = [idx_first, idx_mid, idx_last]
    titles = ["Initial State", "Middle State", "Final State"]

    fig = plt.figure(figsize=(15, 5))

    for i, idx in enumerate(selected_indices):
        frame = frames[idx]
        positions = np.array(frame["positions"])

        ax = fig.add_subplot(1, 3, i + 1, projection="3d")

        # Plot beads
        ax.scatter(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            c="blue",
            s=50,
            alpha=0.8,
            edgecolors="black",
            zorder=5,
        )

        # Plot bonds
        for b_i in range(len(positions) - 1):
            x = [positions[b_i, 0], positions[b_i + 1, 0]]
            y = [positions[b_i, 1], positions[b_i + 1, 1]]
            z = [positions[b_i, 2], positions[b_i + 1, 2]]
            ax.plot(x, y, z, c="gray", linewidth=2, zorder=1)

        # Highlight head (bead 0) and tail (bead N-1)
        ax.scatter(
            positions[0, 0],
            positions[0, 1],
            positions[0, 2],
            c="green",
            s=100,
            label="Head (Bead 0)",
            zorder=6,
        )
        ax.scatter(
            positions[-1, 0],
            positions[-1, 1],
            positions[-1, 2],
            c="red",
            s=100,
            label=f"Tail (Bead {len(positions)-1})",
            zorder=6,
        )

        ax.set_title(
            f"{titles[i]} (Step {frame['timestep_index']})\nRg = {frame['radius_of_gyration']:.2f}"
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Center camera bounding box around Center of Mass
        com = np.array(frame["center_of_mass"])
        ax.set_xlim(com[0] - 8, com[0] + 8)
        ax.set_ylim(com[1] - 8, com[1] + 8)
        ax.set_zlim(com[2] - 8, com[2] + 8)

        if i == 0:
            ax.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved snapshots to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize polymer chain 3D trajectory snapshots"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/trajectory_0000.json",
        help="Path to trajectory JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="figures/snapshots.png",
        help="Path to output PNG image",
    )
    args = parser.parse_args()

    visualize_snapshots(args.input, args.output)


if __name__ == "__main__":
    main()
