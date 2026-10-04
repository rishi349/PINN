#!/usr/bin/env python3
"""
Visualize a polymer chain simulation trajectory as an animated GIF.
"""

import argparse
import sys
import os
from pathlib import Path

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.simulator.numpy_simulator import NumpySimulator, load_config

def main():
    parser = argparse.ArgumentParser(description="Animate polymer simulation")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output", type=str, required=True, help="Output GIF path")
    parser.add_argument("--trajectory", type=str, default=None, help="Path to trajectory JSON file to animate (overrides simulation)")
    parser.add_argument("--steps", type=int, default=1000, help="Steps to simulate")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--beads", type=int, default=None, help="Number of beads (overrides config)")
    parser.add_argument("--max_frames", type=int, default=200, help="Maximum number of frames for GIF")
    parser.add_argument("--continuous", action="store_true", help="Take the first max_frames continuously without downsampling")
    args = parser.parse_args()

    if args.trajectory:
        print(f"Loading trajectory from {args.trajectory}...")
        with open(args.trajectory, "r", encoding="utf-8") as f:
            data = json.load(f)
        frames = data.get("frames", [])
        if len(frames) > args.max_frames:
            if args.continuous:
                print(f"Taking the first {args.max_frames} continuous frames...")
                frames = frames[:args.max_frames]
            else:
                step = max(1, len(frames) // args.max_frames)
                print(f"Downsampling {len(frames)} frames by taking every {step}th frame...")
                frames = frames[::step]
    else:
        # Load config and override steps
        config = load_config(args.config)
        config["simulation"]["T_steps"] = args.steps
        config["simulation"]["n_burnin"] = 0  # Crucial: no burn-in so we get frames immediately
        config["simulation"]["save_every"] = max(1, args.steps // 200) # Save ~200 frames for animation
        
        if args.beads is not None:
            config["chain"]["N"] = args.beads
        
        print(f"Running simulation for {args.steps} steps...")
        sim = NumpySimulator(config)
        trajectory = sim.run(traj_id=999, seed=123, verbose=True)
        
        # Extract positions (Frames, N, 3)
        frames = trajectory["frames"] if "frames" in trajectory else []
        
    if not frames:
        print("No frames found!")
        sys.exit(1)
        
    positions = np.array([f["positions"] for f in frames])
    n_frames = len(positions)
    n_beads = positions.shape[1]
    print(f"Generated {n_frames} frames of {n_beads} beads. Creating animation...")

    # Set up the figure and 3D axis
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Find bounding box for plotting
    all_pos = positions.reshape(-1, 3)
    max_range = np.array([all_pos[:,0].max()-all_pos[:,0].min(), 
                          all_pos[:,1].max()-all_pos[:,1].min(), 
                          all_pos[:,2].max()-all_pos[:,2].min()]).max() / 2.0
                          
    mid_x = (all_pos[:,0].max()+all_pos[:,0].min()) * 0.5
    mid_y = (all_pos[:,1].max()+all_pos[:,1].min()) * 0.5
    mid_z = (all_pos[:,2].max()+all_pos[:,2].min()) * 0.5

    # Initialize plot elements
    line, = ax.plot([], [], [], 'o-', lw=2, markersize=8, color='#2c3e50', 
                    markerfacecolor='#3498db', markeredgecolor='white')
    
    def init():
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.set_title("Polymer Chain Simulation")
        ax.set_axis_off() # clean look
        return line,

    def update(frame):
        pos = positions[frame]
        line.set_data(pos[:, 0], pos[:, 1])
        line.set_3d_properties(pos[:, 2])
        # Slowly rotate camera
        ax.view_init(elev=20., azim=frame * 360 / n_frames)
        return line,

    ani = animation.FuncAnimation(
        fig, update, frames=n_frames, init_func=init, blit=False, interval=1000/args.fps
    )
    
    print(f"Saving to {args.output}...")
    # Save as GIF using Pillow writer
    ani.save(args.output, writer='pillow', fps=args.fps)
    print("Done!")

if __name__ == "__main__":
    main()
