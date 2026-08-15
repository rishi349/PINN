# Physics-Informed GNN for Polymer Chain Dynamics

A controlled study of whether, and how, different ways of injecting physical structure affect long-horizon rollout stability for a coarse-grained polymer chain under overdamped Langevin dynamics.

## Project Status
**Phase:** Month 3 (GNN Dataset & Baseline Models)
**Physics Engine:** ✅ Fully Validated (WCA forces clamped, integration stable, Flory exponent matches theory).
**Next Milestone:** Train baseline message-passing GNN to predict next-step displacements.

## Project Overview

This project compares plain and physics-informed Graph Neural Networks (GNNs) for predicting the dynamics of bead-spring polymer chains. The core question: **does incorporating physics constraints (bond-length conservation, excluded-volume) into the GNN loss function improve long-rollout stability compared to a pure data-driven baseline?**

### Models
| # | Model | Purpose |
|---|---|---|
| 0a | Zero-displacement | Naive baseline |
| 0b | Global-statistics random draw | Naive baseline |
| 1 | Baseline message-passing GNN | Fair comparison point |
| 2 | Physics-informed (bond + excluded-volume losses) | Main comparison |
| 3 | EGNN (equivariant) | Architectural equivariance test |
| 4 | Momentum-conserving | Newton's 3rd law enforcement |

### Physics
- **Dynamics:** Overdamped Langevin (Brownian dynamics)
- **Integrator:** Euler–Maruyama (strong order 1.0 for additive noise)
- **Bond potential:** Harmonic (first version) → FENE (production)
- **Excluded volume:** WCA (Weeks-Chandler-Andersen)
- **Chain lengths:** N=30 (main), N=50 (OOD), N=100/200 (scaling study)

## Project Structure

```
simulator/
├── configs/              # YAML config files per experiment
├── src/
│   ├── physics/          # Force laws, integrators
│   ├── simulator/        # HOOMD-blue + NumPy simulators
│   ├── data/             # Trajectory I/O, graph construction
│   ├── models/           # GNN architectures
│   ├── training/         # Training loops, loss functions
│   └── evaluation/       # Metrics, rollout evaluation
├── tests/                # Unit + integration tests
├── data/                 # Raw trajectories, processed graphs, splits
├── models/               # Saved checkpoints by model type
├── logs/                 # Experiment logs
├── figures/              # Generated plots
├── reports/drafts/       # Validation reports, comparison docs
└── scripts/              # CLI entry points
```

## Setup

### Prerequisites
- Miniconda or Anaconda
- Git

### Installation
```bash
# Clone the repository
git clone https://github.com/rishi349/PINN.git
cd PINN

# Create and activate conda environment
conda env create -f environment.yml
conda activate polymer-gnn

# Verify installation
python -c "import hoomd; import torch; import torch_geometric; print('All imports OK')"
```

## Usage

### 1. Generating Simulation Data
Generate raw trajectories using the validated numpy physics engine.

```bash
# Generate a single test trajectory
python scripts/run_simulation.py --config configs/default.yaml

# Generate the full multi-trajectory dataset for GNN training (Strict Train/Val/Test splitting)
python scripts/generate_data.py --config configs/default.yaml --n-trajectories 5 --output data/raw/N30
```

### 2. Validating Physics & Equilibration
Run these to verify that the core physics engine produces physically valid, correctly equilibrated polymer trajectories.

```bash
# Quick validation (runs a short simulation to catch basic numerical blow-ups)
python scripts/validate_simulator.py --config configs/default.yaml

# Full 3.7M step standard diagnostic dashboard (Rg, Ree, Bond dist, PE)
python scripts/diagnose_equilibration.py --output-dir plots/equilibration

# Full 3.7M step advanced physics validation (autocorrelation, force caps, Flory scaling)
python scripts/verify_remaining.py
```

### 3. Training the GNN
Train the baseline message-passing GNN on the generated dataset.

```bash
# Train on the generated dataset
python scripts/train.py --config configs/default.yaml --data-dir data/raw/N30

# Override default epochs or device
python scripts/train.py --config configs/default.yaml --data-dir data/raw/N30 --epochs 200 --device cuda
```

### 4. Evaluating the Model
Evaluate the trained GNN checkpoint via autoregressive rollout against the ground truth physics.

```bash
python scripts/evaluate.py --checkpoint models/saved/best_model.pt --config configs/default.yaml --data-dir data/raw/N30
```

### 5. Running Unit Tests
Validate the fundamental logic (data loading, exact force derivatives, integration algorithms).

```bash
pytest tests/ -v
```

### 6. Visualization
Plot 3D snapshots (Initial, Middle, Final) of a completed trajectory.

```bash
python scripts/visualize_trajectory.py data/raw/N30/traj_0.json plots/traj_0_vis.png
```

## Reproducibility

Every trajectory and training run records:
- Software versions (HOOMD, PyTorch, PyG, Python, NumPy)
- All random seeds (simulation, split, model init, dataloader shuffle)
- Config file hash and git commit
- SHA256 checksums of data files

## References

- Kremer & Grest bead-spring model
- Satorras et al., EGNN (arXiv:2102.09844)
- Sharma & Fink, Dynami-CAL GraphNet (Nature Comms 17:1045, 2026)
- Sanchez-Gonzalez et al., Graph Network Simulator (arXiv:2002.09405)
