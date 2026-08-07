# Physics-Informed GNN for Polymer Chain Dynamics

A controlled study of whether, and how, different ways of injecting physical structure affect long-horizon rollout stability for a coarse-grained polymer chain under overdamped Langevin dynamics.

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

### Run a simulation
```bash
python scripts/run_simulation.py --config configs/default.yaml
```

### Run tests
```bash
pytest tests/ -v
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
