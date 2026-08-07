# Contributing Guidelines

## Branch Naming

- `main` — stable, validated code only
- `feature/<name>` — new features (e.g., `feature/wca-force`)
- `fix/<name>` — bug fixes (e.g., `fix/bond-force-sign`)
- `experiment/<name>` — experimental runs (e.g., `experiment/lr-sweep`)

## When to Open a PR vs. Push Directly

### Always open a PR (require review) for:
- **Anything touching physics code** (force calculations, loss terms, evaluation metrics)
  - These produce plausible-looking wrong numbers — bugs here are far more expensive to catch later than a crash
- Changes to the dataset schema (§5)
- Changes to the config format
- Changes to evaluation/metrics code

### OK to push directly:
- Documentation updates
- Logging/visualization improvements
- Test additions (that don't change existing tests)
- Script changes that don't affect core logic

## Code Review Rules (from §7.3)

1. **Physics code gets reviewed before merge, always.** No exceptions. This includes:
   - Force implementations (`src/physics/forces.py`)
   - Integrator implementations (`src/physics/integrators.py`)
   - Physics-informed loss terms
   - Evaluation metrics (especially R_g, MSD, temperature estimation)

2. **The reviewer should verify:**
   - Formulas match the plan document (§4, §8)
   - Signs are correct (Newton's 3rd law: F_ij = -F_ji)
   - Units are consistent (reduced units throughout)
   - Edge cases handled (FENE near max extension, WCA at cutoff boundary)

## Commit Messages

Use conventional commit format:
```
type(scope): description

feat(physics): implement WCA excluded-volume force
fix(simulator): correct noise scaling in overdamped integrator
test(forces): add unit tests for harmonic bond at 4 distances
docs(readme): update setup instructions
```

## Testing

- Run `pytest tests/ -v` before any push
- Force unit tests must pass before simulator work begins
- Simulator validation must pass before dataset work begins
- **Do not skip gates** — see §9 in the execution plan
