# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research experiment using **symbolic regression (SR)** to discover mathematical equations governing **catastrophic forgetting** in neural networks. The project tests multiple hypotheses about forgetting dynamics through four experimental phases:

1. **Phase 1**: Linear network baseline (validates methodology against known analytical results)
2. **Phase 2**: Nonlinear networks with lazy-rich regime detection
3. **Phase 3**: Universal subspace analysis (tests if deviation from learned subspace predicts forgetting)
4. **Phase 4**: Trajectory analysis (tests if path through weight space predicts forgetting)

**Key finding**: Task similarity dominates (r = -0.91), explaining 83% of forgetting variance. The trajectory hypothesis was not supported.

## Development Commands

```bash
# Install dependencies (requires Julia for PySR)
pip install -r requirements.txt

# Run Phase 1 experiment (full sweep: ~30 min)
python scripts/run_phase1.py

# Run Phase 1 quick test (~2 min)
python scripts/run_phase1.py --quick

# Run Phase 1 with symbolic regression only (skip data generation)
python scripts/run_phase1.py --sr-only

# Run Phase 3 experiment
python scripts/run_phase3.py

# Run Phase 4 experiment
python scripts/run_phase4.py

# Generate visualizations from existing results
python scripts/generate_visualizations.py

# Use uv to run Python (per user preference)
uv run python scripts/run_phase1.py
```

**Note**: PySR requires Julia installation. Install Julia first, then `pip install pysr`.

## Architecture

### Source Modules (`src/`)

| Module | Purpose |
|--------|---------|
| `models.py` | Linear teacher-student networks, task similarity generation, training loops |
| `nonlinear_models.py` | Two-layer nonlinear networks with FLR/NTK tracking, lazy-rich classification |
| `universal_subspace.py` | Universal Subspace extractor (SVD-based), deviation ratio computation |
| `trajectory_analysis.py` | Comprehensive trajectory metrics (velocity, curvature, momentum, excursion) |
| `data_generation.py` | Experiment configuration, hyperparameter sweeps, dataset generation pipeline |
| `phase2_data_generation.py` | Phase 2 specific data generation with FLR tracking |
| `phase3_data_generation.py` | Phase 3 specific data generation with subspace analysis |
| `phase4_data_generation.py` | Phase 4 specific data generation with dense trajectory tracking |
| `symbolic_regression.py` | PySR wrapper, equation selection, Pareto front analysis |
| `validation.py` | Analytical prediction comparison, validation metrics |

### Data Flow

```
ExperimentConfig → generate_forgetting_dataset() → DataFrame → run_symbolic_regression() → Equations
                         ↓                            ↓                    ↓
                   results/{phase}/data.csv    analysis.json      discovered equations
```

### Key Classes

- **`TaskPair`**: Container for two teacher weight matrices with controlled similarity
- **`LinearTeacher`/`NonlinearTeacher`**: Ground-truth function generators
- **`LinearStudent`/`NonlinearStudent`**: Learnable networks tracking weight changes
- **`UniversalSubspace`**: SVD-based subspace extractor with projection/analysis methods
- **`TrajectoryMetrics`**: Comprehensive trajectory statistics (22 metrics)
- **`ExperimentConfig`**: Hyperparameter sweep specification

### Metrics Computed

| Metric | Definition |
|--------|------------|
| `forgetting` | `loss_t1_after_t2 - loss_t1_after_t1` |
| `FLR` (Feature Learning Rate) | `1 - CKA(K_init, K_current)` |
| `deviation_ratio` | `‖θ_⊥‖ / ‖θ_∥‖` (perpendicular/parallel to subspace) |
| `ntk_alignment` | Cosine similarity between initial and current NTK |

### Results Structure

```
results/
├── phase{1,2,3,4}/
│   ├── *_data.csv              # Raw experimental data
│   ├── *_data_config.json      # Experiment configuration
│   ├── *_analysis.json         # SR results and correlations
│   └── PHASE{N}_RESULTS.md     # Human-readable summary
├── visualizations/             # Generated plots and animations
└── FINAL_SUMMARY.md           # Cross-phase synthesis
```

## Configuration

Phase 1 config example (`config/phase1_config.yaml`):

```yaml
data:
  d_in: 100
  d_out: 10
  widths: [50, 100, 200, 500, 1000]
  similarities: [0.0, 0.1, ..., 1.0]
  learning_rates: [0.001, 0.01, 0.1]
  n_steps: [100, 500, 1000, 2000]
  n_seeds: 5

symbolic_regression:
  niterations: 100
  maxsize: 30
  parsimony: 0.001
```

## Known Issues

1. **Deviation metric instability**: The ratio `‖θ_⊥‖ / ‖θ_∥‖` produces extreme values (up to 60,000) when `‖θ_∥‖ → 0`. Use angle-based metrics for robustness.

2. **Phase 4 analysis correction**: Original analysis contained incorrect correlations; corrected 2026-01-07. The trajectory hypothesis is NOT supported (r ≈ 0.08).

## Key Theoretical References

- Evron et al. (2022): Closed-form forgetting in linear regression
- Goldfarb et al. (2024): Joint effect of task similarity + overparameterization
- Graldi et al. (2025): Lazy-rich transition in continual learning
- Kaushik et al. (2024): Universal Weight Subspace Hypothesis (arXiv:2512.05117)
