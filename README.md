# Catastrophic Forgetting Dynamics

Systematic experiments discovering equations governing catastrophic forgetting in neural networks through symbolic regression and hyperparameter sweeps.

## Key Finding

```
Forgetting ≈ 0.59 - 0.65 × similarity
```

**Task similarity is the dominant predictor (r = -0.91).** Everything else is noise.

## Experiments

| Phase | Experiments | Key Finding |
|-------|-------------|-------------|
| Phase 1 | 1,980 | Linear networks: LR dominates |
| Phase 2 | 1,620 | Nonlinear: Lazy-rich transition at LR ≥ 0.1 |
| Phase 3 | 845 | Subspace analysis: Similarity dominates (r = -0.91) |
| Phase 4 | 675 | Trajectory hypothesis: NOT supported |
| Nested Learning | 700 | Multi-timescale: 15-17% improvement, similarity still r = -0.92 |

**Total: 5,820 experiments**

## Results Summary

### What Predicts Forgetting

| Factor | Correlation | Verdict |
|--------|-------------|---------|
| Task similarity | r = -0.91 | **Dominant** |
| Learning rate | r = +0.15 | Small effect |
| Trajectory metrics | r ≈ 0.08 | Negligible |
| Feature Learning Rate | r ≈ 0.02 | Negligible |
| Network width | — | No effect |

### Nested Learning Follow-up

Tested Google's "Nested Learning" ideas (multi-timescale updates, surprise-based gating):

| Method | Improvement |
|--------|-------------|
| Multi-timescale (2 levels) | 16.7% |
| Surprise-gated | 14.4% |
| Deep nesting (5 levels) | 15.3% |

**Conclusion:** Multi-timescale helps within-distribution (same task type) but doesn't solve cross-task forgetting. Nested Learning solves **context rot**, not **catastrophic forgetting**.

## Practical Implications

| Finding | Advice |
|---------|--------|
| Similarity dominates | **Train similar tasks consecutively** |
| LR threshold | Keep LR < 0.1 to stay in lazy regime |
| Trajectory doesn't matter | Don't monitor trajectory — it's noise |
| Simple models win | Use task similarity as your primary metric |

## Project Structure

```
├── src/
│   ├── models.py                 # Linear/nonlinear network definitions
│   ├── data_generation.py        # Phase 1 data generation
│   ├── phase2_data_generation.py # Phase 2 with FLR tracking
│   ├── phase3_data_generation.py # Phase 3 subspace analysis
│   ├── phase4_data_generation.py # Phase 4 trajectory tracking
│   ├── nested_learning_test.py   # Multi-timescale experiments
│   └── deep_nesting_test.py      # Deep nesting (1-5 levels)
├── results/
│   ├── phase1/ - phase4/         # Phase-specific results
│   ├── FINAL_SUMMARY.md          # Complete summary
│   └── NESTED_LEARNING_REPORT.md # Nested learning analysis
├── config/                       # Experiment configurations
└── requirements.txt              # Dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

Or with uv:
```bash
uv run --with torch --with numpy --with pandas python src/nested_learning_test.py
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- NumPy, Pandas, SciPy
- PySR (optional, for symbolic regression)

## References

- Behrouz, A., et al. (2025). "Nested Learning: Towards Multi-Scale Optimization for Foundation Models." NeurIPS 2025.
- Behrouz, A., et al. (2024). "Titans: Learning to Memorize at Test Time." arXiv:2501.00663.
