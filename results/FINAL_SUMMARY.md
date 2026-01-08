# Symbolic Regression for Catastrophic Forgetting Dynamics

## Final Summary: Four Phases of Discovery (CORRECTED)

**Date:** 2026-01-07 (Phases 1-4), 2026-01-08 (Nested Learning follow-up)
**Total Experiments:** 5,820 runs (5,120 across 4 phases + 700 nested learning)
**Status:** Complete (with Phase 4 corrections and Nested Learning follow-up)

---

## Executive Summary

This experiment used symbolic regression and systematic hyperparameter sweeps to discover equations governing **catastrophic forgetting** in neural networks. Across four phases, we progressed from linear baselines through nonlinear dynamics to test various hypotheses.

### The Journey

```
Phase 1: Linear Networks      → Learning rate dominates
Phase 2: Nonlinear Networks   → Lazy-rich transition discovered
Phase 3: Subspace Analysis    → Similarity dominates (r = -0.91)
Phase 4: Trajectory Analysis  → Trajectory hypothesis NOT supported
```

### The Key Discovery

$$\boxed{\text{Forgetting} \approx 0.59 - 0.65 \cdot \text{similarity}}$$

**Task similarity is the dominant predictor. Everything else is noise.**

---

## Phase-by-Phase Summary

### Phase 1: Linear Network Baseline

**Goal:** Validate methodology by reproducing known analytical results.

**Data:** 1,980 experimental runs

**Key Findings:**
- All 4 theoretical predictions from literature confirmed
- Learning rate is 3× more predictive than task similarity
- Width has NO effect in linear networks (lazy regime only)

**Discovered Equation:**
$$\text{Forgetting} \approx 0.30 \cdot (1-s)^{0.22} \cdot \eta^{0.39} \cdot t^{0.17} - 0.19$$

**R² = 0.63**

| Variable | Correlation |
|----------|-------------|
| Learning rate | +0.71 |
| Weight change | +0.60 |
| Similarity | -0.23 |

---

### Phase 2: Nonlinear Networks & Lazy-Rich Transition

**Goal:** Observe lazy-rich transition in nonlinear networks.

**Data:** 1,620 experimental runs

**Key Findings:**
- **Lazy regime (90.4%)**: Forgetting = 0.060
- **Rich regime (9.6%)**: Forgetting = 0.392 (**6.6× more**)
- **Learning rate ≥ 0.1 is the ONLY trigger** for rich regime
- FLR (Feature Learning Rate) cleanly separates regimes (50× difference)

**Discovered Equation:**
$$\text{Forgetting} \approx 0.40(1-s) + 5.17\eta - 0.12 \cdot \text{FLR} - 0.29$$

**R² = 0.52**

| Regime | % of Runs | Mean Forgetting | Mean FLR |
|--------|-----------|-----------------|----------|
| Lazy | 90.4% | 0.060 | 0.009 |
| Rich | 9.6% | 0.392 | 0.455 |

**Critical threshold:** LR ≥ 0.1 triggers rich regime

---

### Phase 3: Universal Subspace Analysis

**Goal:** Test if subspace deviation predicts forgetting.

**Data:** 845 experimental runs

**Key Findings:**
- **Final deviation has ZERO correlation** with forgetting (r = -0.001)
- **Max deviation during training correlates** (r = +0.68) — but see Phase 4 correction
- Similarity dominates in high-LR regime (r = -0.91)
- FLR correlation drops to r = +0.05

**Discovered Equation:**
$$\text{Forgetting} \approx 0.64 \cdot (1-s) + 0.64 \cdot \eta + 0.05 \cdot \text{FLR} - 0.06$$

**R² = 0.86**

**Critical insight:** Similarity emerges as the dominant predictor.

---

### Phase 4: Trajectory Hypothesis Testing (CORRECTED)

**Goal:** Test if trajectory through weight space predicts forgetting.

**Data:** 675 experimental runs

**Key Findings (CORRECTED):**
- ❌ **Trajectory metrics are WEAK** (r = 0.08-0.10)
- ❌ **The trajectory hypothesis is NOT supported**
- ✓ **Similarity remains dominant** (r = -0.91)
- ⚠️ **Numerical instability** caused misleading outliers in original analysis

**Corrected Correlations:**

| Predictor | Correlation |
|-----------|-------------|
| similarity | **-0.91** (dominant) |
| learning_rate | +0.15 |
| t2_mean_deviation | +0.10 |
| t2_max_deviation | +0.08 |
| flr_after_t2 | +0.02 |

**Simple Model (R² = 0.83):**
$$\text{Forgetting} \approx 0.59 - 0.65 \cdot \text{similarity}$$

---

## The Unified Theory (REVISED)

### One Factor Dominates Forgetting

```
┌─────────────────────────────────────────────────────────────┐
│                    FORGETTING DYNAMICS                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   1. TASK SIMILARITY (s)              ← DOMINANT            │
│      └─ Correlation: r = -0.91                              │
│      └─ Explains 83% of variance alone                      │
│      └─ Higher similarity → less forgetting                 │
│                                                             │
│   2. LEARNING RATE (η)                ← SMALL EFFECT        │
│      └─ Correlation: r = +0.15                              │
│      └─ Triggers regime transition at η ≥ 0.1               │
│      └─ Effect: Higher LR → slightly more forgetting        │
│                                                             │
│   3. TRAJECTORY METRICS               ← NEGLIGIBLE          │
│      └─ Correlation: r ≈ 0.08                               │
│      └─ Numerically unstable                                │
│      └─ Effect: None after controlling for similarity       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### The Master Equation (SIMPLIFIED)

$$\boxed{\text{Forgetting} \approx 0.59 - 0.65 \cdot \text{similarity}}$$

That's it. One variable, R² = 0.83.

---

## Key Discoveries Ranked by Importance

### 1. Task Similarity Dominates ⭐⭐⭐

**Forgetting is almost entirely determined by task similarity.**

r = -0.91 correlation, explaining 83% of variance. This is the only predictor that matters.

### 2. The Lazy-Rich Transition ⭐⭐⭐

**Learning rate ≥ 0.1 triggers a phase transition.**

Below this threshold: 100% lazy regime, minimal forgetting.
Above: 29% chance of rich regime, 6.6× more forgetting.

### 3. Trajectory Hypothesis Failed ⭐⭐

**The trajectory through weight space does NOT predict forgetting.**

Phase 4 tested this hypothesis rigorously. Trajectory metrics (max deviation, path integral, excursion intensity) all have r ≈ 0.08 — essentially noise.

### 4. FLR is a Poor Predictor ⭐

**Feature Learning Rate explains almost nothing.**

r = 0.02-0.05 depending on phase. Not useful for prediction.

### 5. Width is Irrelevant ⭐

**Overparameterization doesn't affect forgetting.**

Confirmed across all phases.

---

## Practical Implications

### For Continual Learning Practitioners

| Finding | Practical Advice |
|---------|------------------|
| Similarity dominates | **Train similar tasks consecutively** |
| LR threshold | Keep LR < 0.1 to stay in lazy regime |
| Trajectory doesn't matter | Don't monitor trajectory — it's noise |
| Simple models win | Use task similarity as your primary metric |

### For Researchers

| Finding | Research Direction |
|---------|-------------------|
| Similarity dominance | Study why similarity matters so much |
| Trajectory failure | Develop better deviation metrics |
| Numerical instability | Use angle-based metrics, not ratios |
| Lazy-rich transition | Study phase transition boundaries |

---

## Model Comparison Across Phases

| Phase | Best R² | Key Predictors | Samples |
|-------|---------|----------------|---------|
| 1 | 0.63 | LR, similarity, steps | 1,980 |
| 2 | 0.52 | LR, FLR, similarity | 1,620 |
| 3 | 0.86 | Similarity, LR | 845 |
| 4 | 0.83 | **Similarity alone** | 675 |

**Note:** Phase 4 achieves R² = 0.83 using **only similarity**. Adding trajectory metrics provides no improvement.

---

## Correlation Evolution Across Phases

| Predictor | Phase 1 | Phase 2 | Phase 3 | Phase 4 |
|-----------|---------|---------|---------|---------|
| Similarity | -0.23 | -0.37 | -0.91 | **-0.91** |
| Learning rate | +0.71 | +0.59 | +0.16 | +0.15 |
| FLR | N/A | +0.28 | +0.05 | +0.02 |
| Max deviation | N/A | N/A | +0.68* | **+0.08** |

*Phase 3 max deviation correlation was calculated differently; Phase 4 direct recalculation shows r = 0.08.

**Pattern:** As experiments focused on high-LR regimes, similarity emerged as the dominant (and only meaningful) predictor.

---

## Technical Details

### Experimental Setup

```yaml
# Common across phases
Input dimension: 50-100
Output dimension: 5-10
Activation: GELU (Phases 2-4)
Optimizer: SGD
Loss: MSE

# Phase-specific
Phase 1: Linear, widths 50-500, LR 0.001-0.1
Phase 2: Nonlinear, widths 32-256, LR 0.001-0.1
Phase 3: Subspace tracking every 30 steps
Phase 4: Dense tracking every 10 steps
```

### Key Metrics Defined

| Metric | Definition |
|--------|------------|
| Forgetting | loss_t1_after_t2 - loss_t1_after_t1 |
| FLR | 1 - CKA(K_init, K_current) |
| Deviation | \|\|θ_⊥\|\| / \|\|θ_∥\|\| |
| Max deviation | max_t(deviation(t)) |
| Path integral | ∫ deviation(t) dt |

### Known Issues

1. **Deviation metric instability**: The ratio ||θ_⊥|| / ||θ_∥|| produces extreme values (up to 60,000) when ||θ_∥|| → 0
2. **Phase 4 analysis bug**: Original analysis script produced incorrect correlations; corrected 2026-01-07

---

## Files Generated

```
results/
├── phase1/
│   ├── forgetting_data.csv
│   ├── config.json
│   └── PHASE1_RESULTS.md
├── phase2/
│   ├── phase2_data.csv
│   ├── phase2_data_config.json
│   └── PHASE2_RESULTS.md
├── phase3/
│   ├── phase3_data.csv
│   ├── phase3_data_config.json
│   └── PHASE3_RESULTS.md
├── phase4/
│   ├── phase4_data.csv
│   ├── phase4_data_config.json
│   ├── phase4_analysis.json (contains errors)
│   └── PHASE4_RESULTS.md (CORRECTED)
├── visualizations/
│   ├── phase_comparison.png
│   ├── trajectory_comparison.png
│   ├── key_findings.png
│   ├── trajectory_animation.gif
│   └── phase4_investigation.png
├── nested_learning_test.csv      # Multi-timescale experiment (450 runs)
├── deep_nesting_test.csv         # Deep nesting experiment (250 runs)
├── deep_nesting_analysis.json    # Deep nesting analysis
├── NESTED_LEARNING_REPORT.md     # Nested learning follow-up report
└── FINAL_SUMMARY.md              # This document (CORRECTED + updated)
```

---

## Conclusions

### What We Learned

1. **Catastrophic forgetting is predictable** — R² = 0.83 with one variable
2. **Task similarity is everything** — r = -0.91, the dominant predictor
3. **The trajectory hypothesis failed** — r ≈ 0.08, not meaningful
4. **Learning rate triggers regime transition** — η ≥ 0.1 → rich regime
5. **Simple models win** — Complex trajectory metrics add nothing

### The Final Word

**Forgetting is about task relationships, not optimization dynamics.**

Train similar tasks together. That's the actionable insight from 5,120 experiments.

---

## Follow-up: Nested Learning Experiments (2026-01-08)

After completing Phases 1-4, we tested whether Google's "Nested Learning" ideas could reduce catastrophic forgetting.

### What We Tested

| Method | Description | Improvement |
|--------|-------------|-------------|
| Multi-timescale | Slow (10% LR) + Fast (100% LR) parameters | **16.7%** |
| Surprise-gated | Updates weighted by gradient magnitude | **14.4%** |
| Deep nesting (5 levels) | 5 timescale levels with 10× decay each | **15.3%** |

### Key Finding

**Multi-timescale helps within-distribution, not across novel tasks.**

| Similarity | Standard | Multi-timescale | Benefit |
|------------|----------|-----------------|---------|
| 0.00 (orthogonal) | 0.723 | 0.614 | 15% |
| 0.75 (similar) | 0.112 | 0.087 | **22%** |

Deeper nesting provides diminishing returns (~4.5% per level), and **similarity correlation remains r = -0.92 to -0.95**.

### Interpretation

Nested Learning solves **context rot** (long-context memory within same task type) not **catastrophic forgetting** (sequential learning of different tasks).

For autoregressive language modeling where all tasks share linguistic structure, nested learning likely helps significantly. For genuinely orthogonal tasks, it provides only marginal improvement.

**See:** `NESTED_LEARNING_REPORT.md` for full details (700 additional experiments).

---

## Future Directions

1. **Why does similarity dominate?** — Theoretical analysis needed
2. **Better deviation metrics** — Angle-based, numerically stable
3. **Multi-task sequences** — Extend beyond 2 tasks
4. **Real architectures** — Test on CNNs, Transformers
5. **Task similarity estimation** — Practical methods for real datasets
6. **Nested learning on structured tasks** — Test with tasks sharing latent structure

---

## Erratum

The original Phase 4 analysis and FINAL_SUMMARY.md contained incorrect correlations for trajectory metrics. The reported r = 0.68 was wrong; the actual correlation is r = 0.08.

This error was caused by a bug in the analysis script that incorrectly computed correlations. The correction was identified on 2026-01-07 by re-analyzing the raw CSV data directly.

---

*Experiment Complete. Task similarity determines forgetting. The trajectory hypothesis is not supported.*

**Total experimental runs:** 5,820 (5,120 + 700 nested learning)
**Total computation time:** ~1.5 hours
**Key insight:** Similarity is all you need (r = -0.91)

