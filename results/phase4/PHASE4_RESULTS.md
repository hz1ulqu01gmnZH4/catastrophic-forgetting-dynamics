# Phase 4 Results: Trajectory Hypothesis Testing

## CORRECTED: Trajectory Hypothesis NOT Supported

**Date:** 2026-01-07
**Status:** Complete (CORRECTED)
**Data Points:** 675 experimental runs

---

## 1. Executive Summary

Phase 4 tested the trajectory hypothesis: **Forgetting ∝ max_t ||θ_⊥(t)|| / ||θ_∥(t)||**

### ❌ HYPOTHESIS NOT SUPPORTED

| Finding | Expected | Actual |
|---------|----------|--------|
| **Trajectory beats FLR** | r >> 0.03 | r = 0.08 (only 4× better) |
| **Max > Final deviation** | Large difference | 0.08 vs 0.05 (marginal) |
| **Path integral matters** | r > 0.3 | r = 0.10 (weak) |
| **Excursion intensity** | r > 0.3 | r = 0.08 (weak) |

### Key Discovery

**Task similarity remains the dominant predictor (r = -0.91).** Trajectory metrics provide minimal additional predictive power (r ≈ 0.08-0.10).

### Corrected Correlations

| Predictor | Actual Correlation |
|-----------|-------------------|
| similarity | **-0.91** (dominant) |
| learning_rate | +0.15 |
| t2_mean_deviation | +0.10 |
| t2_path_integral_deviation | +0.10 |
| t2_max_deviation | +0.08 |
| t2_excursion_intensity | +0.08 |
| flr_after_t2 | +0.02 |

---

## 2. Experimental Configuration

```yaml
Input dimension: 50
Output dimension: 5
Hidden widths: [32, 64, 128]
Activation: GELU
Similarities: [0.0, 0.25, 0.5, 0.75, 1.0]
Learning rates: [0.05, 0.1, 0.15]
Init scales: [0.5, 1.0, 2.0]
Training steps: 300
Track every: 10 steps (dense!)
Seeds: 5
Total runs: 675
```

**Key difference from Phase 3**: Dense tracking every 10 steps (vs 30) to capture full trajectory shape.

---

## 3. Actual Correlations with Forgetting

| Rank | Variable | Correlation | Category |
|------|----------|-------------|----------|
| 1 | **similarity** | **-0.91** | Task property |
| 2 | learning_rate | +0.15 | Hyperparameter |
| 3 | t2_mean_deviation | +0.10 | Trajectory |
| 4 | t2_path_integral_deviation | +0.10 | Trajectory |
| 5 | t2_max_deviation | +0.08 | Trajectory |
| 6 | t2_excursion_intensity | +0.08 | Trajectory |
| 7 | t2_final_deviation | +0.05 | Trajectory |
| 8 | flr_after_t2 | +0.02 | Feature learning |
| 9 | t2_initial_deviation | +0.02 | Trajectory |

### Critical Observation

**Similarity dominates everything else by an order of magnitude.**

All trajectory metrics have correlations between 0.05-0.10, which is statistically weak and practically useless for prediction.

---

## 4. Trajectory Hypothesis Validation

### 4.1 Does Max Deviation Beat Final Deviation?

| Metric | Correlation | Winner |
|--------|-------------|--------|
| t2_max_deviation | +0.08 | Marginal |
| t2_final_deviation | +0.05 | |

**Barely.** The difference (0.08 vs 0.05) is not meaningful.

### 4.2 Does Trajectory Beat FLR?

| Predictor | Correlation | Improvement |
|-----------|-------------|-------------|
| Best trajectory (t2_mean_deviation) | +0.10 | — |
| FLR | +0.02 | 5× worse |

**Marginally.** Trajectory is 5× better than FLR, but both are weak predictors.

### 4.3 Does Path Integral Matter?

| Metric | Correlation | Interpretation |
|--------|-------------|----------------|
| t2_path_integral_deviation | +0.10 | Weak effect |
| t2_area_under_curve | +0.10 | Weak effect |

**Not meaningfully.** r = 0.10 explains only 1% of variance.

### 4.4 Does Excursion Shape Matter?

| Metric | Correlation |
|--------|-------------|
| t2_excursion_intensity | +0.08 |
| t2_deviation_std | +0.08 |
| t2_deviation_range | +0.08 |

**No.** All shape metrics have weak correlations.

---

## 5. Why Did Trajectory Metrics Fail?

### 5.1 Numerical Instability

The deviation metric ||θ_⊥|| / ||θ_∥|| produced extreme outliers:

| Statistic | Value |
|-----------|-------|
| Min | 0.0 |
| Median | 2.1 |
| Mean | 165 |
| Max | **59,779** |

The huge range (0 to 60,000) indicates numerical instability when ||θ_∥|| approaches zero.

### 5.2 Similarity Confounding

High forgetting cases (similarity = 0.0) and low forgetting cases (similarity = 1.0) are completely separated by task similarity alone. Trajectory metrics don't add information beyond what similarity already captures.

### 5.3 The Real Model

A simple linear model using only similarity achieves:

$$\text{Forgetting} \approx 0.59 - 0.65 \cdot \text{similarity}$$

**R² = 0.83** — using just one predictor!

Adding trajectory metrics improves this marginally at best.

---

## 6. What Phase 4 Actually Shows

### 6.1 Similarity is Everything

| Similarity | Mean Forgetting | Std |
|------------|-----------------|-----|
| 0.00 | 0.59 | 0.19 |
| 0.25 | 0.52 | 0.17 |
| 0.50 | 0.38 | 0.12 |
| 0.75 | 0.21 | 0.09 |
| 1.00 | -0.03 | 0.04 |

The relationship is nearly perfectly linear.

### 6.2 Learning Rate Has Small Effect

| Learning Rate | Mean Forgetting |
|---------------|-----------------|
| 0.05 | 0.30 |
| 0.10 | 0.34 |
| 0.15 | 0.37 |

Only a 0.07 difference across the range.

### 6.3 Trajectory Metrics Are Noise

When controlling for similarity, trajectory metrics explain almost no additional variance.

---

## 7. Lessons Learned

### 7.1 The Subspace Deviation Metric is Problematic

The ratio ||θ_⊥|| / ||θ_∥|| is numerically unstable. When the parallel component is small, the ratio explodes to meaningless values.

**Better alternatives for future work:**
- Use ||θ_⊥|| alone (absolute deviation)
- Use angle-based metrics (cosine similarity)
- Regularize the denominator

### 7.2 Dense Tracking Didn't Help

Tracking every 10 steps (vs 30 in Phase 3) produced noisier data without improving predictive power.

### 7.3 Simple Models Win

Task similarity alone predicts forgetting with R² = 0.83. Complex trajectory metrics add nothing meaningful.

---

## 8. Corrected Conclusions

### 8.1 Phase 4 Success Criteria (REVISED)

| Criterion | Target | Achieved | Status |
|-----------|--------|----------|--------|
| Trajectory beats FLR | r_traj >> r_flr | 0.10 vs 0.02 | ⚠️ Marginal |
| Max beats Final deviation | r_max >> r_final | 0.08 vs 0.05 | ⚠️ Marginal |
| Path integral correlates | r > 0.3 | r = 0.10 | ❌ Failed |
| Model R² improvement | > 0.85 | ~0.83 | ❌ No improvement |

### 8.2 Summary

Phase 4 **does NOT validate** the trajectory hypothesis.

Key findings:
1. **Task similarity dominates** (r = -0.91), explaining 83% of variance alone
2. **Trajectory metrics are weak** (r = 0.08-0.10), adding <1% explained variance
3. **Numerical instability** in deviation calculations produced misleading outliers
4. **The journey doesn't matter** — where you start (similarity) determines the outcome

---

## 9. Revised Unified Picture

### What Actually Predicts Forgetting

$$\boxed{\text{Forgetting} \approx 0.59 - 0.65 \cdot \text{similarity}}$$

That's it. R² = 0.83 with one variable.

### The Three Factors (Revised)

1. **Task Similarity** — Dominant effect (r = -0.91)
2. **Learning Rate** — Small effect (r = +0.15)
3. **Trajectory Metrics** — Negligible effect (r ≈ 0.08)

### Implications

- **For practitioners**: Focus on task ordering (similar tasks together)
- **For theory**: The trajectory hypothesis needs better metrics or may be wrong
- **For future work**: Develop numerically stable deviation measures

---

## 10. Files Generated

```
results/phase4/
├── phase4_data.csv           # Raw experimental data (675 rows)
├── phase4_data_config.json   # Experiment configuration
├── phase4_analysis.json      # Full analysis results (contains errors)
└── PHASE4_RESULTS.md         # This document (CORRECTED)
```

---

## 11. Erratum

The original Phase 4 analysis contained incorrect correlations due to a bug in the analysis script. The reported r = 0.68 for trajectory metrics was wrong; the actual correlation is r = 0.08.

This correction was identified on 2026-01-07 by re-analyzing the raw data.

---

*Phase 4 Complete. The trajectory hypothesis is NOT supported: task similarity determines forgetting, trajectory shape does not.*

