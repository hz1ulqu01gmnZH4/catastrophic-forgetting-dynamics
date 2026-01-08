# Phase 3 Results: Universal Subspace Analysis

## Testing the Subspace Deviation Hypothesis

**Date:** 2026-01-07
**Status:** Complete
**Data Points:** 845 experimental runs (55 failed due to numerical instability)

---

## 1. Executive Summary

Phase 3 tested whether **subspace deviation** (from Kaushik et al.'s Universal Subspace hypothesis) predicts forgetting and regime transitions. The results are **surprising and nuanced**:

### Key Findings

| Metric | Correlation with Forgetting | Interpretation |
|--------|----------------------------|----------------|
| **Similarity** | **-0.91** | Dominant predictor |
| Max Deviation (T2) | +0.68 | Strong positive correlation |
| Deviation Increase | +0.21 | Moderate correlation |
| Learning Rate | +0.16 | Weak correlation |
| FLR | +0.05 | Very weak correlation |
| **Deviation After T2** | **-0.001** | No correlation! |

### Critical Discovery

**Subspace deviation at the END of training doesn't predict forgetting.** But **maximum deviation DURING training** (r=0.68) does correlate strongly.

### Best Model (R² = 0.86)

$$\text{Forgetting} = 0.64 \cdot (1-s) + 0.64 \cdot \eta + 0.05 \cdot \text{FLR} + 0.004 \cdot \text{deviation} - 0.06$$

**Similarity dominates** - it explains most of the variance.

---

## 2. Experimental Configuration

```yaml
Input dimension: 50
Output dimension: 5
Hidden widths: [32, 64, 128]
Activation: GELU (best for feature learning signal)
Similarities: [0.0, 0.25, 0.5, 0.75, 1.0]
Learning rates: [0.05, 0.1, 0.15, 0.2]  # Focus on transition region
Init scales: [0.5, 1.0, 2.0]
Training steps: 300
Weight tracking: Every 30 steps
Subspace variance target: 90%
Seeds: 5
Total runs: 900 (845 successful)
```

---

## 3. Subspace Deviation Analysis

### 3.1 What We Measured

- **Deviation After T1**: ||θ_⊥|| / ||θ_∥|| after task 1 training
- **Deviation After T2**: ||θ_⊥|| / ||θ_∥|| after task 2 training
- **Max Deviation T1/T2**: Peak deviation during each task
- **Deviation Increase**: Change from T1 to T2
- **Alignment**: Cosine similarity with fitted subspace

### 3.2 Deviation Statistics

| Metric | Mean | Std |
|--------|------|-----|
| Deviation After T2 | 0.262 | 0.141 |
| Max Deviation T2 | — | — |
| Alignment After T2 | — | — |

### 3.3 The Surprising Non-Result

**Deviation at end of training has ZERO correlation with forgetting** (r = -0.001).

This contradicts the simple hypothesis that "networks that deviate from universal subspace forget more."

---

## 4. What DOES Predict Forgetting?

### 4.1 Correlation Ranking

| Variable | Correlation | Interpretation |
|----------|-------------|----------------|
| Similarity | **-0.91** | Task similarity is overwhelmingly dominant |
| Max Deviation T2 | +0.68 | Peak deviation during training matters |
| Deviation Increase | +0.21 | Growing deviation correlates |
| Learning Rate | +0.16 | Higher LR → more forgetting |
| FLR | +0.05 | Feature learning rate weakly correlates |
| Deviation After T2 | -0.001 | Final deviation is irrelevant |

### 4.2 Why Similarity Dominates

In Phase 3, we focused on the LR ≥ 0.05 regime where most runs show significant feature learning. In this regime:

- **Orthogonal tasks (s=0)**: Mean forgetting ≈ 0.67
- **Identical tasks (s=1)**: Mean forgetting ≈ -0.10 (forward transfer)

The similarity effect is **linear and strong** in the nonlinear regime.

### 4.3 Max Deviation vs Final Deviation

The key insight: **trajectory matters, not endpoint**.

- Max deviation during T2 (r = +0.68): Networks that temporarily deviate far from subspace during learning forget more
- Final deviation (r = -0.001): Where they end up doesn't matter

**Hypothesis**: Forgetting happens during the excursion, not because of the final position.

---

## 5. Regime Analysis

### 5.1 Lazy vs Rich Regime Statistics

| Regime | Count | % | Mean Forgetting | Mean Deviation | Mean FLR |
|--------|-------|---|-----------------|----------------|----------|
| Lazy | 442 | 52% | 0.346 | 0.247 | 0.042 |
| Rich | 403 | 48% | 0.352 | 0.278 | 0.256 |

**Key Observation**: In Phase 3's high-LR regime (0.05-0.2), we see nearly 50/50 lazy/rich split, compared to Phase 2's 90/10 split (which included LR ≤ 0.01).

### 5.2 Deviation Does NOT Cleanly Separate Regimes

| Metric | Lazy Regime | Rich Regime | Ratio |
|--------|-------------|-------------|-------|
| Mean Deviation | 0.247 | 0.278 | 1.12× |

**Disappointing**: Deviation only shows 12% difference between regimes. FLR remains the better regime indicator.

### 5.3 Rich Regime by Learning Rate

| Learning Rate | % Rich | Mean Forgetting |
|---------------|--------|-----------------|
| 0.05 | 30% | 0.286 |
| 0.10 | 44% | 0.350 |
| 0.15 | 46% | 0.375 |
| 0.20 | 74% | 0.392 |

Higher LR consistently produces more rich regime runs and more forgetting.

---

## 6. Model Fitting Results

### 6.1 Candidate Models

| Model | Equation | R² |
|-------|----------|-----|
| Linear Deviation | F = a·deviation + b | 0.000 |
| Deviation + FLR | F = a·deviation + b·FLR + c | 0.003 |
| Threshold Model | F = baseline + penalty·(deviation ≥ θ) | 0.003 |
| **Full Model** | F = a·deviation + b·FLR + c·LR + d·(1-s) + e | **0.856** |

### 6.2 Best Model

$$\boxed{\text{Forgetting} = 0.004 \cdot \text{deviation} + 0.053 \cdot \text{FLR} + 0.643 \cdot \eta + 0.637 \cdot (1-s) - 0.057}$$

**R² = 0.856** (86% variance explained)

**Coefficient Interpretation:**

| Term | Coefficient | Effect |
|------|-------------|--------|
| $(1-s)$ | **+0.637** | +0.06 per 0.1 dissimilarity |
| $\eta$ | **+0.643** | +0.06 per 0.1 LR increase |
| FLR | +0.053 | +0.005 per 0.1 FLR |
| Deviation | +0.004 | Negligible |
| Intercept | -0.057 | Baseline |

### 6.3 Why R² = 0.86?

Phase 3 achieved much higher R² than Phase 2 (0.52) because:

1. **Focused on high-LR regime**: Less noise from lazy regime
2. **Used GELU only**: More consistent activation dynamics
3. **Similarity dominates**: Linear effect is very strong

---

## 7. Transition Boundary Analysis

### 7.1 Optimal Threshold

Using the transition boundary algorithm:

| Metric | Value |
|--------|-------|
| Deviation Threshold | 1.09 |
| Forgetting Threshold | 0.67 |
| Classification Accuracy | 90% |
| Separation | 0.38 |

### 7.2 Threshold Model Performance

The threshold model:
$$F = 0.348 + 0.381 \cdot \mathbb{1}[\text{deviation} \geq 1.09]$$

achieves only R² = 0.003 - **the threshold is too high** to be useful (very few samples exceed it).

---

## 8. Comparison: Phase 2 vs Phase 3

| Metric | Phase 2 | Phase 3 |
|--------|---------|---------|
| Mean Forgetting | +0.083 | +0.349 |
| Regime Split | 90% lazy / 10% rich | 52% lazy / 48% rich |
| LR Range | 0.001-0.1 | 0.05-0.2 |
| Best Model R² | 0.52 | **0.86** |
| Similarity Correlation | -0.37 | **-0.91** |
| LR Correlation | +0.59 | +0.16 |
| Key Predictor | LR | **Similarity** |

### 8.1 Why Results Differ

Phase 3 focused on the **high-LR regime** where:
- Most networks are in or near rich regime
- Similarity becomes the dominant factor
- Learning rate effect saturates
- Deviation metrics show less separation

---

## 9. Key Phase 3 Discoveries

### 9.1 Subspace Deviation Hypothesis: Partially Rejected

❌ **Final deviation doesn't predict forgetting** (r = -0.001)
✅ **Max deviation during training does correlate** (r = +0.68)
✅ **Deviation increase correlates moderately** (r = +0.21)

**Conclusion**: The Universal Subspace hypothesis needs refinement. It's not about where you end up, but how far you travel during learning.

### 9.2 Similarity is King

In the high-LR regime, task similarity explains 83% of forgetting variance alone (r² = 0.83).

### 9.3 FLR Remains Better Regime Indicator Than Deviation

| Metric | Lazy vs Rich Ratio |
|--------|-------------------|
| FLR | 6.1× |
| Deviation | 1.12× |

### 9.4 The Trajectory Hypothesis

**New hypothesis**: Forgetting is determined by the **maximum excursion** from the universal subspace during training, not the final position.

$$\text{Forgetting} \propto \max_t \|\theta_\perp(t)\| / \|\theta_\parallel(t)\|$$

---

## 10. Implications for Phase 4

### 10.1 Questions Answered

✅ Does deviation predict forgetting? **No** (final), **Yes** (max during training)
✅ Is deviation better than FLR? **No** - FLR remains superior
✅ Can we fit transition boundary? **Partially** - threshold too high to be practical

### 10.2 Questions for Phase 4

1. Can we track **trajectory curvature** in weight space?
2. Does the **integral of deviation** over training predict forgetting?
3. Is there a **causal** relationship between excursion and forgetting?
4. Can **gradient direction analysis** explain why max deviation matters?

### 10.3 Recommended Phase 4 Focus

Based on Phase 3 findings:

1. **Track full deviation trajectory**, not just endpoints
2. Compute **path integral** of deviation during training
3. Test whether **deviation velocity** (rate of change) predicts forgetting
4. Explore **information-theoretic measures** of trajectory complexity

---

## 11. Files Generated

```
results/phase3/
├── phase3_data.csv           # Raw experimental data (845 rows)
├── phase3_data_config.json   # Experiment configuration
└── PHASE3_RESULTS.md         # This document
```

---

## 12. Conclusions

### 12.1 Phase 3 Success Criteria

| Criterion | Target | Achieved | Status |
|-----------|--------|----------|--------|
| Measure subspace deviation | Yes | Yes | ✅ |
| Correlate with forgetting | r > 0.3 | r = -0.001 (final), r = 0.68 (max) | ⚠️ |
| Compare to FLR | — | FLR better for regime | ✅ |
| Fit transition boundary | — | Threshold too high | ⚠️ |
| Model R² | > 0.6 | **0.86** | ✅ |

### 12.2 Summary

Phase 3 reveals a **nuanced picture** of the Universal Subspace hypothesis:

1. **Final subspace deviation is NOT predictive** of forgetting (r ≈ 0)
2. **Maximum deviation during training IS predictive** (r = 0.68)
3. **Similarity dominates** in the high-LR regime (r = -0.91)
4. **FLR remains the best regime indicator** (6× separation vs 1.1×)

The key insight: **Forgetting happens during the journey, not because of the destination.**

The discovered equation:

$$\boxed{\text{Forgetting} \approx 0.64 \cdot (1-s) + 0.64 \cdot \eta + 0.05 \cdot \text{FLR} - 0.06}$$

achieves R² = 0.86 in the high-LR regime, with similarity and learning rate as equally important predictors.

---

*Phase 3 Complete. The Subspace Deviation hypothesis is partially supported—trajectory matters, not endpoint.*
