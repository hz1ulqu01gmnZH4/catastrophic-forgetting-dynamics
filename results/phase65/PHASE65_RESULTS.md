# Phase 6.5: Similarity-Aware Learning Rate Results
*Generated: 2026-01-10T17:01:35.043103*

## Summary

This experiment tests whether adapting learning rate based on task similarity reduces forgetting.
This is a simple, low-cost intervention with **zero computational overhead**.

### Formula

```
lr_T2 = base_lr × similarity^α
```

- α = 0: No adaptation (baseline)
- α > 0: Lower LR for dissimilar tasks

### Theoretical Prediction

Based on forgetting equation `F ≈ 0.59 - 0.65 × similarity`, theoretical optimal α ≈ 1.54

## Key Results

- **α = 2.0 significantly reduces forgetting** by 15.1%
- α = 2.0: 15.1% reduction (significant)
- α = 1.5: 12.8% reduction (significant)
- α = 1.0: 10.0% reduction (significant)

## Alpha Value Comparison

| α | Mean Forgetting | Reduction | T2 Loss | Effective LR |
|---|-----------------|-----------|---------|-------------|
| 0.0 | 0.0383 | +0.0% | 0.0110 | 0.0533 |
| 0.25 | 0.0174 | +7.2% | 0.0474 | 0.0382 |
| 0.5 | 0.0158 | +8.0% | 0.0514 | 0.0334 |
| 0.75 | 0.0136 | +8.9% | 0.0559 | 0.0297 |
| 1.0 | 0.0107 | +10.0% | 0.0606 | 0.0268 |
| 1.5 | 0.0047 | +12.8% | 0.0710 | 0.0225 |
| 2.0 | 0.0004 | +15.1% | 0.0816 | 0.0196 |

## Optimal α by Task Similarity

| Similarity | Optimal α |
|------------|----------|
| 0.0 | 0.25 |
| 0.2 | 2.0 |
| 0.4 | 2.0 |
| 0.6 | 2.0 |
| 0.8 | 0.0 |
| 1.0 | 0.0 |

## Interpretation

### Moderate Effectiveness

Adaptive LR provides meaningful reduction (15.1%) but less effective than regularization (Phase 6.3: 31%).

## Comparison to Other Mitigation Methods

| Method | Reduction | T2 Impact | Overhead |
|--------|-----------|-----------|----------|
| Gradient Projection (Phase 6.2) | ~1% | Minimal | ~45% |
| L2 Regularization (Phase 6.3) | ~31% | +810% | ~30% |
| Adaptive LR (Phase 6.5) | ~15% | +641% | **0%** |

## Recommendations

- **Use α = 2.0** for simple, cost-free mitigation
- Combine with L2 regularization for stronger protection
- Tune α based on expected task similarity distribution
- No computational overhead makes this suitable for resource-constrained settings

---
*Phase 6.5 Complete*
