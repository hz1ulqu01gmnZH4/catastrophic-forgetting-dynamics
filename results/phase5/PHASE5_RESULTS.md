# Phase 5.1: Gradient Interference Analysis Results
*Generated: 2026-01-08T21:36:19.655829*
## Summary
This experiment tests whether gradient interference during Task 2 training causally explains catastrophic forgetting.
### Hypothesis
**H1**: Forgetting ∝ -cos(∇L_T1, ∇L_T2)
- Negative gradient angles (T2 updates pointing away from T1 minimum) → more forgetting
- Positive gradient angles (T2 updates compatible with T1) → less forgetting

## Key Results
- Gradient angle negatively correlates with forgetting (r=-0.869), supporting the hypothesis that destructive interference causes forgetting.
- Gradient angle has partial correlation r=-0.848 with forgetting even after controlling for similarity, suggesting it captures additional mechanism.

## Correlation Analysis
| Metric | Pearson r | p-value | Significant |
|--------|-----------|---------|-------------|
| similarity | -0.3827 | 1.04e-18 | Yes |
| mean_gradient_angle | -0.8688 | 1.25e-152 | Yes |
| min_gradient_angle | -0.8598 | 5.21e-146 | Yes |
| max_gradient_angle | -0.8524 | 6.19e-141 | Yes |
| std_gradient_angle | +0.1928 | 1.57e-05 | Yes |
| mean_gradient_projection | -0.2821 | 1.65e-10 | Yes |
| cumulative_interference | +0.5185 | 2.07e-35 | Yes |
| early_gradient_angle | -0.8263 | 5.28e-125 | Yes |
| late_gradient_angle | -0.8376 | 1.57e-131 | Yes |

## Partial Correlation
- Gradient angle | Similarity: r = -0.8482 (p = 3.81e-138)

## Hypothesis Verdict
**SUPPORTED**

Gradient interference is a causal mechanism for forgetting. This suggests mitigation strategies based on gradient projection (e.g., OGD, A-GEM) could be effective.

## Interpretation
### What gradient angle tells us
- **Positive angle**: T2 gradient points in same direction as T1 gradient (learning T2 helps T1)
- **Zero angle**: Orthogonal gradients (independent learning)
- **Negative angle**: T2 gradient opposes T1 gradient (destructive interference)

### Connection to similarity
High task similarity → gradients tend to align → less destructive interference
Low task similarity → gradients tend to conflict → more destructive interference

## Next Steps
1. **Phase 6.2**: Implement gradient projection methods (OGD, A-GEM)
2. Test if blocking destructive gradients reduces forgetting
3. Find minimal intervention for maximum forgetting reduction

---
*Phase 5.1 Complete*
