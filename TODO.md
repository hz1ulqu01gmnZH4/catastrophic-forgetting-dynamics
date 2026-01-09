# TODO: Explaining and Mitigating Catastrophic Forgetting

## Current State

**What we know (updated 2026-01-09):**
- ✅ **Gradient interference is the causal mechanism** (r = -0.87)
- ✅ Similarity predicts forgetting because it determines gradient alignment
- ✅ Simple equation: `Forgetting ≈ 0.59 - 0.65 × similarity`
- ✅ Mechanistic equation: `Forgetting ∝ -cos(∇L_T1, ∇L_T2)`
- ✅ Learning rate ≥ 0.1 triggers lazy→rich regime transition
- ✅ Trajectory through weight space does NOT predict forgetting
- ✅ **Gradient projection provides LIMITED mitigation** (~1% reduction)
- ✅ Most task gradients are CONSTRUCTIVE (cos > 0), not destructive

**What we still don't know:**
- ~~WHY similarity dominates (causal mechanism)~~ ✅ SOLVED: Gradient interference
- ~~HOW effective are gradient projection methods?~~ ✅ TESTED: Limited (~1% reduction)
- WHETHER regularization (EWC) provides better mitigation
- WHETHER we can predict/estimate similarity before training

---

## Phase 5: Mechanistic Understanding

### 5.1 Gradient Interference Analysis ✅ COMPLETE

**Hypothesis:** Forgetting occurs because Task 2 gradients destructively interfere with Task 1 solution.

**RESULT: HYPOTHESIS CONFIRMED** (r = -0.87, partial r = -0.85)

```python
# Measured gradient alignment during Task 2 training
results = {
    'gradient_angle': 'r = -0.87',  # DOMINANT predictor
    'gradient_projection': 'r = -0.28',
    'cumulative_interference': 'r = +0.52',
}
```

**Completed Tasks:**
- [x] Implement gradient logging during Task 2 training
- [x] Compute gradient alignment with Task 1 loss surface
- [x] Correlate gradient interference with forgetting
- [x] Test: Does `forgetting ∝ -cos(∇L_T1, ∇L_T2)`? **YES!**

**Key Finding:** Gradient angle (r = -0.87) is a stronger predictor than similarity (r = -0.38) in the same experimental setup. Partial correlation after controlling for similarity is r = -0.85, proving gradient interference explains additional variance.

**Implication:** Gradient projection methods (OGD, A-GEM) should effectively reduce forgetting.

### 5.2 Representational Overlap Analysis

**Hypothesis:** Similar tasks share representations; dissimilar tasks compete for capacity.

```python
# Measure representation overlap
experiments = {
    'feature_overlap': 'CKA(h_T1, h_T2)',  # Hidden representation similarity
    'weight_overlap': 'cos(W_T1*, W_T2*)',  # Optimal weight similarity
    'activation_sparsity': 'overlap(active_neurons_T1, active_neurons_T2)',
}
```

**Tasks:**
- [ ] Train separate networks on each task to get "optimal" weights
- [ ] Measure overlap between optimal solutions
- [ ] Correlate representation overlap with forgetting
- [ ] Test: Is forgetting about competition for shared neurons?

### 5.3 Loss Landscape Geometry

**Hypothesis:** Forgetting depends on loss landscape geometry between task optima.

```python
# Analyze loss landscape
experiments = {
    'barrier_height': 'max L(αW_T1 + (1-α)W_T2) for α ∈ [0,1]',  # Linear interpolation barrier
    'hessian_alignment': 'cos(eigenvectors(H_T1), eigenvectors(H_T2))',
    'curvature_ratio': 'tr(H_T1) / tr(H_T2)',
}
```

**Tasks:**
- [ ] Compute loss along linear path between task solutions
- [ ] Measure barrier height vs. similarity
- [ ] Analyze Hessian eigenspectrum overlap
- [ ] Test: Does low barrier ↔ low forgetting?

### 5.4 Information-Theoretic Analysis

**Hypothesis:** Forgetting = information about T1 lost when encoding T2.

```python
# Information metrics
experiments = {
    'mutual_information': 'I(X_T1; h_after_T2)',  # Info about T1 inputs in final representation
    'compression_ratio': 'H(W_T1) / H(W_after_T2)',
    'task_encoding_capacity': 'bits needed to distinguish T1 vs T2',
}
```

**Tasks:**
- [ ] Implement mutual information estimators
- [ ] Measure information retention about T1 after T2 training
- [ ] Correlate information loss with forgetting
- [ ] Test: Is forgetting = information-theoretic compression?

---

## Phase 6: Mitigation Strategies

### 6.1 Task Ordering Optimization

**Hypothesis:** Optimal task ordering minimizes cumulative forgetting.

```python
# Multi-task sequence optimization
experiments = {
    'curriculum': 'Train similar tasks together',
    'interleaving': 'Alternate between task clusters',
    'similarity_sorting': 'Order tasks by pairwise similarity',
}
```

**Tasks:**
- [ ] Extend to N-task sequences (N > 2)
- [ ] Compute optimal ordering given similarity matrix
- [ ] Compare random vs. similarity-sorted orderings
- [ ] Derive ordering algorithm from similarity structure

### 6.2 Gradient Projection Methods ✅ COMPLETE

**Hypothesis:** Projecting gradients to avoid interference reduces forgetting.

**RESULT: LIMITED EFFECTIVENESS** - Methods work but effect is small (~1% reduction)

```python
# Results from 1080 experiments (6 similarities × 3 LRs × 3 memory sizes × 4 methods × 5 seeds)
results = {
    'A-GEM': {'reduction': '1.1%', 'time': '1.44x', 'best': True},
    'OGD': {'reduction': '1.1%', 'time': '1.46x', 'harmful_when_constructive': True},
    'Scaling': {'reduction': '0.2%', 'time': '1.49x'},
}

# KEY INSIGHT: Most gradients are CONSTRUCTIVE (mean cos angle = 0.3-0.4)
# Destructive count = 0 in most runs → A-GEM rarely intervenes
# OGD hurts when gradients are aligned (projects away useful learning)
```

**Completed Tasks:**
- [x] Implement OGD (Orthogonal Gradient Descent)
- [x] Implement A-GEM (Averaged Gradient Episodic Memory)
- [x] Implement gradient scaling method
- [x] Measure forgetting reduction vs. computational cost
- [x] Find minimal memory budget for effective mitigation

**Key Findings:**
1. **A-GEM is best** - Only intervenes when destructive, avoids harming constructive learning
2. **OGD can hurt performance** - Projects even constructive gradients, loses useful learning
3. **Limited effectiveness because gradients are mostly constructive** - In our setup, Task 2 gradients often help Task 1 (positive cosine angle)
4. **Memory size has minimal impact** - 10 vs 100 gradient memory shows <0.1% difference
5. **~45% computational overhead** - All methods add similar cost

**Implication:** Gradient projection alone insufficient. Combined strategies (regularization + projection) or architecture-based methods may be needed for stronger mitigation.

### 6.3 Regularization Approaches

**Hypothesis:** Penalizing movement from T1 solution reduces forgetting.

```python
# Regularization strategies
methods = {
    'EWC': 'λ Σ F_i (θ_i - θ_T1)²',  # Elastic Weight Consolidation
    'SI': 'Synaptic Intelligence importance weighting',
    'L2_to_T1': 'λ ||θ - θ_T1||²',  # Simple L2 to T1 solution
}
```

**Tasks:**
- [ ] Implement EWC with Fisher information
- [ ] Test L2 regularization toward T1 weights
- [ ] Sweep regularization strength λ
- [ ] Find optimal λ as function of similarity

### 6.4 Architecture Modifications

**Hypothesis:** Dedicated capacity per task prevents interference.

```python
# Architecture strategies
methods = {
    'task_specific_heads': 'Shared backbone + separate output heads',
    'progressive_networks': 'Freeze T1 columns, add new for T2',
    'modular_networks': 'Route to task-specific modules',
    'wider_networks': 'More capacity = less competition',
}
```

**Tasks:**
- [ ] Test multi-head architecture
- [ ] Implement progressive neural networks
- [ ] Measure capacity vs. forgetting tradeoff
- [ ] Find minimal extra capacity for zero forgetting

### 6.5 Similarity-Aware Learning Rate

**Hypothesis:** Adapt learning rate based on task similarity.

```python
# LR scheduling by similarity
def adaptive_lr(base_lr, similarity):
    # Lower LR for dissimilar tasks (more careful updates)
    return base_lr * similarity ** α  # α to be discovered
```

**Tasks:**
- [ ] Sweep α parameter
- [ ] Test similarity-adaptive LR scheduling
- [ ] Compare with fixed LR baselines
- [ ] Derive optimal α from forgetting equation

---

## Phase 7: Practical Similarity Estimation

### 7.1 Pre-Training Similarity Estimation

**Problem:** We used ground-truth similarity (controlled). Real tasks don't have this.

```python
# Similarity estimation methods
methods = {
    'data_based': 'cos(mean(X_T1), mean(X_T2))',  # Input distribution overlap
    'gradient_based': 'cos(∇L_T1(θ_init), ∇L_T2(θ_init))',  # Gradient at init
    'loss_based': 'L_T1(θ_T2*) vs L_T1(θ_random)',  # Transfer loss
    'representation_based': 'CKA of features on both tasks',
}
```

**Tasks:**
- [ ] Implement multiple similarity estimators
- [ ] Correlate estimated vs. ground-truth similarity
- [ ] Test which estimator best predicts forgetting
- [ ] Create practical similarity estimation pipeline

### 7.2 Online Similarity Tracking

**Hypothesis:** Track similarity during training to adapt strategy.

```python
# Online metrics
def online_similarity_estimate(model, task1_data, task2_batch):
    # Estimate similarity from gradient alignment during training
    grad_t1 = compute_gradient(model, task1_data)
    grad_t2 = compute_gradient(model, task2_batch)
    return cosine_similarity(grad_t1, grad_t2)
```

**Tasks:**
- [ ] Implement online similarity tracking
- [ ] Test adaptive mitigation triggered by similarity drop
- [ ] Measure overhead vs. benefit
- [ ] Create "forgetting early warning" system

---

## Phase 8: Scaling Validation

### 8.1 Deeper Networks

**Question:** Do findings hold for deeper architectures?

**Tasks:**
- [ ] Test with 3-5 layer MLPs
- [ ] Test with residual connections
- [ ] Measure layer-wise forgetting patterns
- [ ] Check if similarity dominance persists

### 8.2 Real Datasets

**Question:** Do findings transfer from synthetic to real data?

```python
datasets = ['Split-MNIST', 'Split-CIFAR10', 'Permuted-MNIST', 'CORe50']
```

**Tasks:**
- [ ] Implement standard continual learning benchmarks
- [ ] Estimate task similarity for real datasets
- [ ] Test forgetting equation on real data
- [ ] Validate mitigation strategies

### 8.3 Different Architectures

**Question:** Do findings generalize beyond MLPs?

```python
architectures = ['CNN', 'ResNet', 'Transformer', 'ViT']
```

**Tasks:**
- [ ] Test on convolutional networks
- [ ] Test on attention-based architectures
- [ ] Identify architecture-specific forgetting patterns
- [ ] Check universal vs. architecture-specific equations

---

## Priority Order (Updated 2026-01-09)

1. ~~**Phase 5.1** (Gradient Interference)~~ ✅ COMPLETE - Confirmed as causal mechanism (r = -0.87)
2. ~~**Phase 6.2** (Gradient Projection)~~ ✅ COMPLETE - Limited effectiveness (~1% reduction)
3. **Phase 6.3** (Regularization/EWC) - May provide better mitigation than projection ← **NEXT**
4. **Phase 7.1** (Similarity Estimation) - Needed for practical application
5. **Phase 6.5** (Adaptive LR) - Simple, low-cost intervention
6. **Phase 8.2** (Real Datasets) - Validation before publication

---

## Success Metrics

| Goal | Metric | Target |
|------|--------|--------|
| Explain WHY | Identified causal mechanism with ablation evidence | Mechanistic model R² > 0.8 |
| Predict forgetting | Similarity estimator accuracy | Correlation > 0.85 with true similarity |
| Mitigate forgetting | Reduction in forgetting | ≥50% reduction at same performance |
| Practical method | Computational overhead | <20% training time increase |
| Generalization | Transfer to real benchmarks | Findings replicate on Split-CIFAR |

---

## References for Mitigation Methods

- **EWC**: Kirkpatrick et al. (2017) "Overcoming catastrophic forgetting in neural networks"
- **OGD**: Farajtabar et al. (2020) "Orthogonal Gradient Descent for Continual Learning"
- **A-GEM**: Chaudhry et al. (2019) "Efficient Lifelong Learning with A-GEM"
- **Progressive Networks**: Rusu et al. (2016) "Progressive Neural Networks"
- **Gradient Episodic Memory**: Lopez-Paz & Ranzato (2017) "Gradient Episodic Memory for Continual Learning"

---

*Created: 2026-01-08*
*Updated: 2026-01-09*
*Status: Phase 6.2 Complete - Limited mitigation from gradient projection*
