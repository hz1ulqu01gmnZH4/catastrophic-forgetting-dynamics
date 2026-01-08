# Experiment Plan: Symbolic Regression for Deep Learning Dynamics

## Discovering Equations for Catastrophic Forgetting and the Lazy-Rich Transition

**Date:** 2026-01-07
**Status:** Proposal

---

## 1. Executive Summary

This experiment aims to use **symbolic regression (SR)** to discover closed-form mathematical equations governing catastrophic forgetting dynamics in neural networks. While extensive analytical work exists for linear models, **no one has systematically applied SR** to discover equations from training trajectories. This represents a genuine research gap.

A key theoretical target is the **lazy-rich transition boundary**—the critical point where networks shift from low-forgetting (lazy) to high-forgetting (rich/feature-learning) regimes. The recent **Universal Subspace Hypothesis** (Kaushik et al., 2024) provides a promising geometric framework for characterizing this transition.

---

## 2. Research Background Summary

### 2.1 Analytical Work on Catastrophic Forgetting

| Paper | Key Contribution | Limitation |
|-------|------------------|------------|
| **Evron et al. (2022)** | Closed-form forgetting in linear regression | Linear only |
| **Asanuma et al. (2021)** | Statistical mechanics of teacher-student forgetting | Single-layer, Gaussian data |
| **Goldfarb et al. (2024)** | Joint effect of task similarity + overparameterization | Linear, two-task |
| **Graldi et al. (2025)** | Lazy-rich transition in continual learning | Mean-field limits |
| **Mori et al. (2024)** | Optimal protocols via control theory | Requires known dynamics |

**Key insight from Graldi et al.:**
> "High feature learning is only beneficial with highly similar tasks... We identify a transition modulated by task similarity where the model exits an effectively lazy regime with low forgetting to enter a rich regime with significant forgetting."

### 2.2 The Gap

Existing work derives equations **from first principles** (statistical mechanics, mean-field theory). Nobody has:
1. Used SR to **discover** these equations from data
2. Found equations for the **lazy-rich transition boundary**
3. Extended analytical results to **nonlinear/deeper networks** via SR

### 2.3 Universal Subspace Hypothesis (Kaushik et al., 2024)

**arXiv:2512.05117** demonstrates:
- Neural networks trained on diverse tasks converge to **shared low-dimensional parametric subspaces**
- This is architecture-specific and layer-wise
- 500+ models (LoRAs, ViTs, LLMs) show same phenomenon

**Relevance to lazy-rich transition:**
- **Lazy regime**: Network stays within pre-existing universal subspace
- **Rich regime**: Network learns new directions, potentially **breaking** the universal subspace
- The transition boundary may be characterizable as: **when does the network leave the universal subspace?**

**Hypothesis:** The forgetting-transition boundary can be expressed as a function of:
$$\text{Forgetting}(\theta) \propto f\left(\frac{\|\Delta\theta_{\perp}\|}{\|\Delta\theta_{\parallel}\|}\right)$$

where $\Delta\theta_{\parallel}$ is weight change within universal subspace and $\Delta\theta_{\perp}$ is orthogonal to it.

---

## 3. Latest Symbolic Regression Methods

### 3.1 Classical Methods

| Tool | Type | Strengths | Best For |
|------|------|-----------|----------|
| **PySR** (Cranmer 2023) | Evolutionary (Julia) | Fast, distributed, SIMD | Production use |
| **AI Feynman 2.0** (Tegmark) | Physics-inspired + NN | Symmetry detection, compositionality | Physics equations |
| **Operon** | Genetic programming | Very fast C++ | Large-scale |

### 3.2 Neural/Transformer Methods

| Tool | Type | Strengths | Best For |
|------|------|-----------|----------|
| **SymFormer** (2022) | Transformer | Fast inference, constants included | Quick exploration |
| **E2E** (Kamienny 2022) | End-to-end transformer | Direct constant prediction | Benchmark problems |
| **Deep Symbolic Regression** (Petersen 2019) | RNN + policy gradient | Risk-seeking optimization | Novel expressions |

### 3.3 LLM-Based Methods (State-of-the-Art 2024-2025)

| Tool | Type | Strengths | Best For |
|------|------|-----------|----------|
| **LLM-SR** (Shojaee 2024) | LLM + evolutionary | Scientific priors, code generation | Scientific discovery |
| **LaSR** (Grayeli 2024) | LLM + concept library | Semantic guidance, interpretability | Complex systems |
| **EGG-SR** (Jiang 2025) | Equality graphs + LLM | Handles symbolic equivalence | Avoiding redundancy |
| **DrSR** (Wang 2025) | Dual reasoning (data + reflection) | Data-driven + feedback loop | Robust discovery |
| **LLM-Meta-SR** (Zhang 2025) | Meta-learning selection operators | Evolves the SR algorithm itself | Best overall performance |

### 3.4 Hybrid Methods

| Tool | Type | Strengths |
|------|------|-----------|
| **T-SHRED** (2025) | Transformer + SINDy attention | Sparse identification in latent space |
| **KAN + LLM** (Harvey 2025) | Kolmogorov-Arnold Networks + vision LLM | Univariate decomposition |

**Recommendation for this experiment:**
- **Primary:** PySR (robust, fast, well-documented)
- **Secondary:** LLM-SR or LaSR (for incorporating physics priors about learning dynamics)
- **Validation:** Compare against analytical predictions

---

## 4. Experiment Design

### 4.1 Phase 1: Linear Network Baseline (Validation)

**Goal:** Reproduce known analytical results via SR to validate methodology.

#### Setup
```python
# Two-task continual linear regression
# Teacher: y = W*x where W* ∈ R^{d_out × d_in}
# Student: y = Wx trained via SGD

d_in, d_out = 100, 10  # Input/output dimensions
n_widths = [50, 100, 200, 500, 1000, 2000]  # Overparameterization sweep
n_similarities = np.linspace(0, 1, 11)  # Task similarity (0=orthogonal, 1=identical)
n_steps = [100, 500, 1000, 5000]  # Training steps per task
learning_rates = [0.001, 0.01, 0.1]
```

#### Measurements
```python
# For each (width, similarity, steps, lr) combination:
measurements = {
    'forgetting': loss_T1_after_T2 - loss_T1_after_T1,
    'forward_transfer': loss_T2_init - loss_T2_random_init,
    'backward_transfer': loss_T1_after_T2 - loss_T1_joint,
    'weight_distance': ||W_after_T2 - W_after_T1||,
    'effective_rank': nuclear_norm(W) / spectral_norm(W),
}
```

#### SR Target
Find: $\text{Forgetting}(n, s, t, \eta) = ?$

**Known analytical form (Goldfarb et al.):**
$$\mathbb{E}[\text{Forgetting}] \approx \frac{c_1}{n} \cdot g(s) + c_2 \cdot \eta \cdot t$$

where $g(s)$ peaks at intermediate similarity $s \approx 0.5$ for overparameterized models.

**Success criterion:** SR rediscovers this functional form.

---

### 4.2 Phase 2: Nonlinear Single-Layer Networks

**Goal:** Extend to nonlinear activations where analytical solutions don't exist.

#### Setup
```python
# Single hidden layer with nonlinear activation
# y = V * σ(W * x)

activations = ['relu', 'tanh', 'gelu', 'sigmoid']
hidden_widths = [64, 128, 256, 512, 1024]
```

#### New Measurements
```python
measurements.update({
    'feature_learning_rate': ||W_t - W_0|| / ||W_0||,  # How much features changed
    'ntk_alignment': cosine_sim(NTK_init, NTK_final),  # Deviation from lazy regime
    'activation_sparsity': mean(σ(Wx) == 0),  # For ReLU
    'hessian_trace': tr(H),  # Curvature
})
```

#### SR Target
Find: $\text{Forgetting}(n, s, t, \eta, \sigma, \text{feature\_learning\_rate}) = ?$

**Hypothesis:** Feature learning rate mediates the transition:
$$\text{Forgetting} \approx f_{\text{lazy}}(\cdot) + \lambda \cdot f_{\text{rich}}(\cdot) \cdot \mathbb{1}[\text{FLR} > \tau]$$

---

### 4.3 Phase 3: Universal Subspace Analysis

**Goal:** Test whether forgetting correlates with deviation from universal subspace.

#### Setup (Following Kaushik et al.)
```python
# Train N models on different tasks
# Extract layer-wise weight matrices
# Compute universal subspace via HOSVD

n_models = 50  # Different tasks
architecture = 'MLP'  # Start simple
layers_to_analyze = ['layer1', 'layer2', 'output']

# For each model after each task:
for model in models:
    for task_idx in range(n_tasks):
        train(model, task_idx)

        # Measure subspace alignment
        for layer in layers_to_analyze:
            W = model.get_weights(layer)

            # Project onto pre-computed universal subspace
            W_parallel = project_onto_subspace(W, universal_subspace[layer])
            W_perp = W - W_parallel

            subspace_deviation = ||W_perp|| / ||W||
```

#### SR Target
Find: $\text{Forgetting}(\text{subspace\_deviation}, s, n, ...) = ?$

**Hypothesis:**
$$\text{Forgetting} \approx \alpha \cdot (\text{subspace\_deviation})^\beta + \gamma$$

If confirmed, this provides a **geometric characterization** of catastrophic forgetting.

---

### 4.4 Phase 4: Lazy-Rich Transition Boundary

**Goal:** Discover the equation for the critical transition point.

#### Key Variables
```python
# Candidate predictors for transition boundary
predictors = {
    'width_depth_ratio': width / depth,
    'lr_width_product': lr * sqrt(width),  # NTK scaling
    'task_similarity': s,
    'init_scale': ||W_0||,
    'data_complexity': intrinsic_dim(X),
    'batch_size_normalized': batch_size / n_samples,
}
```

#### Binary Classification for Transition
```python
# Label each training run as 'lazy' or 'rich' based on:
def classify_regime(run):
    # Criterion 1: NTK change
    if ntk_change(run) < threshold_1:
        return 'lazy'
    # Criterion 2: Feature learning rate
    if feature_learning_rate(run) < threshold_2:
        return 'lazy'
    # Criterion 3: Forgetting pattern
    if forgetting_peaks_at_intermediate_similarity(run):
        return 'rich'  # Rich regime signature
    return 'lazy'
```

#### SR Target
Find the **decision boundary**:
$$\text{Regime} = \text{sign}(f(\text{width}, \text{lr}, \text{similarity}, ...))$$

This is **symbolic classification** rather than regression—use SR on the decision function.

---

## 5. Implementation Plan

### 5.1 Tools and Dependencies

```bash
# Core symbolic regression
pip install pysr  # Primary SR tool
pip install sympy  # Symbolic manipulation

# Neural network training
pip install torch
pip install pytorch-lightning

# Analysis
pip install scikit-learn
pip install umap-learn  # For subspace visualization

# LLM-based SR (optional)
pip install openai  # For LLM-SR experiments
git clone https://github.com/deep-symbolic-mathematics/LLM-SR
```

### 5.2 Data Generation Pipeline

```python
# pseudo-code for data generation

def generate_forgetting_dataset(config):
    """Generate dataset of (hyperparams, measurements) pairs."""

    results = []

    for width in config.widths:
        for similarity in config.similarities:
            for lr in config.learning_rates:
                for seed in range(config.n_seeds):

                    # Create teacher-student setup
                    teacher1, teacher2 = create_teachers(similarity)
                    student = create_student(width)

                    # Train on task 1
                    train(student, teacher1, n_steps=config.steps)
                    loss_T1_after_T1 = evaluate(student, teacher1)
                    W_after_T1 = student.weights.clone()

                    # Train on task 2
                    train(student, teacher2, n_steps=config.steps)
                    loss_T1_after_T2 = evaluate(student, teacher1)
                    loss_T2_after_T2 = evaluate(student, teacher2)
                    W_after_T2 = student.weights.clone()

                    # Compute measurements
                    result = {
                        # Inputs
                        'width': width,
                        'similarity': similarity,
                        'lr': lr,
                        'steps': config.steps,

                        # Outputs
                        'forgetting': loss_T1_after_T2 - loss_T1_after_T1,
                        'weight_change': (W_after_T2 - W_after_T1).norm(),
                        'feature_learning_rate': compute_flr(student),
                        ...
                    }
                    results.append(result)

    return pd.DataFrame(results)
```

### 5.3 Symbolic Regression Pipeline

```python
from pysr import PySRRegressor

def run_symbolic_regression(df, target, predictors):
    """Run SR to find equation for target."""

    X = df[predictors].values
    y = df[target].values

    model = PySRRegressor(
        niterations=100,
        binary_operators=["+", "-", "*", "/", "^"],
        unary_operators=["exp", "log", "sqrt", "abs"],

        # Physics-inspired constraints
        constraints={
            "^": (-1, 1),  # Limit exponent complexity
        },

        # Complexity control
        maxsize=30,
        parsimony=0.001,  # Favor simpler expressions

        # Performance
        procs=8,
        multithreading=True,
    )

    model.fit(X, y)

    return model.equations_  # Returns Pareto front of accuracy vs complexity
```

### 5.4 Validation Against Analytical Predictions

```python
def validate_against_theory(discovered_eq, analytical_eq, test_data):
    """Compare SR-discovered equation against analytical prediction."""

    # Symbolic comparison
    from sympy import simplify, expand
    symbolic_match = simplify(discovered_eq - analytical_eq) == 0

    # Numerical comparison
    y_discovered = eval_equation(discovered_eq, test_data)
    y_analytical = eval_equation(analytical_eq, test_data)

    mse = mean_squared_error(y_discovered, y_analytical)
    correlation = pearsonr(y_discovered, y_analytical)

    return {
        'symbolic_match': symbolic_match,
        'mse': mse,
        'correlation': correlation,
    }
```

---

## 6. Expected Outcomes

### 6.1 Validation Phase
- SR should rediscover known linear model equations
- Establishes methodology validity

### 6.2 Extension Phase
- Discover equations for nonlinear activations
- Quantify how activation function affects forgetting dynamics
- Expected form: corrections to linear equations

### 6.3 Universal Subspace Phase
- **Novel contribution**: Link forgetting to geometric deviation from universal subspace
- Expected discovery: $\text{Forgetting} \propto (\text{subspace\_deviation})^\beta$

### 6.4 Transition Boundary Phase
- **Key contribution**: First closed-form expression for lazy-rich transition
- Enable prediction of when a network will exhibit catastrophic forgetting
- Practical implications for continual learning system design

---

## 7. Potential Challenges and Mitigations

| Challenge | Mitigation |
|-----------|------------|
| High dimensionality | Start with 2-3 predictors, add incrementally |
| Noise in measurements | Average over multiple seeds, use robust SR |
| SR overfitting | Use Pareto front, validate on held-out data |
| Computational cost | Start with small networks, scale up |
| No ground truth for deep networks | Compare against empirical patterns, not equations |

---

## 8. Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Phase 1: Linear baseline | 2 weeks | Validated methodology, reproduced known results |
| Phase 2: Nonlinear extension | 3 weeks | Equations for ReLU/tanh networks |
| Phase 3: Universal subspace | 3 weeks | Geometric characterization of forgetting |
| Phase 4: Transition boundary | 4 weeks | Lazy-rich transition equation |
| Write-up | 2 weeks | Paper draft |

**Total: ~14 weeks**

---

## 9. Success Metrics

1. **Validation**: SR rediscovers ≥80% of known linear model structure
2. **Extension**: Discovered equations predict forgetting with R² > 0.9 on test data
3. **Novelty**: At least one equation form not previously reported in literature
4. **Interpretability**: All discovered equations have ≤10 terms
5. **Generalization**: Equations transfer to held-out architectures/tasks

---

## 10. References

### Catastrophic Forgetting Theory
- McCloskey & Cohen (1989). Catastrophic interference in connectionist networks.
- Evron et al. (2022). How catastrophic can catastrophic forgetting be in linear regression?
- Asanuma et al. (2021). Statistical mechanical analysis of catastrophic forgetting.
- Goldfarb et al. (2024). Joint effect of task similarity and overparameterization.
- Graldi et al. (2025). The importance of being lazy: Scaling limits of continual learning.
- Mori et al. (2024). Optimal protocols for continual learning via statistical physics.

### Universal Subspace
- Kaushik et al. (2024). The Universal Weight Subspace Hypothesis. arXiv:2512.05117

### Symbolic Regression
- Cranmer (2023). PySR: Interpretable machine learning for science.
- Udrescu & Tegmark (2020). AI Feynman 2.0.
- Shojaee et al. (2024). LLM-SR: Scientific equation discovery via programming with LLMs.
- Grayeli et al. (2024). LaSR: Symbolic regression with a learned concept library.

### Scaling Laws for SR
- Otte, Franke & Hutter (2025). Towards scaling laws for symbolic regression.

---

## Appendix A: Code Snippets

### A.1 Task Similarity Generation

```python
import torch

def create_orthogonal_tasks(d_in, d_out, similarity):
    """Create two teacher weight matrices with specified similarity."""

    # First teacher: random orthogonal
    W1 = torch.randn(d_out, d_in)
    W1 = W1 / W1.norm()

    # Second teacher: interpolate between W1 and random orthogonal
    W_random = torch.randn(d_out, d_in)
    W_random = W_random - (W_random @ W1.T @ W1)  # Orthogonalize
    W_random = W_random / W_random.norm()

    W2 = similarity * W1 + (1 - similarity) * W_random
    W2 = W2 / W2.norm()

    return W1, W2
```

### A.2 Feature Learning Rate Computation

```python
def compute_feature_learning_rate(model, X_probe):
    """Measure how much internal representations changed."""

    # Get activations at init vs current
    with torch.no_grad():
        h_init = model.get_hidden(X_probe, use_init_weights=True)
        h_current = model.get_hidden(X_probe)

    # Centered Kernel Alignment
    K_init = h_init @ h_init.T
    K_current = h_current @ h_current.T

    flr = 1 - cka(K_init, K_current)
    return flr
```

### A.3 Universal Subspace Projection

```python
def compute_universal_subspace(weight_matrices, variance_threshold=0.95):
    """Extract universal subspace via HOSVD."""

    # Stack weight matrices: shape (n_models, d_out, d_in)
    X = torch.stack(weight_matrices)

    # Zero-center
    mu = X.mean(dim=0)
    X_centered = X - mu

    # Mode-1 unfolding and SVD
    X_unfolded = X_centered.reshape(X.shape[0], -1)
    U, S, Vh = torch.linalg.svd(X_unfolded, full_matrices=False)

    # Select components explaining threshold variance
    cumvar = (S ** 2).cumsum(0) / (S ** 2).sum()
    n_components = (cumvar < variance_threshold).sum() + 1

    return {
        'basis': Vh[:n_components],
        'singular_values': S[:n_components],
        'mean': mu,
        'n_components': n_components,
    }

def project_onto_subspace(W, subspace):
    """Project weight matrix onto universal subspace."""

    W_flat = (W - subspace['mean']).flatten()
    coeffs = W_flat @ subspace['basis'].T
    W_parallel = (coeffs @ subspace['basis']).reshape(W.shape) + subspace['mean']

    return W_parallel
```

---

## Appendix B: Expected Equation Forms

Based on analytical literature, SR should discover equations resembling:

### Linear Model Forgetting
$$\text{Forgetting} \approx \frac{a}{n^b} \cdot s^c \cdot (1-s)^d + e \cdot \eta \cdot t$$

### Nonlinear Corrections
$$\text{Forgetting}_{\text{nonlinear}} \approx \text{Forgetting}_{\text{linear}} + f(\text{activation}) \cdot g(\text{FLR})$$

### Transition Boundary
$$\tau_{\text{transition}} \approx \frac{\alpha \cdot \sqrt{n}}{\eta \cdot t} \cdot h(s)$$

where crossing $\tau$ switches from lazy to rich regime.

### Subspace-Based Characterization
$$\text{Forgetting} \approx \beta_0 + \beta_1 \cdot \left(\frac{\|\theta_\perp\|}{\|\theta_\parallel\|}\right)^\gamma$$
