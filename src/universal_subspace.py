"""
Universal Subspace Analysis for Phase 3.

Implements the Universal Subspace hypothesis from Kaushik et al. (2024):
- Networks converge to a low-dimensional "universal subspace" of weight space
- Deviation from this subspace correlates with forgetting
- The lazy-rich transition corresponds to leaving the universal subspace

Key concepts:
- Universal Subspace: Principal subspace of trained weight matrices
- Subspace Deviation: ||θ_⊥|| / ||θ_∥|| ratio (perpendicular to parallel)
- Transition Boundary: Where deviation exceeds threshold
"""

import torch
import numpy as np
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class SubspaceAnalysis:
    """Results from universal subspace analysis."""

    # Subspace properties
    subspace_dim: int
    total_variance_explained: float
    singular_values: np.ndarray

    # Weight projections
    parallel_norm: float      # ||θ_∥|| - component in subspace
    perpendicular_norm: float # ||θ_⊥|| - component outside subspace
    deviation_ratio: float    # ||θ_⊥|| / ||θ_∥||

    # Alignment metrics
    alignment_with_subspace: float  # Cosine similarity with subspace
    projection_loss: float          # 1 - (||projection||² / ||original||²)

    def to_dict(self) -> Dict:
        return {
            'subspace_dim': self.subspace_dim,
            'total_variance_explained': self.total_variance_explained,
            'parallel_norm': self.parallel_norm,
            'perpendicular_norm': self.perpendicular_norm,
            'deviation_ratio': self.deviation_ratio,
            'alignment_with_subspace': self.alignment_with_subspace,
            'projection_loss': self.projection_loss,
        }


class UniversalSubspace:
    """
    Universal Subspace extractor using SVD/HOSVD.

    The universal subspace is computed from a collection of trained weights
    and represents the "typical" directions that learning takes.
    """

    def __init__(
        self,
        target_variance: float = 0.95,
        max_dim: Optional[int] = None,
        device: str = "cpu"
    ):
        """
        Args:
            target_variance: Variance to explain (determines subspace dim)
            max_dim: Maximum subspace dimension
            device: Torch device
        """
        self.target_variance = target_variance
        self.max_dim = max_dim
        self.device = device

        # Computed subspace
        self.U: Optional[torch.Tensor] = None  # Left singular vectors (basis)
        self.S: Optional[torch.Tensor] = None  # Singular values
        self.subspace_dim: int = 0
        self.is_fitted: bool = False

        # Statistics
        self.mean_weights: Optional[torch.Tensor] = None
        self.n_samples: int = 0

    def fit(self, weight_matrices: List[torch.Tensor]) -> 'UniversalSubspace':
        """
        Fit universal subspace from collection of weight matrices.

        Args:
            weight_matrices: List of weight tensors (can be different shapes)

        Returns:
            self for chaining
        """
        # Flatten and stack all weights
        flattened = []
        for W in weight_matrices:
            flattened.append(W.flatten().to(self.device))

        # Pad to same length if needed
        max_len = max(w.shape[0] for w in flattened)
        padded = []
        for w in flattened:
            if w.shape[0] < max_len:
                w = torch.cat([w, torch.zeros(max_len - w.shape[0], device=self.device)])
            padded.append(w)

        # Stack into matrix (samples × features)
        W_matrix = torch.stack(padded, dim=0)
        self.n_samples = W_matrix.shape[0]

        # Center the data
        self.mean_weights = W_matrix.mean(dim=0)
        W_centered = W_matrix - self.mean_weights

        # SVD
        U, S, Vh = torch.linalg.svd(W_centered, full_matrices=False)

        # Determine subspace dimension from variance explained
        total_var = (S ** 2).sum()
        cumulative_var = torch.cumsum(S ** 2, dim=0) / total_var

        # Find dimension for target variance
        dim_mask = cumulative_var >= self.target_variance
        if dim_mask.any():
            self.subspace_dim = dim_mask.nonzero()[0, 0].item() + 1
        else:
            self.subspace_dim = len(S)

        # Apply max_dim constraint
        if self.max_dim is not None:
            self.subspace_dim = min(self.subspace_dim, self.max_dim)

        # Store basis (transpose Vh to get column vectors)
        self.U = Vh[:self.subspace_dim, :].T  # (features × subspace_dim)
        self.S = S[:self.subspace_dim]
        self.is_fitted = True

        return self

    def project(self, W: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project weight matrix onto universal subspace.

        Args:
            W: Weight tensor

        Returns:
            (parallel_component, perpendicular_component)
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before project()")

        # Flatten
        w = W.flatten().to(self.device)

        # Pad if needed
        if w.shape[0] < self.U.shape[0]:
            w = torch.cat([w, torch.zeros(self.U.shape[0] - w.shape[0], device=self.device)])
        elif w.shape[0] > self.U.shape[0]:
            w = w[:self.U.shape[0]]

        # Center
        w_centered = w - self.mean_weights

        # Project: w_∥ = U @ (U^T @ w)
        coeffs = self.U.T @ w_centered  # (subspace_dim,)
        w_parallel = self.U @ coeffs    # (features,)
        w_perpendicular = w_centered - w_parallel

        return w_parallel, w_perpendicular

    def analyze(self, W: torch.Tensor) -> SubspaceAnalysis:
        """
        Analyze weight matrix relative to universal subspace.

        Args:
            W: Weight tensor

        Returns:
            SubspaceAnalysis with all metrics
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before analyze()")

        w_parallel, w_perpendicular = self.project(W)

        parallel_norm = w_parallel.norm().item()
        perpendicular_norm = w_perpendicular.norm().item()

        # Deviation ratio (key metric for lazy-rich)
        if parallel_norm > 1e-10:
            deviation_ratio = perpendicular_norm / parallel_norm
        else:
            deviation_ratio = float('inf') if perpendicular_norm > 1e-10 else 0.0

        # Alignment (cosine similarity)
        w_flat = W.flatten().to(self.device)
        if w_flat.shape[0] < self.mean_weights.shape[0]:
            w_flat = torch.cat([w_flat, torch.zeros(self.mean_weights.shape[0] - w_flat.shape[0], device=self.device)])
        elif w_flat.shape[0] > self.mean_weights.shape[0]:
            w_flat = w_flat[:self.mean_weights.shape[0]]

        w_centered = w_flat - self.mean_weights
        total_norm = w_centered.norm().item()

        if total_norm > 1e-10:
            alignment = parallel_norm / total_norm
            projection_loss = 1 - (parallel_norm ** 2) / (total_norm ** 2)
        else:
            alignment = 0.0
            projection_loss = 1.0

        # Variance explained
        variance_explained = (self.S ** 2).sum().item() / max(
            (self.S ** 2).sum().item() + 1e-10, 1e-10
        )

        return SubspaceAnalysis(
            subspace_dim=self.subspace_dim,
            total_variance_explained=variance_explained,
            singular_values=self.S.cpu().numpy(),
            parallel_norm=parallel_norm,
            perpendicular_norm=perpendicular_norm,
            deviation_ratio=deviation_ratio,
            alignment_with_subspace=alignment,
            projection_loss=projection_loss,
        )


def compute_weight_trajectory_subspace(
    weight_history: List[torch.Tensor],
    target_variance: float = 0.95
) -> Tuple[UniversalSubspace, List[SubspaceAnalysis]]:
    """
    Compute subspace from weight trajectory and analyze each point.

    Args:
        weight_history: List of weight tensors during training
        target_variance: Variance to explain

    Returns:
        (fitted_subspace, list_of_analyses)
    """
    # Fit subspace to all weights
    subspace = UniversalSubspace(target_variance=target_variance)
    subspace.fit(weight_history)

    # Analyze each weight
    analyses = [subspace.analyze(W) for W in weight_history]

    return subspace, analyses


def compute_task_transition_subspace(
    weights_task1: List[torch.Tensor],
    weights_task2: List[torch.Tensor],
    target_variance: float = 0.95
) -> Dict:
    """
    Analyze subspace dynamics during task transition.

    Fits subspace on Task 1 final weights, then measures
    deviation during Task 2 training.

    Args:
        weights_task1: Weight trajectory during Task 1
        weights_task2: Weight trajectory during Task 2
        target_variance: Variance to explain

    Returns:
        Dict with analysis results
    """
    # Fit subspace to Task 1 trajectory
    subspace = UniversalSubspace(target_variance=target_variance)
    subspace.fit(weights_task1)

    # Analyze Task 1 trajectory
    task1_analyses = [subspace.analyze(W) for W in weights_task1]

    # Analyze Task 2 trajectory (deviation from Task 1 subspace)
    task2_analyses = [subspace.analyze(W) for W in weights_task2]

    # Extract key metrics
    t1_deviations = [a.deviation_ratio for a in task1_analyses]
    t2_deviations = [a.deviation_ratio for a in task2_analyses]

    return {
        'subspace_dim': subspace.subspace_dim,
        'task1_final_deviation': t1_deviations[-1] if t1_deviations else 0.0,
        'task2_max_deviation': max(t2_deviations) if t2_deviations else 0.0,
        'task2_final_deviation': t2_deviations[-1] if t2_deviations else 0.0,
        'deviation_increase': (
            (t2_deviations[-1] - t1_deviations[-1]) if t1_deviations and t2_deviations else 0.0
        ),
        'task1_trajectory': task1_analyses,
        'task2_trajectory': task2_analyses,
    }


def compute_transition_boundary(
    deviations: np.ndarray,
    forgetting: np.ndarray,
    threshold_percentile: float = 90
) -> Dict:
    """
    Compute transition boundary from deviation-forgetting data.

    The boundary is where deviation predicts high forgetting.

    Args:
        deviations: Array of deviation ratios
        forgetting: Array of forgetting values
        threshold_percentile: Percentile to define "high" forgetting

    Returns:
        Dict with boundary analysis
    """
    # Define high forgetting threshold
    forgetting_threshold = np.percentile(forgetting, threshold_percentile)
    high_forgetting_mask = forgetting >= forgetting_threshold

    # Find deviation threshold that best separates
    sorted_deviations = np.sort(deviations)
    best_threshold = 0.0
    best_separation = 0.0

    for d in sorted_deviations:
        # Split by deviation
        low_dev = forgetting[deviations < d]
        high_dev = forgetting[deviations >= d]

        if len(low_dev) > 0 and len(high_dev) > 0:
            # Separation = difference in means
            separation = high_dev.mean() - low_dev.mean()
            if separation > best_separation:
                best_separation = separation
                best_threshold = d

    # Classification accuracy
    predicted_high = deviations >= best_threshold
    accuracy = (predicted_high == high_forgetting_mask).mean()

    # Correlation
    correlation = np.corrcoef(deviations, forgetting)[0, 1]

    return {
        'deviation_threshold': best_threshold,
        'forgetting_threshold': forgetting_threshold,
        'separation': best_separation,
        'classification_accuracy': accuracy,
        'correlation': correlation,
    }


def fit_transition_equation(
    deviations: np.ndarray,
    forgetting: np.ndarray,
    flr: np.ndarray,
    learning_rate: np.ndarray,
    similarity: np.ndarray
) -> Dict:
    """
    Fit equation for transition boundary using multiple predictors.

    Tests several functional forms:
    1. Linear: F = a*deviation + b
    2. With FLR: F = a*deviation + b*FLR + c
    3. Full: F = a*deviation + b*FLR + c*LR + d*(1-s) + e
    4. Threshold: F = baseline + penalty * (deviation > threshold)

    Args:
        deviations: Subspace deviation ratios
        forgetting: Forgetting values
        flr: Feature learning rates
        learning_rate: Learning rates
        similarity: Task similarities

    Returns:
        Dict with fitted equations and R² values
    """
    from scipy.optimize import curve_fit

    results = {}

    # Clean data
    mask = ~(np.isnan(deviations) | np.isnan(forgetting) |
             np.isnan(flr) | np.isnan(learning_rate) | np.isnan(similarity))
    dev = deviations[mask]
    forg = forgetting[mask]
    f = flr[mask]
    lr = learning_rate[mask]
    s = similarity[mask]

    if len(dev) < 10:
        return {'error': 'Insufficient clean data'}

    # Model 1: Linear deviation
    def linear(X, a, b):
        return a * X + b

    try:
        popt, _ = curve_fit(linear, dev, forg, p0=[1.0, 0.0], maxfev=5000)
        pred = linear(dev, *popt)
        ss_res = ((forg - pred) ** 2).sum()
        ss_tot = ((forg - forg.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        results['linear_deviation'] = {
            'equation': f'F = {popt[0]:.4f} * deviation + {popt[1]:.4f}',
            'coefficients': {'a': popt[0], 'b': popt[1]},
            'r_squared': r2
        }
    except Exception as e:
        results['linear_deviation'] = {'error': str(e)}

    # Model 2: Deviation + FLR
    def deviation_flr(X, a, b, c):
        dev, flr = X
        return a * dev + b * flr + c

    try:
        popt, _ = curve_fit(deviation_flr, (dev, f), forg, p0=[1.0, 1.0, 0.0], maxfev=5000)
        pred = deviation_flr((dev, f), *popt)
        ss_res = ((forg - pred) ** 2).sum()
        ss_tot = ((forg - forg.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        results['deviation_flr'] = {
            'equation': f'F = {popt[0]:.4f} * deviation + {popt[1]:.4f} * FLR + {popt[2]:.4f}',
            'coefficients': {'a': popt[0], 'b': popt[1], 'c': popt[2]},
            'r_squared': r2
        }
    except Exception as e:
        results['deviation_flr'] = {'error': str(e)}

    # Model 3: Full model
    def full_model(X, a, b, c, d, e):
        dev, flr, lr, s = X
        return a * dev + b * flr + c * lr + d * (1 - s) + e

    try:
        popt, _ = curve_fit(full_model, (dev, f, lr, s), forg,
                           p0=[0.5, 0.5, 1.0, 0.5, 0.0], maxfev=5000)
        pred = full_model((dev, f, lr, s), *popt)
        ss_res = ((forg - pred) ** 2).sum()
        ss_tot = ((forg - forg.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        results['full_model'] = {
            'equation': (f'F = {popt[0]:.4f} * deviation + {popt[1]:.4f} * FLR + '
                        f'{popt[2]:.4f} * LR + {popt[3]:.4f} * (1-s) + {popt[4]:.4f}'),
            'coefficients': {'a': popt[0], 'b': popt[1], 'c': popt[2], 'd': popt[3], 'e': popt[4]},
            'r_squared': r2
        }
    except Exception as e:
        results['full_model'] = {'error': str(e)}

    # Model 4: Threshold model
    # Find optimal threshold first
    boundary = compute_transition_boundary(dev, forg)
    threshold = boundary['deviation_threshold']

    def threshold_model(X, baseline, penalty):
        return baseline + penalty * (X >= threshold).astype(float)

    try:
        popt, _ = curve_fit(threshold_model, dev, forg, p0=[0.0, 0.3], maxfev=5000)
        pred = threshold_model(dev, *popt)
        ss_res = ((forg - pred) ** 2).sum()
        ss_tot = ((forg - forg.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        results['threshold_model'] = {
            'equation': f'F = {popt[0]:.4f} + {popt[1]:.4f} * (deviation ≥ {threshold:.4f})',
            'coefficients': {'baseline': popt[0], 'penalty': popt[1], 'threshold': threshold},
            'r_squared': r2
        }
    except Exception as e:
        results['threshold_model'] = {'error': str(e)}

    return results
