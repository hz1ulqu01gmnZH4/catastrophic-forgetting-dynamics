"""
Trajectory Analysis for Phase 4.

Tests the hypothesis: Forgetting ∝ max_t ||θ_⊥(t)|| / ||θ_∥(t)||

Key trajectory metrics:
- Max deviation: Peak excursion from subspace
- Path integral: Cumulative deviation over training
- Deviation velocity: Rate of change of deviation
- Trajectory curvature: Second derivative of path
- Area under curve: Total deviation exposure
- Deviation momentum: deviation × velocity
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy import integrate
from scipy.ndimage import gaussian_filter1d


@dataclass
class TrajectoryMetrics:
    """Comprehensive trajectory metrics for a training run."""

    # Basic deviation metrics
    initial_deviation: float
    final_deviation: float
    max_deviation: float
    min_deviation: float
    mean_deviation: float

    # Path metrics
    path_length: float              # Total distance traveled in weight space
    path_integral_deviation: float  # ∫ deviation(t) dt
    area_under_curve: float         # Same as path integral, normalized by time

    # Velocity metrics
    max_velocity: float             # max |d(deviation)/dt|
    mean_velocity: float            # mean |d(deviation)/dt|
    final_velocity: float           # velocity at end of training

    # Acceleration metrics
    max_acceleration: float         # max |d²(deviation)/dt²|
    mean_acceleration: float        # mean |d²(deviation)/dt²|

    # Curvature metrics
    trajectory_curvature: float     # Average curvature of path
    max_curvature: float            # Peak curvature

    # Momentum metrics (deviation × velocity)
    max_momentum: float
    mean_momentum: float

    # Excursion metrics
    excursion_duration: float       # Time spent above mean deviation
    excursion_intensity: float      # max_deviation - initial_deviation
    return_to_baseline: float       # |final - initial| deviation

    # Volatility
    deviation_std: float            # Standard deviation of deviation trajectory
    deviation_range: float          # max - min

    def to_dict(self) -> Dict:
        return {
            'initial_deviation': self.initial_deviation,
            'final_deviation': self.final_deviation,
            'max_deviation': self.max_deviation,
            'min_deviation': self.min_deviation,
            'mean_deviation': self.mean_deviation,
            'path_length': self.path_length,
            'path_integral_deviation': self.path_integral_deviation,
            'area_under_curve': self.area_under_curve,
            'max_velocity': self.max_velocity,
            'mean_velocity': self.mean_velocity,
            'final_velocity': self.final_velocity,
            'max_acceleration': self.max_acceleration,
            'mean_acceleration': self.mean_acceleration,
            'trajectory_curvature': self.trajectory_curvature,
            'max_curvature': self.max_curvature,
            'max_momentum': self.max_momentum,
            'mean_momentum': self.mean_momentum,
            'excursion_duration': self.excursion_duration,
            'excursion_intensity': self.excursion_intensity,
            'return_to_baseline': self.return_to_baseline,
            'deviation_std': self.deviation_std,
            'deviation_range': self.deviation_range,
        }


def compute_trajectory_metrics(
    deviation_trajectory: np.ndarray,
    weight_trajectory: Optional[List[torch.Tensor]] = None,
    dt: float = 1.0,
    smooth_sigma: float = 1.0
) -> TrajectoryMetrics:
    """
    Compute comprehensive trajectory metrics from deviation time series.

    Args:
        deviation_trajectory: Array of deviation values over time
        weight_trajectory: Optional list of weight tensors for path length
        dt: Time step between measurements
        smooth_sigma: Gaussian smoothing for derivative computation

    Returns:
        TrajectoryMetrics with all computed values
    """
    dev = np.array(deviation_trajectory)
    n = len(dev)

    if n < 2:
        # Return zeros for insufficient data
        return TrajectoryMetrics(
            initial_deviation=dev[0] if n > 0 else 0.0,
            final_deviation=dev[-1] if n > 0 else 0.0,
            max_deviation=dev.max() if n > 0 else 0.0,
            min_deviation=dev.min() if n > 0 else 0.0,
            mean_deviation=dev.mean() if n > 0 else 0.0,
            path_length=0.0, path_integral_deviation=0.0, area_under_curve=0.0,
            max_velocity=0.0, mean_velocity=0.0, final_velocity=0.0,
            max_acceleration=0.0, mean_acceleration=0.0,
            trajectory_curvature=0.0, max_curvature=0.0,
            max_momentum=0.0, mean_momentum=0.0,
            excursion_duration=0.0, excursion_intensity=0.0, return_to_baseline=0.0,
            deviation_std=0.0, deviation_range=0.0
        )

    # Basic statistics
    initial_deviation = dev[0]
    final_deviation = dev[-1]
    max_deviation = dev.max()
    min_deviation = dev.min()
    mean_deviation = dev.mean()
    deviation_std = dev.std()
    deviation_range = max_deviation - min_deviation

    # Smooth for derivative computation
    dev_smooth = gaussian_filter1d(dev, sigma=smooth_sigma) if smooth_sigma > 0 else dev

    # Velocity (first derivative)
    velocity = np.gradient(dev_smooth, dt)
    abs_velocity = np.abs(velocity)
    max_velocity = abs_velocity.max()
    mean_velocity = abs_velocity.mean()
    final_velocity = velocity[-1]

    # Acceleration (second derivative)
    if n > 2:
        acceleration = np.gradient(velocity, dt)
        abs_acceleration = np.abs(acceleration)
        max_acceleration = abs_acceleration.max()
        mean_acceleration = abs_acceleration.mean()
    else:
        max_acceleration = 0.0
        mean_acceleration = 0.0

    # Path integral (area under deviation curve)
    time_points = np.arange(n) * dt
    path_integral_deviation = integrate.trapezoid(dev, time_points)
    total_time = (n - 1) * dt
    area_under_curve = path_integral_deviation / total_time if total_time > 0 else 0.0

    # Path length in weight space
    if weight_trajectory is not None and len(weight_trajectory) > 1:
        path_length = 0.0
        for i in range(1, len(weight_trajectory)):
            diff = weight_trajectory[i] - weight_trajectory[i-1]
            if isinstance(diff, torch.Tensor):
                diff = diff.flatten()
                path_length += diff.norm().item()
            else:
                path_length += np.linalg.norm(diff)
    else:
        # Approximate from deviation changes
        path_length = np.sum(np.abs(np.diff(dev)))

    # Curvature: κ = |d²y/dx²| / (1 + (dy/dx)²)^(3/2)
    if n > 2:
        curvature = np.abs(acceleration) / (1 + velocity**2)**1.5
        # Handle infinities
        curvature = np.where(np.isfinite(curvature), curvature, 0.0)
        trajectory_curvature = curvature.mean()
        max_curvature = curvature.max()
    else:
        trajectory_curvature = 0.0
        max_curvature = 0.0

    # Momentum (deviation × velocity)
    momentum = dev * abs_velocity
    max_momentum = momentum.max()
    mean_momentum = momentum.mean()

    # Excursion metrics
    above_mean = dev > mean_deviation
    excursion_duration = above_mean.sum() / n  # Fraction of time above mean
    excursion_intensity = max_deviation - initial_deviation
    return_to_baseline = abs(final_deviation - initial_deviation)

    return TrajectoryMetrics(
        initial_deviation=initial_deviation,
        final_deviation=final_deviation,
        max_deviation=max_deviation,
        min_deviation=min_deviation,
        mean_deviation=mean_deviation,
        path_length=path_length,
        path_integral_deviation=path_integral_deviation,
        area_under_curve=area_under_curve,
        max_velocity=max_velocity,
        mean_velocity=mean_velocity,
        final_velocity=final_velocity,
        max_acceleration=max_acceleration,
        mean_acceleration=mean_acceleration,
        trajectory_curvature=trajectory_curvature,
        max_curvature=max_curvature,
        max_momentum=max_momentum,
        mean_momentum=mean_momentum,
        excursion_duration=excursion_duration,
        excursion_intensity=excursion_intensity,
        return_to_baseline=return_to_baseline,
        deviation_std=deviation_std,
        deviation_range=deviation_range,
    )


def compute_combined_trajectory_metrics(
    trajectory_t1: np.ndarray,
    trajectory_t2: np.ndarray,
    dt: float = 1.0
) -> Dict[str, float]:
    """
    Compute trajectory metrics for both tasks and their relationship.

    Args:
        trajectory_t1: Deviation trajectory during Task 1
        trajectory_t2: Deviation trajectory during Task 2
        dt: Time step

    Returns:
        Dict with all trajectory metrics
    """
    # Individual task metrics
    metrics_t1 = compute_trajectory_metrics(trajectory_t1, dt=dt)
    metrics_t2 = compute_trajectory_metrics(trajectory_t2, dt=dt)

    # Combined metrics
    full_trajectory = np.concatenate([trajectory_t1, trajectory_t2])
    metrics_full = compute_trajectory_metrics(full_trajectory, dt=dt)

    result = {}

    # Task 1 metrics
    for key, value in metrics_t1.to_dict().items():
        result[f't1_{key}'] = value

    # Task 2 metrics
    for key, value in metrics_t2.to_dict().items():
        result[f't2_{key}'] = value

    # Full trajectory metrics
    for key, value in metrics_full.to_dict().items():
        result[f'full_{key}'] = value

    # Cross-task metrics
    result['t2_vs_t1_max_deviation_ratio'] = (
        metrics_t2.max_deviation / (metrics_t1.max_deviation + 1e-10)
    )
    result['t2_vs_t1_path_integral_ratio'] = (
        metrics_t2.path_integral_deviation / (metrics_t1.path_integral_deviation + 1e-10)
    )
    result['deviation_jump_at_transition'] = (
        trajectory_t2[0] - trajectory_t1[-1] if len(trajectory_t1) > 0 and len(trajectory_t2) > 0 else 0.0
    )

    return result


def fit_trajectory_forgetting_model(
    trajectory_metrics: Dict[str, np.ndarray],
    forgetting: np.ndarray,
    candidate_predictors: Optional[List[str]] = None
) -> Dict[str, Dict]:
    """
    Fit models predicting forgetting from trajectory metrics.

    Args:
        trajectory_metrics: Dict mapping metric names to arrays
        forgetting: Array of forgetting values
        candidate_predictors: List of metric names to try (default: all)

    Returns:
        Dict with fitted models and R² values
    """
    from scipy.optimize import curve_fit

    results = {}

    if candidate_predictors is None:
        candidate_predictors = list(trajectory_metrics.keys())

    # Clean data
    valid_mask = ~np.isnan(forgetting)
    for pred in candidate_predictors:
        if pred in trajectory_metrics:
            valid_mask &= ~np.isnan(trajectory_metrics[pred])

    forg = forgetting[valid_mask]
    n_valid = len(forg)

    if n_valid < 10:
        return {'error': 'Insufficient valid data'}

    # 1. Single predictor models
    for pred in candidate_predictors:
        if pred not in trajectory_metrics:
            continue

        x = trajectory_metrics[pred][valid_mask]

        try:
            # Linear model
            def linear(X, a, b):
                return a * X + b

            popt, _ = curve_fit(linear, x, forg, p0=[0.1, 0.0], maxfev=5000)
            pred_y = linear(x, *popt)
            ss_res = ((forg - pred_y) ** 2).sum()
            ss_tot = ((forg - forg.mean()) ** 2).sum()
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            results[f'linear_{pred}'] = {
                'equation': f'F = {popt[0]:.4f} * {pred} + {popt[1]:.4f}',
                'coefficients': {'a': popt[0], 'b': popt[1]},
                'r_squared': r2,
                'correlation': np.corrcoef(x, forg)[0, 1]
            }
        except Exception as e:
            results[f'linear_{pred}'] = {'error': str(e)}

    # 2. Best single predictor combinations
    best_predictors = sorted(
        [(p, results.get(f'linear_{p}', {}).get('r_squared', 0))
         for p in candidate_predictors if f'linear_{p}' in results],
        key=lambda x: x[1],
        reverse=True
    )[:5]  # Top 5

    # 3. Multi-predictor model with top predictors
    if len(best_predictors) >= 2:
        top_preds = [p for p, _ in best_predictors[:3]]

        try:
            X_multi = np.column_stack([
                trajectory_metrics[p][valid_mask] for p in top_preds
            ])

            # Add intercept
            X_with_intercept = np.column_stack([X_multi, np.ones(n_valid)])

            # Least squares fit
            coeffs, residuals, rank, s = np.linalg.lstsq(X_with_intercept, forg, rcond=None)

            pred_y = X_with_intercept @ coeffs
            ss_res = ((forg - pred_y) ** 2).sum()
            ss_tot = ((forg - forg.mean()) ** 2).sum()
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            eq_parts = [f'{coeffs[i]:.4f} * {top_preds[i]}' for i in range(len(top_preds))]
            equation = 'F = ' + ' + '.join(eq_parts) + f' + {coeffs[-1]:.4f}'

            results['multi_predictor'] = {
                'equation': equation,
                'predictors': top_preds,
                'coefficients': {p: coeffs[i] for i, p in enumerate(top_preds)},
                'intercept': coeffs[-1],
                'r_squared': r2
            }
        except Exception as e:
            results['multi_predictor'] = {'error': str(e)}

    # 4. Summary
    results['summary'] = {
        'best_single_predictor': best_predictors[0] if best_predictors else None,
        'top_5_predictors': best_predictors,
        'n_samples': n_valid
    }

    return results


def identify_trajectory_phases(
    deviation_trajectory: np.ndarray,
    threshold_percentile: float = 75
) -> Dict[str, any]:
    """
    Identify distinct phases in the trajectory.

    Phases:
    - Exploration: Initial deviation increase
    - Peak: Maximum deviation region
    - Consolidation: Return toward baseline

    Args:
        deviation_trajectory: Deviation values over time
        threshold_percentile: Percentile for "high" deviation

    Returns:
        Dict with phase boundaries and statistics
    """
    dev = np.array(deviation_trajectory)
    n = len(dev)

    if n < 3:
        return {'error': 'Insufficient data for phase identification'}

    threshold = np.percentile(dev, threshold_percentile)

    # Find peak
    peak_idx = np.argmax(dev)
    peak_value = dev[peak_idx]

    # Find first crossing above threshold (exploration end)
    above_threshold = dev >= threshold
    if above_threshold.any():
        exploration_end = np.argmax(above_threshold)
    else:
        exploration_end = peak_idx

    # Find last crossing above threshold (consolidation start)
    if above_threshold.any():
        consolidation_start = n - 1 - np.argmax(above_threshold[::-1])
    else:
        consolidation_start = peak_idx

    return {
        'exploration_phase': (0, exploration_end),
        'peak_phase': (exploration_end, consolidation_start),
        'consolidation_phase': (consolidation_start, n - 1),
        'peak_index': peak_idx,
        'peak_value': peak_value,
        'threshold': threshold,
        'exploration_mean': dev[:exploration_end+1].mean() if exploration_end > 0 else dev[0],
        'peak_mean': dev[exploration_end:consolidation_start+1].mean() if consolidation_start > exploration_end else peak_value,
        'consolidation_mean': dev[consolidation_start:].mean() if consolidation_start < n - 1 else dev[-1],
    }
