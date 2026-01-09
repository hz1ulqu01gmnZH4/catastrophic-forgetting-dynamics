"""
Phase 4 Data Generation: Trajectory Hypothesis Testing

Tests the hypothesis: Forgetting ∝ max_t ||θ_⊥(t)|| / ||θ_∥(t)||

Key experiments:
1. Dense trajectory tracking (every 10 steps)
2. Compute path integrals and velocity metrics
3. Test if trajectory shape predicts forgetting
4. Compare trajectory metrics to FLR and final deviation
"""

import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import json
import traceback
from pathlib import Path

from .nonlinear_models import (
    NonlinearTeacher,
    NonlinearStudent,
    create_nonlinear_task_pair,
    evaluate_nonlinear_on_task,
    classify_regime
)
from .universal_subspace import UniversalSubspace
from .trajectory_analysis import (
    compute_trajectory_metrics,
    compute_combined_trajectory_metrics,
    fit_trajectory_forgetting_model,
    TrajectoryMetrics
)


@dataclass
class Phase4Config:
    """Configuration for Phase 4 trajectory experiments."""

    # Dimensions
    d_in: int = 50
    d_out: int = 5

    # Focus on settings that showed interesting dynamics
    hidden_widths: List[int] = field(default_factory=lambda: [32, 64, 128])

    # GELU for consistent feature learning
    activations: List[str] = field(default_factory=lambda: ["gelu"])

    # Task similarity
    similarities: List[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])

    # Focus on transition region from Phase 2/3
    learning_rates: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.15])

    # Init scales
    init_scales: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])

    # Training
    n_steps: int = 300
    n_seeds: int = 5
    batch_size: int = 64
    noise_std: float = 0.0

    # DENSE trajectory tracking for Phase 4
    track_every: int = 10  # Track every 10 steps (vs 30 in Phase 3)

    # Subspace fitting
    subspace_variance_target: float = 0.90

    # Evaluation
    n_eval_samples: int = 500
    n_probe_samples: int = 200

    # Device
    device: str = "cpu"

    def total_runs(self) -> int:
        return (
            len(self.hidden_widths) *
            len(self.activations) *
            len(self.similarities) *
            len(self.learning_rates) *
            len(self.init_scales) *
            self.n_seeds
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'd_in': self.d_in,
            'd_out': self.d_out,
            'hidden_widths': self.hidden_widths,
            'activations': self.activations,
            'similarities': self.similarities,
            'learning_rates': self.learning_rates,
            'init_scales': self.init_scales,
            'n_steps': self.n_steps,
            'n_seeds': self.n_seeds,
            'batch_size': self.batch_size,
            'track_every': self.track_every,
            'subspace_variance_target': self.subspace_variance_target,
        }


def train_with_dense_tracking(
    student: NonlinearStudent,
    teacher: NonlinearTeacher,
    n_steps: int,
    lr: float,
    batch_size: int = 64,
    noise_std: float = 0.0,
    track_every: int = 10,
    X_probe: Optional[torch.Tensor] = None,
    subspace: Optional[UniversalSubspace] = None
) -> Tuple[Dict[str, List], List[float], List[torch.Tensor]]:
    """
    Train student with dense weight trajectory tracking.

    Args:
        student: Student network
        teacher: Teacher network
        n_steps: Training steps
        lr: Learning rate
        batch_size: Batch size
        noise_std: Label noise
        track_every: Track weights every N steps
        X_probe: Probe data for FLR
        subspace: Optional fitted subspace for deviation tracking

    Returns:
        (history, deviation_trajectory, weight_trajectory)
    """
    import torch.nn.functional as F

    optimizer = torch.optim.SGD(student.parameters(), lr=lr)
    device = student.device

    history = {'loss': [], 'flr': []}
    deviation_trajectory = []
    weight_trajectory = []

    if X_probe is None:
        X_probe = torch.randn(200, student.d_in, device=device)

    # Record initial state
    W_init = torch.cat([
        student.W1.flatten().detach().clone(),
        student.W2.flatten().detach().clone()
    ])
    weight_trajectory.append(W_init)

    # Initial deviation (only if subspace is fitted)
    if subspace is not None and subspace.is_fitted:
        try:
            analysis = subspace.analyze(W_init)
            deviation_trajectory.append(analysis.deviation_ratio)
        except Exception:
            deviation_trajectory.append(0.0)
    else:
        deviation_trajectory.append(0.0)

    for step in range(n_steps):
        # Generate batch
        X, y = teacher.generate_data(batch_size, noise_std=noise_std, device=device)

        # Training step
        optimizer.zero_grad()
        y_pred = student(X)
        loss = F.mse_loss(y_pred, y)
        loss.backward()
        optimizer.step()

        history['loss'].append(loss.item())

        # Dense tracking
        if step % track_every == 0 or step == n_steps - 1:
            W_current = torch.cat([
                student.W1.flatten().detach().clone(),
                student.W2.flatten().detach().clone()
            ])
            weight_trajectory.append(W_current)

            # Compute deviation from subspace (if fitted)
            if subspace is not None and subspace.is_fitted:
                try:
                    analysis = subspace.analyze(W_current)
                    deviation_trajectory.append(analysis.deviation_ratio)
                except Exception:
                    deviation_trajectory.append(deviation_trajectory[-1] if deviation_trajectory else 0.0)
            else:
                deviation_trajectory.append(0.0)

            # FLR
            flr = student.compute_feature_learning_rate(X_probe)
            history['flr'].append(flr)

    return history, deviation_trajectory, weight_trajectory


def run_phase4_experiment(
    config: Phase4Config,
    hidden_width: int,
    activation: str,
    similarity: float,
    learning_rate: float,
    init_scale: float,
    seed: int
) -> Dict[str, Any]:
    """Run a single Phase 4 experiment with full trajectory analysis."""

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = config.device

    # Create task pair
    task_pair = create_nonlinear_task_pair(
        d_in=config.d_in,
        d_hidden=hidden_width,
        d_out=config.d_out,
        similarity=similarity,
        activation=activation,
        device=device
    )

    teacher1 = NonlinearTeacher(
        task_pair.teacher1_W1, task_pair.teacher1_W2, activation
    )
    teacher2 = NonlinearTeacher(
        task_pair.teacher2_W1, task_pair.teacher2_W2, activation
    )

    # Create student
    student = NonlinearStudent(
        d_in=config.d_in,
        d_hidden=hidden_width,
        d_out=config.d_out,
        activation=activation,
        init_scale=init_scale,
        device=device
    )

    # Probe data
    X_probe = torch.randn(config.n_probe_samples, config.d_in, device=device)

    # Evaluate random init
    loss_t2_random = evaluate_nonlinear_on_task(student, teacher2, config.n_eval_samples)

    # === Store initial weights ===
    W_init = torch.cat([
        student.W1.flatten().detach().clone(),
        student.W2.flatten().detach().clone()
    ])

    # === Train on Task 1 with weight tracking (no subspace yet) ===
    history_t1_raw, _, weights_t1 = train_with_dense_tracking(
        student, teacher1,
        n_steps=config.n_steps,
        lr=learning_rate,
        batch_size=config.batch_size,
        noise_std=config.noise_std,
        track_every=config.track_every,
        X_probe=X_probe,
        subspace=None  # No subspace yet
    )

    # === Fit subspace on Task 1 trajectory ===
    subspace = UniversalSubspace(
        target_variance=config.subspace_variance_target,
        device=device
    )

    try:
        subspace.fit(weights_t1)
    except Exception as e:
        raise RuntimeError(
            f"Subspace fitting failed on trajectory of {len(weights_t1)} weight snapshots. "
            f"Cannot proceed with degraded single-point subspace: {e}"
        ) from e

    # Recompute Task 1 deviations with fitted subspace
    deviation_t1 = []
    for W in weights_t1:
        try:
            analysis = subspace.analyze(W)
            deviation_t1.append(analysis.deviation_ratio)
        except Exception:
            deviation_t1.append(0.0)

    # Measurements after task 1
    loss_t1_after_t1 = evaluate_nonlinear_on_task(student, teacher1, config.n_eval_samples)
    flr_after_t1 = student.compute_feature_learning_rate(X_probe)
    ntk_alignment_t1 = student.compute_ntk_alignment(X_probe)
    regime_after_t1 = classify_regime(flr_after_t1, ntk_alignment_t1)

    # Save weights after task 1
    student.save_weights_after_t1()
    W_after_t1 = torch.cat([
        student.W1.flatten().detach().clone(),
        student.W2.flatten().detach().clone()
    ])

    # === Train on Task 2 with dense tracking ===
    history_t2, deviation_t2, weights_t2 = train_with_dense_tracking(
        student, teacher2,
        n_steps=config.n_steps,
        lr=learning_rate,
        batch_size=config.batch_size,
        noise_std=config.noise_std,
        track_every=config.track_every,
        X_probe=X_probe,
        subspace=subspace  # Use Task 1 subspace
    )

    # Measurements after task 2
    loss_t1_after_t2 = evaluate_nonlinear_on_task(student, teacher1, config.n_eval_samples)
    loss_t2_after_t2 = evaluate_nonlinear_on_task(student, teacher2, config.n_eval_samples)
    flr_after_t2 = student.compute_feature_learning_rate(X_probe)
    ntk_alignment_t2 = student.compute_ntk_alignment(X_probe)
    regime_after_t2 = classify_regime(flr_after_t2, ntk_alignment_t2)

    # Compute forgetting
    forgetting = loss_t1_after_t2 - loss_t1_after_t1

    # === Compute trajectory metrics ===
    trajectory_metrics = compute_combined_trajectory_metrics(
        np.array(deviation_t1),
        np.array(deviation_t2),
        dt=config.track_every
    )

    # Build result
    result = {
        # Configuration
        'hidden_width': hidden_width,
        'activation': activation,
        'similarity': similarity,
        'learning_rate': learning_rate,
        'init_scale': init_scale,
        'seed': seed,

        # Primary measurements
        'forgetting': forgetting,
        'forward_transfer': loss_t2_random - loss_t2_after_t2,
        'loss_t1_after_t1': loss_t1_after_t1,
        'loss_t1_after_t2': loss_t1_after_t2,
        'loss_t2_after_t2': loss_t2_after_t2,

        # FLR and regime
        'flr_after_t1': flr_after_t1,
        'flr_after_t2': flr_after_t2,
        'regime_after_t1': regime_after_t1,
        'regime_after_t2': regime_after_t2,

        # Subspace info
        'subspace_dim': subspace.subspace_dim,

        # Derived
        'overparameterization': hidden_width / config.d_out,
        'total_params': (config.d_in * hidden_width) + (hidden_width * config.d_out),
    }

    # Add all trajectory metrics
    result.update(trajectory_metrics)

    return result


def generate_phase4_dataset(
    config: Phase4Config,
    output_path: Optional[Path] = None,
    show_progress: bool = True
) -> pd.DataFrame:
    """Generate full Phase 4 dataset."""

    results = []
    failed_experiments = []
    total = config.total_runs()

    iterator = tqdm(total=total, desc="Phase 4 experiments") if show_progress else None

    for hidden_width in config.hidden_widths:
        for activation in config.activations:
            for similarity in config.similarities:
                for lr in config.learning_rates:
                    for init_scale in config.init_scales:
                        for seed in range(config.n_seeds):
                            try:
                                result = run_phase4_experiment(
                                    config=config,
                                    hidden_width=hidden_width,
                                    activation=activation,
                                    similarity=similarity,
                                    learning_rate=lr,
                                    init_scale=init_scale,
                                    seed=seed
                                )
                                results.append(result)
                            except Exception as e:
                                failed_experiments.append({
                                    'config': {
                                        'hidden_width': hidden_width,
                                        'activation': activation,
                                        'similarity': similarity,
                                        'learning_rate': lr,
                                        'init_scale': init_scale,
                                        'seed': seed
                                    },
                                    'error': str(e),
                                    'traceback': traceback.format_exc()
                                })
                                continue

                            if iterator:
                                iterator.update(1)

    if iterator:
        iterator.close()

    # Check failure rate
    failure_rate = len(failed_experiments) / total if total > 0 else 0
    if failure_rate > 0.05:
        raise RuntimeError(
            f"Experiment failure rate {failure_rate:.1%} exceeds 5% threshold. "
            f"Failed {len(failed_experiments)}/{total} experiments. "
            f"First failure: {failed_experiments[0] if failed_experiments else 'N/A'}"
        )
    elif failed_experiments:
        print(f"Warning: {len(failed_experiments)}/{total} experiments failed ({failure_rate:.1%})")

    df = pd.DataFrame(results)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        config_path = output_path.parent / f"{output_path.stem}_config.json"
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)

    return df


def analyze_phase4_results(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Comprehensive analysis of Phase 4 trajectory results.

    Tests the trajectory hypothesis: Does path shape predict forgetting
    better than endpoint?
    """
    results = {}

    # 1. Basic statistics
    results['basic_stats'] = {
        'n_samples': len(df),
        'mean_forgetting': df['forgetting'].mean(),
        'std_forgetting': df['forgetting'].std(),
    }

    # 2. Identify trajectory metric columns
    trajectory_cols = [col for col in df.columns if any(
        prefix in col for prefix in ['t1_', 't2_', 'full_', 'deviation']
    )]

    # 3. Correlations with forgetting
    correlations = {}
    for col in trajectory_cols + ['flr_after_t2', 'learning_rate', 'similarity']:
        if col in df.columns:
            clean = df[[col, 'forgetting']].dropna()
            if len(clean) > 10:
                corr = clean['forgetting'].corr(clean[col])
                correlations[col] = corr

    # Sort by absolute correlation
    correlations = dict(sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True))
    results['correlations_with_forgetting'] = correlations

    # 4. Top trajectory predictors
    trajectory_correlations = {k: v for k, v in correlations.items()
                               if any(prefix in k for prefix in ['t1_', 't2_', 'full_'])}
    results['top_trajectory_predictors'] = dict(list(trajectory_correlations.items())[:10])

    # 5. Fit trajectory-based model
    # Get numeric trajectory columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    trajectory_numeric = [col for col in trajectory_cols if col in numeric_cols]

    if len(trajectory_numeric) > 0:
        # Prepare data
        clean_df = df.dropna(subset=['forgetting'] + trajectory_numeric[:10])

        if len(clean_df) > 20:
            metrics_dict = {col: clean_df[col].values for col in trajectory_numeric[:10]}
            forgetting_arr = clean_df['forgetting'].values

            model_results = fit_trajectory_forgetting_model(
                metrics_dict, forgetting_arr, trajectory_numeric[:10]
            )
            results['trajectory_models'] = model_results

    # 6. Compare trajectory vs simple predictors
    clean_df = df.dropna(subset=['forgetting', 'flr_after_t2', 'similarity', 'learning_rate'])
    if len(clean_df) > 10:
        corr_flr = clean_df['forgetting'].corr(clean_df['flr_after_t2'])
        corr_sim = clean_df['forgetting'].corr(clean_df['similarity'])

        # Best trajectory predictor
        best_traj = max(trajectory_correlations.items(), key=lambda x: abs(x[1])) if trajectory_correlations else (None, 0)

        results['predictor_comparison'] = {
            'flr_correlation': corr_flr,
            'similarity_correlation': corr_sim,
            'best_trajectory_predictor': best_traj[0],
            'best_trajectory_correlation': best_traj[1],
            'trajectory_beats_flr': abs(best_traj[1]) > abs(corr_flr) if best_traj[0] else False,
        }

    # 7. Regime analysis with trajectory metrics
    if 'regime_after_t2' in df.columns:
        regime_stats = {}
        for col in ['t2_max_deviation', 't2_path_integral_deviation', 't2_excursion_intensity']:
            if col in df.columns:
                by_regime = df.groupby('regime_after_t2')[col].agg(['mean', 'std']).round(4)
                regime_stats[col] = by_regime.to_dict()
        results['trajectory_by_regime'] = regime_stats

    return results
