"""
Phase 3 Data Generation: Universal Subspace Analysis

Generates dataset for discovering:
- Correlation between subspace deviation and forgetting
- Transition boundary in subspace coordinates
- Whether deviation predicts lazy-rich regime

Based on Phase 2 findings:
- Focus on LR = 0.1 region where transition occurs
- Use GELU activation for clearer signal
- Measure subspace deviation alongside FLR
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
from .universal_subspace import (
    UniversalSubspace,
    SubspaceAnalysis,
    compute_transition_boundary,
    fit_transition_equation
)


@dataclass
class Phase3Config:
    """Configuration for Phase 3 subspace experiments."""

    # Dimensions
    d_in: int = 50
    d_out: int = 5

    # Focus on widths that showed regime variation in Phase 2
    hidden_widths: List[int] = field(default_factory=lambda: [32, 64, 128])

    # GELU showed best feature learning signal
    activations: List[str] = field(default_factory=lambda: ["gelu"])

    # Task similarity - full range
    similarities: List[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])

    # Focus on LR = 0.1 region where transition occurs
    # Also include neighbors to see boundary
    learning_rates: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.15, 0.2])

    # Init scales affect FLR
    init_scales: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])

    # Training
    n_steps: int = 300
    n_seeds: int = 5
    batch_size: int = 64
    noise_std: float = 0.0

    # Weight tracking for subspace analysis
    track_weights_every: int = 30  # Track every 30 steps

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
            'track_weights_every': self.track_weights_every,
            'subspace_variance_target': self.subspace_variance_target,
        }


@dataclass
class Phase3Result:
    """Results from a single Phase 3 experimental run."""

    # Configuration
    hidden_width: int
    activation: str
    similarity: float
    learning_rate: float
    init_scale: float
    seed: int

    # Primary measurements (from Phase 2)
    forgetting: float
    forward_transfer: float
    loss_t1_after_t1: float
    loss_t1_after_t2: float
    loss_t2_after_t2: float

    # FLR metrics (from Phase 2)
    flr_after_t1: float
    flr_after_t2: float

    # Regime classification
    regime_after_t1: str
    regime_after_t2: str

    # NEW: Subspace deviation metrics
    subspace_dim: int
    deviation_after_t1: float        # ||θ_⊥|| / ||θ_∥|| after task 1
    deviation_after_t2: float        # ||θ_⊥|| / ||θ_∥|| after task 2
    max_deviation_t1: float          # Max deviation during task 1
    max_deviation_t2: float          # Max deviation during task 2
    deviation_increase: float        # Increase from t1 to t2
    alignment_after_t1: float        # Alignment with subspace after t1
    alignment_after_t2: float        # Alignment with subspace after t2

    # Weight trajectory metrics
    weight_path_length_t1: float     # Total weight change during t1
    weight_path_length_t2: float     # Total weight change during t2

    # Derived
    overparameterization: float
    total_params: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            'hidden_width': self.hidden_width,
            'activation': self.activation,
            'similarity': self.similarity,
            'learning_rate': self.learning_rate,
            'init_scale': self.init_scale,
            'seed': self.seed,
            'forgetting': self.forgetting,
            'forward_transfer': self.forward_transfer,
            'loss_t1_after_t1': self.loss_t1_after_t1,
            'loss_t1_after_t2': self.loss_t1_after_t2,
            'loss_t2_after_t2': self.loss_t2_after_t2,
            'flr_after_t1': self.flr_after_t1,
            'flr_after_t2': self.flr_after_t2,
            'regime_after_t1': self.regime_after_t1,
            'regime_after_t2': self.regime_after_t2,
            'subspace_dim': self.subspace_dim,
            'deviation_after_t1': self.deviation_after_t1,
            'deviation_after_t2': self.deviation_after_t2,
            'max_deviation_t1': self.max_deviation_t1,
            'max_deviation_t2': self.max_deviation_t2,
            'deviation_increase': self.deviation_increase,
            'alignment_after_t1': self.alignment_after_t1,
            'alignment_after_t2': self.alignment_after_t2,
            'weight_path_length_t1': self.weight_path_length_t1,
            'weight_path_length_t2': self.weight_path_length_t2,
            'overparameterization': self.overparameterization,
            'total_params': self.total_params,
        }


def train_with_weight_tracking(
    student: NonlinearStudent,
    teacher: NonlinearTeacher,
    n_steps: int,
    lr: float,
    batch_size: int = 64,
    noise_std: float = 0.0,
    track_every: int = 30,
    X_probe: Optional[torch.Tensor] = None
) -> Tuple[Dict[str, List], List[torch.Tensor]]:
    """
    Train student on task while tracking weight trajectory.

    Returns:
        (history, weight_trajectory)
    """
    import torch.nn.functional as F

    optimizer = torch.optim.SGD(student.parameters(), lr=lr)
    device = student.device

    history = {
        'loss': [],
        'flr': [],
    }
    weight_trajectory = []

    if X_probe is None:
        X_probe = torch.randn(200, student.d_in, device=device)

    # Record initial weights
    weight_trajectory.append(torch.cat([
        student.W1.flatten().detach().clone(),
        student.W2.flatten().detach().clone()
    ]))

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

        # Track weights and FLR periodically
        if step % track_every == 0 or step == n_steps - 1:
            weight_trajectory.append(torch.cat([
                student.W1.flatten().detach().clone(),
                student.W2.flatten().detach().clone()
            ]))
            flr = student.compute_feature_learning_rate(X_probe)
            history['flr'].append(flr)

    return history, weight_trajectory


def compute_weight_path_length(trajectory: List[torch.Tensor]) -> float:
    """Compute total path length through weight space."""
    if len(trajectory) < 2:
        return 0.0

    path_length = 0.0
    for i in range(1, len(trajectory)):
        path_length += (trajectory[i] - trajectory[i-1]).norm().item()

    return path_length


def run_phase3_experiment(
    config: Phase3Config,
    hidden_width: int,
    activation: str,
    similarity: float,
    learning_rate: float,
    init_scale: float,
    seed: int,
    reference_subspace: Optional[UniversalSubspace] = None
) -> Phase3Result:
    """Run a single Phase 3 experiment with subspace tracking."""

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

    # Evaluate random init on task 2 (for forward transfer baseline)
    loss_t2_random = evaluate_nonlinear_on_task(student, teacher2, config.n_eval_samples)

    # === Train on Task 1 with weight tracking ===
    history_t1, trajectory_t1 = train_with_weight_tracking(
        student, teacher1,
        n_steps=config.n_steps,
        lr=learning_rate,
        batch_size=config.batch_size,
        noise_std=config.noise_std,
        track_every=config.track_weights_every,
        X_probe=X_probe
    )

    # Measurements after task 1
    loss_t1_after_t1 = evaluate_nonlinear_on_task(student, teacher1, config.n_eval_samples)
    flr_after_t1 = student.compute_feature_learning_rate(X_probe)
    ntk_alignment_t1 = student.compute_ntk_alignment(X_probe)
    regime_after_t1 = classify_regime(flr_after_t1, ntk_alignment_t1)

    weight_path_t1 = compute_weight_path_length(trajectory_t1)

    # Save weights after task 1
    student.save_weights_after_t1()
    W_after_t1 = torch.cat([
        student.W1.flatten().detach().clone(),
        student.W2.flatten().detach().clone()
    ])

    # === Train on Task 2 with weight tracking ===
    history_t2, trajectory_t2 = train_with_weight_tracking(
        student, teacher2,
        n_steps=config.n_steps,
        lr=learning_rate,
        batch_size=config.batch_size,
        noise_std=config.noise_std,
        track_every=config.track_weights_every,
        X_probe=X_probe
    )

    # Measurements after task 2
    loss_t1_after_t2 = evaluate_nonlinear_on_task(student, teacher1, config.n_eval_samples)
    loss_t2_after_t2 = evaluate_nonlinear_on_task(student, teacher2, config.n_eval_samples)
    flr_after_t2 = student.compute_feature_learning_rate(X_probe)
    ntk_alignment_t2 = student.compute_ntk_alignment(X_probe)
    regime_after_t2 = classify_regime(flr_after_t2, ntk_alignment_t2)

    weight_path_t2 = compute_weight_path_length(trajectory_t2)

    W_after_t2 = torch.cat([
        student.W1.flatten().detach().clone(),
        student.W2.flatten().detach().clone()
    ])

    # === Subspace Analysis ===
    # If no reference subspace provided, fit on this run's trajectory
    all_weights = trajectory_t1 + trajectory_t2
    if reference_subspace is None:
        subspace = UniversalSubspace(
            target_variance=config.subspace_variance_target,
            device=device
        )
        subspace.fit(all_weights)
    else:
        subspace = reference_subspace

    # Analyze deviations
    deviations_t1 = []
    for W in trajectory_t1:
        analysis = subspace.analyze(W)
        deviations_t1.append(analysis.deviation_ratio)

    deviations_t2 = []
    for W in trajectory_t2:
        analysis = subspace.analyze(W)
        deviations_t2.append(analysis.deviation_ratio)

    analysis_t1 = subspace.analyze(W_after_t1)
    analysis_t2 = subspace.analyze(W_after_t2)

    # Compute metrics
    forgetting = loss_t1_after_t2 - loss_t1_after_t1
    forward_transfer = loss_t2_random - evaluate_nonlinear_on_task(
        NonlinearStudent(config.d_in, hidden_width, config.d_out, activation, init_scale, device),
        teacher2, config.n_eval_samples
    )

    total_params = (config.d_in * hidden_width) + (hidden_width * config.d_out)

    return Phase3Result(
        hidden_width=hidden_width,
        activation=activation,
        similarity=similarity,
        learning_rate=learning_rate,
        init_scale=init_scale,
        seed=seed,
        forgetting=forgetting,
        forward_transfer=forward_transfer,
        loss_t1_after_t1=loss_t1_after_t1,
        loss_t1_after_t2=loss_t1_after_t2,
        loss_t2_after_t2=loss_t2_after_t2,
        flr_after_t1=flr_after_t1,
        flr_after_t2=flr_after_t2,
        regime_after_t1=regime_after_t1,
        regime_after_t2=regime_after_t2,
        subspace_dim=subspace.subspace_dim,
        deviation_after_t1=analysis_t1.deviation_ratio,
        deviation_after_t2=analysis_t2.deviation_ratio,
        max_deviation_t1=max(deviations_t1) if deviations_t1 else 0.0,
        max_deviation_t2=max(deviations_t2) if deviations_t2 else 0.0,
        deviation_increase=analysis_t2.deviation_ratio - analysis_t1.deviation_ratio,
        alignment_after_t1=analysis_t1.alignment_with_subspace,
        alignment_after_t2=analysis_t2.alignment_with_subspace,
        weight_path_length_t1=weight_path_t1,
        weight_path_length_t2=weight_path_t2,
        overparameterization=hidden_width / config.d_out,
        total_params=total_params,
    )


def generate_phase3_dataset(
    config: Phase3Config,
    output_path: Optional[Path] = None,
    show_progress: bool = True
) -> pd.DataFrame:
    """Generate full Phase 3 dataset."""

    results = []
    failed_experiments = []
    total = config.total_runs()

    iterator = tqdm(total=total, desc="Phase 3 experiments") if show_progress else None

    for hidden_width in config.hidden_widths:
        for activation in config.activations:
            for similarity in config.similarities:
                for lr in config.learning_rates:
                    for init_scale in config.init_scales:
                        for seed in range(config.n_seeds):
                            try:
                                result = run_phase3_experiment(
                                    config=config,
                                    hidden_width=hidden_width,
                                    activation=activation,
                                    similarity=similarity,
                                    learning_rate=lr,
                                    init_scale=init_scale,
                                    seed=seed
                                )
                                results.append(result.to_dict())
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


def analyze_phase3_results(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Comprehensive analysis of Phase 3 results.

    Returns:
        Dict with all analysis results
    """
    results = {}

    # 1. Basic statistics
    results['basic_stats'] = {
        'n_samples': len(df),
        'mean_forgetting': df['forgetting'].mean(),
        'std_forgetting': df['forgetting'].std(),
        'mean_deviation_t2': df['deviation_after_t2'].mean(),
        'std_deviation_t2': df['deviation_after_t2'].std(),
    }

    # 2. Correlation analysis
    correlations = {}
    for col in ['deviation_after_t2', 'max_deviation_t2', 'deviation_increase',
                'flr_after_t2', 'learning_rate', 'similarity', 'alignment_after_t2']:
        if col in df.columns:
            corr = df['forgetting'].corr(df[col])
            correlations[col] = corr
    results['correlations_with_forgetting'] = correlations

    # 3. Regime analysis
    regime_stats = df.groupby('regime_after_t2').agg({
        'forgetting': ['mean', 'std', 'count'],
        'deviation_after_t2': ['mean', 'std'],
        'flr_after_t2': ['mean', 'std'],
    }).round(4)
    results['regime_stats'] = regime_stats.to_dict()

    # 4. Transition boundary analysis
    clean_df = df.dropna(subset=['deviation_after_t2', 'forgetting'])
    if len(clean_df) > 10:
        boundary = compute_transition_boundary(
            clean_df['deviation_after_t2'].values,
            clean_df['forgetting'].values
        )
        results['transition_boundary'] = boundary

    # 5. Equation fitting
    clean_df = df.dropna(subset=['deviation_after_t2', 'forgetting', 'flr_after_t2',
                                  'learning_rate', 'similarity'])
    if len(clean_df) > 10:
        equations = fit_transition_equation(
            clean_df['deviation_after_t2'].values,
            clean_df['forgetting'].values,
            clean_df['flr_after_t2'].values,
            clean_df['learning_rate'].values,
            clean_df['similarity'].values
        )
        results['fitted_equations'] = equations

    # 6. Deviation by hyperparameters
    deviation_by_lr = df.groupby('learning_rate')['deviation_after_t2'].agg(['mean', 'std']).round(4)
    results['deviation_by_lr'] = deviation_by_lr.to_dict()

    deviation_by_init = df.groupby('init_scale')['deviation_after_t2'].agg(['mean', 'std']).round(4)
    results['deviation_by_init_scale'] = deviation_by_init.to_dict()

    # 7. Is deviation better predictor than FLR?
    if 'flr_after_t2' in df.columns and 'deviation_after_t2' in df.columns:
        clean_df = df.dropna(subset=['forgetting', 'flr_after_t2', 'deviation_after_t2'])
        if len(clean_df) > 10:
            corr_flr = clean_df['forgetting'].corr(clean_df['flr_after_t2'])
            corr_dev = clean_df['forgetting'].corr(clean_df['deviation_after_t2'])
            results['predictor_comparison'] = {
                'flr_correlation': corr_flr,
                'deviation_correlation': corr_dev,
                'better_predictor': 'deviation' if abs(corr_dev) > abs(corr_flr) else 'flr'
            }

    return results
