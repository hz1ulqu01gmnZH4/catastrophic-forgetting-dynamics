"""
Phase 2 Data Generation: Nonlinear Networks

Generates dataset for discovering equations governing:
- Catastrophic forgetting in nonlinear networks
- Lazy-rich transition boundary
- Feature learning rate effects
"""

import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import json
import traceback
from pathlib import Path

from .nonlinear_models import (
    NonlinearTeacher,
    NonlinearStudent,
    create_nonlinear_task_pair,
    train_nonlinear_on_task,
    evaluate_nonlinear_on_task,
    classify_regime
)


@dataclass
class Phase2Config:
    """Configuration for Phase 2 nonlinear experiments."""

    # Dimensions
    d_in: int = 50
    d_out: int = 5

    # Hidden layer widths (key variable for lazy-rich transition)
    hidden_widths: List[int] = field(default_factory=lambda: [32, 64, 128, 256, 512])

    # Activations to test
    activations: List[str] = field(default_factory=lambda: ["relu", "tanh", "gelu"])

    # Task similarity
    similarities: List[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])

    # Learning rates (key for lazy-rich transition)
    learning_rates: List[float] = field(default_factory=lambda: [0.001, 0.01, 0.1])

    # Initialization scales (affects initial NTK)
    init_scales: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])

    # Training
    n_steps: int = 500
    n_seeds: int = 3
    batch_size: int = 64
    noise_std: float = 0.0

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
        }


@dataclass
class Phase2Result:
    """Results from a single Phase 2 experimental run."""

    # Configuration
    hidden_width: int
    activation: str
    similarity: float
    learning_rate: float
    init_scale: float
    seed: int

    # Primary measurements
    forgetting: float
    forward_transfer: float
    loss_t1_after_t1: float
    loss_t1_after_t2: float
    loss_t2_after_t2: float

    # Feature learning rate (key Phase 2 metric)
    flr_after_t1: float          # FLR after task 1
    flr_after_t2: float          # FLR after task 2 (cumulative)
    flr_w1_after_t1: float       # Weight-based FLR for W1
    flr_w2_after_t1: float       # Weight-based FLR for W2

    # NTK alignment (lazy-rich indicator)
    ntk_alignment_after_t1: float
    ntk_alignment_after_t2: float

    # Regime classification
    regime_after_t1: str         # "lazy", "rich", or "transition"
    regime_after_t2: str

    # Weight changes
    w1_change_t1: float
    w2_change_t1: float
    w1_change_t2: float
    w2_change_t2: float

    # Derived metrics
    overparameterization: float  # hidden_width / d_out
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
            'flr_w1_after_t1': self.flr_w1_after_t1,
            'flr_w2_after_t1': self.flr_w2_after_t1,
            'ntk_alignment_after_t1': self.ntk_alignment_after_t1,
            'ntk_alignment_after_t2': self.ntk_alignment_after_t2,
            'regime_after_t1': self.regime_after_t1,
            'regime_after_t2': self.regime_after_t2,
            'w1_change_t1': self.w1_change_t1,
            'w2_change_t1': self.w2_change_t1,
            'w1_change_t2': self.w1_change_t2,
            'w2_change_t2': self.w2_change_t2,
            'overparameterization': self.overparameterization,
            'total_params': self.total_params,
        }


def run_phase2_experiment(
    config: Phase2Config,
    hidden_width: int,
    activation: str,
    similarity: float,
    learning_rate: float,
    init_scale: float,
    seed: int
) -> Phase2Result:
    """Run a single Phase 2 experiment."""

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

    # Probe data for FLR measurement
    X_probe = torch.randn(config.n_probe_samples, config.d_in, device=device)

    # Evaluate random init on task 2 (for forward transfer baseline)
    loss_t2_random = evaluate_nonlinear_on_task(student, teacher2, config.n_eval_samples)

    # === Train on Task 1 ===
    history_t1 = train_nonlinear_on_task(
        student, teacher1,
        n_steps=config.n_steps,
        lr=learning_rate,
        batch_size=config.batch_size,
        noise_std=config.noise_std,
        X_probe=X_probe
    )

    # Measurements after task 1
    loss_t1_after_t1 = evaluate_nonlinear_on_task(student, teacher1, config.n_eval_samples)
    loss_t2_after_t1 = evaluate_nonlinear_on_task(student, teacher2, config.n_eval_samples)

    flr_after_t1 = student.compute_feature_learning_rate(X_probe)
    ntk_alignment_after_t1 = student.compute_ntk_alignment(X_probe)
    weight_flr_t1 = student.compute_weight_flr()

    regime_after_t1 = classify_regime(flr_after_t1, ntk_alignment_after_t1)

    # Save weights after task 1
    student.save_weights_after_t1()
    W1_after_t1 = student.W1.clone().detach()
    W2_after_t1 = student.W2.clone().detach()
    w1_change_t1 = (W1_after_t1 - student._W1_init).norm().item()
    w2_change_t1 = (W2_after_t1 - student._W2_init).norm().item()

    # === Train on Task 2 ===
    history_t2 = train_nonlinear_on_task(
        student, teacher2,
        n_steps=config.n_steps,
        lr=learning_rate,
        batch_size=config.batch_size,
        noise_std=config.noise_std,
        X_probe=X_probe
    )

    # Measurements after task 2
    loss_t1_after_t2 = evaluate_nonlinear_on_task(student, teacher1, config.n_eval_samples)
    loss_t2_after_t2 = evaluate_nonlinear_on_task(student, teacher2, config.n_eval_samples)

    flr_after_t2 = student.compute_feature_learning_rate(X_probe)
    ntk_alignment_after_t2 = student.compute_ntk_alignment(X_probe)

    regime_after_t2 = classify_regime(flr_after_t2, ntk_alignment_after_t2)

    weight_change_t2 = student.compute_weight_change_t2()

    # Compute metrics
    forgetting = loss_t1_after_t2 - loss_t1_after_t1
    forward_transfer = loss_t2_random - loss_t2_after_t1

    total_params = (config.d_in * hidden_width) + (hidden_width * config.d_out)

    return Phase2Result(
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
        flr_w1_after_t1=weight_flr_t1['flr_w1'],
        flr_w2_after_t1=weight_flr_t1['flr_w2'],
        ntk_alignment_after_t1=ntk_alignment_after_t1,
        ntk_alignment_after_t2=ntk_alignment_after_t2,
        regime_after_t1=regime_after_t1,
        regime_after_t2=regime_after_t2,
        w1_change_t1=w1_change_t1,
        w2_change_t1=w2_change_t1,
        w1_change_t2=weight_change_t2['w1_change_t2'],
        w2_change_t2=weight_change_t2['w2_change_t2'],
        overparameterization=hidden_width / config.d_out,
        total_params=total_params,
    )


def generate_phase2_dataset(
    config: Phase2Config,
    output_path: Optional[Path] = None,
    show_progress: bool = True
) -> pd.DataFrame:
    """Generate full Phase 2 dataset."""

    results = []
    failed_experiments = []
    total = config.total_runs()

    iterator = tqdm(total=total, desc="Phase 2 experiments") if show_progress else None

    for hidden_width in config.hidden_widths:
        for activation in config.activations:
            for similarity in config.similarities:
                for lr in config.learning_rates:
                    for init_scale in config.init_scales:
                        for seed in range(config.n_seeds):
                            try:
                                result = run_phase2_experiment(
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
