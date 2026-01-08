"""
Data generation pipeline for catastrophic forgetting experiments.

Generates dataset of (hyperparameters, measurements) pairs for symbolic regression.
"""

import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import json
from pathlib import Path

from .models import (
    LinearTeacher,
    LinearStudent,
    create_task_pair,
    train_on_task,
    evaluate_on_task
)


@dataclass
class ExperimentConfig:
    """Configuration for forgetting experiment sweep."""

    # Dimensions
    d_in: int = 100
    d_out: int = 10

    # Hyperparameter sweeps
    widths: List[int] = field(default_factory=lambda: [50, 100, 200, 500, 1000])
    similarities: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    learning_rates: List[float] = field(default_factory=lambda: [0.001, 0.01, 0.1])
    n_steps_list: List[int] = field(default_factory=lambda: [100, 500, 1000])

    # Repetitions
    n_seeds: int = 5

    # Training settings
    batch_size: int = 64
    noise_std: float = 0.0

    # Evaluation
    n_eval_samples: int = 1000

    # Device
    device: str = "cpu"

    def total_runs(self) -> int:
        """Total number of experimental runs."""
        return (
            len(self.widths) *
            len(self.similarities) *
            len(self.learning_rates) *
            len(self.n_steps_list) *
            self.n_seeds
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'd_in': self.d_in,
            'd_out': self.d_out,
            'widths': self.widths,
            'similarities': self.similarities,
            'learning_rates': self.learning_rates,
            'n_steps_list': self.n_steps_list,
            'n_seeds': self.n_seeds,
            'batch_size': self.batch_size,
            'noise_std': self.noise_std,
            'n_eval_samples': self.n_eval_samples,
            'device': self.device,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ExperimentConfig':
        """Create from dictionary."""
        return cls(**d)


@dataclass
class RunResult:
    """Results from a single experimental run."""

    # Input hyperparameters
    width: int
    similarity: float
    learning_rate: float
    n_steps: int
    seed: int

    # Primary measurements
    forgetting: float           # Loss on T1 after T2 - Loss on T1 after T1
    forward_transfer: float     # Performance improvement on T2 due to T1
    loss_t1_after_t1: float     # Loss on task 1 after training on task 1
    loss_t1_after_t2: float     # Loss on task 1 after training on task 2
    loss_t2_after_t2: float     # Loss on task 2 after training on task 2

    # Weight-based measurements
    weight_change_t1: float     # ||W_after_T1 - W_init||
    weight_change_t2: float     # ||W_after_T2 - W_after_T1||
    weight_change_total: float  # ||W_after_T2 - W_init||

    # Derived metrics
    effective_rank_init: float
    effective_rank_final: float
    weight_norm_init: float
    weight_norm_final: float

    # Overparameterization ratio
    overparameterization: float  # width * d_in / (d_out * d_in) = width / d_out

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'width': self.width,
            'similarity': self.similarity,
            'learning_rate': self.learning_rate,
            'n_steps': self.n_steps,
            'seed': self.seed,
            'forgetting': self.forgetting,
            'forward_transfer': self.forward_transfer,
            'loss_t1_after_t1': self.loss_t1_after_t1,
            'loss_t1_after_t2': self.loss_t1_after_t2,
            'loss_t2_after_t2': self.loss_t2_after_t2,
            'weight_change_t1': self.weight_change_t1,
            'weight_change_t2': self.weight_change_t2,
            'weight_change_total': self.weight_change_total,
            'effective_rank_init': self.effective_rank_init,
            'effective_rank_final': self.effective_rank_final,
            'weight_norm_init': self.weight_norm_init,
            'weight_norm_final': self.weight_norm_final,
            'overparameterization': self.overparameterization,
        }


def compute_effective_rank(W: torch.Tensor) -> float:
    """
    Compute effective rank of weight matrix.

    Effective rank = nuclear norm / spectral norm
    This measures the "spread" of singular values.
    """
    try:
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        nuclear = S.sum()
        spectral = S[0]
        if spectral < 1e-10:
            return 0.0
        return (nuclear / spectral).item()
    except Exception:
        return float('nan')


def run_single_experiment(
    config: ExperimentConfig,
    width: int,
    similarity: float,
    learning_rate: float,
    n_steps: int,
    seed: int
) -> RunResult:
    """
    Run a single forgetting experiment.

    1. Create task pair with specified similarity
    2. Train student on task 1
    3. Measure loss on task 1
    4. Train student on task 2
    5. Measure loss on both tasks
    6. Compute forgetting and other metrics
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = config.device

    # Create task pair
    task_pair = create_task_pair(
        d_in=config.d_in,
        d_out=config.d_out,
        similarity=similarity,
        normalize=True,
        device=device
    )

    teacher1 = LinearTeacher(task_pair.teacher1)
    teacher2 = LinearTeacher(task_pair.teacher2)

    # Create student (note: for linear model, "width" doesn't apply directly
    # In this baseline, we use width as a scaling factor for effective capacity
    # For true width experiments, see Phase 2 with hidden layers)

    # For linear model: we simulate overparameterization via input dimension scaling
    # or simply track width as metadata for later phases
    student = LinearStudent(
        d_in=config.d_in,
        d_out=config.d_out,
        init_scale=1.0,
        device=device
    )

    W_init = student.weights.clone().detach()
    effective_rank_init = compute_effective_rank(W_init)
    weight_norm_init = W_init.norm().item()

    # Evaluate random initialization on task 2 (for forward transfer baseline)
    loss_t2_random = evaluate_on_task(student, teacher2, config.n_eval_samples, device)

    # Train on task 1
    train_on_task(
        student, teacher1,
        n_steps=n_steps,
        lr=learning_rate,
        batch_size=config.batch_size,
        noise_std=config.noise_std,
        device=device
    )

    W_after_t1 = student.weights.clone().detach()
    weight_change_t1 = (W_after_t1 - W_init).norm().item()

    # Evaluate after task 1
    loss_t1_after_t1 = evaluate_on_task(student, teacher1, config.n_eval_samples, device)
    loss_t2_after_t1 = evaluate_on_task(student, teacher2, config.n_eval_samples, device)

    # Reset baseline for weight change measurement
    student.reset_initial_weights()

    # Train on task 2
    train_on_task(
        student, teacher2,
        n_steps=n_steps,
        lr=learning_rate,
        batch_size=config.batch_size,
        noise_std=config.noise_std,
        device=device
    )

    W_after_t2 = student.weights.clone().detach()
    weight_change_t2 = (W_after_t2 - W_after_t1).norm().item()
    weight_change_total = (W_after_t2 - W_init).norm().item()

    # Evaluate after task 2
    loss_t1_after_t2 = evaluate_on_task(student, teacher1, config.n_eval_samples, device)
    loss_t2_after_t2 = evaluate_on_task(student, teacher2, config.n_eval_samples, device)

    effective_rank_final = compute_effective_rank(W_after_t2)
    weight_norm_final = W_after_t2.norm().item()

    # Compute metrics
    forgetting = loss_t1_after_t2 - loss_t1_after_t1
    forward_transfer = loss_t2_random - loss_t2_after_t1  # Positive = beneficial transfer

    return RunResult(
        width=width,
        similarity=similarity,
        learning_rate=learning_rate,
        n_steps=n_steps,
        seed=seed,
        forgetting=forgetting,
        forward_transfer=forward_transfer,
        loss_t1_after_t1=loss_t1_after_t1,
        loss_t1_after_t2=loss_t1_after_t2,
        loss_t2_after_t2=loss_t2_after_t2,
        weight_change_t1=weight_change_t1,
        weight_change_t2=weight_change_t2,
        weight_change_total=weight_change_total,
        effective_rank_init=effective_rank_init,
        effective_rank_final=effective_rank_final,
        weight_norm_init=weight_norm_init,
        weight_norm_final=weight_norm_final,
        overparameterization=width / config.d_out,
    )


def generate_forgetting_dataset(
    config: ExperimentConfig,
    output_path: Optional[Path] = None,
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Generate full dataset of forgetting measurements.

    Sweeps over all hyperparameter combinations and records measurements.

    Args:
        config: Experiment configuration
        output_path: Optional path to save results
        show_progress: Whether to show progress bar

    Returns:
        DataFrame with all measurements
    """
    results = []

    total = config.total_runs()
    iterator = tqdm(total=total, desc="Generating data") if show_progress else None

    for width in config.widths:
        for similarity in config.similarities:
            for lr in config.learning_rates:
                for n_steps in config.n_steps_list:
                    for seed in range(config.n_seeds):
                        result = run_single_experiment(
                            config=config,
                            width=width,
                            similarity=similarity,
                            learning_rate=lr,
                            n_steps=n_steps,
                            seed=seed
                        )
                        results.append(result.to_dict())

                        if iterator is not None:
                            iterator.update(1)

    if iterator is not None:
        iterator.close()

    df = pd.DataFrame(results)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        # Also save config
        config_path = output_path.parent / f"{output_path.stem}_config.json"
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)

    return df


def load_dataset(path: Path) -> tuple:
    """Load dataset and config from files."""
    df = pd.read_csv(path)
    config_path = path.parent / f"{path.stem}_config.json"

    if config_path.exists():
        with open(config_path, 'r') as f:
            config = ExperimentConfig.from_dict(json.load(f))
    else:
        config = None

    return df, config
