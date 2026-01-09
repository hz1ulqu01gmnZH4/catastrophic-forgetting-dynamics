"""
Gradient Interference Analysis for Catastrophic Forgetting (Phase 5.1).

This module measures how Task 2 gradients interfere with Task 1 solution:
- Gradient alignment: cos(∇L_T1, ∇L_T2)
- Gradient projection: ∇L_T2 · ∇L_T1 / ||∇L_T1||²
- Interference magnitude: ||∇L_T2 projected onto T1 gradient subspace||

Hypothesis: Forgetting ∝ -cos(∇L_T1, ∇L_T2)
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import json

from .models import (
    LinearTeacher,
    LinearStudent,
    create_task_pair,
    evaluate_on_task
)


@dataclass
class GradientMetrics:
    """Gradient interference metrics at a single training step."""
    step: int

    # Core gradient alignment metrics
    gradient_angle: float           # cos(∇L_T1, ∇L_T2) - alignment between task gradients
    gradient_projection: float      # ∇L_T2 · ∇L_T1 / ||∇L_T1||² - T2 grad projection onto T1
    interference_magnitude: float   # ||∇L_T2|| * |cos(∇L_T1, ∇L_T2)| - destructive component

    # Gradient norms
    grad_t1_norm: float            # ||∇L_T1||
    grad_t2_norm: float            # ||∇L_T2||

    # Loss values at this step
    loss_t1: float                 # Current loss on Task 1
    loss_t2: float                 # Current loss on Task 2

    def to_dict(self) -> Dict[str, float]:
        return {
            'step': self.step,
            'gradient_angle': self.gradient_angle,
            'gradient_projection': self.gradient_projection,
            'interference_magnitude': self.interference_magnitude,
            'grad_t1_norm': self.grad_t1_norm,
            'grad_t2_norm': self.grad_t2_norm,
            'loss_t1': self.loss_t1,
            'loss_t2': self.loss_t2,
        }


@dataclass
class GradientInterferenceResult:
    """Full results from gradient interference analysis."""

    # Experiment parameters
    similarity: float
    learning_rate: float
    n_steps: int
    seed: int

    # Forgetting outcome
    forgetting: float
    loss_t1_after_t1: float
    loss_t1_after_t2: float
    loss_t2_after_t2: float

    # Aggregated gradient metrics (during T2 training)
    mean_gradient_angle: float           # Average cos(∇L_T1, ∇L_T2) over T2 training
    min_gradient_angle: float            # Minimum (most negative = most destructive)
    max_gradient_angle: float            # Maximum (most aligned)
    std_gradient_angle: float            # Variability in alignment

    mean_gradient_projection: float      # Average projection strength
    cumulative_interference: float       # Sum of interference over all steps

    # Early vs late training comparison
    early_gradient_angle: float          # Average angle in first 20% of T2 training
    late_gradient_angle: float           # Average angle in last 20% of T2 training

    # Per-step metrics (optional, for detailed analysis)
    step_metrics: List[GradientMetrics] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'similarity': self.similarity,
            'learning_rate': self.learning_rate,
            'n_steps': self.n_steps,
            'seed': self.seed,
            'forgetting': self.forgetting,
            'loss_t1_after_t1': self.loss_t1_after_t1,
            'loss_t1_after_t2': self.loss_t1_after_t2,
            'loss_t2_after_t2': self.loss_t2_after_t2,
            'mean_gradient_angle': self.mean_gradient_angle,
            'min_gradient_angle': self.min_gradient_angle,
            'max_gradient_angle': self.max_gradient_angle,
            'std_gradient_angle': self.std_gradient_angle,
            'mean_gradient_projection': self.mean_gradient_projection,
            'cumulative_interference': self.cumulative_interference,
            'early_gradient_angle': self.early_gradient_angle,
            'late_gradient_angle': self.late_gradient_angle,
        }


def compute_gradient(
    student: LinearStudent,
    teacher: LinearTeacher,
    batch_size: int = 64,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Compute gradient of loss with respect to student weights.

    Returns flattened gradient vector.
    """
    student.zero_grad()
    X, y = teacher.generate_data(batch_size, noise_std=0.0, device=device)
    y_pred = student(X)
    loss = ((y_pred - y) ** 2).mean()
    loss.backward()

    # Get gradient as flat vector
    grad = student.weights.grad.detach().clone().flatten()
    return grad, loss.item()


def compute_gradient_alignment(grad1: torch.Tensor, grad2: torch.Tensor) -> Tuple[float, float, float]:
    """
    Compute gradient alignment metrics.

    Args:
        grad1: Gradient for Task 1 (reference direction)
        grad2: Gradient for Task 2 (current training direction)

    Returns:
        (cos_angle, projection, interference_magnitude)

    Raises:
        ValueError: If either gradient has near-zero norm (degenerate case)
    """
    norm1 = grad1.norm()
    norm2 = grad2.norm()

    if norm1 < 1e-10 or norm2 < 1e-10:
        raise ValueError(
            f"Degenerate gradient detected: ||grad1||={norm1.item():.2e}, ||grad2||={norm2.item():.2e}. "
            f"Cannot compute gradient alignment with zero gradient."
        )

    # Cosine similarity (gradient angle)
    cos_angle = (grad1 @ grad2) / (norm1 * norm2)
    cos_angle = cos_angle.item()

    # Projection: ∇L_T2 · ∇L_T1 / ||∇L_T1||²
    # Positive = T2 grad points toward T1 minimum
    # Negative = T2 grad points away from T1 minimum
    projection = (grad2 @ grad1) / (norm1 ** 2)
    projection = projection.item()

    # Interference magnitude: how much T2 update affects T1 direction
    # = ||∇L_T2|| * |cos(angle)| when cos < 0 (destructive)
    # = 0 when cos >= 0 (constructive or orthogonal)
    if cos_angle < 0:
        interference = norm2.item() * abs(cos_angle)
    else:
        interference = 0.0

    return cos_angle, projection, interference


def train_with_gradient_logging(
    student: LinearStudent,
    teacher_train: LinearTeacher,
    teacher_reference: LinearTeacher,
    n_steps: int,
    lr: float,
    batch_size: int = 64,
    log_every: int = 1,
    device: str = "cpu"
) -> List[GradientMetrics]:
    """
    Train student while logging gradient alignment with reference task.

    Args:
        student: Student network to train
        teacher_train: Teacher for training (Task 2)
        teacher_reference: Teacher for reference gradients (Task 1)
        n_steps: Number of training steps
        lr: Learning rate
        batch_size: Batch size
        log_every: Log metrics every N steps
        device: Device

    Returns:
        List of GradientMetrics for each logged step
    """
    optimizer = torch.optim.SGD(student.parameters(), lr=lr)
    metrics_list = []

    for step in range(n_steps):
        # Compute gradients for both tasks (before update)
        grad_t1, loss_t1 = compute_gradient(student, teacher_reference, batch_size, device)
        grad_t2, loss_t2 = compute_gradient(student, teacher_train, batch_size, device)

        # Log metrics
        if step % log_every == 0:
            cos_angle, projection, interference = compute_gradient_alignment(grad_t1, grad_t2)

            metrics = GradientMetrics(
                step=step,
                gradient_angle=cos_angle,
                gradient_projection=projection,
                interference_magnitude=interference,
                grad_t1_norm=grad_t1.norm().item(),
                grad_t2_norm=grad_t2.norm().item(),
                loss_t1=loss_t1,
                loss_t2=loss_t2
            )
            metrics_list.append(metrics)

        # Perform actual training step on Task 2
        optimizer.zero_grad()
        X, y = teacher_train.generate_data(batch_size, noise_std=0.0, device=device)
        y_pred = student(X)
        loss = ((y_pred - y) ** 2).mean()
        loss.backward()
        optimizer.step()

    return metrics_list


def run_gradient_interference_experiment(
    d_in: int,
    d_out: int,
    similarity: float,
    learning_rate: float,
    n_steps: int,
    seed: int,
    batch_size: int = 64,
    n_eval_samples: int = 1000,
    log_every: int = 1,
    device: str = "cpu"
) -> GradientInterferenceResult:
    """
    Run single gradient interference experiment.

    1. Create task pair with specified similarity
    2. Train on Task 1
    3. Train on Task 2 while logging gradient alignment
    4. Compute forgetting and correlate with gradient metrics
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create task pair
    task_pair = create_task_pair(
        d_in=d_in,
        d_out=d_out,
        similarity=similarity,
        normalize=True,
        device=device
    )

    teacher1 = LinearTeacher(task_pair.teacher1)
    teacher2 = LinearTeacher(task_pair.teacher2)

    # Create student
    student = LinearStudent(d_in=d_in, d_out=d_out, init_scale=1.0, device=device)

    # Train on Task 1 (no gradient logging needed)
    optimizer = torch.optim.SGD(student.parameters(), lr=learning_rate)
    for _ in range(n_steps):
        optimizer.zero_grad()
        X, y = teacher1.generate_data(batch_size, noise_std=0.0, device=device)
        y_pred = student(X)
        loss = ((y_pred - y) ** 2).mean()
        loss.backward()
        optimizer.step()

    # Evaluate after Task 1
    loss_t1_after_t1 = evaluate_on_task(student, teacher1, n_eval_samples, device)

    # Train on Task 2 WITH gradient logging
    step_metrics = train_with_gradient_logging(
        student=student,
        teacher_train=teacher2,
        teacher_reference=teacher1,
        n_steps=n_steps,
        lr=learning_rate,
        batch_size=batch_size,
        log_every=log_every,
        device=device
    )

    # Evaluate after Task 2
    loss_t1_after_t2 = evaluate_on_task(student, teacher1, n_eval_samples, device)
    loss_t2_after_t2 = evaluate_on_task(student, teacher2, n_eval_samples, device)

    # Compute forgetting
    forgetting = loss_t1_after_t2 - loss_t1_after_t1

    # Aggregate gradient metrics
    angles = [m.gradient_angle for m in step_metrics]
    projections = [m.gradient_projection for m in step_metrics]
    interferences = [m.interference_magnitude for m in step_metrics]

    # Early vs late comparison (first/last 20%)
    n_metrics = len(step_metrics)
    early_cutoff = max(1, int(0.2 * n_metrics))
    late_cutoff = max(1, int(0.8 * n_metrics))

    early_angles = angles[:early_cutoff]
    late_angles = angles[late_cutoff:]

    return GradientInterferenceResult(
        similarity=similarity,
        learning_rate=learning_rate,
        n_steps=n_steps,
        seed=seed,
        forgetting=forgetting,
        loss_t1_after_t1=loss_t1_after_t1,
        loss_t1_after_t2=loss_t1_after_t2,
        loss_t2_after_t2=loss_t2_after_t2,
        mean_gradient_angle=float(np.mean(angles)),
        min_gradient_angle=float(np.min(angles)),
        max_gradient_angle=float(np.max(angles)),
        std_gradient_angle=float(np.std(angles)),
        mean_gradient_projection=float(np.mean(projections)),
        cumulative_interference=float(np.sum(interferences)),
        early_gradient_angle=float(np.mean(early_angles)),
        late_gradient_angle=float(np.mean(late_angles)),
        step_metrics=step_metrics
    )


@dataclass
class Phase5Config:
    """Configuration for Phase 5.1 experiment."""

    d_in: int = 100
    d_out: int = 10

    # Sweep parameters
    similarities: List[float] = field(default_factory=lambda:
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    learning_rates: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.1])
    n_steps_list: List[int] = field(default_factory=lambda: [100, 500, 1000])

    n_seeds: int = 5
    batch_size: int = 64
    n_eval_samples: int = 1000
    log_every: int = 10  # Log gradient metrics every N steps
    device: str = "cpu"

    def total_runs(self) -> int:
        return (
            len(self.similarities) *
            len(self.learning_rates) *
            len(self.n_steps_list) *
            self.n_seeds
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'd_in': self.d_in,
            'd_out': self.d_out,
            'similarities': self.similarities,
            'learning_rates': self.learning_rates,
            'n_steps_list': self.n_steps_list,
            'n_seeds': self.n_seeds,
            'batch_size': self.batch_size,
            'n_eval_samples': self.n_eval_samples,
            'log_every': self.log_every,
            'device': self.device,
        }
