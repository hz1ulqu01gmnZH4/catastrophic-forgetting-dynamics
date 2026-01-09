"""
Gradient Projection Methods for Mitigating Catastrophic Forgetting (Phase 6.2).

Implements three gradient modification strategies:
1. OGD (Orthogonal Gradient Descent) - Project T2 gradient orthogonal to T1 gradients
2. A-GEM (Averaged Gradient Episodic Memory) - Project only if destructive
3. Gradient Scaling - Scale gradient by alignment factor

Based on Phase 5.1 finding: Forgetting ∝ -cos(∇L_T1, ∇L_T2)
These methods aim to block or reduce destructive gradient interference.

References:
- OGD: Farajtabar et al. (2020) "Orthogonal Gradient Descent for Continual Learning"
- A-GEM: Chaudhry et al. (2019) "Efficient Lifelong Learning with A-GEM"
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
from pathlib import Path
import json

from .models import (
    LinearTeacher,
    LinearStudent,
    create_task_pair,
    evaluate_on_task
)


class ProjectionMethod(Enum):
    """Available gradient projection methods."""
    NONE = "none"           # Baseline: no projection
    OGD = "ogd"             # Orthogonal Gradient Descent
    AGEM = "agem"           # Averaged Gradient Episodic Memory
    SCALING = "scaling"     # Gradient scaling by alignment


@dataclass
class GradientMemory:
    """
    Memory buffer for storing reference gradients from previous tasks.

    For OGD: Stores gradient directions to project orthogonal to
    For A-GEM: Stores representative gradients for interference check
    """
    gradients: List[torch.Tensor] = field(default_factory=list)
    max_size: int = 100  # Maximum number of gradients to store

    def add(self, grad: torch.Tensor):
        """Add gradient to memory (FIFO if at capacity)."""
        if len(self.gradients) >= self.max_size:
            self.gradients.pop(0)
        self.gradients.append(grad.detach().clone())

    def get_reference_gradient(self) -> Optional[torch.Tensor]:
        """Get averaged reference gradient."""
        if not self.gradients:
            return None
        stacked = torch.stack(self.gradients)
        return stacked.mean(dim=0)

    def clear(self):
        """Clear memory."""
        self.gradients.clear()


def project_orthogonal(grad: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """
    Project gradient orthogonal to reference direction (OGD).

    g_proj = g - (g · r / ||r||²) * r

    This removes the component of grad that points along reference,
    keeping only the orthogonal component.
    """
    ref_norm_sq = (reference @ reference)
    if ref_norm_sq < 1e-10:
        return grad  # Reference is zero, return unchanged

    projection_coeff = (grad @ reference) / ref_norm_sq
    projected = grad - projection_coeff * reference
    return projected


def project_agem(grad: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """
    A-GEM projection: Only project if gradient is destructive.

    If grad · reference < 0 (destructive interference):
        Project grad to be orthogonal to reference
    Else:
        Keep grad unchanged (constructive or orthogonal)

    This is more permissive than OGD - only blocks harmful updates.
    """
    dot_product = grad @ reference

    if dot_product >= 0:
        # Constructive or orthogonal - no projection needed
        return grad

    # Destructive - project orthogonal
    ref_norm_sq = (reference @ reference)
    if ref_norm_sq < 1e-10:
        return grad

    projection_coeff = dot_product / ref_norm_sq
    projected = grad - projection_coeff * reference
    return projected


def scale_by_alignment(grad: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """
    Scale gradient by alignment factor.

    scaling = 1 - |cos(grad, reference)|

    When perfectly aligned (cos=1) or anti-aligned (cos=-1): scaling=0
    When orthogonal (cos=0): scaling=1

    This softly reduces gradients that would interfere.
    """
    grad_norm = grad.norm()
    ref_norm = reference.norm()

    if grad_norm < 1e-10 or ref_norm < 1e-10:
        return grad

    cos_angle = (grad @ reference) / (grad_norm * ref_norm)
    scaling = 1.0 - abs(cos_angle.item())

    return grad * scaling


def apply_projection(
    grad: torch.Tensor,
    reference: Optional[torch.Tensor],
    method: ProjectionMethod
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Apply gradient projection method.

    Returns:
        (projected_gradient, metrics_dict)
    """
    if reference is None or method == ProjectionMethod.NONE:
        return grad, {'projection_applied': False, 'grad_change': 0.0}

    original_norm = grad.norm().item()

    if method == ProjectionMethod.OGD:
        projected = project_orthogonal(grad, reference)
    elif method == ProjectionMethod.AGEM:
        projected = project_agem(grad, reference)
    elif method == ProjectionMethod.SCALING:
        projected = scale_by_alignment(grad, reference)
    else:
        raise ValueError(f"Unknown projection method: {method}")

    projected_norm = projected.norm().item()
    grad_change = abs(projected_norm - original_norm) / max(original_norm, 1e-10)

    # Compute angle between original and reference
    ref_norm = reference.norm().item()
    if original_norm > 1e-10 and ref_norm > 1e-10:
        cos_angle = ((grad @ reference) / (original_norm * ref_norm)).item()
    else:
        cos_angle = 0.0

    metrics = {
        'projection_applied': True,
        'original_norm': original_norm,
        'projected_norm': projected_norm,
        'grad_change': grad_change,
        'cos_angle_with_ref': cos_angle,
        'was_destructive': cos_angle < 0,
    }

    return projected, metrics


def train_with_projection(
    student: LinearStudent,
    teacher_train: LinearTeacher,
    teacher_reference: LinearTeacher,
    n_steps: int,
    lr: float,
    method: ProjectionMethod,
    memory: GradientMemory,
    batch_size: int = 64,
    memory_update_freq: int = 10,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Train student with gradient projection.

    Args:
        student: Student network to train
        teacher_train: Teacher for current task (Task 2)
        teacher_reference: Teacher for reference task (Task 1)
        n_steps: Number of training steps
        lr: Learning rate
        method: Projection method to use
        memory: Gradient memory (populated during Task 1)
        batch_size: Batch size
        memory_update_freq: How often to update reference gradient
        device: Device

    Returns:
        Training metrics including projection statistics
    """
    optimizer = torch.optim.SGD(student.parameters(), lr=lr)

    metrics = {
        'projection_count': 0,
        'destructive_count': 0,
        'total_grad_change': 0.0,
        'cos_angles': [],
    }

    for step in range(n_steps):
        # Get reference gradient from memory
        reference = memory.get_reference_gradient()

        # Compute gradient for current task
        optimizer.zero_grad()
        X, y = teacher_train.generate_data(batch_size, noise_std=0.0, device=device)
        y_pred = student(X)
        loss = ((y_pred - y) ** 2).mean()
        loss.backward()

        # Get and flatten gradient
        grad = student.weights.grad.detach().clone().flatten()

        # Apply projection
        projected_grad, proj_metrics = apply_projection(grad, reference, method)

        # Update metrics
        if proj_metrics['projection_applied']:
            metrics['projection_count'] += 1
            metrics['total_grad_change'] += proj_metrics['grad_change']
            metrics['cos_angles'].append(proj_metrics['cos_angle_with_ref'])
            if proj_metrics.get('was_destructive', False):
                metrics['destructive_count'] += 1

        # Apply projected gradient
        with torch.no_grad():
            student.weights.grad.copy_(projected_grad.view_as(student.weights))

        optimizer.step()

    # Compute summary statistics
    if metrics['cos_angles']:
        metrics['mean_cos_angle'] = float(np.mean(metrics['cos_angles']))
        metrics['mean_grad_change'] = metrics['total_grad_change'] / max(metrics['projection_count'], 1)
    else:
        metrics['mean_cos_angle'] = 0.0
        metrics['mean_grad_change'] = 0.0

    del metrics['cos_angles']  # Don't store full list

    return metrics


def collect_task1_gradients(
    student: LinearStudent,
    teacher: LinearTeacher,
    memory: GradientMemory,
    n_samples: int = 50,
    batch_size: int = 64,
    device: str = "cpu"
):
    """
    Collect gradient samples from Task 1 for memory.

    Called after Task 1 training to populate gradient memory.
    """
    memory.clear()

    for _ in range(n_samples):
        student.zero_grad()
        X, y = teacher.generate_data(batch_size, noise_std=0.0, device=device)
        y_pred = student(X)
        loss = ((y_pred - y) ** 2).mean()
        loss.backward()

        grad = student.weights.grad.detach().clone().flatten()
        memory.add(grad)


@dataclass
class ProjectionExperimentResult:
    """Results from a single projection experiment."""

    # Configuration
    similarity: float
    learning_rate: float
    n_steps: int
    method: str
    seed: int
    memory_size: int

    # Forgetting metrics
    forgetting: float
    forgetting_reduction: float  # vs. baseline (no projection)
    loss_t1_after_t1: float
    loss_t1_after_t2: float
    loss_t2_after_t2: float

    # Projection metrics
    projection_count: int
    destructive_count: int
    mean_cos_angle: float
    mean_grad_change: float

    # Computational cost
    relative_time: float  # Time relative to baseline

    def to_dict(self) -> Dict[str, Any]:
        return {
            'similarity': self.similarity,
            'learning_rate': self.learning_rate,
            'n_steps': self.n_steps,
            'method': self.method,
            'seed': self.seed,
            'memory_size': self.memory_size,
            'forgetting': self.forgetting,
            'forgetting_reduction': self.forgetting_reduction,
            'loss_t1_after_t1': self.loss_t1_after_t1,
            'loss_t1_after_t2': self.loss_t1_after_t2,
            'loss_t2_after_t2': self.loss_t2_after_t2,
            'projection_count': self.projection_count,
            'destructive_count': self.destructive_count,
            'mean_cos_angle': self.mean_cos_angle,
            'mean_grad_change': self.mean_grad_change,
            'relative_time': self.relative_time,
        }


def run_projection_experiment(
    d_in: int,
    d_out: int,
    similarity: float,
    learning_rate: float,
    n_steps: int,
    method: ProjectionMethod,
    seed: int,
    memory_size: int = 50,
    batch_size: int = 64,
    n_eval_samples: int = 1000,
    device: str = "cpu",
    baseline_forgetting: Optional[float] = None
) -> ProjectionExperimentResult:
    """
    Run single gradient projection experiment.

    1. Train on Task 1
    2. Collect Task 1 gradients into memory
    3. Train on Task 2 with gradient projection
    4. Measure forgetting reduction
    """
    import time

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

    # Create gradient memory
    memory = GradientMemory(max_size=memory_size)

    # Train on Task 1 (standard training, no projection)
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

    # Collect Task 1 gradients into memory
    collect_task1_gradients(student, teacher1, memory, n_samples=memory_size,
                           batch_size=batch_size, device=device)

    # Train on Task 2 with projection
    start_time = time.time()
    proj_metrics = train_with_projection(
        student=student,
        teacher_train=teacher2,
        teacher_reference=teacher1,
        n_steps=n_steps,
        lr=learning_rate,
        method=method,
        memory=memory,
        batch_size=batch_size,
        device=device
    )
    elapsed_time = time.time() - start_time

    # Evaluate after Task 2
    loss_t1_after_t2 = evaluate_on_task(student, teacher1, n_eval_samples, device)
    loss_t2_after_t2 = evaluate_on_task(student, teacher2, n_eval_samples, device)

    # Compute forgetting
    forgetting = loss_t1_after_t2 - loss_t1_after_t1

    # Compute forgetting reduction if baseline provided
    if baseline_forgetting is not None and baseline_forgetting > 0:
        forgetting_reduction = 1.0 - (forgetting / baseline_forgetting)
    else:
        forgetting_reduction = 0.0

    return ProjectionExperimentResult(
        similarity=similarity,
        learning_rate=learning_rate,
        n_steps=n_steps,
        method=method.value,
        seed=seed,
        memory_size=memory_size,
        forgetting=forgetting,
        forgetting_reduction=forgetting_reduction,
        loss_t1_after_t1=loss_t1_after_t1,
        loss_t1_after_t2=loss_t1_after_t2,
        loss_t2_after_t2=loss_t2_after_t2,
        projection_count=proj_metrics['projection_count'],
        destructive_count=proj_metrics['destructive_count'],
        mean_cos_angle=proj_metrics['mean_cos_angle'],
        mean_grad_change=proj_metrics['mean_grad_change'],
        relative_time=elapsed_time,  # Will be normalized later
    )


@dataclass
class Phase6Config:
    """Configuration for Phase 6.2 experiment."""

    d_in: int = 100
    d_out: int = 10

    # Sweep parameters
    similarities: List[float] = field(default_factory=lambda:
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    learning_rates: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.1])
    n_steps_list: List[int] = field(default_factory=lambda: [500])
    memory_sizes: List[int] = field(default_factory=lambda: [10, 50, 100])

    # Methods to compare
    methods: List[str] = field(default_factory=lambda:
        ['none', 'ogd', 'agem', 'scaling'])

    n_seeds: int = 5
    batch_size: int = 64
    n_eval_samples: int = 1000
    device: str = "cpu"

    def total_runs(self) -> int:
        return (
            len(self.similarities) *
            len(self.learning_rates) *
            len(self.n_steps_list) *
            len(self.memory_sizes) *
            len(self.methods) *
            self.n_seeds
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'd_in': self.d_in,
            'd_out': self.d_out,
            'similarities': self.similarities,
            'learning_rates': self.learning_rates,
            'n_steps_list': self.n_steps_list,
            'memory_sizes': self.memory_sizes,
            'methods': self.methods,
            'n_seeds': self.n_seeds,
            'batch_size': self.batch_size,
            'n_eval_samples': self.n_eval_samples,
            'device': self.device,
        }
