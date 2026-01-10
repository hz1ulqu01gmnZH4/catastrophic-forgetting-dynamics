"""
Similarity-Aware Learning Rate for Catastrophic Forgetting Mitigation.

Hypothesis: Adapt learning rate based on task similarity.
- Lower LR for dissimilar tasks (more careful updates)
- Higher LR for similar tasks (can learn faster)

Formula: lr_t2 = base_lr * similarity^α
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import time


@dataclass
class AdaptiveLRResult:
    """Result from an adaptive LR experiment."""
    alpha: float
    similarity: float
    base_lr: float
    effective_lr: float
    seed: int
    forgetting: float
    forgetting_reduction: float
    loss_t1_after_t1: float
    loss_t1_after_t2: float
    loss_t2_after_t2: float
    relative_time: float


@dataclass
class Phase65Config:
    """Configuration for Phase 6.5 adaptive LR experiments."""
    d_in: int = 100
    d_out: int = 10
    n_samples: int = 1000
    n_steps: int = 500
    similarities: List[float] = field(default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    base_learning_rates: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.1])
    # α values: 0 = no adaptation, 0.5 = sqrt scaling, 1 = linear, 2 = quadratic
    alpha_values: List[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
    n_seeds: int = 5
    min_lr_ratio: float = 0.01  # Minimum LR as ratio of base (avoid zero LR)


def compute_adaptive_lr(
    base_lr: float,
    similarity: float,
    alpha: float,
    min_lr_ratio: float = 0.01
) -> float:
    """
    Compute similarity-adapted learning rate.

    lr_t2 = base_lr * max(similarity^α, min_lr_ratio)

    Args:
        base_lr: Base learning rate
        similarity: Task similarity (0 to 1)
        alpha: Scaling exponent (0 = no adaptation, higher = more aggressive)
        min_lr_ratio: Minimum LR as ratio of base (prevents zero LR)

    Returns:
        Adapted learning rate for Task 2
    """
    if alpha == 0:
        return base_lr

    # Avoid zero LR for zero similarity
    if similarity <= 0:
        return base_lr * min_lr_ratio

    scaled = similarity ** alpha
    return base_lr * max(scaled, min_lr_ratio)


def train_with_adaptive_lr(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    n_steps: int,
    lr: float
) -> Tuple[List[float], float]:
    """
    Train model with specified learning rate.

    Returns:
        (losses, elapsed_time)
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    losses = []

    start_time = time.time()

    for step in range(n_steps):
        optimizer.zero_grad()
        output = model(X)
        loss = nn.functional.mse_loss(output, y)

        # Check for NaN/Inf
        if not torch.isfinite(loss):
            losses.append(float('nan'))
            break

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    elapsed = time.time() - start_time
    return losses, elapsed


def run_adaptive_lr_experiment(
    similarity: float,
    base_lr: float,
    alpha: float,
    seed: int,
    config: Phase65Config,
    baseline_time: Optional[float] = None
) -> AdaptiveLRResult:
    """
    Run single adaptive LR experiment.

    Train on T1 with base_lr, train on T2 with adapted LR.
    """
    from src.models import create_task_pair, LinearStudent, LinearTeacher

    torch.manual_seed(seed)
    device = "cpu"

    # Create tasks
    task_pair = create_task_pair(
        d_in=config.d_in,
        d_out=config.d_out,
        similarity=similarity,
        device=device
    )

    teacher1 = LinearTeacher(task_pair.teacher1)
    teacher2 = LinearTeacher(task_pair.teacher2)

    # Generate data
    X1, y1 = teacher1.generate_data(config.n_samples, device=device)
    X2, y2 = teacher2.generate_data(config.n_samples, device=device)

    # Create student
    student = LinearStudent(config.d_in, config.d_out, device=device)

    # Train on Task 1 with base LR
    _, t1_time = train_with_adaptive_lr(
        model=student,
        X=X1,
        y=y1,
        n_steps=config.n_steps,
        lr=base_lr
    )

    # Evaluate on T1 after T1 training
    with torch.no_grad():
        loss_t1_after_t1 = nn.functional.mse_loss(student(X1), y1).item()

    # Compute adaptive LR for Task 2
    effective_lr = compute_adaptive_lr(
        base_lr=base_lr,
        similarity=similarity,
        alpha=alpha,
        min_lr_ratio=config.min_lr_ratio
    )

    # Train on Task 2 with adaptive LR
    _, t2_time = train_with_adaptive_lr(
        model=student,
        X=X2,
        y=y2,
        n_steps=config.n_steps,
        lr=effective_lr
    )

    # Evaluate
    with torch.no_grad():
        loss_t1_after_t2 = nn.functional.mse_loss(student(X1), y1).item()
        loss_t2_after_t2 = nn.functional.mse_loss(student(X2), y2).item()

    forgetting = loss_t1_after_t2 - loss_t1_after_t1

    total_time = t1_time + t2_time
    relative_time = total_time / baseline_time if baseline_time else 1.0

    return AdaptiveLRResult(
        alpha=alpha,
        similarity=similarity,
        base_lr=base_lr,
        effective_lr=effective_lr,
        seed=seed,
        forgetting=forgetting,
        forgetting_reduction=0.0,  # Computed later against baseline
        loss_t1_after_t1=loss_t1_after_t1,
        loss_t1_after_t2=loss_t1_after_t2,
        loss_t2_after_t2=loss_t2_after_t2,
        relative_time=relative_time
    )


def theoretical_optimal_alpha(forgetting_coefficient: float = 0.65) -> float:
    """
    Derive theoretical optimal α from forgetting equation.

    From Phase 1: Forgetting ≈ 0.59 - 0.65 × similarity

    If we want to reduce forgetting proportionally:
    - When similarity is low, we need smaller LR updates
    - The gradient of forgetting w.r.t. similarity is -0.65

    Optimal α should balance:
    - Lower LR for dissimilar tasks (reduce interference)
    - Not too low (need to learn T2)

    Heuristic: α ≈ 1 / |gradient| = 1 / 0.65 ≈ 1.5
    """
    return 1.0 / abs(forgetting_coefficient)
