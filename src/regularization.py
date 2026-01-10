"""
Regularization methods for mitigating catastrophic forgetting.

Implements:
- EWC (Elastic Weight Consolidation): λ Σ F_i (θ_i - θ_T1)²
- L2 regularization toward T1 weights: λ ||θ - θ_T1||²
- Online EWC: Running average of Fisher information
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class RegularizationMethod(Enum):
    """Available regularization methods."""
    NONE = "none"
    L2 = "l2"
    EWC = "ewc"
    ONLINE_EWC = "online_ewc"


@dataclass
class FisherInformation:
    """Stores Fisher Information for EWC."""
    fisher: Dict[str, torch.Tensor] = field(default_factory=dict)
    optimal_params: Dict[str, torch.Tensor] = field(default_factory=dict)

    def compute(
        self,
        model: nn.Module,
        data_loader: List[Tuple[torch.Tensor, torch.Tensor]],
        n_samples: int = 200
    ) -> None:
        """
        Compute Fisher Information matrix (diagonal approximation).

        F_i = E[(∂log p(y|x,θ) / ∂θ_i)²]

        For MSE loss, this simplifies to gradient squared.
        """
        # Store optimal parameters
        for name, param in model.named_parameters():
            self.optimal_params[name] = param.data.clone()
            self.fisher[name] = torch.zeros_like(param.data)

        model.eval()
        n_computed = 0

        for X, y in data_loader:
            if n_computed >= n_samples:
                break

            batch_size = min(X.shape[0], n_samples - n_computed)
            X = X[:batch_size]
            y = y[:batch_size]

            # Compute gradients for each sample
            for i in range(batch_size):
                model.zero_grad()
                output = model(X[i:i+1])
                loss = nn.functional.mse_loss(output, y[i:i+1])
                loss.backward()

                # Accumulate squared gradients
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        self.fisher[name] += param.grad.data ** 2

                n_computed += 1

        # Normalize by number of samples
        if n_computed == 0:
            raise ValueError("No samples computed for Fisher Information")

        for name in self.fisher:
            self.fisher[name] /= n_computed


@dataclass
class OnlineFisherInformation:
    """Online/running average Fisher Information for Online EWC."""
    fisher: Dict[str, torch.Tensor] = field(default_factory=dict)
    optimal_params: Dict[str, torch.Tensor] = field(default_factory=dict)
    gamma: float = 0.95  # Decay factor for running average
    task_count: int = 0

    def update(
        self,
        model: nn.Module,
        data_loader: List[Tuple[torch.Tensor, torch.Tensor]],
        n_samples: int = 200
    ) -> None:
        """Update Fisher with new task's information."""
        # Compute new Fisher
        new_fisher = FisherInformation()
        new_fisher.compute(model, data_loader, n_samples)

        if self.task_count == 0:
            # First task: just copy
            self.fisher = new_fisher.fisher
            self.optimal_params = new_fisher.optimal_params
        else:
            # Running average
            for name in new_fisher.fisher:
                self.fisher[name] = (
                    self.gamma * self.fisher[name] +
                    (1 - self.gamma) * new_fisher.fisher[name]
                )
                self.optimal_params[name] = new_fisher.optimal_params[name]

        self.task_count += 1


def compute_ewc_loss(
    model: nn.Module,
    fisher_info: FisherInformation,
    lambda_ewc: float
) -> torch.Tensor:
    """
    Compute EWC regularization loss.

    L_EWC = (λ/2) Σ_i F_i (θ_i - θ*_i)²
    """
    ewc_loss = torch.tensor(0.0, device=next(model.parameters()).device)

    for name, param in model.named_parameters():
        if name in fisher_info.fisher:
            fisher = fisher_info.fisher[name]
            optimal = fisher_info.optimal_params[name]
            ewc_loss += (fisher * (param - optimal) ** 2).sum()

    return (lambda_ewc / 2) * ewc_loss


def compute_l2_loss(
    model: nn.Module,
    reference_params: Dict[str, torch.Tensor],
    lambda_l2: float
) -> torch.Tensor:
    """
    Compute L2 regularization loss toward reference weights.

    L_L2 = (λ/2) ||θ - θ_ref||²
    """
    l2_loss = torch.tensor(0.0, device=next(model.parameters()).device)

    for name, param in model.named_parameters():
        if name in reference_params:
            ref = reference_params[name]
            l2_loss += ((param - ref) ** 2).sum()

    return (lambda_l2 / 2) * l2_loss


def check_gradient_explosion(loss: torch.Tensor) -> bool:
    """Check if loss indicates gradient explosion (NaN or Inf)."""
    return not torch.isfinite(loss).item()


@dataclass
class RegularizationResult:
    """Result from a regularization experiment."""
    method: str
    lambda_value: float
    similarity: float
    learning_rate: float
    seed: int
    forgetting: float
    forgetting_reduction: float
    loss_t1_after_t1: float
    loss_t1_after_t2: float
    loss_t2_after_t2: float
    ewc_loss_mean: float  # Mean EWC/L2 penalty during T2 training
    ewc_loss_final: float  # Final EWC/L2 penalty
    relative_time: float


@dataclass
class Phase63Config:
    """Configuration for Phase 6.3 regularization experiments."""
    d_in: int = 100
    d_out: int = 10
    n_samples: int = 1000
    n_steps: int = 500
    similarities: List[float] = field(default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    learning_rates: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.1])
    lambda_values: List[float] = field(default_factory=lambda: [0.0, 0.01, 0.1, 1.0, 10.0, 100.0])
    methods: List[str] = field(default_factory=lambda: ["none", "l2", "ewc"])
    n_seeds: int = 5
    fisher_samples: int = 200


def train_with_regularization(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    n_steps: int,
    lr: float,
    method: RegularizationMethod,
    lambda_value: float,
    fisher_info: Optional[FisherInformation] = None,
    reference_params: Optional[Dict[str, torch.Tensor]] = None
) -> Tuple[List[float], List[float], float]:
    """
    Train model with regularization.

    Returns:
        (task_losses, reg_losses, relative_time)
    """
    import time

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    task_losses = []
    reg_losses = []

    start_time = time.time()

    for step in range(n_steps):
        optimizer.zero_grad()

        # Task loss
        output = model(X)
        task_loss = nn.functional.mse_loss(output, y)

        # Regularization loss
        if method == RegularizationMethod.NONE or lambda_value == 0:
            reg_loss = torch.tensor(0.0, device=X.device)
        elif method == RegularizationMethod.L2:
            if reference_params is None:
                raise ValueError("L2 regularization requires reference_params")
            reg_loss = compute_l2_loss(model, reference_params, lambda_value)
        elif method in (RegularizationMethod.EWC, RegularizationMethod.ONLINE_EWC):
            if fisher_info is None:
                raise ValueError("EWC regularization requires fisher_info")
            reg_loss = compute_ewc_loss(model, fisher_info, lambda_value)
        else:
            raise ValueError(f"Unknown regularization method: {method}")

        total_loss = task_loss + reg_loss

        # Check for gradient explosion
        if check_gradient_explosion(total_loss):
            # Return early with NaN indicators
            task_losses.append(float('nan'))
            reg_losses.append(float('nan'))
            elapsed = time.time() - start_time
            return task_losses, reg_losses, elapsed

        total_loss.backward()
        optimizer.step()

        task_losses.append(task_loss.item())
        reg_losses.append(reg_loss.item())

    elapsed = time.time() - start_time

    return task_losses, reg_losses, elapsed


def run_regularization_experiment(
    similarity: float,
    lr: float,
    method: str,
    lambda_value: float,
    seed: int,
    config: Phase63Config,
    baseline_time: Optional[float] = None
) -> RegularizationResult:
    """
    Run single regularization experiment.

    Train on T1, compute Fisher/store weights, train on T2 with regularization.
    """
    import time
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

    # Train on Task 1 (no regularization)
    start_time = time.time()
    optimizer = torch.optim.SGD(student.parameters(), lr=lr)
    for _ in range(config.n_steps):
        optimizer.zero_grad()
        output = student(X1)
        loss = nn.functional.mse_loss(output, y1)
        loss.backward()
        optimizer.step()

    t1_train_time = time.time() - start_time

    # Evaluate on T1 after T1 training
    with torch.no_grad():
        loss_t1_after_t1 = nn.functional.mse_loss(student(X1), y1).item()

    # Store reference for regularization
    reg_method = RegularizationMethod(method)
    fisher_info = None
    reference_params = None

    if reg_method == RegularizationMethod.L2:
        reference_params = {
            name: param.data.clone()
            for name, param in student.named_parameters()
        }
    elif reg_method in (RegularizationMethod.EWC, RegularizationMethod.ONLINE_EWC):
        fisher_info = FisherInformation()
        # Create data loader format
        data_loader = [(X1, y1)]
        fisher_info.compute(student, data_loader, config.fisher_samples)

    # Train on Task 2 with regularization
    task_losses, reg_losses, t2_train_time = train_with_regularization(
        model=student,
        X=X2,
        y=y2,
        n_steps=config.n_steps,
        lr=lr,
        method=reg_method,
        lambda_value=lambda_value,
        fisher_info=fisher_info,
        reference_params=reference_params
    )

    # Evaluate
    with torch.no_grad():
        loss_t1_after_t2 = nn.functional.mse_loss(student(X1), y1).item()
        loss_t2_after_t2 = nn.functional.mse_loss(student(X2), y2).item()

    forgetting = loss_t1_after_t2 - loss_t1_after_t1

    total_time = t1_train_time + t2_train_time
    relative_time = total_time / baseline_time if baseline_time else 1.0

    return RegularizationResult(
        method=method,
        lambda_value=lambda_value,
        similarity=similarity,
        learning_rate=lr,
        seed=seed,
        forgetting=forgetting,
        forgetting_reduction=0.0,  # Computed later against baseline
        loss_t1_after_t1=loss_t1_after_t1,
        loss_t1_after_t2=loss_t1_after_t2,
        loss_t2_after_t2=loss_t2_after_t2,
        ewc_loss_mean=sum(reg_losses) / len(reg_losses) if reg_losses else 0.0,
        ewc_loss_final=reg_losses[-1] if reg_losses else 0.0,
        relative_time=relative_time
    )
