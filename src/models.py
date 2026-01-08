"""
Linear network models for teacher-student catastrophic forgetting experiments.

This module implements the core neural network components:
- Teacher networks (ground truth functions)
- Student networks (learners)
- Task similarity generation
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class TaskPair:
    """Container for a pair of related tasks with known similarity."""
    teacher1: torch.Tensor  # Weight matrix for task 1
    teacher2: torch.Tensor  # Weight matrix for task 2
    similarity: float       # Cosine similarity between tasks
    d_in: int
    d_out: int


class LinearTeacher:
    """
    Linear teacher network: y = W* @ x

    Generates ground-truth labels for student training.
    """

    def __init__(self, weights: torch.Tensor):
        """
        Args:
            weights: Teacher weight matrix of shape (d_out, d_in)
        """
        self.weights = weights
        self.d_out, self.d_in = weights.shape

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Generate labels: y = W* @ x"""
        return x @ self.weights.T

    def generate_data(
        self,
        n_samples: int,
        noise_std: float = 0.0,
        device: str = "cpu"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate training data from this teacher.

        Args:
            n_samples: Number of samples to generate
            noise_std: Standard deviation of label noise
            device: Device for tensors

        Returns:
            (X, y) tuple of input features and labels
        """
        X = torch.randn(n_samples, self.d_in, device=device)
        y = self(X)

        if noise_std > 0:
            y = y + torch.randn_like(y) * noise_std

        return X, y


class LinearStudent(nn.Module):
    """
    Linear student network: y = W @ x

    Learns to approximate teacher functions via gradient descent.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        init_scale: float = 1.0,
        device: str = "cpu"
    ):
        """
        Args:
            d_in: Input dimension
            d_out: Output dimension
            init_scale: Scale factor for weight initialization
            device: Device for parameters
        """
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out

        # Initialize weights with controlled scale
        W_init = torch.randn(d_out, d_in, device=device) * init_scale / (d_in ** 0.5)
        self.weights = nn.Parameter(W_init)

        # Store initial weights for feature learning rate computation
        self._initial_weights = W_init.clone().detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: y = W @ x"""
        return x @ self.weights.T

    def get_weight_change(self) -> float:
        """Compute L2 distance from initial weights."""
        return (self.weights - self._initial_weights).norm().item()

    def get_weight_change_normalized(self) -> float:
        """Compute normalized weight change (feature learning rate proxy)."""
        change = (self.weights - self._initial_weights).norm()
        init_norm = self._initial_weights.norm()
        if init_norm < 1e-8:
            return float('inf')
        return (change / init_norm).item()

    def reset_initial_weights(self):
        """Update initial weights reference (call after task 1 training)."""
        self._initial_weights = self.weights.clone().detach()


def create_task_pair(
    d_in: int,
    d_out: int,
    similarity: float,
    normalize: bool = True,
    device: str = "cpu"
) -> TaskPair:
    """
    Create two teacher weight matrices with specified similarity.

    Uses Gram-Schmidt-like orthogonalization to control exact similarity.

    Args:
        d_in: Input dimension
        d_out: Output dimension
        similarity: Target cosine similarity in [0, 1]
        normalize: Whether to normalize teacher weights
        device: Device for tensors

    Returns:
        TaskPair containing both teachers and metadata
    """
    if not 0 <= similarity <= 1:
        raise ValueError(f"Similarity must be in [0, 1], got {similarity}")

    # First teacher: random matrix
    W1 = torch.randn(d_out, d_in, device=device)
    if normalize:
        W1 = W1 / W1.norm()

    # Second teacher: interpolate between W1 and orthogonal random
    W_random = torch.randn(d_out, d_in, device=device)

    # Gram-Schmidt: make W_random orthogonal to W1
    W1_flat = W1.flatten()
    W_random_flat = W_random.flatten()
    projection = (W_random_flat @ W1_flat) / (W1_flat @ W1_flat)
    W_orthogonal_flat = W_random_flat - projection * W1_flat
    W_orthogonal = W_orthogonal_flat.reshape(d_out, d_in)

    if normalize:
        W_orthogonal = W_orthogonal / W_orthogonal.norm()

    # Interpolate: W2 = similarity * W1 + sqrt(1 - similarity^2) * W_orthogonal
    # This gives exact cosine similarity = similarity
    W2 = similarity * W1 + (1 - similarity**2)**0.5 * W_orthogonal

    if normalize:
        W2 = W2 / W2.norm()

    # Verify similarity
    actual_sim = (W1.flatten() @ W2.flatten()) / (W1.norm() * W2.norm())
    assert abs(actual_sim.item() - similarity) < 1e-5, \
        f"Similarity mismatch: expected {similarity}, got {actual_sim.item()}"

    return TaskPair(
        teacher1=W1,
        teacher2=W2,
        similarity=similarity,
        d_in=d_in,
        d_out=d_out
    )


def train_step(
    student: LinearStudent,
    X: torch.Tensor,
    y: torch.Tensor,
    optimizer: torch.optim.Optimizer
) -> float:
    """
    Single training step.

    Args:
        student: Student network
        X: Input batch
        y: Target batch
        optimizer: Optimizer

    Returns:
        Loss value
    """
    optimizer.zero_grad()
    y_pred = student(X)
    loss = ((y_pred - y) ** 2).mean()
    loss.backward()
    optimizer.step()
    return loss.item()


def train_on_task(
    student: LinearStudent,
    teacher: LinearTeacher,
    n_steps: int,
    lr: float,
    batch_size: int = 64,
    noise_std: float = 0.0,
    device: str = "cpu"
) -> list:
    """
    Train student on a single task.

    Args:
        student: Student network
        teacher: Teacher to learn from
        n_steps: Number of training steps
        lr: Learning rate
        batch_size: Batch size for SGD
        noise_std: Label noise
        device: Device

    Returns:
        List of loss values during training
    """
    optimizer = torch.optim.SGD(student.parameters(), lr=lr)
    losses = []

    for step in range(n_steps):
        X, y = teacher.generate_data(batch_size, noise_std=noise_std, device=device)
        loss = train_step(student, X, y, optimizer)
        losses.append(loss)

    return losses


def evaluate_on_task(
    student: LinearStudent,
    teacher: LinearTeacher,
    n_samples: int = 1000,
    device: str = "cpu"
) -> float:
    """
    Evaluate student performance on a task.

    Args:
        student: Student network
        teacher: Teacher defining the task
        n_samples: Number of evaluation samples
        device: Device

    Returns:
        Mean squared error
    """
    student.eval()
    with torch.no_grad():
        X, y = teacher.generate_data(n_samples, noise_std=0.0, device=device)
        y_pred = student(X)
        mse = ((y_pred - y) ** 2).mean().item()
    student.train()
    return mse
