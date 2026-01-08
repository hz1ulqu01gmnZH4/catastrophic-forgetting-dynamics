"""
Nonlinear network models for Phase 2 experiments.

Extends linear baseline with:
- Single hidden layer with nonlinear activations
- Feature learning rate (FLR) measurement
- NTK alignment tracking
- Lazy-rich regime detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List, Callable
from dataclasses import dataclass
import numpy as np


@dataclass
class NonlinearTaskPair:
    """Container for nonlinear teacher-student task pair."""
    teacher1_W1: torch.Tensor  # First layer weights for task 1
    teacher1_W2: torch.Tensor  # Second layer weights for task 1
    teacher2_W1: torch.Tensor  # First layer weights for task 2
    teacher2_W2: torch.Tensor  # Second layer weights for task 2
    similarity: float
    d_in: int
    d_hidden: int
    d_out: int
    activation: str


class NonlinearTeacher:
    """
    Two-layer nonlinear teacher: y = W2 @ σ(W1 @ x)
    """

    def __init__(
        self,
        W1: torch.Tensor,
        W2: torch.Tensor,
        activation: str = "relu"
    ):
        self.W1 = W1  # (d_hidden, d_in)
        self.W2 = W2  # (d_out, d_hidden)
        self.activation = activation
        self.d_hidden, self.d_in = W1.shape
        self.d_out = W2.shape[0]

    def _activate(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "relu":
            return F.relu(x)
        elif self.activation == "tanh":
            return torch.tanh(x)
        elif self.activation == "gelu":
            return F.gelu(x)
        elif self.activation == "sigmoid":
            return torch.sigmoid(x)
        elif self.activation == "linear":
            return x
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        h = self._activate(x @ self.W1.T)
        return h @ self.W2.T

    def generate_data(
        self,
        n_samples: int,
        noise_std: float = 0.0,
        device: str = "cpu"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        X = torch.randn(n_samples, self.d_in, device=device)
        y = self(X)
        if noise_std > 0:
            y = y + torch.randn_like(y) * noise_std
        return X, y


class NonlinearStudent(nn.Module):
    """
    Two-layer nonlinear student: y = W2 @ σ(W1 @ x)

    Tracks feature learning rate and NTK alignment.
    """

    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        d_out: int,
        activation: str = "relu",
        init_scale: float = 1.0,
        device: str = "cpu"
    ):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.d_out = d_out
        self.activation = activation
        self.device = device

        # NTK parameterization: scale by 1/sqrt(fan_in)
        self.W1 = nn.Parameter(
            torch.randn(d_hidden, d_in, device=device) * init_scale / np.sqrt(d_in)
        )
        self.W2 = nn.Parameter(
            torch.randn(d_out, d_hidden, device=device) * init_scale / np.sqrt(d_hidden)
        )

        # Store initial weights for FLR computation
        self._W1_init = self.W1.clone().detach()
        self._W2_init = self.W2.clone().detach()

        # Store weights after task 1 for task 2 comparison
        self._W1_after_t1 = None
        self._W2_after_t1 = None

    def _activate(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "relu":
            return F.relu(x)
        elif self.activation == "tanh":
            return torch.tanh(x)
        elif self.activation == "gelu":
            return F.gelu(x)
        elif self.activation == "sigmoid":
            return torch.sigmoid(x)
        elif self.activation == "linear":
            return x
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._activate(x @ self.W1.T)
        return h @ self.W2.T

    def get_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Get hidden layer activations."""
        with torch.no_grad():
            return self._activate(x @ self.W1.T)

    def get_hidden_with_init_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Get hidden activations using initial weights."""
        with torch.no_grad():
            return self._activate(x @ self._W1_init.T)

    def compute_feature_learning_rate(self, X_probe: torch.Tensor) -> float:
        """
        Compute Feature Learning Rate (FLR) using CKA.

        FLR = 1 - CKA(K_init, K_current)

        Where K is the kernel matrix of hidden representations.
        Low FLR = lazy regime (features don't change)
        High FLR = rich regime (features change significantly)
        """
        h_init = self.get_hidden_with_init_weights(X_probe)
        h_current = self.get_hidden(X_probe)

        # Compute kernel matrices
        K_init = h_init @ h_init.T
        K_current = h_current @ h_current.T

        # Center kernels
        n = K_init.shape[0]
        H = torch.eye(n, device=self.device) - torch.ones(n, n, device=self.device) / n
        K_init_c = H @ K_init @ H
        K_current_c = H @ K_current @ H

        # Compute CKA
        hsic_init_current = (K_init_c * K_current_c).sum()
        hsic_init = (K_init_c * K_init_c).sum()
        hsic_current = (K_current_c * K_current_c).sum()

        denom = torch.sqrt(hsic_init * hsic_current)
        if denom < 1e-10:
            return 1.0  # Maximum FLR if one kernel is degenerate

        cka = hsic_init_current / denom
        flr = 1.0 - cka.item()
        return max(0.0, min(1.0, flr))  # Clamp to [0, 1]

    def compute_weight_flr(self) -> Dict[str, float]:
        """
        Compute weight-based feature learning rate.

        Returns separate FLR for each layer.
        """
        w1_change = (self.W1 - self._W1_init).norm() / self._W1_init.norm()
        w2_change = (self.W2 - self._W2_init).norm() / self._W2_init.norm()

        return {
            'flr_w1': w1_change.item(),
            'flr_w2': w2_change.item(),
            'flr_total': (w1_change + w2_change).item() / 2
        }

    def save_weights_after_t1(self):
        """Save weights after task 1 training."""
        self._W1_after_t1 = self.W1.clone().detach()
        self._W2_after_t1 = self.W2.clone().detach()

    def compute_weight_change_t2(self) -> Dict[str, float]:
        """Compute weight change during task 2 training."""
        if self._W1_after_t1 is None:
            return {'w1_change': 0.0, 'w2_change': 0.0, 'total_change': 0.0}

        w1_change = (self.W1 - self._W1_after_t1).norm().item()
        w2_change = (self.W2 - self._W2_after_t1).norm().item()

        return {
            'w1_change_t2': w1_change,
            'w2_change_t2': w2_change,
            'total_change_t2': w1_change + w2_change
        }

    def compute_ntk_alignment(self, X_probe: torch.Tensor) -> float:
        """
        Compute NTK alignment between init and current.

        High alignment = lazy regime
        Low alignment = rich regime
        """
        # Compute NTK at initialization
        ntk_init = self._compute_ntk(X_probe, use_init=True)
        ntk_current = self._compute_ntk(X_probe, use_init=False)

        # Compute alignment (cosine similarity of flattened NTKs)
        ntk_init_flat = ntk_init.flatten()
        ntk_current_flat = ntk_current.flatten()

        alignment = F.cosine_similarity(
            ntk_init_flat.unsqueeze(0),
            ntk_current_flat.unsqueeze(0)
        ).item()

        return alignment

    def _compute_ntk(self, X: torch.Tensor, use_init: bool = False) -> torch.Tensor:
        """Compute empirical NTK matrix."""
        n = X.shape[0]
        ntk = torch.zeros(n, n, device=self.device)

        # Use appropriate weights
        W1 = self._W1_init if use_init else self.W1
        W2 = self._W2_init if use_init else self.W2

        # Compute Jacobian-based NTK approximation
        # For efficiency, use the formula: NTK ≈ J @ J.T
        with torch.no_grad():
            h = self._activate(X @ W1.T)  # (n, d_hidden)

            # Gradient w.r.t. W2: outer product of h and output gradient
            # Gradient w.r.t. W1: backprop through activation

            # Simplified: use feature kernel as proxy
            # K1 = X @ X.T (input kernel)
            # K2 = h @ h.T (hidden kernel)

            K_hidden = h @ h.T
            ntk = K_hidden  # Simplified NTK proxy

        return ntk


def create_nonlinear_task_pair(
    d_in: int,
    d_hidden: int,
    d_out: int,
    similarity: float,
    activation: str = "relu",
    device: str = "cpu"
) -> NonlinearTaskPair:
    """
    Create two nonlinear teachers with specified similarity.

    Similarity is defined on the combined weight space.
    """
    # First teacher: random weights
    W1_1 = torch.randn(d_hidden, d_in, device=device) / np.sqrt(d_in)
    W2_1 = torch.randn(d_out, d_hidden, device=device) / np.sqrt(d_hidden)

    # Second teacher: interpolate with random
    W1_random = torch.randn(d_hidden, d_in, device=device) / np.sqrt(d_in)
    W2_random = torch.randn(d_out, d_hidden, device=device) / np.sqrt(d_hidden)

    # Orthogonalize random weights w.r.t. first teacher
    def orthogonalize(v, u):
        """Make v orthogonal to u."""
        proj = (v.flatten() @ u.flatten()) / (u.flatten() @ u.flatten() + 1e-10)
        return v - proj * u

    W1_orth = orthogonalize(W1_random, W1_1)
    W2_orth = orthogonalize(W2_random, W2_1)

    # Normalize
    W1_1_norm = W1_1 / (W1_1.norm() + 1e-10)
    W1_orth_norm = W1_orth / (W1_orth.norm() + 1e-10)
    W2_1_norm = W2_1 / (W2_1.norm() + 1e-10)
    W2_orth_norm = W2_orth / (W2_orth.norm() + 1e-10)

    # Interpolate
    W1_2 = similarity * W1_1_norm + np.sqrt(1 - similarity**2) * W1_orth_norm
    W2_2 = similarity * W2_1_norm + np.sqrt(1 - similarity**2) * W2_orth_norm

    # Rescale to original magnitude
    W1_2 = W1_2 * W1_1.norm() / (W1_2.norm() + 1e-10)
    W2_2 = W2_2 * W2_1.norm() / (W2_2.norm() + 1e-10)

    return NonlinearTaskPair(
        teacher1_W1=W1_1,
        teacher1_W2=W2_1,
        teacher2_W1=W1_2,
        teacher2_W2=W2_2,
        similarity=similarity,
        d_in=d_in,
        d_hidden=d_hidden,
        d_out=d_out,
        activation=activation
    )


def train_nonlinear_step(
    student: NonlinearStudent,
    X: torch.Tensor,
    y: torch.Tensor,
    optimizer: torch.optim.Optimizer
) -> float:
    """Single training step for nonlinear student."""
    optimizer.zero_grad()
    y_pred = student(X)
    loss = F.mse_loss(y_pred, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def train_nonlinear_on_task(
    student: NonlinearStudent,
    teacher: NonlinearTeacher,
    n_steps: int,
    lr: float,
    batch_size: int = 64,
    noise_std: float = 0.0,
    X_probe: Optional[torch.Tensor] = None,
    track_flr_every: int = 100
) -> Dict[str, List[float]]:
    """
    Train nonlinear student on a single task.

    Returns training history including FLR tracking.
    """
    optimizer = torch.optim.SGD(student.parameters(), lr=lr)
    device = student.device

    history = {
        'loss': [],
        'flr': [],
        'ntk_alignment': []
    }

    # Create probe data for FLR measurement
    if X_probe is None:
        X_probe = torch.randn(200, student.d_in, device=device)

    for step in range(n_steps):
        X, y = teacher.generate_data(batch_size, noise_std=noise_std, device=device)
        loss = train_nonlinear_step(student, X, y, optimizer)
        history['loss'].append(loss)

        # Track FLR periodically
        if step % track_flr_every == 0 or step == n_steps - 1:
            flr = student.compute_feature_learning_rate(X_probe)
            history['flr'].append(flr)

            ntk_align = student.compute_ntk_alignment(X_probe)
            history['ntk_alignment'].append(ntk_align)

    return history


def evaluate_nonlinear_on_task(
    student: NonlinearStudent,
    teacher: NonlinearTeacher,
    n_samples: int = 1000
) -> float:
    """Evaluate student on task."""
    student.eval()
    device = student.device
    with torch.no_grad():
        X, y = teacher.generate_data(n_samples, noise_std=0.0, device=device)
        y_pred = student(X)
        mse = F.mse_loss(y_pred, y).item()
    student.train()
    return mse


def classify_regime(
    flr: float,
    ntk_alignment: float,
    flr_threshold: float = 0.1,
    ntk_threshold: float = 0.9
) -> str:
    """
    Classify training regime as lazy or rich.

    Lazy regime: low FLR, high NTK alignment
    Rich regime: high FLR, low NTK alignment
    """
    if flr < flr_threshold and ntk_alignment > ntk_threshold:
        return "lazy"
    elif flr > flr_threshold or ntk_alignment < ntk_threshold:
        return "rich"
    else:
        return "transition"
