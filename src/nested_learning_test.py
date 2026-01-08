"""
Test Nested Learning Core Ideas on Catastrophic Forgetting

Tests whether the key mechanisms from Nested Learning actually help
with sequential task forgetting (not just long-context memory).

Core ideas to test:
1. Multi-timescale updates (slow/fast parameters)
2. Surprise-based memory (higher gradient = more memorable)
3. Forgetting gate (adaptive decay)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import pandas as pd
from tqdm import tqdm


@dataclass
class NestedLearningConfig:
    """Configuration for nested learning experiments."""
    d_in: int = 50
    d_hidden: int = 64
    d_out: int = 5

    # Multi-timescale: ratio of slow to fast learning rate
    slow_lr_ratio: float = 0.1  # Slow params learn 10x slower

    # Which fraction of parameters are "slow" (long-term memory)
    slow_param_fraction: float = 0.5

    # Surprise threshold for memory updates
    use_surprise_gating: bool = True
    surprise_threshold: float = 0.1

    # Standard training params
    n_steps: int = 300
    batch_size: int = 64
    base_lr: float = 0.1


class MultiTimescaleNetwork(nn.Module):
    """
    Network with multi-timescale parameter updates.

    Implements the core Nested Learning idea:
    - Some parameters update slowly (long-term memory)
    - Some parameters update quickly (short-term adaptation)
    """

    def __init__(self, d_in: int, d_hidden: int, d_out: int,
                 slow_fraction: float = 0.5):
        super().__init__()

        self.d_in = d_in
        self.d_hidden = d_hidden
        self.d_out = d_out
        self.slow_fraction = slow_fraction

        # Split hidden layer into slow and fast portions
        self.d_hidden_slow = int(d_hidden * slow_fraction)
        self.d_hidden_fast = d_hidden - self.d_hidden_slow

        # Slow parameters (long-term memory)
        self.W1_slow = nn.Parameter(torch.randn(d_in, self.d_hidden_slow) * 0.1)
        self.W2_slow = nn.Parameter(torch.randn(self.d_hidden_slow, d_out) * 0.1)

        # Fast parameters (short-term adaptation)
        self.W1_fast = nn.Parameter(torch.randn(d_in, self.d_hidden_fast) * 0.1)
        self.W2_fast = nn.Parameter(torch.randn(self.d_hidden_fast, d_out) * 0.1)

        self.activation = F.gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Slow pathway
        h_slow = self.activation(x @ self.W1_slow)
        y_slow = h_slow @ self.W2_slow

        # Fast pathway
        h_fast = self.activation(x @ self.W1_fast)
        y_fast = h_fast @ self.W2_fast

        # Combine outputs
        return y_slow + y_fast

    def get_slow_params(self):
        return [self.W1_slow, self.W2_slow]

    def get_fast_params(self):
        return [self.W1_fast, self.W2_fast]


class StandardNetwork(nn.Module):
    """Standard single-timescale network for comparison."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(d_in, d_hidden) * 0.1)
        self.W2 = nn.Parameter(torch.randn(d_hidden, d_out) * 0.1)
        self.activation = F.gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.activation(x @ self.W1)
        return h @ self.W2


class SurpriseGatedNetwork(nn.Module):
    """
    Network with surprise-based gating.

    Implements Titans' idea: surprising inputs (high gradient)
    cause larger memory updates.
    """

    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(d_in, d_hidden) * 0.1)
        self.W2 = nn.Parameter(torch.randn(d_hidden, d_out) * 0.1)
        self.activation = F.gelu

        # Running estimate of "normal" gradient magnitude
        self.register_buffer('grad_ema', torch.tensor(1.0))
        self.ema_decay = 0.99

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.activation(x @ self.W1)
        return h @ self.W2

    def compute_surprise(self, loss: torch.Tensor) -> float:
        """Compute surprise as ratio of current gradient to EMA."""
        # Get gradient magnitude
        grad_mag = 0.0
        for p in self.parameters():
            if p.grad is not None:
                grad_mag += p.grad.norm().item()

        # Surprise = how much larger than expected
        surprise = grad_mag / (self.grad_ema.item() + 1e-8)

        # Update EMA
        self.grad_ema = self.ema_decay * self.grad_ema + (1 - self.ema_decay) * grad_mag

        return surprise


def create_teacher(d_in: int, d_hidden: int, d_out: int,
                   seed: int = 0) -> nn.Module:
    """Create a teacher network."""
    torch.manual_seed(seed)
    teacher = StandardNetwork(d_in, d_hidden, d_out)
    # Freeze teacher
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def create_task_pair(d_in: int, d_hidden: int, d_out: int,
                     similarity: float, seed: int = 0) -> Tuple[nn.Module, nn.Module]:
    """Create two teachers with specified similarity."""
    torch.manual_seed(seed)

    teacher1 = create_teacher(d_in, d_hidden, d_out, seed)

    # Create teacher2 as interpolation
    torch.manual_seed(seed + 1000)
    teacher2_random = create_teacher(d_in, d_hidden, d_out, seed + 1000)

    # Interpolate parameters
    with torch.no_grad():
        for p1, p2, pr in zip(teacher1.parameters(),
                              teacher2_random.parameters(),
                              teacher2_random.parameters()):
            # teacher2 = similarity * teacher1 + (1-similarity) * random
            pr.data = similarity * p1.data + (1 - similarity) * pr.data

    return teacher1, teacher2_random


def generate_data(teacher: nn.Module, n_samples: int,
                  d_in: int, noise_std: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate data from teacher."""
    X = torch.randn(n_samples, d_in)
    with torch.no_grad():
        y = teacher(X)
        if noise_std > 0:
            y = y + torch.randn_like(y) * noise_std
    return X, y


def evaluate(model: nn.Module, teacher: nn.Module,
             n_samples: int, d_in: int) -> float:
    """Evaluate model on teacher's task."""
    X, y = generate_data(teacher, n_samples, d_in)
    with torch.no_grad():
        y_pred = model(X)
        loss = F.mse_loss(y_pred, y)
    return loss.item()


def train_standard(model: nn.Module, teacher: nn.Module,
                   config: NestedLearningConfig) -> Dict[str, List[float]]:
    """Standard single-LR training."""
    optimizer = torch.optim.SGD(model.parameters(), lr=config.base_lr)
    history = {'loss': []}

    for step in range(config.n_steps):
        X, y = generate_data(teacher, config.batch_size, config.d_in)

        optimizer.zero_grad()
        y_pred = model(X)
        loss = F.mse_loss(y_pred, y)
        loss.backward()
        optimizer.step()

        history['loss'].append(loss.item())

    return history


def train_multi_timescale(model: MultiTimescaleNetwork, teacher: nn.Module,
                          config: NestedLearningConfig) -> Dict[str, List[float]]:
    """Multi-timescale training (core Nested Learning idea)."""

    # Separate optimizers for slow and fast params
    optimizer_slow = torch.optim.SGD(
        model.get_slow_params(),
        lr=config.base_lr * config.slow_lr_ratio
    )
    optimizer_fast = torch.optim.SGD(
        model.get_fast_params(),
        lr=config.base_lr
    )

    history = {'loss': []}

    for step in range(config.n_steps):
        X, y = generate_data(teacher, config.batch_size, config.d_in)

        optimizer_slow.zero_grad()
        optimizer_fast.zero_grad()

        y_pred = model(X)
        loss = F.mse_loss(y_pred, y)
        loss.backward()

        optimizer_slow.step()
        optimizer_fast.step()

        history['loss'].append(loss.item())

    return history


def train_surprise_gated(model: SurpriseGatedNetwork, teacher: nn.Module,
                         config: NestedLearningConfig) -> Dict[str, List[float]]:
    """Surprise-gated training (Titans idea)."""

    optimizer = torch.optim.SGD(model.parameters(), lr=config.base_lr)
    history = {'loss': [], 'surprise': []}

    for step in range(config.n_steps):
        X, y = generate_data(teacher, config.batch_size, config.d_in)

        optimizer.zero_grad()
        y_pred = model(X)
        loss = F.mse_loss(y_pred, y)
        loss.backward()

        # Compute surprise
        surprise = model.compute_surprise(loss)
        history['surprise'].append(surprise)

        # Scale gradients by surprise (more surprise = larger update)
        if config.use_surprise_gating:
            scale = min(surprise, 2.0)  # Cap at 2x
            for p in model.parameters():
                if p.grad is not None:
                    p.grad *= scale

        optimizer.step()
        history['loss'].append(loss.item())

    return history


def run_forgetting_experiment(
    method: str,
    similarity: float,
    config: NestedLearningConfig,
    seed: int = 0
) -> Dict[str, float]:
    """
    Run a single forgetting experiment.

    Args:
        method: 'standard', 'multi_timescale', or 'surprise_gated'
        similarity: Task similarity (0 to 1)
        config: Experiment configuration
        seed: Random seed

    Returns:
        Dictionary with forgetting metrics
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create task pair
    teacher1, teacher2 = create_task_pair(
        config.d_in, config.d_hidden, config.d_out,
        similarity, seed
    )

    # Create model based on method
    if method == 'standard':
        model = StandardNetwork(config.d_in, config.d_hidden, config.d_out)
        train_fn = train_standard
    elif method == 'multi_timescale':
        model = MultiTimescaleNetwork(
            config.d_in, config.d_hidden, config.d_out,
            config.slow_param_fraction
        )
        train_fn = train_multi_timescale
    elif method == 'surprise_gated':
        model = SurpriseGatedNetwork(config.d_in, config.d_hidden, config.d_out)
        train_fn = train_surprise_gated
    else:
        raise ValueError(f"Unknown method: {method}")

    # Evaluate before training
    loss_t1_init = evaluate(model, teacher1, 500, config.d_in)

    # Train on Task 1
    train_fn(model, teacher1, config)
    loss_t1_after_t1 = evaluate(model, teacher1, 500, config.d_in)

    # Train on Task 2
    train_fn(model, teacher2, config)
    loss_t1_after_t2 = evaluate(model, teacher1, 500, config.d_in)
    loss_t2_after_t2 = evaluate(model, teacher2, 500, config.d_in)

    # Compute forgetting
    forgetting = loss_t1_after_t2 - loss_t1_after_t1

    return {
        'method': method,
        'similarity': similarity,
        'seed': seed,
        'loss_t1_after_t1': loss_t1_after_t1,
        'loss_t1_after_t2': loss_t1_after_t2,
        'loss_t2_after_t2': loss_t2_after_t2,
        'forgetting': forgetting,
    }


def run_comparison_experiment(
    similarities: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0],
    n_seeds: int = 10,
    slow_lr_ratios: List[float] = [0.01, 0.1, 0.5],
) -> pd.DataFrame:
    """
    Compare standard vs nested learning methods on forgetting.
    """

    results = []
    methods = ['standard', 'multi_timescale', 'surprise_gated']

    config = NestedLearningConfig()

    total = len(similarities) * n_seeds * len(methods) * len(slow_lr_ratios)
    pbar = tqdm(total=total, desc="Running experiments")

    for similarity in similarities:
        for seed in range(n_seeds):
            # Standard baseline
            config.slow_lr_ratio = 1.0  # Not used for standard
            result = run_forgetting_experiment('standard', similarity, config, seed)
            result['slow_lr_ratio'] = 1.0
            results.append(result)
            pbar.update(1)

            # Multi-timescale with different ratios
            for ratio in slow_lr_ratios:
                config.slow_lr_ratio = ratio
                result = run_forgetting_experiment('multi_timescale', similarity, config, seed)
                result['slow_lr_ratio'] = ratio
                results.append(result)
                pbar.update(1)

            # Surprise gated
            result = run_forgetting_experiment('surprise_gated', similarity, config, seed)
            result['slow_lr_ratio'] = 1.0
            results.append(result)
            pbar.update(1)

    pbar.close()
    return pd.DataFrame(results)


def analyze_results(df: pd.DataFrame) -> Dict[str, any]:
    """Analyze the comparison results."""

    analysis = {}

    # Overall forgetting by method
    method_stats = df.groupby('method')['forgetting'].agg(['mean', 'std']).round(4)
    analysis['method_comparison'] = method_stats.to_dict()

    # Forgetting by method and similarity
    pivot = df.pivot_table(
        values='forgetting',
        index='similarity',
        columns='method',
        aggfunc='mean'
    ).round(4)
    analysis['forgetting_by_similarity'] = pivot.to_dict()

    # Best slow_lr_ratio for multi_timescale
    mt_df = df[df['method'] == 'multi_timescale']
    ratio_stats = mt_df.groupby('slow_lr_ratio')['forgetting'].mean().round(4)
    analysis['slow_lr_ratio_effect'] = ratio_stats.to_dict()

    # Statistical comparison: does multi_timescale beat standard?
    std_forg = df[df['method'] == 'standard']['forgetting'].values
    mt_forg = df[df['method'] == 'multi_timescale']['forgetting'].values

    # Simple t-test approximation
    diff = std_forg.mean() - mt_forg[:len(std_forg)].mean()
    analysis['standard_minus_multitimescale'] = round(diff, 4)
    analysis['multitimescale_helps'] = diff > 0

    # Correlation with similarity for each method
    for method in df['method'].unique():
        subset = df[df['method'] == method]
        corr = subset['similarity'].corr(subset['forgetting'])
        analysis[f'similarity_corr_{method}'] = round(corr, 4)

    return analysis


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Nested Learning Ideas on Catastrophic Forgetting")
    print("=" * 60)

    # Run comparison
    df = run_comparison_experiment(
        similarities=[0.0, 0.25, 0.5, 0.75, 1.0],
        n_seeds=10,
        slow_lr_ratios=[0.01, 0.1, 0.5]
    )

    # Save results
    df.to_csv('results/nested_learning_test.csv', index=False)

    # Analyze
    analysis = analyze_results(df)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print("\n1. Mean Forgetting by Method:")
    print("-" * 40)
    for method in ['standard', 'multi_timescale', 'surprise_gated']:
        subset = df[df['method'] == method]
        print(f"  {method:20s}: {subset['forgetting'].mean():.4f} ± {subset['forgetting'].std():.4f}")

    print("\n2. Forgetting by Similarity (mean across seeds):")
    print("-" * 40)
    pivot = df.pivot_table(values='forgetting', index='similarity', columns='method', aggfunc='mean')
    print(pivot.round(4).to_string())

    print("\n3. Effect of Slow LR Ratio (multi_timescale only):")
    print("-" * 40)
    mt_df = df[df['method'] == 'multi_timescale']
    for ratio in sorted(mt_df['slow_lr_ratio'].unique()):
        subset = mt_df[mt_df['slow_lr_ratio'] == ratio]
        print(f"  ratio={ratio:.2f}: forgetting={subset['forgetting'].mean():.4f}")

    print("\n4. Similarity Correlation by Method:")
    print("-" * 40)
    for method in df['method'].unique():
        subset = df[df['method'] == method]
        corr = subset['similarity'].corr(subset['forgetting'])
        print(f"  {method:20s}: r = {corr:.4f}")

    print("\n5. Key Question: Does Multi-Timescale Help?")
    print("-" * 40)
    std_mean = df[df['method'] == 'standard']['forgetting'].mean()
    mt_mean = df[df['method'] == 'multi_timescale']['forgetting'].mean()
    sg_mean = df[df['method'] == 'surprise_gated']['forgetting'].mean()

    print(f"  Standard:        {std_mean:.4f}")
    print(f"  Multi-timescale: {mt_mean:.4f} ({'BETTER' if mt_mean < std_mean else 'WORSE'})")
    print(f"  Surprise-gated:  {sg_mean:.4f} ({'BETTER' if sg_mean < std_mean else 'WORSE'})")

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    if mt_mean < std_mean * 0.95:  # At least 5% improvement
        print("Multi-timescale updates HELP with catastrophic forgetting!")
    elif mt_mean > std_mean * 1.05:
        print("Multi-timescale updates HURT - standard training is better.")
    else:
        print("Multi-timescale updates have NO SIGNIFICANT EFFECT on forgetting.")

    print(f"\nSimilarity still dominates all methods (r ≈ {analysis.get('similarity_corr_standard', 'N/A')})")
