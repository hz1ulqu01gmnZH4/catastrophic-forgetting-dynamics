#!/usr/bin/env python3
"""
Test: Does forgetting reduction scale with nesting depth?

Tests networks with 1, 2, 3, 4, 5 levels of timescale hierarchy.
Each level has progressively slower learning rate.

Timescale hierarchy:
- Level 1: Standard (single LR)
- Level 2: Fast + Slow
- Level 3: Fast + Medium + Slow
- Level 4: Fast + Medium + Slow + Very Slow
- Level 5: Fast + Medium-Fast + Medium + Slow + Very Slow
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Dict
from pathlib import Path
import json


@dataclass
class Config:
    d_in: int = 50
    d_out: int = 5
    d_hidden_total: int = 128  # Total hidden units, split across levels
    base_lr: float = 0.1
    lr_decay_per_level: float = 0.1  # Each level is 10x slower
    t1_steps: int = 300
    t2_steps: int = 300
    seed: int = 42


class DeepNestedNetwork(nn.Module):
    """Network with N levels of timescale hierarchy."""

    def __init__(self, d_in: int, d_hidden_total: int, d_out: int, n_levels: int):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.n_levels = n_levels

        # Split hidden units across levels (equal split)
        self.d_per_level = d_hidden_total // n_levels

        # Create parameters for each level
        self.W1_levels = nn.ParameterList()
        self.W2_levels = nn.ParameterList()

        for i in range(n_levels):
            W1 = nn.Parameter(torch.randn(d_in, self.d_per_level) * 0.1)
            W2 = nn.Parameter(torch.randn(self.d_per_level, d_out) * 0.1)
            self.W1_levels.append(W1)
            self.W2_levels.append(W2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sum contributions from all levels
        out = torch.zeros(x.shape[0], self.d_out)
        for i in range(self.n_levels):
            h = F.gelu(x @ self.W1_levels[i])
            out = out + h @ self.W2_levels[i]
        return out / self.n_levels  # Average contributions

    def get_params_by_level(self) -> List[List[nn.Parameter]]:
        """Return parameters grouped by level (0=fastest, N-1=slowest)."""
        params = []
        for i in range(self.n_levels):
            params.append([self.W1_levels[i], self.W2_levels[i]])
        return params


def generate_teacher(d_in: int, d_out: int, seed: int) -> torch.Tensor:
    """Generate a random linear teacher."""
    torch.manual_seed(seed)
    return torch.randn(d_in, d_out) * 0.5


def generate_correlated_teacher(W1: torch.Tensor, similarity: float, seed: int) -> torch.Tensor:
    """Generate a teacher with specified similarity to W1."""
    torch.manual_seed(seed)
    W_random = torch.randn_like(W1) * 0.5
    return similarity * W1 + (1 - similarity) * W_random


def train_deep_nested(
    model: DeepNestedNetwork,
    teacher: torch.Tensor,
    config: Config,
    n_steps: int
) -> float:
    """Train with level-specific learning rates."""

    # Create optimizer for each level
    optimizers = []
    params_by_level = model.get_params_by_level()

    for level_idx, level_params in enumerate(params_by_level):
        # Level 0 = fastest, level N-1 = slowest
        lr = config.base_lr * (config.lr_decay_per_level ** level_idx)
        opt = torch.optim.SGD(level_params, lr=lr)
        optimizers.append(opt)

    # Training loop
    for step in range(n_steps):
        x = torch.randn(32, config.d_in)
        y_true = x @ teacher
        y_pred = model(x)
        loss = F.mse_loss(y_pred, y_true)

        # Update all levels
        for opt in optimizers:
            opt.zero_grad()
        loss.backward()
        for opt in optimizers:
            opt.step()

    # Final loss
    with torch.no_grad():
        x = torch.randn(1000, config.d_in)
        y_true = x @ teacher
        y_pred = model(x)
        return F.mse_loss(y_pred, y_true).item()


def run_experiment(
    n_levels: int,
    similarity: float,
    config: Config,
    seed: int
) -> Dict:
    """Run a single experiment with specified nesting depth."""

    torch.manual_seed(seed)

    # Create model
    model = DeepNestedNetwork(
        config.d_in, config.d_hidden_total, config.d_out, n_levels
    )

    # Generate teachers
    W1 = generate_teacher(config.d_in, config.d_out, seed)
    W2 = generate_correlated_teacher(W1, similarity, seed + 1000)

    # Train on Task 1
    loss_t1_after_t1 = train_deep_nested(model, W1, config, config.t1_steps)

    # Train on Task 2
    loss_t1_after_t2_start = None
    with torch.no_grad():
        x = torch.randn(1000, config.d_in)
        y_true = x @ W1
        y_pred = model(x)
        loss_t1_after_t2_start = F.mse_loss(y_pred, y_true).item()

    train_deep_nested(model, W2, config, config.t2_steps)

    # Measure forgetting
    with torch.no_grad():
        x = torch.randn(1000, config.d_in)
        y_true = x @ W1
        y_pred = model(x)
        loss_t1_after_t2 = F.mse_loss(y_pred, y_true).item()

    forgetting = loss_t1_after_t2 - loss_t1_after_t1

    return {
        'n_levels': n_levels,
        'similarity': similarity,
        'seed': seed,
        'loss_t1_after_t1': loss_t1_after_t1,
        'loss_t1_after_t2': loss_t1_after_t2,
        'forgetting': forgetting,
        'lr_decay': config.lr_decay_per_level,
    }


def main():
    print("=" * 60)
    print("DEEP NESTING EXPERIMENT")
    print("Testing if forgetting reduction scales with nesting depth")
    print("=" * 60)

    config = Config()

    # Experiment grid
    n_levels_list = [1, 2, 3, 4, 5]
    similarities = [0.0, 0.25, 0.5, 0.75, 1.0]
    seeds = list(range(10))  # 10 seeds per condition

    results = []
    total = len(n_levels_list) * len(similarities) * len(seeds)

    print(f"\nRunning {total} experiments...")
    print(f"Levels: {n_levels_list}")
    print(f"Similarities: {similarities}")
    print(f"Seeds per condition: {len(seeds)}")
    print(f"LR decay per level: {config.lr_decay_per_level}")
    print()

    for n_levels in n_levels_list:
        for sim in similarities:
            for seed in seeds:
                result = run_experiment(n_levels, sim, config, seed)
                results.append(result)

        # Print progress
        completed = len(results)
        print(f"Completed {n_levels}-level networks: {completed}/{total}")

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Save raw data
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    df.to_csv(output_dir / "deep_nesting_test.csv", index=False)

    # Analysis
    print("\n" + "=" * 60)
    print("RESULTS: Forgetting by Nesting Depth")
    print("=" * 60)

    # Mean forgetting by n_levels
    summary = df.groupby('n_levels')['forgetting'].agg(['mean', 'std', 'count'])
    print("\nMean Forgetting by Depth:")
    print(summary)

    # Improvement relative to 1-level
    baseline = summary.loc[1, 'mean']
    print(f"\nBaseline (1-level): {baseline:.4f}")
    print("\nImprovement vs Baseline:")
    for n_levels in n_levels_list:
        mean_f = summary.loc[n_levels, 'mean']
        improvement = (baseline - mean_f) / baseline * 100
        print(f"  {n_levels}-level: {mean_f:.4f} ({improvement:+.1f}%)")

    # By similarity
    print("\n" + "-" * 40)
    print("Forgetting by Depth and Similarity:")
    pivot = df.pivot_table(
        values='forgetting',
        index='n_levels',
        columns='similarity',
        aggfunc='mean'
    )
    print(pivot.round(4))

    # Correlation with similarity (still dominates?)
    print("\n" + "-" * 40)
    print("Correlation with Similarity by Depth:")
    for n_levels in n_levels_list:
        subset = df[df['n_levels'] == n_levels]
        corr = subset['similarity'].corr(subset['forgetting'])
        print(f"  {n_levels}-level: r = {corr:.3f}")

    # Does deeper = better? Statistical test
    print("\n" + "-" * 40)
    print("Does Deeper Nesting Help?")

    # Compare adjacent levels
    for i in range(len(n_levels_list) - 1):
        n1 = n_levels_list[i]
        n2 = n_levels_list[i + 1]
        f1 = df[df['n_levels'] == n1]['forgetting'].values
        f2 = df[df['n_levels'] == n2]['forgetting'].values

        diff = f1.mean() - f2.mean()
        # Simple t-test
        pooled_std = np.sqrt((f1.std()**2 + f2.std()**2) / 2)
        t_stat = diff / (pooled_std * np.sqrt(2/len(f1)))

        sig = "✓" if abs(t_stat) > 2.0 else "✗"
        print(f"  {n1} → {n2}: Δ = {diff:+.4f} (t = {t_stat:.2f}) {sig}")

    # Save analysis
    analysis = {
        'summary_by_depth': summary.to_dict(),
        'pivot_table': pivot.to_dict(),
        'baseline_1_level': baseline,
        'config': {
            'd_in': config.d_in,
            'd_out': config.d_out,
            'd_hidden_total': config.d_hidden_total,
            'base_lr': config.base_lr,
            'lr_decay_per_level': config.lr_decay_per_level,
            'steps': config.t1_steps,
        }
    }

    with open(output_dir / "deep_nesting_analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)

    # Check if improvement scales
    improvements = []
    for n_levels in n_levels_list:
        mean_f = summary.loc[n_levels, 'mean']
        imp = (baseline - mean_f) / baseline * 100
        improvements.append(imp)

    # Linear regression of improvement vs depth
    x = np.array(n_levels_list)
    y = np.array(improvements)
    slope = np.cov(x, y)[0, 1] / np.var(x)

    if slope > 1.0:  # More than 1% improvement per level
        print(f"✓ YES: Improvement scales with depth (+{slope:.1f}% per level)")
    elif slope > 0:
        print(f"⚠ MARGINAL: Slight scaling (+{slope:.1f}% per level)")
    else:
        print(f"✗ NO: No scaling with depth (slope = {slope:.1f}%)")

    print(f"\nBut similarity correlation remains r ≈ {df['similarity'].corr(df['forgetting']):.2f}")
    print("Task similarity still dominates forgetting dynamics.")


if __name__ == "__main__":
    main()
