#!/usr/bin/env python3
"""
Phase 6.5: Similarity-Aware Learning Rate

Tests whether adapting learning rate based on task similarity reduces forgetting.
This is a simple, low-cost intervention with no computational overhead.

Hypothesis: lr_t2 = base_lr * similarity^α
- α = 0: No adaptation (baseline)
- α > 0: Lower LR for dissimilar tasks
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from src.adaptive_lr import (
    Phase65Config,
    run_adaptive_lr_experiment,
    AdaptiveLRResult,
    theoretical_optimal_alpha,
)


def run_phase65_experiments(config: Phase65Config) -> List[AdaptiveLRResult]:
    """Run all Phase 6.5 experiments."""
    results = []

    # Count total experiments
    total = (
        len(config.similarities) *
        len(config.base_learning_rates) *
        len(config.alpha_values) *
        config.n_seeds
    )

    print("=" * 60)
    print("Phase 6.5: Similarity-Aware Learning Rate")
    print("=" * 60)
    print(f"Total runs: {total}")
    print(f"Similarities: {config.similarities}")
    print(f"Base learning rates: {config.base_learning_rates}")
    print(f"Alpha values: {config.alpha_values}")
    print(f"Seeds: {config.n_seeds}")
    print(f"Theoretical optimal α: {theoretical_optimal_alpha():.2f}")
    print("=" * 60)

    # Get baseline time
    baseline_result = run_adaptive_lr_experiment(
        similarity=0.5,
        base_lr=0.05,
        alpha=0.0,
        seed=0,
        config=config,
        baseline_time=None
    )
    baseline_time = baseline_result.relative_time

    # Run all experiments
    pbar = tqdm(total=total, desc="Running experiments")

    for similarity in config.similarities:
        for base_lr in config.base_learning_rates:
            for alpha in config.alpha_values:
                for seed in range(config.n_seeds):
                    result = run_adaptive_lr_experiment(
                        similarity=similarity,
                        base_lr=base_lr,
                        alpha=alpha,
                        seed=seed,
                        config=config,
                        baseline_time=baseline_time
                    )
                    results.append(result)
                    pbar.update(1)

    pbar.close()
    return results


def analyze_results(results: List[AdaptiveLRResult]) -> Dict[str, Any]:
    """Analyze adaptive LR experiment results."""
    df = pd.DataFrame([asdict(r) for r in results])

    analysis = {
        'n_experiments': len(df),
        'timestamp': datetime.now().isoformat(),
        'theoretical_optimal_alpha': theoretical_optimal_alpha(),
    }

    # Compute forgetting reduction relative to baseline (alpha=0)
    baseline_df = df[df['alpha'] == 0.0].groupby(
        ['similarity', 'base_lr', 'seed']
    )['forgetting'].first().reset_index()
    baseline_df = baseline_df.rename(columns={'forgetting': 'baseline_forgetting'})

    df = df.merge(baseline_df, on=['similarity', 'base_lr', 'seed'], how='left')

    # Handle division by near-zero
    df['forgetting_reduction'] = np.where(
        df['baseline_forgetting'].abs() > 1e-10,
        (df['baseline_forgetting'] - df['forgetting']) / df['baseline_forgetting'].abs(),
        0.0
    )

    # Alpha comparison
    print("\n" + "=" * 60)
    print("Adaptive LR Analysis Results")
    print("=" * 60)

    print("\n" + "-" * 50)
    print("Alpha Value Comparison (averaged across all conditions)")
    print("-" * 50)

    alpha_stats = df.groupby('alpha').agg({
        'forgetting': ['mean', 'std'],
        'forgetting_reduction': ['mean', 'std'],
        'loss_t2_after_t2': ['mean', 'std'],
        'effective_lr': ['mean', 'std'],
    }).round(4)

    alpha_stats.columns = ['_'.join(col).strip() for col in alpha_stats.columns.values]
    print(alpha_stats)

    # Statistical tests vs baseline (alpha=0)
    print("\n" + "-" * 50)
    print("Statistical Tests vs Baseline (α=0)")
    print("-" * 50)

    baseline_forgetting = df[df['alpha'] == 0.0]['forgetting'].values
    test_results = {}

    for alpha in sorted(df['alpha'].unique()):
        if alpha == 0.0:
            continue

        alpha_forgetting = df[df['alpha'] == alpha]['forgetting'].values
        n_samples = min(len(baseline_forgetting), len(alpha_forgetting))

        t_stat, p_value = stats.ttest_ind(
            baseline_forgetting[:n_samples],
            alpha_forgetting[:n_samples]
        )

        mean_reduction = df[df['alpha'] == alpha]['forgetting_reduction'].mean()
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""

        print(f"\nα = {alpha}:")
        print(f"  Mean reduction: {mean_reduction*100:.1f}%")
        print(f"  t-statistic: {t_stat:.3f}")
        print(f"  p-value: {p_value:.2e} {sig}")

        test_results[str(alpha)] = {
            'mean_reduction': float(mean_reduction),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': bool(p_value < 0.05),
        }

    # Find best alpha
    best_alpha_row = df.groupby('alpha')['forgetting'].mean().idxmin()
    best_reduction = df[df['alpha'] == best_alpha_row]['forgetting_reduction'].mean()

    print("\n" + "-" * 50)
    print(f"Best α = {best_alpha_row} (mean reduction: {best_reduction*100:.1f}%)")
    print("-" * 50)

    # Effect by similarity
    print("\n" + "-" * 50)
    print("Forgetting Reduction by Task Similarity")
    print("-" * 50)

    for alpha in [0.5, 1.0, 1.5, 2.0]:
        if alpha not in df['alpha'].values:
            continue

        print(f"\nα = {alpha}:")
        alpha_df = df[df['alpha'] == alpha]

        for sim in sorted(df['similarity'].unique()):
            sim_baseline = df[(df['alpha'] == 0.0) & (df['similarity'] == sim)]['forgetting'].mean()
            sim_alpha = alpha_df[alpha_df['similarity'] == sim]['forgetting'].mean()
            reduction = (sim_baseline - sim_alpha) / abs(sim_baseline) if abs(sim_baseline) > 1e-10 else 0
            eff_lr = alpha_df[alpha_df['similarity'] == sim]['effective_lr'].mean()
            print(f"  Sim {sim}: {reduction*100:+.1f}% reduction, effective LR={eff_lr:.4f}")

    # T2 performance impact
    print("\n" + "-" * 50)
    print("Task 2 Performance Impact (plasticity check)")
    print("-" * 50)

    baseline_t2 = df[df['alpha'] == 0.0]['loss_t2_after_t2'].mean()
    print(f"\nBaseline T2 loss (α=0): {baseline_t2:.4f}")

    for alpha in sorted(df['alpha'].unique()):
        if alpha == 0.0:
            continue
        t2_loss = df[df['alpha'] == alpha]['loss_t2_after_t2'].mean()
        t2_change = (t2_loss - baseline_t2) / baseline_t2 * 100
        print(f"  α={alpha}: T2 loss={t2_loss:.4f} ({t2_change:+.1f}% vs baseline)")

    # Optimal alpha per similarity
    print("\n" + "-" * 50)
    print("Optimal α by Similarity")
    print("-" * 50)

    optimal_alphas = {}
    for sim in sorted(df['similarity'].unique()):
        sim_df = df[df['similarity'] == sim]
        alpha_perf = sim_df.groupby('alpha')['forgetting'].mean()
        best_alpha = alpha_perf.idxmin()
        best_forgetting = alpha_perf.min()
        optimal_alphas[str(sim)] = float(best_alpha)
        print(f"  Similarity {sim}: optimal α={best_alpha}, forgetting={best_forgetting:.4f}")

    # Best configurations
    print("\n" + "=" * 60)
    print("Best Configurations")
    print("=" * 60)

    best_configs = df.groupby(['alpha', 'base_lr']).agg({
        'forgetting': 'mean',
        'forgetting_reduction': 'mean',
        'loss_t2_after_t2': 'mean',
    }).reset_index()

    best_configs = best_configs[best_configs['alpha'] > 0]
    best_configs = best_configs.sort_values('forgetting_reduction', ascending=False)

    print("\nTop 5 Configurations:")
    for i, row in best_configs.head(5).iterrows():
        print(f"  α={row['alpha']}, base_lr={row['base_lr']}: "
              f"{row['forgetting_reduction']*100:.1f}% reduction, "
              f"T2 loss={row['loss_t2_after_t2']:.4f}")

    # Compile analysis
    analysis['alpha_comparison'] = alpha_stats.to_dict()
    analysis['statistical_tests'] = test_results
    analysis['optimal_alphas'] = optimal_alphas
    analysis['best_alpha'] = float(best_alpha_row)
    analysis['best_reduction'] = float(best_reduction)

    return analysis, df


def generate_report(analysis: Dict[str, Any], df: pd.DataFrame, output_dir: Path) -> None:
    """Generate markdown report."""
    report = f"""# Phase 6.5: Similarity-Aware Learning Rate Results
*Generated: {analysis['timestamp']}*

## Summary

This experiment tests whether adapting learning rate based on task similarity reduces forgetting.
This is a simple, low-cost intervention with **zero computational overhead**.

### Formula

```
lr_T2 = base_lr × similarity^α
```

- α = 0: No adaptation (baseline)
- α > 0: Lower LR for dissimilar tasks

### Theoretical Prediction

Based on forgetting equation `F ≈ 0.59 - 0.65 × similarity`, theoretical optimal α ≈ {analysis['theoretical_optimal_alpha']:.2f}

## Key Results

"""
    # Best alpha
    best_alpha = analysis['best_alpha']
    best_reduction = analysis['best_reduction']

    if best_reduction > 0.1:
        report += f"- **α = {best_alpha} significantly reduces forgetting** by {best_reduction*100:.1f}%\n"
    elif best_reduction > 0.01:
        report += f"- α = {best_alpha} provides modest reduction of {best_reduction*100:.1f}%\n"
    else:
        report += f"- Adaptive LR provides **limited effectiveness** (~{best_reduction*100:.1f}% reduction)\n"

    # Statistical results
    tests = analysis['statistical_tests']
    for alpha, result in sorted(tests.items(), key=lambda x: x[1]['mean_reduction'], reverse=True)[:3]:
        sig = "significant" if result['significant'] else "not significant"
        report += f"- α = {alpha}: {result['mean_reduction']*100:.1f}% reduction ({sig})\n"

    # Comparison table
    report += "\n## Alpha Value Comparison\n\n"
    report += "| α | Mean Forgetting | Reduction | T2 Loss | Effective LR |\n"
    report += "|---|-----------------|-----------|---------|-------------|\n"

    for alpha in sorted(df['alpha'].unique()):
        alpha_df = df[df['alpha'] == alpha]
        mean_forg = alpha_df['forgetting'].mean()
        mean_red = alpha_df['forgetting_reduction'].mean()
        t2_loss = alpha_df['loss_t2_after_t2'].mean()
        eff_lr = alpha_df['effective_lr'].mean()
        report += f"| {alpha} | {mean_forg:.4f} | {mean_red*100:+.1f}% | {t2_loss:.4f} | {eff_lr:.4f} |\n"

    # Optimal alpha by similarity
    report += "\n## Optimal α by Task Similarity\n\n"
    report += "| Similarity | Optimal α |\n"
    report += "|------------|----------|\n"

    for sim, alpha in sorted(analysis['optimal_alphas'].items(), key=lambda x: float(x[0])):
        report += f"| {sim} | {alpha} |\n"

    # Interpretation
    report += "\n## Interpretation\n\n"

    baseline_t2 = df[df['alpha'] == 0.0]['loss_t2_after_t2'].mean()
    best_t2 = df[df['alpha'] == best_alpha]['loss_t2_after_t2'].mean()
    t2_change = (best_t2 - baseline_t2) / baseline_t2 * 100

    if best_reduction > 0.1 and t2_change < 50:
        report += f"### Effective Low-Cost Mitigation\n\n"
        report += f"α = {best_alpha} provides {best_reduction*100:.1f}% forgetting reduction "
        report += f"with only {t2_change:.1f}% T2 performance impact. "
        report += f"This is a highly efficient intervention with zero computational overhead.\n\n"
    elif best_reduction > 0.01:
        report += f"### Moderate Effectiveness\n\n"
        report += f"Adaptive LR provides meaningful reduction ({best_reduction*100:.1f}%) "
        report += f"but less effective than regularization (Phase 6.3: 31%).\n\n"
    else:
        report += f"### Limited Effectiveness\n\n"
        report += f"Adaptive LR shows limited effectiveness in this setup. "
        report += f"The similarity-LR relationship may be more complex than power-law scaling.\n\n"

    # Comparison to other methods
    report += "## Comparison to Other Mitigation Methods\n\n"
    report += "| Method | Reduction | T2 Impact | Overhead |\n"
    report += "|--------|-----------|-----------|----------|\n"
    report += "| Gradient Projection (Phase 6.2) | ~1% | Minimal | ~45% |\n"
    report += "| L2 Regularization (Phase 6.3) | ~31% | +810% | ~30% |\n"
    report += f"| Adaptive LR (Phase 6.5) | ~{best_reduction*100:.0f}% | {t2_change:+.0f}% | **0%** |\n"

    # Recommendations
    report += "\n## Recommendations\n\n"

    if best_reduction > 0.05:
        report += f"- **Use α = {best_alpha}** for simple, cost-free mitigation\n"
        report += "- Combine with L2 regularization for stronger protection\n"
    else:
        report += "- Adaptive LR alone provides limited benefit\n"
        report += "- Consider combining with other methods\n"

    report += "- Tune α based on expected task similarity distribution\n"
    report += "- No computational overhead makes this suitable for resource-constrained settings\n"

    report += "\n---\n*Phase 6.5 Complete*\n"

    # Save report
    report_path = output_dir / "PHASE65_RESULTS.md"
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"\nReport saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Run Phase 6.5 adaptive LR experiments")
    parser.add_argument("--quick", action="store_true", help="Quick test with reduced parameters")
    parser.add_argument("--output-dir", type=str, default="results/phase65", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.quick:
        config = Phase65Config(
            similarities=[0.0, 0.5, 1.0],
            base_learning_rates=[0.05],
            alpha_values=[0.0, 0.5, 1.0, 1.5],
            n_seeds=2,
            n_steps=200,
        )
    else:
        config = Phase65Config()

    # Run experiments
    results = run_phase65_experiments(config)

    # Save raw results
    df = pd.DataFrame([asdict(r) for r in results])
    df.to_csv(output_dir / "phase65_adaptive_lr.csv", index=False)

    # Save config
    with open(output_dir / "phase65_config.json", 'w') as f:
        json.dump(asdict(config), f, indent=2)

    # Analyze
    analysis, df = analyze_results(results)

    # Save analysis
    with open(output_dir / "phase65_analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2, default=str)

    # Generate report
    generate_report(analysis, df, output_dir)

    print("\n" + "=" * 60)
    print("Phase 6.5 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
