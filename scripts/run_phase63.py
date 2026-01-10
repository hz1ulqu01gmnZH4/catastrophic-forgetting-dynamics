#!/usr/bin/env python3
"""
Phase 6.3: Regularization Approaches for Catastrophic Forgetting

Tests whether EWC and L2 regularization reduce forgetting better than
gradient projection methods (Phase 6.2 showed ~1% reduction).

Methods:
- L2: λ/2 ||θ - θ_T1||²
- EWC: λ/2 Σ F_i (θ_i - θ_T1)² where F is Fisher Information
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import torch
from scipy import stats
from tqdm import tqdm

from src.regularization import (
    Phase63Config,
    run_regularization_experiment,
    RegularizationResult,
)


def run_phase63_experiments(config: Phase63Config) -> List[RegularizationResult]:
    """Run all Phase 6.3 experiments."""
    results = []

    # Count total experiments
    total = (
        len(config.similarities) *
        len(config.learning_rates) *
        len(config.methods) *
        len(config.lambda_values) *
        config.n_seeds
    )

    print("=" * 60)
    print("Phase 6.3: Regularization Approaches")
    print("=" * 60)
    print(f"Total runs: {total}")
    print(f"Similarities: {config.similarities}")
    print(f"Learning rates: {config.learning_rates}")
    print(f"Methods: {config.methods}")
    print(f"Lambda values: {config.lambda_values}")
    print(f"Seeds: {config.n_seeds}")
    print("=" * 60)

    # Get baseline time (no regularization, lambda=0)
    baseline_result = run_regularization_experiment(
        similarity=0.5,
        lr=0.05,
        method="none",
        lambda_value=0.0,
        seed=0,
        config=config,
        baseline_time=None
    )
    baseline_time = baseline_result.relative_time  # This is actual time in seconds

    # Run all experiments
    pbar = tqdm(total=total, desc="Running experiments")

    for similarity in config.similarities:
        for lr in config.learning_rates:
            for method in config.methods:
                # For "none" method, only run with lambda=0
                lambdas = [0.0] if method == "none" else config.lambda_values

                for lambda_val in lambdas:
                    for seed in range(config.n_seeds):
                        result = run_regularization_experiment(
                            similarity=similarity,
                            lr=lr,
                            method=method,
                            lambda_value=lambda_val,
                            seed=seed,
                            config=config,
                            baseline_time=baseline_time
                        )
                        results.append(result)
                        pbar.update(1)

    pbar.close()
    return results


def analyze_results(results: List[RegularizationResult]) -> Dict[str, Any]:
    """Analyze regularization experiment results."""
    df = pd.DataFrame([asdict(r) for r in results])

    analysis = {
        'n_experiments': len(df),
        'timestamp': datetime.now().isoformat(),
    }

    # Compute forgetting reduction relative to baseline (none method)
    baseline_df = df[df['method'] == 'none'].groupby(
        ['similarity', 'learning_rate', 'seed']
    )['forgetting'].first().reset_index()
    baseline_df = baseline_df.rename(columns={'forgetting': 'baseline_forgetting'})

    df = df.merge(baseline_df, on=['similarity', 'learning_rate', 'seed'], how='left')
    df['forgetting_reduction'] = (df['baseline_forgetting'] - df['forgetting']) / df['baseline_forgetting'].abs().clip(lower=1e-10)

    # Method comparison
    method_stats = df.groupby('method').agg({
        'forgetting': ['mean', 'std'],
        'forgetting_reduction': ['mean', 'std'],
        'relative_time': ['mean', 'std'],
        'loss_t2_after_t2': ['mean', 'std'],
    }).round(4)

    # Flatten column names for JSON serialization
    method_stats.columns = ['_'.join(col).strip() for col in method_stats.columns.values]

    print("\n" + "=" * 60)
    print("Regularization Analysis Results")
    print("=" * 60)

    print("\n" + "-" * 50)
    print("Method Comparison (averaged across all conditions)")
    print("-" * 50)
    print(method_stats)

    # Statistical tests vs baseline
    print("\n" + "-" * 50)
    print("Statistical Tests vs Baseline (none)")
    print("-" * 50)

    baseline_forgetting = df[df['method'] == 'none']['forgetting'].values
    test_results = {}

    for method in ['l2', 'ewc']:
        method_df = df[df['method'] == method]

        # Group by lambda to find best
        lambda_stats = method_df.groupby('lambda_value').agg({
            'forgetting': 'mean',
            'forgetting_reduction': 'mean',
            'loss_t2_after_t2': 'mean',
        }).round(4)

        print(f"\n{method.upper()} by Lambda:")
        print(lambda_stats)

        # Find best lambda (most forgetting reduction)
        best_lambda = lambda_stats['forgetting_reduction'].idxmax()
        best_reduction = lambda_stats.loc[best_lambda, 'forgetting_reduction']

        # Statistical test for best lambda
        best_forgetting = method_df[method_df['lambda_value'] == best_lambda]['forgetting'].values

        # Match samples for paired test
        n_samples = min(len(baseline_forgetting), len(best_forgetting))
        t_stat, p_value = stats.ttest_ind(baseline_forgetting[:n_samples], best_forgetting[:n_samples])

        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""

        print(f"\n{method.upper()} Best Configuration:")
        print(f"  Best λ: {best_lambda}")
        print(f"  Mean reduction: {best_reduction*100:.1f}%")
        print(f"  t-statistic: {t_stat:.3f}")
        print(f"  p-value: {p_value:.2e} {sig}")

        test_results[method] = {
            'best_lambda': float(best_lambda),
            'best_reduction': float(best_reduction),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': bool(p_value < 0.05),
        }

    # Forgetting by similarity and lambda
    print("\n" + "-" * 50)
    print("Forgetting Reduction by Task Similarity (best λ)")
    print("-" * 50)

    for method in ['l2', 'ewc']:
        best_lambda = test_results[method]['best_lambda']
        method_df = df[(df['method'] == method) & (df['lambda_value'] == best_lambda)]

        print(f"\n{method.upper()} (λ={best_lambda}):")
        for sim in sorted(df['similarity'].unique()):
            sim_baseline = df[(df['method'] == 'none') & (df['similarity'] == sim)]['forgetting'].mean()
            sim_method = method_df[method_df['similarity'] == sim]['forgetting'].mean()
            reduction = (sim_baseline - sim_method) / abs(sim_baseline) if abs(sim_baseline) > 1e-10 else 0
            print(f"  Similarity {sim}: {reduction*100:+.1f}% reduction (baseline={sim_baseline:.4f}, {method}={sim_method:.4f})")

    # Lambda sweep analysis
    print("\n" + "-" * 50)
    print("Optimal Lambda by Similarity")
    print("-" * 50)

    optimal_lambdas = {}
    for method in ['l2', 'ewc']:
        method_df = df[df['method'] == method]
        optimal_lambdas[method] = {}

        print(f"\n{method.upper()}:")
        for sim in sorted(df['similarity'].unique()):
            sim_df = method_df[method_df['similarity'] == sim]
            # Filter out NaN values
            lambda_perf = sim_df.groupby('lambda_value')['forgetting'].mean().dropna()
            if len(lambda_perf) > 0:
                best_lambda = lambda_perf.idxmin()
                best_forgetting = lambda_perf.min()
                optimal_lambdas[method][str(sim)] = float(best_lambda)  # Use string key for JSON
                print(f"  Similarity {sim}: optimal λ={best_lambda}, forgetting={best_forgetting:.4f}")
            else:
                optimal_lambdas[method][str(sim)] = 0.0
                print(f"  Similarity {sim}: no valid results")

    # Best overall method
    print("\n" + "=" * 60)
    print("Best Method Analysis")
    print("=" * 60)

    # Find best (method, lambda) combination
    best_configs = df.groupby(['method', 'lambda_value']).agg({
        'forgetting': 'mean',
        'forgetting_reduction': 'mean',
        'relative_time': 'mean',
        'loss_t2_after_t2': 'mean',
    }).reset_index()

    # Exclude baseline
    best_configs = best_configs[best_configs['method'] != 'none']

    # Sort by forgetting reduction
    best_configs = best_configs.sort_values('forgetting_reduction', ascending=False)

    print("\nTop 5 Configurations:")
    for i, row in best_configs.head(5).iterrows():
        efficiency = row['forgetting_reduction'] / row['relative_time'] if row['relative_time'] > 0 else 0
        print(f"  {row['method'].upper()} λ={row['lambda_value']}: "
              f"{row['forgetting_reduction']*100:.1f}% reduction, "
              f"{row['relative_time']:.2f}x time, "
              f"T2 loss={row['loss_t2_after_t2']:.4f}")

    # Check if T2 performance is hurt
    print("\n" + "-" * 50)
    print("Task 2 Performance Impact (plasticity check)")
    print("-" * 50)

    baseline_t2 = df[df['method'] == 'none']['loss_t2_after_t2'].mean()
    print(f"\nBaseline T2 loss: {baseline_t2:.4f}")

    for method in ['l2', 'ewc']:
        method_df = df[df['method'] == method]
        for lambda_val in sorted(method_df['lambda_value'].unique()):
            lambda_df = method_df[method_df['lambda_value'] == lambda_val]
            t2_loss = lambda_df['loss_t2_after_t2'].mean()
            t2_degradation = (t2_loss - baseline_t2) / baseline_t2 * 100
            print(f"  {method.upper()} λ={lambda_val}: T2 loss={t2_loss:.4f} ({t2_degradation:+.1f}% vs baseline)")

    # Compile analysis
    analysis['method_comparison'] = method_stats.to_dict()
    analysis['statistical_tests'] = test_results
    analysis['optimal_lambdas'] = optimal_lambdas
    analysis['best_configurations'] = best_configs.head(10).to_dict('records')

    return analysis, df


def generate_report(analysis: Dict[str, Any], df: pd.DataFrame, output_dir: Path) -> None:
    """Generate markdown report."""
    report = f"""# Phase 6.3: Regularization Approaches Results
*Generated: {analysis['timestamp']}*

## Summary

This experiment tests whether regularization methods (EWC, L2) reduce catastrophic forgetting better than gradient projection (Phase 6.2: ~1% reduction).

### Methods Tested

| Method | Description |
|--------|-------------|
| **Baseline** | No regularization (standard SGD) |
| **L2** | λ/2 \\|\\|θ - θ_T1\\|\\|² |
| **EWC** | λ/2 Σ F_i (θ_i - θ_T1)² with Fisher Information |

## Key Results

"""
    # Best method
    tests = analysis['statistical_tests']
    best_method = max(tests.keys(), key=lambda m: tests[m]['best_reduction'])
    best_reduction = tests[best_method]['best_reduction']
    best_lambda = tests[best_method]['best_lambda']

    if best_reduction > 0.1:  # More than 10% reduction
        report += f"- **{best_method.upper()} significantly reduces forgetting** by {best_reduction*100:.1f}% (λ={best_lambda})\n"
    elif best_reduction > 0.01:  # More than 1% reduction
        report += f"- {best_method.upper()} provides modest reduction of {best_reduction*100:.1f}% (λ={best_lambda})\n"
    else:
        report += f"- Regularization provides **limited effectiveness** (~{best_reduction*100:.1f}% reduction)\n"

    for method, result in tests.items():
        sig = "significant" if result['significant'] else "not significant"
        report += f"- {method.upper()}: {result['best_reduction']*100:.1f}% reduction at λ={result['best_lambda']} ({sig})\n"

    # Method comparison table
    report += "\n## Method Comparison (Best λ for Each)\n\n"
    report += "| Method | Best λ | Mean Forgetting | Reduction | T2 Loss |\n"
    report += "|--------|--------|-----------------|-----------|----------|\n"

    baseline_forgetting = df[df['method'] == 'none']['forgetting'].mean()
    baseline_t2 = df[df['method'] == 'none']['loss_t2_after_t2'].mean()
    report += f"| NONE | - | {baseline_forgetting:.4f} | +0.0% | {baseline_t2:.4f} |\n"

    for method, result in tests.items():
        best_lambda = result['best_lambda']
        method_df = df[(df['method'] == method) & (df['lambda_value'] == best_lambda)]
        mean_forgetting = method_df['forgetting'].mean()
        t2_loss = method_df['loss_t2_after_t2'].mean()
        report += f"| {method.upper()} | {best_lambda} | {mean_forgetting:.4f} | {result['best_reduction']*100:+.1f}% | {t2_loss:.4f} |\n"

    # Optimal lambda by similarity
    report += "\n## Optimal λ by Task Similarity\n\n"
    report += "| Similarity | L2 Optimal λ | EWC Optimal λ |\n"
    report += "|------------|--------------|---------------|\n"

    opt = analysis['optimal_lambdas']
    for sim in sorted(opt['l2'].keys()):
        report += f"| {sim} | {opt['l2'][sim]} | {opt['ewc'][sim]} |\n"

    # Interpretation
    report += "\n## Interpretation\n\n"

    if best_reduction > 0.2:
        report += "### Strong Mitigation Effect\n\n"
        report += f"{best_method.upper()} with λ={best_lambda} provides substantial forgetting reduction. "
        report += "This validates the weight consolidation approach for catastrophic forgetting.\n\n"
    elif best_reduction > 0.05:
        report += "### Moderate Mitigation Effect\n\n"
        report += f"Regularization provides meaningful but not complete protection. "
        report += "Best configuration achieves {best_reduction*100:.1f}% reduction.\n\n"
    else:
        report += "### Limited Mitigation Effect\n\n"
        report += "Regularization methods show limited effectiveness in this setup, "
        report += "similar to gradient projection (Phase 6.2). "
        report += "Architecture-based methods or combined strategies may be needed.\n\n"

    # Recommendations
    report += "## Recommendations\n\n"

    if best_reduction > 0.05:
        report += f"- **Use {best_method.upper()} with λ={best_lambda}** for best forgetting reduction\n"
    else:
        report += "- Regularization alone provides limited benefit in this setting\n"

    report += "- Consider combined approach: regularization + gradient projection + task-specific heads\n"
    report += "- Tune λ based on expected task similarity\n"
    report += "- Monitor T2 performance to avoid excessive stability-plasticity tradeoff\n"

    # Connection to previous phases
    report += "\n## Connection to Previous Phases\n\n"
    report += "- **Phase 5.1**: Gradient interference is causal mechanism (r = -0.87)\n"
    report += "- **Phase 6.2**: Gradient projection provides ~1% reduction\n"
    report += f"- **Phase 6.3**: Regularization provides ~{best_reduction*100:.1f}% reduction\n"

    report += "\n---\n*Phase 6.3 Complete*\n"

    # Save report
    report_path = output_dir / "PHASE63_RESULTS.md"
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"\nReport saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Run Phase 6.3 regularization experiments")
    parser.add_argument("--quick", action="store_true", help="Quick test with reduced parameters")
    parser.add_argument("--output-dir", type=str, default="results/phase63", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.quick:
        config = Phase63Config(
            similarities=[0.0, 0.5, 1.0],
            learning_rates=[0.05],
            lambda_values=[0.0, 1.0, 100.0],
            methods=["none", "l2", "ewc"],
            n_seeds=2,
            n_steps=200,
        )
    else:
        config = Phase63Config()

    # Run experiments
    results = run_phase63_experiments(config)

    # Save raw results
    df = pd.DataFrame([asdict(r) for r in results])
    df.to_csv(output_dir / "phase63_regularization.csv", index=False)

    # Save config
    with open(output_dir / "phase63_config.json", 'w') as f:
        json.dump(asdict(config), f, indent=2)

    # Analyze
    analysis, df = analyze_results(results)

    # Save analysis
    with open(output_dir / "phase63_analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2, default=str)

    # Generate report
    generate_report(analysis, df, output_dir)

    print("\n" + "=" * 60)
    print("Phase 6.3 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
