#!/usr/bin/env python3
"""
Phase 5.1: Gradient Interference Analysis

Tests the hypothesis: Forgetting ∝ -cos(∇L_T1, ∇L_T2)

This script:
1. Runs experiments measuring gradient alignment during Task 2 training
2. Correlates gradient interference metrics with forgetting
3. Tests whether gradient angle predicts forgetting beyond similarity
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
import json
import argparse
from datetime import datetime

from src.gradient_interference import (
    Phase5Config,
    run_gradient_interference_experiment,
)


def run_phase5_experiment(
    config: Phase5Config,
    output_dir: Path,
    quick: bool = False
) -> pd.DataFrame:
    """Run full Phase 5.1 gradient interference experiment."""

    if quick:
        # Quick test configuration
        config.similarities = [0.0, 0.5, 1.0]
        config.learning_rates = [0.01]
        config.n_steps_list = [100]
        config.n_seeds = 2
        config.log_every = 20

    print(f"\n{'='*60}")
    print("Phase 5.1: Gradient Interference Analysis")
    print(f"{'='*60}")
    print(f"Total runs: {config.total_runs()}")
    print(f"Similarities: {config.similarities}")
    print(f"Learning rates: {config.learning_rates}")
    print(f"Training steps: {config.n_steps_list}")
    print(f"Seeds: {config.n_seeds}")
    print(f"{'='*60}\n")

    results = []
    total = config.total_runs()

    with tqdm(total=total, desc="Running experiments") as pbar:
        for similarity in config.similarities:
            for lr in config.learning_rates:
                for n_steps in config.n_steps_list:
                    for seed in range(config.n_seeds):
                        result = run_gradient_interference_experiment(
                            d_in=config.d_in,
                            d_out=config.d_out,
                            similarity=similarity,
                            learning_rate=lr,
                            n_steps=n_steps,
                            seed=seed,
                            batch_size=config.batch_size,
                            n_eval_samples=config.n_eval_samples,
                            log_every=config.log_every,
                            device=config.device
                        )
                        results.append(result.to_dict())
                        pbar.update(1)

    df = pd.DataFrame(results)

    # Save data
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "phase5_gradient_interference.csv", index=False)

    # Save config
    with open(output_dir / "phase5_config.json", 'w') as f:
        json.dump(config.to_dict(), f, indent=2)

    return df


def analyze_gradient_interference(df: pd.DataFrame, output_dir: Path) -> dict:
    """Analyze correlation between gradient metrics and forgetting."""

    print("\n" + "="*60)
    print("Gradient Interference Analysis Results")
    print("="*60)

    analysis = {
        'timestamp': datetime.now().isoformat(),
        'n_samples': len(df),
        'correlations': {},
        'hypothesis_tests': {},
        'key_findings': []
    }

    # Use all valid forgetting values (positive = true forgetting, negative = forward transfer)
    df_valid = df[df['forgetting'].notna()].copy()
    n_positive = (df_valid['forgetting'] > 0).sum()
    n_negative = (df_valid['forgetting'] <= 0).sum()
    print(f"\nTotal samples: {len(df_valid)}")
    print(f"  Positive forgetting (actual forgetting): {n_positive}")
    print(f"  Negative forgetting (forward transfer): {n_negative}")

    if len(df_valid) < 3:
        print("\nERROR: Not enough samples for statistical analysis")
        analysis['key_findings'].append("Insufficient samples for analysis")
        return analysis

    # Compute correlations with forgetting
    metrics_to_test = [
        'similarity',
        'mean_gradient_angle',
        'min_gradient_angle',
        'max_gradient_angle',
        'std_gradient_angle',
        'mean_gradient_projection',
        'cumulative_interference',
        'early_gradient_angle',
        'late_gradient_angle'
    ]

    print("\nCorrelations with Forgetting:")
    print("-" * 50)

    for metric in metrics_to_test:
        if metric in df_valid.columns:
            r, p = stats.pearsonr(df_valid['forgetting'], df_valid[metric])
            analysis['correlations'][metric] = {
                'pearson_r': float(r),
                'p_value': float(p),
                'significant': bool(p < 0.05)
            }
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {metric:30s}: r = {r:+.4f} (p = {p:.2e}) {sig}")

    # Key hypothesis test: Does gradient angle predict forgetting?
    print("\n" + "="*60)
    print("Hypothesis Test: forgetting ∝ -cos(∇L_T1, ∇L_T2)")
    print("="*60)

    # Main hypothesis: negative correlation between gradient angle and forgetting
    r_angle, p_angle = stats.pearsonr(df_valid['forgetting'], df_valid['mean_gradient_angle'])
    print(f"\nGradient Angle vs Forgetting:")
    print(f"  Pearson r: {r_angle:+.4f}")
    print(f"  P-value: {p_angle:.2e}")

    if r_angle < 0 and p_angle < 0.05:
        print("  ✓ HYPOTHESIS SUPPORTED: Negative gradient angles (destructive interference)")
        print("    correlate with higher forgetting")
        analysis['hypothesis_tests']['gradient_angle_hypothesis'] = 'SUPPORTED'
        analysis['key_findings'].append(
            f"Gradient angle negatively correlates with forgetting (r={r_angle:.3f}), "
            "supporting the hypothesis that destructive interference causes forgetting."
        )
    elif r_angle > 0 and p_angle < 0.05:
        print("  ✗ HYPOTHESIS REJECTED: Positive gradient angles correlate with forgetting")
        print("    (opposite of prediction)")
        analysis['hypothesis_tests']['gradient_angle_hypothesis'] = 'REJECTED'
        analysis['key_findings'].append(
            f"Gradient angle positively correlates with forgetting (r={r_angle:.3f}), "
            "contradicting the destructive interference hypothesis."
        )
    else:
        print("  ? INCONCLUSIVE: No significant correlation found")
        analysis['hypothesis_tests']['gradient_angle_hypothesis'] = 'INCONCLUSIVE'
        analysis['key_findings'].append(
            "No significant correlation between gradient angle and forgetting."
        )

    # Compare gradient angle to similarity as predictor
    r_sim, p_sim = stats.pearsonr(df_valid['forgetting'], df_valid['similarity'])
    print(f"\nSimilarity vs Forgetting (baseline):")
    print(f"  Pearson r: {r_sim:+.4f}")
    print(f"  P-value: {p_sim:.2e}")

    # Partial correlation: gradient angle controlling for similarity
    print("\n" + "="*60)
    print("Partial Correlation Analysis")
    print("="*60)

    # Residualize gradient angle on similarity
    from sklearn.linear_model import LinearRegression
    X_sim = df_valid['similarity'].values.reshape(-1, 1)
    y_angle = df_valid['mean_gradient_angle'].values
    y_forgetting = df_valid['forgetting'].values

    model = LinearRegression()
    model.fit(X_sim, y_angle)
    angle_residuals = y_angle - model.predict(X_sim)

    model.fit(X_sim, y_forgetting)
    forgetting_residuals = y_forgetting - model.predict(X_sim)

    r_partial, p_partial = stats.pearsonr(forgetting_residuals, angle_residuals)

    print(f"\nPartial correlation (gradient angle | similarity):")
    print(f"  r_partial: {r_partial:+.4f}")
    print(f"  P-value: {p_partial:.2e}")

    if abs(r_partial) > 0.1 and p_partial < 0.05:
        print("  ✓ Gradient angle provides additional explanatory power beyond similarity")
        analysis['key_findings'].append(
            f"Gradient angle has partial correlation r={r_partial:.3f} with forgetting "
            "even after controlling for similarity, suggesting it captures additional mechanism."
        )
    else:
        print("  ✗ Gradient angle does NOT explain variance beyond similarity")
        analysis['key_findings'].append(
            "Gradient angle does not explain additional variance in forgetting beyond similarity."
        )

    analysis['partial_correlation'] = {
        'gradient_angle_given_similarity': float(r_partial),
        'p_value': float(p_partial)
    }

    # Analyze by similarity groups
    print("\n" + "="*60)
    print("Gradient Angle by Similarity Group")
    print("="*60)

    df_valid['sim_group'] = pd.cut(df_valid['similarity'],
                                    bins=[0, 0.3, 0.7, 1.0],
                                    labels=['Low (0-0.3)', 'Medium (0.3-0.7)', 'High (0.7-1.0)'])

    group_stats = df_valid.groupby('sim_group').agg({
        'mean_gradient_angle': ['mean', 'std'],
        'forgetting': ['mean', 'std'],
        'cumulative_interference': ['mean', 'std']
    }).round(4)

    print("\n", group_stats.to_string())

    # Save analysis
    with open(output_dir / "phase5_analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2)

    return analysis


def generate_phase5_report(df: pd.DataFrame, analysis: dict, output_dir: Path):
    """Generate markdown report for Phase 5.1 results."""

    report = []
    report.append("# Phase 5.1: Gradient Interference Analysis Results\n")
    report.append(f"*Generated: {analysis['timestamp']}*\n")

    report.append("## Summary\n")
    report.append("This experiment tests whether gradient interference during Task 2 training ")
    report.append("causally explains catastrophic forgetting.\n")

    report.append("### Hypothesis\n")
    report.append("**H1**: Forgetting ∝ -cos(∇L_T1, ∇L_T2)\n")
    report.append("- Negative gradient angles (T2 updates pointing away from T1 minimum) → more forgetting\n")
    report.append("- Positive gradient angles (T2 updates compatible with T1) → less forgetting\n")

    report.append("\n## Key Results\n")
    for finding in analysis['key_findings']:
        report.append(f"- {finding}\n")

    report.append("\n## Correlation Analysis\n")
    report.append("| Metric | Pearson r | p-value | Significant |\n")
    report.append("|--------|-----------|---------|-------------|\n")

    for metric, stats_data in analysis['correlations'].items():
        sig = "Yes" if stats_data['significant'] else "No"
        report.append(f"| {metric} | {stats_data['pearson_r']:+.4f} | {stats_data['p_value']:.2e} | {sig} |\n")

    report.append("\n## Partial Correlation\n")
    if 'partial_correlation' in analysis:
        pc = analysis['partial_correlation']
        report.append(f"- Gradient angle | Similarity: r = {pc['gradient_angle_given_similarity']:+.4f} ")
        report.append(f"(p = {pc['p_value']:.2e})\n")

    report.append("\n## Hypothesis Verdict\n")
    verdict = analysis['hypothesis_tests'].get('gradient_angle_hypothesis', 'NOT TESTED')
    report.append(f"**{verdict}**\n")

    if verdict == 'SUPPORTED':
        report.append("\nGradient interference is a causal mechanism for forgetting. ")
        report.append("This suggests mitigation strategies based on gradient projection (e.g., OGD, A-GEM) ")
        report.append("could be effective.\n")
    elif verdict == 'REJECTED':
        report.append("\nGradient interference does NOT explain forgetting as hypothesized. ")
        report.append("The mechanism may be more subtle or similarity captures the causal factor directly.\n")
    else:
        report.append("\nNo conclusive evidence either way. More experiments may be needed.\n")

    report.append("\n## Interpretation\n")
    report.append("### What gradient angle tells us\n")
    report.append("- **Positive angle**: T2 gradient points in same direction as T1 gradient ")
    report.append("(learning T2 helps T1)\n")
    report.append("- **Zero angle**: Orthogonal gradients (independent learning)\n")
    report.append("- **Negative angle**: T2 gradient opposes T1 gradient (destructive interference)\n")

    report.append("\n### Connection to similarity\n")
    report.append("High task similarity → gradients tend to align → less destructive interference\n")
    report.append("Low task similarity → gradients tend to conflict → more destructive interference\n")

    report.append("\n## Next Steps\n")
    if verdict == 'SUPPORTED':
        report.append("1. **Phase 6.2**: Implement gradient projection methods (OGD, A-GEM)\n")
        report.append("2. Test if blocking destructive gradients reduces forgetting\n")
        report.append("3. Find minimal intervention for maximum forgetting reduction\n")
    else:
        report.append("1. **Phase 5.2**: Investigate representational overlap analysis\n")
        report.append("2. **Phase 5.3**: Study loss landscape geometry\n")
        report.append("3. Similarity may be the fundamental cause, not gradient interference\n")

    report.append("\n---\n*Phase 5.1 Complete*\n")

    with open(output_dir / "PHASE5_RESULTS.md", 'w') as f:
        f.writelines(report)

    print(f"\nReport saved to: {output_dir / 'PHASE5_RESULTS.md'}")


def main():
    parser = argparse.ArgumentParser(description="Run Phase 5.1: Gradient Interference Analysis")
    parser.add_argument("--quick", action="store_true", help="Quick test run")
    parser.add_argument("--analysis-only", action="store_true",
                       help="Only run analysis on existing data")
    parser.add_argument("--output-dir", type=str, default="results/phase5",
                       help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.analysis_only:
        # Load existing data
        data_path = output_dir / "phase5_gradient_interference.csv"
        if not data_path.exists():
            print(f"ERROR: No data found at {data_path}")
            print("Run without --analysis-only first to generate data.")
            sys.exit(1)
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} rows from {data_path}")
    else:
        # Run experiment
        config = Phase5Config()
        df = run_phase5_experiment(config, output_dir, quick=args.quick)

    # Analyze results
    analysis = analyze_gradient_interference(df, output_dir)

    # Generate report
    generate_phase5_report(df, analysis, output_dir)

    print("\n" + "="*60)
    print("Phase 5.1 Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
