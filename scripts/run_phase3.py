#!/usr/bin/env python3
"""
Phase 3: Universal Subspace Analysis

Run experiments to discover:
1. Whether subspace deviation predicts forgetting
2. The transition boundary equation
3. If deviation is better predictor than FLR
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from src.phase3_data_generation import (
    Phase3Config,
    generate_phase3_dataset,
    analyze_phase3_results
)
from src.universal_subspace import (
    compute_transition_boundary,
    fit_transition_equation
)


def main():
    print("=" * 60)
    print("PHASE 3: Universal Subspace Analysis")
    print("=" * 60)

    # Configuration focused on transition region
    config = Phase3Config(
        d_in=50,
        d_out=5,
        hidden_widths=[32, 64, 128],
        activations=["gelu"],  # Best for feature learning signal
        similarities=[0.0, 0.25, 0.5, 0.75, 1.0],
        learning_rates=[0.05, 0.1, 0.15, 0.2],  # Focus on transition region
        init_scales=[0.5, 1.0, 2.0],
        n_steps=300,
        n_seeds=5,
        track_weights_every=30,
        subspace_variance_target=0.90,
    )

    print(f"\nConfiguration:")
    print(f"  Hidden widths: {config.hidden_widths}")
    print(f"  Learning rates: {config.learning_rates}")
    print(f"  Similarities: {config.similarities}")
    print(f"  Init scales: {config.init_scales}")
    print(f"  Total runs: {config.total_runs()}")

    # Generate dataset
    print("\n" + "-" * 60)
    print("Generating Phase 3 dataset...")
    print("-" * 60)

    output_path = Path(__file__).parent.parent / "results" / "phase3" / "phase3_data.csv"
    df = generate_phase3_dataset(config, output_path=output_path, show_progress=True)

    print(f"\nGenerated {len(df)} data points")
    print(f"Saved to: {output_path}")

    # Analyze results
    print("\n" + "=" * 60)
    print("PHASE 3 ANALYSIS RESULTS")
    print("=" * 60)

    analysis = analyze_phase3_results(df)

    # 1. Basic Statistics
    print("\n1. BASIC STATISTICS")
    print("-" * 40)
    stats = analysis['basic_stats']
    print(f"  Samples: {stats['n_samples']}")
    print(f"  Mean Forgetting: {stats['mean_forgetting']:.4f}")
    print(f"  Std Forgetting: {stats['std_forgetting']:.4f}")
    print(f"  Mean Deviation (T2): {stats['mean_deviation_t2']:.4f}")
    print(f"  Std Deviation (T2): {stats['std_deviation_t2']:.4f}")

    # 2. Correlations with Forgetting
    print("\n2. CORRELATIONS WITH FORGETTING")
    print("-" * 40)
    for var, corr in sorted(analysis['correlations_with_forgetting'].items(),
                            key=lambda x: abs(x[1]), reverse=True):
        print(f"  {var}: {corr:+.4f}")

    # 3. Regime Statistics
    print("\n3. REGIME STATISTICS")
    print("-" * 40)
    regime_df = df.groupby('regime_after_t2').agg({
        'forgetting': ['mean', 'std', 'count'],
        'deviation_after_t2': ['mean', 'std'],
        'flr_after_t2': ['mean', 'std'],
    }).round(4)
    print(regime_df.to_string())

    # 4. Transition Boundary
    print("\n4. TRANSITION BOUNDARY")
    print("-" * 40)
    if 'transition_boundary' in analysis:
        boundary = analysis['transition_boundary']
        print(f"  Deviation threshold: {boundary['deviation_threshold']:.4f}")
        print(f"  Forgetting threshold: {boundary['forgetting_threshold']:.4f}")
        print(f"  Separation: {boundary['separation']:.4f}")
        print(f"  Classification accuracy: {boundary['classification_accuracy']:.4f}")
        print(f"  Correlation: {boundary['correlation']:.4f}")

    # 5. Fitted Equations
    print("\n5. FITTED EQUATIONS")
    print("-" * 40)
    if 'fitted_equations' in analysis:
        for model_name, result in analysis['fitted_equations'].items():
            if 'error' in result:
                print(f"\n  {model_name}: ERROR - {result['error']}")
            else:
                print(f"\n  {model_name}:")
                print(f"    Equation: {result['equation']}")
                print(f"    R² = {result['r_squared']:.4f}")

    # 6. Deviation by Learning Rate
    print("\n6. DEVIATION BY LEARNING RATE")
    print("-" * 40)
    deviation_by_lr = df.groupby('learning_rate').agg({
        'deviation_after_t2': ['mean', 'std'],
        'forgetting': ['mean', 'std'],
        'regime_after_t2': lambda x: (x == 'rich').mean()
    }).round(4)
    deviation_by_lr.columns = ['dev_mean', 'dev_std', 'forg_mean', 'forg_std', 'rich_pct']
    print(deviation_by_lr.to_string())

    # 7. Predictor Comparison
    print("\n7. DEVIATION vs FLR AS PREDICTORS")
    print("-" * 40)
    if 'predictor_comparison' in analysis:
        comp = analysis['predictor_comparison']
        print(f"  FLR correlation with forgetting: {comp['flr_correlation']:.4f}")
        print(f"  Deviation correlation with forgetting: {comp['deviation_correlation']:.4f}")
        print(f"  Better predictor: {comp['better_predictor'].upper()}")

    # 8. Key Discovery Summary
    print("\n" + "=" * 60)
    print("KEY PHASE 3 DISCOVERIES")
    print("=" * 60)

    # Check if deviation predicts regime
    clean_df = df.dropna(subset=['deviation_after_t2', 'regime_after_t2'])
    lazy_dev = clean_df[clean_df['regime_after_t2'] == 'lazy']['deviation_after_t2'].mean()
    rich_dev = clean_df[clean_df['regime_after_t2'] == 'rich']['deviation_after_t2'].mean()

    print(f"\n  Lazy regime mean deviation: {lazy_dev:.4f}")
    print(f"  Rich regime mean deviation: {rich_dev:.4f}")
    if rich_dev > 0:
        print(f"  Ratio (rich/lazy): {rich_dev/lazy_dev:.2f}x")

    # Best model
    if 'fitted_equations' in analysis:
        best_model = max(
            [(k, v) for k, v in analysis['fitted_equations'].items() if 'r_squared' in v],
            key=lambda x: x[1]['r_squared'],
            default=(None, None)
        )
        if best_model[0]:
            print(f"\n  Best model: {best_model[0]}")
            print(f"  R² = {best_model[1]['r_squared']:.4f}")
            print(f"  {best_model[1]['equation']}")

    # Save analysis
    import json

    analysis_path = output_path.parent / "phase3_analysis.json"

    # Convert to JSON-serializable format
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        else:
            return obj

    with open(analysis_path, 'w') as f:
        json.dump(make_serializable(analysis), f, indent=2)

    print(f"\n\nAnalysis saved to: {analysis_path}")

    return df, analysis


if __name__ == "__main__":
    df, analysis = main()
