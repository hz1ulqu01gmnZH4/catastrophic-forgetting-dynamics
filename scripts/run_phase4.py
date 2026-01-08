#!/usr/bin/env python3
"""
Phase 4: Trajectory Hypothesis Testing

Tests: Forgetting ∝ max_t ||θ_⊥(t)|| / ||θ_∥(t)||

Key questions:
1. Does path integral of deviation predict forgetting?
2. Does deviation velocity predict forgetting?
3. Is max deviation better than final deviation?
4. Can trajectory shape predict regime?
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from src.phase4_data_generation import (
    Phase4Config,
    generate_phase4_dataset,
    analyze_phase4_results
)


def main():
    print("=" * 60)
    print("PHASE 4: Trajectory Hypothesis Testing")
    print("=" * 60)
    print("\nHypothesis: Forgetting ∝ max_t ||θ_⊥(t)|| / ||θ_∥(t)||")

    # Configuration with dense tracking
    config = Phase4Config(
        d_in=50,
        d_out=5,
        hidden_widths=[32, 64, 128],
        activations=["gelu"],
        similarities=[0.0, 0.25, 0.5, 0.75, 1.0],
        learning_rates=[0.05, 0.1, 0.15],
        init_scales=[0.5, 1.0, 2.0],
        n_steps=300,
        n_seeds=5,
        track_every=10,  # Dense tracking!
        subspace_variance_target=0.90,
    )

    print(f"\nConfiguration:")
    print(f"  Hidden widths: {config.hidden_widths}")
    print(f"  Learning rates: {config.learning_rates}")
    print(f"  Track every: {config.track_every} steps (dense)")
    print(f"  Total runs: {config.total_runs()}")

    # Generate dataset
    print("\n" + "-" * 60)
    print("Generating Phase 4 dataset with dense trajectory tracking...")
    print("-" * 60)

    output_path = Path(__file__).parent.parent / "results" / "phase4" / "phase4_data.csv"
    df = generate_phase4_dataset(config, output_path=output_path, show_progress=True)

    print(f"\nGenerated {len(df)} data points")
    print(f"Saved to: {output_path}")

    # Analyze results
    print("\n" + "=" * 60)
    print("PHASE 4 ANALYSIS RESULTS")
    print("=" * 60)

    analysis = analyze_phase4_results(df)

    # 1. Basic Statistics
    print("\n1. BASIC STATISTICS")
    print("-" * 40)
    stats = analysis['basic_stats']
    print(f"  Samples: {stats['n_samples']}")
    print(f"  Mean Forgetting: {stats['mean_forgetting']:.4f}")
    print(f"  Std Forgetting: {stats['std_forgetting']:.4f}")

    # 2. Top Correlations with Forgetting
    print("\n2. TOP CORRELATIONS WITH FORGETTING")
    print("-" * 40)
    for i, (var, corr) in enumerate(analysis['correlations_with_forgetting'].items()):
        if i >= 15:
            break
        marker = "***" if abs(corr) > 0.5 else "**" if abs(corr) > 0.3 else "*" if abs(corr) > 0.1 else ""
        print(f"  {var}: {corr:+.4f} {marker}")

    # 3. Top Trajectory Predictors (excluding similarity/LR)
    print("\n3. TOP TRAJECTORY PREDICTORS")
    print("-" * 40)
    for var, corr in analysis.get('top_trajectory_predictors', {}).items():
        marker = "***" if abs(corr) > 0.5 else "**" if abs(corr) > 0.3 else ""
        print(f"  {var}: {corr:+.4f} {marker}")

    # 4. Predictor Comparison
    print("\n4. TRAJECTORY vs SIMPLE PREDICTORS")
    print("-" * 40)
    if 'predictor_comparison' in analysis:
        comp = analysis['predictor_comparison']
        print(f"  FLR correlation: {comp['flr_correlation']:.4f}")
        print(f"  Similarity correlation: {comp['similarity_correlation']:.4f}")
        print(f"  Best trajectory predictor: {comp['best_trajectory_predictor']}")
        print(f"  Best trajectory correlation: {comp['best_trajectory_correlation']:.4f}")
        print(f"  Trajectory beats FLR: {comp['trajectory_beats_flr']}")

    # 5. Trajectory Models
    print("\n5. TRAJECTORY-BASED MODELS")
    print("-" * 40)
    if 'trajectory_models' in analysis:
        models = analysis['trajectory_models']

        # Best single predictors
        single_models = [(k, v) for k, v in models.items()
                        if k.startswith('linear_') and 'r_squared' in v]
        single_models.sort(key=lambda x: x[1]['r_squared'], reverse=True)

        print("\n  Top 5 single-predictor models:")
        for name, result in single_models[:5]:
            pred_name = name.replace('linear_', '')
            print(f"    {pred_name}: R² = {result['r_squared']:.4f}")
            print(f"      {result['equation']}")

        # Multi-predictor model
        if 'multi_predictor' in models and 'r_squared' in models['multi_predictor']:
            mp = models['multi_predictor']
            print(f"\n  Multi-predictor model: R² = {mp['r_squared']:.4f}")
            print(f"    {mp['equation']}")

    # 6. Key Trajectory Metrics by Regime
    print("\n6. TRAJECTORY METRICS BY REGIME")
    print("-" * 40)
    if 'trajectory_by_regime' in analysis:
        for metric, stats in analysis['trajectory_by_regime'].items():
            print(f"\n  {metric}:")
            if 'mean' in stats:
                for regime, val in stats['mean'].items():
                    print(f"    {regime}: {val:.4f}")

    # 7. Key Discoveries
    print("\n" + "=" * 60)
    print("KEY PHASE 4 DISCOVERIES")
    print("=" * 60)

    # Find best trajectory predictor vs FLR
    traj_corrs = analysis.get('top_trajectory_predictors', {})
    best_traj_name, best_traj_corr = max(traj_corrs.items(), key=lambda x: abs(x[1])) if traj_corrs else (None, 0)

    flr_corr = analysis.get('predictor_comparison', {}).get('flr_correlation', 0)
    sim_corr = analysis.get('predictor_comparison', {}).get('similarity_correlation', 0)

    print(f"\n  Best trajectory predictor: {best_traj_name}")
    print(f"    Correlation: {best_traj_corr:+.4f}")
    print(f"  FLR correlation: {flr_corr:+.4f}")
    print(f"  Similarity correlation: {sim_corr:+.4f}")

    if best_traj_name and abs(best_traj_corr) > abs(flr_corr):
        print(f"\n  ✅ TRAJECTORY HYPOTHESIS SUPPORTED!")
        print(f"     {best_traj_name} beats FLR as predictor")
        print(f"     |{best_traj_corr:.3f}| > |{flr_corr:.3f}|")
    else:
        print(f"\n  ⚠️ Trajectory metrics don't beat FLR")
        print(f"     But may provide complementary information")

    # Check if max_deviation beats final_deviation
    max_dev_corr = traj_corrs.get('t2_max_deviation', 0)
    final_dev_corr = traj_corrs.get('t2_final_deviation', 0)
    if abs(max_dev_corr) > abs(final_dev_corr):
        print(f"\n  ✅ Max deviation ({max_dev_corr:+.3f}) beats final deviation ({final_dev_corr:+.3f})")
        print(f"     Confirms: trajectory matters, not endpoint")

    # Check path integral
    path_int_corr = traj_corrs.get('t2_path_integral_deviation', 0)
    print(f"\n  Path integral correlation: {path_int_corr:+.4f}")
    if abs(path_int_corr) > 0.3:
        print(f"     ✅ Cumulative deviation exposure matters")

    # Save analysis
    import json

    analysis_path = output_path.parent / "phase4_analysis.json"

    def make_serializable(obj):
        if isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
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
