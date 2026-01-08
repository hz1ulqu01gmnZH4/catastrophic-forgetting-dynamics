#!/usr/bin/env python3
"""
Phase 1: Linear Network Baseline Experiment

This script runs the complete Phase 1 experiment:
1. Generate forgetting data across hyperparameter sweep
2. Run symbolic regression to discover equations
3. Validate against known analytical predictions
4. Save results and generate report

Usage:
    python scripts/run_phase1.py [--quick] [--output-dir results/phase1]

Options:
    --quick         Run with reduced sweep for testing
    --output-dir    Directory for results (default: results/phase1)
    --device        Device to use (default: cpu)
    --sr-only       Skip data generation, run SR on existing data
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generation import ExperimentConfig, generate_forgetting_dataset, load_dataset
from src.symbolic_regression import (
    run_symbolic_regression,
    run_sr_with_physics_priors,
    save_sr_results,
    select_best_equations
)
from src.validation import (
    get_analytical_predictions,
    validate_sr_against_analytical,
    compute_validation_metrics,
    summarize_validation
)


def get_quick_config() -> ExperimentConfig:
    """Reduced config for quick testing."""
    return ExperimentConfig(
        d_in=50,
        d_out=5,
        widths=[25, 50, 100],
        similarities=[0.0, 0.25, 0.5, 0.75, 1.0],
        learning_rates=[0.01, 0.1],
        n_steps_list=[100, 500],
        n_seeds=3,
        batch_size=32,
    )


def get_full_config() -> ExperimentConfig:
    """Full config for comprehensive experiment."""
    return ExperimentConfig(
        d_in=100,
        d_out=10,
        widths=[50, 100, 200, 500, 1000],
        similarities=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        learning_rates=[0.001, 0.01, 0.1],
        n_steps_list=[100, 500, 1000, 2000],
        n_seeds=5,
        batch_size=64,
    )


def run_data_generation(config: ExperimentConfig, output_dir: Path) -> Path:
    """Generate forgetting dataset."""
    print("\n" + "=" * 60)
    print("PHASE 1: DATA GENERATION")
    print("=" * 60)

    print(f"\nConfiguration:")
    print(f"  Input dim: {config.d_in}")
    print(f"  Output dim: {config.d_out}")
    print(f"  Widths: {config.widths}")
    print(f"  Similarities: {config.similarities}")
    print(f"  Learning rates: {config.learning_rates}")
    print(f"  Steps: {config.n_steps_list}")
    print(f"  Seeds: {config.n_seeds}")
    print(f"  Total runs: {config.total_runs()}")

    output_path = output_dir / "forgetting_data.csv"

    df = generate_forgetting_dataset(
        config=config,
        output_path=output_path,
        show_progress=True
    )

    print(f"\nGenerated {len(df)} data points")
    print(f"Saved to: {output_path}")

    # Print summary statistics
    print("\nForgetting statistics:")
    print(f"  Mean: {df['forgetting'].mean():.4f}")
    print(f"  Std: {df['forgetting'].std():.4f}")
    print(f"  Min: {df['forgetting'].min():.4f}")
    print(f"  Max: {df['forgetting'].max():.4f}")

    return output_path


def run_symbolic_regression_phase(df, output_dir: Path, quick: bool = False):
    """Run symbolic regression on forgetting data."""
    print("\n" + "=" * 60)
    print("PHASE 1: SYMBOLIC REGRESSION")
    print("=" * 60)

    # Define predictor sets to try
    predictor_sets = [
        # Basic predictors
        ['similarity', 'learning_rate', 'n_steps'],
        # With overparameterization
        ['similarity', 'learning_rate', 'n_steps', 'overparameterization'],
        # Weight-based
        ['similarity', 'weight_change_t1', 'weight_change_t2'],
    ]

    targets = ['forgetting', 'forward_transfer']

    # SR parameters
    sr_params = {
        'niterations': 50 if quick else 100,
        'maxsize': 20 if quick else 30,
        'parsimony': 0.002 if quick else 0.001,
        'procs': 4,
    }

    all_results = []

    for target in targets:
        print(f"\n--- Target: {target} ---")

        for i, predictors in enumerate(predictor_sets):
            # Check all predictors exist
            missing = [p for p in predictors if p not in df.columns]
            if missing:
                print(f"  Skipping predictor set {i+1}: missing {missing}")
                continue

            print(f"\n  Predictor set {i+1}: {predictors}")

            try:
                result = run_symbolic_regression(
                    df=df,
                    target=target,
                    predictors=predictors,
                    aggregate_seeds=True,
                    **sr_params
                )

                print(f"    Best equation: {result.best_equation}")
                print(f"    Complexity: {result.best_complexity}")
                print(f"    R² score: {result.r2_score:.4f}")

                all_results.append(result)

            except ImportError as e:
                print(f"    Error: {e}")
                print("    Install PySR: pip install pysr")
                break
            except Exception as e:
                print(f"    Error: {e}")
                continue

    # Save results
    if all_results:
        save_sr_results(all_results, output_dir, "phase1")

        # Select best equations
        best = select_best_equations(all_results, complexity_threshold=15, r2_threshold=0.7)
        print(f"\n{len(best)} equations meet quality criteria (complexity≤15, R²≥0.7)")

    return all_results


def run_validation_phase(df, sr_results, output_dir: Path):
    """Validate SR results against analytical predictions."""
    print("\n" + "=" * 60)
    print("PHASE 1: VALIDATION")
    print("=" * 60)

    # Get analytical predictions
    analytical = get_analytical_predictions()

    # Split data for validation
    from sklearn.model_selection import train_test_split
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    all_comparisons = {}
    all_metrics = {}

    for result in sr_results:
        if result.target != 'forgetting':
            continue

        print(f"\nValidating: {result.best_equation}")

        # Compare with analytical
        comparisons = validate_sr_against_analytical(
            sr_equation=result.best_equation,
            analytical_predictions=analytical,
            df=df,
            target=result.target
        )
        all_comparisons[result.best_equation] = comparisons

        # Compute validation metrics
        metrics = compute_validation_metrics(
            df_train=df_train,
            df_test=df_test,
            sr_equation=result.best_equation,
            predictors=result.predictors,
            target=result.target
        )
        all_metrics[result.best_equation] = metrics

        # Print key metrics
        if 'test_r2' in metrics:
            print(f"  Test R²: {metrics['test_r2']:.4f}")
        if 'generalization_gap' in metrics:
            print(f"  Generalization gap: {metrics['generalization_gap']:.4f}")

    # Generate summary
    if sr_results:
        summary = summarize_validation(
            sr_results=[r for r in sr_results if r.target == 'forgetting'],
            analytical_comparisons=all_comparisons.get(sr_results[0].best_equation, {}),
            validation_metrics=all_metrics.get(sr_results[0].best_equation, {})
        )

        print("\n" + summary)

        # Save summary
        summary_path = output_dir / "validation_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(summary)
        print(f"\nSaved summary to: {summary_path}")

    return all_comparisons, all_metrics


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Linear Network Baseline")
    parser.add_argument('--quick', action='store_true', help="Quick test run")
    parser.add_argument('--output-dir', type=str, default='results/phase1',
                       help="Output directory")
    parser.add_argument('--device', type=str, default='cpu', help="Device")
    parser.add_argument('--sr-only', action='store_true',
                       help="Skip data generation, use existing data")
    args = parser.parse_args()

    # Setup
    output_dir = Path(__file__).parent.parent / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PHASE 1: LINEAR NETWORK BASELINE EXPERIMENT")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Configuration
    config = get_quick_config() if args.quick else get_full_config()
    config.device = args.device

    # Step 1: Data Generation
    data_path = output_dir / "forgetting_data.csv"

    if args.sr_only and data_path.exists():
        print(f"\nLoading existing data from {data_path}")
        df, _ = load_dataset(data_path)
    else:
        data_path = run_data_generation(config, output_dir)
        df, _ = load_dataset(data_path)

    # Step 2: Symbolic Regression
    try:
        sr_results = run_symbolic_regression_phase(df, output_dir, quick=args.quick)
    except ImportError:
        print("\nPySR not installed. Skipping symbolic regression.")
        print("Install with: pip install pysr")
        sr_results = []

    # Step 3: Validation
    if sr_results:
        try:
            comparisons, metrics = run_validation_phase(df, sr_results, output_dir)
        except Exception as e:
            print(f"\nValidation error: {e}")

    # Final report
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print(f"Finished: {datetime.now().isoformat()}")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)

    # Save run metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'config': config.to_dict(),
        'quick_mode': args.quick,
        'n_data_points': len(df),
        'n_sr_results': len(sr_results) if sr_results else 0,
    }

    with open(output_dir / "run_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
