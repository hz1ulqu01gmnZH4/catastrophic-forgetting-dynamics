"""
Symbolic regression pipeline using PySR.

Discovers closed-form equations for forgetting dynamics from experimental data.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
from dataclasses import dataclass


@dataclass
class SRResult:
    """Results from symbolic regression run."""
    target: str
    predictors: List[str]
    equations: pd.DataFrame  # Pareto front of equations
    best_equation: str
    best_complexity: int
    best_loss: float
    r2_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'target': self.target,
            'predictors': self.predictors,
            'best_equation': self.best_equation,
            'best_complexity': self.best_complexity,
            'best_loss': self.best_loss,
            'r2_score': self.r2_score,
        }


def prepare_data_for_sr(
    df: pd.DataFrame,
    target: str,
    predictors: List[str],
    aggregate_seeds: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for symbolic regression.

    Args:
        df: DataFrame with measurements
        target: Target variable name
        predictors: List of predictor variable names
        aggregate_seeds: Whether to average over random seeds

    Returns:
        (X, y) arrays for SR
    """
    if aggregate_seeds and 'seed' in df.columns:
        # Average over seeds to reduce noise
        group_cols = [p for p in predictors if p in df.columns]
        df_agg = df.groupby(group_cols)[target].mean().reset_index()
        X = df_agg[predictors].values
        y = df_agg[target].values
    else:
        X = df[predictors].values
        y = df[target].values

    # Remove NaN/inf values
    mask = np.isfinite(y)
    for i in range(X.shape[1]):
        mask &= np.isfinite(X[:, i])

    X = X[mask]
    y = y[mask]

    return X, y


def run_symbolic_regression(
    df: pd.DataFrame,
    target: str,
    predictors: List[str],
    aggregate_seeds: bool = True,
    niterations: int = 100,
    maxsize: int = 30,
    parsimony: float = 0.001,
    binary_operators: Optional[List[str]] = None,
    unary_operators: Optional[List[str]] = None,
    constraints: Optional[Dict] = None,
    procs: int = 4,
    random_state: int = 42,
    **kwargs
) -> SRResult:
    """
    Run symbolic regression to discover equations.

    Args:
        df: DataFrame with measurements
        target: Target variable to predict
        predictors: Predictor variable names
        aggregate_seeds: Average over random seeds first
        niterations: Number of SR iterations
        maxsize: Maximum equation complexity
        parsimony: Complexity penalty (higher = simpler equations)
        binary_operators: Binary operators to use
        unary_operators: Unary operators to use
        constraints: Operator constraints
        procs: Number of processes
        random_state: Random seed
        **kwargs: Additional PySR arguments

    Returns:
        SRResult with discovered equations
    """
    try:
        from pysr import PySRRegressor
    except ImportError:
        raise ImportError(
            "PySR not installed. Install with: pip install pysr\n"
            "Also requires Julia. See: https://astroautomata.com/PySR/"
        )

    # Prepare data
    X, y = prepare_data_for_sr(df, target, predictors, aggregate_seeds)

    if len(y) < 10:
        raise ValueError(f"Not enough data points ({len(y)}) for symbolic regression")

    # Default operators
    if binary_operators is None:
        binary_operators = ["+", "-", "*", "/", "^"]
    if unary_operators is None:
        unary_operators = ["exp", "log", "sqrt", "abs"]

    # Default constraints
    if constraints is None:
        constraints = {
            "^": (-1, 1),  # Limit exponent complexity
        }

    # Create and fit model
    model = PySRRegressor(
        niterations=niterations,
        binary_operators=binary_operators,
        unary_operators=unary_operators,
        constraints=constraints,
        maxsize=maxsize,
        parsimony=parsimony,
        procs=procs,
        random_state=random_state,
        progress=True,
        **kwargs
    )

    # Fit with variable names
    model.fit(X, y, variable_names=predictors)

    # Extract results
    equations_df = model.equations_

    # Find best equation (lowest loss on Pareto front)
    best_idx = equations_df['loss'].idxmin()
    best_eq = equations_df.loc[best_idx]

    # Compute R² score
    y_pred = model.predict(X)
    ss_res = ((y - y_pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot

    return SRResult(
        target=target,
        predictors=predictors,
        equations=equations_df,
        best_equation=str(best_eq['equation']),
        best_complexity=int(best_eq['complexity']),
        best_loss=float(best_eq['loss']),
        r2_score=float(r2),
    )


def run_sr_with_physics_priors(
    df: pd.DataFrame,
    target: str,
    predictors: List[str],
    expected_form: str = "power_law",
    **kwargs
) -> SRResult:
    """
    Run SR with physics-informed constraints.

    Args:
        df: Data
        target: Target variable
        predictors: Predictors
        expected_form: Expected functional form:
            - "power_law": a * x^b type
            - "exponential": a * exp(b*x) type
            - "rational": polynomial / polynomial
            - "additive": sum of terms
        **kwargs: Additional SR arguments

    Returns:
        SRResult
    """
    # Adjust operators based on expected form
    if expected_form == "power_law":
        binary_ops = ["+", "-", "*", "/", "^"]
        unary_ops = ["sqrt", "abs"]
        maxsize = 20
    elif expected_form == "exponential":
        binary_ops = ["+", "-", "*", "/"]
        unary_ops = ["exp", "log", "sqrt"]
        maxsize = 25
    elif expected_form == "rational":
        binary_ops = ["+", "-", "*", "/"]
        unary_ops = ["sqrt", "abs"]
        maxsize = 30
    else:  # additive
        binary_ops = ["+", "-", "*"]
        unary_ops = ["sqrt", "exp", "log"]
        maxsize = 35

    return run_symbolic_regression(
        df=df,
        target=target,
        predictors=predictors,
        binary_operators=binary_ops,
        unary_operators=unary_ops,
        maxsize=maxsize,
        **kwargs
    )


def compare_with_analytical(
    discovered_eq: str,
    analytical_form: str,
    test_data: pd.DataFrame,
    predictors: List[str],
    target: str
) -> Dict[str, float]:
    """
    Compare discovered equation with analytical prediction.

    Args:
        discovered_eq: Discovered equation string (sympy format)
        analytical_form: Known analytical equation string
        test_data: Test data for numerical comparison
        predictors: Predictor names
        target: Target name

    Returns:
        Dictionary with comparison metrics
    """
    import sympy as sp
    from scipy.stats import pearsonr

    # Create sympy symbols
    symbols = {p: sp.Symbol(p) for p in predictors}

    # Parse equations
    try:
        discovered_expr = sp.sympify(discovered_eq, locals=symbols)
        analytical_expr = sp.sympify(analytical_form, locals=symbols)
    except Exception as e:
        return {
            'parse_error': str(e),
            'symbolic_match': False,
            'correlation': float('nan'),
            'mse': float('nan'),
        }

    # Symbolic comparison
    try:
        diff = sp.simplify(discovered_expr - analytical_expr)
        symbolic_match = diff == 0
    except Exception:
        symbolic_match = False

    # Numerical comparison
    X, y_true = prepare_data_for_sr(test_data, target, predictors, aggregate_seeds=True)

    # Evaluate both equations
    discovered_func = sp.lambdify(list(symbols.values()), discovered_expr, 'numpy')
    analytical_func = sp.lambdify(list(symbols.values()), analytical_expr, 'numpy')

    try:
        y_discovered = discovered_func(*[X[:, i] for i in range(X.shape[1])])
        y_analytical = analytical_func(*[X[:, i] for i in range(X.shape[1])])

        correlation = pearsonr(y_discovered.flatten(), y_analytical.flatten())[0]
        mse = float(np.mean((y_discovered - y_analytical) ** 2))
    except Exception as e:
        correlation = float('nan')
        mse = float('nan')

    return {
        'symbolic_match': symbolic_match,
        'correlation': correlation,
        'mse': mse,
        'discovered_simplified': str(sp.simplify(discovered_expr)),
        'analytical_simplified': str(sp.simplify(analytical_expr)),
    }


def save_sr_results(
    results: List[SRResult],
    output_dir: Path,
    experiment_name: str
):
    """Save SR results to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save summary
    summary = []
    for r in results:
        summary.append(r.to_dict())

    with open(output_dir / f"{experiment_name}_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # Save equation details
    for i, r in enumerate(results):
        r.equations.to_csv(output_dir / f"{experiment_name}_{r.target}_equations.csv", index=False)


def select_best_equations(
    results: List[SRResult],
    complexity_threshold: int = 15,
    r2_threshold: float = 0.8
) -> List[SRResult]:
    """
    Filter SR results to select best equations.

    Args:
        results: List of SR results
        complexity_threshold: Maximum acceptable complexity
        r2_threshold: Minimum R² score

    Returns:
        Filtered results meeting criteria
    """
    filtered = []
    for r in results:
        if r.best_complexity <= complexity_threshold and r.r2_score >= r2_threshold:
            filtered.append(r)

    # Sort by R² score
    filtered.sort(key=lambda x: x.r2_score, reverse=True)

    return filtered
