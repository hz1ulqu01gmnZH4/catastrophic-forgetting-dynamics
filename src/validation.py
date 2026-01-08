"""
Validation module for comparing SR discoveries against analytical predictions.

Implements comparisons against known results from:
- Evron et al. (2022): Linear regression forgetting
- Goldfarb et al. (2024): Joint effect of similarity and overparameterization
"""

import numpy as np
import pandas as pd
import sympy as sp
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.stats import pearsonr
from scipy.optimize import curve_fit


@dataclass
class AnalyticalPrediction:
    """Container for analytical prediction and metadata."""
    name: str
    equation_str: str
    equation_sympy: sp.Expr
    source: str
    assumptions: List[str]
    parameter_ranges: Dict[str, Tuple[float, float]]


# Known analytical forms from literature
def get_analytical_predictions() -> Dict[str, AnalyticalPrediction]:
    """
    Return dictionary of known analytical equations.

    Based on:
    - Goldfarb et al. (2024): "Joint effect of task similarity and overparameterization"
    - Evron et al. (2022): "How catastrophic can catastrophic forgetting be in linear regression?"
    """

    # Define symbols
    n = sp.Symbol('overparameterization', positive=True)  # width / d_out
    s = sp.Symbol('similarity')
    t = sp.Symbol('n_steps', positive=True)
    eta = sp.Symbol('learning_rate', positive=True)

    predictions = {}

    # 1. Basic linear model forgetting (Evron et al. style)
    # Forgetting ~ O(1/n) for large overparameterization
    predictions['basic_overparameterization'] = AnalyticalPrediction(
        name="Basic Overparameterization Scaling",
        equation_str="c / overparameterization",
        equation_sympy=sp.Symbol('c') / n,
        source="Evron et al. (2022)",
        assumptions=["Linear model", "Two tasks", "Sufficient training"],
        parameter_ranges={'overparameterization': (1.0, 100.0)}
    )

    # 2. Task similarity effect (Goldfarb et al.)
    # Forgetting peaks at intermediate similarity for underparameterized
    # Decreases monotonically with similarity for overparameterized
    predictions['similarity_effect_underparameterized'] = AnalyticalPrediction(
        name="Similarity Effect (Underparameterized)",
        equation_str="a * similarity * (1 - similarity)",
        equation_sympy=sp.Symbol('a') * s * (1 - s),
        source="Goldfarb et al. (2024)",
        assumptions=["Linear model", "width < d_in", "Gaussian data"],
        parameter_ranges={'similarity': (0.0, 1.0), 'overparameterization': (0.1, 1.0)}
    )

    predictions['similarity_effect_overparameterized'] = AnalyticalPrediction(
        name="Similarity Effect (Overparameterized)",
        equation_str="a * (1 - similarity)",
        equation_sympy=sp.Symbol('a') * (1 - s),
        source="Goldfarb et al. (2024)",
        assumptions=["Linear model", "width > d_in", "Gaussian data"],
        parameter_ranges={'similarity': (0.0, 1.0), 'overparameterization': (1.0, 100.0)}
    )

    # 3. Combined effect (our synthesis)
    predictions['combined_linear'] = AnalyticalPrediction(
        name="Combined Linear Model",
        equation_str="(c1 / overparameterization) * (1 - similarity)^c2 + c3 * learning_rate * n_steps",
        equation_sympy=(
            sp.Symbol('c1') / n * (1 - s)**sp.Symbol('c2') +
            sp.Symbol('c3') * eta * t
        ),
        source="Synthesis of Goldfarb + Evron",
        assumptions=["Linear model", "Two tasks", "SGD training"],
        parameter_ranges={
            'overparameterization': (0.5, 50.0),
            'similarity': (0.0, 1.0),
            'learning_rate': (0.001, 0.1),
            'n_steps': (100, 5000)
        }
    )

    # 4. Gradient flow approximation (continuous time limit)
    predictions['gradient_flow'] = AnalyticalPrediction(
        name="Gradient Flow Limit",
        equation_str="(1 - similarity^2) * exp(-c * n_steps * learning_rate)",
        equation_sympy=(1 - s**2) * sp.exp(-sp.Symbol('c') * t * eta),
        source="Continuous-time analysis",
        assumptions=["Small learning rate", "Linear model", "Continuous time"],
        parameter_ranges={
            'similarity': (0.0, 1.0),
            'learning_rate': (0.0001, 0.01),
            'n_steps': (100, 10000)
        }
    )

    return predictions


def fit_analytical_form(
    df: pd.DataFrame,
    prediction: AnalyticalPrediction,
    target: str = 'forgetting'
) -> Dict[str, any]:
    """
    Fit free parameters in analytical form to data.

    Args:
        df: Data with measurements
        prediction: Analytical prediction to fit
        target: Target variable name

    Returns:
        Dictionary with fitted parameters and metrics
    """
    # Get symbols and free parameters
    expr = prediction.equation_sympy
    all_symbols = expr.free_symbols

    # Separate data variables from free parameters
    data_vars = ['overparameterization', 'similarity', 'n_steps', 'learning_rate',
                 'width', 'lr']
    param_symbols = [s for s in all_symbols if str(s) not in data_vars]
    data_symbols = [s for s in all_symbols if str(s) in data_vars]

    if not param_symbols:
        # No free parameters to fit
        return {'error': 'No free parameters in analytical form'}

    # Create fitting function
    def model_func(X, *params):
        param_dict = dict(zip([str(s) for s in param_symbols], params))

        # Map column names to symbol names
        col_map = {
            'width': 'overparameterization',  # We may use width in data
            'lr': 'learning_rate',
        }

        results = np.zeros(len(X))
        for i in range(len(X)):
            subs = param_dict.copy()
            for j, sym in enumerate(data_symbols):
                sym_name = str(sym)
                # Find matching column
                for col in df.columns:
                    if col == sym_name or col_map.get(col) == sym_name:
                        subs[sym_name] = X[i, j]
                        break

            try:
                val = float(expr.subs(subs))
                results[i] = val
            except Exception:
                results[i] = np.nan

        return results

    # Prepare data
    X_cols = []
    for sym in data_symbols:
        sym_name = str(sym)
        found = False
        for col in df.columns:
            if col == sym_name or (col == 'learning_rate' and sym_name == 'learning_rate'):
                X_cols.append(col)
                found = True
                break
        if not found:
            # Try mapping
            for col, mapped in [('width', 'overparameterization'), ('lr', 'learning_rate')]:
                if col in df.columns and sym_name == mapped:
                    X_cols.append(col)
                    found = True
                    break
        if not found:
            return {'error': f'Missing column for {sym_name}'}

    X = df[X_cols].values
    y = df[target].values

    # Remove NaN
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X = X[mask]
    y = y[mask]

    if len(y) < len(param_symbols) + 1:
        return {'error': 'Not enough data points'}

    # Initial parameter guess
    p0 = [1.0] * len(param_symbols)

    try:
        popt, pcov = curve_fit(model_func, X, y, p0=p0, maxfev=10000)

        # Compute metrics
        y_pred = model_func(X, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot

        fitted_params = dict(zip([str(s) for s in param_symbols], popt))

        return {
            'fitted_params': fitted_params,
            'r2': r2,
            'rmse': np.sqrt(np.mean((y - y_pred) ** 2)),
            'param_std': dict(zip([str(s) for s in param_symbols], np.sqrt(np.diag(pcov)))),
            'n_points': len(y),
        }

    except Exception as e:
        return {'error': str(e)}


def validate_sr_against_analytical(
    sr_equation: str,
    analytical_predictions: Dict[str, AnalyticalPrediction],
    df: pd.DataFrame,
    target: str = 'forgetting'
) -> Dict[str, Dict]:
    """
    Compare SR-discovered equation against all analytical predictions.

    Args:
        sr_equation: Discovered equation string
        analytical_predictions: Dictionary of analytical predictions
        df: Test data
        target: Target variable

    Returns:
        Dictionary mapping prediction name to comparison results
    """
    results = {}

    # Parse SR equation
    try:
        sr_expr = sp.sympify(sr_equation)
    except Exception as e:
        return {'parse_error': str(e)}

    for name, pred in analytical_predictions.items():
        # Check symbolic similarity
        try:
            # Simplify both
            sr_simple = sp.simplify(sr_expr)
            an_simple = sp.simplify(pred.equation_sympy)

            # Check if structurally similar (same terms)
            sr_terms = set(str(t) for t in sr_simple.as_ordered_terms())
            an_terms = set(str(t) for t in an_simple.as_ordered_terms())
            term_overlap = len(sr_terms & an_terms) / max(len(sr_terms | an_terms), 1)
        except Exception:
            term_overlap = 0.0

        # Fit analytical form
        fit_result = fit_analytical_form(df, pred, target)

        results[name] = {
            'analytical_form': pred.equation_str,
            'term_overlap': term_overlap,
            'fit_result': fit_result,
            'source': pred.source,
            'assumptions': pred.assumptions,
        }

    return results


def compute_validation_metrics(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    sr_equation: str,
    predictors: List[str],
    target: str = 'forgetting'
) -> Dict[str, float]:
    """
    Compute comprehensive validation metrics for SR equation.

    Args:
        df_train: Training data
        df_test: Held-out test data
        sr_equation: Discovered equation
        predictors: Predictor variables
        target: Target variable

    Returns:
        Dictionary of metrics
    """
    import sympy as sp

    # Parse equation
    symbols = {p: sp.Symbol(p) for p in predictors}
    try:
        expr = sp.sympify(sr_equation, locals=symbols)
        func = sp.lambdify(list(symbols.values()), expr, 'numpy')
    except Exception as e:
        return {'parse_error': str(e)}

    metrics = {}

    for name, df in [('train', df_train), ('test', df_test)]:
        X = df[predictors].values
        y_true = df[target].values

        # Remove NaN
        mask = np.isfinite(y_true)
        X = X[mask]
        y_true = y_true[mask]

        try:
            y_pred = func(*[X[:, i] for i in range(X.shape[1])])

            # R² score
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = 1 - ss_res / ss_tot

            # RMSE
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

            # Correlation
            corr = pearsonr(y_true.flatten(), y_pred.flatten())[0]

            # Max error
            max_err = np.max(np.abs(y_true - y_pred))

            metrics[f'{name}_r2'] = r2
            metrics[f'{name}_rmse'] = rmse
            metrics[f'{name}_correlation'] = corr
            metrics[f'{name}_max_error'] = max_err
            metrics[f'{name}_n_points'] = len(y_true)

        except Exception as e:
            metrics[f'{name}_error'] = str(e)

    # Generalization gap
    if 'train_r2' in metrics and 'test_r2' in metrics:
        metrics['generalization_gap'] = metrics['train_r2'] - metrics['test_r2']

    return metrics


def summarize_validation(
    sr_results: List,
    analytical_comparisons: Dict[str, Dict],
    validation_metrics: Dict[str, float]
) -> str:
    """
    Generate human-readable validation summary.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("VALIDATION SUMMARY")
    lines.append("=" * 60)

    # SR Results
    lines.append("\n1. SYMBOLIC REGRESSION RESULTS")
    lines.append("-" * 40)
    for r in sr_results:
        lines.append(f"Target: {r.target}")
        lines.append(f"  Best equation: {r.best_equation}")
        lines.append(f"  Complexity: {r.best_complexity}")
        lines.append(f"  R² score: {r.r2_score:.4f}")
        lines.append("")

    # Analytical Comparison
    lines.append("\n2. COMPARISON WITH ANALYTICAL PREDICTIONS")
    lines.append("-" * 40)
    for name, comp in analytical_comparisons.items():
        lines.append(f"\n{name}:")
        lines.append(f"  Analytical form: {comp['analytical_form']}")
        lines.append(f"  Source: {comp['source']}")
        if 'fit_result' in comp and 'r2' in comp['fit_result']:
            lines.append(f"  Fitted R²: {comp['fit_result']['r2']:.4f}")
        lines.append(f"  Term overlap: {comp['term_overlap']:.2%}")

    # Validation Metrics
    lines.append("\n3. VALIDATION METRICS")
    lines.append("-" * 40)
    for k, v in validation_metrics.items():
        if isinstance(v, float):
            lines.append(f"  {k}: {v:.4f}")
        else:
            lines.append(f"  {k}: {v}")

    lines.append("\n" + "=" * 60)

    return "\n".join(lines)
