# Code Quality Assurance Report

**Codebase**: `/home/ak/blog/experiments/deeplearning-dynamics`
**Scan Date**: 2026-01-09
**Files Checked**: 19
**Verdict**: **FAIL**

---

## Executive Summary

| Category | Count | Severity |
|----------|-------|----------|
| Fallback Violations (HIGH) | 5 | Blocking |
| Fallback Violations (MEDIUM) | 9 | Should Fix |
| Implementation Shortcuts | 2 | Documented |
| Error Handling Issues | 3 | Should Fix |
| Missing Test Coverage | Yes | Risk |

The codebase has **5 blocking issues** that violate the zero-tolerance policy for silent failures and fallback functionality. These must be fixed before the code can be considered production-ready.

---

## HIGH Severity: Fallback Violations

### HIGH-1: Silent SVD Failure

**File**: `src/data_generation.py:154-155`

```python
except Exception:
    return float('nan')
```

**Issue**: `compute_effective_rank()` silently returns NaN when SVD fails instead of propagating the error. This masks numerical instability and can corrupt downstream analysis.

**Fix**: Raise explicit error with context:
```python
except Exception as e:
    raise RuntimeError(f"SVD failed in compute_effective_rank: {e}")
```

---

### HIGH-2: Silent Experiment Continuation (Phase 2)

**File**: `src/phase2_data_generation.py:323-325`

```python
except Exception as e:
    print(f"Error: {e}")
    continue
```

**Issue**: Failed experiments are silently skipped. This can lead to biased results if certain configurations systematically fail, and there's no tracking of which experiments failed or why.

**Fix**: Track failures and enforce threshold:
```python
failed_experiments = []
# ... in loop:
except Exception as e:
    failed_experiments.append({
        'config': current_config,
        'error': str(e),
        'traceback': traceback.format_exc()
    })
    continue

# After loop:
failure_rate = len(failed_experiments) / total_experiments
if failure_rate > 0.05:
    raise RuntimeError(
        f"Experiment failure rate {failure_rate:.1%} exceeds 5% threshold. "
        f"Failed configs: {failed_experiments}"
    )
```

---

### HIGH-3: Silent Experiment Continuation (Phase 3)

**File**: `src/phase3_data_generation.py:454-456`

```python
except Exception as e:
    print(f"Error: {e}")
    continue
```

**Issue**: Same pattern as HIGH-2. Silent data loss in Phase 3 experiments.

**Fix**: Same as HIGH-2.

---

### HIGH-4: Silent Experiment Continuation (Phase 4)

**File**: `src/phase4_data_generation.py:397-399`

```python
except Exception as e:
    print(f"Error: {e}")
    continue
```

**Issue**: Same pattern as HIGH-2. Silent data loss in Phase 4 experiments.

**Fix**: Same as HIGH-2.

---

### HIGH-5: Silent Fallback to Degraded Subspace Fitting

**File**: `src/phase4_data_generation.py:276-278`

```python
except Exception:
    # Fallback: use initial weights only
    subspace.fit([W_init])
```

**Issue**: When subspace fitting fails on the weight trajectory, code silently falls back to fitting with only initial weights. This produces a degenerate 1-dimensional subspace that doesn't represent the actual learning dynamics.

**Fix**: Raise error or at minimum flag the result:
```python
except Exception as e:
    raise RuntimeError(
        f"Subspace fitting failed on trajectory of {len(weight_snapshots)} snapshots. "
        f"Cannot fall back to degraded mode: {e}"
    )
```

---

## MEDIUM Severity: Fallback Violations

### MEDIUM-1: Silent Zero Deviation

**File**: `src/phase4_data_generation.py:156-160`

```python
except Exception:
    deviation_trajectory.append(0.0)
```

**Issue**: Falls back to 0.0 when subspace analysis fails, hiding numerical issues.

---

### MEDIUM-2: Silent Last-Value Propagation

**File**: `src/phase4_data_generation.py:186-190`

```python
except Exception:
    deviation_trajectory.append(deviation_trajectory[-1] if deviation_trajectory else 0.0)
```

**Issue**: Propagates last value or zero on failure, masking when analysis breaks.

---

### MEDIUM-3: Silent Zero Deviation (Phase 3)

**File**: `src/phase3_data_generation.py:285-287`

```python
except Exception:
    deviations_t1.append(0.0)
```

**Issue**: Same silent fallback pattern.

---

### MEDIUM-4: Silent NaN in Validation

**File**: `src/validation.py:168-169`

```python
except Exception:
    results[i] = np.nan
```

**Issue**: Silent NaN substitution in `fit_analytical_form()` hides fitting failures.

---

### MEDIUM-5: Silent Zero Overlap

**File**: `src/validation.py:273-275`

```python
except Exception:
    term_overlap = 0.0
```

**Issue**: Silent fallback in `validate_sr_against_analytical()`.

---

### MEDIUM-6: Error Converted to Dict Key

**File**: `src/symbolic_regression.py:262-268`

```python
except Exception as e:
    return {
        'parse_error': str(e),
        ...
    }
```

**Issue**: Errors converted to dictionary keys instead of being raised. Callers must check for `parse_error` key.

---

### MEDIUM-7: Bare Except Clause

**File**: `scripts/generate_visualizations.py:369-370`

```python
except:
    deviations.append(deviations[-1] if deviations else 0)
```

**Issue**: Bare `except:` clause catches all exceptions including KeyboardInterrupt and SystemExit. Also uses silent fallback.

**Fix**:
```python
except Exception as e:
    raise RuntimeError(f"Deviation computation failed at step {i}: {e}")
```

---

### MEDIUM-8: Infinity/Zero Return on Degenerate Case

**File**: `src/universal_subspace.py:193-195`

```python
if parallel_norm > 1e-10:
    deviation_ratio = perpendicular_norm / parallel_norm
else:
    deviation_ratio = float('inf') if perpendicular_norm > 1e-10 else 0.0
```

**Issue**: Returns inf/0 on degenerate cases instead of raising. This is documented in CLAUDE.md as a known issue but still violates the no-fallback policy.

---

### MEDIUM-9: Maximum FLR on Degenerate Kernel

**File**: `src/nonlinear_models.py:177-178`

```python
if denom < 1e-10:
    return 1.0  # Maximum FLR if one kernel is degenerate
```

**Issue**: Silent fallback to 1.0 (maximum FLR) when kernel computation is degenerate.

---

## Implementation Compromises

### SHORTCUT-1: Simplified NTK Computation

**File**: `src/nonlinear_models.py:260-263`

```python
K_hidden = h @ h.T
ntk = K_hidden  # Simplified NTK proxy
```

**Status**: Documented in code comments. Uses feature kernel as proxy for full NTK. Acceptable for research purposes but affects accuracy.

---

### SHORTCUT-2: Width Parameter Metadata-Only

**File**: `src/data_generation.py:193-198`

```python
# For linear model: we simulate overparameterization via input dimension scaling
# or simply track width as metadata for later phases
```

**Status**: Documented. Width parameter doesn't affect Phase 1 linear model behavior - tracked as metadata only.

---

## Error Handling Issues

### ERROR-1: None Return on Missing Config

**File**: `src/data_generation.py:343-350`

```python
if config_path.exists():
    with open(config_path, 'r') as f:
        config = ExperimentConfig.from_dict(json.load(f))
else:
    config = None
```

**Issue**: Returns None when config file is missing instead of raising FileNotFoundError.

---

### ERROR-2: Swallowed ImportError

**File**: `scripts/run_phase1.py:285-290`

```python
try:
    sr_results = run_symbolic_regression_phase(...)
except ImportError:
    print("\nPySR not installed...")
    sr_results = []
```

**Issue**: Swallows ImportError and continues with empty results. Script should fail if PySR is required.

---

### ERROR-3: Broad Exception Catch in Validation

**File**: `scripts/run_phase1.py:293-297`

```python
try:
    comparisons, metrics = run_validation_phase(...)
except Exception as e:
    print(f"\nValidation error: {e}")
```

**Issue**: Catches all exceptions and continues silently without re-raising.

---

## Test Quality Issues

**Finding**: No pytest test files found in the codebase.

The files `src/nested_learning_test.py` and `src/deep_nesting_test.py` are experimental scripts that run experiments, not automated test suites.

**Risk**: No automated regression testing. All quality assurance is manual/visual.

**Recommendation**: Add pytest test coverage for core functions:
- `src/models.py`: Test training loops, loss computation
- `src/data_generation.py`: Test config parsing, metric computation
- `src/symbolic_regression.py`: Test equation parsing, Pareto selection
- `src/validation.py`: Test analytical comparisons

---

## Positive Observations

- No hardcoded credentials or secrets
- No security bypasses detected
- Well-organized modular code structure
- Comprehensive docstrings throughout
- No TODO/FIXME comments indicating incomplete work
- Configuration properly externalized
- No empty functions or stub implementations
- Clear separation of phases (1-4)

---

## Required Actions

### Blocking (Must Fix)

1. **Replace all `except Exception: continue` patterns** with error tracking that records failed configurations and raises if failure rate exceeds threshold (5%)

2. **Replace silent fallbacks** (NaN, 0.0, inf returns) with explicit assertions or raises

3. **Remove bare `except:` clause** in `generate_visualizations.py`

4. **Remove degraded fallback** in Phase 4 subspace fitting

5. **Raise on SVD failure** in `compute_effective_rank()`

### Recommended

6. Add pytest test coverage for core modules

7. Convert ImportError handling in `run_phase1.py` to fail-fast behavior

8. Replace None return with FileNotFoundError in config loading

---

## Appendix: Files Scanned

```
src/
├── data_generation.py
├── deep_nesting_test.py
├── gradient_interference.py
├── models.py
├── nested_learning_test.py
├── nonlinear_models.py
├── phase2_data_generation.py
├── phase3_data_generation.py
├── phase4_data_generation.py
├── symbolic_regression.py
├── trajectory_analysis.py
├── universal_subspace.py
└── validation.py

scripts/
├── generate_visualizations.py
├── run_phase1.py
├── run_phase3.py
├── run_phase4.py
└── visualize_phase4.py

config/
└── phase1_config.yaml
```

---

*Report generated by code-qa agent (no-fallback-functionality-enforcer + compromise-checker)*
