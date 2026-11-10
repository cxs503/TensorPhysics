# Statistical Coverage Audit (SBC) — N03 Deliverable

**Mission**: m9362 / p270w1-n03-statistical-coverage-audit  
**Date**: 2026-11-10  
**Branch**: agent/m9362  

## Summary

A domain-agnostic coverage audit engine was built and applied to two mission cases.
The engine measures whether claimed 95% confidence intervals actually cover the true
parameter at the stated rate, using Monte-Carlo repetition with known ground truth.

## Deliverables

### 1. Coverage Audit Engine
- **Path**: `tools/stats/coverage_audit.py`
- **Interface**: Protocol-based (TruthGenerator, Estimator) — zero domain semantics
- **Features**:
  - Three error-injection levels (pure noise, model structure error, heteroscedastic)
  - Three-state verdict (WELL_CALIBRATED / CONSERVATIVE / DISTORTED)
  - Wilson score interval for coverage uncertainty
  - Stopping rule: minimum n for ±0.02 resolution
  - Coverage convergence curves

### 2. Test Suite
- **Path**: `tests/test_coverage_audit.py`
- **Tests**: 16 tests, all passing
- **Coverage**: Judge logic, Wilson interval, stopping rule, well-calibrated/biased/conservative estimators, JSON export, convergence curves

### 3. Case Studies
- **Path**: `tools/stats/case_studies.py`
- **Case 1 (m7272)**: Weibull shape parameter m, n=8, bootstrap B=500
- **Case 2 (m7269)**: σ divergence, N_peak≈60, chi-squared CI

### 4. Results
- **Path**: `artifacts/coverage_audit/case_studies.json`
- **Figures**: `artifacts/coverage_audit/figures/*.png`

## Key Findings

### Case 1: Weibull m (m7272)
| Level | Coverage | Verdict |
|-------|----------|---------|
| 0 (pure noise) | 0.798 | DISTORTED |
| 1 (model error) | 0.906 | DISTORTED |
| 2 (heteroscedastic) | 0.874 | DISTORTED |

**Conclusion**: The log-linear bootstrap CI severely under-covers. The bootstrap
percentile method is anti-conservative for small samples (n=8).

### Case 2: σ divergence (m7269)
| Level | Coverage | Verdict |
|-------|----------|---------|
| 0 (pure noise) | 0.960 | WELL_CALIBRATED |
| 1 (heavy-tailed) | 0.724 | DISTORTED |
| 2 (heteroscedastic) | 0.646 | DISTORTED |

**Conclusion**: The chi-squared CI is well-calibrated under Gaussian assumptions
but fails catastrophically under model error. Heavy tails and heteroscedasticity
are common in peak stress data, making this a critical finding.

## Recommendations

1. **m7272**: Replace bootstrap percentile CI with BCa (bias-corrected accelerated)
   bootstrap or profile likelihood for small samples.
2. **m7269**: Use robust variance estimators or bootstrap when non-Gaussianity
   is suspected. The chi-squared CI should not be blindly trusted.
3. **General**: All CI claims should be audited via this tool before being used
   for decision-making.

## Stopping Rule

For ±0.02 resolution on coverage estimation:
- n_min = 2401 replications (worst-case p=0.5)
- Our N=500 achieves Wilson HW ≈ 0.02–0.04 (adequate for diagnostics)

## Reproducibility

- Seed: 42
- Replications: 500
- All RNG via numpy.random.default_rng (deterministic)
- Full audit chain in `artifacts/coverage_audit/`

## Report

Submitted to Tensor API: report id=11035
