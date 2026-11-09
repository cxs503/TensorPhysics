# UQ PCE Sobol' Gate: Polynomial Chaos Uncertainty Quantification

**Status**: implemented and verified · **Package**: `tensorphysics.uq` ·
**Reproduce**: `python scripts/uq_pce_sobol_gate_e2e.py`

## 1. Overview

The PCE Sobol' Gate provides a non-intrusive polynomial chaos expansion (PCE)
framework for global sensitivity analysis and uncertainty quantification. It
answers three questions in a machine-adjudicated way:

1. **Which inputs matter?** First-order, total-effect, and second-order Sobol'
   indices with bootstrap confidence intervals.
2. **How wide is the output band?** 95% prediction interval width `U` against
   an engineering threshold (default 15%).
3. **Can we trust the result?** Three-state verdict: PASS / FAIL / WITHHELD
   with explicit reason codes.

The method is **non-intrusive**: the solver is a black box
`points (n, d) -> values (n,)`. No solver internals are accessed.

## 2. Mathematical Foundation

### 2.1 Polynomial Chaos Expansion

Given a QoI `Y = f(X)` where `X = (X_1, ..., X_d)` are uncertain inputs with
known marginal distributions, the PCE approximates:

```
Y ≈ Ŷ = Σ_{α ∈ A} c_α Ψ_α(X)
```

where:
- `A` is a truncated multi-index set (total degree `≤ p`)
- `Ψ_α` are orthonormal polynomials (Legendre for uniform, Hermite for normal)
- `c_α` are coefficients computed by least-squares or LARS (sparse)

### 2.2 Analytic Sobol' Indices

The orthonormality of the basis yields **analytic** Sobol' indices:

```
S_α = c_α² / Var[Y]    (first-order and higher)
S_i = Σ_{α: α_i>0, α_j=0 ∀j≠i} S_α    (first-order)
S_Ti = Σ_{α: α_i>0} S_α    (total-effect)
S_ij = Σ_{α: α_i>0, α_j>0, α_k=0 ∀k≠i,j} S_α    (second-order)
```

No additional model evaluations are needed beyond the PCE fit.

### 2.3 Leave-One-Out Error

The surrogate quality is measured by the normalized leave-one-out error:

```
ε_LOO = (1/N) Σ_{i=1}^N [(Y_i - Ŷ_{-i}) / (Y_mean - Y_i)]²
```

where `Ŷ_{-i}` is the prediction from the PCE fitted without sample `i`.
A threshold `ε_LOO ≤ 0.1` is enforced; exceeding it triggers WITHHELD.

## 3. Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. SampleSpec: marginals (uniform/normal/lognormal) + copula    │
│    → unit hypercube [0,1]^d                                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. Sampler: Scrambled Sobol' / LHS / MC                         │
│    → design points with sha256 hash for reproducibility         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. Adapter: solver fan-out (callable or cluster shards)         │
│    → hash-pinned merge, NaN bookkeeping                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. PCE Fit: sparse LARS or OLS                                  │
│    → coefficients c_α, ε_LOO, R²                                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. Sobol' Indices: analytic from PCE + bootstrap CI             │
│    → first, total, second-order                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 6. Monte Carlo (optional): Saltelli/Jansen pick-freeze          │
│    → independent second opinion, first/total/second order       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 7. Cross-check: |ΔS| between PCE and MC                         │
│    → agreement flag if max relative deviation ≤ 5%              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 8. Verdict: 95% output interval U vs threshold (default 15%)    │
│    → PASS / FAIL / WITHHELD (+ reason codes)                    │
└─────────────────────────────────────────────────────────────────┘
```

## 4. Usage

### 4.1 Quick Start

```python
from tensorphysics.uq import run_uq_study, ishigami, ishigami_spec

study = run_uq_study(
    ishigami,
    ishigami_spec(),
    n_samples=1024,
    degree=8,
    run_mc=True,
    n_base_mc=1 << 14,
    n_boot=200,
)

print(f"Status: {study.status}")
print(f"First-order: {study.pce_sobol.first}")
print(f"Total-effect: {study.pce_sobol.total}")
print(f"ε_LOO: {study.surrogate.eps_loo:.4f}")
```

### 4.2 Custom Solver

```python
import numpy as np
from tensorphysics.uq import (
    run_uq_study, SampleSpec, Marginal, CallableAdapter
)

def my_solver(points: np.ndarray) -> np.ndarray:
    """points: (n, d) physical → values: (n,)"""
    return np.sum(points**2, axis=1)

spec = SampleSpec(
    marginals=[Marginal("uniform", low=0.0, high=1.0) for _ in range(3)],
    names=["x1", "x2", "x3"],
)

study = run_uq_study(my_solver, spec, n_samples=512, degree=4)
```

### 4.3 End-to-End Script

```bash
python scripts/uq_pce_sobol_gate_e2e.py --out artifacts/uq_gate
```

This runs the full pipeline on the Ishigami function (analytic indices known)
and the Sobol' G-function (d=4), producing:
- `ishigami_study.json`: full study results
- `g_function_study.json`: Sobol' G-function results
- `figures/`: convergence plots, index comparisons

### 4.4 Report Generation

```bash
python scripts/uq_pce_sobol_gate_report.py \
    --study artifacts/uq_gate/ishigami_study.json \
    --out artifacts/uq_gate/report.md
```

Generates a markdown report with:
- Input specification and sample hash
- Surrogate quality metrics (ε_LOO, R², n_terms)
- Sobol' indices table with bootstrap CI
- Cross-check results (PCE vs MC)
- Verdict with reason codes

## 5. Acceptance Criteria

The following criteria are machine-checked by `scripts/run_uq_acceptance.py`:

| ID | Criterion | Limit | Method |
|----|-----------|-------|--------|
| A1 | Ishigami PCE vs analytic: \|ΔS\| ≤ 1e-3 | 1e-3 | Absolute error |
| A2 | Ishigami MC vs analytic: relative error ≤ 5% | 0.05 | Relative error |
| A3 | ε_LOO decreases with degree (Spearman ≤ -0.9) | -0.9 | Monotonicity |
| A4 | ε_LOO decreases with sample count (Spearman ≤ -0.9) | -0.9 | Monotonicity |
| A5 | Determinism: same spec/seed → identical indices | exact | Bit-identical |
| A6 | WITHHELD on NaN/too few samples/high ε_LOO | - | Reason codes |
| A7 | PCE vs MC cross-check: relative deviation ≤ 5% | 0.05 | Agreement |
| A8 | No solver kernel touched | - | Hash check |
| A9 | Sobol' G-function: \|ΔS\| ≤ 5e-3 (d=4) | 5e-3 | Absolute error |
| A10 | Ship benchmark: narrow→PASS, wide→FAIL at 15% | - | Verdict |

## 6. Verdict System

### 6.1 Three-State Logic

- **PASS**: 95% interval width `U ≤ threshold` (default 15%)
- **FAIL**: `U > threshold`
- **WITHHELD**: surrogate quality insufficient (reason codes W01-W08)

### 6.2 Reason Codes

| Code | Meaning |
|------|---------|
| W01 | Non-finite values in samples |
| W02 | Too few samples (< min_samples) |
| W03 | ε_LOO > eps_loo_max (default 0.1) |
| W04 | PCE fit failed |
| W05 | Degenerate output (zero variance) |
| W06 | Bootstrap CI too wide |
| W07 | Cross-check missing or failed |
| W08 | Surrogate not available |

### 6.3 Engineering Interpretation

The threshold is an **engineering judgment**, not a statistical quantity. For
ship resistance:
- 15% threshold: typical for early-stage design
- 5% threshold: detailed design, model-test correlation required

The 95% interval `U` is a **prediction interval** of the QoI, not the
confidence interval of its mean. It answers: "given the input uncertainty,
how wide is the band on the output?"

## 7. Limitations and Applicability

### 7.1 Smoothness Assumption

PCE assumes the QoI is **smooth** in the inputs at the chosen total degree.
Discontinuous or highly non-smooth functions may require:
- Higher polynomial degree
- Adaptive enrichment (not yet implemented)
- Alternative surrogates (Gaussian processes, neural networks)

### 7.2 Correlated Inputs

For correlated inputs (Gaussian copula), the Sobol' indices are in the
**latent coordinate system**. The verdict attaches note `UQ-N01-CORRELATED`.
The Monte Carlo route only supports independent inputs.

### 7.3 Computational Cost

- PCE route: `N` solver evaluations + O(N p^d) fit
- MC route: `N_base (d+2)` solver evaluations
- Bootstrap: `n_boot` resamples (cheap, no solver calls)

For expensive solvers, the PCE route is preferred (analytic indices, no
additional evaluations). The MC route serves as an independent check.

## 8. References

1. Xiu, D., & Karniadakis, G. E. (2002). The Wiener-Askey polynomial chaos
   for stochastic differential equations. *SIAM Journal on Scientific
   Computing*, 24(2), 610-644.

2. Sobol', I. M. (2001). Global sensitivity indices for nonlinear mathematical
   models and their Monte Carlo estimates. *Mathematics and Computers in
   Simulation*, 55(1-3), 271-280.

3. Saltelli, A. (2002). Making best use of model evaluations to compute
   sensitivity indices. *Computer Physics Communications*, 145(2), 280-297.

4. Blatman, G., & Sudret, B. (2011). Adaptive sparse polynomial chaos
   expansion based on least angle regression. *Journal of Computational
   Physics*, 230(6), 2345-2367.

5. ASME V&V 20-2009. Standard for Verification and Validation in
   Computational Fluid Mechanics and Heat Transfer. ASME.
