# ISO 19906 Ice-Pressure Partial-Factor Calibration

## Overview

This module implements the **target-reliability inverse calibration** of the
ISO 19906 ice-pressure design format. Given a target annual reliability index
$\beta_T = 4.0$ (per ISO 2394), it finds the partial factor pair
$(\gamma_f, \psi)$ such that a member designed by the format

$$\gamma_f \, F_k \le R_k / \psi$$

attains exactly the target reliability.

## Physical Background

ISO 19906 describes ice actions through a **pressure–area relationship**:

$$p(A) = p_{\text{ref}} \left(\frac{A}{A_{\text{ref}}}\right)^{-ex}$$

where $ex \approx 0.4$ for global pressures. The ice force on the structure is

$$F = k_F \, p(A) \, A = k_F \, p_{\text{ref}} \, A_{\text{ref}}^{ex} \, A^{1-ex}$$

Because $ex > 0$, the pressure and area that together produce the annual
extreme force are **negatively correlated**: a large contact area comes with
a low pressure. This negative correlation reduces the force scatter compared
to an independent combination of the same marginals.

## Method

### 1. Joint p–A State

The ice pressure and contact area are modeled as lognormal random variables
with correlation fixed by the ISO power law:

$$\rho_{\ln p, \ln A} = -ex \, \sigma_{\ln A} / \sigma_{\ln p}$$

The Nataf transform converts this log-space correlation to the physical-space
correlation required by the reliability solver.

### 2. Target-Reliability Inverse

The limit state depends on $(\gamma_f, \psi)$ only through the product
$\kappa = \gamma_f \psi$ (a member sized with $R_k = \kappa F_k$). The inverse
is a two-equation system:

$$F_1(\gamma_f, \psi) = \beta(\kappa) - \beta_T = 0$$
$$F_2(\gamma_f, \psi) = \gamma_f - F^*(\kappa) / F_k = 0$$

- $F_1$: code-calibration condition (achieved reliability equals target)
- $F_2$: JCSS/ISO 2394 design-value split (load factor = design-point action / characteristic action)

The system is solved by damped Newton iteration in log-space
($y = (\ln \gamma_f, \ln \psi)$) with three dispersed starting points to
verify uniqueness.

### 3. Verification

The FORM solution is cross-checked against:
- **Design-point importance sampling** (200k samples)
- **Crude Monte Carlo** (4M samples)
- **Rao–Blackwell MC** (resistance integrated analytically)
- **Closed-form ln-space index** (exact for the lognormal problem)

### 4. ISO 19906 Table Comparison

The calibrated pair is compared row-by-row against ISO 19906 Annex A tables
(load factors × resistance factors). For each table pair that falls more than
15% short of the calibrated $\kappa^*$, a **non-conservative working example**
is quantified (achieved $\beta$, $P_f$ ratio, resistance shortfall).

### 5. Negative Control

The same calibration is repeated with $\rho_{\ln p, \ln A} = 0$ (same
marginals, no correlation). The calibrated $\gamma_f$ must increase by at
least 5% — confirming that the negative correlation is physically meaningful
and not an artifact.

## Quick Start

```python
from tensorphysics.iso19906_calibration import run_full_calibration

results = run_full_calibration()
print(f"γ_f = {results['inverse']['solution']['gamma_f']:.4f}")
print(f"ψ   = {results['inverse']['solution']['psi']:.4f}")
print(f"β   = {results['inverse']['solution']['beta_achieved']:.6f}")
print(f"All criteria passed: {results['criteria_summary']['all_passed']}")
```

## Configuration

```python
config = {
    "model": {
        "p_ref_mpa": 2.5,        # Reference pressure [MPa]
        "exponent": 0.40,         # p-A exponent
        "a_ref_m2": 5.0,          # Reference area [m²]
        "a_median_m2": 5.0,       # Median annual extreme area [m²]
        "a_cov": 0.35,            # Area COV
        "pressure_model_cov": 0.15,
        "load_model_cov": 0.10,
        "resistance_cov": 0.12,
    },
    "beta_target": 4.0,
    "samples_mc": 4_000_000,
    "samples_is": 200_000,
}
results = run_full_calibration(config=config)
```

## Output Structure

The returned dictionary is JSON-serialisable and contains:

| Key | Description |
|---|---|
| `study` | Metadata (node, title, repository, design format) |
| `model` | PAreaForceModel parameters |
| `force_statistics` | Log-space moments of the ice force |
| `inverse` | Calibrated $(\gamma_f, \psi, \kappa)$, Newton trace |
| `crosscheck` | FORM/IS/MC comparison |
| `sensitivity` | Adjoint derivatives $\partial\beta/\partial\theta$ |
| `iso_deviations` | Table-by-table comparison with ISO 19906 |
| `negative_control` | $\rho=0$ control experiment |
| `machine_criteria` | Pass/fail rows with measured values |
| `criteria_summary` | Aggregate pass/fail count |

## Machine Acceptance Criteria

| Criterion | Description | Tolerance |
|---|---|---|
| (a) | $\beta$ accuracy | $\|\beta - \beta_T\| \le 0.1$ |
| (b) | ISO table deviation | $\le 10\%$ for at least one pair |
| (c) | MC convergence | $\text{CoV}(P_f) < 10\%$ at $10^6$ samples |
| (d) | Negative control | $\Delta\gamma_f \ge 5\%$ |

## References

- ISO 19906:2019 — Petroleum and natural gas industries — Fixed steel
  offshore structures in Arctic and ice-covered waters.
- ISO 2394:2015 — General principles on reliability of structures.
- ISO 19906 Annex A — Partial factor design format for ice actions.

## Module Structure

```
tensorphysics/
├── iso19906_calibration.py          # Top-level API (this module)
└── ice_partial_factors/
    ├── __init__.py                   # Package exports
    ├── p_area_joint.py               # Joint p-A force model
    ├── target_reliability_inverse.py # Newton inverse solver
    ├── target_inverse_checks.py      # FORM/IS/MC cross-checks
    ├── target_inverse_study.py       # Study assembly
    ├── calibration.py                # Original plate-stress calibration
    ├── pressure_area.py              # p-A trend + Gumbel/GEV fitting
    ├── limit_state.py                # Ice-belt plate limit state
    └── iso_tables.py                 # ISO 19906 table data
```
