# Winkler Elastic Foundation Plate Model (SIR-PC2|F01)

## Overview

This module implements the **Winkler elastic foundation plate** model within the
`tensorphysics.ice_sheet` sub-package, targeting ice mechanics for floating ice
sheets. The model solves the governing biharmonic equation on a Winkler
foundation and validates against the Hetényi-type Navier analytical solution.

## Physical Model

A thin elastic plate of dimensions *a × b* and thickness *h* rests on a Winkler
elastic foundation with modulus *k_w* and is subjected to a uniform transverse
load *q*. The governing equation is:

    D ∇⁴w + k_w w = q

where *D = Eh³/[12(1−ν²)]* is the flexural rigidity.

For a **floating ice sheet**, the Winkler modulus arises from buoyancy
(Archimedes' principle):

    k_w = ρ_w · g

A unit downward deflection displaces additional water, generating a restoring
pressure proportional to the deflection.

## Solvers

### Navier Analytical Solution

For a simply supported rectangular plate, the deflection is expressed as a
double Fourier sine series:

    w(x,y) = Σ_m Σ_n W_mn sin(mπx/a) sin(nπy/b)

with coefficients:

    W_mn = q_mn / [D π⁴ (m²/a² + n²/b²)² + k_w]
    q_mn = 16q / (mn π²)   for m, n odd

This is the Hetényi-type analytical reference for plates on elastic foundation.

### Finite Difference Solver

The biharmonic equation is split into two coupled Laplacian equations:

    ∇²w = φ
    ∇²φ = (q − k_w w) / D

with simply supported BCs: *w = 0*, *∇²w = 0* on the boundary. The resulting
linear system is solved directly via `numpy.linalg.solve`.

## Acceptance Criteria

| Criterion | Value | Threshold | Status |
|-----------|-------|-----------|--------|
| center_deflection_relative_error | 3.48×10⁻⁴ | < 0.05 | ✅ PASS |
| kw0_free_plate_error | 0.0 m | < 1×10⁻¹⁰ m | ✅ PASS |
| pytest test_winkler_plate | 13/13 passed | PASS | ✅ PASS |

## Module Structure

```
src/tensorphysics/ice_sheet/
├── __init__.py
└── winkler_plate.py       # Core implementation

tests/
└── test_winkler_plate.py  # 13 pytest test cases

artifacts/
├── generate_figures.py
├── report_data.json
├── winkler_deflection_contour.png
├── winkler_centerline_comparison.png
└── winkler_convergence.png
```

## Usage

```python
from tensorphysics.ice_sheet import (
    WinklerPlateParams,
    winkler_kw_parametric,
    navier_center_deflection,
    navier_deflection_field,
    solve_winkler_plate_fd,
)

# Set up a floating ice sheet
params = WinklerPlateParams(
    a=100.0, b=100.0, h=2.0,      # 100m × 100m, 2m thick
    E=5.0e9, nu=0.33,              # Sea ice properties
    k_w=winkler_kw_parametric(),   # Buoyancy: k_w = ρ_w·g ≈ 10055 Pa/m
    q=1000.0,                      # 1 kPa uniform load
)

# Analytical center deflection
w_c = navier_center_deflection(params)  # ≈ 63.27 mm

# Full deflection field (Navier)
x, y, w = navier_deflection_field(params, nx=51, ny=51)

# Finite difference solution
x, y, w = solve_winkler_plate_fd(params, nx=51, ny=51)
```
