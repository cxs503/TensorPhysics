"""Verification & validation (V&V) machine gate for cross-solver benchmarks.

Public surface
--------------
* :mod:`tensorphysics.vv.gci` -- Richardson extrapolation, observed order of
  accuracy, Roache grid convergence index (GCI), asymptotic-range checks and
  the PASS/WARN/FAIL gate that fuses them.
* :mod:`tensorphysics.vv.validation` -- ASME V&V 20 validation uncertainty
  ``U_val = sqrt(U_num**2 + U_input**2)`` and the ``|E| <= U_val`` acceptance
  test, plus a table driver.
* :mod:`tensorphysics.vv.adapters` -- normalisation of benchmark JSON emitted
  by the TensorLBM / TensorFVM / TensorFEM / TensorDEM / TensorBEM
  repositories into a single observation record consumed by the gate.

Command line
------------
``python -m tensorphysics.vv --obs obs.json`` (equivalently
``python -m tensorphysics.vv.gci --obs obs.json``).
"""
from .gci import (
    GatePolicy,
    GateVerdict,
    GridConvergenceStudy,
    Observation,
    evaluate,
    grid_convergence_study,
    observed_order,
    richardson_extrapolated,
)
from .validation import (
    ValidationResult,
    evaluate_validation_table,
    validation_uncertainty,
    validate,
)

__all__ = [
    "GatePolicy",
    "GateVerdict",
    "GridConvergenceStudy",
    "Observation",
    "ValidationResult",
    "evaluate",
    "evaluate_validation_table",
    "grid_convergence_study",
    "observed_order",
    "richardson_extrapolated",
    "validate",
    "validation_uncertainty",
]