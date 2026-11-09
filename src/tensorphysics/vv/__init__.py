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
  repositories (and the ship-performance platform) into a single observation
  record consumed by the gate.
* :mod:`tensorphysics.vv.resolvability` -- the ``R = delta / dx``
  discriminability law (W10-N07): a declared physical feature that is subgrid
  turns the verdict into ``UNRESOLVED``, a missing declaration into ``WITHHELD``.
* :mod:`tensorphysics.vv.uncertainty` -- the full ASME V&V 20 uncertainty
  ledger ``U_num`` / ``U_input`` / ``U_val`` plus the reference term ``U_ref``
  that decides whether the comparison can discriminate at all.
* :mod:`tensorphysics.vv.cross_solver` -- the cross-solver agreement gate:
  inverse-variance weighted consensus, the largest pairwise agreement band and
  the PASS/WARN/FAIL verdict for a set of solvers reporting the same
  observable with no reference available.
* :mod:`tensorphysics.vv.schema` -- dependency-free validation of the
  cross-repository benchmark JSON contract.

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
from .resolvability import (
    DiscriminabilityPolicy,
    FeatureSpec,
    ResolvabilityResult,
    discriminability,
    required_grid_spacing,
)
from .uncertainty import UncertaintyLedger, compare, ledger_from_study, u_num_from_study
from .validation import (
    ValidationResult,
    evaluate_validation_table,
    validation_uncertainty,
    validate,
)
from .cross_solver import (
    AgreementPolicy,
    AgreementResult,
    SolverSample,
    agreement_spread,
    consensus,
    cross_solver_agreement,
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
    "DiscriminabilityPolicy",
    "FeatureSpec",
    "ResolvabilityResult",
    "UncertaintyLedger",
    "compare",
    "discriminability",
    "ledger_from_study",
    "required_grid_spacing",
    "u_num_from_study",
    "AgreementPolicy",
    "AgreementResult",
    "SolverSample",
    "agreement_spread",
    "consensus",
    "cross_solver_agreement",
]