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

High-level API (class-based, complementary to the functional gate)
------------------------------------------------------------------
* :mod:`tensorphysics.vv.richardson` -- :class:`RichardsonExtrapolation` and
  :class:`GCIResult`: a thin wrapper around the ``gci`` functional gate for
  the common "three grids, one observable" case.
* :mod:`tensorphysics.vv.asme_vv20` -- :class:`ASMEVV20` and
  :class:`UncertaintyBudget`: a thin wrapper around the ``validation``
  functional gate for the full ASME V&V 20 uncertainty budget.
* :mod:`tensorphysics.vv.cross_solver` -- :class:`CrossSolverAdapter`,
  :class:`CanonicalCase` and the canonical-case factories
  (``create_boundary_layer_case`` / ``create_cylinder_drag_case`` /
  ``create_lid_driven_cavity_case``): a thin adapter layer for registering
  solver results and comparing them on canonical test cases.
* :mod:`tensorphysics.vv.core` -- :class:`VVCore` and :class:`CoreVerdict`:
  the unified V&V machine gate composing the convergence, ASME V&V 20
  validation and cross-solver agreement stages into one fused verdict
  (``evaluate_core`` one-call entry point).

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
    CanonicalCase,
    ComparisonResult,
    ComparisonStatus,
    CrossSolverAdapter,
    SolverSample,
    agreement_spread,
    consensus,
    create_boundary_layer_case,
    create_cylinder_drag_case,
    create_lid_driven_cavity_case,
    cross_solver_agreement,
)
from .richardson import (
    ConvergenceStatus,
    GCIResult,
    RichardsonExtrapolation,
)
from .asme_vv20 import (
    ASMEVV20,
    UncertaintyBudget,
)
from .core import (
    CoreVerdict,
    VVCore,
    evaluate_core,
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
    # High-level API
    "CanonicalCase",
    "ComparisonResult",
    "ComparisonStatus",
    "CrossSolverAdapter",
    "create_boundary_layer_case",
    "create_cylinder_drag_case",
    "create_lid_driven_cavity_case",
    "ConvergenceStatus",
    "GCIResult",
    "RichardsonExtrapolation",
    "ASMEVV20",
    "UncertaintyBudget",
    # Core gate (m7729 / PR #199 re-open, m8768)
    "CoreVerdict",
    "VVCore",
    "evaluate_core",
]
