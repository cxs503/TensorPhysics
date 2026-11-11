"""Operating-window envelope engine -- boundaries, not anecdotes.

An *operating window* is the subset of a declared parameter span on which every
constraint of a case holds.  Everywhere else in this repository such a window is
either a paragraph of prose, a hand-tuned one-off script, or a dense grid scan
whose answer depends on the grid.  This package replaces all three with one
stage-ordered mechanism:

===  ============================================================  ==========================
stage module                                                        question answered
===  ============================================================  ==========================
1    :mod:`~tensorphysics.envelope.constraints`                     what is the constraint
     (:mod:`~tensorphysics.envelope.expressions`)                   network, as data?
2    :mod:`~tensorphysics.envelope.grid`                            coarse feasibility mask,
                                                                     active set, cell area
3    :mod:`~tensorphysics.envelope.marching`                        where exactly is the
     (:mod:`~tensorphysics.envelope.roots`)                         boundary, and how many
                                                                     separate regions?
4    :mod:`~tensorphysics.envelope.contracts`                       is the answer
                                                                     believable?
5    :mod:`~tensorphysics.envelope.engine`                          one result object per
     (:mod:`~tensorphysics.envelope.artifact`)                      case and condition
===  ============================================================  ==========================

Zero domain vocabulary
----------------------
Names, spans, coefficients, tolerances, monotonicity expectations and the
discrete setting lattice all arrive through a case configuration
(:mod:`~tensorphysics.envelope.case`, schema
``tensorphysics.envelope.case.v1``).  Nothing in this package knows what a
parameter means, which unit it carries, or which constraint "should" bind.  The
shipped case files live outside the package (``scenarios/envelope_cases/``) and
are ordinary configuration data.

Semantics that are enforced, not assumed
----------------------------------------
* **status** is derived, never declared: ``feasible`` iff the marched structure
  is non-empty.
* **empty window** returns ``infeasible`` plus a nearest-feasible report with an
  explicit residual and the resolution of the search that produced it.  No
  boundary, no area, no extrapolated number escapes an empty window
  (:func:`~tensorphysics.envelope.contracts.check_infeasible_semantics`).
* **discrete feasibility** is applied after the continuous window and reported
  separately, so a continuous boundary is never presented as an attainable
  setting.
* **every check is falsifiable**: monotonicity, convexity, boundary order,
  empty-window semantics, resolution and window structure each return pass/fail
  with the numbers behind the verdict.
* **the probe scale is a number**: a segment's verdict comes from probes at a
  declared spacing (:mod:`~tensorphysics.envelope.subgrid`), not from a single
  midpoint sample.  A disagreement between probes is subdivided until it is
  settled; a span that cannot be settled is reported as unresolved and fails the
  ``resolution`` check instead of being averaged into a window.

Quick start::

    from tensorphysics.envelope import EnvelopeEngine, load_case

    case = load_case("scenarios/envelope_cases/<case>.json")
    result = EnvelopeEngine(case).run(condition="<label>")
    print(result.status, result.to_dict()["intervals"])
    assert result.contracts.ok
"""

from __future__ import annotations

from .artifact import artifact_digest, code_digest, read_artifact, write_artifact
from .case import (
    AxesSpec,
    CaseError,
    Condition,
    ContractPolicy,
    EnvelopeConfigError,
    SCHEMA,
    WindowCase,
    canonical_digest,
    case_from_dict,
    load_case,
)
from .constraints import (
    Constraint,
    ConstraintContract,
    ConstraintKind,
    ConstraintSet,
    EvalCounter,
    Evaluation,
    Lattice,
)
from .contracts import (
    CheckOutcome,
    ContractReport,
    boundary_order,
    check_area_consistency,
    check_area_vs_grid,
    check_boundary_order,
    check_convexity,
    check_infeasible_semantics,
    check_monotonicity,
    measure_component_boundary_error,
    symmetric_eigenvalues,
)
from .engine import (
    AnalyticReference,
    EnvelopeEngine,
    OrderStudy,
    WindowResult,
    WindowStatus,
    order_study,
)
from .expressions import ExpressionError, compile_expression, evaluate_expression, free_names
from .grid import NearestPoint, PlaneScan, brute_cell_area, nearest_feasible_point, scan_plane
from .marching import (
    BoundaryInterval,
    Component,
    DiscreteWindow,
    PlaneWindow,
    Window1D,
    march_plane,
    snap_lattice,
    window_1d,
)
from .roots import RootResult, bisect, brent, bracket, refine_bracket
from .subgrid import (
    ResolutionLedger,
    ResolutionPolicy,
    Span,
    check_span_partition,
)

__all__ = [
    "AnalyticReference",
    "AxesSpec",
    "BoundaryInterval",
    "CaseError",
    "CheckOutcome",
    "Component",
    "Condition",
    "Constraint",
    "ConstraintContract",
    "ConstraintKind",
    "ConstraintSet",
    "ContractPolicy",
    "ContractReport",
    "DiscreteWindow",
    "EnvelopeConfigError",
    "EnvelopeEngine",
    "EvalCounter",
    "Evaluation",
    "ExpressionError",
    "Lattice",
    "NearestPoint",
    "OrderStudy",
    "PlaneScan",
    "PlaneWindow",
    "RootResult",
    "SCHEMA",
    "Window1D",
    "WindowCase",
    "WindowResult",
    "WindowStatus",
    "artifact_digest",
    "bisect",
    "boundary_order",
    "bracket",
    "brent",
    "brute_cell_area",
    "canonical_digest",
    "case_from_dict",
    "check_area_consistency",
    "check_area_vs_grid",
    "check_boundary_order",
    "check_convexity",
    "check_infeasible_semantics",
    "check_monotonicity",
    "code_digest",
    "compile_expression",
    "evaluate_expression",
    "free_names",
    "load_case",
    "march_plane",
    "measure_component_boundary_error",
    "nearest_feasible_point",
    "order_study",
    "read_artifact",
    "refine_bracket",
    "ResolutionLedger",
    "ResolutionPolicy",
    "Span",
    "check_span_partition",
    "scan_plane",
    "snap_lattice",
    "symmetric_eigenvalues",
    "window_1d",
    "write_artifact",
]