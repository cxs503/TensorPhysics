"""Failure-mode taxonomy for campaign DAG nodes that end in honest FAIL.

The taxonomy answers one operational question: *given a node that failed, what
should the scheduler do next?*  It is deliberately **domain-neutral**: classes
are named after the *method-level* mechanism that produced the FAIL (a missing
baseline, a non-converged iteration, a fidelity assumption, a missing input, a
defect, ...), never after a physics application.  Domain vocabulary reaches the
classifier only through external data files (``data/*.json``), which keeps the
code reusable across the twelve solver repositories.

Six classes are defined; ``RANDOM_BASELINE = 1 / 6`` is therefore the accuracy
of a chance labeler and is used as the reference line in all back-tests.

Each class carries

* ``definition`` -- one-sentence, testable statement of the mechanism;
* ``discriminating_signals`` -- the evidence fields that must be inspected to
  assign the class (these are the seed of the decision table's trigger fields);
* ``boundary_note`` -- how the class differs from its nearest neighbour, i.e.
  the confusion that the dual-annotation study is expected to hit.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

RANDOM_BASELINE = 1.0 / 6.0


class FailureClass(str, Enum):
    """Method-level failure mechanisms (order is stable; used for matrices)."""

    BASELINE_ABSENT = "BASELINE_ABSENT"
    TOLERANCE_EXCEEDED = "TOLERANCE_EXCEEDED"
    CONVERGENCE_STALL = "CONVERGENCE_STALL"
    MODEL_FIDELITY_LIMIT = "MODEL_FIDELITY_LIMIT"
    DATA_INSUFFICIENCY = "DATA_INSUFFICIENCY"
    IMPLEMENTATION_DEFECT = "IMPLEMENTATION_DEFECT"

    @property
    def code(self) -> str:
        return self.value


CLASSES: tuple[FailureClass, ...] = tuple(FailureClass)
CLASS_CODES: tuple[str, ...] = tuple(c.code for c in CLASSES)


@dataclass(frozen=True)
class ClassSpec:
    """Definition + discriminating evidence of one failure class."""

    code: str
    definition: str
    discriminating_signals: tuple[str, ...]
    boundary_note: str

    @property
    def cls(self) -> FailureClass:
        return FailureClass(self.code)


SPECS: tuple[ClassSpec, ...] = (
    ClassSpec(
        code=FailureClass.BASELINE_ABSENT.code,
        definition=(
            "The computation ran, but no independent reference (analytic limit, "
            "benchmark, or measurement) exists to adjudicate its accuracy, so the "
            "grading criterion cannot be discharged."
        ),
        discriminating_signals=("reference_lexeme", "absence_lexeme"),
        boundary_note=(
            "Differs from DATA_INSUFFICIENCY: the deliverable exists and is "
            "quantified; what is missing is the *yardstick*, not the input."
        ),
    ),
    ClassSpec(
        code=FailureClass.TOLERANCE_EXCEEDED.code,
        definition=(
            "A reference exists and the reported quantity misses the declared "
            "tolerance band by a quantified margin."
        ),
        discriminating_signals=("quantified_gap", "threshold_lexeme", "reference_lexeme"),
        boundary_note=(
            "Differs from MODEL_FIDELITY_LIMIT: the deviation is reported as a "
            "number against a reference, not explained away as an assumption."
        ),
    ),
    ClassSpec(
        code=FailureClass.CONVERGENCE_STALL.code,
        definition=(
            "An iterative process (linear solve, nonlinear loop, refinement study, "
            "sampling estimator) did not reach the requested tolerance or order."
        ),
        discriminating_signals=("convergence_lexeme", "residual_magnitude"),
        boundary_note=(
            "Differs from TOLERANCE_EXCEEDED: the target value itself is unreliable "
            "because the iterate never settled, so no deviation can be trusted."
        ),
    ),
    ClassSpec(
        code=FailureClass.MODEL_FIDELITY_LIMIT.code,
        definition=(
            "The result is bound by a declared simplification, idealisation, or "
            "validity range; accuracy cannot improve without changing the model "
            "form or leaving the envelope."
        ),
        discriminating_signals=("assumption_lexeme", "envelope_lexeme"),
        boundary_note=(
            "Differs from IMPLEMENTATION_DEFECT: the code is believed correct and "
            "the limitation is intrinsic to the chosen formulation."
        ),
    ),
    ClassSpec(
        code=FailureClass.DATA_INSUFFICIENCY.code,
        definition=(
            "Inputs, artefacts, calibration records, or delivered numbers are "
            "missing, so the node cannot be graded at all (withheld rather than "
            "failed)."
        ),
        discriminating_signals=("absence_lexeme", "artifact_lexeme", "withheld_lexeme"),
        boundary_note=(
            "Differs from BASELINE_ABSENT: nothing was delivered or nothing was "
            "measured, so even a yardstick would not help yet."
        ),
    ),
    ClassSpec(
        code=FailureClass.IMPLEMENTATION_DEFECT.code,
        definition=(
            "A concrete coding, unit, interface, or bookkeeping error was "
            "identified and is fixable without changing the method."
        ),
        discriminating_signals=("defect_lexeme",),
        boundary_note=(
            "Differs from MODEL_FIDELITY_LIMIT: a patch is expected to close the "
            "gap; if a fix was already applied and the gap persists, the class is "
            "re-assigned."
        ),
    ),
)

SPEC_BY_CODE: dict[str, ClassSpec] = {spec.code: spec for spec in SPECS}


def spec(code: str) -> ClassSpec:
    """Return the :class:`ClassSpec` for *code*, raising ``KeyError`` if unknown."""
    try:
        return SPEC_BY_CODE[code]
    except KeyError:  # pragma: no cover - defensive
        raise KeyError(f"unknown failure class: {code!r}") from None


__all__ = [
    "CLASSES",
    "CLASS_CODES",
    "RANDOM_BASELINE",
    "ClassSpec",
    "FailureClass",
    "SPECS",
    "SPEC_BY_CODE",
    "spec",
]