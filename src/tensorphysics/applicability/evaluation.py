"""One evaluation of a registered capability at one point of the axis space.

Three independent failure signals are checked here (stage 2 criteria):

(a) **acceptance criterion** -- the measured scalar against its reference /
    tolerance window (``Criterion``);
(b) **monotonicity / sign law** -- a *sequence* property, therefore applied by
    the probing operator (``probe.py``) across the scan, see
    :func:`law_violation`;
(c) **numerical pathology** -- non-finite value, declared failure flags, or a
    convergence-order collapse (``FailurePolicy.pathology``).

A sample is *in domain* only when it passes all of them; anything that raises,
diverges or is unknown is **not** certified (fail-closed).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from .capability import CapabilitySpec
from .axis import AxisSpec

__all__ = ["Evaluation", "evaluate_point", "law_violation", "PASS", "CRITERION", "PATHOLOGY",
           "LAW", "ERROR", "IN_STATUSES"]

PASS = "pass"
CRITERION = "criterion"
PATHOLOGY = "pathology"
LAW = "law"
ERROR = "error"
IN_STATUSES = (PASS,)

#: Numerically meaningful failures: the model went pathological (divergence,
#: NaN, breakdown).  Everything else is a wiring surprise and is reported as
#: ``error`` -- also not certified, but never silently upgraded to "out of
#: domain due to physics".
_NUMERIC_EXCEPTIONS = (ValueError, ArithmeticError, FloatingPointError, ZeroDivisionError,
                       OverflowError, RuntimeError)


@dataclass(frozen=True)
class Evaluation:
    """Outcome of one capability evaluation, with the reason chain."""

    capability_id: str
    point: Mapping[str, float]
    value: float | None
    reference: float | None
    flags: Mapping[str, Any]
    status: str
    reasons: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.status in IN_STATUSES

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "point": dict(self.point),
            "value": self.value,
            "reference": self.reference,
            "flags": dict(self.flags),
            "status": self.status,
            "reasons": list(self.reasons),
        }


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out


def evaluate_point(spec: CapabilitySpec, point: Mapping[str, float]) -> Evaluation:
    """Evaluate the registered evaluator and apply criterion (a) and (c)."""
    resolved = {axis.name: float(point[axis.name]) for axis in spec.axes}
    try:
        raw = spec.evaluator.call(resolved)
    except _NUMERIC_EXCEPTIONS as exc:
        return Evaluation(spec.capability_id, resolved, None, None, {},
                          PATHOLOGY, (f"evaluator raised {type(exc).__name__}: {exc}",))
    except Exception as exc:  # pragma: no cover - wiring surprise, fail-closed
        return Evaluation(spec.capability_id, resolved, None, None, {},
                          ERROR, (f"evaluator failed with {type(exc).__name__}: {exc}",))

    if isinstance(raw, Mapping):
        value = _as_float(raw.get(spec.evaluator.value_key))
        reference = _as_float(raw.get(spec.evaluator.reference_key))
        flags = dict(raw.get(spec.evaluator.flags_key) or {})
    else:
        value = _as_float(raw)
        reference = None
        flags = {}

    bad, detail = spec.policy.pathology(value, flags)
    if bad:
        return Evaluation(spec.capability_id, resolved, value, reference, flags, PATHOLOGY, (detail,))

    ok, margin, detail = spec.policy.criterion.check(value, reference)
    if not ok:
        return Evaluation(spec.capability_id, resolved, value, reference, flags, CRITERION, (detail,))
    sign_problem = spec.policy.sign_violation(value)
    if sign_problem is not None:
        return Evaluation(spec.capability_id, resolved, value, reference, flags, LAW, (sign_problem,))
    return Evaluation(spec.capability_id, resolved, value, reference, flags, PASS,
                      (f"in domain ({detail}, margin={margin:.3f})",))


def _at_floor(axis: AxisSpec, *values: float | None) -> bool:
    if axis.floor is None:
        return False
    return all(v is not None and math.isfinite(v) and abs(v) <= axis.floor for v in values)


def law_violation(axis: AxisSpec, previous: float | None, current: float | None) -> str | None:
    """Criterion (b): monotonicity law of the measured scalar along the axis.

    The *sign* law is a point property of the measured scalar and is therefore
    enforced in :func:`evaluate_point` (``FailurePolicy.sign``); monotonicity is a
    sequence property and lives here.

    Two values at or below the declared round-off ``floor`` are treated as equal,
    so estimator noise at round-off cannot fabricate a law break.
    """
    if current is None or not math.isfinite(current):
        return None  # already pathological, do not double-report
    if axis.monotone == "none" or previous is None or not math.isfinite(previous):
        return None
    if _at_floor(axis, previous, current):
        return None
    slack = axis.monotone_slack * max(abs(previous), abs(current))
    if axis.monotone == "increasing" and current < previous - slack:
        return (f"monotonicity law: expected non-decreasing along {axis.name}, "
                f"{previous:.6g} -> {current:.6g} (slack {slack:.3e})")
    if axis.monotone == "decreasing" and current > previous + slack:
        return (f"monotonicity law: expected non-increasing along {axis.name}, "
                f"{previous:.6g} -> {current:.6g} (slack {slack:.3e})")
    return None