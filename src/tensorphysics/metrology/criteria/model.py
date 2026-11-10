"""Data model for metrological error criteria.

An *error criterion* is promoted here from a free-form string to a structure with
metric semantics::

    (y, y_ref, tau, metric + unit + condition)

where ``y`` is the measured value, ``y_ref`` the reference (benchmark / analytic /
baseline) value, ``tau`` the tolerance and ``metric`` one of

``RELATIVE``    ``|y - y_ref| / |y_ref|``  (dimensionless, degenerate at ``y_ref -> 0``)
``ABSOLUTE``    ``|y - y_ref|``            (engineering unit)
``LOG_RATIO``   ``|ln(y / y_ref)|``        (dimensionless, symmetric, needs same sign)
``SIGNED``      ``y - y_ref``              (keeps the sign of the deviation)

The module is domain neutral: units are opaque strings, the engineering *scale* and
the *noise floor* are injected by the caller (see :mod:`tensorphysics.metrology.criteria.degeneracy`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

# --------------------------------------------------------------------------------------
# verdict vocabulary -- deliberately closed: "PARTIAL" is not representable
# --------------------------------------------------------------------------------------
PASS = "PASS"
FAIL = "FAIL"
WITHHELD = "WITHHELD"
NOT_APPLICABLE = "NOT_APPLICABLE"
VERDICTS = (PASS, FAIL, WITHHELD, NOT_APPLICABLE)

# --------------------------------------------------------------------------------------
# metric vocabulary
# --------------------------------------------------------------------------------------
RELATIVE = "relative"
ABSOLUTE = "absolute"
LOG_RATIO = "log_ratio"
SIGNED = "signed"
METRICS = (RELATIVE, ABSOLUTE, LOG_RATIO, SIGNED)

#: default relative half-width of the engineering band used as noise proxy when the
#: artifact does not record one.  Rationale: engineering reports are written with six
#: significant digits, so 1e-6 * scale is the smallest difference that carries meaning.
DEFAULT_NOISE_REL = 1e-6


_ID_RX = __import__("re").compile(r"^\s*(?:[A-Za-z]{0,3}[-_ ]?\d+[a-z]?|\d+|[a-fA-F])\s*$")


def _is_identifier(text: str | None) -> bool:
    """True for table-row ids (``C1``, ``a``, ``锚2``) that do not name a quantity."""
    t = (text or "").strip()
    if len(t) < 6:
        return True
    return bool(_ID_RX.match(t))


def finite(value: Any) -> float | None:
    """Return ``value`` as a finite float or ``None``.

    ``None``, non-numeric objects, ``NaN`` and ``+-inf`` all collapse to ``None`` so that
    every downstream comparison is total and no verdict can be produced from a
    non-finite number by accident.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


def sign(value: float | None) -> int | None:
    """Return -1, 0 or +1 for a finite value, ``None`` otherwise."""
    f = finite(value)
    if f is None:
        return None
    if f > 0.0:
        return 1
    if f < 0.0:
        return -1
    return 0


@dataclass(frozen=True)
class ParsedCriterion:
    """The metric-semantic content of a criterion string."""

    raw: str
    metric: str
    tolerance: float | None
    tolerance_kind: str  # "fraction" | "absolute"
    unit: str | None = None
    condition: str | None = None
    pattern: str = ""

    @property
    def has_tolerance(self) -> bool:
        return self.tolerance is not None and self.tolerance > 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "metric": self.metric,
            "tolerance": self.tolerance,
            "tolerance_kind": self.tolerance_kind,
            "unit": self.unit,
            "condition": self.condition,
            "pattern": self.pattern,
        }


UNPARSED = ParsedCriterion(
    raw="",
    metric="unparsed",
    tolerance=None,
    tolerance_kind="unknown",
    pattern="none",
)


@dataclass(frozen=True)
class CriterionRecord:
    """One historical criterion instance harvested from an artifact.

    ``y`` / ``y_ref`` may be unknown (``None``) when the artifact only recorded a
    relative error; that absence is itself metrologically meaningful and is reported as
    ``WITHHELD`` instead of silently passing.
    """

    rid: str
    source: str
    measurand: str
    criterion: str
    parsed: ParsedCriterion = UNPARSED
    y: float | None = None
    y_ref: float | None = None
    reported_error: float | None = None  # fraction (0.03 == 3 %)
    unit: str | None = None
    scale: float | None = None
    noise: float | None = None
    family: str | None = None
    old_verdict: str | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)

    # -- derived -----------------------------------------------------------------------
    @property
    def family_key(self) -> str:
        if self.family:
            return self.family
        if _is_identifier(self.measurand):
            # "C1"/"a"/"锚2" are table row ids, not measurands: two rows of the same table
            # must never be treated as competing criteria for one quantity.
            return f"{self.source}::{self.rid}"
        return f"{self.source}::{self.measurand}"

    def has_pair(self) -> bool:
        return finite(self.y) is not None and finite(self.y_ref) is not None

    def derived_relative_error(self) -> float | None:
        """``|y - y_ref| / |y_ref|`` with explicit divergence semantics.

        * ``y_ref != 0`` -> finite value;
        * ``y_ref == 0`` and ``y == 0`` -> ``0.0`` (legitimate exact zero);
        * ``y_ref == 0`` and ``y != 0`` -> ``+inf`` (the degenerate case: the metric is
          unbounded, which is exactly what criterion (e) of P266W2-N09 forbids).
        """
        y = finite(self.y)
        r = finite(self.y_ref)
        if y is None or r is None:
            return None
        if r == 0.0:
            return 0.0 if y == 0.0 else math.inf
        return abs(y - r) / abs(r)

    def with_fields(self, **kwargs: Any) -> "CriterionRecord":
        return replace(self, **kwargs)

    def as_dict(self) -> dict[str, Any]:
        return {
            "rid": self.rid,
            "source": self.source,
            "measurand": self.measurand,
            "criterion": self.criterion,
            "parsed": self.parsed.as_dict(),
            "y": self.y,
            "y_ref": self.y_ref,
            "reported_error": self.reported_error,
            "unit": self.unit,
            "scale": self.scale,
            "noise": self.noise,
            "family": self.family_key,
            "old_verdict": self.old_verdict,
            "derived_relative_error": self.derived_relative_error(),
        }


@dataclass(frozen=True)
class MetricEvaluation:
    """One metric value computed for a record, with its own threshold and verdict."""

    metric: str
    value: float | None
    threshold: float | None
    status: str
    formula: str
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "value": self.value,
            "threshold": self.threshold,
            "status": self.status,
            "formula": self.formula,
            "note": self.note,
        }


@dataclass(frozen=True)
class Verdict:
    """Outcome of a criterion evaluation (baseline, adaptive, or scalar check)."""

    status: str
    metric: str
    value: float | None = None
    threshold: float | None = None
    direction: int | None = None
    reasons: tuple[str, ...] = ()
    evaluations: tuple[MetricEvaluation, ...] = ()

    @property
    def decided(self) -> bool:
        return self.status in (PASS, FAIL)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "metric": self.metric,
            "value": self.value,
            "threshold": self.threshold,
            "direction": self.direction,
            "reasons": list(self.reasons),
            "evaluations": [e.as_dict() for e in self.evaluations],
        }