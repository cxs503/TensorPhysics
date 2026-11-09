"""Applicability envelope objects and the fail-closed out-of-domain verdict.

Stage 4 of the machinery: a result carries ``domain_status`` in
``{IN, OUT, UNKNOWN}``.  When the status is ``OUT`` the grading field is forced
to ``OUT_OF_DOMAIN`` and a ``REAL`` write is *refused* -- a number produced
outside the envelope can never be published as a validated capability.

The conventions are deliberately fail-closed:

* a query outside the *searched span* of an axis is ``UNKNOWN`` (the mechanism
  never extrapolates its own scan);
* a query inside a detected interval is re-evaluated at that point: a failing
  point property overrides the interval classification with ``OUT``;
* only ``IN`` leaves a requested grade untouched.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from .refine import Boundary

__all__ = [
    "EnvelopeFormatError",
    "DomainStatus",
    "OUT_OF_DOMAIN",
    "GradeWriteError",
    "Interval",
    "AxisEnvelope",
    "ApplicabilityEnvelope",
    "Verdict",
    "GradeRecord",
    "assign_grade",
    "declaration_compat",
    "DomainCompat",
    "AxisCompat",
    "INSIDE",
    "OUTSIDE",
    "UNSEARCHED",
]

INSIDE = "inside"
OUTSIDE = "outside"
UNSEARCHED = "unsearched"
OUT_OF_DOMAIN = "OUT_OF_DOMAIN"


class EnvelopeFormatError(ValueError):
    """A serialized envelope cannot be reconstructed -- fail closed, never guess."""


class DomainStatus(str, Enum):
    """Whether a point is inside the capability's detected applicability domain."""

    IN = "IN"
    OUT = "OUT"
    UNKNOWN = "UNKNOWN"


class GradeWriteError(RuntimeError):
    """A ``REAL`` grade was requested for a result outside its applicability domain."""


def _number(payload: Mapping[str, Any], key: str, context: str):
    """Read a numeric field *without* changing its type.

    A quantised axis legitimately serializes snapped values as integers (``22``),
    so coercing every number to ``float`` would break the exact round trip that
    the artifact digest is computed over.
    """
    value = _field(payload, key, context, lambda v: v)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EnvelopeFormatError(f"{context}: field {key!r} is not a number: {value!r}")
    return value


def _as_number(value: Any, label: str, context: str):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EnvelopeFormatError(f"{context}: {label} is not a number: {value!r}")
    return value


def _field(payload: Mapping[str, Any], key: str, context: str, cast):
    if not isinstance(payload, Mapping) or key not in payload:
        raise EnvelopeFormatError(f"{context}: missing required field {key!r}")
    try:
        return cast(payload[key])
    except (TypeError, ValueError) as exc:
        raise EnvelopeFormatError(f"{context}: field {key!r} is not a valid {cast.__name__}: {exc}") from None


@dataclass(frozen=True)
class Interval:
    """A contiguous in-domain interval; ``*_open`` flags mean "no failure found"."""

    low: float
    high: float
    low_open: bool = False
    high_open: bool = False

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Interval":
        return cls(low=_number(payload, "low", "Interval"),
                   high=_number(payload, "high", "Interval"),
                   low_open=bool(payload.get("low_open", False)),
                   high_open=bool(payload.get("high_open", False)))

    def to_dict(self) -> dict[str, Any]:
        return {"low": self.low, "high": self.high,
                "low_open": self.low_open, "high_open": self.high_open}


@dataclass(frozen=True)
class AxisEnvelope:
    """Detected applicability domain of one parameter axis."""

    name: str
    span_low: float
    span_high: float
    intervals: tuple[Interval, ...] = ()
    boundaries: tuple[Boundary, ...] = ()
    confidence: float = 1.0
    samples_total: int = 0
    samples_in: int = 0
    unresolved: bool = False
    notes: tuple[str, ...] = ()

    def locate(self, value: float) -> str:
        if not (self.span_low <= value <= self.span_high):
            return UNSEARCHED
        return INSIDE if any(iv.contains(value) for iv in self.intervals) else OUTSIDE

    def hull(self) -> Interval | None:
        if not self.intervals:
            return None
        return Interval(min(iv.low for iv in self.intervals),
                        max(iv.high for iv in self.intervals),
                        any(iv.low_open for iv in self.intervals),
                        any(iv.high_open for iv in self.intervals))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AxisEnvelope":
        context = "AxisEnvelope"
        span = _field(payload, "searched_span", context, list)
        if not isinstance(span, list) or len(span) != 2:
            raise EnvelopeFormatError(f"{context}: 'searched_span' must be [low, high]")
        boundaries = payload.get("boundaries") or []
        intervals = payload.get("intervals") or []
        try:
            return cls(
                name=str(_field(payload, "axis", context, str)),
                span_low=_as_number(span[0], "searched_span[0]", context),
                span_high=_as_number(span[1], "searched_span[1]", context),
                intervals=tuple(Interval.from_dict(item) for item in intervals),
                boundaries=tuple(Boundary.from_dict(item) for item in boundaries),
                confidence=_as_number(payload.get("confidence", 1.0), "confidence", context),
                samples_total=int(payload.get("samples_total", 0)),
                samples_in=int(payload.get("samples_in", 0)),
                unresolved=bool(payload.get("unresolved", False)),
                notes=tuple(str(n) for n in (payload.get("notes") or ())),
            )
        except EnvelopeFormatError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise EnvelopeFormatError(f"{context}: malformed payload: {exc}") from None

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": self.name,
            "searched_span": [self.span_low, self.span_high],
            "intervals": [iv.to_dict() for iv in self.intervals],
            "boundaries": [b.to_dict() for b in self.boundaries],
            "confidence": self.confidence,
            "samples_total": self.samples_total,
            "samples_in": self.samples_in,
            "unresolved": self.unresolved,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class Verdict:
    """The stage-4 answer for one query point."""

    capability_id: str
    point: Mapping[str, float]
    domain_status: DomainStatus
    axis_status: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()
    value: float | None = None
    reference: float | None = None

    @property
    def grade(self) -> str:
        """The grading field of the result: ``OUT_OF_DOMAIN`` when outside."""
        return OUT_OF_DOMAIN if self.domain_status is DomainStatus.OUT else self.domain_status.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "point": dict(self.point),
            "domain_status": self.domain_status.value,
            "grade": self.grade,
            "axis_status": {k: dict(v) for k, v in self.axis_status.items()},
            "reasons": list(self.reasons),
            "value": self.value,
            "reference": self.reference,
        }


@dataclass(frozen=True)
class ApplicabilityEnvelope:
    """The per-axis envelope of one capability plus its query interface."""

    capability_id: str
    axes: Mapping[str, AxisEnvelope]
    nominal: Mapping[str, float] = field(default_factory=dict)
    digest: str = ""
    generated_utc: str = ""
    declared: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    samples_total: int = 0
    samples_in: int = 0

    def classify(self, point: Mapping[str, float], *, spec=None, verify: bool = True) -> Verdict:
        """Return the domain status of ``point`` (fail-closed)."""
        statuses: dict[str, dict[str, Any]] = {}
        reasons: list[str] = []
        overall = DomainStatus.IN
        for name, axis_env in self.axes.items():
            if name not in point:
                raise KeyError(f"point is missing axis {name!r}")
            value = float(point[name])
            where = axis_env.locate(value)
            if where == UNSEARCHED:
                statuses[name] = {"status": DomainStatus.UNKNOWN.value, "value": value,
                                  "reason": "outside searched span",
                                  "span": [axis_env.span_low, axis_env.span_high]}
                reasons.append(f"{name}={value:.6g}: outside searched span "
                               f"[{axis_env.span_low:.6g}, {axis_env.span_high:.6g}]")
                overall = _worse(overall, DomainStatus.UNKNOWN)
            elif where == OUTSIDE:
                statuses[name] = {"status": DomainStatus.OUT.value, "value": value,
                                  "reason": "outside detected in-domain interval",
                                  "intervals": [iv.to_dict() for iv in axis_env.intervals]}
                reasons.append(f"{name}={value:.6g}: outside detected in-domain interval")
                overall = _worse(overall, DomainStatus.OUT)
            elif axis_env.unresolved:
                statuses[name] = {"status": DomainStatus.UNKNOWN.value, "value": value,
                                  "reason": "no in-domain sample could be evaluated"}
                reasons.append(f"{name}={value:.6g}: no in-domain sample could be evaluated")
                overall = _worse(overall, DomainStatus.UNKNOWN)
            else:
                statuses[name] = {"status": DomainStatus.IN.value, "value": value,
                                  "reason": "inside detected in-domain interval"}
        value: float | None = None
        reference: float | None = None
        if verify and spec is not None:
            from .evaluation import evaluate_point

            evaluation = evaluate_point(spec, point)
            value, reference = evaluation.value, evaluation.reference
            if not evaluation.ok:
                statuses["evaluation"] = {"status": DomainStatus.OUT.value,
                                          "reason": evaluation.status,
                                          "detail": list(evaluation.reasons)}
                reasons.extend(evaluation.reasons)
                overall = _worse(overall, DomainStatus.OUT)
        return Verdict(
            capability_id=self.capability_id,
            point={k: float(v) for k, v in point.items()},
            domain_status=overall,
            axis_status=statuses,
            reasons=tuple(reasons),
            value=value,
            reference=reference,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "nominal": dict(self.nominal),
            "digest": self.digest,
            "generated_utc": self.generated_utc,
            "declared": {k: dict(v) for k, v in self.declared.items()},
            "samples_total": self.samples_total,
            "samples_in": self.samples_in,
            "axes": {name: env.to_dict() for name, env in self.axes.items()},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ApplicabilityEnvelope":
        """Rebuild an envelope from :meth:`to_dict` output.

        This is the consumer side of the mechanism: a persisted artifact can be
        reloaded in another process and used to gate grades *without* re-running
        the probe or having the evaluator importable.  Round-tripping is exact,
        which is what lets :mod:`~.artifact` verify the suite digest after load.
        """
        context = "ApplicabilityEnvelope"
        raw_axes = _field(payload, "axes", context, dict)
        if not isinstance(raw_axes, Mapping):
            raise EnvelopeFormatError(f"{context}: 'axes' must be a mapping")
        axes: dict[str, AxisEnvelope] = {}
        for key, item in raw_axes.items():
            axis_envelope = AxisEnvelope.from_dict(item)
            if axis_envelope.name != key:
                raise EnvelopeFormatError(
                    f"{context}: axis key {key!r} does not match serialized name "
                    f"{axis_envelope.name!r}")
            axes[key] = axis_envelope
        declared = payload.get("declared") or {}
        if not isinstance(declared, Mapping):
            raise EnvelopeFormatError(f"{context}: 'declared' must be a mapping")
        nominal = payload.get("nominal") or {}
        if not isinstance(nominal, Mapping):
            raise EnvelopeFormatError(f"{context}: 'nominal' must be a mapping")
        return cls(
            capability_id=str(_field(payload, "capability_id", context, str)),
            axes=axes,
            nominal=dict(nominal),
            digest=str(payload.get("digest", "")),
            generated_utc=str(payload.get("generated_utc", "")),
            declared={str(k): dict(v) for k, v in declared.items()},
            samples_total=int(payload.get("samples_total", 0)),
            samples_in=int(payload.get("samples_in", 0)),
        )


_SEVERITY = {DomainStatus.IN: 0, DomainStatus.UNKNOWN: 1, DomainStatus.OUT: 2}


def _worse(a: DomainStatus, b: DomainStatus) -> DomainStatus:
    return a if _SEVERITY[a] >= _SEVERITY[b] else b


@dataclass(frozen=True)
class GradeRecord:
    """Result of passing a verdict through the grading guard."""

    capability_id: str
    requested: str
    effective: str
    domain_status: str
    accepted: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "requested": self.requested,
            "effective": self.effective,
            "domain_status": self.domain_status,
            "accepted": self.accepted,
            "reason": self.reason,
        }


def assign_grade(verdict: Verdict, requested: str = "REAL", *, strict: bool = True) -> GradeRecord:
    """Apply the out-of-domain grading rule.

    ``OUT``   -> effective grade ``OUT_OF_DOMAIN``; a ``REAL`` request raises
                 :class:`GradeWriteError` when ``strict`` (default) and is
                 otherwise recorded as refused (``accepted=False``).
    ``UNKNOWN`` -> ``REAL`` is downgraded to ``WITHHELD`` (silence never passes).
    ``IN``    -> the requested grade is passed through unchanged.
    """
    requested = str(requested).upper()
    if verdict.domain_status is DomainStatus.OUT:
        reason = "out of applicability domain: " + "; ".join(verdict.reasons[:3])
        if strict and requested == "REAL":
            raise GradeWriteError(
                f"{verdict.capability_id}: refusing REAL for an out-of-domain result ({reason})")
        return GradeRecord(verdict.capability_id, requested, OUT_OF_DOMAIN,
                           verdict.domain_status.value, False, reason)
    if verdict.domain_status is DomainStatus.UNKNOWN:
        effective = "WITHHELD" if requested == "REAL" else requested
        return GradeRecord(verdict.capability_id, requested, effective,
                           verdict.domain_status.value, effective == requested,
                           "domain status UNKNOWN: REAL downgraded to WITHHELD")
    return GradeRecord(verdict.capability_id, requested, requested,
                       verdict.domain_status.value, True, "inside applicability domain")


# ---------------------------------------------------------------------------
# mechanism boundary vs. human declaration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AxisCompat:
    axis: str
    mechanism_low: float | None
    mechanism_high: float | None
    declared_low: float | None
    declared_high: float | None
    slack: float
    verdict: str
    wider_low_by: float
    wider_high_by: float
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "mechanism_low": self.mechanism_low,
            "mechanism_high": self.mechanism_high,
            "declared_low": self.declared_low,
            "declared_high": self.declared_high,
            "slack": self.slack,
            "verdict": self.verdict,
            "wider_low_by": self.wider_low_by,
            "wider_high_by": self.wider_high_by,
            "source": self.source,
        }


@dataclass(frozen=True)
class DomainCompat:
    """Comparison of the detected envelope against the human declaration."""

    capability_id: str
    verdict: str
    axes: tuple[AxisCompat, ...] = ()

    @property
    def ok(self) -> bool:
        return self.verdict in ("NOT_WIDER", "CONSERVATIVE")

    def to_dict(self) -> dict[str, Any]:
        return {"capability_id": self.capability_id, "verdict": self.verdict,
                "ok": self.ok, "axes": [a.to_dict() for a in self.axes]}


def declaration_compat(envelope: ApplicabilityEnvelope, spec) -> DomainCompat:
    """Is the mechanism boundary no wider than the human declaration?

    ``NOT_WIDER``  detected in-domain hull inside the declared interval (or equal
                   within the declared ``slack``) -- the mechanism does not
                   certify anything the documentation does not claim.
    ``CONSERVATIVE`` detected hull strictly inside with a margin (documented
                   claim was conservative; informational, still acceptable).
    ``WIDER_THAN_DECLARED`` the mechanism certifies a region outside the declared
                   interval -- overreach, must be tightened.
    ``NO_DECLARATION`` no human statement for this capability.
    """
    if not spec.declared:
        return DomainCompat(spec.capability_id, "NO_DECLARATION")
    reports: list[AxisCompat] = []
    overall = "NOT_WIDER"
    for name, declared in sorted(spec.declared.items()):
        axis_env = envelope.axes[name]
        hull = axis_env.hull()
        span = axis_env.span_high - axis_env.span_low
        # ``declared.slack`` is a fraction of the searched span (comparison tolerance
        # for estimator noise; it never justifies certifying more than the declaration)
        slack = declared.slack * span
        if hull is None:
            reports.append(AxisCompat(name, None, None, declared.low, declared.high, slack,
                                      "NO_DOMAIN_DETECTED", 0.0, 0.0, declared.source))
            overall = "NO_DOMAIN_DETECTED"
            continue
        wider_low = 0.0
        if declared.low is not None and hull.low < declared.low:
            wider_low = declared.low - hull.low
            if wider_low <= slack:
                wider_low = 0.0
        wider_high = 0.0
        if declared.high is not None and hull.high > declared.high:
            wider_high = hull.high - declared.high
            if wider_high <= slack:
                wider_high = 0.0
        if wider_low > 0.0 or wider_high > 0.0:
            verdict = "WIDER_THAN_DECLARED"
            overall = "WIDER_THAN_DECLARED"
        elif overall == "NOT_WIDER":
            inside_margin = (
                (hull.low > declared.low if declared.low is not None else False)
                or (hull.high < declared.high if declared.high is not None else False)
            )
            verdict = "CONSERVATIVE" if inside_margin else "NOT_WIDER"
            if verdict == "CONSERVATIVE" and overall == "NOT_WIDER":
                overall = "CONSERVATIVE"
        else:
            verdict = "NOT_WIDER"
        reports.append(AxisCompat(name, hull.low, hull.high, declared.low, declared.high,
                                  slack, verdict, wider_low, wider_high, declared.source))
    return DomainCompat(spec.capability_id, overall, tuple(reports))