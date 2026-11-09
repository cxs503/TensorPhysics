"""Stage 2 of the envelope machinery: the probing operator.

For one registered capability the operator walks every axis along a *geometric
gradient scan* (stage 2), marks each sample with the three criteria of the
evaluation layer plus the sequence laws, and hands every in/out neighbour pair
to the bisection of stage 3.  The assembled per-axis intervals become an
:class:`~.envelope.ApplicabilityEnvelope` (stage 4).

Nothing here is capability specific: the scan, the laws and the refinement all
read the declaration loaded from configuration.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from .axis import AxisSpec
from .capability import CapabilityRegistry, CapabilitySpec
from .envelope import ApplicabilityEnvelope, AxisEnvelope, Interval, declaration_compat
from .evaluation import ERROR, LAW, PASS, Evaluation, evaluate_point, law_violation
from .refine import DEFAULT_REL_TOL, Boundary, bisect_boundary

__all__ = ["ScanRecord", "ProbeReport", "probe_axis", "probe_capability", "probe_registry",
           "EnvelopeSuite", "CONTENT_VOLATILE_FIELDS", "envelope_content_material"]

#: Envelope fields that carry run metadata rather than content.  They are
#: excluded from the suite digest, otherwise the same envelopes probed twice
#: would hash differently and the artifact identity could never be reproduced
#: (and therefore never be verified by a later consumer).
CONTENT_VOLATILE_FIELDS = ("generated_utc",)


def envelope_content_material(payload: Mapping[str, Any]) -> dict[str, Any]:
    """The digestable content of a serialized envelope (volatile metadata dropped)."""
    return {k: v for k, v in payload.items() if k not in CONTENT_VOLATILE_FIELDS}


@dataclass(frozen=True)
class ScanRecord:
    """One geometric gradient scan of one axis (the other axes at their nominal)."""

    axis: AxisSpec
    values: tuple[float, ...]
    evaluations: tuple[Evaluation, ...]
    law_notes: tuple[str, ...] = ()

    @property
    def statuses(self) -> tuple[str, ...]:
        return tuple(ev.status for ev in self.evaluations)

    @property
    def in_count(self) -> int:
        return sum(1 for s in self.statuses if s == PASS)

    def measured(self) -> tuple[float | None, ...]:
        return tuple(ev.value for ev in self.evaluations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis.name,
            "values": list(self.values),
            "measures": list(self.measured()),
            "statuses": list(self.statuses),
            "law_notes": list(self.law_notes),
        }


@dataclass(frozen=True)
class ProbeReport:
    capability_id: str
    scans: Mapping[str, ScanRecord]
    envelope: ApplicabilityEnvelope
    compat: Any = None

    def to_dict(self) -> dict[str, Any]:
        payload = {"capability_id": self.capability_id,
                   "scans": {k: v.to_dict() for k, v in self.scans.items()},
                   "envelope": self.envelope.to_dict()}
        if self.compat is not None:
            payload["declaration_compat"] = self.compat.to_dict()
        return payload


@dataclass(frozen=True)
class EnvelopeSuite:
    """All envelopes produced by one run, with process provenance."""

    reports: Mapping[str, ProbeReport]
    digest: str          # content digest: reproducible across runs, timestamp excluded
    generated_utc: str
    config_digest: str = ""
    rel_tol: float = DEFAULT_REL_TOL

    def envelopes(self) -> dict[str, ApplicabilityEnvelope]:
        return {k: v.envelope for k, v in self.reports.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "tensorphysics.applicability.envelopes.v1",
            "generated_utc": self.generated_utc,
            "config_digest": self.config_digest,
            "suite_digest": self.digest,
            "rel_tol": self.rel_tol,
            "capabilities": [self.reports[k].to_dict() for k in sorted(self.reports)],
        }


def probe_axis(spec: CapabilitySpec, axis: AxisSpec) -> ScanRecord:
    """Geometric gradient scan of a single axis, laws applied across the sequence."""
    base = spec.nominal_point()
    values = axis.coordinates()
    evaluations: list[Evaluation] = []
    law_notes: list[str] = []
    previous_value: float | None = None
    for i, value in enumerate(values):
        point = dict(base)
        point[axis.name] = float(value)
        evaluation = evaluate_point(spec, point)
        if law_notes_set := law_violation(axis, previous_value, evaluation.value):
            note = f"{axis.name}={value:.6g}: {law_notes_set}"
            law_notes.append(note)
            evaluation = Evaluation(evaluation.capability_id, evaluation.point, evaluation.value,
                                    evaluation.reference, evaluation.flags, LAW,
                                    evaluation.reasons + (note,))
        evaluations.append(evaluation)
        if evaluation.value is not None:
            previous_value = evaluation.value
    return ScanRecord(axis=axis, values=values, evaluations=tuple(evaluations),
                      law_notes=tuple(law_notes))


def _point_predicate(spec: CapabilitySpec, axis: AxisSpec, base: Mapping[str, float]):
    def inside(value: float) -> bool:
        point = dict(base)
        point[axis.name] = float(value)
        return evaluate_point(spec, point).ok

    return inside


def _build_axis_envelope(spec: CapabilitySpec, scan: ScanRecord, *,
                         rel_tol: float, refine: bool) -> AxisEnvelope:
    """Assemble one axis envelope: in-domain runs, refined boundaries, confidence."""
    axis = scan.axis
    statuses = scan.statuses
    base = spec.nominal_point()
    boundaries: dict[int, Boundary] = {}   # key: index i of the pair (i, i+1)
    notes: list[str] = list(scan.law_notes)
    n = len(scan.values)
    span = axis.width

    def _snap(boundary: Boundary) -> Boundary:
        if axis.quantum is None:
            return boundary
        value = axis.snap(boundary.value)
        width = axis.quantum
        return Boundary(
            axis=boundary.axis, side=boundary.side, value=value, width=width,
            span_fraction=width / span if span else 0.0,
            inside_value=boundary.inside_value, outside_value=boundary.outside_value,
            source=boundary.source, iterations=boundary.iterations,
            converged=boundary.converged and width <= rel_tol * span,
            confidence=boundary.confidence if width <= rel_tol * span else 0.5,
        )

    # ---- stage 2: maximal runs of in-domain samples ---------------------
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, status in enumerate(statuses):
        if status == PASS and start is None:
            start = i
        elif status != PASS and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, n - 1))

    # ---- stage 3: refine every in/out neighbour pair --------------------
    for i in range(n - 1):
        a_in, b_in = statuses[i] == PASS, statuses[i + 1] == PASS
        if a_in == b_in:
            continue
        inside_value = scan.values[i] if a_in else scan.values[i + 1]
        outside_value = scan.values[i + 1] if a_in else scan.values[i]
        outside_status = statuses[i + 1] if a_in else statuses[i]
        source = "law" if outside_status in (LAW, ERROR) else outside_status
        if not refine or source in ("law", "error"):
            midpoint = 0.5 * (inside_value + outside_value)
            boundaries[i] = Boundary(
                axis=axis.name,
                side="low" if outside_value < inside_value else "high",
                value=midpoint,
                width=abs(outside_value - inside_value),
                span_fraction=abs(outside_value - inside_value) / span if span else 0.0,
                inside_value=inside_value,
                outside_value=outside_value,
                source=source,
                iterations=0,
                converged=False,
                confidence=0.5,
            )
            if source in ("law", "error"):
                notes.append(f"{axis.name}: {source} transition at sampling resolution "
                             f"({outside_value:.6g} outside vs {inside_value:.6g} inside); "
                             f"not bisectable")
            continue
        boundaries[i] = _snap(bisect_boundary(axis, inside_value, outside_value,
                                              _point_predicate(spec, axis, base),
                                              source=source, rel_tol=rel_tol))

    # ---- in-domain intervals, endpoints taken from the refined edges ----
    intervals: list[Interval] = []
    for lo_i, hi_i in runs:
        low, low_open = axis.low, True
        if lo_i > 0:
            low_open = False
            boundary = boundaries.get(lo_i - 1)
            low = boundary.value if boundary is not None else scan.values[lo_i]
        high, high_open = axis.high, True
        if hi_i < n - 1:
            high_open = False
            boundary = boundaries.get(hi_i)
            high = boundary.value if boundary is not None else scan.values[hi_i]
        intervals.append(Interval(min(low, high), max(low, high), low_open, high_open))

    samples_in = sum(1 for s in statuses if s == PASS)
    unresolved = samples_in == 0 and any(s == ERROR for s in statuses)
    ordered = tuple(boundaries[k] for k in sorted(boundaries))
    if ordered:
        confidence = min(b.confidence for b in ordered)
    else:
        confidence = 1.0 if samples_in else 0.0
    if axis.quantum is not None:
        notes.append(f"{axis.name}: quantised axis (quantum={axis.quantum:g}); "
                     f"boundary resolved to one quantum")
    return AxisEnvelope(
        name=axis.name,
        span_low=axis.low,
        span_high=axis.high,
        intervals=tuple(intervals),
        boundaries=ordered,
        confidence=confidence,
        samples_total=n,
        samples_in=samples_in,
        unresolved=unresolved,
        notes=tuple(notes),
    )


def _digest(material: Any) -> str:
    blob = json.dumps(material, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def probe_capability(spec: CapabilitySpec, *, rel_tol: float = DEFAULT_REL_TOL,
                     refine: bool = True) -> ProbeReport:
    """Probe every axis of one capability and assemble its envelope."""
    scans: dict[str, ScanRecord] = {}
    axis_envelopes: dict[str, AxisEnvelope] = {}
    for axis in spec.axes:
        scan = probe_axis(spec, axis)
        scans[axis.name] = scan
        axis_envelopes[axis.name] = _build_axis_envelope(spec, scan, rel_tol=rel_tol, refine=refine)
    envelope = ApplicabilityEnvelope(
        capability_id=spec.capability_id,
        axes=axis_envelopes,
        nominal=spec.nominal_point(),
        digest=_digest(spec.digest_material()),
        generated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        declared={k: v.to_dict() for k, v in spec.declared.items()},
        samples_total=sum(env.samples_total for env in axis_envelopes.values()),
        samples_in=sum(env.samples_in for env in axis_envelopes.values()),
    )
    compat = declaration_compat(envelope, spec) if spec.declared else None
    return ProbeReport(capability_id=spec.capability_id, scans=scans, envelope=envelope, compat=compat)


def probe_registry(registry: CapabilityRegistry, *, rel_tol: float = DEFAULT_REL_TOL,
                   refine: bool = True, ids: Iterable[str] | None = None) -> EnvelopeSuite:
    """Probe every registered capability (or the subset ``ids``)."""
    selected = list(registry.ids()) if ids is None else list(ids)
    reports: dict[str, ProbeReport] = {}
    for capability_id in selected:
        reports[capability_id] = probe_capability(registry.get(capability_id),
                                                  rel_tol=rel_tol, refine=refine)
    # content only: the digest must be reproducible across runs (see
    # CONTENT_VOLATILE_FIELDS) so a stored artifact can be verified later
    material = {k: envelope_content_material(v.envelope.to_dict())
                for k, v in reports.items()}
    return EnvelopeSuite(
        reports=reports,
        digest=_digest(material),
        generated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        rel_tol=rel_tol,
    )