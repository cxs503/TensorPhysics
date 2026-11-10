"""Degeneracy classification of error criteria (three types, P266W2-N09).

``T1  zero reference``      ``|y_ref| <= 3 sigma`` or ``|y_ref| < 0.1 * scale``: the
                            denominator of a relative criterion is at (or below) the
                            noise floor, so the ratio is dominated by round-off.
``T2  sign flip / zero crossing``
                            ``y`` and ``y_ref`` have opposite signs, or ``y_ref == 0``
                            while ``y != 0``: a relative criterion can then never fall
                            below 1, so it cannot discriminate anymore.
``T3  multi-metric without priority``
                            the same measurand is judged by two or more metric kinds in
                            the same artifact with no declared precedence: which one wins
                            is decided by whoever reads the report ("partial pass").

The classifier is three-valued per type: ``hit`` / ``clear`` / ``undecidable``.  A type
is *undecidable* when the evidence needed to compute it is absent (typically: a report
that recorded only a relative error and dropped the reference magnitude).  Degeneracy is
reported as ``flagged = hit or undecidable`` so that missing evidence can never be
mistaken for a healthy criterion.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .model import (
    ABSOLUTE,
    DEFAULT_NOISE_REL,
    LOG_RATIO,
    RELATIVE,
    SIGNED,
    CriterionRecord,
    finite,
    sign,
)

T1 = "T1_zero_reference"
T2 = "T2_sign_flip"
T3 = "T3_multi_metric_no_priority"
DEGENERACY_TYPES = (T1, T2, T3)

#: scale fraction below which a reference is considered "engineering-wise negligible"
T1_SCALE_FRACTION = 0.1
#: reference/noise multiple below which the reference is considered "at the noise floor"
T1_NOISE_MULTIPLE = 3.0

_HIT = "hit"
_CLEAR = "clear"
_NA = "not_applicable"
_UNDECIDABLE = "undecidable"


@dataclass(frozen=True)
class Degeneracy:
    """Result of the three-type classification of one criterion record."""

    classes: tuple[str, ...] = ()
    undecidable: tuple[str, ...] = ()
    details: Mapping[str, str] = None  # type: ignore[assignment]
    scale: float | None = None
    noise: float | None = None
    family: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "details", dict(self.details or {}))

    @property
    def flagged(self) -> bool:
        return bool(self.classes) or bool(self.undecidable)

    @property
    def provable(self) -> bool:
        return bool(self.classes)

    def has(self, kind: str) -> bool:
        return kind in self.classes

    def as_dict(self) -> dict[str, Any]:
        return {
            "classes": list(self.classes),
            "undecidable": list(self.undecidable),
            "details": dict(self.details),
            "flagged": self.flagged,
            "provable": self.provable,
            "scale": self.scale,
            "noise": self.noise,
            "family": self.family,
        }


def resolve_scale(
    record: CriterionRecord,
    family_scale: float | None = None,
    noise_rel: float = DEFAULT_NOISE_REL,
) -> tuple[float | None, float | None, str]:
    """Resolve ``(scale, noise_floor, provenance)`` for one record.

    Priority: artifact-declared ``scale`` -> family scale (median reference magnitude of
    the same measurand family) -> local magnitude ``max(|y|, |y_ref|)``.  The noise floor
    is the artifact-declared ``noise`` or ``noise_rel * scale``.
    """
    s_ext = finite(record.scale) or finite(family_scale)
    y = finite(record.y)
    r = finite(record.y_ref)
    s_loc = max(abs(v) for v in (y, r) if v is not None) if (y is not None or r is not None) else None
    scale = s_ext if s_ext and s_ext > 0 else (s_loc if s_loc and s_loc > 0 else None)
    prov = "declared" if finite(record.scale) else ("family" if finite(family_scale) else ("local" if scale else "none"))
    noise = finite(record.noise)
    if noise is None and scale is not None:
        noise = noise_rel * scale
    return scale, noise, prov


def _t1(record: CriterionRecord, scale: float | None, noise: float | None, external: bool) -> tuple[str, str]:
    parsed = record.parsed
    if parsed.metric not in (RELATIVE, LOG_RATIO):
        return _NA, "metric is not denominator-normalised"
    r = finite(record.y_ref)
    if r is None:
        return _UNDECIDABLE, "reference magnitude not recorded; |y_ref| cannot be bounded"
    if r == 0.0:
        return _HIT, "y_ref == 0 exactly -> relative error unbounded"
    if noise is not None and abs(r) <= T1_NOISE_MULTIPLE * noise:
        return _HIT, f"|y_ref|={abs(r):.6g} <= 3*noise={T1_NOISE_MULTIPLE * noise:.6g}"
    if external and scale is not None and abs(r) < T1_SCALE_FRACTION * scale:
        return _HIT, f"|y_ref|={abs(r):.6g} < 0.1*scale={T1_SCALE_FRACTION * scale:.6g}"
    return _CLEAR, f"|y_ref|={abs(r):.6g} above noise/scale floor"


def _t2(record: CriterionRecord) -> tuple[str, str]:
    # a zero crossing only destroys discriminability for denominator-normalised or
    # explicitly sign-aware metrics; an absolute band handles y_ref == 0 perfectly well
    if record.parsed.metric not in (RELATIVE, LOG_RATIO, SIGNED):
        return _NA, "metric is sign-blind and denominator-free"
    y = finite(record.y)
    r = finite(record.y_ref)
    if y is None or r is None:
        if record.parsed.metric == SIGNED or "sign" in (record.criterion or "").lower():
            return _UNDECIDABLE, "sign check declared but y/y_ref missing"
        return _NA, "no numeric pair"
    sy, sr = sign(y), sign(r)
    if sr == 0 and sy != 0:
        return _HIT, "y_ref == 0 with y != 0 -> zero denominator (criterion crosses zero)"
    if sy is not None and sr is not None and sy * sr < 0:
        return _HIT, f"sign(y)={sy:+d} vs sign(y_ref)={sr:+d}"
    return _CLEAR, "signs consistent"


def classify(
    record: CriterionRecord,
    *,
    family_scale: float | None = None,
    multi_metric_families: Sequence[str] | Iterable[str] = (),
    noise_rel: float = DEFAULT_NOISE_REL,
) -> Degeneracy:
    """Classify one record into the three degeneracy types."""
    scale, noise, prov = resolve_scale(record, family_scale, noise_rel)
    external = prov in ("declared", "family")
    hits: list[str] = []
    undec: list[str] = []
    details: dict[str, str] = {}

    for kind, (state, why) in ((T1, _t1(record, scale, noise, external)), (T2, _t2(record))):
        if state == _HIT:
            hits.append(kind)
        elif state == _UNDECIDABLE:
            undec.append(kind)
        details[kind] = f"{state}: {why}"

    fams = set(multi_metric_families)
    if record.family_key in fams:
        hits.append(T3)
        details[T3] = f"hit: family '{record.family_key}' carries several metric kinds with no precedence"
    else:
        details[T3] = "clear: no competing metric kind in family"

    details["scale_provenance"] = prov
    return Degeneracy(
        classes=tuple(c for c in DEGENERACY_TYPES if c in hits),
        undecidable=tuple(c for c in DEGENERACY_TYPES if c in undec),
        details=details,
        scale=scale,
        noise=noise,
        family=record.family_key,
    )


def multi_metric_families(
    records: Iterable[CriterionRecord],
    priority_keys: Sequence[str] = ("priority", "precedence", "metric_priority", "primary_metric"),
) -> set[str]:
    """Families where two or more metric kinds compete without a declared priority."""
    seen: dict[str, set[str]] = {}
    has_priority: set[str] = set()
    for rec in records:
        fam = rec.family_key
        seen.setdefault(fam, set()).add(rec.parsed.metric)
        if any(k in rec.evidence for k in priority_keys):
            has_priority.add(fam)
    return {fam for fam, kinds in seen.items() if len({k for k in kinds if k != "unparsed"}) >= 2 and fam not in has_priority}


def analyze(
    records: Iterable[CriterionRecord],
    *,
    noise_rel: float = DEFAULT_NOISE_REL,
) -> list[tuple[CriterionRecord, Degeneracy]]:
    """Classify a whole corpus, resolving family scales internally."""
    records = list(records)
    fams = multi_metric_families(records)
    scales: dict[str, float] = {}
    for rec in records:
        r = finite(rec.y_ref)
        if r is not None and r != 0.0:
            scales.setdefault(rec.family_key, []).append(abs(r))  # type: ignore[arg-type]
    medians = {fam: _median(vals) for fam, vals in scales.items()}  # type: ignore[arg-type]
    return [(rec, classify(rec, family_scale=medians.get(rec.family_key), multi_metric_families=fams, noise_rel=noise_rel)) for rec in records]


def _median(values: Sequence[float]) -> float:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return 0.0
    n = len(vals)
    mid = n // 2
    return vals[mid] if n % 2 else 0.5 * (vals[mid - 1] + vals[mid])