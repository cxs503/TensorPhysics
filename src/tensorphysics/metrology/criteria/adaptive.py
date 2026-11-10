"""Adaptive criterion family: metric-semantic replacements for ``relative error < tau``.

Four members, all bounded by construction:

``scaled_delta``  ``|y - y_ref| / max(|y_ref|, 0.1*scale)``  -- order-of-magnitude
                  normalisation: the denominator can no longer collapse when the
                  reference tends to zero.
``log_ratio``     ``|ln(y / y_ref)| <= ln(1 + tau)``  -- symmetric band, so a 2x too large
                  and a 2x too small result are judged equally (the plain relative error
                  is asymmetric: it saturates at 100 % for under-estimates).
``signed_abs``    ``|y - y_ref| <= tau * scale``  -- engineering-unit threshold, keeps the
                  sign so a sign flip can never pass.
``merge``         precedence ``absolute -> log_ratio -> relative``; a FAIL of any
                  mandatory member is decisive ("no double standard"), and the outcome
                  vocabulary is closed to ``PASS / FAIL / WITHHELD`` -- ``PARTIAL`` is
                  not representable.

Guarantees (tested):

* no division by a zero or near-zero denominator -> ``None`` instead of ``inf``/``NaN``;
* ``y_ref -> 0`` yields ``WITHHELD`` or a finite scaled value, never an unbounded ratio;
* ``None``/``NaN`` inputs propagate to ``WITHHELD``, never to ``PASS``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

from .degeneracy import Degeneracy, T1, T2, T3, classify, resolve_scale
from .model import (
    ABSOLUTE,
    DEFAULT_NOISE_REL,
    FAIL,
    LOG_RATIO,
    PASS,
    RELATIVE,
    SIGNED,
    WITHHELD,
    CriterionRecord,
    MetricEvaluation,
    Verdict,
    finite,
    sign,
)

__all__ = [
    "scaled_delta",
    "log_ratio",
    "signed_abs",
    "bounded_relative_error",
    "judge",
    "merge_verdicts",
    "deviation_fraction",
    "direction",
    "LOW_SCALE_FRACTION",
]

#: fraction of the engineering scale used as the floor of the normalisation denominator
LOW_SCALE_FRACTION = 0.1


def _abs_or_none(x: float | None) -> float | None:
    return None if x is None else abs(x)


def scaled_delta(
    y: Any,
    y_ref: Any,
    *,
    scale: float | None = None,
    noise: float | None = None,
    noise_rel: float = DEFAULT_NOISE_REL,
) -> float | None:
    """Magnitude-normalised deviation ``|y - y_ref| / max(|y_ref|, 0.1*scale)``.

    Returns ``None`` when no meaningful denominator exists (both ``y_ref`` and the
    engineering scale are zero/unknown), which downstream becomes ``WITHHELD``.
    >>> scaled_delta(1.1, 1.0)
    0.1
    >>> scaled_delta(1.0, 0.0, scale=1.0)
    10.0
    >>> scaled_delta(1.0, 0.0) is None
    True
    """
    yy, rr = finite(y), finite(y_ref)
    if yy is None or rr is None:
        return None
    s = finite(scale)
    denom = abs(rr)
    if s is not None and s > 0:
        denom = max(denom, LOW_SCALE_FRACTION * s)
    if noise is not None and noise > 0:
        denom = max(denom, 3.0 * noise)
    if denom <= 0.0:
        return None
    value = abs(yy - rr) / denom
    return value if math.isfinite(value) else None


def log_ratio(y: Any, y_ref: Any, *, noise: float | None = None) -> float | None:
    """``|ln(y / y_ref)|`` -- only defined for same-sign non-zero values.

    >>> round(log_ratio(2.0, 1.0), 12)
    0.69314718056
    >>> log_ratio(1.0, -1.0) is None
    True
    >>> log_ratio(0.0, 1.0) is None
    True
    """
    yy, rr = finite(y), finite(y_ref)
    if yy is None or rr is None or yy == 0.0 or rr == 0.0:
        return None
    if (yy > 0) != (rr > 0):
        return None
    if noise is not None and noise > 0 and (abs(yy) <= noise or abs(rr) <= noise):
        return None
    value = abs(math.log(abs(yy) / abs(rr)))
    return value if math.isfinite(value) else None


def signed_abs(y: Any, y_ref: Any) -> float | None:
    """``y - y_ref`` (sign preserving); ``None`` when either input is non-finite.

    >>> signed_abs(0.99, 1.0)
    -0.010000000000000009
    """
    yy, rr = finite(y), finite(y_ref)
    if yy is None or rr is None:
        return None
    value = yy - rr
    return value if math.isfinite(value) else None


def bounded_relative_error(
    y: Any,
    y_ref: Any,
    *,
    scale: float | None = None,
    noise_rel: float = DEFAULT_NOISE_REL,
) -> tuple[float | None, str]:
    """Relative error that is *bounded by construction*.

    Returns ``(value, status)`` where status is ``"defined"`` or ``"withheld"``.  The
    function never returns ``inf`` or ``NaN``: when ``|y_ref|`` sits at the noise floor
    the ratio is not a measurement and the caller must treat the criterion as
    ``WITHHELD``.

    >>> bounded_relative_error(0.0, 0.0)
    (0.0, 'defined')
    >>> bounded_relative_error(1.0, 0.0)
    (None, 'withheld')
    """
    yy, rr = finite(y), finite(y_ref)
    if yy is None or rr is None:
        return None, "withheld"
    s = finite(scale)
    magnitude = max(abs(s) if s is not None else 0.0, abs(yy), abs(rr))
    floor = 3.0 * noise_rel * magnitude if magnitude > 0.0 else 0.0
    if abs(rr) <= floor:
        if abs(yy) <= floor:
            return 0.0, "defined"  # both at the noise floor: an exact zero is legitimate
        return None, "withheld"  # reference at (or below) the noise floor: ratio meaningless
    value = abs(yy - rr) / abs(rr)
    return (value, "defined") if math.isfinite(value) else (None, "withheld")


def deviation_fraction(
    y: Any,
    y_ref: Any,
    *,
    scale: float | None = None,
) -> float | None:
    """``|y - y_ref| / scale`` (0..inf) -- the physical size of the deviation."""
    yy, rr = finite(y), finite(y_ref)
    if yy is None or rr is None:
        return None
    s = finite(scale)
    if s is None or s <= 0:
        s = abs(rr) if rr != 0.0 else abs(yy)
    if s <= 0:
        return None
    return abs(yy - rr) / s


def direction(y: Any, y_ref: Any) -> int | None:
    """Direction of the deviation: ``+1`` measured high, ``-1`` measured low, ``0`` exact."""
    d = signed_abs(y, y_ref)
    return sign(d) if d is not None else None


def _threshold_abs(record: CriterionRecord, scale: float | None) -> float | None:
    parsed = record.parsed
    tol = finite(parsed.tolerance)
    if tol is None or tol <= 0:
        return None
    if parsed.tolerance_kind == "absolute" or parsed.metric == ABSOLUTE:
        return tol
    if scale is None or scale <= 0:
        return None
    return tol * scale


def judge(
    record: CriterionRecord,
    degeneracy: Degeneracy | None = None,
    *,
    family_scale: float | None = None,
    multi_metric_families: tuple[str, ...] = (),
    noise_rel: float = DEFAULT_NOISE_REL,
) -> Verdict:
    """Judge one record with the adaptive criterion family.

    Decision procedure (documented, deterministic):

    1. non-finite / missing numbers -> ``WITHHELD``;
    2. ``T2`` sign flip -> ``FAIL`` (or ``WITHHELD`` if the flip is inside the noise
       floor);
    3. ``T1`` zero reference -> mandatory ``signed_abs`` with ``tau * scale``; the
       degenerate relative metric is reported as advisory only;
    4. otherwise -> precedence ``absolute -> log_ratio -> relative``, and any mandatory
       member that FAILs makes the record FAIL ("no double standard").

    >>> from .parser import extract_quadruple
    >>> rec = extract_quadruple("相对误差 < 2%", 1.01, 1.0)
    >>> judge(rec).status
    'PASS'
    >>> rec2 = extract_quadruple("相对误差 < 2%", -1.01, 1.0)
    >>> judge(rec2).status
    'FAIL'
    """
    deg = degeneracy if degeneracy is not None else classify(
        record, family_scale=family_scale, multi_metric_families=multi_metric_families, noise_rel=noise_rel
    )
    scale, noise, _prov = resolve_scale(record, family_scale, noise_rel)
    y, r = finite(record.y), finite(record.y_ref)
    evaluations: list[MetricEvaluation] = []
    reasons: list[str] = []

    tol = finite(record.parsed.tolerance)
    tol_abs = _threshold_abs(record, scale)

    # ---- metric evaluations (always computed, so the alternative numbers are auditable)
    sd = scaled_delta(y, r, scale=scale, noise=noise, noise_rel=noise_rel)
    if sd is not None and tol is not None:
        evaluations.append(MetricEvaluation(RELATIVE, sd, tol, PASS if sd <= tol else FAIL,
                                            "|y-y_ref|/max(|y_ref|,0.1*scale)", "magnitude-normalised"))
    lr = log_ratio(y, r, noise=noise)
    log_thr = math.log1p(tol) if tol is not None else None
    if lr is not None and log_thr is not None:
        evaluations.append(MetricEvaluation(LOG_RATIO, lr, log_thr, PASS if lr <= log_thr else FAIL,
                                            "|ln(y/y_ref)|<=ln(1+tau)", "symmetric band"))
    sa = signed_abs(y, r)
    if sa is not None and tol_abs is not None:
        evaluations.append(MetricEvaluation(SIGNED, abs(sa), tol_abs, PASS if abs(sa) <= tol_abs else FAIL,
                                            "|y-y_ref|<=tau*scale", "engineering-unit threshold"))

    # ---- evidence availability
    if y is None and r is None and record.reported_error is None:
        return Verdict(WITHHELD, record.parsed.metric or "unparsed", None, tol, None,
                       ("y and y_ref both missing",), tuple(evaluations))
    if tol is None:
        return Verdict(WITHHELD, record.parsed.metric or "unparsed", None, None, None,
                       ("criterion carries no parseable tolerance",), tuple(evaluations))
    if y is not None and r is None and record.parsed.tolerance_kind == "absolute":
        # "max|A-B| <= 1e-12" with the deviation itself archived: no reference needed
        return Verdict(PASS if abs(y) <= tol else FAIL, SIGNED, abs(y), tol, sign(y),
                       ("absolute criterion with a recorded deviation and no reference (decidable)",),
                       tuple(evaluations))
    if y is None or r is None:
        eps = finite(record.reported_error)
        if eps is None or eps < 0.0:
            return Verdict(WITHHELD, record.parsed.metric, None, tol, None,
                           ("no recorded outcome: neither the pair nor a relative error was archived",),
                           tuple(evaluations))
        if tol is None:
            return Verdict(WITHHELD, record.parsed.metric, eps, None, None, ("no parseable tolerance",), tuple(evaluations))
        if record.parsed.tolerance_kind == "absolute" or record.parsed.metric == ABSOLUTE:
            # an absolute band needs no reference magnitude: the recorded deviation decides
            return Verdict(PASS if abs(eps) <= tol else FAIL, SIGNED, abs(eps), tol, sign(eps),
                           ("absolute criterion with a recorded absolute deviation",), tuple(evaluations))
        # A recorded relative error alone still determines the *symmetric* band up to the
        # sign of the deviation: |ln(1 + eps)| for an over-estimate, |ln(1 - eps)| for an
        # under-estimate.  We take the worse of the two (conservative, direction-ambiguous)
        # so the replacement criterion is decidable without inventing a reference value.
        log_bound = abs(math.log1p(eps)) if eps < 1.0 else math.inf
        if eps < 1.0:
            log_bound = max(log_bound, abs(math.log1p(-eps)))
        log_thr2 = math.log1p(tol) if record.parsed.tolerance_kind == "fraction" else None
        if log_thr2 is None or not math.isfinite(log_bound):
            return Verdict(WITHHELD, LOG_RATIO, None, log_thr2, None,
                           ("recorded relative error cannot be mapped onto a bounded metric",),
                           tuple(evaluations))
        status = PASS if log_bound <= log_thr2 else FAIL
        return Verdict(status, LOG_RATIO, log_bound, log_thr2, None,
                       ("relative error recorded without its reference: judged with the worst-case symmetric log band",),
                       tuple(evaluations))

    dirn = direction(y, r)

    # ---- 2. sign flip
    if T2 in deg.classes:
        gap = abs(y - r)
        if noise is not None and gap <= 3.0 * noise:
            return Verdict(WITHHELD, SIGNED, gap, 3.0 * noise, dirn,
                           ("sign flip inside the noise floor: sign is not resolvable",), tuple(evaluations))
        thr = tol_abs if tol_abs is not None else tol
        if tol_abs is not None and gap <= tol_abs:
            # sign is inverted but the magnitude is within the engineering band: still a
            # sign error, so the verdict is FAIL (sign is a physical invariant here).
            return Verdict(FAIL, SIGNED, gap, tol_abs, dirn,
                           ("sign flip: relative criterion is structurally unbounded (>1); sign must be preserved",),
                           tuple(evaluations))
        return Verdict(FAIL, SIGNED, gap, thr, dirn,
                       ("sign flip: deviation exceeds the engineering band",), tuple(evaluations))

    # ---- 3. zero reference
    if T1 in deg.classes:
        if tol_abs is None:
            if sd is not None:
                return Verdict(PASS if sd <= tol else FAIL, RELATIVE, sd, tol, dirn,
                               ("zero reference without engineering scale: magnitude-normalised fallback",),
                               tuple(evaluations))
            return Verdict(WITHHELD, SIGNED, abs(y - r), None, dirn,
                           ("zero reference and no engineering scale: bound not computable",), tuple(evaluations))
        gap = abs(y - r)
        return Verdict(PASS if gap <= tol_abs else FAIL, SIGNED, gap, tol_abs, dirn,
                       ("reference at noise floor: relative metric degenerates, signed absolute band applied",),
                       tuple(evaluations))

    # ---- 4. regular case
    mandatory: list[MetricEvaluation] = []
    if tol_abs is not None and record.parsed.metric in (ABSOLUTE, SIGNED):
        mandatory = [e for e in evaluations if e.metric == SIGNED]
    elif log_thr is not None and lr is not None:
        mandatory = [e for e in evaluations if e.metric == LOG_RATIO]
    else:
        mandatory = [e for e in evaluations if e.metric == RELATIVE]
    if not mandatory:
        if tol_abs is not None:
            mandatory = [e for e in evaluations if e.metric == SIGNED]
        elif sd is not None:
            mandatory = [MetricEvaluation(RELATIVE, sd, tol, PASS if sd <= tol else FAIL,
                                          "|y-y_ref|/max(|y_ref|,0.1*scale)", "fallback")]
    if not mandatory:
        return Verdict(WITHHELD, record.parsed.metric, None, tol, dirn,
                       ("no adaptive metric is computable for this record",), tuple(evaluations))
    if any(e.status == FAIL for e in mandatory):
        best = max(mandatory, key=lambda e: (e.threshold is not None, e.value or 0.0))
        return Verdict(FAIL, best.metric, best.value, best.threshold, dirn,
                       ("deviation exceeds the adaptive band (no double standard: a FAIL member is decisive)",),
                       tuple(evaluations))
    best = mandatory[0]
    return Verdict(PASS, best.metric, best.value, best.threshold, dirn, (), tuple(evaluations))


def merge_verdicts(verdicts: list[Verdict], policy: str = "strictest") -> Verdict:
    """Merge several metric verdicts into one, without ever producing ``PARTIAL``.

    ``strictest`` (default): any ``FAIL`` -> ``FAIL``; else any ``PASS`` -> ``PASS``;
    else ``WITHHELD``.  ``absolute_first``: the first decided verdict in the given order
    wins, but a FAIL of any earlier *mandatory* member still dominates because callers
    order absolute before relative.

    >>> merge_verdicts([Verdict(PASS, "relative"), Verdict(FAIL, "absolute")]).status
    'FAIL'
    >>> merge_verdicts([Verdict(WITHHELD, "relative")]).status
    'WITHHELD'
    """
    if not verdicts:
        return Verdict(WITHHELD, "none", None, None, None, ("empty verdict set",))
    if policy == "absolute_first":
        order = {ABSOLUTE: 0, SIGNED: 0, LOG_RATIO: 1, RELATIVE: 2}
        ordered = sorted(verdicts, key=lambda v: order.get(v.metric, 3))
        for v in ordered:
            if v.decided:
                return v
        return ordered[0]
    if any(v.status == FAIL for v in verdicts):
        return next(v for v in verdicts if v.status == FAIL)
    if any(v.status == PASS for v in verdicts):
        return next(v for v in verdicts if v.status == PASS)
    return verdicts[0]

def judge_family(
    records: list[CriterionRecord],
    *,
    degeneracies: dict[str, Degeneracy] | None = None,
    family_scale: float | None = None,
    noise_rel: float = DEFAULT_NOISE_REL,
) -> dict[str, Any]:
    """Judge all criteria of one measurand family and merge them without a double standard.

    This is the T3 remedy: when an artifact states ``relative error < 2 %`` *and*
    ``max|delta| <= 0.5`` for the same measurand and never says which one wins, the
    family verdict is the strictest decided member.  The per-record verdicts are returned
    as well, so the "partial pass" that a human reader would produce is visible.

    >>> from .parser import extract_quadruple
    >>> a = extract_quadruple("相对误差 < 2%", 1.001, 1.0, rid="a")
    >>> b = extract_quadruple("max|Δ| <= 1e-4", 1.001, 1.0, rid="b")
    >>> judge_family([a, b])["merged"].status
    'FAIL'
    """
    degs = degeneracies or {}
    per = []
    for rec in records:
        v = judge(rec, degs.get(rec.rid), family_scale=family_scale, noise_rel=noise_rel)
        per.append({"rid": rec.rid, "criterion": rec.criterion, "metric": rec.parsed.metric,
                    "status": v.status, "value": v.value, "threshold": v.threshold})
    merged = merge_verdicts([judge(rec, degs.get(rec.rid), family_scale=family_scale, noise_rel=noise_rel) for rec in records],
                            policy="strictest")
    return {
        "family": records[0].family_key if records else None,
        "n_members": len(records),
        "members": per,
        "merged": merged,
        "merged_status": merged.status,
        "merged_metric": merged.metric,
        "merged_value": merged.value,
        "partial_pass_risk": len({m["status"] for m in per}) > 1,
    }
