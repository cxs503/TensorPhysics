"""Adaptive acceptance-criterion family.

For each degeneracy class the plain relative-error criterion is replaced by a metric that
stays *bounded* and *interpretable*:

======================  ==========================================  ==============================
degeneracy              remedy metric                               why it is well defined
======================  ==========================================  ==============================
T1 reference→0          magnitude-normalised ``|y-y_ref|/scale``    denominator is an engineering
                        with tolerance widened to the noise floor   scale, bounded away from 0;
                        ``max(tau, 3*sigma/scale)``                 you cannot demand better
                                                                    than the noise
T2 sign crossing        symmetric log ratio ``|ln(y/y_ref)|`` when   even in the sign; a sign flip
                        both > 0, else signed absolute error        is FAIL by construction
T3 plural tolerances    deterministic merge rule: **strictest       removes the "dual standard"
                        registered tau wins**, absolute metric      degree of freedom
                        before relative (never both)
======================  ==========================================  ==============================

Two rules are structural, not stylistic:

1. **No unbounded numbers leave this module.**  ``error`` is ``None`` or a finite float.
   A caller that wants a percentage can read ``relative_error`` (which is ``None`` when the
   reference vanishes) but can never get ``inf``.
2. **No PARTIAL.**  Every evaluation returns PASS / FAIL / WITHHELD.  ``WITHHELD`` is the
   explicit "cannot be judged" state; it is never counted as a pass.

Merge rule for T3 (:func:`merge_multi_metric`): among competing tolerances take the
*smallest* (strictest) one — the registered tau wins, the looser "判据带" is a post-hoc
escape hatch and is dropped.  When an absolute and a relative criterion coexist, the
**absolute** one wins (its value does not depend on the vanishing reference).  Applying
both and keeping the better outcome is forbidden: that is the dual-standard pathology.
"""
from __future__ import annotations

import math
from typing import Iterable

from .degeneracy import DegeneracyReport, classify
from .model import (
    Criterion,
    Degeneracy,
    Evaluation,
    Metric,
    Verdict,
    relative_error,
)


def _finite(x: float | None) -> float | None:
    if x is None or not isinstance(x, (int, float)):
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def magnitude_normalised_error(criterion: Criterion) -> float | None:
    """``|y - y_ref| / scale_y`` — bounded by construction (effective scale > 0)."""
    m = criterion.measurement
    y, yr = _finite(m.y), _finite(m.y_ref)
    if y is None or yr is None:
        return None
    return _finite(abs(y - yr) / m.effective_scale())


def log_ratio_error(criterion: Criterion) -> float | None:
    """``|ln(y/y_ref)|`` — only defined for positive values (T2 remedy)."""
    m = criterion.measurement
    y, yr = _finite(m.y), _finite(m.y_ref)
    if y is None or yr is None or y <= 0.0 or yr <= 0.0:
        return None
    return _finite(abs(math.log(y / yr)))


def signed_error(criterion: Criterion) -> float | None:
    """``y - y_ref`` keeping the sign (T2 remedy when values are non-positive)."""
    m = criterion.measurement
    y, yr = _finite(m.y), _finite(m.y_ref)
    if y is None or yr is None:
        return None
    return _finite(y - yr)


def excess_over_bound(criterion: Criterion) -> float | None:
    """``max(0, y - y_ref)`` — 0 when the one-sided engineering threshold is met."""
    m = criterion.measurement
    y, yr = _finite(m.y), _finite(m.y_ref)
    if y is None or yr is None:
        return None
    return _finite(max(0.0, y - yr))


def merge_multi_metric(criterion: Criterion) -> tuple[float, Metric, str]:
    """Deterministic merge of competing tolerances (T3 remedy).

    Returns ``(tau, metric, rule)`` with ``rule`` documenting the choice verbatim.
    """
    taus = [t for t in criterion.all_taus if t is not None]
    if not taus:
        raise ValueError("criterion has no tolerance")
    strictest = min(taus)
    metric = criterion.metric
    if len(set(taus)) > 1:
        rule = (f"strictest_tau_wins: {sorted(set(taus))} -> {strictest:g}"
                f" (looser bands dropped; metric from registration: {metric.value})")
    else:
        rule = "single_tau"
    return strictest, metric, rule


def evaluate(
    criterion: Criterion,
    *,
    degeneracy: DegeneracyReport | None = None,
    ref_text: str | None = None,
    verdict_text: str | None = None,
    sign_text: str | None = None,
) -> Evaluation:
    """Evaluate ``criterion`` under the adaptive family (see module docstring)."""
    m = criterion.measurement
    dg = degeneracy or classify(criterion, ref_text=ref_text, verdict_text=verdict_text,
                                sign_text=sign_text)
    rel = relative_error(m.y, m.y_ref)

    # -- 0a. tolerance not in the metric's unit space -> metric mismatch ---------
    if not criterion.tau_available:
        return Evaluation(criterion=criterion, metric=criterion.metric, error=None,
                          tau=criterion.tau, verdict=Verdict.WITHHELD,
                          reason="tolerance_missing_in_metric_space",
                          degeneracy=dg.kind, remedy="withhold_metric_mismatch",
                          relative_error=rel)

    # -- 0. missing data: explicit WITHHELD, never a pass -----------------------
    if m.y is None or m.y_ref is None:
        why = [w for w, flag in (("y_not_measured", m.y is None),
                                 ("reference_absent", m.y_ref is None)) if flag]
        return Evaluation(criterion=criterion, metric=criterion.metric, error=None,
                          tau=criterion.tau, verdict=Verdict.WITHHELD,
                          reason="+".join(why), degeneracy=dg.kind,
                          remedy="withhold_no_numeric_claim", relative_error=rel)

    # -- 1. resolve the standard (T3) before choosing a metric ------------------
    tau, metric, merge_rule = merge_multi_metric(criterion)
    notes: list[str] = []
    if dg.kind is Degeneracy.T3_MULTI_METRIC_NO_PRIORITY or criterion.competing_taus:
        notes.append(merge_rule)

    # -- 2. metric selection by degeneracy -------------------------------------
    scale = m.effective_scale()
    if dg.kind is Degeneracy.T1_REFERENCE_NEAR_ZERO and m.reference_is_placeholder:
        # self-referential reference ("平衡方程本身", "自我加密"): the recorded number is already
        # a dimensionless residual, so there is nothing unbounded to repair - but the ledger
        # keeps the T1 label because a self-consistency check is not an external validation.
        err = rel
        tau_eff = tau
        metric_rep = Metric.RELATIVE
        notes.append("self_referential_reference: judged on the reported dimensionless "
                     "residual (NOT an external validation)")
    elif dg.kind is Degeneracy.T1_REFERENCE_NEAR_ZERO:
        delta = abs((m.y or 0.0) - (m.y_ref or 0.0))
        if m.sigma_noise is not None and delta <= 3.0 * abs(m.sigma_noise):
            # the difference is inside the noise floor: no claim either way (not a pass)
            return Evaluation(criterion=criterion, metric=Metric.RELATIVE, error=None,
                              tau=criterion.tau, verdict=Verdict.WITHHELD,
                              reason="difference_within_noise_floor", degeneracy=dg.kind,
                              remedy="withhold_indistinguishable_from_noise",
                              relative_error=rel)
        err = magnitude_normalised_error(criterion)
        tau_eff = tau
        if m.sigma_noise is not None and scale > 0:
            tau_eff = max(tau, 3.0 * abs(m.sigma_noise) / scale)
        notes.append("magnitude_normalised: |y-y_ref|/scale_y"
                     f" (scale={scale:.6g}, tau_eff={tau_eff:.6g})")
        metric_rep = Metric.RELATIVE
    elif dg.kind is Degeneracy.T2_SIGN_CROSSING:
        lr = log_ratio_error(criterion)
        if lr is not None:
            err, tau_eff = lr, math.log1p(max(tau, 0.0))
            metric_rep = Metric.LOG_RATIO
            notes.append("log_ratio_symmetric_band: |ln(y/y_ref)| <= ln(1+tau)")
        else:
            se = signed_error(criterion)
            err = _finite(abs(se)) if se is not None else None
            tau_eff = tau * scale
            metric_rep = Metric.SIGNED
            notes.append("signed_absolute: |y-y_ref| in engineering unit")
    elif metric is Metric.ABSOLUTE:
        se = signed_error(criterion)
        err = _finite(abs(se)) if se is not None else None
        tau_eff = tau
        metric_rep = Metric.ABSOLUTE
    elif metric is Metric.LOG_RATIO:
        err, tau_eff = log_ratio_error(criterion), math.log1p(max(tau, 0.0))
        metric_rep = Metric.LOG_RATIO
    elif metric is Metric.BOUND:
        # one-sided engineering threshold ("<= 20L", "<= 1e-10"): the error is the *excess*
        # over the bound, zero when inside - bounded, non-negative, metric-explicit.
        err = excess_over_bound(criterion)
        tau_eff = tau
        metric_rep = Metric.BOUND
        notes.append("one_sided_bound: excess = max(0, y - y_ref)")
    elif metric is Metric.SIGNED:
        se = signed_error(criterion)
        err = _finite(abs(se)) if se is not None else None
        tau_eff = tau
        metric_rep = Metric.SIGNED
    else:  # RELATIVE (non-degenerate)
        if rel is not None:
            err, tau_eff, metric_rep = rel, tau, Metric.RELATIVE
        else:  # y_ref == 0 slipped through classification: bounded fallback
            err = magnitude_normalised_error(criterion)
            tau_eff = tau
            metric_rep = Metric.RELATIVE
            notes.append("fallback_magnitude_normalised (relative undefined: y_ref==0)")

    err = _finite(err)
    if err is None:
        return Evaluation(criterion=criterion, metric=metric_rep, error=None,
                          tau=criterion.tau, verdict=Verdict.WITHHELD,
                          reason="error_not_computable", degeneracy=dg.kind,
                          remedy="withhold_metric_undefined", relative_error=rel)

    verdict = Verdict.PASS if err <= tau_eff else Verdict.FAIL
    reason = "within_tolerance" if verdict is Verdict.PASS else "exceeds_tolerance"

    se = signed_error(criterion)
    if dg.kind is Degeneracy.T2_SIGN_CROSSING and se is not None and m.y_ref != 0:
        if (se > 0) != (m.y_ref > 0):
            reason = "sign_flip"
            verdict = Verdict.FAIL  # a wrong-sign result never passes, however small |Δ|
            notes.append("sign_flip_forces_fail")

    return Evaluation(
        criterion=criterion, metric=metric_rep, error=err, tau=tau_eff, verdict=verdict,
        reason=reason + (";" + ";".join(notes) if notes else ""),
        degeneracy=dg.kind, remedy=";".join(notes),
        signed_error=se, relative_error=rel,
    )


def evaluate_baseline(criterion: Criterion) -> Evaluation:
    """The *baseline* criterion: plain relative error ``|Δ|/|y_ref| < tau``.

    This is what historical reports actually used.  It reproduces the pathologies under
    measurement: a vanishing reference yields "not computable" (the author then prints a
    hand-wavy "部分"), and a reference range has no defined denominator.
    """
    m = criterion.measurement
    err = relative_error(m.y, m.y_ref)
    if not criterion.tau_available:
        return Evaluation(criterion=criterion, metric=Metric.RELATIVE, error=err,
                          tau=criterion.tau, verdict=Verdict.WITHHELD,
                          reason="baseline_tolerance_unknown", relative_error=err)
    if err is None:
        return Evaluation(criterion=criterion, metric=Metric.RELATIVE, error=None,
                          tau=criterion.tau, verdict=Verdict.WITHHELD,
                          reason="baseline_relative_error_undefined", relative_error=None)
    verdict = Verdict.PASS if err < criterion.tau else Verdict.FAIL
    return Evaluation(criterion=criterion, metric=Metric.RELATIVE, error=err,
                      tau=criterion.tau, verdict=verdict,
                      reason="baseline_relative_lt_tau", relative_error=err)


def evaluate_all(criteria: Iterable[Criterion], **kw) -> list[Evaluation]:
    return [evaluate(c, **kw) for c in criteria]