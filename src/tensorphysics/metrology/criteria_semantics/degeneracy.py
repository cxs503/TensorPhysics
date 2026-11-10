"""Degeneracy classifier for acceptance criteria (three types, T1/T2/T3).

A criterion is *degenerate* when its recorded error number cannot be interpreted as a
measurement of agreement.  Three structural causes are recognised; each carries a
numeric witness so the classification is auditable (no vibes-based labelling):

``T1_REFERENCE_NEAR_ZERO``
    ``|y_ref| <= max(3*sigma_noise, 0.1*scale_y)``.  The relative error denominator is at
    or below the measurement noise floor, so ``Δ/|y_ref|`` is unbounded: the printed
    percentage is a property of the noise, not of the model.  Also raised when the source
    text says the reference is exactly zero / the identity itself.

``T2_SIGN_CROSSING``
    ``y`` and ``y_ref`` have opposite signs, or the quantity is flagged sign-sensitive and
    ``|y|`` is within a few tolerances of zero.  Around the crossing the relative error is
    a step function of the perturbation (``+100 %`` vs ``-100 %`` for the same |Δ|).

``T3_MULTI_METRIC_NO_PRIORITY``
    Two or more tolerances / reference values apply to the same quantity with no stated
    priority (``<3%`` in the header vs ``<10% 判据带`` in the cell; reference given as a
    range ``0.04-0.06``).  The verdict is then a choice of standard, not a measurement;
    the dual-standard pattern is what lets a 60 % deviation be filed as "部分通过".

Classification reads both channels: numbers when present, provenance text as evidence.
The textual channel is recorded in ``evidence`` so a human reviewer can re-derive the label.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .model import Criterion, Degeneracy, Measurement

#: relative floor below which a reference is considered "no reference" in provenance text
_NO_REF_PATTERNS = (
    r"无基准", r"缺基准", r"无离线", r"无解析", r"无实测", r"无实验", r"不存在", r"未评估",
    r"不适用", r"withheld",
    r"self[- ]consisten", r"自一致", r"no reference", r"n/a",
)
#: textual evidence that the reference magnitude is *exactly zero / the identity itself*
#: a reference that *is* zero: the cell ends with "=0" / says "精确 0" / "应为 0" / "极限 0".
#: A cell that merely *contains* a zero (e.g. a standards table "1 kHz=0, 100 Hz=-19.1") is
#: not a zero reference and must not fire.
_EXACT_ZERO_PATTERNS = (
    r"精确\s*0\s*$",
    r"=\s*0(?:\.0*)?\s*(?:[A-Za-z%°/]+)?\s*$",
    r"应为\s*0", r"极限\s*0", r"平衡条件", r"基准\s*=?\s*0\s*$",
)
#: containment criteria: an interval used as an acceptance *set* is not an ambiguous
#: reference value (nothing has to be chosen), so it is not T3
_CONTAINMENT_PATTERNS = (
    r"落在带内", r"在带内", r"带内", r"within (?:the )?band", r"落在区间", r"包络.*落在",
)
#: textual evidence that the reference is the implementation itself (self-consistency, not
#: an external validation).  Recorded as a separate witness: it is a real measurement
#: pathology but it is NOT the spec's T1 (whose definition is a vanishing reference magnitude).
_SELF_REFERENTIAL_PATTERNS = (
    r"恒等", r"平衡方程", r"逐位", r"残差", r"自我加密", r"self[- ]refin", r"自洽",
    r"自参照", r"同式复算", r"同一几何", r"同上", r"重新渲染", r"sha256", r"自身基准",
)
_RANGE_PATTERN = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:-|–|~|～|to)\s*(-?\d+(?:\.\d+)?)")
_MULTI_TAU_PATTERN = re.compile(r"(?:<|≤|<=|不超过)\s*(\d+(?:\.\d+)?)\s*%")
_NUM_PAT = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
_BAND_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*%\s*判据带|判据带\s*(\d+(?:\.\d+)?)\s*%")


@dataclass
class DegeneracyReport:
    """Classification result with the numeric/textual witness that produced it."""

    kind: Degeneracy
    witnesses: dict[str, Any] = field(default_factory=dict)
    #: all degeneracy types that fired (a criterion can fire more than one)
    also: tuple[Degeneracy, ...] = ()
    #: reference is the implementation itself (self-consistency): reported separately from
    #: the T1/T2/T3 taxonomy, which is about the reference *magnitude*
    self_referential: bool = False

    @property
    def is_degenerate(self) -> bool:
        return self.kind is not Degeneracy.NONE

    def as_dict(self) -> dict[str, Any]:
        return {
            "degeneracy": self.kind.value,
            "also": [d.value for d in self.also],
            "self_referential": self.self_referential,
            "witnesses": self.witnesses,
        }


def near_zero_threshold(scale_y: float | None, sigma_noise: float | None) -> float:
    """The T1 trigger: ``max(3*sigma_noise, 0.1*scale_y)`` with safe fallbacks."""
    cands = []
    if sigma_noise is not None:
        cands.append(3.0 * abs(sigma_noise))
    if scale_y is not None:
        cands.append(0.1 * abs(scale_y))
    if not cands:
        return 0.0
    return max(cands)


def classify(
    criterion: Criterion,
    *,
    ref_text: str | None = None,
    verdict_text: str | None = None,
    sign_text: str | None = None,
    error_text: str | None = None,
    self_referential_as_degeneracy: bool = False,
) -> DegeneracyReport:
    """Classify the metrology degeneracy of ``criterion``.

    ``ref_text`` / ``verdict_text`` / ``sign_text`` are optional provenance strings
    (the report cells the criterion was parsed from); they are used as *evidence* and
    echoed into ``witnesses`` so the label is reproducible.
    """
    m: Measurement = criterion.measurement
    fired: list[tuple[Degeneracy, dict[str, Any]]] = []

    # -- 0. exact agreement is metric-invariant: nothing can be degenerate ---------
    if criterion.reported_error_is_zero or (m.y == 0.0 and (m.y_ref is None or m.y_ref == 0.0)):
        return DegeneracyReport(Degeneracy.NONE, {"why": "exact_zero_error"},
                                self_referential=False)

    # -- 0a. reference declared absent: the reference magnitude is undefined (T1) --
    if m.reference_absent:
        return DegeneracyReport(Degeneracy.T1_REFERENCE_NEAR_ZERO,
                                {"why": "reference_absent", "ref_text": ref_text},
                                self_referential=False)

    # -- 0c. nothing numeric to judge: missing measurement, not a degeneracy -----
    if m.y is None and m.y_ref is None and m.ref_range is None:
        return DegeneracyReport(Degeneracy.NONE, {"why": "no_numeric_content"})
    # -- 0a. containment criterion (interval used as an acceptance set) ----------
    if error_text and any(re.search(p, error_text, re.I) for p in _CONTAINMENT_PATTERNS):
        return DegeneracyReport(Degeneracy.NONE,
                                {"why": "containment_criterion", "error_text": error_text},
                                self_referential=False)


    # ---- T1: reference magnitude at/below the noise floor ---------------------
    thr = near_zero_threshold(m.scale_y, m.sigma_noise)
    t1_w: dict[str, Any] = {}
    if m.y is not None and m.y == 0.0 and m.y_ref == 0.0:
        t1_w = {}                        # exact agreement: |Δ|=0 is metric-invariant
    elif m.y_ref is None:
        if criterion.metric is not Degeneracy and criterion.metric.value == "relative":
            t1_w["reference"] = "absent"   # a relative criterion with no reference magnitude
    else:
        if m.y_ref == 0.0:
            t1_w["y_ref"] = 0.0
            t1_w["why"] = "exact_zero_reference"
        elif thr > 0 and abs(m.y_ref) <= thr:
            t1_w["y_ref"] = m.y_ref
            t1_w["threshold"] = thr
            t1_w["why"] = "below_noise_floor"
    self_ref = False
    for _text in (ref_text, verdict_text):
        if _text and any(re.search(p, _text, re.I) for p in _NO_REF_PATTERNS):
            t1_w.setdefault("why", "text_no_reference")
            t1_w["ref_text"] = _text
    if verdict_text and re.search(r"条件数发散|分母.*发散|趋零", verdict_text):
        t1_w.setdefault("why", "text_denominator_diverges")
        t1_w["verdict_text"] = verdict_text
    if ref_text:
        low = ref_text.lower()
        if any(re.search(p, low) for p in _EXACT_ZERO_PATTERNS):
            t1_w.setdefault("why", "text_exact_zero_reference")
            t1_w["ref_text"] = ref_text
        if any(re.search(p, low) for p in _NO_REF_PATTERNS):
            t1_w.setdefault("why", "text_no_reference")
            t1_w["ref_text"] = ref_text
        self_ref = any(re.search(p, low) for p in _SELF_REFERENTIAL_PATTERNS)
    if self_ref and self_referential_as_degeneracy and not t1_w:
        # legacy behaviour (kept for the before/after comparison of the labelling round)
        t1_w = {"why": "text_self_referential", "ref_text": ref_text}
    if t1_w:
        fired.append((Degeneracy.T1_REFERENCE_NEAR_ZERO, t1_w))

    # ---- T2: sign crossing / sign-sensitive quantity near zero ----------------
    t2_w: dict[str, Any] = {}
    if m.y is not None and m.y_ref is not None and m.y != 0.0 and m.y_ref != 0.0:
        if (m.y > 0) != (m.y_ref > 0):
            t2_w = {"y": m.y, "y_ref": m.y_ref, "why": "opposite_signs"}
    if criterion.sign_sensitive and m.y is not None and m.y_ref is not None:
        # a quantity the author declares sign-sensitive cannot be judged by a symmetric
        # relative error: the sign carries meaning, so T2 always applies
        t2_w = {**t2_w, "sign_sensitive": True, "y": m.y, "y_ref": m.y_ref,
                "why": t2_w.get("why", "sign_sensitive_declared")}
    if sign_text and re.search(r"[Δ∆]\s*\w*\s*[<>]\s*0|变号|跨零|反号|sign[- ]flip|符号相反", sign_text):
        t2_w = {**t2_w, "sign_text": sign_text}
        t2_w.setdefault("why", "text_sign_sensitive")
    if t2_w:
        fired.append((Degeneracy.T2_SIGN_CROSSING, t2_w))

    # ---- T3: plural tolerances / reference range with no priority -------------
    t3_w: dict[str, Any] = {}
    if len(criterion.competing_taus) > 0:
        t3_w["taus"] = list(criterion.all_taus)
        t3_w["why"] = "competing_tolerances"
    containment = any(re.search(p, error_text or "", re.I) for p in _CONTAINMENT_PATTERNS)
    if m.ref_range is not None and not containment:
        t3_w["ref_range"] = list(m.ref_range)
        t3_w.setdefault("why", "reference_range_no_priority")
    if containment:
        t3_w = {}
    if ref_text and not containment:
        mm = _RANGE_PATTERN.search(ref_text)
        if mm:
            try:
                lo, hi = float(mm.group(1)), float(mm.group(2))
                if lo != hi:
                    t3_w.setdefault("ref_range", [lo, hi])
                    t3_w.setdefault("why", "text_reference_range")
                    t3_w["ref_text"] = ref_text
            except ValueError:  # pragma: no cover - defensive
                pass
    if verdict_text:
        taus = [float(v) for v in _MULTI_TAU_PATTERN.findall(verdict_text)]
        for g1, g2 in _BAND_PATTERN.findall(verdict_text):
            taus.append(float(g1 or g2))
        loosened = re.findall(rf"放宽(?:到|至)\s*({_NUM_PAT})\s*%", verdict_text)
        taus += [float(v) for v in loosened]
        uniq = sorted({t for t in taus if t > 0})
        if len(uniq) >= 2:
            t3_w["taus"] = uniq
            t3_w.setdefault("why", "text_dual_standard")
            t3_w["verdict_text"] = verdict_text
    if t3_w:
        fired.append((Degeneracy.T3_MULTI_METRIC_NO_PRIORITY, t3_w))

    if not fired:
        wit = {"self_referential": True, "ref_text": ref_text} if self_ref else {}
        return DegeneracyReport(Degeneracy.NONE, wit, self_referential=self_ref)
    # Priority of reporting: T1 (unbounded) > T2 (discontinuous) > T3 (ambiguous standard)
    order = {
        Degeneracy.T1_REFERENCE_NEAR_ZERO: 0,
        Degeneracy.T2_SIGN_CROSSING: 1,
        Degeneracy.T3_MULTI_METRIC_NO_PRIORITY: 2,
    }
    fired.sort(key=lambda kv: order[kv[0]])
    kind, wit = fired[0]
    return DegeneracyReport(kind, wit, also=tuple(k for k, _ in fired[1:]),
                            self_referential=self_ref)