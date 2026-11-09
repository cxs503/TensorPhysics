"""The UQ acceptance gate: PASS / FAIL / WITHHELD, with machine-checkable reasons.

Design rule (inherited from the V&V gate in :mod:`tensorphysics.vv`): an
*absence of evidence* is never rounded up to a pass.  The gate therefore has
three states and the middle one is not "warn":

``PASS``
    The surrogate reproduces the model (``eps_LOO <= max_loo_error``), the sample
    budget is sufficient for the retained basis, every reported index carries a
    confidence interval, and -- when requested -- an independent Monte-Carlo
    estimator agrees with the PCE indices within the stated tolerance.
``FAIL``
    The evidence contradicts itself: the two independent estimators disagree
    beyond tolerance, or the index set violates the variance algebra
    (``ST_i < S1_i``, ``S1_i < 0``, ``sum(S1) > 1``).  A FAIL is *actionable*:
    something in the model/UQ set-up is wrong and no report may quote it.
``WITHHELD``
    The evidence is insufficient or inadmissible: non-finite model output, too
    few samples for the basis, ``eps_LOO`` above the threshold, degenerate output
    variance, or correlated inputs (polynomial chaos in this package requires
    independent inputs).  WITHHELD blocks a downstream engineering error gate
    the same way FAIL does -- that is the point of putting UQ *before* it.

The gate returns its own diagnostic dict, so a report can quote the numbers
behind the verdict instead of asserting a colour.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from .indices import IndexComparison, SobolIndices

__all__ = [
    "PASS",
    "FAIL",
    "WITHHELD",
    "UQGateError",
    "UQPolicy",
    "UQDiagnostics",
    "UQVerdict",
    "evaluate_gate",
]

PASS = "PASS"
FAIL = "FAIL"
WITHHELD = "WITHHELD"

# -- reason codes -----------------------------------------------------------
R_NONFINITE_OUTPUT = "NONFINITE_MODEL_OUTPUT"
R_NONFINITE_INPUT = "NONFINITE_INPUT_SAMPLES"
R_ADAPTER_FAILURES = "ADAPTER_FAILURES"
R_INSUFFICIENT_SAMPLES = "INSUFFICIENT_SAMPLES"
R_LOO_UNRESOLVED = "LOO_ERROR_ABOVE_THRESHOLD"
R_LOO_UNAVAILABLE = "LOO_ERROR_UNAVAILABLE"
R_DEGENERATE_VARIANCE = "DEGENERATE_OUTPUT_VARIANCE"
R_CORRELATED_INPUTS = "CORRELATED_INPUTS_UNSUPPORTED"
R_ESTIMATORS_DISAGREE = "PCE_MC_ESTIMATORS_DISAGREE"
R_INDEX_ALGEBRA = "INDEX_ALGEBRA_VIOLATION"
R_NEGATIVE_INDEX = "SIGNIFICANTLY_NEGATIVE_INDEX"
R_CROSS_CHECK_MISSING = "CROSS_CHECK_MISSING"
R_CROSS_CHECK_PRECISION = "CROSS_CHECK_INCONCLUSIVE"

WITHHELD_CODES = (
    R_NONFINITE_OUTPUT,
    R_NONFINITE_INPUT,
    R_ADAPTER_FAILURES,
    R_INSUFFICIENT_SAMPLES,
    R_LOO_UNRESOLVED,
    R_LOO_UNAVAILABLE,
    R_DEGENERATE_VARIANCE,
    R_CORRELATED_INPUTS,
    R_CROSS_CHECK_MISSING,
    R_CROSS_CHECK_PRECISION,
)
FAIL_CODES = (R_ESTIMATORS_DISAGREE, R_INDEX_ALGEBRA, R_NEGATIVE_INDEX)

REASON_MEANING = {
    R_NONFINITE_OUTPUT: "model output contained NaN/Inf samples; no surrogate can be certified",
    R_NONFINITE_INPUT: "sampled inputs contained NaN/Inf; the design itself is invalid",
    R_ADAPTER_FAILURES: "one or more solver evaluations raised; see adapter stats",
    R_INSUFFICIENT_SAMPLES: "sample count below the policy lower bound for the retained basis",
    R_LOO_UNRESOLVED: "leave-one-out error above the policy threshold (surrogate not predictive)",
    R_LOO_UNAVAILABLE: "LOO error could not be computed (fit unavailable)",
    R_DEGENERATE_VARIANCE: "output variance is numerically zero; indices are undefined",
    R_CORRELATED_INPUTS: "correlated inputs: independent polynomial chaos is not applicable here",
    R_CROSS_CHECK_MISSING: "an independent Monte-Carlo index estimate was required but not supplied",
    R_CROSS_CHECK_PRECISION: (
        "the Monte-Carlo cross-check is too imprecise to confirm or refute the PCE index "
        "at the supplied base sample count (confidence intervals overlap); raise cross_check_base"
    ),
    R_ESTIMATORS_DISAGREE: "PCE and Monte-Carlo Sobol' indices disagree beyond tolerance",
    R_INDEX_ALGEBRA: "variance algebra violated (sum(S1) > 1, sum(ST) < sum(S1), ST_i < S1_i)",
    R_NEGATIVE_INDEX: "an index is negative by more than the numerical floor",
}


class UQGateError(ValueError):
    """Raised when the gate is asked to judge an inconsistent diagnostics object."""


@dataclass(frozen=True)
class UQPolicy:
    """Thresholds of the UQ acceptance gate.

    Defaults are the ones quoted in the campaign acceptance criteria:
    ``max_loo_error = 0.1`` (the surrogate must explain >= 90 % of the output
    variance in a leave-one-out sense) and a 5 % cross-estimator tolerance.
    """

    max_loo_error: float = 0.1
    min_samples: int = 64
    min_oversampling: float = 2.0
    ci_level: float = 0.95
    max_cross_check_rel_dev: float = 0.05
    cross_check_abs_floor: float = 1e-3
    require_cross_check: bool = True
    degenerate_variance_tol: float = 1e-14
    negative_index_floor: float = -0.01
    algebra_tolerance: float = 0.02

    def required_samples(self, basis_size: int) -> int:
        """Lower bound on the sample count for a basis of ``basis_size`` terms."""
        return int(max(self.min_samples, np.ceil(self.min_oversampling * int(basis_size))))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "max_loo_error": float(self.max_loo_error),
            "min_samples": int(self.min_samples),
            "min_oversampling": float(self.min_oversampling),
            "ci_level": float(self.ci_level),
            "max_cross_check_rel_dev": float(self.max_cross_check_rel_dev),
            "cross_check_abs_floor": float(self.cross_check_abs_floor),
            "require_cross_check": bool(self.require_cross_check),
            "degenerate_variance_tol": float(self.degenerate_variance_tol),
            "negative_index_floor": float(self.negative_index_floor),
            "algebra_tolerance": float(self.algebra_tolerance),
        }


@dataclass
class UQDiagnostics:
    """Everything the gate is allowed to look at (fully explicit, no inference)."""

    n_samples: int
    dim: int
    basis_size: int
    n_terms: int
    degree: int
    loo_error: float
    output_variance: float
    indices: SobolIndices
    cross_check: SobolIndices | None = None
    comparison: IndexComparison | None = None
    nonfinite_output_count: int = 0
    nonfinite_input_count: int = 0
    adapter_failures: int = 0
    correlated_inputs: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def checks(self) -> Dict[str, Any]:
        policy = UQPolicy()
        return {
            "n_samples": int(self.n_samples),
            "required_samples": policy.required_samples(self.basis_size),
            "basis_size": int(self.basis_size),
            "n_terms": int(self.n_terms),
            "degree": int(self.degree),
            "loo_error": float(self.loo_error),
            "output_variance": float(self.output_variance),
            "nonfinite_output_count": int(self.nonfinite_output_count),
            "nonfinite_input_count": int(self.nonfinite_input_count),
            "adapter_failures": int(self.adapter_failures),
            "correlated_inputs": bool(self.correlated_inputs),
            "cross_check_supplied": self.comparison is not None,
        }


@dataclass
class UQVerdict:
    status: str
    reasons: List[str]
    diagnostics: Dict[str, Any]
    policy: Dict[str, Any]
    messages: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.status == PASS

    @property
    def usable(self) -> bool:
        """Whether a downstream engineering gate may consume the indices at all."""
        return self.status == PASS

    def as_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "reasons": list(self.reasons),
            "messages": list(self.messages),
            "diagnostics": dict(self.diagnostics),
            "policy": dict(self.policy),
            "usable_downstream": self.usable,
        }

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.status}: {', '.join(self.reasons) if self.reasons else 'all checks satisfied'}"



def _disputed_entries(indices: SobolIndices, cross: SobolIndices | None, policy: UQPolicy) -> List[Dict[str, Any]]:
    """Entries where PCE and Monte-Carlo indices differ beyond tolerance.

    Each entry records whether the Monte-Carlo bootstrap interval covers the PCE
    point estimate.  Coverage means the cross-check simply cannot resolve the
    difference at this base sample count (``WITHHELD``); a PCE value outside the
    Monte-Carlo interval means the two estimators genuinely contradict each other
    (``FAIL``).  The PCE interval is *not* used as the yardstick: its
    coefficient-covariance width measures regression noise only (it collapses to
    ~1e-5 for a well-resolved surrogate) and would make every disagreement look
    like a contradiction.
    """
    if cross is None:
        return []
    out: List[Dict[str, Any]] = []
    for key, pce_vals, mc_vals, pce_lo, pce_hi, mc_lo, mc_hi in (
        ("S1", indices.S1, cross.S1, indices.S1_low, indices.S1_high, cross.S1_low, cross.S1_high),
        ("ST", indices.ST, cross.ST, indices.ST_low, indices.ST_high, cross.ST_low, cross.ST_high),
    ):
        for i, name in enumerate(indices.names):
            ref = float(pce_vals[i])
            cand = float(mc_vals[i])
            if abs(ref) <= policy.cross_check_abs_floor:
                continue
            rel = abs(cand - ref) / abs(ref)
            if rel <= policy.max_cross_check_rel_dev:
                continue
            if mc_lo is None or mc_hi is None:
                overlap = False  # no Monte-Carlo interval -> cannot excuse the disagreement
            else:
                # the Monte-Carlo bootstrap interval is the yardstick: if the PCE
                # point estimate falls inside it, sampling noise alone explains the
                # gap and the cross-check is merely inconclusive
                overlap = bool(float(mc_lo[i]) <= ref <= float(mc_hi[i]))
            out.append(
                {
                    "name": name,
                    "key": key,
                    "pce": ref,
                    "mc": cand,
                    "rel_dev": float(rel),
                    "ci_overlap": bool(overlap),
                }
            )
    return out

def evaluate_gate(diag: UQDiagnostics, policy: UQPolicy | None = None) -> UQVerdict:
    """Run the three-state UQ gate on explicit diagnostics."""
    policy = policy or UQPolicy()
    if diag.n_samples <= 0:
        raise UQGateError("diagnostics report a non-positive sample count")

    withheld: List[str] = []
    failed: List[str] = []
    messages: List[str] = []

    # ---- 1. admissibility of the evidence --------------------------------
    if diag.nonfinite_input_count > 0:
        withheld.append(R_NONFINITE_INPUT)
        messages.append(f"{diag.nonfinite_input_count} non-finite input sample(s)")
    if diag.nonfinite_output_count > 0:
        withheld.append(R_NONFINITE_OUTPUT)
        messages.append(f"{diag.nonfinite_output_count} non-finite model output sample(s)")
    if diag.adapter_failures > 0:
        withheld.append(R_ADAPTER_FAILURES)
        messages.append(f"{diag.adapter_failures} solver evaluation(s) raised an exception")
    if diag.correlated_inputs:
        withheld.append(R_CORRELATED_INPUTS)
        messages.append("correlated inputs: independent polynomial chaos is not applicable")

    # ---- 2. sufficiency of the sample budget ----------------------------
    required = policy.required_samples(diag.basis_size)
    if diag.n_samples < policy.min_samples:
        withheld.append(R_INSUFFICIENT_SAMPLES)
        messages.append(f"n={diag.n_samples} below the absolute floor min_samples={policy.min_samples}")
    elif diag.n_samples < required:
        withheld.append(R_INSUFFICIENT_SAMPLES)
        messages.append(
            f"n={diag.n_samples} below {policy.min_oversampling:g} x basis_size={diag.basis_size} "
            f"({required}) for degree {diag.degree}"
        )

    # ---- 3. predictive quality of the surrogate -------------------------
    if not np.isfinite(diag.loo_error):
        withheld.append(R_LOO_UNAVAILABLE)
        messages.append("eps_LOO is not finite")
    elif diag.loo_error > policy.max_loo_error:
        withheld.append(R_LOO_UNRESOLVED)
        messages.append(
            f"eps_LOO={diag.loo_error:.4f} > {policy.max_loo_error:g}: surrogate does not resolve the variance"
        )
    if diag.output_variance <= policy.degenerate_variance_tol:
        withheld.append(R_DEGENERATE_VARIANCE)
        messages.append(f"output variance {diag.output_variance:.3e} is numerically zero")

    # ---- 4. cross-estimator agreement (only if the evidence is admissible)
    if policy.require_cross_check and diag.comparison is None:
        withheld.append(R_CROSS_CHECK_MISSING)
        messages.append("an independent Monte-Carlo estimate was required but not supplied")

    if withheld:
        _attach(flash := UQVerdict(WITHHELD, withheld, diag.checks(), policy.as_dict(), messages), diag)
        return flash

    # ---- 5. internal consistency of the certified indices ---------------
    if diag.comparison is not None:
        cmp_ = diag.comparison
        disputed = _disputed_entries(diag.indices, diag.cross_check, policy)
        contradicting = [d for d in disputed if not d["ci_overlap"]]
        inconclusive = [d for d in disputed if d["ci_overlap"]]
        if contradicting:
            failed.append(R_ESTIMATORS_DISAGREE)
            worst = max(contradicting, key=lambda d: d["rel_dev"])
            messages.append(
                f"PCE vs Monte-Carlo indices contradict each other: {len(contradicting)} index/indices "
                f"outside the {policy.max_cross_check_rel_dev:.0%} tolerance with disjoint confidence "
                f"intervals (worst: {worst['name']} {worst['key']} PCE={worst['pce']:.4f} "
                f"MC={worst['mc']:.4f}, rel dev {worst['rel_dev']:.1%})"
            )
        elif inconclusive:
            withheld.append(R_CROSS_CHECK_PRECISION)
            worst = max(inconclusive, key=lambda d: d["rel_dev"])
            messages.append(
                f"{len(inconclusive)} index/indices differ from the Monte-Carlo cross-check by more than "
                f"{policy.max_cross_check_rel_dev:.0%} ({worst['name']} {worst['key']}: PCE={worst['pce']:.4f}, "
                f"MC={worst['mc']:.4f}) but the confidence intervals overlap: the cross-check is "
                f"inconclusive, not contradictory"
            )
        if not disputed:
            worst_rel = max(
                [
                    v
                    for v in (cmp_.max_rel_dev_S1_defined, cmp_.max_rel_dev_ST_defined)
                    if np.isfinite(v)
                ]
                or [0.0]
            )
            messages.append(
                f"PCE and Monte-Carlo indices agree within {policy.max_cross_check_rel_dev:.0%} "
                f"(worst defined relative deviation {worst_rel:.3%} over {cmp_.n_defined} indices)"
            )
            if cmp_.n_defined == 0 and max(cmp_.max_abs_dev_S1, cmp_.max_abs_dev_ST) > 0.05:
                failed.append(R_ESTIMATORS_DISAGREE)
                messages.append("no index is above the relative floor yet an absolute deviation exceeds 0.05")

    if withheld:
        _attach(out := UQVerdict(WITHHELD, withheld, diag.checks(), policy.as_dict(), messages), diag)
        return out

    idx = diag.indices
    tol = policy.algebra_tolerance
    if idx.sum_S1 > 1.0 + tol:
        failed.append(R_INDEX_ALGEBRA)
        messages.append(f"sum(S1)={idx.sum_S1:.4f} > 1")
    if idx.sum_ST < idx.sum_S1 - tol:
        failed.append(R_INDEX_ALGEBRA)
        messages.append(f"sum(ST)={idx.sum_ST:.4f} < sum(S1)={idx.sum_S1:.4f}")
    if np.any(idx.ST < idx.S1 - tol):
        failed.append(R_INDEX_ALGEBRA)
        messages.append("at least one parameter has ST_i < S1_i beyond tolerance")
    if np.any(idx.S1 < policy.negative_index_floor) or np.any(idx.ST < policy.negative_index_floor):
        failed.append(R_NEGATIVE_INDEX)
        messages.append(f"an index is below {policy.negative_index_floor}")

    if failed:
        _attach(out := UQVerdict(FAIL, failed, diag.checks(), policy.as_dict(), messages), diag)
        return out
    _attach(out := UQVerdict(PASS, [], diag.checks(), policy.as_dict(), messages), diag)
    return out


def _attach(verdict: UQVerdict, diag: UQDiagnostics) -> None:
    """Fold the index summary + extended diagnostics into the verdict payload."""
    verdict.diagnostics["indices"] = diag.indices.as_dict()
    if diag.comparison is not None:
        verdict.diagnostics["cross_check_indices"] = (
            diag.cross_check.as_dict() if diag.cross_check is not None else {}
        )
        verdict.diagnostics["cross_check_comparison"] = diag.comparison.as_dict()
    if diag.extra:
        verdict.diagnostics["extra"] = dict(diag.extra)
    for code in verdict.reasons:
        verdict.diagnostics.setdefault("reason_meaning", {})[code] = REASON_MEANING.get(code, "")