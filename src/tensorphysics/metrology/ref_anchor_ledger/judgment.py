"""
Three-state judgment for validation criteria with uncertainty.

The judgment rewrites the traditional "|Δ| ≤ 15%" criterion into a
statistical proposition under k=2 coverage:

    |Δ| ≤ k · √(U_num² + U_ref²)

Three states:
- ROBUST:       |Δ| ≤ k · √(U_num² + U_ref²)
                → Criterion clearly satisfied within combined uncertainty
- CRITICAL:     k · √(...) < |Δ| ≤ 2k · √(...)
                → Criterion borderline; may pass or fail depending on
                  uncertainty realization
- INDETERMINATE: U_ref missing or too large, or |Δ| > 2k · √(...)
                → Cannot make a reliable judgment; fail-closed

The fail-closed principle: when uncertainty is too large or missing,
the judgment defaults to INDETERMINATE (not ROBUST). This prevents
false positives from poorly characterized anchors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ThreeStateJudgment(Enum):
    """Three-state judgment result."""
    ROBUST = "robust"           # Clearly satisfied
    CRITICAL = "critical"       # Borderline
    INDETERMINATE = "indeterminate"  # Cannot judge → fail-closed


@dataclass(frozen=True)
class JudgmentResult:
    """
    Result of a three-state judgment.

    Attributes
    ----------
    judgment : ThreeStateJudgment
        The three-state classification
    delta : float
        Absolute difference |computed - reference|
    U_combined : float
        Combined expanded uncertainty k·√(U_num² + U_ref²)
    U_num : float
        Numerical (computational) uncertainty at k=2
    U_ref : float
        Reference anchor uncertainty at k=2
    k : float
        Coverage factor (default 2)
    threshold_robust : float
        Threshold for ROBUST: k·√(U_num² + U_ref²)
    threshold_critical : float
        Threshold for CRITICAL: 2k·√(U_num² + U_ref²)
    reason : str
        Human-readable explanation of the judgment
    """
    judgment: ThreeStateJudgment
    delta: float
    U_combined: float
    U_num: float
    U_ref: float
    k: float
    threshold_robust: float
    threshold_critical: float
    reason: str

    @property
    def passed(self) -> bool:
        """Whether the criterion is considered passed (ROBUST only)."""
        return self.judgment == ThreeStateJudgment.ROBUST

    @property
    def fail_closed(self) -> bool:
        """Whether this is a fail-closed result (INDETERMINATE)."""
        return self.judgment == ThreeStateJudgment.INDETERMINATE


def compute_combined_uncertainty(
    U_num: float,
    U_ref: float,
    k: float = 2.0,
) -> float:
    """
    Compute combined expanded uncertainty.

    U_combined = k · √(U_num² + U_ref²)

    Parameters
    ----------
    U_num : float
        Numerical uncertainty (standard uncertainty × k)
    U_ref : float
        Reference anchor uncertainty (standard uncertainty × k)
    k : float
        Coverage factor (default 2 for 95% confidence)

    Returns
    -------
    float
        Combined expanded uncertainty
    """
    return k * math.sqrt(U_num**2 + U_ref**2)


def judge_criterion(
    delta: float,
    U_num: float,
    U_ref: Optional[float] = None,
    k: float = 2.0,
    critical_multiplier: float = 2.0,
) -> JudgmentResult:
    """
    Three-state judgment of a validation criterion.

    Parameters
    ----------
    delta : float
        Absolute difference |computed - reference|
    U_num : float
        Numerical uncertainty at k=2
    U_ref : float or None
        Reference anchor uncertainty at k=2. If None, judgment is INDETERMINATE.
    k : float
        Coverage factor (default 2)
    critical_multiplier : float
        Multiplier for CRITICAL threshold (default 2.0, so CRITICAL zone is
        between k·√(...) and 2k·√(...))

    Returns
    -------
    JudgmentResult
        Three-state judgment with full metadata.
    """
    # Handle missing U_ref → fail-closed
    if U_ref is None or U_ref < 0:
        U_combined = float('inf')
        return JudgmentResult(
            judgment=ThreeStateJudgment.INDETERMINATE,
            delta=abs(delta),
            U_combined=float('inf'),
            U_num=U_num,
            U_ref=U_ref if U_ref is not None else 0.0,
            k=k,
            threshold_robust=float('inf'),
            threshold_critical=float('inf'),
            reason="U_ref missing or negative → fail-closed (INDETERMINATE)",
        )

    # Compute combined uncertainty
    U_combined = compute_combined_uncertainty(U_num, U_ref, k)
    threshold_robust = U_combined
    threshold_critical = critical_multiplier * U_combined
    abs_delta = abs(delta)

    # Three-state classification
    if abs_delta <= threshold_robust:
        judgment = ThreeStateJudgment.ROBUST
        reason = (
            f"|Δ|={abs_delta:.4g} ≤ k·√(U_num²+U_ref²)={threshold_robust:.4g} "
            f"→ ROBUST (criterion clearly satisfied)"
        )
    elif abs_delta <= threshold_critical:
        judgment = ThreeStateJudgment.CRITICAL
        reason = (
            f"k·√(...)={threshold_robust:.4g} < |Δ|={abs_delta:.4g} "
            f"≤ 2k·√(...)={threshold_critical:.4g} "
            f"→ CRITICAL (borderline)"
        )
    else:
        judgment = ThreeStateJudgment.INDETERMINATE
        reason = (
            f"|Δ|={abs_delta:.4g} > 2k·√(...)={threshold_critical:.4g} "
            f"→ INDETERMINATE (exceeds critical zone, fail-closed)"
        )

    return JudgmentResult(
        judgment=judgment,
        delta=abs_delta,
        U_combined=U_combined,
        U_num=U_num,
        U_ref=U_ref,
        k=k,
        threshold_robust=threshold_robust,
        threshold_critical=threshold_critical,
        reason=reason,
    )


def batch_judge(
    deltas: list,
    U_nums: list,
    U_refs: list,
    k: float = 2.0,
) -> list:
    """
    Batch judgment for multiple criteria.

    Parameters
    ----------
    deltas : list of float
        Absolute differences
    U_nums : list of float
        Numerical uncertainties
    U_refs : list of float or None
        Reference uncertainties

    Returns
    -------
    list of JudgmentResult
    """
    return [
        judge_criterion(d, u_n, u_r, k=k)
        for d, u_n, u_r in zip(deltas, U_nums, U_refs)
    ]


def judgment_statistics(results: list) -> dict:
    """
    Compute statistics over a batch of judgments.

    Returns
    -------
    dict
        Counts and fractions for each state.
    """
    total = len(results)
    if total == 0:
        return {"total": 0, "robust": 0, "critical": 0, "indeterminate": 0}

    counts = {
        ThreeStateJudgment.ROBUST: 0,
        ThreeStateJudgment.CRITICAL: 0,
        ThreeStateJudgment.INDETERMINATE: 0,
    }
    for r in results:
        counts[r.judgment] += 1

    return {
        "total": total,
        "robust": counts[ThreeStateJudgment.ROBUST],
        "critical": counts[ThreeStateJudgment.CRITICAL],
        "indeterminate": counts[ThreeStateJudgment.INDETERMINATE],
        "robust_fraction": counts[ThreeStateJudgment.ROBUST] / total,
        "critical_fraction": counts[ThreeStateJudgment.CRITICAL] / total,
        "indeterminate_fraction": counts[ThreeStateJudgment.INDETERMINATE] / total,
        "fail_closed_count": counts[ThreeStateJudgment.INDETERMINATE],
    }
