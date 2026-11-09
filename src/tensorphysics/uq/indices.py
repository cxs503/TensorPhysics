"""Variance-based global sensitivity measures (Sobol' indices) as a data object.

One container is shared by every estimator in this package (polynomial-chaos
analytical indices and Saltelli/Jansen Monte-Carlo indices) so that the
cross-check in :mod:`tensorphysics.uq.gate` compares like with like:

* ``S1``  first-order index ``Var(E[Y|X_i]) / Var(Y)``,
* ``ST``  total index ``E[Var(Y|X_-i)] / Var(Y)`` (includes every interaction),
* ``S2``  closed second-order index ``Var(E[Y|X_i, X_j]) / Var(Y) - S1_i - S1_j``,
* the closed-form interaction sum rule ``sum(S1) <= 1 <= sum(ST)``.

Confidence intervals are 95 % percentile intervals from an estimator-specific
resampling scheme (coefficient-covariance sampling for PCE, bootstrap over base
samples for Saltelli), stored alongside the point value so a report can never
quote an index without its interval.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

__all__ = ["SobolIndices", "IndexComparison", "compare_indices"]


@dataclass
class SobolIndices:
    names: Sequence[str]
    S1: np.ndarray
    ST: np.ndarray
    variance: float
    estimator: str
    ci_level: float = 0.95
    S1_low: np.ndarray | None = None
    S1_high: np.ndarray | None = None
    ST_low: np.ndarray | None = None
    ST_high: np.ndarray | None = None
    S2: Dict[str, float] = field(default_factory=dict)
    S2_low: Dict[str, float] = field(default_factory=dict)
    S2_high: Dict[str, float] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.names = list(self.names)
        self.S1 = np.asarray(self.S1, dtype=float).ravel()
        self.ST = np.asarray(self.ST, dtype=float).ravel()
        if self.S1.size != len(self.names) or self.ST.size != len(self.names):
            raise ValueError("S1/ST must have one entry per name")

    # -- summary statistics ----------------------------------------------
    @property
    def sum_S1(self) -> float:
        return float(np.sum(self.S1))

    @property
    def sum_ST(self) -> float:
        return float(np.sum(self.ST))

    def interaction_share(self) -> float:
        """``sum(ST) - sum(S1)``: variance attributable to interactions (x2 mean)."""
        return float(self.sum_ST - self.sum_S1)

    def rank_by_total(self) -> list[tuple[str, float]]:
        order = np.argsort(-self.ST)
        return [(self.names[i], float(self.ST[i])) for i in order]

    # -- serialisation ----------------------------------------------------
    def as_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "estimator": self.estimator,
            "ci_level": float(self.ci_level),
            "variance": float(self.variance),
            "names": list(self.names),
            "S1": [float(v) for v in self.S1],
            "ST": [float(v) for v in self.ST],
            "sum_S1": self.sum_S1,
            "sum_ST": self.sum_ST,
            "interaction_share": self.interaction_share(),
        }
        if self.S1_low is not None and self.S1_high is not None:
            out["S1_ci_low"] = [float(v) for v in self.S1_low]
            out["S1_ci_high"] = [float(v) for v in self.S1_high]
        if self.ST_low is not None and self.ST_high is not None:
            out["ST_ci_low"] = [float(v) for v in self.ST_low]
            out["ST_ci_high"] = [float(v) for v in self.ST_high]
        if self.S2:
            out["S2"] = {k: float(v) for k, v in self.S2.items()}
            out["S2_ci_low"] = {k: float(v) for k, v in self.S2_low.items()}
            out["S2_ci_high"] = {k: float(v) for k, v in self.S2_high.items()}
        if self.meta:
            out["meta"] = dict(self.meta)
        return out

    def table(self) -> str:
        """Fixed-width table for reports/logs."""
        head = f"{'parameter':<24}{'S1':>10}{'ST':>10}{'95% CI(S1)':>24}"
        lines = [head, "-" * len(head)]
        for i, name in enumerate(self.names):
            ci = ""
            if self.S1_low is not None and self.S1_high is not None:
                ci = f"[{self.S1_low[i]:+.4f}, {self.S1_high[i]:+.4f}]"
            lines.append(f"{name:<24}{self.S1[i]:>10.5f}{self.ST[i]:>10.5f}{ci:>24}")
        lines.append(f"{'sum':<24}{self.sum_S1:>10.5f}{self.sum_ST:>10.5f}")
        return "\n".join(lines)


@dataclass
class IndexComparison:
    """Pairwise agreement between two index estimates."""

    max_abs_dev_S1: float
    max_rel_dev_S1: float
    max_abs_dev_ST: float
    max_rel_dev_ST: float
    #: relative deviations restricted to entries whose reference magnitude is
    #: above ``abs_floor`` -- the only ones for which a relative number is
    #: meaningful (an analytically zero index has no relative error).
    max_rel_dev_S1_defined: float = float("nan")
    max_rel_dev_ST_defined: float = float("nan")
    n_defined: int = 0
    detail: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "max_abs_dev_S1": self.max_abs_dev_S1,
            "max_rel_dev_S1": self.max_rel_dev_S1,
            "max_abs_dev_ST": self.max_abs_dev_ST,
            "max_rel_dev_ST": self.max_rel_dev_ST,
            "max_rel_dev_S1_defined": self.max_rel_dev_S1_defined,
            "max_rel_dev_ST_defined": self.max_rel_dev_ST_defined,
            "n_defined": int(self.n_defined),
            "detail": dict(self.detail),
        }


def _rel_dev(a: float, b: float, floor: float) -> float:
    denom = max(abs(b), floor)
    return float(abs(a - b) / denom)


def compare_indices(
    reference: SobolIndices,
    candidate: SobolIndices,
    *,
    abs_floor: float = 1e-3,
) -> IndexComparison:
    """Compare two index sets entry-wise (``candidate`` against ``reference``).

    Relative deviations use ``max(|reference|, abs_floor)`` as denominator so
    that an index that is analytically zero (e.g. ``S1`` of a pure interaction
    input) yields a finite, meaningful number instead of ``inf``.
    """
    if list(reference.names) != list(candidate.names):
        raise ValueError("index sets must be over the same parameters")
    dS1 = np.abs(np.asarray(candidate.S1) - np.asarray(reference.S1))
    dST = np.abs(np.asarray(candidate.ST) - np.asarray(reference.ST))
    rS1 = [_rel_dev(float(a), float(b), abs_floor) for a, b in zip(candidate.S1, reference.S1)]
    rST = [_rel_dev(float(a), float(b), abs_floor) for a, b in zip(candidate.ST, reference.ST)]
    detail = {
        "per_parameter": [
            {
                "name": name,
                "S1_ref": float(reference.S1[i]),
                "S1_cand": float(candidate.S1[i]),
                "S1_abs_dev": float(dS1[i]),
                "S1_rel_dev": rS1[i],
                "ST_ref": float(reference.ST[i]),
                "ST_cand": float(candidate.ST[i]),
                "ST_abs_dev": float(dST[i]),
                "ST_rel_dev": rST[i],
            }
            for i, name in enumerate(reference.names)
        ],
        "abs_floor": float(abs_floor),
    }
    defined_s1 = [r for r, ref in zip(rS1, reference.S1) if abs(float(ref)) > abs_floor]
    defined_st = [r for r, ref in zip(rST, reference.ST) if abs(float(ref)) > abs_floor]
    return IndexComparison(
        max_abs_dev_S1=float(dS1.max()),
        max_rel_dev_S1=float(max(rS1)),
        max_abs_dev_ST=float(dST.max()),
        max_rel_dev_ST=float(max(rST)),
        max_rel_dev_S1_defined=float(max(defined_s1)) if defined_s1 else float("nan"),
        max_rel_dev_ST_defined=float(max(defined_st)) if defined_st else float("nan"),
        n_defined=len(defined_st),
        detail=detail,
    )