"""Cross-solver agreement gate: mutual consistency of a solver ensemble.

The V&V machine gate has two halves.  :mod:`tensorphysics.vv.validation`
judges *one* solver against a *reference* (experiment or analytic solution);
:mod:`tensorphysics.vv.gci` judges *one* solver's own grid convergence.  This
module answers the third question, the one that has no reference at all:

    *N solvers each report a value for the same observable on the same
    problem.  Do they agree, and what consensus value -- with an uncertainty
    -- may be published?*

It is the cross-solver complement of the single-solver grid-convergence gate:
no experiment, no analytic solution, only the mutual consistency of the
ensemble.  A solver that disagrees with the ensemble is flagged as an
outlier; the consensus value and its uncertainty are what may be published.
A FAIL is never overwritten by a nicer number.

Conventions
-----------
``samples`` is a collection of :class:`SolverSample`, one per solver.  Each
sample carries the solver name, the reported value ``f``, and an optional
1-sigma uncertainty ``u`` (normally that solver's own finest-grid GCI).

The **consensus** is the inverse-variance weighted mean

    f_c = sum_i w_i f_i / sum_i w_i,      w_i = 1 / u_i**2

with uncertainty ``u_c = 1 / sqrt(sum_i w_i)``.  When no sample declares a
positive uncertainty the consensus is the plain arithmetic mean and its
uncertainty is left ``None`` (the gate then relies on the spread band alone).
A study that mixes declared and undeclared uncertainties is rejected: the
caller must be explicit about which regime it is in.

The **agreement band** is the largest pairwise disagreement

    spread     = max_{i<j} |f_i - f_j|
    spread_rel = spread / |f_c|

and the gate is deliberately conservative: a band wider than
``fail_spread`` (or a solver more than ``outlier_sigma`` of its own standard
deviations from the consensus) is a hard FAIL, a band wider than
``warn_spread`` is a WARN, otherwise PASS.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .gci import FAIL, PASS, WARN

__all__ = [
    "PASS",
    "WARN",
    "FAIL",
    "AgreementPolicy",
    "AgreementResult",
    "SolverSample",
    "agreement_spread",
    "consensus",
    "cross_solver_agreement",
]


# ---------------------------------------------------------------------------
# data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SolverSample:
    """One solver's result for one observable on one problem.

    ``u`` is the solver's own 1-sigma uncertainty on ``f`` (normally its
    finest-grid GCI).  ``h`` is optional context (grid size) and is carried
    for provenance but not used by the gate.
    """

    solver: str
    f: float
    u: float | None = None
    h: float | None = None
    label: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.solver, str) or not self.solver.strip():
            raise ValueError(f"solver name must be a non-empty string, got {self.solver!r}")
        if not isinstance(self.f, (int, float)) or not math.isfinite(self.f):
            raise ValueError(f"f must be finite, got {self.f!r}")
        if self.u is not None and (not math.isfinite(self.u) or self.u < 0.0):
            raise ValueError(f"u must be finite and non-negative, got {self.u!r}")
        if self.h is not None and (not math.isfinite(self.h) or self.h <= 0.0):
            raise ValueError(f"h must be finite and positive, got {self.h!r}")

    def declares_uncertainty(self) -> bool:
        return self.u is not None and self.u > 0.0

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"solver": self.solver, "f": self.f}
        if self.u is not None:
            out["u"] = self.u
        if self.h is not None:
            out["h"] = self.h
        if self.label:
            out["label"] = self.label
        return out

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SolverSample":
        diff: dict[str, Any] = {}
        solver = payload.get("solver", payload.get("name", payload.get("method")))
        if solver is not None:
            diff["solver"] = str(solver)
        f = payload.get("f", payload.get("value", payload.get("metric")))
        if f is not None:
            diff["f"] = f
        for key in ("u", "h", "label"):
            if payload.get(key) is not None:
                diff[key] = payload[key]
        if "solver" not in diff:
            raise ValueError(f"sample needs a solver name, got keys {sorted(payload)}")
        if "f" not in diff:
            raise ValueError(f"sample needs a value, got keys {sorted(payload)}")
        return cls(**diff)


@dataclass(frozen=True)
class AgreementPolicy:
    """Thresholds of the cross-solver agreement gate.

    ``warn_spread`` / ``fail_spread`` are bands on the largest pairwise
    disagreement: relative (a fraction of ``|f_c|``) by default, absolute
    when ``relative`` is False or the consensus is zero.  ``outlier_sigma``
    flags a solver whose value lies more than that many of its own standard
    deviations from the consensus.
    """

    warn_spread: float = 0.02
    fail_spread: float = 0.05
    outlier_sigma: float = 3.0
    relative: bool = True

    def __post_init__(self) -> None:
        for name in ("warn_spread", "fail_spread"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
        if self.fail_spread < self.warn_spread:
            raise ValueError(
                f"fail_spread ({self.fail_spread}) must be >= warn_spread ({self.warn_spread})"
            )
        if not math.isfinite(self.outlier_sigma) or self.outlier_sigma <= 0.0:
            raise ValueError(f"outlier_sigma must be finite and positive, got {self.outlier_sigma!r}")

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass(frozen=True)
class AgreementResult:
    """Machine verdict of the cross-solver agreement gate."""

    status: str
    quantity: str
    n_solvers: int
    solvers: tuple[str, ...]
    consensus: float
    consensus_u: float | None
    spread: float
    spread_rel: float | None
    widest_pair: tuple[str, str]
    deviations: tuple[dict[str, Any], ...]
    outliers: tuple[str, ...]
    checks: tuple[dict[str, Any], ...]
    policy: AgreementPolicy = field(default_factory=AgreementPolicy)

    @property
    def reasons(self) -> tuple[str, ...]:
        return tuple(c["detail"] for c in self.checks if c["status"] != PASS)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "tensorphysics.vv.cross_solver/1",
            "status": self.status,
            "quantity": self.quantity,
            "n_solvers": self.n_solvers,
            "solvers": list(self.solvers),
            "consensus": self.consensus,
            "consensus_u": self.consensus_u,
            "spread": self.spread,
            "spread_rel": self.spread_rel,
            "widest_pair": list(self.widest_pair),
            "deviations": list(self.deviations),
            "outliers": list(self.outliers),
            "checks": list(self.checks),
            "reasons": list(self.reasons),
            "policy": self.policy.as_dict(),
        }


# ---------------------------------------------------------------------------
# numerical helpers
# ---------------------------------------------------------------------------


def _validated_samples(samples: Iterable[SolverSample]) -> tuple[SolverSample, ...]:
    ordered = tuple(samples)
    if len(ordered) < 2:
        raise ValueError("a cross-solver comparison needs at least two solvers")
    names = [s.solver for s in ordered]
    if len(set(names)) != len(names):
        raise ValueError(f"solver names must be unique, got {names}")
    declaring = [s for s in ordered if s.declares_uncertainty()]
    if declaring and len(declaring) != len(ordered):
        raise ValueError(
            "either every sample declares a positive uncertainty or none do; "
            f"declaring={sorted(s.solver for s in declaring)}, "
            f"silent={sorted(s.solver for s in ordered if not s.declares_uncertainty())}"
        )
    return ordered


def consensus(samples: Iterable[SolverSample]) -> tuple[float, float | None]:
    """Inverse-variance weighted mean and its 1-sigma uncertainty.

    Returns ``(f_c, u_c)``.  ``u_c`` is ``None`` when no sample declares a
    positive uncertainty (plain mean, no uncertainty attached).
    """
    ordered = _validated_samples(samples)
    if all(s.declares_uncertainty() for s in ordered):
        weights = [1.0 / (s.u * s.u) for s in ordered]
        total = math.fsum(weights)
        f_c = math.fsum(w * s.f for w, s in zip(weights, ordered)) / total
        u_c = 1.0 / math.sqrt(total)
        return f_c, u_c
    # no sample declares a positive uncertainty: plain mean, no uncertainty
    f_c = math.fsum(s.f for s in ordered) / len(ordered)
    return f_c, None


def agreement_spread(
    samples: Iterable[SolverSample],
) -> tuple[float, float | None, tuple[str, str]]:
    """Largest pairwise disagreement.

    Returns ``(spread, spread_rel, widest_pair)`` where ``spread`` is the
    maximum absolute pairwise difference, ``spread_rel`` is that divided by
    ``|f_c|`` (``None`` when the consensus is zero), and ``widest_pair`` names
    the two solvers achieving the maximum.
    """
    ordered = _validated_samples(samples)
    f_c, _ = consensus(ordered)
    spread = 0.0
    widest = (ordered[0].solver, ordered[1].solver)
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            d = abs(ordered[i].f - ordered[j].f)
            if d > spread:
                spread = d
                widest = (ordered[i].solver, ordered[j].solver)
    spread_rel = spread / abs(f_c) if f_c != 0.0 else None
    return spread, spread_rel, widest


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------


def cross_solver_agreement(
    samples: Iterable[SolverSample],
    *,
    quantity: str = "",
    policy: AgreementPolicy | None = None,
) -> AgreementResult:
    """Judge the mutual consistency of a set of solver results.

    ``samples`` is one :class:`SolverSample` per solver.  The gate reports the
    consensus value and its uncertainty, the agreement band, and a
    PASS / WARN / FAIL verdict with a per-check account.
    """
    ordered = _validated_samples(samples)
    pol = policy if policy is not None else AgreementPolicy()
    f_c, u_c = consensus(ordered)
    spread, spread_rel, widest = agreement_spread(ordered)

    band = spread_rel if (pol.relative and spread_rel is not None) else spread
    band_kind = "relative" if (pol.relative and spread_rel is not None) else "absolute"

    checks: list[dict[str, Any]] = []
    if band > pol.fail_spread:
        spread_status = FAIL
    elif band > pol.warn_spread:
        spread_status = WARN
    else:
        spread_status = PASS
    checks.append(
        {
            "check": "spread",
            "status": spread_status,
            "band": band,
            "band_kind": band_kind,
            "warn": pol.warn_spread,
            "fail": pol.fail_spread,
            "detail": (
                f"widest pair {widest[0]}/{widest[1]} disagrees by {band:.6g} "
                f"({band_kind}) against warn={pol.warn_spread:g} fail={pol.fail_spread:g}"
            ),
        }
    )

    deviations: list[dict[str, Any]] = []
    outliers: list[str] = []
    has_u = all(s.declares_uncertainty() for s in ordered)
    for s in ordered:
        dev = abs(s.f - f_c)
        dev_sigma = dev / s.u if (has_u and s.u > 0.0) else None
        entry: dict[str, Any] = {
            "solver": s.solver,
            "f": s.f,
            "deviation": dev,
            "deviation_sigma": dev_sigma,
        }
        deviations.append(entry)
        if dev_sigma is not None and dev_sigma > pol.outlier_sigma:
            outliers.append(s.solver)
    outlier_status = FAIL if outliers else PASS
    checks.append(
        {
            "check": "outlier",
            "status": outlier_status,
            "outlier_sigma": pol.outlier_sigma,
            "detail": (
                "no solver exceeds "
                f"{pol.outlier_sigma:g} sigma of its own uncertainty"
                if not outliers
                else f"outliers beyond {pol.outlier_sigma:g} sigma: {', '.join(outliers)}"
            ),
        }
    )

    if FAIL in (c["status"] for c in checks):
        status = FAIL
    elif WARN in (c["status"] for c in checks):
        status = WARN
    else:
        status = PASS

    return AgreementResult(
        status=status,
        quantity=quantity,
        n_solvers=len(ordered),
        solvers=tuple(s.solver for s in ordered),
        consensus=f_c,
        consensus_u=u_c,
        spread=spread,
        spread_rel=spread_rel,
        widest_pair=widest,
        deviations=tuple(deviations),
        outliers=tuple(outliers),
        checks=tuple(checks),
        policy=pol,
    )
