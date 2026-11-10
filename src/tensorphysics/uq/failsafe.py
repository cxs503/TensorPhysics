"""Fail-safe degradation: certify with polynomial chaos **or** quasi-Monte-Carlo.

Why this module exists
----------------------
A chaos surrogate is cheap and gives *analytic* Sobol' indices, but it is only
admissible when the surrogate actually represents the solver: too few solver
runs, a discontinuous QoI, or a badly chosen truncation all produce a
confident-looking expansion with a poor leave-one-out score.  Silently shipping
those indices is the failure mode this gate exists to prevent.

The policy is therefore three-state and explicit:

``PCE``
    ``Q^2 >= q2_min`` (leave-one-out) **and** the fit converged **and** there
    are enough samples.  The PCE indices are primary; the Saltelli estimate (when
    requested) stays the independent cross-check.
``MONTECARLO``
    Any of the above fails.  The quasi-Monte-Carlo (scrambled Sobol' sequence,
    Saltelli/Jansen pick-freeze) estimate is *promoted to primary*, the output
    band is estimated by direct solver sampling, and the verdict carries the
    non-blocking note ``UQ-N05-MC-FAILSAFE`` plus the reason the shift happened.

``Q^2`` is the leave-one-out coefficient of determination of the surrogate,

    Q^2 = 1 - eps_LOO^2,

because :class:`~tensorphysics.uq.pce.PCESurrogate` stores ``eps_loo`` as the
LOO residual RMS normalised by ``std(y)``; this is exactly
``1 - SSE_LOO / SST``.

Design notes
------------
* The decision is **predeclared**: ``q2_min`` (default 0.5) and the sample
  floor are inputs, fixed before the fit is looked at.
* Degrading does **not** upgrade evidence: the sampling route is honest about
  its own cost (:class:`~tensorphysics.uq.sobol_mc.SobolMCResult`
  ``n_model_evaluations``) and carries the Monte-Carlo noise diagnostic
  ``UQ-N02-NOISY-MC`` exactly as before.
* The wrapper never mutates the study returned by
  :func:`~tensorphysics.uq.workflow.run_uq_study`; it returns a
  :class:`FailsafeStudy` that references it, so the plain-workflow behaviour is
  unchanged (existing callers see no difference).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .pce import PCESurrogate, SobolIndices
from .sample_spec import SampleSpec
from .sampler import draw_unit
from .sobol_mc import SobolMCResult, saltelli_indices
from .verdict import (
    FAIL,
    PASS,
    DEFAULT_EPS_LOO_MAX,
    WITHHELD,
    OutputBudget,
    UQVerdict,
    decide,
    min_samples_floor,
)
from .workflow import UQStudy, run_uq_study

__all__ = [
    "FailsafeDecision",
    "FailsafeStudy",
    "MODE_MONTECARLO",
    "MODE_PCE",
    "assess",
    "direct_budget",
    "indices_from_mc",
    "loo_q2",
    "run_with_failsafe",
]

MODE_PCE = "pce"
MODE_MONTECARLO = "montecarlo"

#: Default quality floor: Q^2 below this means the surrogate explains less than
#: half of the output variance and must not be used to certify indices.
DEFAULT_Q2_MIN = 0.5


def loo_q2(surrogate: PCESurrogate | None) -> float | None:
    """Leave-one-out coefficient of determination ``Q^2 = 1 - eps_LOO^2``."""
    if surrogate is None:
        return None
    q2 = 1.0 - float(surrogate.eps_loo) ** 2
    return q2 if math.isfinite(q2) else None


@dataclass(frozen=True)
class FailsafeDecision:
    """Predeclared mode decision for one study."""

    mode: str
    triggered: bool
    q2: float | None
    q2_min: float
    n_valid: int
    n_samples: int
    n_terms: int | None
    q2_threshold_met: bool
    reasons: tuple[str, ...]
    fit_error: str | None = None

    @property
    def degraded(self) -> bool:
        return self.mode == MODE_MONTECARLO

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "triggered": bool(self.triggered),
            "q2": None if self.q2 is None else float(self.q2),
            "q2_min": float(self.q2_min),
            "q2_threshold_met": bool(self.q2_threshold_met),
            "n_valid": int(self.n_valid),
            "n_samples": int(self.n_samples),
            "n_terms": None if self.n_terms is None else int(self.n_terms),
            "reasons": list(self.reasons),
            "fit_error": self.fit_error,
        }


def assess(
    surrogate: PCESurrogate | None,
    *,
    n_valid: int,
    n_samples: int,
    q2_min: float = DEFAULT_Q2_MIN,
    fit_error: str | None = None,
    min_samples: int | None = None,
    dim: int | None = None,
) -> FailsafeDecision:
    """Decide ``PCE`` vs ``MONTECARLO`` **before** any index is reported."""
    if not (0.0 < q2_min < 1.0) or not math.isfinite(q2_min):
        raise ValueError(f"q2_min must be in (0, 1), got {q2_min!r}")
    reasons: list[str] = []
    q2 = loo_q2(surrogate)
    if surrogate is None:
        reasons.append("PCE-FIT-FAILED")
        if fit_error:
            reasons.append(f"PCE-FIT-ERROR: {fit_error}")
    elif q2 is None or q2 < q2_min:
        reasons.append("PCE-Q2-BELOW-FLOOR")
    floor = min_samples if min_samples is not None else min_samples_floor(
        dim if dim is not None else (surrogate.dim if surrogate is not None else 1)
    )
    if n_valid < floor:
        reasons.append("PCE-TOOFEW-SAMPLES")
    triggered = bool(reasons)
    return FailsafeDecision(
        mode=MODE_MONTECARLO if triggered else MODE_PCE,
        triggered=triggered,
        q2=q2,
        q2_min=float(q2_min),
        n_valid=int(n_valid),
        n_samples=int(n_samples),
        n_terms=None if surrogate is None else int(surrogate.n_terms),
        q2_threshold_met=bool(q2 is not None and q2 >= q2_min),
        reasons=tuple(reasons),
        fit_error=fit_error,
    )


def indices_from_mc(mc: SobolMCResult) -> SobolIndices:
    """Promote a pick-freeze estimate to the primary index set."""
    return SobolIndices(
        first=mc.first,
        total=mc.total,
        second=dict(mc.second),
        variance=float(mc.variance),
        mean=float(mc.mean),
        first_ci=mc.first_ci,
        total_ci=mc.total_ci,
    )


def direct_budget(
    solver: Any,
    spec: SampleSpec,
    *,
    n_mc: int = 4096,
    seed: int = 20240,
) -> OutputBudget:
    """95 % prediction interval of the QoI by **direct** quasi-Monte-Carlo.

    No surrogate is involved: the band is measured on solver output, which is
    why it remains admissible when the chaos route has failed.  The price is
    ``n_mc`` solver runs.
    """
    if n_mc < 32:
        raise ValueError(f"n_mc must be >= 32, got {n_mc}")
    unit = draw_unit(n_mc, spec.dim, rule="sobol", seed=seed)
    points, _ = spec.transform(unit)
    values = np.asarray(solver.evaluate(points), dtype=float).reshape(-1)
    if values.size != n_mc:
        raise ValueError(f"solver returned {values.size} values, expected {n_mc}")
    if not np.all(np.isfinite(values)):
        raise ValueError("solver returned non-finite values during the direct budget")
    mean = float(np.mean(values))
    lo, hi = np.percentile(values, [2.5, 97.5])
    width = float(hi - lo)
    return OutputBudget(
        mean=mean,
        sd=float(np.std(values, ddof=1)),
        lower95=float(lo),
        upper95=float(hi),
        width=width,
        relative_width=width / abs(mean) if mean != 0.0 else float("inf"),
        n_mc=int(values.size),
        definition="prediction_interval_direct_solver_failsafe",
    )


@dataclass(frozen=True, eq=False)
class FailsafeStudy:
    """A study plus the predeclared route decision that produced its indices."""

    decision: FailsafeDecision
    study: UQStudy
    indices: SobolIndices | None
    mc: SobolMCResult | None
    budget: OutputBudget | None
    verdict: UQVerdict
    metadata: dict[str, Any]

    @property
    def mode(self) -> str:
        return self.decision.mode

    @property
    def status(self) -> str:
        return self.verdict.status

    def as_dict(self) -> dict[str, Any]:
        names = self.study.spec.names
        out = {
            "mode": self.mode,
            "status": self.status,
            "decision": self.decision.as_dict(),
            "study": self.study.as_dict(),
            "verdict": self.verdict.as_dict(),
            "metadata": self.metadata,
        }
        if self.indices is not None:
            out["indices"] = self.indices.as_dict(names=names)
        if self.mc is not None:
            out["mc"] = self.mc.as_dict(names=names)
        if self.budget is not None:
            out["budget_primary"] = self.budget.as_dict()
        return out


def run_with_failsafe(
    solver: Any,
    spec: SampleSpec,
    *,
    n_samples: int,
    degree: int,
    method: str = "lars",
    seed: int = 20240,
    rule: str = "sobol",
    q2_min: float = DEFAULT_Q2_MIN,
    min_samples: int | None = None,
    failsafe_n_base: int | None = None,
    failsafe_budget_n_mc: int = 4096,
    threshold: float = 0.15,
    eps_loo_max: float | None = None,
    shards: int = 1,
    runner: Callable[..., Any] | None = None,
    **study_kwargs: Any,
) -> FailsafeStudy:
    """Run a study, then certify with PCE or degrade to quasi-Monte-Carlo.

    Parameters beyond those of :func:`~tensorphysics.uq.workflow.run_uq_study`
    ---------------------------------------------------------------------
    q2_min:
        Predeclared quality floor for the surrogate (default 0.5).
    failsafe_n_base:
        Base sample size of the pick-freeze design used when the PCE route is
        rejected (default ``max(1 << 12, 64 * dim)``).
    failsafe_budget_n_mc:
        Direct solver samples used for the output band in degraded mode.

    The PCE training design is always re-used; degradation is decided from the
    surrogate that was actually fitted, not from a pilot fit.
    """
    plain = run_uq_study(
        solver,
        spec,
        n_samples=n_samples,
        degree=degree,
        method=method,
        seed=seed,
        rule=rule,
        threshold=threshold,
        shards=shards,
        runner=runner,
        run_mc=bool(study_kwargs.pop("run_mc", True)),
        **study_kwargs,
    )
    n_valid = int(plain.results.n_valid)
    decision = assess(
        plain.surrogate,
        n_valid=n_valid,
        n_samples=int(n_samples),
        q2_min=q2_min,
        fit_error=plain.metadata.get("fit_error"),
        min_samples=min_samples,
        dim=spec.dim,
    )
    names = plain.spec.names
    if not decision.triggered:
        return FailsafeStudy(
            decision=decision,
            study=plain,
            indices=plain.pce_sobol,
            mc=plain.mc,
            budget=plain.budget,
            verdict=plain.verdict,
            metadata={
                "route": MODE_PCE,
                "pce_quality": decision.as_dict(),
                "workflow_version": "tpuq-failsafe-v1",
            },
        )

    adapter = solver if hasattr(solver, "evaluate") else None
    if adapter is None:
        from .adapter import CallableAdapter

        adapter = CallableAdapter(solver, names=names)

    def _evaluate_mc(unit_points: np.ndarray) -> np.ndarray:
        phys, _ = plain.spec.transform(unit_points)
        out = np.asarray(adapter.evaluate(phys), dtype=float).reshape(-1)
        if not np.all(np.isfinite(out)):
            raise ValueError("solver returned non-finite values in the fail-safe MC design")
        return out

    n_base = int(failsafe_n_base or max(1 << 12, 64 * plain.spec.dim))
    mc = saltelli_indices(
        _evaluate_mc,
        plain.spec.dim,
        n_base=n_base,
        seed=seed + 17,
        rule="sobol",
        second_order=True,
    )
    budget = direct_budget(
        adapter,
        plain.spec,
        n_mc=int(failsafe_budget_n_mc),
        seed=seed + 29,
    )
    # ``decide`` attaches the UQ-N05-MC-FAILSAFE note itself when the band is
    # certified directly; the machine-readable reason for the shift lives in
    # ``decision.reasons`` and in ``details`` below.
    verdict = decide(
        budget=budget,
        threshold=threshold,
        surrogate=None,
        mc=mc,
        cross_check=None,
        samples=plain.samples,
        results=plain.results,
        eps_loo_max=DEFAULT_EPS_LOO_MAX if eps_loo_max is None else eps_loo_max,
        # in degraded mode the sample-sufficiency criterion is the MC design
        # size (reported below), not the PCE training count
        min_samples=0,
        require_cross_check=False,
        budget_certified_directly=True,
        details={
            "route": MODE_MONTECARLO,
            "failsafe_reasons": list(decision.reasons),
            "n_base_mc": n_base,
            "n_mc_evaluations": int(mc.n_model_evaluations),
            "n_direct_budget": int(failsafe_budget_n_mc),
        },
    )
    return FailsafeStudy(
        decision=decision,
        study=plain,
        indices=indices_from_mc(mc),
        mc=mc,
        budget=budget,
        verdict=verdict,
        metadata={
            "route": MODE_MONTECARLO,
            "pce_quality": decision.as_dict(),
            "n_base_mc": n_base,
            "n_mc_evaluations": int(mc.n_model_evaluations),
            "n_direct_budget": int(failsafe_budget_n_mc),
            "workflow_version": "tpuq-failsafe-v1",
            "statuses": {"pce_route_would_be": plain.status, "primary": verdict.status},
            "status_constants": [PASS, FAIL, WITHHELD],
        },
    )