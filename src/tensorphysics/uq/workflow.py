"""End-to-end UQ workflow: sample -> solver fan-out -> chaos -> Sobol -> verdict.

``run_uq_study`` is the single entry point that the rest of the ecosystem
calls.  It only talks to the solver through a black-box adapter, so *no
solver kernel is imported or modified*: the workflow is a consumer of the
``points -> values`` contract (see :mod:`tensorphysics.uq.adapter`).

The evidence chain is pinned at every step:

``spec fingerprint`` -> ``sample-set hash`` -> ``solver values`` ->
``surrogate (LOO)`` -> ``PCE Sobol indices`` + ``Saltelli Sobol indices`` ->
``cross-check`` -> ``95% output interval`` -> ``PASS/FAIL/WITHHELD``.

Nothing in the chain is inferred from a previous step's *name*: the hash
travels with the results and the verdict re-checks it, so a study cannot be
silently re-used with a different design.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .adapter import CallableAdapter, SampleResults, SolverAdapter, fanout_evaluate
from .pce import PCESurrogate, SobolIndices, bootstrap_sobol, fit_pce
from .sample_spec import SampleSpec
from .sampler import SampleSet, draw_unit, sample
from .sobol_mc import SobolMCResult, saltelli_indices
from .verdict import (
    DEFAULT_EPS_LOO_MAX,
    DEFAULT_THRESHOLD,
    CrossCheck,
    OutputBudget,
    Reason,
    UQVerdict,
    cross_check_indices,
    decide,
    output_budget,
)

__all__ = ["UQStudy", "run_uq_study"]


def _as_adapter(adapter: Any, spec: SampleSpec) -> SolverAdapter:
    if hasattr(adapter, "evaluate") and callable(getattr(adapter, "evaluate")):
        return adapter  # type: ignore[return-value]
    if callable(adapter):
        return CallableAdapter(adapter, names=spec.names)
    raise TypeError(
        f"adapter must be a SolverAdapter or a callable, got {type(adapter)!r}"
    )


@dataclass(frozen=True, eq=False)
class UQStudy:
    """Complete, JSON-serialisable record of one UQ study."""

    spec: SampleSpec
    samples: SampleSet
    results: SampleResults = field(repr=False)
    surrogate: PCESurrogate | None = field(repr=False)
    pce_sobol: SobolIndices | None = field(repr=False)
    mc: SobolMCResult | None = field(repr=False)
    cross_check: CrossCheck | None
    budget: OutputBudget | None
    verdict: UQVerdict
    budget_direct: OutputBudget | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        return self.verdict.status

    def as_dict(self) -> dict[str, Any]:
        names = self.spec.names
        out: dict[str, Any] = {
            "status": self.status,
            "spec": self.spec.as_dict(),
            "spec_hash": self.spec.fingerprint(),
            "samples": self.samples.as_dict(),
            "results": self.results.as_dict(),
            "verdict": self.verdict.as_dict(),
            "metadata": dict(self.metadata),
        }
        if self.surrogate is not None:
            out["surrogate"] = self.surrogate.as_dict(names)
        if self.pce_sobol is not None:
            out["pce_sobol"] = self.pce_sobol.as_dict(names)
        if self.mc is not None:
            out["mc_sobol"] = self.mc.as_dict(names)
        if self.cross_check is not None:
            out["cross_check"] = self.cross_check.as_dict()
        if self.budget is not None:
            out["budget"] = self.budget.as_dict()
        if self.budget_direct is not None:
            out["budget_direct"] = self.budget_direct.as_dict()
        return out


def run_uq_study(
    adapter: SolverAdapter | Callable[[np.ndarray], np.ndarray],
    spec: SampleSpec,
    *,
    n_samples: int = 1024,
    degree: int = 8,
    method: str = "lars",
    max_terms: int | None = None,
    rule: str = "sobol",
    seed: int = 20240,
    n_boot: int = 200,
    bootstrap_seed: int | None = None,
    run_mc: bool = True,
    n_base_mc: int = 1 << 16,
    mc_seed: int | None = None,
    bootstrap_mc: int = 100,
    second_order: bool = True,
    budget_n_mc: int = 20000,
    budget_seed: int = 9091,
    direct_budget_n_mc: int = 0,
    threshold: float = DEFAULT_THRESHOLD,
    eps_loo_max: float = DEFAULT_EPS_LOO_MAX,
    min_samples: int | None = None,
    min_samples_per_term: float = 4.0,
    cross_check_rel_tol: float = 0.05,
    cross_check_rel_tol_second: float = 0.25,
    cross_check_floor: float = 0.02,
    require_cross_check: bool = True,
    shards: int = 1,
    runner: Callable[[SolverAdapter, np.ndarray, np.ndarray], Any] | None = None,
) -> UQStudy:
    """Run the full non-intrusive UQ pipeline and adjudicate it.

    Parameters
    ----------
    adapter:
        Black-box solver (``points -> values``) or a :class:`SolverAdapter`.
    spec:
        Uncertain-input specification.
    n_samples, degree, method, max_terms:
        Polynomial-chaos budget: number of solver runs, total polynomial
        degree, fitting method (``"lars"`` / ``"ols"``) and the cap on active
        terms for the greedy selector.
    rule, seed:
        Sampling rule and seed of the training design (both enter the hash).
    n_boot:
        Bootstrap resamples for the confidence interval of the PCE indices.
    run_mc, n_base_mc, mc_seed, bootstrap_mc, second_order:
        Monte-Carlo cross-check (independent second opinion).  Skipping it
        (``run_mc=False``) while ``require_cross_check`` stays True withholds the
        verdict with ``UQ-W07-NOMC`` and records the ``UQ-N04-NO-CROSSCHECK``
        note: a single-method study cannot certify itself.
    budget_n_mc, budget_seed:
        Size and seed of the fresh draw used to propagate the surrogate into
        the 95% output interval.
    direct_budget_n_mc:
        When positive, also evaluate the *solver* on that many fresh points to
        obtain a direct Monte-Carlo interval as a cross-check of the surrogate
        interval (costs one solver block; 0 disables).
    threshold, eps_loo_max, min_samples, min_samples_per_term,
    cross_check_rel_tol, cross_check_rel_tol_second, cross_check_floor,
    require_cross_check:
        Acceptance-gate policy, see :mod:`tensorphysics.uq.verdict`.
    shards, runner:
        Solver fan-out (see :func:`tensorphysics.uq.adapter.fanout_evaluate`).
    """
    t_start = time.perf_counter()
    spec_checked = spec if isinstance(spec, SampleSpec) else None
    if spec_checked is None:
        raise TypeError(f"spec must be a SampleSpec, got {type(spec)!r}")
    solver = _as_adapter(adapter, spec_checked)
    design = sample(spec_checked, n_samples, rule=rule, seed=seed)
    t_sampled = time.perf_counter()
    results = fanout_evaluate(solver, design, shards=shards, runner=runner)
    t_solved = time.perf_counter()

    _, coords = spec_checked.from_latent(design.latent)
    values = results.values
    valid = np.isfinite(values)
    n_valid = int(np.count_nonzero(valid))

    surrogate: PCESurrogate | None = None
    pce_sobol: SobolIndices | None = None
    fit_error: str | None = None
    if n_valid >= 2:
        try:
            surrogate = fit_pce(
                coords[valid],
                values[valid],
                degree=degree,
                families=spec_checked.families,
                method=method,
                max_terms=max_terms,
            )
            if n_boot:
                pce_sobol = bootstrap_sobol(
                    surrogate,
                    coords[valid],
                    values[valid],
                    n_boot=n_boot,
                    seed=bootstrap_seed if bootstrap_seed is not None else seed + 11,
                )
            else:
                pce_sobol = surrogate.sobol()
        except (ValueError, np.linalg.LinAlgError) as exc:  # recorded, not hidden
            fit_error = f"{type(exc).__name__}: {exc}"
    t_fitted = time.perf_counter()

    mc: SobolMCResult | None = None
    mc_error: str | None = None
    if run_mc and n_valid >= 2:
        def _evaluate_mc(unit_points: np.ndarray) -> np.ndarray:
            # the pick-freeze design lives on the unit hypercube; map it through
            # the specification before touching the solver
            phys, _ = spec_checked.transform(unit_points)
            out = np.asarray(solver.evaluate(phys), dtype=float).reshape(-1)
            if not np.all(np.isfinite(out)):
                raise ValueError("solver returned non-finite values during the MC cross-check")
            return out

        try:
            mc = saltelli_indices(
                _evaluate_mc,
                spec_checked.dim,
                n_base=n_base_mc,
                seed=mc_seed if mc_seed is not None else seed + 3,
                second_order=second_order,
                bootstrap=bootstrap_mc,
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            mc_error = f"{type(exc).__name__}: {exc}"
    t_mc = time.perf_counter()

    cross: CrossCheck | None = None
    if pce_sobol is not None and mc is not None:
        cross = cross_check_indices(
            pce_sobol,
            mc,
            names=spec_checked.names,
            rel_tol=cross_check_rel_tol,
            floor=cross_check_floor,
            rel_tol_second=cross_check_rel_tol_second,
        )

    budget: OutputBudget | None = None
    if surrogate is not None:
        try:
            budget = output_budget(
                surrogate, spec_checked, n_mc=budget_n_mc, seed=budget_seed
            )
        except ValueError:
            budget = None

    budget_direct: OutputBudget | None = None
    if direct_budget_n_mc and n_valid >= 2:
        unit = draw_unit(direct_budget_n_mc, spec_checked.dim, rule="sobol", seed=budget_seed + 1)
        pts, _ = spec_checked.transform(unit)
        vals = np.asarray(solver.evaluate(pts), dtype=float).reshape(-1)
        if np.all(np.isfinite(vals)):
            mean = float(np.mean(vals))
            lo, hi = np.percentile(vals, [2.5, 97.5])
            width = float(hi - lo)
            budget_direct = OutputBudget(
                mean=mean,
                sd=float(np.std(vals, ddof=1)),
                lower95=float(lo),
                upper95=float(hi),
                width=width,
                relative_width=width / abs(mean) if mean != 0.0 else float("inf"),
                n_mc=int(vals.size),
                definition="prediction_interval_direct_solver",
            )

    details: dict[str, Any] = {
        "fit_error": fit_error,
        "mc_error": mc_error,
        "n_samples_requested": int(n_samples),
        "n_valid": n_valid,
        "degree": int(degree),
        "method": method,
        "rule": rule,
        "seed": int(seed),
        "shards": int(shards),
        "n_base_mc": int(n_base_mc) if run_mc else 0,
    }
    if budget_direct is not None and budget is not None:
        details["budget_surrogate_vs_direct_rel_diff"] = float(
            abs(budget.relative_width - budget_direct.relative_width)
        )
    extra_notes: list[Reason] = []
    if not run_mc:
        extra_notes.append(
            Reason(
                "UQ-N04-NO-CROSSCHECK",
                "Monte-Carlo cross-check explicitly disabled by the caller: the "
                "indices rest on the chaos surrogate alone",
                blocking=False,
            )
        )
    verdict = decide(
        budget=budget,
        threshold=threshold,
        surrogate=surrogate,
        mc=mc,
        cross_check=cross,
        samples=design,
        results=results,
        eps_loo_max=eps_loo_max,
        min_samples=min_samples,
        min_samples_per_term=min_samples_per_term,
        require_cross_check=require_cross_check,
        extra_notes=extra_notes,
        details=details,
    )
    t_end = time.perf_counter()
    metadata = {
        "timing_s": {
            "sample": t_sampled - t_start,
            "solve": t_solved - t_sampled,
            "fit": t_fitted - t_solved,
            "cross_check": t_mc - t_fitted,
            "total": t_end - t_start,
        },
        "n_model_evaluations": int(n_valid + (mc.n_model_evaluations if mc else 0)
                                    + (direct_budget_n_mc if budget_direct else 0)),
        "workflow_version": "tpuq-workflow-v1",
    }
    return UQStudy(
        spec=spec_checked,
        samples=design,
        results=results,
        surrogate=surrogate,
        pce_sobol=pce_sobol,
        mc=mc,
        cross_check=cross,
        budget=budget,
        verdict=verdict,
        budget_direct=budget_direct,
        metadata=metadata,
    )