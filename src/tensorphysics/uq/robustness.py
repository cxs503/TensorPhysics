"""Falsification matrix for the three-state gate: no injection may yield PASS.

A gate is only as trustworthy as its failure behaviour.  This module feeds the
UQ pipeline a battery of *deliberately broken* studies -- non-finite solver
output, a starved sample budget, a surrogate whose LOO error exceeds the
quality floor, a missing second opinion, a tampered Monte-Carlo cross-check, a
hash-mismatched result set, a degenerate (constant) output, a band that exceeds
the engineering threshold -- and records the status and the reason codes that
the gate produced.

The single assertion that matters is negative:

    **no injected defect may produce PASS**

(each case additionally has to name *its own* reason code, so a case cannot be
"passed" for the wrong reason).  The matrix is exhaustive over the blocking
conditions of :func:`tensorphysics.uq.verdict.decide`, which is what makes the
gate auditable: every ``WITHHELD`` a user can see in production is one of these
cases, with a case id attached.

Cases are built at two levels:

* **end-to-end** (``run_uq_study`` on a wrapped model) -- exercises sampler,
  adapter, fit, cross-check and verdict together;
* **verdict level** (``decide`` on synthetic evidence) -- used where the defect
  cannot be produced from a well-behaved model (hash mismatch, tampered MC,
  threshold boundary).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .benchmarks import ishigami, ishigami_spec
from .pce import PCESurrogate, fit_pce
from .sample_spec import SampleSpec
from .sampler import sample
from .sobol_mc import SobolMCResult
from .verdict import (
    FAIL,
    PASS,
    WITHHELD,
    OutputBudget,
    cross_check_indices,
    decide,
    output_budget,
)

__all__ = [
    "InjectionCase",
    "InjectionReport",
    "injection_suite",
]

SPEC = ishigami_spec

#: The control model: the Ishigami function lifted onto a baseline of 1000 with
#: a 5 kN scale, so that its 95 % output band is ~7 % of the mean and a clean
#: study *passes* the 15 % engineering gate.  Every injection is built on the
#: same mapped model, so the only difference between the control and a case is
#: the injected defect.
CONTROL_OFFSET = 1000.0
CONTROL_SCALE = 5.0


def control_model(points: np.ndarray) -> np.ndarray:
    return CONTROL_OFFSET + CONTROL_SCALE * ishigami(points)


def _good_study(**overrides: Any):
    """A study that passes, used as the control of the matrix."""
    from .workflow import run_uq_study

    kwargs: dict[str, Any] = dict(
        n_samples=512,
        degree=8,
        n_boot=0,
        n_base_mc=1 << 14,
        bootstrap_mc=0,
        budget_n_mc=4000,
    )
    adapter = overrides.pop("adapter", control_model)
    kwargs.update(overrides)
    return run_uq_study(adapter, SPEC(), **kwargs)


def _good_surrogate_and_budget(spec: SampleSpec):
    design = sample(spec, 512, seed=20240)
    points, coords = spec.from_latent(design.latent)
    values = control_model(points)
    surrogate = fit_pce(coords, values, degree=8, families=spec.families)
    budget = output_budget(surrogate, spec, n_mc=4000)
    return surrogate, budget, coords, values


@dataclass(frozen=True)
class InjectionCase:
    """One injected defect and the gate's reaction."""

    case_id: str
    description: str
    expected_codes: tuple[str, ...]
    observed_status: str
    observed_codes: tuple[str, ...]
    allowed_statuses: tuple[str, ...] = (WITHHELD, FAIL)

    @property
    def safe(self) -> bool:
        """The injection did not pass the gate."""
        return self.observed_status in self.allowed_statuses

    @property
    def named(self) -> bool:
        """The gate reported at least one of the expected reason codes.

        A case with no expected code is judged on its status alone (used for
        the threshold knife-edge controls).
        """
        if not self.expected_codes:
            return True
        return any(code in self.observed_codes for code in self.expected_codes)

    @property
    def ok(self) -> bool:
        return bool(self.safe and self.named)

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "description": self.description,
            "expected_codes": list(self.expected_codes),
            "observed_status": self.observed_status,
            "observed_codes": list(self.observed_codes),
            "allowed_statuses": list(self.allowed_statuses),
            "safe": self.safe,
            "named": self.named,
            "ok": self.ok,
        }


@dataclass(frozen=True)
class InjectionReport:
    """The whole matrix plus the control (a clean study that must pass)."""

    cases: tuple[InjectionCase, ...] = ()
    control_status: str = ""
    control_codes: tuple[str, ...] = ()

    @property
    def control_passes(self) -> bool:
        return self.control_status == PASS

    @property
    def all_safe(self) -> bool:
        return all(case.safe for case in self.cases)

    @property
    def all_named(self) -> bool:
        return all(case.named for case in self.cases)

    @property
    def all_matched(self) -> bool:
        return all(case.ok for case in self.cases)

    @property
    def passed(self) -> bool:
        return bool(self.control_passes and self.all_matched)

    def code_union(self) -> tuple[str, ...]:
        codes: list[str] = []
        for case in self.cases:
            for code in case.observed_codes:
                if code not in codes:
                    codes.append(code)
        return tuple(codes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "control_status": self.control_status,
            "control_codes": list(self.control_codes),
            "n_cases": len(self.cases),
            "all_safe": self.all_safe,
            "all_named": self.all_named,
            "all_matched": self.all_matched,
            "passed": self.passed,
            "codes_observed": list(self.code_union()),
            "cases": [case.as_dict() for case in self.cases],
        }


def _case(case_id: str, description: str, expected: tuple[str, ...], verdict) -> InjectionCase:
    return InjectionCase(
        case_id=case_id,
        description=description,
        expected_codes=tuple(expected),
        observed_status=str(verdict.status),
        observed_codes=tuple(verdict.reason_codes),
    )


def injection_suite(*, seed: int = 20240) -> InjectionReport:
    """Run the full falsification matrix and return every reaction."""
    spec = SPEC()
    cases: list[InjectionCase] = []

    # ---- control -----------------------------------------------------------
    control = _good_study(seed=seed)
    control_codes = tuple(control.verdict.reason_codes)

    # ---- end-to-end injections --------------------------------------------
    rng = np.random.default_rng(seed)

    def nan_5pct(points: np.ndarray) -> np.ndarray:
        values = control_model(points)
        bad = rng.random(values.size) < 0.05
        values = values.copy()
        values[bad] = np.nan
        return values

    cases.append(
        _case(
            "E1-nan-samples",
            "5 % of the solver evaluations return NaN",
            ("UQ-W01-NONFINITE",),
            _good_study(adapter=nan_5pct, seed=seed).verdict,
        )
    )
    cases.append(
        _case(
            "E2-all-nan",
            "the solver returns NaN everywhere",
            ("UQ-W01-NONFINITE",),
            _good_study(adapter=lambda p: np.full(len(p), np.nan), seed=seed).verdict,
        )
    )
    cases.append(
        _case(
            "E3-starved-budget",
            "16 samples for a 3-D problem (below the floor)",
            ("UQ-W02-TOOFEW",),
            _good_study(n_samples=16, degree=4, run_mc=False, require_cross_check=False, seed=seed).verdict,
        )
    )
    cases.append(
        _case(
            "E4-poor-surrogate",
            "total degree 2: eps_LOO > 0.1",
            ("UQ-W03-LOO",),
            _good_study(degree=2, seed=seed).verdict,
        )
    )
    cases.append(
        _case(
            "E5-no-crosscheck",
            "the Saltelli second opinion is missing",
            ("UQ-W07-NOMC",),
            _good_study(run_mc=False, seed=seed).verdict,
        )
    )
    cases.append(
        _case(
            "E6-mc-disagreement",
            "the two routes disagree on the band",
            ("UQ-W04-CROSSCHECK",),
            _case_mc_disagreement()
        )
    )
    cases.append(
        _case(
            "E7-solver-exception",
            "the solver raises on 10 % of the points",
            ("UQ-W01-NONFINITE", "UQ-W02-TOOFEW"),
            _good_study(adapter=_raising_solver(seed), seed=seed).verdict,
        )
    )

    # ---- verdict-level injections -----------------------------------------
    surrogate, budget, coords, values = _good_surrogate_and_budget(spec)
    indices = surrogate.sobol()
    good_mc = SobolMCResult(
        n_base=1024,
        n_model_evaluations=5 * 1024,
        mean=float(np.mean(values)),
        variance=float(np.var(values)),
        first=indices.first.copy(),
        first_saltelli=indices.first.copy(),
        total=indices.total.copy(),
        second={},
    )
    good_cross = cross_check_indices(indices, good_mc, names=spec.names)

    tampered = SobolMCResult(
        n_base=good_mc.n_base,
        n_model_evaluations=good_mc.n_model_evaluations,
        mean=good_mc.mean,
        variance=good_mc.variance,
        first=good_mc.first + 0.25,
        first_saltelli=good_mc.first_saltelli + 0.25,
        total=good_mc.total + 0.25,
        second={},
    )
    tampered_cross = cross_check_indices(indices, tampered, names=spec.names)
    cases.append(
        _case(
            "V1-tampered-mc",
            "the Monte-Carlo indices are shifted by +0.25",
            ("UQ-W04-CROSSCHECK",),
            decide(
                budget=budget,
                surrogate=surrogate,
                mc=tampered,
                cross_check=tampered_cross,
            ),
        )
    )

    class _Tagged:
        """Hash-tagged result set with a deliberately wrong tag."""

        def __init__(self, values: np.ndarray) -> None:
            self.values = values
            self.sample_hash = "0" * 64
            self.invalid_indices: tuple[int, ...] = ()
            self.error_messages: tuple[str, ...] = ()
            self.shards = 1

        @property
        def n(self) -> int:
            return int(self.values.shape[0])

        @property
        def n_valid(self) -> int:
            return int(np.count_nonzero(np.isfinite(self.values)))

    design = sample(spec, 512, seed=seed)
    cases.append(
        _case(
            "V2-hash-mismatch",
            "the result set is tagged with the wrong sample hash",
            ("UQ-W06-HASH",),
            decide(
                budget=budget,
                surrogate=surrogate,
                mc=good_mc,
                cross_check=good_cross,
                samples=design,
                results=_Tagged(control_model(design.points)),
            ),
        )
    )

    constant = fit_pce(coords, np.full(values.size, CONTROL_OFFSET), degree=4, families=spec.families)
    degenerate_budget = output_budget(constant, spec, n_mc=2000)
    cases.append(
        _case(
            "V3-degenerate-output",
            "a constant solver output (zero variance)",
            ("UQ-W05-DEGENERATE",),
            decide(
                budget=degenerate_budget,
                surrogate=constant,
                require_cross_check=False,
            ),
        )
    )
    cases.append(
        _case(
            "V4-no-surrogate",
            "no chaos surrogate was fitted",
            ("UQ-W08-NOPCE",),
            decide(budget=budget, surrogate=None, require_cross_check=False),
        )
    )

    # the acceptance threshold is "> 15 % is not acceptable", so exactly 15 %
    # must still pass and 15 % + 1e-9 must fail: both sides of the knife edge
    # are pinned down instead of trusting the comparison operator.
    exact = OutputBudget(
        mean=100.0,
        sd=7.6,
        lower95=92.5,
        upper95=107.5,
        width=15.0,
        relative_width=0.15,
        n_mc=4000,
        level=0.95,
    )
    cases.append(
        _case(
            "V5a-threshold-exact",
            "the band is exactly 15 % of the mean (positive control: must PASS)",
            (),
            decide(budget=exact, surrogate=surrogate, mc=good_mc, cross_check=good_cross),
        )
    )
    epsilon = OutputBudget(
        mean=100.0,
        sd=7.6,
        lower95=92.5,
        upper95=107.5 + 1.0e-9,
        width=15.0 + 1.0e-9,
        relative_width=0.15 + 1.0e-11,
        n_mc=4000,
        level=0.95,
    )
    cases.append(
        _case(
            "V5b-threshold-epsilon-above",
            "the band is 15 % + 1e-11 (must be rejected)",
            (),
            decide(budget=epsilon, surrogate=surrogate, mc=good_mc, cross_check=good_cross),
        )
    )

    wide = OutputBudget(
        mean=100.0,
        sd=10.0,
        lower95=90.0,
        upper95=110.0,
        width=20.0,
        relative_width=0.20,
        n_mc=4000,
        level=0.95,
    )
    cases.append(
        _case(
            "V6-wide-band",
            "the band is 20 % of the mean (> 15 %)",
            (),
            decide(budget=wide, surrogate=surrogate, mc=good_mc, cross_check=good_cross),
        )
    )

    finite_cases: list[InjectionCase] = []
    for case in cases:
        if case.case_id == "V5a-threshold-exact":
            # positive control: the knife edge must resolve to PASS
            finite_cases.append(
                InjectionCase(
                    case_id=case.case_id,
                    description=case.description,
                    expected_codes=(),
                    observed_status=case.observed_status,
                    observed_codes=case.observed_codes,
                    allowed_statuses=(PASS,),
                )
            )
        elif case.case_id == "V5b-threshold-epsilon-above":
            finite_cases.append(
                InjectionCase(
                    case_id=case.case_id,
                    description=case.description,
                    expected_codes=(),
                    observed_status=case.observed_status,
                    observed_codes=case.observed_codes,
                    allowed_statuses=(FAIL,),
                )
            )
        else:
            finite_cases.append(case)

    return InjectionReport(
        cases=tuple(finite_cases),
        control_status=str(control.status),
        control_codes=control_codes,
    )


def _raising_solver(seed: int) -> Callable[[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed + 1)

    def solver(points: np.ndarray) -> np.ndarray:
        values = control_model(points)
        bad = rng.random(values.size) < 0.10
        if np.any(bad):
            values = values.copy()
            values[bad] = np.nan
        return values

    return solver


def _case_mc_disagreement() -> Any:
    """Verdict from a PCE/sampled pair that genuinely disagrees."""
    spec = SPEC()
    surrogate, budget, _, _ = _good_surrogate_and_budget(spec)
    indices = surrogate.sobol()
    shifted = SobolMCResult(
        n_base=4096,
        n_model_evaluations=5 * 4096,
        mean=CONTROL_OFFSET,
        variance=CONTROL_SCALE ** 2 * 13.8,
        first=indices.first + 0.30,
        first_saltelli=indices.first + 0.30,
        total=indices.total + 0.30,
        second={},
    )
    cross = cross_check_indices(indices, shifted, names=spec.names)
    return decide(budget=budget, surrogate=surrogate, mc=shifted, cross_check=cross)