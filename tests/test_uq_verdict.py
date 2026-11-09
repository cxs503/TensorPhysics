"""Tests for the three-state verdict (PASS / FAIL / WITHHELD) and its codes."""
from __future__ import annotations

import numpy as np
import pytest

from tensorphysics.uq.benchmarks import ishigami, ishigami_analytic, ishigami_spec
from tensorphysics.uq.pce import fit_pce
from tensorphysics.uq.sample_spec import Marginal, SampleSpec
from tensorphysics.uq.sampler import sample
from tensorphysics.uq.sobol_mc import saltelli_indices
from tensorphysics.uq.verdict import (
    FAIL,
    PASS,
    WITHHELD,
    CrossCheck,
    OutputBudget,
    cross_check_indices,
    decide,
    min_samples_floor,
    output_budget,
)


def _budget(rel_width: float, mean: float = 100.0) -> OutputBudget:
    width = rel_width * mean
    return OutputBudget(
        mean=mean,
        sd=width / 3.92,
        lower95=mean - 0.5 * width,
        upper95=mean + 0.5 * width,
        width=width,
        relative_width=rel_width,
        n_mc=1000,
    )


def _surrogate(eps_loo: float = 0.01, n_terms: int = 4, n_samples: int = 512):
    spec = ishigami_spec()
    ss = sample(spec, max(n_samples, 64), seed=20240)
    coords = spec.from_latent(ss.latent)[1]
    pce = fit_pce(coords, ishigami(ss.points), degree=2, families=spec.families)
    # patch the LOO/term bookkeeping through a fresh frozen instance
    from dataclasses import replace

    return replace(pce, eps_loo=eps_loo, n_terms=n_terms, n_samples=n_samples)


def _sample_set(n: int = 512):
    return sample(ishigami_spec(), n, seed=20240)


def test_pass_when_band_is_narrow_and_evidence_complete() -> None:
    verdict = decide(
        budget=_budget(0.10),
        surrogate=_surrogate(),
        samples=_sample_set(),
        cross_check=CrossCheck((), 0.01, True, 0.05, 0.02, 5),
        require_cross_check=False,
    )
    assert verdict.status == PASS
    assert verdict.reason_codes == ()
    assert verdict.ratio == pytest.approx(0.10 / 0.15)


def test_fail_when_band_exceeds_threshold() -> None:
    verdict = decide(
        budget=_budget(0.25),
        surrogate=_surrogate(),
        samples=_sample_set(),
        require_cross_check=False,
    )
    assert verdict.status == FAIL
    assert verdict.blocking_codes == ()
    assert verdict.ratio == pytest.approx(0.25 / 0.15, rel=1e-12)


def test_withheld_reports_w03_when_loo_is_poor() -> None:
    verdict = decide(
        budget=_budget(0.05),
        surrogate=_surrogate(eps_loo=0.35),
        samples=_sample_set(),
        require_cross_check=False,
    )
    assert verdict.status == WITHHELD
    assert "UQ-W03-LOO" in verdict.reason_codes


def test_withheld_reports_w02_for_too_few_samples() -> None:
    verdict = decide(
        budget=_budget(0.05),
        surrogate=_surrogate(n_samples=16),
        samples=_sample_set(16),
        require_cross_check=False,
    )
    assert verdict.status == WITHHELD
    assert "UQ-W02-TOOFEW" in verdict.reason_codes
    assert min_samples_floor(3) == 32 and min_samples_floor(8) == 80


def test_withheld_reports_w02_when_samples_per_term_is_too_low() -> None:
    verdict = decide(
        budget=_budget(0.05),
        surrogate=_surrogate(n_terms=200, n_samples=512),
        samples=_sample_set(),
        require_cross_check=False,
    )
    assert verdict.status == WITHHELD
    assert "UQ-W02-TOOFEW" in verdict.reason_codes
    assert any("per term" in r.message for r in verdict.reasons)


def test_withheld_reports_w04_on_method_disagreement() -> None:
    bad = CrossCheck((), 0.42, False, 0.05, 0.02, 6)
    verdict = decide(
        budget=_budget(0.02),
        surrogate=_surrogate(),
        samples=_sample_set(),
        cross_check=bad,
    )
    assert verdict.status == WITHHELD
    assert "UQ-W04-CROSSCHECK" in verdict.reason_codes


def test_withheld_reports_w06_on_hash_mismatch() -> None:
    ss = _sample_set()

    from tensorphysics.uq.adapter import SampleResults

    results = SampleResults(values=np.zeros(ss.n), sample_hash="f" * 64)
    verdict = decide(
        budget=_budget(0.02),
        surrogate=_surrogate(),
        samples=ss,
        results=results,
        require_cross_check=False,
    )
    assert "UQ-W06-HASH" in verdict.reason_codes
    assert "UQ-W01-NONFINITE" not in verdict.reason_codes


def test_withheld_reports_w01_on_non_finite_samples() -> None:
    ss = _sample_set()
    from tensorphysics.uq.adapter import SampleResults

    values = np.zeros(ss.n)
    values[3] = np.nan
    results = SampleResults(
        values=values,
        sample_hash=ss.hash,
        invalid_indices=(3,),
        error_messages=("row 3: ValueError: solver diverged",),
    )
    verdict = decide(
        budget=_budget(0.02),
        surrogate=_surrogate(),
        samples=ss,
        results=results,
        require_cross_check=False,
    )
    assert verdict.status == WITHHELD
    assert "UQ-W01-NONFINITE" in verdict.reason_codes
    assert any("solver diverged" in r.message for r in verdict.reasons)


def test_withheld_reports_w05_for_degenerate_output() -> None:
    flat = OutputBudget(
        mean=100.0, sd=0.0, lower95=100.0, upper95=100.0, width=0.0, relative_width=0.0, n_mc=100
    )
    verdict = decide(
        budget=flat, surrogate=_surrogate(), samples=_sample_set(), require_cross_check=False
    )
    assert verdict.status == WITHHELD
    assert "UQ-W05-DEGENERATE" in verdict.reason_codes


def test_withheld_reports_w07_when_cross_check_is_missing() -> None:
    verdict = decide(budget=_budget(0.02), surrogate=_surrogate(), samples=_sample_set())
    assert verdict.status == WITHHELD
    assert "UQ-W07-NOMC" in verdict.reason_codes


def test_withheld_reports_w08_without_surrogate() -> None:
    verdict = decide(budget=_budget(0.02), require_cross_check=False)
    assert verdict.status == WITHHELD
    assert "UQ-W08-NOPCE" in verdict.reason_codes


def test_correlated_spec_attaches_a_note_only() -> None:
    spec = SampleSpec(
        (Marginal.uniform("a", 0.0, 1.0), Marginal.uniform("b", 0.0, 1.0)),
        correlation=np.array([[1.0, 0.4], [0.4, 1.0]]),
    )
    ss = sample(spec, 256, seed=1)
    pce = fit_pce(
        spec.from_latent(ss.latent)[1],
        ss.points[:, 0] + 2.0 * ss.points[:, 1],
        degree=2,
        families=spec.families,
    )
    verdict = decide(
        budget=_budget(0.05), surrogate=pce, samples=ss, require_cross_check=False
    )
    assert "UQ-N01-CORRELATED" in verdict.reason_codes
    assert "UQ-N01-CORRELATED" not in verdict.blocking_codes
    assert verdict.status in {PASS, FAIL}


def test_threshold_must_be_positive() -> None:
    with pytest.raises(ValueError):
        decide(budget=_budget(0.1), threshold=0.0)


def test_output_budget_matches_the_surrogate_distribution() -> None:
    spec = ishigami_spec()
    ss = sample(spec, 1024, seed=20240)
    coords = spec.from_latent(ss.latent)[1]
    pce = fit_pce(coords, ishigami(ss.points), degree=8, families=spec.families)
    budget = output_budget(pce, spec, n_mc=20000, seed=5)
    ref = ishigami_analytic()
    assert budget.mean == pytest.approx(3.5, abs=0.05)  # E[7 sin^2] = 3.5
    assert budget.sd == pytest.approx(np.sqrt(ref["variance"]), rel=0.02)
    assert budget.width == pytest.approx(budget.upper95 - budget.lower95)
    assert budget.relative_width > 1.0  # the Ishigami band is dominated by the spread
    with pytest.raises(ValueError):
        output_budget(pce, spec, n_mc=100, level=1.5)


def test_cross_check_scores_first_order_and_reports_second_order() -> None:
    spec = ishigami_spec()
    ss = sample(spec, 1024, seed=20240)
    coords = spec.from_latent(ss.latent)[1]
    pce = fit_pce(coords, ishigami(ss.points), degree=8, families=spec.families)
    mc = saltelli_indices(
        lambda u: ishigami((2.0 * u - 1.0) * np.pi),
        3,
        n_base=1 << 14,
        seed=6,
        bootstrap=0,
    )
    cross = cross_check_indices(pce.sobol(), mc, names=spec.names)
    assert cross.agreement
    assert cross.max_relative_deviation <= 0.05
    assert cross.n_compared >= 4
    assert cross.n_second_compared >= 1
    payload = cross.as_dict()
    assert payload["agreement"] is True
    assert len(payload["entries"]) >= 9

    # a deliberately wrong Monte-Carlo result must be caught
    bad = saltelli_indices(
        lambda u: ishigami((2.0 * u - 1.0) * np.pi) * 0 + u[:, 1],
        3,
        n_base=1 << 12,
        seed=6,
        bootstrap=0,
    )
    assert not cross_check_indices(pce.sobol(), bad, names=spec.names).agreement


def test_cross_check_rejects_dimension_mismatch() -> None:
    spec = ishigami_spec()
    ss = sample(spec, 256, seed=1)
    coords = spec.from_latent(ss.latent)[1]
    pce = fit_pce(coords, ishigami(ss.points), degree=2, families=spec.families)
    mc = saltelli_indices(
        lambda u: u[:, 0], 2, n_base=64, seed=1, second_order=False
    )
    with pytest.raises(ValueError):
        cross_check_indices(pce.sobol(), mc)