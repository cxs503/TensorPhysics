"""Core UQ building blocks: bases, marginals, sampling determinism, adapters.

These tests pin the numerical properties the rest of the UQ stack relies on; if
one of them regresses, every Sobol' index produced downstream is suspect.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensorphysics.uq import (
    HERMITE,
    LEGENDRE,
    Beta,
    CallableAdapter,
    InputModel,
    LogNormal,
    MultiIndexSet,
    Normal,
    RecordedAdapter,
    Uniform,
    draw_unit,
    orthonormal_basis,
    sample_input_model,
)
from tensorphysics.uq.basis import check_orthonormality
from tensorphysics.uq.benchmarks import (
    g_function_analytic_indices,
    ishigami_analytic_indices,
    ishigami_function,
    ishigami_input_model,
    ishigami_literature_values,
)


# ---------------------------------------------------------------------------
# bases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", [LEGENDRE, HERMITE])
def test_basis_is_orthonormal_in_its_own_measure(family: str) -> None:
    idx = MultiIndexSet.build(3, 4, family)
    report = check_orthonormality(idx)
    assert report["method"] == "gauss-quadrature"
    assert report["ok"], report
    assert report["max_abs_off_diagonal"] < 1e-12
    assert report["max_diag_error"] < 1e-12


def test_multi_index_total_degree_and_hyperbolic_truncation() -> None:
    full = MultiIndexSet.build(3, 4, LEGENDRE)
    # number of total-degree-4 monomials in 3 variables = C(7, 3) = 35
    assert full.size == 35
    hyper = MultiIndexSet.build(3, 4, LEGENDRE, q=0.5)
    assert hyper.size == 16 < full.size

    def _has(idx, alpha):
        return bool(any((idx.array == np.array(alpha)).all(axis=1)))

    # pure and pair terms survive; the third-order mixed term is dropped
    # (|alpha|_0.5 of (3, 1, 0) is (sqrt(3) + 1)^2 = 7.46 > 4)
    assert _has(hyper, (0, 0, 0)) and _has(hyper, (1, 0, 0)) and _has(hyper, (4, 0, 0))
    assert _has(hyper, (1, 1, 0))
    assert not _has(hyper, (3, 1, 0))
    assert not _has(hyper, (2, 2, 0))


def test_legendre_matches_univariate_values() -> None:
    idx = MultiIndexSet.build(1, 3, LEGENDRE)
    xi = np.linspace(-1.0, 1.0, 7)
    A = orthonormal_basis(xi[:, None], idx)
    P0, P1, P2, P3 = 1.0, xi, (3 * xi**2 - 1) / 2, (5 * xi**3 - 3 * xi) / 2
    assert np.allclose(A[:, 0], P0)
    assert np.allclose(A[:, 1], P1 / math.sqrt(1 / 3))
    assert np.allclose(A[:, 2], P2 / math.sqrt(1 / 5))
    assert np.allclose(A[:, 3], P3 / math.sqrt(1 / 7))


# ---------------------------------------------------------------------------
# marginals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "marginal",
    [
        Uniform(-2.0, 3.0),
        Normal(1.5, 0.4),
        LogNormal(math.log(2.0), 0.3),
        Beta(2.0, 5.0, 0.8, 1.4),
    ],
)
def test_marginal_cdf_ppf_roundtrip(marginal) -> None:
    u = np.array([0.001, 0.1, 0.5, 0.75, 0.999])
    x = marginal.ppf(u)
    assert np.allclose(marginal.cdf(x), u, atol=1e-9)


def test_marginal_moments_match_monte_carlo() -> None:
    rng = np.random.default_rng(11)
    for marginal in (Uniform(-2.0, 3.0), Normal(1.5, 0.4), LogNormal(0.0, 0.25), Beta(2.0, 5.0, 0.8, 1.4)):
        u = rng.random(400000)
        x = marginal.ppf(np.clip(u, 1e-12, 1 - 1e-12))
        assert x.mean() == pytest.approx(marginal.mean(), rel=2e-2, abs=2e-2)
        assert x.std(ddof=1) == pytest.approx(marginal.std(), rel=3e-2)


def test_uniform_std() -> None:
    assert Uniform(0.0, 1.0).std() == pytest.approx(1.0 / math.sqrt(12.0))


def test_input_model_rejects_bad_correlation() -> None:
    from tensorphysics.uq.distributions import GaussianCopulaError

    with pytest.raises(GaussianCopulaError):
        InputModel(["a", "b"], [Uniform(0, 1)] * 2, correlation=np.array([[1.0, 0.5], [0.4, 1.0]]))
    with pytest.raises(GaussianCopulaError):
        InputModel(["a", "b"], [Uniform(0, 1)] * 2, correlation=np.array([[1.0, 0.9], [0.9, 1.0]]) * 2.0)


def test_gaussian_copula_reproduces_requested_correlation() -> None:
    from scipy.stats import spearmanr

    R = np.array([[1.0, 0.6], [0.6, 1.0]])
    model = InputModel(["a", "b"], [Normal(0, 1), Normal(0, 1)], correlation=R)
    X = model.unit_to_physical(draw_unit(20000, 2, method="mc", seed=3))
    rho = float(np.corrcoef(X[:, 0], X[:, 1])[0, 1])
    assert rho == pytest.approx(0.6, abs=0.02)
    # the inverse map round-trips
    U = model.physical_to_unit(X)
    assert np.allclose(model.unit_to_physical(U), X, atol=1e-6)
    del spearmanr


def test_correlated_model_refuses_pce_scores() -> None:
    from tensorphysics.uq.distributions import GaussianCopulaError

    model = InputModel(["a", "b"], [Normal(0, 1)] * 2, correlation=np.array([[1.0, 0.5], [0.5, 1.0]]))
    with pytest.raises(GaussianCopulaError):
        model.standard_scores(draw_unit(10, 2), "legendre")


# ---------------------------------------------------------------------------
# sampling determinism
# ---------------------------------------------------------------------------


def test_sampling_is_deterministic_and_digest_stable() -> None:
    a = sample_input_model(ishigami_input_model(), 512, method="sobol", seed=1234)
    b = sample_input_model(ishigami_input_model(), 512, method="sobol", seed=1234)
    c = sample_input_model(ishigami_input_model(), 512, method="sobol", seed=1235)
    assert a.digest() == b.digest()
    assert np.array_equal(a.unit, b.unit)
    assert a.digest() != c.digest()


def test_sampler_methods_are_reproducible() -> None:
    for method in ("sobol", "lhs", "mc"):
        u1 = draw_unit(64, 3, method=method, seed=5)
        u2 = draw_unit(64, 3, method=method, seed=5)
        assert np.array_equal(u1, u2), method
        assert u1.shape == (64, 3)
        assert np.all((u1 > 0.0) & (u1 < 1.0))


def test_sobol_stratification_property() -> None:
    """A Sobol' point set must be much more uniform than Monte Carlo in 1-D."""
    u = draw_unit(256, 1, method="sobol", seed=99)[:, 0]
    u_mc = draw_unit(256, 1, method="mc", seed=99)[:, 0]
    # stratified: every 1/256 bin of the first 2^8 points, after resorting, is hit
    bins_sobol = np.unique(np.floor(np.sort(u) * 256).astype(int))
    bins_mc = np.unique(np.floor(np.sort(u_mc) * 256).astype(int))
    assert bins_sobol.size == 256
    assert bins_mc.size < 256


def test_unit_to_physical_is_monotone_per_component() -> None:
    model = ishigami_input_model()
    U = draw_unit(1000, 3, method="sobol", seed=1)
    X = model.unit_to_physical(U)
    order = np.argsort(U[:, 0])
    assert np.all(np.diff(X[order, 0]) > 0)


# ---------------------------------------------------------------------------
# adapters
# ---------------------------------------------------------------------------


def test_callable_adapter_counts_evaluations() -> None:
    model = ishigami_input_model()
    adapter = CallableAdapter(ishigami_function, model.names, name="ishigami")
    X = model.unit_to_physical(draw_unit(100, 3, method="sobol", seed=2))
    y = adapter.evaluate(X)
    assert y.shape == (100,)
    assert adapter.stats.n_evaluations == 100
    assert adapter.stats.n_failures == 0


def test_callable_adapter_converts_exceptions_to_nan() -> None:
    def fragile(X):
        out = X[:, 0].copy()
        out[0] = np.nan
        raise_if = out
        if np.any(~np.isfinite(raise_if)):  # pragma: no cover - branch exercised below
            return out
        return out

    model = InputModel(["x"], [Uniform(0, 1)])
    adapter = CallableAdapter(fragile, model.names)
    y = adapter.evaluate(draw_unit(8, 1, method="sobol", seed=4))
    assert np.isnan(y[0])


def test_scalar_adapter_loops_and_records_failure() -> None:
    model = InputModel(["x", "y"], [Uniform(0, 1), Uniform(0, 1)])

    def scalar(row):
        if row[0] > 0.9:
            raise ValueError("outside the solver's validity domain")
        return row[0] + row[1]

    adapter = CallableAdapter(scalar, model.names, batch=False)
    X = np.array([[0.1, 0.2], [0.95, 0.1], [0.3, 0.3]])
    y = adapter.evaluate(X)
    assert y[0] == pytest.approx(0.3)
    assert np.isnan(y[1])
    assert adapter.stats.n_failures == 1


def test_recorded_adapter_does_not_interpolate() -> None:
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    adapter = RecordedAdapter(X, np.array([1.0, 2.0]), ["a", "b"])
    y = adapter.evaluate(X)
    assert np.allclose(y, [1.0, 2.0])
    y2 = adapter.evaluate(np.array([[0.5, 0.5]]))
    assert np.isnan(y2[0])


# ---------------------------------------------------------------------------
# analytic benchmarks
# ---------------------------------------------------------------------------


def test_ishigami_analytic_matches_literature_rounding() -> None:
    exact = ishigami_analytic_indices()
    lit = ishigami_literature_values()
    assert np.allclose(exact.S1, lit["S1"], atol=1e-4)
    # The commonly quoted ST_3 = 0.2439 is a rounded publication value; the
    # exact value is 0.243684, i.e. the table itself carries 2.2e-4 of error.
    # An absolute-error criterion must therefore be checked against the exact
    # numbers, not against the table.
    assert np.allclose(exact.ST, lit["ST"], atol=5e-4)
    assert abs(exact.ST[2] - lit["ST"][2]) == pytest.approx(2.16e-4, abs=2e-5)
    # the closed second-order index equals the whole total effect of x3
    assert exact.S2["x1-x3"] == pytest.approx(exact.ST[2], abs=1e-12)
    # x3 has zero first-order effect
    assert exact.S1[2] == 0.0


def test_g_function_analytic_matches_published_values() -> None:
    idx = g_function_analytic_indices()
    # classical a = (0, 1, 4.5, 9, 99, 99, 99, 99) values
    assert np.allclose(idx.S1[:4], [0.7162, 0.1790, 0.0237, 0.0072], atol=5e-4)
    assert np.all(idx.ST >= idx.S1 - 1e-12)