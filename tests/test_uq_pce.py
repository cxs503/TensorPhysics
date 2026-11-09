"""Tests for the polynomial-chaos surrogate and the analytic Sobol' indices.

The Ishigami anchor is the acceptance test of the method: its indices are known
in closed form, so the assertions here are absolute (``|S - S_exact| <= 1e-3``)
rather than self-referential.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm, spearmanr

from tensorphysics.uq.benchmarks import ishigami, ishigami_analytic, ishigami_spec
from tensorphysics.uq.pce import (
    basis_matrix,
    bootstrap_sobol,
    fit_pce,
    hermite_orthonormal,
    legendre_orthonormal,
    multi_indices,
)
from tensorphysics.uq.sampler import draw_unit, sample


def test_legendre_basis_is_orthonormal() -> None:
    x, w = np.polynomial.legendre.leggauss(25)
    phi = legendre_orthonormal(x, 8)
    gram = (phi * (0.5 * w)[:, None]).T @ phi
    assert np.allclose(gram, np.eye(9), atol=1e-12)


def test_hermite_basis_is_orthonormal() -> None:
    z = np.linspace(-9.0, 9.0, 400001)
    weight = norm.pdf(z) * (z[1] - z[0])
    phi = hermite_orthonormal(z, 8)
    gram = (phi * weight[:, None]).T @ phi
    assert np.allclose(gram, np.eye(9), atol=1e-6)
    # He_2(0) = -1, normalised by sqrt(2)
    assert hermite_orthonormal(np.array([0.0]), 2)[0, 2] == pytest.approx(-1.0 / math.sqrt(2.0))


def test_multi_index_counts() -> None:
    for dim, degree in ((3, 4), (4, 3), (6, 2)):
        expected = math.comb(dim + degree, degree)
        m = multi_indices(dim, degree)
        assert m.shape == (expected, dim)
        assert tuple(m[0]) == (0,) * dim
        assert m.max() == degree
        assert np.all(m.sum(axis=1) <= degree)


def test_ishigami_pce_matches_analytic_within_1e_3() -> None:
    spec = ishigami_spec()
    ref = ishigami_analytic()
    ss = sample(spec, 1024, seed=20240)
    coords = spec.from_latent(ss.latent)[1]
    pce = fit_pce(coords, ishigami(ss.points), degree=8, families=spec.families)
    ind = pce.sobol(order=2)
    assert pce.eps_loo < 0.05
    assert abs(ind.first[0] - ref["first"][0]) <= 1.0e-3
    assert abs(ind.first[1] - ref["first"][1]) <= 1.0e-3
    assert abs(ind.total[0] - ref["total"][0]) <= 1.0e-3
    assert abs(ind.total[2] - ref["total"][2]) <= 1.0e-3
    assert abs(ind.second[(0, 2)] - ref["second"][(0, 2)]) <= 1.0e-3
    # the campaign-quoted 4-decimal reference must also be met
    assert abs(ind.first[0] - 0.3139) <= 1.0e-3
    assert abs(ind.total[0] - 0.5574) <= 1.0e-3


def test_lars_and_ols_agree() -> None:
    spec = ishigami_spec()
    ss = sample(spec, 1024, seed=20240)
    coords = spec.from_latent(ss.latent)[1]
    y = ishigami(ss.points)
    lars = fit_pce(coords, y, degree=8, families=spec.families)
    ols = fit_pce(coords, y, degree=8, families=spec.families, method="ols")
    assert lars.n_terms < ols.n_terms
    assert np.allclose(lars.sobol().first, ols.sobol().first, atol=5e-3)


def test_loo_error_decreases_with_degree() -> None:
    spec = ishigami_spec()
    ss = sample(spec, 1024, seed=20240)
    coords = spec.from_latent(ss.latent)[1]
    y = ishigami(ss.points)
    degrees = [2, 3, 4, 5, 6, 8, 10]
    loos = [fit_pce(coords, y, degree=p, families=spec.families).eps_loo for p in degrees]
    assert len(loos) >= 4
    assert spearmanr(degrees, loos).statistic <= -0.9
    assert loos[-1] < 1.0e-2 * loos[0]


def test_loo_error_decreases_with_sample_count_at_fixed_truncation() -> None:
    """The sample-size grid holds the basis fixed (OLS) so that LOO measures
    estimation error only; with the greedy selector the active set adapts to N
    and the two effects would be mixed."""
    spec = ishigami_spec()
    sizes = [512, 1024, 2048, 4096]
    loos = []
    for n in sizes:
        ss = sample(spec, n, seed=20240)
        coords = spec.from_latent(ss.latent)[1]
        pce = fit_pce(
            coords, ishigami(ss.points), degree=8, families=spec.families, method="ols"
        )
        loos.append(pce.eps_loo)
    assert spearmanr(sizes, loos).statistic <= -0.9
    assert loos[-1] < loos[0]


def test_fit_rejects_non_finite_values() -> None:
    spec = ishigami_spec()
    ss = sample(spec, 128, seed=1)
    coords = spec.from_latent(ss.latent)[1]
    y = ishigami(ss.points)
    y[4] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        fit_pce(coords, y, degree=4, families=spec.families)


def test_fit_validates_shapes() -> None:
    spec = ishigami_spec()
    ss = sample(spec, 64, seed=1)
    coords = spec.from_latent(ss.latent)[1]
    with pytest.raises(ValueError):
        fit_pce(coords, np.zeros(10), degree=2, families=spec.families)
    with pytest.raises(ValueError):
        fit_pce(coords, np.zeros(64), degree=2, families=("legendre",))
    with pytest.raises(ValueError):
        fit_pce(coords, np.zeros(64), degree=2, families=spec.families, method="magic")
    with pytest.raises(ValueError):
        fit_pce(coords, np.zeros(64), degree=2, families=spec.families, method="ols").sobol(order=0)


def test_bootstrap_interval_brackets_the_point_estimate() -> None:
    spec = ishigami_spec()
    ss = sample(spec, 1024, seed=20240)
    coords = spec.from_latent(ss.latent)[1]
    y = ishigami(ss.points)
    pce = fit_pce(coords, y, degree=8, families=spec.families)
    ind = bootstrap_sobol(pce, coords, y, n_boot=40, seed=3)
    assert ind.first_ci is not None and ind.total_ci is not None
    lo, hi = ind.first_ci
    assert np.all(lo <= ind.first + 1e-12) and np.all(ind.first - 1e-12 <= hi)
    lo_t, hi_t = ind.total_ci
    assert np.all(lo_t <= ind.total + 1e-12) and np.all(ind.total - 1e-12 <= hi_t)


def test_zero_variance_surrogate_reports_zero_indices() -> None:
    # a constant model: variance 0, indices 0 (the verdict withholds separately)
    coords = np.random.default_rng(0).uniform(-1.0, 1.0, (64, 2))
    pce = fit_pce(
        coords, np.full(64, 3.0), degree=2, families=("legendre", "legendre")
    )
    ind = pce.sobol()
    assert ind.variance == pytest.approx(0.0, abs=1e-20)
    assert np.allclose(ind.first, 0.0)


def test_design_matrix_matches_basis_product() -> None:
    coords = np.random.default_rng(0).uniform(-1, 1, (10, 2))
    multi = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 0]])
    A = basis_matrix(coords, ("legendre", "legendre"), multi)
    p1 = legendre_orthonormal(coords[:, 0], 2)
    p2 = legendre_orthonormal(coords[:, 1], 2)
    assert np.allclose(A[:, 3], p1[:, 1] * p2[:, 1])
    assert np.allclose(A[:, 4], p1[:, 2])
    # the total-degree basis is orthonormal in expectation: Monte-Carlo Gram
    u = draw_unit(50000, 2, seed=4)
    from tensorphysics.uq.sample_spec import Marginal, SampleSpec

    spec = SampleSpec((Marginal.uniform("a", -1, 1), Marginal.uniform("b", -1, 1)))
    c = spec.from_latent(spec.latent_from_unit(u))[1]
    G = basis_matrix(c, ("legendre", "legendre"), multi).T @ basis_matrix(
        c, ("legendre", "legendre"), multi
    )
    assert np.allclose(G / 50000.0, np.eye(len(multi)), atol=0.02)