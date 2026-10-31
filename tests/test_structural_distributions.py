"""Verification of the distribution / Nataf layer of the structural reliability module."""
from __future__ import annotations

from math import asin, log, log1p, pi, sqrt

import pytest

from tensorphysics.structural import (
    Distribution,
    ExponentialDistribution,
    GumbelMaxDistribution,
    LognormalDistribution,
    NatafTransform,
    NormalDistribution,
    bivariate_normal_cdf,
    gumbel_max_from_moments,
    lognormal_from_moments,
    norm_cdf,
    norm_pdf,
    norm_ppf,
    normal_from_moments,
)
from tensorphysics.structural.distributions import (
    _GH_NODES,
    _GH_WEIGHTS,
    clip_probability,
)


def test_norm_cdf_known_values():
    assert norm_cdf(0.0) == pytest.approx(0.5, abs=1e-15)
    assert norm_cdf(1.959963984540054) == pytest.approx(0.975, abs=1e-12)
    assert norm_cdf(-3.0) == pytest.approx(0.0013498980316301, rel=1e-12)
    assert norm_cdf(-8.0) == pytest.approx(6.22096057427178e-16, rel=1e-9)
    # symmetry
    for z in (0.3, 1.0, 2.5, 4.5):
        assert norm_cdf(z) + norm_cdf(-z) == pytest.approx(1.0, abs=1e-15)


def test_norm_ppf_roundtrip_and_reference_quantiles():
    for z in (-7.0, -3.3, -0.5, 0.0, 0.75, 2.0, 5.5):
        # in the extreme tail the round trip is limited by the 1e-16 absolute
        # resolution of the CDF value itself, not by the quantile routine
        assert norm_ppf(norm_cdf(z)) == pytest.approx(z, abs=1e-8)
    assert norm_ppf(0.975) == pytest.approx(1.959963984540054, rel=1e-12)
    assert norm_ppf(1e-6) == pytest.approx(-4.753424308822899, rel=1e-10)
    assert norm_ppf(1.0 - 1e-6) == pytest.approx(4.753424308822899, rel=1e-10)
    assert norm_ppf(0.5) == pytest.approx(0.0, abs=1e-12)


def test_norm_pdf_normalises():
    # rectangle rule over a wide range
    step = 0.01
    total = 0.0
    x = -10.0
    while x < 10.0:
        total += norm_pdf(x) * step
        x += step
    assert total == pytest.approx(1.0, abs=1e-8)


def test_clip_probability_bounds():
    assert clip_probability(0.0) > 0.0
    assert clip_probability(1.0) < 1.0
    assert clip_probability(0.3) == pytest.approx(0.3, abs=0.0)


def test_gauss_hermite_table_moments():
    assert sum(_GH_WEIGHTS) == pytest.approx(1.0, abs=1e-14)
    for power, reference in ((2, 1.0), (4, 3.0), (6, 15.0), (8, 105.0)):
        value = sum(w * z**power for z, w in zip(_GH_NODES, _GH_WEIGHTS))
        assert value == pytest.approx(reference, rel=1e-10)
    # symmetry of the nodes
    assert _GH_NODES[0] == pytest.approx(-_GH_NODES[-1], rel=1e-15)


def test_gauss_hermite_table_matches_numpy():
    numpy = pytest.importorskip("numpy")
    nodes, weights = numpy.polynomial.hermite.hermgauss(20)
    reference_nodes = nodes * sqrt(2.0)
    reference_weights = weights / sqrt(pi)
    for z, z_ref in zip(_GH_NODES, reference_nodes):
        assert z == pytest.approx(float(z_ref), rel=1e-14)
    for w, w_ref in zip(_GH_WEIGHTS, reference_weights):
        assert w == pytest.approx(float(w_ref), rel=1e-14)


def test_bivariate_normal_cdf_independence():
    for h, k in ((0.0, 0.0), (1.0, -0.5), (-2.0, 1.5)):
        assert bivariate_normal_cdf(h, k, 0.0) == pytest.approx(norm_cdf(h) * norm_cdf(k), abs=1e-10)


def test_bivariate_normal_cdf_closed_form_and_symmetry():
    for rho in (-0.7, -0.2, 0.0, 0.4, 0.9):
        assert bivariate_normal_cdf(0.0, 0.0, rho) == pytest.approx(
            0.25 + asin(rho) / (2.0 * pi), abs=1e-9
        )
        # P(Z1>-h, Z2>k) identity: Phi2(h,k,rho) + Phi2(-h,k,-rho) = Phi(k)
        h, k = 1.3, -0.4
        assert bivariate_normal_cdf(h, k, rho) + bivariate_normal_cdf(-h, k, -rho) == pytest.approx(
            norm_cdf(k), abs=1e-8
        )


def test_marginal_roundtrip_normal_lognormal_gumbel_exponential():
    marginals = [
        NormalDistribution(mu=1.5, sigma=0.7),
        LognormalDistribution(mu_ln=log(200.0), sigma_ln=0.2),
        GumbelMaxDistribution(loc=1.0, scale=0.3),
        ExponentialDistribution(lam=1.5),
    ]
    for marginal in marginals:
        for z in (-3.0, -1.0, 0.0, 0.8, 2.5):
            x = marginal.ppf_z(z)
            assert marginal.z_of_x(x) == pytest.approx(z, abs=1e-9), marginal.name
        for x in (0.2, 0.7, 1.4):
            p = marginal.cdf(x)
            if 1e-12 < p < 1 - 1e-12:
                assert marginal.cdf(marginal.ppf(p)) == pytest.approx(p, rel=1e-9)


def test_marginal_moments():
    lognormal = lognormal_from_moments(mean=355.0, cov=0.08)
    assert lognormal.mean() == pytest.approx(355.0, rel=1e-12)
    assert lognormal.std() == pytest.approx(28.4, rel=1e-3)

    gumbel = gumbel_max_from_moments(mean=2.5, cov=0.28)
    assert gumbel.mean() == pytest.approx(2.5, rel=1e-12)
    assert gumbel.std() == pytest.approx(0.70, rel=1e-3)
    # the Gumbel scale follows sigma * sqrt(6)/pi
    assert gumbel.scale == pytest.approx(0.70 * sqrt(6.0) / pi, rel=1e-3)

    normal = normal_from_moments(mean=2.0, cov=0.25)
    assert normal.sigma == pytest.approx(0.5, rel=1e-12)


def test_gumbel_quantile_far_tail_is_stable():
    gumbel = gumbel_max_from_moments(mean=2.5, cov=0.28)
    # exact expression loc - scale*log(-log(Phi(z))) with Phi(z) = 1 - Phi(-z)
    for z in (7.0, 9.0, 11.0):
        tail = clip_probability(norm_cdf(-z))
        reference = gumbel.loc - gumbel.scale * log(-log1p(-tail))
        assert gumbel.ppf_z(z) == pytest.approx(reference, rel=1e-10)
    # the naive route through the saturated probability stalls; the stable one does not
    naive = gumbel.ppf(clip_probability(norm_cdf(11.0)))
    assert gumbel.ppf_z(11.0) > naive * 1.05
    assert gumbel.ppf_z(11.0) > gumbel.ppf_z(9.0) > gumbel.ppf_z(7.0)


def test_nataf_solves_gaussian_copula_correlation_for_mixed_pair():
    marginals = [gumbel_max_from_moments(2.5, 0.28), lognormal_from_moments(1.0, 0.10)]
    target = 0.25
    transform = NatafTransform(marginals, [[1.0, target], [target, 1.0]])
    rho0 = transform.rho0[0][1]
    # the Nataf correction must differ from the physical correlation ...
    assert rho0 != pytest.approx(target, abs=1e-4)
    # ... and reproduce it when the pair integral is re-evaluated
    assert transform._pair_integral(0, 1, rho0) == pytest.approx(target, abs=1e-6)
    assert transform._pair_integral(0, 1, target) != pytest.approx(target, abs=1e-4)


def test_nataf_normal_pairs_need_no_correction():
    marginals = [NormalDistribution(0.0, 1.0), NormalDistribution(5.0, 2.0), NormalDistribution(-1.0, 0.4)]
    correlation = [[1.0, 0.3, 0.0], [0.3, 1.0, -0.2], [0.0, -0.2, 1.0]]
    transform = NatafTransform(marginals, correlation)
    for i in range(3):
        for j in range(3):
            assert transform.rho0[i][j] == pytest.approx(correlation[i][j], abs=1e-12)


def test_nataf_transform_roundtrip():
    marginals = [
        gumbel_max_from_moments(2.5, 0.28),
        lognormal_from_moments(355.0, 0.08),
        NormalDistribution(2.0, 0.5),
        lognormal_from_moments(1.0, 0.10),
    ]
    correlation = [
        [1.0, 0.0, 0.0, 0.25],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.25, 0.0, 0.0, 1.0],
    ]
    transform = NatafTransform(marginals, correlation)
    u = [1.2, -0.7, 0.3, 2.1]
    assert transform.x_to_u(transform.u_to_x(u)) == pytest.approx(u, abs=1e-9)


def test_nataf_monte_carlo_sample_correlation():
    marginals = [gumbel_max_from_moments(2.5, 0.28), lognormal_from_moments(1.0, 0.10)]
    correlation = [[1.0, 0.25], [0.25, 1.0]]
    transform = NatafTransform(marginals, correlation)
    samples = transform.sample_physical(20000, seed=11)
    xs = [row[0] for row in samples]
    ys = [row[1] for row in samples]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / len(xs)
    sx = sqrt(sum((a - mx) ** 2 for a in xs) / len(xs))
    sy = sqrt(sum((b - my) ** 2 for b in ys) / len(ys))
    assert cov / (sx * sy) == pytest.approx(0.25, abs=0.03)
    assert mx == pytest.approx(marginals[0].mean(), rel=0.03)
    assert my == pytest.approx(1.0, rel=0.02)


def test_nataf_rejects_invalid_correlation_matrix():
    marginals = [NormalDistribution(0.0, 1.0), NormalDistribution(0.0, 1.0)]
    with pytest.raises(ValueError):
        NatafTransform(marginals, [[1.0, 1.0], [1.0, 1.0]])
    with pytest.raises(ValueError):
        NatafTransform(marginals, [[1.0, 0.5], [-0.5, 1.0]])  # not symmetric


def test_distribution_base_interface_is_enforced():
    with pytest.raises(NotImplementedError):
        Distribution().cdf(1.0)
    with pytest.raises(ValueError):
        NormalDistribution(mu=0.0, sigma=-1.0)
    with pytest.raises(ValueError):
        GumbelMaxDistribution(loc=0.0, scale=0.0)
    with pytest.raises(ValueError):
        ExponentialDistribution(lam=0.0)