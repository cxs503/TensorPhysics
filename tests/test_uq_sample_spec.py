"""Tests for the uncertain-input specification (marginals + copula)."""
from __future__ import annotations

import numpy as np
import pytest

from tensorphysics.uq.sample_spec import Marginal, SampleSpec


def test_marginal_validation_rejects_bad_definitions() -> None:
    with pytest.raises(ValueError):
        Marginal.uniform("x", 1.0, 1.0)
    with pytest.raises(ValueError):
        Marginal.normal("x", 0.0, 0.0)
    with pytest.raises(ValueError):
        Marginal("x", "weibull", (1.0, 2.0))
    with pytest.raises(ValueError):
        Marginal.normal("x", float("nan"), 1.0)
    with pytest.raises(ValueError):
        Marginal("x", "uniform", (0.0, 1.0, 2.0))  # type: ignore[arg-type]


def test_spec_validation() -> None:
    m = Marginal.uniform("x", 0.0, 1.0)
    with pytest.raises(ValueError):
        SampleSpec((m, m))  # duplicate names
    with pytest.raises(ValueError):
        SampleSpec((m,), correlation=np.ones((1, 2)))
    with pytest.raises(ValueError):
        SampleSpec(
            (m, Marginal.uniform("y", 0.0, 1.0)),
            correlation=np.array([[1.0, 0.5], [0.4, 1.0]]),
        )
    with pytest.raises(ValueError):
        SampleSpec(
            (m, Marginal.uniform("y", 0.0, 1.0)),
            correlation=np.array([[1.0, 1.0], [1.0, 1.0]]),  # singular
        )


def test_marginal_moments_match_empirical() -> None:
    rng = np.random.default_rng(7)
    u = rng.random(200000)
    for m in (
        Marginal.uniform("u", -2.0, 6.0),
        Marginal.normal("n", 1.5, 0.7),
        Marginal.lognormal("ln", 0.2, 0.3),
    ):
        x = m.from_unit(u)
        mean, sd = m.moments()
        assert x.mean() == pytest.approx(mean, rel=5e-3)
        assert x.std() == pytest.approx(sd, rel=2e-2)


def test_marginal_basis_family_and_support() -> None:
    assert Marginal.uniform("a", 0, 1).basis_family == "legendre"
    assert Marginal.normal("b", 0, 1).basis_family == "hermite"
    assert Marginal.lognormal("c", 0, 1).basis_family == "hermite"
    assert Marginal.uniform("a", 2, 3).support == (2.0, 3.0)
    assert Marginal.normal("b", 0, 1).support == (-np.inf, np.inf)


def test_transform_and_physical_inverse_round_trip() -> None:
    spec = SampleSpec(
        (
            Marginal.uniform("L", 100.0, 200.0),
            Marginal.normal("B", 30.0, 0.5),
            Marginal.lognormal("k", 0.0, 0.1),
        )
    )
    unit = np.random.default_rng(3).random((64, 3))
    points, coords = spec.transform(unit)
    assert points.shape == coords.shape == (64, 3)
    # physical support respected
    assert np.all((points[:, 0] >= 100.0) & (points[:, 0] <= 200.0))
    assert np.all(points[:, 2] > 0.0)
    # legendre coordinate lives in [-1, 1]
    assert np.all(np.abs(coords[:, 0]) <= 1.0)
    # inverse map recovers the latent coordinates
    z = spec.latent_from_unit(unit)
    assert np.allclose(spec.physical_to_latent(points), z, atol=1e-10)


def test_copula_reproduces_latent_correlation() -> None:
    R = np.array([[1.0, 0.6], [0.6, 1.0]])
    spec = SampleSpec(
        (Marginal.uniform("a", 0.0, 1.0), Marginal.normal("b", 0.0, 1.0)), correlation=R
    )
    unit = np.random.default_rng(11).random((100000, 2))
    z = spec.latent_from_unit(unit)
    # the latent coordinates are correlated as specified ...
    assert np.corrcoef(z.T)[0, 1] == pytest.approx(0.6, abs=0.01)
    # ... while the inverse map returns independent latent coordinates
    points, _ = spec.from_latent(z)
    z_back = spec.physical_to_latent(points)
    assert np.corrcoef(z_back.T)[0, 1] == pytest.approx(0.0, abs=0.02)
    # discrete marginal: Spearman of the copula is (6/pi) asin(R/2)
    rho = (6.0 / np.pi) * np.arcsin(0.6 / 2.0)
    from scipy.stats import spearmanr

    assert spearmanr(points[:, 0], points[:, 1]).statistic == pytest.approx(rho, abs=0.02)


def test_fingerprint_is_stable_and_sensitive() -> None:
    a = SampleSpec((Marginal.uniform("x", 0.0, 1.0), Marginal.normal("y", 0.0, 1.0)))
    b = SampleSpec((Marginal.uniform("x", 0.0, 1.0), Marginal.normal("y", 0.0, 1.0)))
    c = SampleSpec((Marginal.uniform("x", 0.0, 1.0000001), Marginal.normal("y", 0.0, 1.0)))
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != c.fingerprint()
    assert a.as_dict()["version"] == "tpuq-spec-v1"
    assert not a.correlated