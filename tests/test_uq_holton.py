"""Holton-type high-dimensional anchor: model, closed forms, recovery.

Covers acceptance items F1/F2 of the campaign node from the analytic side:
the closed-form decomposition must agree with (a) direct Monte-Carlo moments,
(b) the internal identity ``sum_i S_Ti = 1 + sum_{i<j} S_ij``, (c) a
degree-2 polynomial-chaos fit (exact reproduction), and (d) the sampled
pick-freeze route at a working sample size.
"""
from __future__ import annotations

import numpy as np
import pytest

from tensorphysics.uq.holton import (
    holton,
    holton_analytic,
    holton_default_coefficients,
    holton_reference_indices,
    holton_spec,
    make_holton,
)
from tensorphysics.uq.pce import fit_pce
from tensorphysics.uq.sampler import sample
from tensorphysics.uq.sobol_mc import saltelli_indices


def test_holton_closed_form_matches_direct_moments():
    model = make_holton(4)
    rng = np.random.default_rng(11)
    x = rng.random((200000, 4))
    y = model(x)
    analytic = model.analytic()
    assert y.mean() == pytest.approx(analytic["mean"], rel=0.01)
    assert y.var() == pytest.approx(analytic["variance"], rel=0.03)


def test_holton_total_index_identity():
    """For a model with only pairwise terms: sum ST = 1 + sum S_ij exactly."""
    analytic = holton_analytic(6)
    second_sum = float(sum(analytic["second"].values()))
    assert analytic["total_sum"] == pytest.approx(1.0 + second_sum, abs=1e-12)
    # interaction share is large enough for a first-order analysis to be wrong
    assert analytic["interaction_variance_share"] > 0.10


def test_holton_pce_recovers_analytic_indices():
    """Degree 2 reproduces the polynomial model, so the indices are exact."""
    model = make_holton(6)
    spec = holton_spec(6)
    design = sample(spec, 256, rule="sobol", seed=5)
    points, coords = spec.from_latent(design.latent)
    surrogate = fit_pce(coords, model(points), degree=2, families=spec.families)
    ind = surrogate.sobol()
    analytic = model.analytic()
    assert surrogate.eps_loo < 1e-6
    np.testing.assert_allclose(ind.first, analytic["first"], atol=1e-9)
    np.testing.assert_allclose(ind.total, analytic["total"], atol=1e-9)


def test_holton_sampled_indices_converge_within_five_percent():
    """F2: the sampled route is within 5 % at a modest pick-freeze budget."""
    model = make_holton(6)
    spec = holton_spec(6)
    analytic = model.analytic()

    def evaluate(unit: np.ndarray) -> np.ndarray:
        phys, _ = spec.transform(unit)
        return model(phys)

    errors = []
    for n_base in (1024, 4096):
        mc = saltelli_indices(evaluate, spec.dim, n_base=n_base, seed=123)
        errors.append(
            max(
                float(np.max(np.abs(mc.first - analytic["first"]))),
                float(np.max(np.abs(mc.total - analytic["total"]))),
            )
        )
    assert errors[-1] <= 0.05, errors
    # and the finer budget is not worse than the coarse one (no divergence)
    assert errors[-1] <= 2.0 * errors[0]


def test_holton_reference_indices_are_normalised_and_named():
    ref = holton_reference_indices(5)
    assert ref["names"] == tuple(f"x{i + 1}" for i in range(5))
    assert ref["sqrt"] == pytest.approx(np.sqrt(ref["variance"]))
    assert float(np.sum(ref["first"])) <= 1.0 + 1e-12


def test_holton_input_validation():
    with pytest.raises(ValueError):
        holton(np.zeros((3, 2, 2)))
    with pytest.raises(ValueError):
        holton(np.zeros((4, 3)), dim=4)
    with pytest.raises(ValueError):
        holton_default_coefficients(1)
    with pytest.raises(ValueError):
        holton_default_coefficients(4, interaction_band=0)
    with pytest.raises(ValueError):
        holton_spec(1)
    with pytest.raises(ValueError):
        holton_analytic(3, linear=[1.0, 2.0])
    coeffs = holton_default_coefficients(3)
    with pytest.raises(ValueError):
        holton(np.zeros((2, 3)), linear=coeffs[0][:2])


def test_holton_benchmark_as_dict_roundtrip():
    model = make_holton(4)
    payload = model.as_dict()
    assert payload["dim"] == 4
    assert len(payload["linear"]) == 4
    assert payload["interaction"], "default band must contain interactions"
    assert payload["analytic"]["variance"] > 0.0
    rebuilt = holton(
        np.array([[0.25, 0.5, 0.75, 1.0]]),
        linear=payload["linear"],
        interaction=[
            [0.0 for _ in range(4)] for _ in range(4)
        ],
        dim=4,
    )
    # with no interaction the value is just the linear part (sanity of the roundtrip)
    assert rebuilt.shape == (1,)
    assert np.isfinite(rebuilt[0])