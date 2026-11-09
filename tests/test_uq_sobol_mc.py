"""Tests for the Saltelli/Jansen Monte-Carlo Sobol' estimator."""
from __future__ import annotations

import numpy as np
import pytest

from tensorphysics.uq.benchmarks import ishigami, ishigami_analytic
from tensorphysics.uq.sampler import draw_unit
from tensorphysics.uq.sobol_mc import saltelli_design, saltelli_indices


def _ishigami_on_unit(unit: np.ndarray) -> np.ndarray:
    return ishigami((2.0 * unit - 1.0) * np.pi)


def test_design_blocks_are_consistent() -> None:
    blocks = saltelli_design(3, 64, seed=11)
    assert set(blocks) == {"A", "B", "A_B_0", "A_B_1", "A_B_2", "A_B_0_1", "A_B_0_2", "A_B_1_2"}
    A, B = blocks["A"], blocks["B"]
    # A_B_i equals A except in column i, where it equals B
    np.testing.assert_array_equal(blocks["A_B_1"][:, 0], A[:, 0])
    np.testing.assert_array_equal(blocks["A_B_1"][:, 1], B[:, 1])
    np.testing.assert_array_equal(blocks["A_B_0_2"][:, 2], B[:, 2])
    # A and B must not be row-paired: with 4096 rows the permutation leaves the
    # sample correlation at noise level (two directly seeded Sobol' sets give
    # |corr| ~ 0.5-0.9, which silently breaks the pick-freeze identity)
    big = saltelli_design(3, 4096, seed=11)
    for k in range(3):
        assert abs(np.corrcoef(big["A"][:, k], big["B"][:, k])[0, 1]) < 0.05


def test_ishigami_indices_within_5_percent() -> None:
    ref = ishigami_analytic()
    res = saltelli_indices(
        _ishigami_on_unit, 3, n_base=1 << 16, seed=4, bootstrap=0
    )
    # first and total order: relative error <= 5 %
    for i in (0, 1):
        assert abs(res.first[i] - ref["first"][i]) / ref["first"][i] <= 0.05
    for i in range(3):
        assert abs(res.total[i] - ref["total"][i]) / ref["total"][i] <= 0.05
    # the (1, 3) interaction is the only non-zero second-order term
    assert res.second[(0, 2)] == pytest.approx(ref["second"][(0, 2)], rel=0.10)
    assert abs(res.second[(0, 1)]) < 0.02
    # both first-order estimators are reported and agree crudely
    assert np.max(res.estimator_spread) < 0.05


def test_deterministic_for_a_fixed_seed() -> None:
    a = saltelli_indices(_ishigami_on_unit, 3, n_base=1024, seed=21, second_order=False)
    b = saltelli_indices(_ishigami_on_unit, 3, n_base=1024, seed=21, second_order=False)
    assert np.array_equal(a.first, b.first)
    assert np.array_equal(a.total, b.total)
    c = saltelli_indices(_ishigami_on_unit, 3, n_base=1024, seed=22, second_order=False)
    assert not np.array_equal(a.first, c.first)


def test_bootstrap_intervals_bracket_estimates() -> None:
    res = saltelli_indices(
        _ishigami_on_unit, 3, n_base=4096, seed=8, bootstrap=40, second_order=False
    )
    assert res.first_ci is not None
    lo, hi = res.first_ci
    assert np.all(lo <= res.first) and np.all(res.first <= hi)
    lo_t, hi_t = res.total_ci
    assert np.all(lo_t <= res.total) and np.all(res.total <= hi_t)


def test_zero_variance_model_is_rejected() -> None:
    with pytest.raises(ValueError, match="zero variance"):
        saltelli_indices(lambda u: np.full(u.shape[0], 2.5), 2, n_base=32)


def test_non_finite_output_is_rejected() -> None:
    def bad(unit: np.ndarray) -> np.ndarray:
        y = _ishigami_on_unit(unit)
        y[0] = np.nan
        return y

    with pytest.raises(ValueError, match="non-finite"):
        saltelli_indices(bad, 3, n_base=64)


def test_argument_validation() -> None:
    with pytest.raises(ValueError):
        saltelli_indices(_ishigami_on_unit, 2, n_base=1)
    with pytest.raises(ValueError):
        saltelli_design(0, 16)
    with pytest.raises(TypeError):
        saltelli_indices("not callable", 2)  # type: ignore[arg-type]


def test_model_evaluation_count_is_reported() -> None:
    res = saltelli_indices(_ishigami_on_unit, 3, n_base=256, seed=1, second_order=True)
    assert res.n_model_evaluations == 256 * (2 + 3 + 3)
    res2 = saltelli_indices(_ishigami_on_unit, 3, n_base=256, seed=1, second_order=False)
    assert res2.n_model_evaluations == 256 * (2 + 3)


def test_to_dict_is_serialisable() -> None:
    import json

    res = saltelli_indices(_ishigami_on_unit, 3, n_base=512, seed=2, second_order=False)
    payload = res.as_dict(("a", "b", "c"))
    assert json.loads(json.dumps(payload))["n_base"] == 512
    assert set(payload["first_jansen"]) == {"a", "b", "c"}


def test_unit_design_matches_sampler_convention() -> None:
    # A and B partition one 2*n_base scrambled-Sobol' design (row-permuted)
    blocks = saltelli_design(2, 32, seed=5)
    ref = draw_unit(64, 2, seed=5)
    for k in range(2):
        pooled = np.sort(np.vstack([blocks["A"], blocks["B"]])[:, k])
        np.testing.assert_allclose(pooled, np.sort(ref[:, k]), atol=0.0)
    assert blocks["A"].min() >= 0.0 and blocks["A"].max() < 1.0