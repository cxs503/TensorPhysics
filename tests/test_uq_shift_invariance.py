"""Regression test of the sampled-estimator defect fixed in mission IN145-N08.

Sobol' indices are invariant under the shift ``f -> f + c``.  The Jansen
first-order and the total-order estimators only use differences of model
outputs and were always invariant; the Saltelli (2010) first-order estimator and
the closed second-order estimator multiplied such a difference by a **raw** model
value, so the output offset leaked in as ``c * O(sigma / sqrt(N))``.

Measured on the pre-fix code (Ishigami, ``n_base = 2**14``, exact
``S_13 = 0.2437``):

    shift        first_saltelli                  S_13
    0            0.3034  0.4335 -0.0028          0.2559
    1e3         -2.039  -1.2971 -0.1204          1.4685
    1e4        -23.1212 -16.8725 -1.1783        12.3817

A ship resistance of O(1e3) kN, a pressure of O(1e5) Pa, or a lift of O(1e6) N
all sit in that regime, so the bug was reachable from the domain, not only from
a synthetic benchmark.  The test below fails on the unmodified baseline and
passes after centring the multiplier on the grand mean.
"""
from __future__ import annotations

import numpy as np
import pytest

from tensorphysics.uq import ishigami, ishigami_analytic, ishigami_spec
from tensorphysics.uq.sobol_mc import saltelli_indices


def _evaluator(shift: float):
    spec = ishigami_spec()

    def evaluate(unit: np.ndarray) -> np.ndarray:
        physical, _ = spec.transform(unit)
        return shift + ishigami(physical)

    return evaluate


def test_sampled_indices_are_shift_invariant() -> None:
    base = saltelli_indices(_evaluator(0.0), 3, n_base=1 << 13, seed=20240, second_order=True)
    shifted = saltelli_indices(_evaluator(1.0e4), 3, n_base=1 << 13, seed=20240, second_order=True)
    assert np.allclose(base.first, shifted.first, atol=1e-12)
    assert np.allclose(base.total, shifted.total, atol=1e-12)
    # the two Saltelli estimators are the ones that used a raw model value
    assert np.allclose(base.first_saltelli, shifted.first_saltelli, atol=1e-12)
    for key, value in base.second.items():
        assert shifted.second[key] == pytest.approx(value, abs=1e-12)


def test_shifted_estimators_stay_close_to_the_analytic_indices() -> None:
    analytic = ishigami_analytic()
    shifted = saltelli_indices(_evaluator(1.0e4), 3, n_base=1 << 14, seed=20240, second_order=True)
    assert abs(shifted.first[0] - analytic["first"][0]) < 0.05
    assert abs(shifted.first[1] - analytic["first"][1]) < 0.05
    assert abs(shifted.total[0] - analytic["total"][0]) < 0.06
    assert abs(shifted.total[2] - analytic["total"][2]) < 0.03
    s13 = shifted.second[(0, 2)]
    assert abs(s13 - analytic["second"][(0, 2)]) / analytic["second"][(0, 2)] < 0.10
    # the null second-order indices must stay near zero, not explode
    assert abs(shifted.second[(0, 1)]) < 0.02
    assert abs(shifted.second[(1, 2)]) < 0.02


def test_scaling_does_not_change_the_indices() -> None:
    spec = ishigami_spec()

    def scaled(unit: np.ndarray) -> np.ndarray:
        physical, _ = spec.transform(unit)
        return -7.5 * ishigami(physical)

    base = saltelli_indices(_evaluator(0.0), 3, n_base=1 << 13, seed=20240, second_order=False)
    flipped = saltelli_indices(scaled, 3, n_base=1 << 13, seed=20240, second_order=False)
    assert np.allclose(base.first, flipped.first, atol=1e-12)
    assert np.allclose(base.total, flipped.total, atol=1e-12)