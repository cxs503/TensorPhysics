"""Tests for the cross-solver agreement gate.

Every expected number below is hand-recomputed from the documented formulas
(inverse-variance weighted consensus, largest pairwise spread, sigma-based
outlier test), so the table is an independent check of the implementation,
not a snapshot.
"""
from __future__ import annotations

import math

import pytest

from tensorphysics.vv.cross_solver import (
    FAIL,
    PASS,
    WARN,
    AgreementPolicy,
    SolverSample,
    agreement_spread,
    consensus,
    cross_solver_agreement,
)


def S(solver, f, u=None, h=None):
    return SolverSample(solver=solver, f=f, u=u, h=h)


# (label, samples, expected status, expected consensus, expected spread_rel, expected outliers)
HAND_TABLE = [
    (
        "identical",
        [S("A", 1.0, 0.01), S("B", 1.0, 0.01)],
        PASS,
        1.0,
        0.0,
        (),
    ),
    (
        "tight",
        [S("A", 1.00, 0.01), S("B", 1.01, 0.01)],
        PASS,
        1.005,
        0.01 / 1.005,
        (),
    ),
    (
        "warn",
        [S("A", 1.00, 0.01), S("B", 1.03, 0.01)],
        WARN,
        1.015,
        0.03 / 1.015,
        (),
    ),
    (
        "fail_spread",
        [S("A", 1.00, 0.02), S("B", 1.06, 0.02)],
        FAIL,
        1.03,
        0.06 / 1.03,
        (),
    ),
    (
        "outlier",
        [S("A", 1.00, 0.001), S("B", 1.00, 0.001), S("C", 1.005, 0.001)],
        FAIL,
        (1.00 + 1.00 + 1.005) / 3.0,
        0.005 / ((1.00 + 1.00 + 1.005) / 3.0),
        ("C",),
    ),
    (
        "weighted",
        [S("A", 1.0, 0.01), S("B", 1.02, 0.02)],
        PASS,
        1.004,
        0.02 / 1.004,
        (),
    ),
    (
        "zero_consensus",
        [S("A", -0.5, 0.01), S("B", 0.5, 0.01)],
        FAIL,
        0.0,
        None,
        ("A", "B"),
    ),
    (
        "plain_mean",
        [S("A", 1.0), S("B", 1.02), S("C", 1.04)],
        WARN,
        1.02,
        0.04 / 1.02,
        (),
    ),
]


@pytest.mark.parametrize(
    "label,samples,status,consensus_value,spread_rel,outliers",
    HAND_TABLE,
    ids=[row[0] for row in HAND_TABLE],
)
def test_hand_table(label, samples, status, consensus_value, spread_rel, outliers):
    result = cross_solver_agreement(samples, quantity="drag")
    assert result.status == status, (label, result.status, status)
    assert result.consensus == pytest.approx(consensus_value, rel=1e-12, abs=1e-15)
    if spread_rel is None:
        assert result.spread_rel is None
    else:
        assert result.spread_rel == pytest.approx(spread_rel, rel=1e-12, abs=1e-15)
    assert result.outliers == outliers
    assert result.quantity == "drag"
    assert result.n_solvers == len(samples)
    assert result.solvers == tuple(s.solver for s in samples)


def test_weighted_consensus_is_inverse_variance():
    # A: f=1.0 u=0.01 (w=1e4); B: f=1.02 u=0.02 (w=2.5e3)
    # f_c = (1e4*1.0 + 2.5e3*1.02)/(1.25e4) = 1.004
    # u_c = 1/sqrt(1.25e4) = 0.008944271909999159
    f_c, u_c = consensus([S("A", 1.0, 0.01), S("B", 1.02, 0.02)])
    assert f_c == pytest.approx(1.004, rel=1e-14)
    assert u_c == pytest.approx(1.0 / math.sqrt(12500.0), rel=1e-14)


def test_plain_mean_when_no_uncertainty():
    f_c, u_c = consensus([S("A", 1.0), S("B", 1.02), S("C", 1.04)])
    assert f_c == pytest.approx(1.02, rel=1e-14)
    assert u_c is None


def test_zero_consensus_falls_back_to_absolute_band():
    result = cross_solver_agreement([S("A", -0.5, 0.01), S("B", 0.5, 0.01)])
    assert result.consensus == pytest.approx(0.0, abs=1e-15)
    assert result.spread_rel is None
    assert result.spread == pytest.approx(1.0, rel=1e-15)
    assert result.status == FAIL


def test_outlier_fails_even_when_spread_band_is_small():
    # two solvers agree tightly; C is 3.33 sigma off -> FAIL via outlier check
    result = cross_solver_agreement(
        [S("A", 1.00, 0.001), S("B", 1.00, 0.001), S("C", 1.005, 0.001)],
        quantity="lift",
    )
    assert result.status == FAIL
    assert result.outliers == ("C",)
    # the spread band itself is well inside the warn threshold
    assert result.spread_rel < 0.02
    assert any(c["check"] == "outlier" and c["status"] == FAIL for c in result.checks)


def test_reasons_are_nonpass_details():
    result = cross_solver_agreement([S("A", 1.00, 0.01), S("B", 1.06, 0.01)])
    assert result.status == FAIL
    assert len(result.reasons) >= 1
    assert all(isinstance(r, str) and r for r in result.reasons)


def test_agreement_spread_reports_widest_pair():
    spread, spread_rel, pair = agreement_spread(
        [S("A", 1.00), S("B", 1.01), S("C", 1.05)]
    )
    assert spread == pytest.approx(0.05, rel=1e-14)
    assert pair == ("A", "C")
    assert spread_rel == pytest.approx(0.05 / 1.02, rel=1e-14)


def test_needs_at_least_two_solvers():
    with pytest.raises(ValueError):
        cross_solver_agreement([S("A", 1.0)])
    with pytest.raises(ValueError):
        consensus([S("A", 1.0, 0.01)])


def test_duplicate_solver_names_rejected():
    with pytest.raises(ValueError):
        cross_solver_agreement([S("A", 1.0, 0.01), S("A", 1.01, 0.01)])


def test_mixed_uncertainty_rejected():
    with pytest.raises(ValueError):
        cross_solver_agreement([S("A", 1.0, 0.01), S("B", 1.01)])


def test_policy_validation():
    with pytest.raises(ValueError):
        AgreementPolicy(warn_spread=0.05, fail_spread=0.02)
    with pytest.raises(ValueError):
        AgreementPolicy(outlier_sigma=0.0)
    with pytest.raises(ValueError):
        AgreementPolicy(warn_spread=-1.0)


def test_sample_validation():
    with pytest.raises(ValueError):
        S("", 1.0)
    with pytest.raises(ValueError):
        S("A", float("nan"))
    with pytest.raises(ValueError):
        S("A", 1.0, u=-1.0)
    with pytest.raises(ValueError):
        S("A", 1.0, h=0.0)


def test_from_mapping_and_roundtrip():
    s = SolverSample.from_mapping(
        {"solver": "TensorLBM", "f": 1.0, "u": 0.01, "h": 0.001, "label": "x"}
    )
    assert (s.solver, s.f, s.u, s.h, s.label) == ("TensorLBM", 1.0, 0.01, 0.001, "x")
    assert s.as_dict() == {
        "solver": "TensorLBM",
        "f": 1.0,
        "u": 0.01,
        "h": 0.001,
        "label": "x",
    }
    # aliases
    s2 = SolverSample.from_mapping({"name": "X", "value": 2.0})
    assert (s2.solver, s2.f) == ("X", 2.0)
    with pytest.raises(ValueError):
        SolverSample.from_mapping({"f": 1.0})
    with pytest.raises(ValueError):
        SolverSample.from_mapping({"solver": "A"})


def test_integration_four_solver_drag():
    # four solvers on a drag coefficient, all within ~0.0001 -> PASS
    samples = [
        S("TensorLBM", 0.0120, 0.0004),
        S("TensorFVM", 0.0120, 0.0005),
        S("TensorFEM", 0.0121, 0.0003),
        S("TensorBEM", 0.0120, 0.0006),
    ]
    result = cross_solver_agreement(samples, quantity="Cd")
    assert result.status == PASS
    assert result.n_solvers == 4
    assert result.solvers == ("TensorLBM", "TensorFVM", "TensorFEM", "TensorBEM")
    assert result.consensus_u is not None and result.consensus_u > 0.0
    assert result.consensus == pytest.approx(0.0120, rel=1e-2)
    assert result.spread_rel is not None and result.spread_rel < 0.02
    assert result.outliers == ()
    d = result.as_dict()
    assert d["schema"] == "tensorphysics.vv.cross_solver/1"
    assert d["status"] == PASS
    assert set(d) >= {"consensus", "consensus_u", "spread", "spread_rel", "checks", "policy"}
