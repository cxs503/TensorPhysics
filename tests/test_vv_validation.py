"""Tests for the ASME V&V 20 validation-uncertainty gate.

The ten cases below are hand-recomputed: the expected ``U_val``, the comparison
error and the PASS/FAIL verdict are written as literals computed from
``U_val = sqrt(U_num**2 + U_input**2)`` and ``|f_sim - f_exp| <= U_val``, so
the table is an independent check of the implementation, not a snapshot.
"""
from __future__ import annotations

import math

import pytest

from tensorphysics.vv import validate, validation_uncertainty
from tensorphysics.vv.validation import evaluate_validation_table

# (label, f_sim, f_exp, U_num, U_input, expected U_val, expected |E|, expected status)
HAND_TABLE = [
    ("case01", 1.00, 1.05, 0.03, 0.03, 0.04242640687119285, 0.050000000000000044, "FAIL"),
    ("case02", 1.00, 1.02, 0.02, 0.01, 0.022360679774997898, 0.020000000000000018, "PASS"),
    ("case03", 2.00, 2.00, 0.10, 0.10, 0.1414213562373095, 0.0, "PASS"),
    ("case04", 10.00, 10.50, 0.00, 0.50, 0.5, 0.5, "PASS"),          # |E| == U_val, inclusive
    ("case05", 10.00, 10.5000001, 0.00, 0.50, 0.5, 0.5000001, "FAIL"),
    ("case06", 0.50, 0.45, 0.02, 0.03, 0.03605551275463989, 0.050000000000000044, "FAIL"),
    ("case07", -1.00, -1.02, 0.01, 0.01, 0.014142135623730951, 0.020000000000000018, "FAIL"),
    ("case08", 100.0, 100.0, 5.00, 0.00, 5.0, 0.0, "PASS"),
    ("case09", 1.00, 1.00, 0.00, 0.00, 0.0, 0.0, "PASS"),            # exact match only
    ("case10", 1.00, 1.001, 0.00, 0.00, 0.0, 0.001, "FAIL"),         # zero band, non-zero error
]


def test_validation_uncertainty_is_the_quadrature_sum():
    assert validation_uncertainty(0.03, 0.04) == pytest.approx(0.05, rel=1e-15)
    assert validation_uncertainty(0.0, 0.0) == 0.0
    assert validation_uncertainty(7.0) == pytest.approx(7.0)


def test_validation_uncertainty_rejects_negative_inputs():
    with pytest.raises(ValueError):
        validation_uncertainty(-1.0, 0.1)
    with pytest.raises(ValueError):
        validation_uncertainty(1.0, float("nan"))


def test_hand_table_matches_the_implementation_exactly():
    results = evaluate_validation_table(
        [
            {"label": label, "f_sim": f_sim, "f_exp": f_exp, "u_num": u_num, "u_input": u_input}
            for label, f_sim, f_exp, u_num, u_input, _u, _e, _s in HAND_TABLE
        ]
    )
    assert len(results) == 10
    for result, row in zip(results, HAND_TABLE):
        label, f_sim, f_exp, u_num, u_input, u_expected, e_expected, status = row
        assert result.label == label
        assert result.u_val == pytest.approx(u_expected, rel=1e-12, abs=1e-15)
        assert abs(result.error) == pytest.approx(e_expected, rel=1e-12, abs=1e-15)
        assert result.status == status, (label, result.status, status)
        assert result.passed is (status == "PASS")


def test_hand_table_accepts_mapping_rows():
    rows = [
        {"label": label, "f_sim": f_sim, "f_exp": f_exp, "u_num": u_num, "u_input": u_input}
        for label, f_sim, f_exp, u_num, u_input, _u, _e, _s in HAND_TABLE
    ]
    results = evaluate_validation_table(rows)
    assert [r.status for r in results] == [row[7] for row in HAND_TABLE]


def test_boundary_is_inclusive():
    exact = validate(10.0, 10.5, u_num=0.3, u_input=0.4)  # U_val = 0.5, |E| = 0.5
    assert exact.ratio == pytest.approx(1.0)
    assert exact.passed is True
    beyond = validate(10.0, 10.5 + 1e-12, u_num=0.3, u_input=0.4)
    assert beyond.passed is False


def test_ratio_is_reported_for_marginal_cases():
    result = validate(1.0, 1.02, u_num=0.01, u_input=0.0)
    assert result.ratio == pytest.approx(2.0, rel=1e-12)
    assert result.status == "FAIL"
    assert result.relative_error == pytest.approx(-0.02 / 1.02, rel=1e-12)


def test_zero_band_with_exact_match_passes_and_diverges_otherwise():
    exact = validate(3.0, 3.0, u_num=0.0, u_input=0.0)
    assert exact.passed is True and exact.ratio == 0.0
    other = validate(3.0, 3.0000001, u_num=0.0, u_input=0.0)
    assert other.passed is False and math.isinf(other.ratio)


def test_table_rejects_incomplete_rows():
    with pytest.raises(ValueError):
        evaluate_validation_table([{"f_sim": 1.0, "f_exp": 1.0, "u_num": 0.1}])
    with pytest.raises(ValueError):
        evaluate_validation_table([(1.0, 1.0)])