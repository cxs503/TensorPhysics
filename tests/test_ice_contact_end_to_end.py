"""End-to-end acceptance harness for the ice/structure contact model."""

import json
import math

import pytest

from tensorphysics.ice_contact.verification import (
    case_studies,
    end_to_end_case,
    run_all_criteria,
    verify_competition_gates,
    verify_friction_melting_steady,
    verify_pressure_melting,
)


def test_specified_case_returns_finite_histories():
    """1 MPa / 0.1 m/s / -10 degC: film and friction histories must be usable."""
    case = end_to_end_case(t_end=5.0e-3, n_samples=51)
    history = case["history"]
    assert len(history["time"]) == 51
    for key in ("film_thickness", "friction_coefficient", "friction_melting", "refreezing"):
        assert all(math.isfinite(value) for value in history[key])
    assert all(h >= 0.0 for h in history["film_thickness"])
    assert all(0.0 < mu <= 1.0 for mu in history["friction_coefficient"])
    assert case["melt_point_depression"] == pytest.approx(0.0742, abs=5e-4)


def test_specified_case_is_refreezing_dominated():
    case = end_to_end_case(t_end=5.0e-3, n_samples=51)
    rates = case["steady_rates"]
    assert rates["refreezing"] > rates["friction_melting"]
    assert case["steady_state"]["exists"] is False
    assert case["film_lifetime"] is not None and case["film_lifetime"] < 5.0e-3


def test_companion_cases_exercise_the_other_two_branches():
    cases = {case["name"]: case for case in case_studies()}
    friction = cases["friction_melting_1MPa_1mps_253K"]
    assert friction["steady_state"]["exists"] is True
    assert friction["steady_state"]["film_thickness"] > 1.0e-7
    pressure = cases["pressure_melting_20MPa_5mmps_272p5K"]
    assert pressure["steady_state"]["exists"] is True
    assert pressure["steady_rates"]["pressure_melting"] > 0.0


def test_criteria_a_and_b_pass():
    assert verify_pressure_melting()["passed"] is True
    assert verify_friction_melting_steady()["passed"] is True


def test_criteria_d_passes():
    assert verify_competition_gates()["passed"] is True


def test_full_harness_is_json_serialisable():
    results = run_all_criteria()
    payload = json.dumps(results, sort_keys=True)
    assert len(payload) > 1000
    assert results["summary"]["all_passed"] is True
    assert set(results["summary"]["criteria"]) == {
        "a_pressure_melting",
        "b_friction_melting_steady",
        "c_refreezing_stefan",
        "d_competition_gates",
        "e_end_to_end",
    }