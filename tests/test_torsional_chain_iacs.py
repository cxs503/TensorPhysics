"""C6 - IACS Polar Class screening, Miner fatigue; and C4 - ice torque transfer."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorphysics.torsional_chain.acceptance import c4_ice_torque_transfer, c6_iacs_check
from tensorphysics.torsional_chain.excitation import IceMillingTorque
from tensorphysics.torsional_chain.iacs import (POLAR_CLASS_TORSION_LIMITS,
                                                allowable_torsional_stress,
                                                exposure_life, fatigue_limit,
                                                iacs_torsion_check, miner_damage, sn_life)
from tensorphysics.torsional_chain.theory import sdof_amplitude


@pytest.fixture(scope="module")
def c4():
    return c4_ice_torque_transfer()


@pytest.fixture(scope="module")
def c6():
    return c6_iacs_check()


def test_sn_life_above_and_below_the_knee():
    knee = fatigue_limit(45.0e6, 2.0e6, 3.0, 1.0e7)
    assert knee == pytest.approx(45.0e6 * (0.2) ** (1.0 / 3.0), rel=1e-12)
    assert sn_life(90.0e6) == pytest.approx(2.0e6 * (0.5) ** 3)
    assert sn_life(30.0e6) == pytest.approx(2.0e6 * (1.5) ** 3)
    assert sn_life(0.5 * knee) == math.inf
    assert sn_life(knee * (1.0 + 1e-9)) < 1.0e7


def test_miner_damage_is_additive_and_life_inverts_it():
    damage = miner_damage([90.0e6, 30.0e6], [1.0e5, 1.0e6])
    expected = 1.0e5 / (2.0e6 * 0.125) + 1.0e6 / (2.0e6 * 3.375)
    assert damage == pytest.approx(expected, rel=1e-12)
    assert miner_damage([20.0e6], [1.0e12]) == 0.0  # below the knee


def test_exposure_life_hours_and_years():
    out = exposure_life(cycles_per_hour=1.0e4, stress_amplitude=90.0e6)
    assert out["cycles_to_failure"] == pytest.approx(2.5e5)
    assert out["life_hours"] == pytest.approx(25.0)
    assert out["life_years"] == pytest.approx(25.0 / 6000.0)


def test_allowable_stress_monotone_in_polar_class():
    values = [allowable_torsional_stress(pc) for pc in sorted(POLAR_CLASS_TORSION_LIMITS)]
    assert values == sorted(values)
    assert allowable_torsional_stress("PC2", transient=True) > allowable_torsional_stress("PC2")
    with pytest.raises(ValueError):
        allowable_torsional_stress("PC9")


def test_iacs_check_reports_model_parameter_status():
    out = iacs_torsion_check("PC3", 10.0e6, 1.0e4)
    assert out["parameter_status"] == "model_parameter"
    assert out["stress_ok"] is True
    assert out["life_ok"] is True
    assert out["stress_utilization"] == pytest.approx(10.0e6 / (34.0e6))


def test_iacs_check_flags_overload_and_short_life():
    out = iacs_torsion_check("PC1", 200.0e6, 1.0e6)
    assert out["stress_ok"] is False
    assert out["life_ok"] is False
    assert out["life_years"] < 20.0


def test_ice_torque_spectrum_shape_and_scaling():
    ice = IceMillingTorque.from_ice_condition(0.5, 1.5, 120.0, 4)
    assert ice.frequencies[0] == pytest.approx(8.0)
    assert ice.mean_torque == pytest.approx(6.0e4 * 0.5 * 1.5)
    assert ice.amplitudes[0] == pytest.approx(1.2e4 * 0.5 * 1.5)
    assert ice.amplitudes[1] < ice.amplitudes[0]
    # analytic one-sided line amplitude carries the phase
    amp = ice.spectrum_amplitude(8.0)
    assert abs(amp) == pytest.approx(ice.amplitudes[0], rel=1e-12)
    grid = np.linspace(0.0, 4.0, 4001)
    assert np.mean(ice.torque(grid)) == pytest.approx(ice.mean_torque, abs=0.01 * ice.mean_torque)


def test_ice_torque_rejects_bad_conditions():
    with pytest.raises(ValueError):
        IceMillingTorque.from_ice_condition(0.0, 1.5, 120.0, 4)
    with pytest.raises(ValueError):
        IceMillingTorque(0.0, 0.0, 1.0e4)


def test_frf_matches_closed_form_at_every_frequency(c4):
    assert c4["criterion"] == "C4"
    assert c4["worst_frf_error"] <= 0.03
    for row in c4["frf_cases"]:
        assert row["amplitude_rad"] > 0.0


def test_frf_peak_position_matches_theory(c4):
    assert c4["pass"] is True
    assert c4["peak_position_error"] <= 0.03
    assert c4["peak_frequency_hz"] == pytest.approx(c4["peak_frequency_theory_hz"], rel=0.03)


def test_ice_line_spectrum_follows_the_transfer_function(c4):
    assert c4["worst_ice_line_error"] <= 0.03
    # each synthetic ice line must reproduce |H(omega)| * T_n
    for line in c4["ice_lines"]:
        assert line["amplitude_rad"] == pytest.approx(line["amplitude_analytic"], rel=0.03)


def test_transfer_amplitude_equals_analytic_expression():
    """Sanity anchor independent of the chain: |H| at resonance is 1/(c omega_n)."""
    k, j, c = 3.9e7, 1.2e3, 21633.0
    omega_n = math.sqrt(k / j)
    assert sdof_amplitude(omega_n, j, k, c) == pytest.approx(1.0 / (c * omega_n), rel=2e-3)


def test_iacs_three_polar_classes_pass(c6):
    assert c6["criterion"] == "C6"
    assert c6["pass"] is True
    assert [row["polar_class"] for row in c6["polar_class_checks"]] == ["PC2", "PC4", "PC6"]
    for row in c6["polar_class_checks"]:
        assert row["stress_ok"] is True
        assert row["life_ok"] is True


def test_iacs_gate_is_not_vacuous(c6):
    """A hypothetical resonant ice-milling case must be rejected by the same gate."""
    assert c6["severe_discriminates"] is True
    severe = c6["severe_resonance_case"]
    assert severe["stress_ok"] is False or severe["life_ok"] is False