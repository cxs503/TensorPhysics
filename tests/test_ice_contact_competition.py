"""Three-mechanism competition: decomposition, onset speed and the phase map."""

import pytest

from tensorphysics.ice_contact.adhesion import IceAdhesionModel
from tensorphysics.ice_contact.competition import (
    direction_gates,
    dominant_mechanism,
    melting_onset_velocity,
    melting_onset_velocity_analytic,
    scan_mechanism_map,
)
from tensorphysics.ice_contact.film import ContactState, FilmModel
from tensorphysics.ice_contact.pressure_melting import (
    melt_point_depression,
    melt_temperature,
    pressure_melting_threshold,
)
from tensorphysics.ice_contact.properties import IceWaterProperties

PROPS = IceWaterProperties()
MODEL = FilmModel(adhesion=IceAdhesionModel())


def _equilibrated(pressure: float, velocity: float, temperature: float) -> ContactState:
    return ContactState(
        pressure=pressure,
        velocity=velocity,
        ice_temperature=temperature,
        structure_temperature=melt_temperature(pressure),
    )


def test_onset_velocity_matches_its_closed_form():
    state = _equilibrated(1.0e6, 1.0, 263.15)
    numeric = melting_onset_velocity(MODEL, state)
    analytic = melting_onset_velocity_analytic(MODEL, state)
    assert numeric is not None
    assert 0.1 < analytic < 5.0
    assert abs(numeric - analytic) / analytic < 0.05


def test_onset_velocity_grows_for_a_cold_structure():
    """A cold slider adds a Stefan sink, so melting needs a much higher speed."""
    warm_slider = melting_onset_velocity(MODEL, _equilibrated(1.0e6, 1.0, 263.15))
    cold_slider = melting_onset_velocity(
        MODEL,
        ContactState(
            pressure=1.0e6, velocity=1.0, ice_temperature=263.15, structure_temperature=263.15
        ),
    )
    assert cold_slider > 5.0 * warm_slider


def test_dominant_mechanism_labels_follow_the_physics():
    cold = ContactState(
        pressure=1.0e6, velocity=0.05, ice_temperature=263.15, structure_temperature=263.15
    )
    assert dominant_mechanism(MODEL, cold, MODEL.film_floor) == "refreezing"
    hot = _equilibrated(1.0e6, 8.0, 263.15)
    assert dominant_mechanism(MODEL, hot, 1.0e-6) == "friction_melting"
    pressurised = _equilibrated(2.0e7, 0.002, 272.9)
    assert dominant_mechanism(MODEL, pressurised, 1.0e-6) == "pressure_melting"


def test_mechanism_map_covers_all_three_mechanisms():
    rows = scan_mechanism_map(
        MODEL,
        pressures=[0.5e6, 1.0e6, 4.0e6, 2.0e7],
        velocities=[0.002, 0.05, 0.5, 5.0],
        temperatures=[272.9, 268.15, 263.15, 248.15],
    )
    labels = {row["dominant_mechanism"] for row in rows}
    assert {"friction_melting", "pressure_melting", "refreezing"} <= labels
    assert all(row["friction_coefficient"] > 0.0 for row in rows)
    assert all(row["film_thickness"] >= 0.0 for row in rows)


def test_mechanism_map_records_pressure_melting_driving_force():
    rows = scan_mechanism_map(
        MODEL, pressures=[1.0e6, 1.5e8], velocities=[0.05], temperatures=[263.15]
    )
    low, high = rows
    assert low["melt_point_depression"] < 0.1
    assert high["melt_point_depression"] > 10.0
    assert low["superheat"] < 0.0 < high["superheat"]


def test_direction_gates_all_pass():
    gates = direction_gates(MODEL)
    assert set(gates) == {
        "G1_friction_melting_V",
        "G2_load_p",
        "G3_thermal_T",
        "G4_pressure_melting",
        "G5_mechanism_coverage",
    }
    failed = [name for name, gate in gates.items() if not gate["passed"]]
    assert failed == []


def test_pressure_melting_threshold_is_asperity_scale():
    threshold = pressure_melting_threshold(263.15, PROPS)
    assert 1.0e8 < threshold < 2.0e8
    assert melt_point_depression(threshold, PROPS) == pytest.approx(10.0, rel=1e-6)