"""Temperature dependence of ice adhesion through the interfacial premelted layer."""

import pytest

from tensorphysics.ice_contact.adhesion import (
    IceAdhesionModel,
    qll_thickness,
    vdw_bridging_stress,
)
from tensorphysics.ice_contact.properties import IceWaterProperties

PROPS = IceWaterProperties()
MODEL = IceAdhesionModel()


def test_adhesion_peaks_at_the_documented_supercooling():
    temperatures = [PROPS.melting_temperature - d for d in (1, 2, 5, 8, 9.9, 10, 10.1, 15, 25, 40)]
    values = [MODEL.adhesion_strength(t) for t in temperatures]
    peak_index = values.index(max(values))
    assert temperatures[peak_index] == pytest.approx(MODEL.peak_temperature(), abs=0.11)


def test_warm_branch_is_linear_in_supercooling():
    """Below the peak, sigma ~ (T_m - T) with the premelting scaling exponent 1."""
    ratios = []
    for depression in (2.0, 4.0, 6.0, 8.0):
        sigma = MODEL.adhesion_strength(PROPS.melting_temperature - depression)
        ratios.append(sigma / depression)
    assert max(ratios) == pytest.approx(min(ratios), rel=1e-9)
    assert MODEL.adhesion_strength(PROPS.melting_temperature - 5.0) == pytest.approx(
        MODEL.peak_strength * 0.5, rel=1e-9
    )


def test_adhesion_magnitude_on_steel_is_in_the_measured_range():
    """Ice-on-steel shear strength at -10 degC is reported as 0.2-1.5 MPa."""
    sigma = MODEL.adhesion_strength(PROPS.melting_temperature - 10.0)
    assert 0.2e6 <= sigma <= 1.5e6
    assert sigma == pytest.approx(0.9e6, rel=1e-9)


def test_adhesion_collapses_at_the_melting_point_and_keeps_a_floor():
    assert MODEL.adhesion_strength(PROPS.melting_temperature) == 0.0
    assert MODEL.adhesion_strength(PROPS.melting_temperature + 1.0) == 0.0
    sigma_warm = MODEL.adhesion_strength(PROPS.melting_temperature - 0.5)
    assert sigma_warm < 0.1 * MODEL.peak_strength
    sigma_cold = MODEL.adhesion_strength(PROPS.melting_temperature - 120.0)
    assert sigma_cold == pytest.approx(MODEL.floor_fraction * MODEL.peak_strength, rel=1e-9)


def test_adhesion_is_monotone_on_both_branches():
    warm = [MODEL.adhesion_strength(PROPS.melting_temperature - d) for d in (0.5, 2, 5, 8, 10)]
    assert all(b > a for a, b in zip(warm, warm[1:]))
    cold = [MODEL.adhesion_strength(PROPS.melting_temperature - d) for d in (10.5, 15, 20, 30)]
    assert all(b < a for a, b in zip(cold, cold[1:]))


def test_qll_thickness_follows_the_premelting_power_law():
    """h ~ (T_m - T)^(-1/3): the log-log slope must be -1/3."""
    from math import log

    d1, d2 = 1.0, 10.0
    h1 = qll_thickness(PROPS.melting_temperature - d1, PROPS)
    h2 = qll_thickness(PROPS.melting_temperature - d2, PROPS)
    slope = log(h1 / h2) / log(d1 / d2)
    assert slope == pytest.approx(-1.0 / 3.0, rel=1e-9)
    assert 1e-10 < h2 < h1 < 1e-8


def test_vdw_bridging_stress_is_capped_and_increasing_with_supercooling():
    stresses = [
        vdw_bridging_stress(PROPS.melting_temperature - d, PROPS) for d in (0.2, 0.5, 1, 5, 20)
    ]
    assert all(b >= a for a, b in zip(stresses, stresses[1:]))
    assert max(stresses) <= 1.5e6


def test_boundary_friction_cap_scales_inversely_with_pressure():
    cap_low = MODEL.boundary_friction_cap(1.0e5, PROPS.melting_temperature - 10.0)
    cap_high = MODEL.boundary_friction_cap(1.0e7, PROPS.melting_temperature - 10.0)
    assert cap_low / cap_high == pytest.approx(100.0, rel=1e-9)


def test_invalid_adhesion_parameters_are_rejected():
    with pytest.raises(ValueError):
        IceAdhesionModel(peak_strength=-1.0)
    with pytest.raises(ValueError):
        IceAdhesionModel(floor_fraction=1.0)
    with pytest.raises(ValueError):
        qll_thickness(PROPS.melting_temperature + 1.0, PROPS)