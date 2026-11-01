"""Pressure melting of ice Ih: Clausius-Clapeyron verification."""

import pytest

from tensorphysics.ice_contact.pressure_melting import (
    clapeyron_slope,
    integrate_melt_temperature,
    melt_point_depression,
    melt_temperature,
    melt_temperature_linear,
    pressure_melting_threshold,
    superheat,
)
from tensorphysics.ice_contact.properties import IceWaterProperties

PROPS = IceWaterProperties()


def test_clapeyron_slope_matches_literature_value():
    """Slope dT/dp must reproduce the tabulated -0.0742 K/MPa within 1 %."""
    slope = clapeyron_slope(PROPS) * 1.0e6
    assert slope < 0.0
    assert abs(abs(slope) - 0.0742) / 0.0742 < 0.01


def test_rk4_integration_matches_analytic_integral():
    """Numerical march of dT/dp against T(p) = T0 exp(dv p / Lf): error < 1 %."""
    for pressure in (1.0e6, 1.0e7, 5.0e7, 1.0e8):
        _, temperatures = integrate_melt_temperature(pressure, 200, PROPS)
        analytic = melt_temperature(pressure, PROPS)
        assert abs(temperatures[-1] - analytic) / analytic < 0.01


def test_linear_approximation_is_accurate_at_moderate_pressure():
    """The tangent-line form is within 1 % of the exact integral below 10 MPa."""
    for pressure in (1.0e6, 5.0e6, 1.0e7):
        linear = melt_temperature_linear(pressure, PROPS)
        exact = melt_temperature(pressure, PROPS)
        assert abs(linear - exact) / exact < 0.01


def test_melt_point_depression_is_positive_and_monotone():
    values = [melt_point_depression(p, PROPS) for p in (0.0, 1.0e6, 1.0e7, 1.0e8)]
    assert values[0] == pytest.approx(0.0, abs=1e-12)
    assert all(b > a for a, b in zip(values, values[1:]))
    assert values[1] == pytest.approx(0.0742, rel=0.01)


def test_pressure_melting_threshold_round_trip():
    """The threshold pressure must depress the melting point onto the ice temperature."""
    for ice_temperature in (268.15, 263.15, 253.15):
        threshold = pressure_melting_threshold(ice_temperature, PROPS)
        assert melt_temperature(threshold, PROPS) == pytest.approx(ice_temperature, rel=1e-9)
        assert superheat(threshold, ice_temperature, PROPS) == pytest.approx(0.0, abs=1e-9)
    assert pressure_melting_threshold(273.15, PROPS) == 0.0


def test_superheat_sign_switches_at_the_threshold():
    threshold = pressure_melting_threshold(263.15, PROPS)
    assert superheat(0.5 * threshold, 263.15, PROPS) < 0.0
    assert superheat(1.05 * threshold, 263.15, PROPS) > 0.0


def test_invalid_arguments_are_rejected():
    with pytest.raises(ValueError):
        melt_temperature(-1.0, PROPS)
    with pytest.raises(ValueError):
        integrate_melt_temperature(1.0e6, 0, PROPS)
    with pytest.raises(ValueError):
        pressure_melting_threshold(-5.0, PROPS)