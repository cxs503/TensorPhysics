"""Stribeck friction curve extended with the melt-film parameter."""

import pytest

from tensorphysics.ice_contact.properties import IceWaterProperties
from tensorphysics.ice_contact.tribology import (
    StribeckParams,
    asperity_load_fraction,
    film_parameter,
    friction_coefficient,
    lubrication_regime,
    stribeck_curve,
    stribeck_curve_minimum,
    viscous_friction_coefficient,
)

PROPS = IceWaterProperties()
PARAMS = StribeckParams()


def test_lubrication_regime_thresholds():
    assert lubrication_regime(0.5e-6, PARAMS) == "boundary"
    assert lubrication_regime(2.0e-6, PARAMS) == "mixed"
    assert lubrication_regime(10.0e-6, PARAMS) == "hydrodynamic"
    assert film_parameter(2.0e-6, 1.0e-6) == pytest.approx(2.0)


def test_asperity_load_fraction_decays_and_saturates():
    values = [asperity_load_fraction(h, PARAMS) for h in (1e-9, 1e-7, 1e-6, 1e-5)]
    assert all(b < a for a, b in zip(values, values[1:]))
    assert values[0] > 0.999
    assert values[-1] < 1e-3


def test_viscous_friction_scales_with_velocity_and_inverse_film():
    base = viscous_friction_coefficient(1.0e-6, 1.0e6, 1.0, 273.15, PROPS, PARAMS)
    faster = viscous_friction_coefficient(1.0e-6, 1.0e6, 2.0, 273.15, PROPS, PARAMS)
    thicker = viscous_friction_coefficient(2.0e-6, 1.0e6, 1.0, 273.15, PROPS, PARAMS)
    heavier = viscous_friction_coefficient(1.0e-6, 2.0e6, 1.0, 273.15, PROPS, PARAMS)
    assert faster / base == pytest.approx(2.0, rel=1e-9)
    assert thicker / base == pytest.approx(0.5, rel=1e-9)
    assert heavier / base == pytest.approx(0.5, rel=1e-9)


def test_friction_coefficient_between_the_floor_and_the_boundary_value():
    kwargs = dict(pressure=1.0e6, velocity=1.0, temperature=263.15, params=PARAMS, props=PROPS)
    thin = friction_coefficient(1e-9, **kwargs)
    thick = friction_coefficient(1e-4, **kwargs)
    # at the film floor the asperity share is ~0.999, so the residual viscous
    # contribution keeps the coefficient within ~3 % of the dry boundary value
    assert thin == pytest.approx(PARAMS.mu_boundary, rel=0.03)
    assert thick == pytest.approx(PARAMS.mu_floor, rel=1e-9)
    assert thick < friction_coefficient(1e-6, **kwargs) < thin


def test_friction_coefficient_is_monotone_decreasing_in_film_thickness():
    values = [
        friction_coefficient(h, 1.0e6, 1.0, 263.15, PARAMS, PROPS)
        for h in (1e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5)
    ]
    assert all(b <= a * (1.0 + 1e-12) for a, b in zip(values, values[1:]))


def test_viscosity_reduces_the_viscous_branch_with_warming():
    cold = viscous_friction_coefficient(1.0e-6, 1.0e6, 1.0, 273.15, PROPS, PARAMS)
    warm = viscous_friction_coefficient(1.0e-6, 1.0e6, 1.0, 293.15, PROPS, PARAMS)
    assert warm < cold


def test_stribeck_curve_has_an_interior_minimum():
    hersey = [0.001 * i for i in range(1, 1500)]
    values = [stribeck_curve(h) for h in hersey]
    index = values.index(min(values))
    assert 0 < index < len(values) - 1
    assert all(b < a for a, b in zip(values[: index + 1], values[1: index + 1]))
    assert all(b > a for a, b in zip(values[index:], values[index + 1:]))
    h_star, mu_star = stribeck_curve_minimum()
    assert h_star == pytest.approx(hersey[index], rel=0.05)
    assert mu_star == pytest.approx(min(values), rel=1e-3)


def test_invalid_tribology_arguments_are_rejected():
    with pytest.raises(ValueError):
        StribeckParams(mu_boundary=-0.1)
    with pytest.raises(ValueError):
        StribeckParams(boundary_limit=3.0, hydrodynamic_limit=1.0)
    with pytest.raises(ValueError):
        film_parameter(-1.0, 1e-6)
    with pytest.raises(ValueError):
        friction_coefficient(1e-6, 0.0, 1.0, 263.15, PARAMS, PROPS)