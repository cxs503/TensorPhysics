"""Tests for the flexural-gravity dispersion module (W5-N05)."""

from __future__ import annotations

import numpy as np
import pytest

from tensorphysics.ice_wave import flexural


D_ICE = flexural.flexural_rigidity(1.5)


@pytest.mark.parametrize(
    "k,h",
    [(0.001, 2.0), (0.005, 5.0), (0.02, 100.0), (0.1, 1000.0), (0.5, 1000.0), (2.0, 5000.0)],
)
def test_spectral_vs_analytical_residual_below_1e12(k, h):
    """Acceptance criterion (a): spectral BVP vs analytical dispersion < 1e-12."""
    assert flexural.spectral_residual(k, h, D_ICE, n_modes=32) < 1e-12


def test_depth_truncation_error_negligible():
    """Truncating the depth at k*h = 18 changes omega^2 by less than 1e-13."""
    for k, h in [(0.5, 1000.0), (2.0, 5000.0)]:
        assert flexural.depth_truncation_error(k, h, D_ICE) < 1e-13


def test_residual_improves_with_more_modes():
    coarse = flexural.spectral_residual(0.5, 1000.0, D_ICE, n_modes=12)
    fine = flexural.spectral_residual(0.5, 1000.0, D_ICE, n_modes=32)
    assert fine < coarse


def test_shallow_water_limit():
    """For k h -> 0 the dispersion must approach omega^2 = g k^2 h."""
    # k h = 1e-5: the leading correction is (k h)^2 / 3 = 3e-11.
    k, h = 1e-5, 1.0
    omega2 = flexural.flexural_gravity_dispersion(k, h, D_ICE)
    reference = flexural.GRAVITY * k**2 * h
    assert abs(omega2 - reference) / reference < 1e-9


def test_deep_water_flexural_limit():
    """For k h -> infinity: omega^2 = g k + D k^5 / rho_w."""
    k, h = 0.05, 5000.0
    omega2 = flexural.flexural_gravity_dispersion(k, h, D_ICE)
    reference = flexural.GRAVITY * k + D_ICE * k**5 / flexural.RHO_WATER
    assert abs(omega2 - reference) / reference < 1e-12


def test_zero_rigidity_recovers_open_water_wave():
    k, h = 0.1, 20.0
    omega2 = flexural.flexural_gravity_dispersion(k, h, 0.0)
    assert omega2 == pytest.approx(flexural.GRAVITY * k * np.tanh(k * h), rel=1e-15)


def test_group_speed_matches_finite_difference_of_omega():
    for k in (0.005, 0.02, 0.05, 0.2, 0.5):
        dk = 1e-6 * k
        fd = (
            np.sqrt(flexural.flexural_gravity_dispersion(k + dk, 500.0, D_ICE))
            - np.sqrt(flexural.flexural_gravity_dispersion(k - dk, 500.0, D_ICE))
        ) / (2 * dk)
        analytic = flexural.group_speed(k, 500.0, D_ICE)
        assert abs(float(fd) - analytic) / analytic < 1e-8


def test_phase_speed_minimum_matches_analytic_wavenumber():
    """Deep-water flexural branch: c^2 = g/k + D k^3 / rho, minimum at
    k = (rho g / (3 D))^(1/4)."""
    k_analytic = (flexural.RHO_WATER * flexural.GRAVITY / (3.0 * D_ICE)) ** 0.25
    grid = np.logspace(np.log10(k_analytic) - 2.0, np.log10(k_analytic) + 2.0, 20001)
    speeds = np.array(
        [flexural.phase_speed(k, 1.0e5, D_ICE) for k in grid]
    )
    k_numeric = float(grid[int(np.argmin(speeds))])
    assert abs(k_numeric - k_analytic) / k_analytic < 1e-3


def test_wavenumber_inverts_dispersion():
    omega, h = 0.628, 100.0
    k = flexural.flexural_gravity_wavenumber(omega, h, D_ICE)
    assert flexural.flexural_gravity_dispersion(k, h, D_ICE) == pytest.approx(omega**2, rel=1e-12)
    # deep-water gravity asymptote for small rigidity contributions
    k_gravity = flexural.flexural_gravity_wavenumber(0.628, 1.0e5, 0.0)
    assert k_gravity == pytest.approx(0.628**2 / flexural.GRAVITY, rel=1e-10)


def test_rigidity_formula_and_units():
    D = flexural.flexural_rigidity(2.0, 5.0e9, 0.3)
    expected = 5.0e9 * 8.0 / (12.0 * (1.0 - 0.09))
    assert D == pytest.approx(expected, rel=1e-15)
    # Dimension check: D k^5 / rho has the same units as g k (1/s^2).
    k = 0.1
    assert D * k**5 / flexural.RHO_WATER > 0.0
    assert np.isfinite(np.sqrt(flexural.flexural_gravity_dispersion(k, 10.0, D)))


@pytest.mark.parametrize(
    "call",
    [
        lambda: flexural.flexural_gravity_dispersion(0.0, 10.0, D_ICE),
        lambda: flexural.flexural_gravity_dispersion(0.1, -1.0, D_ICE),
        lambda: flexural.flexural_gravity_dispersion(0.1, 10.0, -5.0),
        lambda: flexural.flexural_gravity_dispersion(float("nan"), 10.0, D_ICE),
        lambda: flexural.flexural_rigidity(0.0),
        lambda: flexural.flexural_rigidity(1.0, poisson_ratio=0.7),
        lambda: flexural.spectral_dispersion(0.1, 10.0, D_ICE, n_modes=2),
        lambda: flexural.flexural_gravity_wavenumber(-1.0, 10.0, D_ICE),
    ],
)
def test_invalid_inputs_raise(call):
    with pytest.raises(flexural.FlexuralWaveError):
        call()