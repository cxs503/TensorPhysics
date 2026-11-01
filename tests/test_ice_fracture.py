"""Flexural fracture criterion, crack location and ring-crack radius."""
import numpy as np
import pytest

from tensorphysics.ice.fracture import (
    bottom_stress_field,
    crack_initiation,
    fracture_energy,
    max_tensile_stress,
    principal_stress,
    ring_crack_radius,
    top_stress_field,
)


def _bowl(E=9.0e9, h=1.5, nu=0.3, curv=2.0e-5, size=64, extent=100.0):
    """Sagging bowl: w = -curv * r^2 / 2 with w > 0 downwards."""
    x = np.linspace(-extent, extent, size)
    X, Y = np.meshgrid(x, x, indexing="xy")
    der = {"xx": np.full_like(X, -curv), "yy": np.full_like(X, -curv),
           "xy": np.zeros_like(X)}
    return X, Y, der


def test_stress_sign_and_isotropy_of_a_bowl():
    E, h, nu = 9.0e9, 1.5, 0.3
    X, Y, der = _bowl(E, h, nu)
    bot = bottom_stress_field(der, E, h, nu)
    top = top_stress_field(der, E, h, nu)
    C = E * h / (2.0 * (1.0 - nu**2))
    assert bot.sigma_xx[0, 0] == pytest.approx(C * (1.0 + nu) * 2.0e-5)
    # a sagging plate has its bottom in tension and the top in compression
    assert bot.sigma_xx[0, 0] > 0.0
    assert top.sigma_xx[0, 0] == pytest.approx(-bot.sigma_xx[0, 0])
    s1, s2, theta = principal_stress(bot)
    assert s1[0, 0] == pytest.approx(bot.sigma_xx[0, 0], rel=1e-9)
    assert s2[0, 0] == pytest.approx(s1[0, 0], rel=1e-9)


def test_principal_stress_pure_shear():
    from tensorphysics.ice.fracture import StressField

    f = StressField(np.zeros((3, 3)), np.zeros((3, 3)), np.full((3, 3), 2.0e5))
    s1, s2, _ = principal_stress(f)
    assert s1[0, 0] == pytest.approx(2.0e5)
    assert s2[0, 0] == pytest.approx(-2.0e5)


def test_crack_initiation_detection():
    extent, size = 100.0, 65
    x = np.linspace(-extent, extent, size)
    X, Y = np.meshgrid(x, x, indexing="xy")
    s1 = 2.0e5 * np.exp(-(X**2 + Y**2) / (2.0 * 20.0**2))
    assert s1[0, 0] < 1.0e5 < s1[size // 2, size // 2]
    out = crack_initiation(s1, X, Y, strength=1.0e5, characteristic_length=20.0, time=0.5)
    assert out["triggered"] is True
    assert out["time_s"] == pytest.approx(0.5)
    assert out["radius_from_contact_m"] == pytest.approx(0.0, abs=1e-9)
    out2 = crack_initiation(s1, X, Y, strength=1.0e12)
    assert out2["triggered"] is False


def test_ring_crack_radius_outside_the_contact():
    size = 129
    x = np.linspace(-200.0, 200.0, size)
    X, Y = np.meshgrid(x, x, indexing="xy")
    R = np.hypot(X, Y)
    field = np.exp(-0.5 * ((R - 60.0) / 8.0) ** 2) * 1.0e6 - 2.0e6 * np.exp(-0.5 * (R / 5.0) ** 2)
    out = ring_crack_radius(field, X, Y, contact_xy=(0.0, 0.0), inner_radius=20.0,
                            characteristic_length=30.0)
    assert out["ring_radius_m"] == pytest.approx(60.0, abs=4.0)
    assert out["ring_radius_over_lc"] == pytest.approx(2.0, abs=0.15)


def test_fracture_energy_scaling():
    e1 = fracture_energy(10.0, 1.0)
    assert e1["Gc_J_m2"] == pytest.approx(1.2e5**2 / 9.0e9)
    assert e1["energy_J"] == pytest.approx(e1["Gc_J_m2"] * 10.0)
    with pytest.raises(ValueError):
        fracture_energy(-1.0, 1.0)
