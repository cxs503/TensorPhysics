"""Bow geometry, pressure-area crushing law and the ramming dynamics."""
import numpy as np
import pytest

from tensorphysics.ice.contact import (
    CrushingLaw,
    SphericalBow,
    WedgeBow,
    crushing_force,
    pressure_area,
    ram_rigid_closed_form,
    simulate_ram_rigid,
)


def test_pressure_area_law_limits():
    law = CrushingLaw(coefficient=2.6e6, exponent=0.5, pressure_min=1e5, pressure_max=1.2e7)
    assert law.pressure(1.0) == pytest.approx(2.6e6)
    assert law.pressure(100.0) == pytest.approx(2.6e5)
    assert law.pressure(1e-6) == law.pressure_max
    assert law.pressure(1e6) == law.pressure_min
    assert crushing_force(1.0) == pytest.approx(2.6e6)
    assert pressure_area(4.0) == pytest.approx(1.3e6)


def test_bow_contact_area_geometry():
    wedge = WedgeBow(half_angle_deg=45.0, max_contact_width=10.0)
    assert wedge.contact_width(1.0) == pytest.approx(2.0)
    assert wedge.contact_area(1.0, 2.0) == pytest.approx(4.0)
    assert wedge.contact_width(20.0) == pytest.approx(10.0)      # capped
    sphere = SphericalBow(radius=5.0, max_contact_width=10.0)
    assert sphere.contact_width(0.5) == pytest.approx(2.0 * np.sqrt(5.0))
    assert sphere.contact_area(0.5, 2.0) == pytest.approx(2.0 * np.sqrt(5.0) * 2.0)


def test_closed_form_matches_time_integration():
    bow = WedgeBow(22.0, 12.0)
    law = CrushingLaw(coefficient=2.6e6)
    sol = simulate_ram_rigid(8.0e6, 1.5, bow, 1.5, law, dt=2.0e-4)
    closed = ram_rigid_closed_form(8.0e6, 1.5, bow, 1.5, law)
    assert sol.peak_force == pytest.approx(closed["peak_force_N"], rel=0.03)
    # the ram converts exactly its kinetic energy into crushing work
    ke = 0.5 * 8.0e6 * 1.5**2
    assert sol.crushing_energy() == pytest.approx(ke, rel=0.02)
    assert sol.time_to_stop > 0.0
    assert sol.speed_m_s[-1] < 0.01 * sol.speed_m_s[0]


def test_scaling_of_the_peak_force():
    """F_peak = K^(2/3) ((3/4) M V^2)^(1/3): F ~ M^(1/3) V^(2/3)."""
    bow = WedgeBow(22.0, 50.0)
    law = CrushingLaw(coefficient=2.6e6)
    f1 = ram_rigid_closed_form(2.0e6, 1.0, bow, 1.0, law)["peak_force_N"]
    f2 = ram_rigid_closed_form(16.0e6, 1.0, bow, 1.0, law)["peak_force_N"]
    assert f2 / f1 == pytest.approx(8.0 ** (1.0 / 3.0), rel=1e-9)
    f3 = ram_rigid_closed_form(2.0e6, 4.0, bow, 1.0, law)["peak_force_N"]
    assert f3 / f1 == pytest.approx(4.0 ** (2.0 / 3.0), rel=1e-9)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        CrushingLaw(coefficient=-1.0)
    with pytest.raises(ValueError):
        CrushingLaw(exponent=1.5)
    with pytest.raises(ValueError):
        simulate_ram_rigid(0.0, 1.0, WedgeBow(), 1.0)
    with pytest.raises(ValueError):
        ram_rigid_closed_form(1e6, 1.0, SphericalBow(), 1.0)
