"""Transient thermal shock of a semi-infinite solid (Carslaw & Jaeger anchor)."""

from __future__ import annotations

import numpy as np
import pytest

from tensorphysics.thermo_mechanical import (
    DEFAULT_MATERIAL,
    kappa_thermal,
    thermal_shock_fe,
)

TIMES = (0.1, 1.0, 10.0)


@pytest.fixture(scope="module")
def shock():
    return thermal_shock_fe(list(TIMES))


def test_surface_stress_matches_carslaw_jaeger(shock):
    """Surface thermal-shock stress at t = 0.1/1/10 s vs -E*alpha*dT/(1-nu)."""
    mat = DEFAULT_MATERIAL
    ref = -mat["E"] * mat["alpha"] * (mat["Ts"] - mat["T0"]) / (1.0 - mat["nu"])
    for val in shock["surface_stress_fe"]:
        assert abs(val - ref) / abs(ref) <= 0.05


def test_temperature_profile_against_erfc(shock):
    """FE temperature vs the erfc similarity solution (<1% of the step)."""
    for T, T_ex in zip(shock["T"], shock["T_exact"]):
        assert np.max(np.abs(T - T_ex)) / 20.0 < 0.01


def test_stress_depth_profile_against_erfc(shock):
    """In-plane stress depth profile vs the analytic erfc profile (<5%)."""
    for err in shock["max_rel_err_profile"]:
        assert err <= 0.05


def test_surface_stays_traction_free(shock):
    """sigma_yy must vanish: the eigenstrain field is compatible (free surface)."""
    scale = abs(shock["surface_stress_exact"][0])
    assert shock["max_sigma_yy"] / scale < 1.0e-6


def test_diffusivity_and_boundary_layer_scale():
    """kappa = k/(rho c) and the diffusion length sqrt(kappa t) grow like sqrt(t)."""
    mat = DEFAULT_MATERIAL
    kap = kappa_thermal(mat["k"], mat["rho"], mat["c"])
    assert kap == pytest.approx(1.282051282051282e-05, rel=1e-6)
    ratio = np.sqrt(TIMES[2] / TIMES[0])
    assert ratio == pytest.approx(10.0)
    assert np.sqrt(kap * TIMES[0]) < 2.0e-3      # 1.1 mm boundary layer at 0.1 s


def test_surface_step_is_immediate():
    """A Dirichlet temperature step on the surface reaches Ts at t = 0+."""
    out = thermal_shock_fe([0.01], material=None, y_max=0.05)
    mat = DEFAULT_MATERIAL
    assert abs(out["T"][0][0] - mat["Ts"]) < 1e-9