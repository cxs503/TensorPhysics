"""Thermal stress concentration at a circular hole (Timoshenko-Goodier anchor)."""

from __future__ import annotations

import numpy as np
import pytest

from tensorphysics.thermo_mechanical import (
    annulus_thermal_stress,
    hole_fe_sweep,
    thermal_stress_concentration,
)

A = 0.01
B = 0.20               # b/a = 20 (hole in a wide plate)
TA, TB = 100.0, 0.0    # hole hotter than the far field
E = 200.0e9
ALPHA = 12.0e-6
NU = 0.3


def test_kt_fe_matches_analytic_within_5pct():
    """Ring-element K_t vs exact Lame-type thermo-elastic solution (<=5%)."""
    fe = hole_fe_sweep(A, B, TA, TB, E, ALPHA, nu=NU, n_elems=128)
    assert fe["rel_error_Kt"] <= 0.05          # like-for-like (same radius)
    assert fe["rel_error_Kt_edge"] <= 0.05     # hole-edge extrapolated value
    ex = thermal_stress_concentration(A, B, TA, TB, E, ALPHA)
    assert fe["K_t_exact_edge"] == pytest.approx(ex["K_t"], rel=1e-10)


def test_free_boundary_tractions_vanish():
    """sigma_r(a) = sigma_r(b) = 0 identically in the analytic solution."""
    r = np.array([A, 0.25 * A + 0.75 * B, B])
    s_r, _ = annulus_thermal_stress(r, A, B, TA, TB, E, ALPHA)
    scale = abs(thermal_stress_concentration(A, B, TA, TB, E, ALPHA)["sigma_theta"])
    assert abs(s_r[0]) < 1.0e-9 * scale
    assert abs(s_r[-1]) < 1.0e-9 * scale


def test_hoop_stress_is_compressive_at_hot_hole():
    """A hotter hole is restrained by the colder rim -> compressive hoop edge."""
    res = thermal_stress_concentration(A, B, TA, TB, E, ALPHA)
    assert res["sigma_theta"] < 0.0
    assert abs(res["K_t"]) > 0.5      # order-unity thermal stress concentration


def test_kt_mesh_independence():
    """K_t changes by <5% between 128 and 256 elements (converged quantity)."""
    k128 = hole_fe_sweep(A, B, TA, TB, E, ALPHA, nu=NU, n_elems=128)["K_t_fe"]
    k256 = hole_fe_sweep(A, B, TA, TB, E, ALPHA, nu=NU, n_elems=256)["K_t_fe"]
    assert abs(k128 - k256) / abs(k256) < 0.05


def test_max_hoop_stress_at_hole_edge():
    """The hoop stress extremum sits at the hole edge, not in the interior."""
    r = np.linspace(A, B, 400)
    _, s_t = annulus_thermal_stress(r, A, B, TA, TB, E, ALPHA)
    assert np.argmax(np.abs(s_t)) == 0