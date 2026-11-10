"""Tests for the DEM-FEM ice-structure coupling face (node p380w2-n06)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorphysics.ice_structure_dem_fem.coupling import (
    iacs_patch_height,
    patch_load_from_spectrum,
    transfer_impulse_consistent,
)
from tensorphysics.ice_structure_dem_fem.dem_ice import (
    IceSliceConfig,
    floating_beam_break_force,
    simulate_ice_slice,
)
from tensorphysics.ice_structure_dem_fem.fem_plate import (
    INPLANE,
    PlateConfig,
    PlateMaterial,
    _layer_state,
    navier_reference_max_stress,
    solve_plate_history,
)
from tensorphysics.ice_structure_dem_fem.lindqvist_ref import reference_for


# ---------------------------------------------------------------------------
# reference faces
# ---------------------------------------------------------------------------
def test_floating_beam_break_force_matches_beam_theory():
    cfg = IceSliceConfig(h=1.0)
    D = cfg.E * cfg.h ** 3 / (12.0 * (1.0 - cfg.nu ** 2))
    Lc = (4.0 * D / (cfg.rho_w * cfg.g)) ** 0.25
    expected = 4.0 * (cfg.sigma_f * cfg.h ** 2 / 6.0) / Lc
    assert floating_beam_break_force(cfg) == pytest.approx(expected, rel=1e-12)


def test_navier_reference_reproduces_classic_square_plate_coefficient():
    """Timoshenko's alpha = 0.00406 for a uniformly loaded SSSS square plate."""
    cfg = PlateConfig(a=2.4, b=2.4, t=0.02)
    q = 1.0e4
    ref = navier_reference_max_stress(cfg, q, n_terms=59)
    D = cfg.material.E * cfg.t ** 3 / (12.0 * (1 - cfg.material.nu ** 2))
    alpha = ref["max_deflection_m"] * D / (q * cfg.a ** 4)
    assert alpha == pytest.approx(0.00406, rel=0.02)


def test_lindqvist_adapter_line_force_reading():
    ref = reference_for(1.0, 3.0)
    assert ref.peak_per_metre == pytest.approx(ref.r_crush_n + ref.r_break_n)
    assert ref.mean_per_metre == pytest.approx(ref.r_total_n / 24.0)
    assert ref.r_crush_n > 0 and ref.r_total_n == pytest.approx(
        ref.r_crush_n + ref.r_break_n + ref.r_submersion_n)
    with pytest.raises(ValueError):
        reference_for(0.0, 3.0)


# ---------------------------------------------------------------------------
# DEM face
# ---------------------------------------------------------------------------
def test_dem_generates_contact_spectrum_and_fracture():
    res = simulate_ice_slice(IceSliceConfig(h=1.0, speed=3.0, t_end=0.08, rows=3,
                                            length_m=3.0))
    assert res.force_x.size > 10
    assert res.peak_force > 0.0
    assert np.all(np.isfinite(res.force_x))
    assert res.n_bonds > 0
    assert res.n_broken_bonds > 0
    # the spectrum must be non-monotone (sawtooth from breaking events)
    f = res.force_x
    assert np.max(f) > 2.0 * np.mean(f)
    assert res.dt > 0 and res.time[1] == pytest.approx(res.dt)


def test_dem_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        IceSliceConfig(h=0.0)
    with pytest.raises(ValueError):
        IceSliceConfig(speed=0.0)
    with pytest.raises(ValueError):
        IceSliceConfig(rows=1)


# ---------------------------------------------------------------------------
# FEM face
# ---------------------------------------------------------------------------
def test_layer_return_map_uniaxial_yield():
    mat = PlateMaterial()
    eps = np.zeros((1, 6))
    eps[0, 0] = 0.002  # 0.2 % uniaxial strain, well above yield strain
    sig, eps_p, alpha, _ = _layer_state(eps[:, INPLANE], np.zeros((1, 6)),
                                        np.zeros(1), mat, np.zeros(1))
    q = math.sqrt(sig[0, 0] ** 2 - sig[0, 0] * sig[0, 1] + sig[0, 1] ** 2
                  + 3.0 * sig[0, 3] ** 2)
    assert alpha[0] > 0.0
    assert q == pytest.approx(mat.sigma_y + mat.H * alpha[0], rel=1e-8)
    # plane-stress condensation tolerance (fixed point on eps_z)
    assert abs(sig[0, 2]) < 1e-3 * mat.sigma_y


def test_fem_elastic_limit_matches_navier():
    cfg = PlateConfig(nx=10, ny=40)
    q = 0.05e6
    res = solve_plate_history(cfg, [0.0, q])
    ref = navier_reference_max_stress(cfg, q, n_terms=39)
    assert res.converged
    assert res.max_abs_sigma_x[-1] == pytest.approx(ref["max_surface_stress_Pa"], rel=0.05)
    assert abs(res.centre_deflection[-1]) == pytest.approx(ref["max_deflection_m"], rel=0.05)


def test_fem_negative_control_zero_load():
    cfg = PlateConfig(nx=6, ny=24)
    res = solve_plate_history(cfg, np.zeros(5))
    assert np.max(np.abs(res.centre_deflection)) < 1e-12
    assert np.max(np.abs(res.max_abs_sigma_x)) < 1e-12
    assert res.energy["plastic_dissipation_J"] == 0.0


def test_fem_plastic_branch_is_regular_and_bounded():
    # 20 mm ice-belt plate: first yield near 0.52 MPa, collapse near 0.8 MPa
    cfg = PlateConfig(a=0.6, b=2.4, t=0.020, nx=8, ny=32)
    q_probe = 0.70e6
    res = solve_plate_history(cfg, np.linspace(0.0, q_probe, 11))
    assert res.converged
    assert np.all(np.isfinite(res.max_vm_stress))
    assert res.max_alpha_p[-1] > 0.0
    assert res.max_alpha_p[-1] < 0.05
    assert res.energy["work_energy_closure"] < 1e-6


# ---------------------------------------------------------------------------
# coupling face
# ---------------------------------------------------------------------------
def test_impulse_conserving_transfer():
    t_src = np.linspace(0.0, 0.3, 3001)
    f_src = np.abs(np.sin(37.0 * t_src)) + 0.1
    t_dst = np.linspace(0.0, 0.3, 61)
    f_dst = transfer_impulse_consistent(t_src, f_src, t_dst)
    i_src = float(np.trapezoid(f_src, t_src))
    i_dst = float(np.sum(f_dst[:-1] * np.diff(t_dst)))
    assert abs(i_dst - i_src) / i_src < 1e-6
    assert np.all(f_dst >= 0.0)


def test_patch_load_scales_line_force_to_pressure():
    res = simulate_ice_slice(IceSliceConfig(h=0.5, speed=2.0, t_end=0.05, rows=3,
                                            length_m=2.0))
    load = patch_load_from_spectrum(res)
    assert load.patch_height == pytest.approx(iacs_patch_height(0.5))
    assert load.peak_pressure == pytest.approx(res.peak_force / load.patch_height, rel=1e-12)
    with pytest.raises(ValueError):
        patch_load_from_spectrum(res, patch_height=0.0)