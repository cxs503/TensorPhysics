"""Verification tests for imperfections, ice loading, nonlinear ultimate
strength and the IACS UR I ultimate-limit-state check."""

from __future__ import annotations

import numpy as np
import pytest

from tensorphysics.plate_buckling import PlateGeometry, PlateModel
from tensorphysics.plate_buckling.benchmarks import faulkner_ultimate
from tensorphysics.plate_buckling.imperfection import (
    ImperfectionSpec,
    apply_imperfection,
    critical_mode_orders,
    deflection_amplitude,
    residual_stress_field,
)
from tensorphysics.plate_buckling.loads import (
    PATCH_ASPECT_RATIO,
    PATCH_HEIGHT_FACTOR,
    IcePatchLoad,
    ice_patch_pressure,
)
from tensorphysics.plate_buckling.uls_check import uls_check

SIGMA_Y = 313e6
BETA_TO_T = 0.03898      # t = b * 0.03898 / beta for E = 206 GPa, sy = 313 MPa


def _plate(beta=2.0, a=3.0, nx=10, ny=4) -> PlateGeometry:
    return PlateGeometry(a=a, b=1.0, t=BETA_TO_T / beta, sigma_y=SIGMA_Y,
                         nx=nx, ny=ny)


# --------------------------------------------------------------------------
# imperfections
# --------------------------------------------------------------------------
def test_imperfection_amplitude_levels():
    geom = _plate()
    w_slight = deflection_amplitude(geom, "slight")
    w_avg = deflection_amplitude(geom, "average")
    w_sev = deflection_amplitude(geom, "severe")
    assert w_avg == pytest.approx(0.1 * geom.beta**2 * geom.t)
    assert w_slight < w_avg < w_sev
    with pytest.raises(ValueError):
        deflection_amplitude(geom, "unknown")


def test_critical_mode_orders_follow_aspect_ratio():
    assert critical_mode_orders(_plate(a=3.0))["m"] == 3
    assert critical_mode_orders(_plate(a=5.0))["m"] == 5
    assert critical_mode_orders(_plate(a=2.0))["n"] == 1


def test_initial_deflection_field_is_applied_with_full_hermite_dofs():
    geom = _plate()
    model = PlateModel(geom, "SSSS")
    spec = ImperfectionSpec.standard(geom, level="average", sigma_r_ratio=0.0)
    apply_imperfection(model, spec)
    w = model.deflection_field()
    # nodal sampling: the peak of the sine may fall between nodes
    assert np.abs(w).max() == pytest.approx(spec.w0max, rel=0.05)
    # the stored reference field must be consistent (stress free at lam = 0)
    f, _, _ = model.assemble(model.u, need_tangent=False)
    assert np.linalg.norm(f) < 1e-6 * geom.t * geom.b * SIGMA_Y


def test_residual_stress_field_is_self_equilibrated():
    field = residual_stress_field(31.3e6, b=1.0, t=0.0195, edge_fraction=0.2)
    force = field["axial_force_per_width"][0]
    # residual stresses must be self-equilibrated (trapezoidal quadrature)
    assert abs(force) < 2e-3 * 31.3e6 * 0.0195
    assert field["sigma_x"].max() == pytest.approx(31.3e6)


# --------------------------------------------------------------------------
# ice patch load
# --------------------------------------------------------------------------
def test_ice_patch_geometry_follows_iacs_ur_i():
    patch = IcePatchLoad(h_ice=1.0, p_ice=5.0e6)
    assert patch.height == pytest.approx(PATCH_HEIGHT_FACTOR * 1.0)
    assert patch.width / patch.height == pytest.approx(PATCH_ASPECT_RATIO)


def test_ice_patch_load_resultant_equals_pressure_times_patch_area():
    geom = _plate(ny=5)      # element edges aligned with the 0.6 h patch
    model = PlateModel(geom, "SSSS")
    patch = IcePatchLoad(h_ice=1.0, p_ice=5.0e6)
    F = patch.nodal_load(model)
    total = F[6 * np.arange(geom.n_nodes) + 2].sum()
    x0, x1, y0, y1 = patch.box(geom)
    expected = patch.p_ice * (x1 - x0) * (y1 - y0)
    assert total == pytest.approx(expected, rel=0.02)


def test_ice_pressure_scaling_with_thickness():
    assert ice_patch_pressure(1.0, p_ref=5.0e6, h_ref=1.0) == pytest.approx(5.0e6)
    assert ice_patch_pressure(0.5, p_ref=5.0e6, h_ref=1.0) == pytest.approx(5.0e6)


# --------------------------------------------------------------------------
# nonlinear ultimate strength
# --------------------------------------------------------------------------
@pytest.mark.parametrize("beta", [1.5, 3.0])
def test_arc_length_traces_limit_point(beta):
    geom = _plate(beta=beta, nx=8, ny=4)
    model = PlateModel(geom, "SSSS")
    spec = ImperfectionSpec.standard(geom, level="average", sigma_r_ratio=0.0)
    apply_imperfection(model, spec)
    res = model.solve_arc_length(model.compression_load(SIGMA_Y),
                                 max_steps=12, sigma_ref=SIGMA_Y,
                                 ds0=6e-4)
    path = res["path"]
    assert len(path) >= 4
    lam = np.array([p["lam"] for p in path])
    w = np.array([p["w_monitor"] for p in path])
    ult = res["lam_ultimate"]
    assert 0.3 < ult < 1.0
    assert ult == pytest.approx(max(lam))
    # deflections grow through the analysis (post-buckling response)
    assert abs(w[-1]) > abs(w[0])
    # the traced peak is the maximum load factor reached in the run
    assert ult == pytest.approx(lam.max())


def test_ultimate_strength_trend_matches_faulkner_reference():
    """sigma_u/sigma_y must follow the von Karman trend and stay within 20 %."""
    results = {}
    for beta in (1.5, 3.0):
        geom = _plate(beta=beta, nx=8, ny=4)
        model = PlateModel(geom, "SSSS")
        apply_imperfection(
            model,
            ImperfectionSpec.standard(geom, level="average", sigma_r_ratio=0.0))
        res = model.solve_arc_length(model.compression_load(SIGMA_Y),
                                     max_steps=12, sigma_ref=SIGMA_Y,
                                     ds0=6e-4)
        results[beta] = res["lam_ultimate"]
    assert results[1.5] > results[3.0]           # stockier plate is stronger
    for beta, ult in results.items():
        assert ult == pytest.approx(faulkner_ultimate(beta), rel=0.10)


def test_imperfection_sensitivity_is_monotone():
    """A larger initial deflection must not increase the ultimate strength."""
    geom = _plate(beta=2.0, nx=8, ny=4)
    ults = []
    for level in ("slight", "severe"):
        model = PlateModel(geom, "SSSS")
        apply_imperfection(
            model, ImperfectionSpec.standard(geom, level=level, sigma_r_ratio=0.0))
        res = model.solve_arc_length(model.compression_load(SIGMA_Y),
                                     max_steps=10, sigma_ref=SIGMA_Y,
                                     ds0=6e-4)
        ults.append(res["lam_ultimate"])
    assert ults[0] >= ults[1] * (1.0 - 0.02)


# --------------------------------------------------------------------------
# ULS check
# --------------------------------------------------------------------------
def test_uls_check_passes_with_margin():
    res = uls_check(sigma_y=SIGMA_Y, sigma_vm_max=250e6, p_ice=3.0e6,
                    p_water=0.1e6, p_ult=8.0e6)
    assert res.passed
    assert res.utilisation < 1.0
    assert res.safety_factor > res.required_safety_factor


def test_uls_check_fails_when_capacity_is_exceeded():
    res = uls_check(sigma_y=SIGMA_Y, sigma_vm_max=400e6, p_ice=6.0e6,
                    p_water=0.2e6, p_ult=4.0e6)
    assert not res.passed
    assert res.utilisation > 1.0
    assert not res.pass_stress


def test_uls_utilisation_scales_with_load_factor():
    base = uls_check(SIGMA_Y, 200e6, 3.0e6, 0.1e6, 8.0e6)
    factored = uls_check(SIGMA_Y, 200e6, 3.0e6, 0.1e6, 8.0e6,
                         factors={"gamma_ice": 2.5})
    assert factored.utilisation > base.utilisation