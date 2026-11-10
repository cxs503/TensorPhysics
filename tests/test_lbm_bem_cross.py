"""Acceptance tests for the LBM <-> BEM cross-validation framework (p84w3-n10).

The tests are deliberately *cheap* (they run in well under a minute) and target
the pieces of evidence the campaign criteria rest on:

1. dual-format geometry contract  — vertex/normal agreement < 1e-9;
2. panel method vs analytic cylinder potential flow ``Cp = 1 - 4 sin^2(theta)``;
3. far-field decay of the potential disturbance (2-D doublet: ``|u'| ~ r^-2``);
4. LBM collision viscosity through the periodic shear-wave decay;
5. NumPy vs compiled-kernel backend equivalence (same physics);
6. Thwaites integral boundary layer vs the Blasius flat plate;
7. LBM cylinder at Re = 100 — mass conservation, non-zero drag, bounded field;
8. criteria-evaluation logic (including the honest "not adjudicable" path);
9. artifact-level check of the generated ``criteria.json`` when it exists.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.lbm_bem_cross import (  # noqa: E402
    PanelMethod2D,
    analytic_cylinder_cp,
    bilinear_sample,
    evaluate_criteria,
    geometry_contract_check,
    nurbs_circle,
    polar_samples,
    thwaites_flat_plate_check,
)
from tensorphysics.lbm_bem_cross import _kernel  # noqa: E402
from tensorphysics.lbm_bem_cross.lbm import (  # noqa: E402
    CylinderLBM,
    LBMConfig,
    shear_wave_viscosity,
)

ART = ROOT / "artifacts" / "lbm_bem_cross"


# ---------------------------------------------------------------------------
# 1. geometry contract (criterion 4)
# ---------------------------------------------------------------------------

def test_geometry_dual_format_contract(tmp_path):
    """NURBS and STL paths must deliver identical vertices and normals."""
    rep = geometry_contract_check(1.0, 96, tmp_path)
    assert rep["n_vertices_nurbs"] == rep["n_vertices_stl"] == 96
    assert rep["max_vertex_deviation"] < 1e-9
    assert rep["max_normal_deviation"] < 1e-9
    assert rep["passed"] is True
    # area identity: the polygon approximates pi/4 from below (inscribed)
    g = nurbs_circle(1.0, 96)
    assert 0.99 * math.pi / 4 < g.area <= math.pi / 4
    assert abs(g.hydraulic_diameter - 1.0) < 5e-3


def test_geometry_theta_is_monotone_and_normals_outward():
    g = nurbs_circle(1.0, 64)
    th = g.theta()
    assert np.all(np.diff(th) > 0)
    xm, ym, nrm = g.segment_midpoints()
    assert np.all(xm * nrm[:, 0] + ym * nrm[:, 1] > 0)


# ---------------------------------------------------------------------------
# 2./3. BEM verification
# ---------------------------------------------------------------------------

def test_panel_method_matches_analytic_cylinder():
    g = nurbs_circle(1.0, 128)
    pm = PanelMethod2D(g, u_inf=1.0)
    sol = pm.solve()
    assert sol.residual < 1e-8
    assert np.max(np.abs(sol.cp - analytic_cylinder_cp(sol.theta))) < 1e-6
    # d'Alembert paradox: zero potential-flow drag
    assert abs(pm.drag_coefficient_pressure()) < 1e-8
    # surface velocity max ratio (2 U_inf at the shoulder)
    assert abs(float(sol.u_t.max()) - 2.0) < 1e-2


def test_potential_far_field_decays_like_a_doublet():
    """2-D potential disturbance decays as r^-2 (velocity), not r^-1."""
    g = nurbs_circle(1.0, 128)
    pm = PanelMethod2D(g, u_inf=1.0)
    pm.solve()
    xs, ys, meta = polar_samples(1.0, r_min_d=5.0, r_max_d=20.0, n_r=8, n_phi=48)
    u, v = pm.velocity_field(xs, ys)
    r = np.ravel(np.hypot(xs, ys))
    amp = np.ravel(np.hypot(u - 1.0, v))
    mask = amp > 1e-14
    slope = np.polyfit(np.log(r[mask]), np.log(amp[mask]), 1)[0]
    assert -2.15 < slope < -1.85, slope
    assert meta["wake_half_angle_deg"] == 30.0


def test_bilinear_sampling_roundtrip():
    field = np.add.outer(np.arange(10.0), np.arange(12.0) * 2.0)  # f[y, x]
    val = bilinear_sample(field, np.array([3.5]), np.array([4.25]))
    expect = 4.25 + 2.0 * 3.5
    assert abs(float(val[0]) - expect) < 1e-9


# ---------------------------------------------------------------------------
# 4./5. LBM verification
# ---------------------------------------------------------------------------

def test_shear_wave_viscosity_reproduces_tau():
    rep = shear_wave_viscosity(nu=0.02, n_steps=1500)
    assert rep["relative_error"] < 0.02, rep


def test_backends_are_equivalent():
    """The compiled kernel and the NumPy reference must agree."""
    cfg = LBMConfig(re=100.0, diameter=10, u_in=0.08, up_d=4, down_d=8,
                    half_width_d=4, init="uniform", backend="numpy")
    a = CylinderLBM(cfg)
    a.run(300, probe_every=100)
    import dataclasses
    b = CylinderLBM(dataclasses.replace(cfg, backend="auto"))
    if not _kernel.available():
        pytest.skip(f"compiled kernel unavailable: {_kernel.build_error()}")
    assert b.backend == "c"
    b.run(300, probe_every=100)
    assert np.max(np.abs(a.u - b.u)) < 1e-4
    assert abs(a.probe[-1]["cd"] - b.probe[-1]["cd"]) < 1e-3


def test_lbm_cylinder_re100_smoke():
    """Short Re=100 run: mass conserved, finite field, plausible drag."""
    cfg = LBMConfig(re=100.0, diameter=10, u_in=0.08, up_d=5, down_d=10,
                    half_width_d=5, init="potential")
    lb = CylinderLBM(cfg)
    lb.run(1200, probe_every=200)
    assert np.isfinite(lb.u).all() and np.isfinite(lb.v).all()
    # mass conservation: inlet flux == outlet flux within 1 %
    fin = float(lb.u[:, 1].sum())
    fout = float(lb.u[:, -2].sum())
    assert abs(fin - fout) / abs(fin) < 0.01
    cd = float(lb.probe[-1]["cd"])
    assert 0.3 < cd < 3.0, cd
    # the near-wall tangential velocity must reverse (separation) behind the body
    sep = lb.separation_angle()
    assert math.radians(60.0) < sep < math.pi, math.degrees(sep)


# ---------------------------------------------------------------------------
# 6. boundary layer verification
# ---------------------------------------------------------------------------

def test_thwaites_matches_blasius():
    rep = thwaites_flat_plate_check()
    assert rep["max_rel_err_cf_vs_blasius"] < 0.02, rep
    assert rep["max_rel_err_theta_vs_thwaites_exact"] < 0.01, rep


# ---------------------------------------------------------------------------
# 8. criteria logic (including the honest non-adjudicable path)
# ---------------------------------------------------------------------------

def test_criteria_evaluation_paths():
    cd = {"cd_lbm_vs_bem": 0.08, "cd_lbm": 1.2, "cd_bem_vs_reference": 0.9,
          "cd_lbm_vs_reference": 0.02, "cd_tolerance": 0.10}
    cp = {"max_rel_deviation": 0.02}
    far = {"l2_deviation": 0.01, "max_deviation": 0.03}
    contract = {"max_vertex_deviation": 0.0, "max_normal_deviation": 0.0, "passed": True}
    ok = evaluate_criteria(cd, cp, far, contract, lbm_valid=True)
    assert ok["1a_cd_deviation"]["passed"] and ok["2_cp_front"]["passed"]
    assert ok["3_far_field"]["passed"] and ok["4_geometry_contract"]["passed"]
    bad = evaluate_criteria(cd, {"max_rel_deviation": 0.5}, far, contract, lbm_valid=True)
    assert bad["2_cp_front"]["passed"] is False
    # a diverged LBM must be reported as not adjudicable, never as a pass
    div = evaluate_criteria(cd, cp, far, contract, lbm_valid=False)
    assert div["1a_cd_deviation"]["adjudicable"] is False
    assert div["1a_cd_deviation"]["passed"] is False
    assert div["4_geometry_contract"]["passed"] is True


# ---------------------------------------------------------------------------
# 9. artifact-level check of the generated report
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not (ART / "criteria.json").exists(),
                    reason="campaign artifacts not generated in this checkout")
def test_report_artifacts_are_consistent():
    crit = json.loads((ART / "criteria.json").read_text())
    assert crit["campaign_node"] == "p84w3-n10-lbm-bem-cross-validation"
    for name in ("re100", "re1000", "re10000"):
        assert name in crit["cases"]
        case = crit["cases"][name]
        # the geometry contract must hold for every case, adjudicable or not
        assert case["criteria"]["4_geometry_contract"]["passed"] is True
        assert case["criteria"]["4_geometry_contract"]["max_vertex_deviation"] < 1e-9
        if case["lbm_solution_valid"]:
            assert case["cd"]["cd_lbm"] is not None
            assert case["far_field"]["n_points"] > 100
        else:
            # honest non-adjudicable verdict, with a physical reason attached
            assert case["criteria"]["1a_cd_deviation"]["adjudicable"] is False
            assert "reason" in case["criteria"]["1a_cd_deviation"]
    assert (ART / "fig_cd.png").exists()