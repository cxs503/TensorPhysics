"""Tests for the ice-belt local-structure FEM cross-check (IN11-H4).

Coverage map
------------
* pressure-area law and its input guards (unknown Polar Class, non-positive
  area / pressure / spacing, bad exponent) -- the required negative cases;
* the FEM leg against the closed-form Navier solution (< 5 % gate) for uniform
  and patch loading, plus mesh convergence;
* the reused-element pins: the local bicubic Hermite rows must equal
  ``PlateModel.kern["Nw"]`` and a full-element consistent patch load must equal
  ``PlateModel.pressure_load``;
* the dual-chain comparison against the ship-platform B1 chain, including the
  closed-form deviation attribution and the honest ``NOT_AVAILABLE`` path.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from tensorphysics.hull_icebelt_fem import (
    analytic,
    iacs_pressure,
    panel_fem,
    stiffener,
    xcheck,
)
from tensorphysics.plate_buckling import PlateModel


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def crosscheck():
    return xcheck.run_crosscheck()


@pytest.fixture(scope="module")
def verify_panel():
    return xcheck.VERIFY_PANEL


# ---------------------------------------------------------------------------
# pressure-area law / input guards
# ---------------------------------------------------------------------------
def test_pressure_area_law_anchor_and_scaling():
    """p(1 m^2) is the declared 1.2 MPa and the law is a -0.5 power law."""
    assert iacs_pressure.pressure_area_law(1.0) == pytest.approx(1.2e6)
    assert iacs_pressure.pressure_area_law(4.0) == pytest.approx(0.6e6)
    assert iacs_pressure.pressure_area_law(0.25) == pytest.approx(2.4e6)
    # strictly decreasing in area (the physical content of the area effect)
    areas = np.logspace(-2, 2, 25)
    p = [iacs_pressure.pressure_area_law(a) for a in areas]
    assert all(p[i] > p[i + 1] for i in range(len(p) - 1))
    # PC6 anchor: 1.5 m design ice thickness
    assert iacs_pressure.design_ice_thickness("PC6") == pytest.approx(1.5)


@pytest.mark.parametrize("bad", ["PC9", "1A", "pc6-super", "", None, 6])
def test_unknown_ice_class_rejected(bad):
    """An unknown / non-string ice class must never fall back to a default."""
    with pytest.raises(iacs_pressure.IACSPressureError):
        iacs_pressure.design_ice_thickness(bad)
    with pytest.raises(iacs_pressure.IACSPressureError):
        iacs_pressure.design_patch_pressure(bad, stiffener_spacing=0.35)


@pytest.mark.parametrize("area", [0.0, -1.0, -1e-6, float("nan"),
                                  float("inf")])
def test_nonpositive_patch_area_rejected(area):
    """Zero, negative or non-finite areas are rejected before any 0**-0.5."""
    with pytest.raises(iacs_pressure.IACSPressureError):
        iacs_pressure.pressure_area_law(area)


@pytest.mark.parametrize("kwargs", [
    {"p_ref": 0.0},
    {"p_ref": -1.2e6},
    {"a_ref": 0.0},
    {"exponent": 0.0},
    {"exponent": -0.5},
    {"exponent": 1.5},
])
def test_bad_law_parameters_rejected(kwargs):
    with pytest.raises(iacs_pressure.IACSPressureError):
        iacs_pressure.pressure_area_law(0.5, **kwargs)


@pytest.mark.parametrize("h,s", [(0.0, 0.35), (-1.5, 0.35), (1.5, 0.0),
                                 (1.5, -0.35), (float("nan"), 0.35)])
def test_patch_area_rejects_nonpositive_inputs(h, s):
    with pytest.raises(iacs_pressure.IACSPressureError):
        iacs_pressure.patch_area(h, s)


@pytest.mark.parametrize("s", [0.0, -0.35, float("nan")])
def test_design_pressure_rejects_nonpositive_spacing(s):
    with pytest.raises(iacs_pressure.IACSPressureError):
        iacs_pressure.design_patch_pressure("PC6", s)
    with pytest.raises(iacs_pressure.IACSPressureError):
        iacs_pressure.patch_area(1.5, s, panel_length=2.0)


def test_patch_area_conventions_differ_as_documented():
    """Clipped (panel) and unclipped (contact patch) areas are not confused."""
    a_patch = iacs_pressure.patch_area(1.5, 0.35)
    a_panel = iacs_pressure.patch_area(1.5, 0.35, panel_length=2.0)
    assert a_patch == pytest.approx(0.6 * 1.5 * 2.4 * 1.5)
    assert a_panel == pytest.approx(0.35 * 2.0)
    assert a_panel < a_patch


# ---------------------------------------------------------------------------
# negative inputs on the structural side
# ---------------------------------------------------------------------------
def test_panel_rejects_invalid_specification():
    with pytest.raises(ValueError):
        panel_fem.IceBeltPanel(a=2.0, b=0.35, t=-0.030)
    with pytest.raises(ValueError):
        panel_fem.IceBeltPanel(a=2.0, b=0.0, t=0.030)
    with pytest.raises(ValueError):
        panel_fem.IceBeltPanel(a=2.0, b=0.35, t=0.030, grade="WOOD")
    with pytest.raises(ValueError):
        panel_fem.IceBeltPanel(a=2.0, b=0.35, t=0.030, gate="loose")


def test_negative_pressure_and_bad_patch_rejected(verify_panel):
    with pytest.raises(ValueError):
        verify_panel.solve(-1.0e5)
    with pytest.raises(ValueError):
        verify_panel.solve(1.0e5, patch_box=(1.0, 1.0, 0.0, 0.5))
    with pytest.raises(ValueError):
        verify_panel.solve(1.0e5, patch_box=(-0.2, 1.0, 0.0, 0.5))
    with pytest.raises(ValueError):
        verify_panel.solve(1.0e5, load_rule="magic")
    with pytest.raises(ValueError):
        verify_panel.patch_box(0.0)


def test_stiffener_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        stiffener.StiffenerProfile(web_h=0.0, web_t=0.012, flange_b=0.12,
                                  flange_t=0.018, spacing=0.35, t_plate=0.03,
                                  span=2.0)
    with pytest.raises(ValueError):
        stiffener.StiffenerProfile(web_h=0.4, web_t=0.012, flange_b=0.12,
                                   flange_t=0.0, spacing=0.35, t_plate=0.03,
                                   span=2.0)
    prof = xcheck.STIFFENER
    with pytest.raises(ValueError):
        prof.response(-1.0)
    with pytest.raises(ValueError):
        stiffener.stiffener_response(prof, -1.0, 0.35)


def test_geometry_gate_is_reported_not_hidden():
    """The stocky design panel reports its beta excursion instead of raising."""
    checks = xcheck.DESIGN_PANEL.checks()
    assert checks["gate"] == "relaxed"
    assert checks["gate_pass"] is True
    assert checks["beta"] < 1.0
    assert checks["beta_inside_buckling_envelope"] is False
    strict = xcheck.VERIFY_PANEL.checks()
    assert strict["gate_pass"] is True
    assert strict["gate_numbers"]["beta"] > 1.0


# ---------------------------------------------------------------------------
# reused-element pins
# ---------------------------------------------------------------------------
def test_hermite_rows_match_reused_kernel(verify_panel):
    """The local Hermite rows equal the reused element's ``Nw`` exactly."""
    g = verify_panel.geometry()
    model = PlateModel(g, bc="SSSS")
    gx = np.polynomial.legendre.leggauss(4)[0]
    for gi, (xi, eta) in enumerate(
            [(gx[i], gx[j]) for j in range(4) for i in range(4)]):
        row = panel_fem.hermite_w_row(model.ex, model.ey,
                                      0.5 * (1.0 + xi), 0.5 * (1.0 + eta))
        assert np.allclose(row, model.kern["Nw"][gi], atol=1e-12)


def test_consistent_patch_load_reproduces_uniform_load(verify_panel):
    """A patch covering whole elements reproduces ``pressure_load`` exactly."""
    g = verify_panel.geometry()
    model = PlateModel(g, bc="SSSS")
    q = 1.0e6
    F_patch = panel_fem.consistent_patch_load(model, q, (0.0, g.a, 0.0, g.b))
    F_ref = model.pressure_load(q)
    assert np.allclose(F_patch, F_ref, atol=1e-6)
    # total transverse load equals q * loaded area
    w_dofs = 6 * np.arange(g.n_nodes) + 2
    assert F_patch[w_dofs].sum() == pytest.approx(q * g.a * g.b, rel=1e-9)


def test_patch_load_scales_linearly_with_pressure(verify_panel):
    g = verify_panel.geometry()
    model = PlateModel(g, bc="SSSS")
    box = verify_panel.patch_box(xcheck.VERIFY_PATCH_THICKNESS)
    f1 = panel_fem.consistent_patch_load(model, 1.0e5, box)
    f3 = panel_fem.consistent_patch_load(model, 3.0e5, box)
    assert np.allclose(3.0 * f1, f3, atol=1e-9)


# ---------------------------------------------------------------------------
# FEM vs closed form
# ---------------------------------------------------------------------------
def test_fem_uniform_pressure_matches_navier(verify_panel):
    g = verify_panel.geometry()
    q = xcheck.VERIFY_UNIFORM_PRESSURE
    sol = verify_panel.solve(q)
    ana = analytic.navier_field(g, [(0.0, g.a, 0.0, g.b)], q, g.a / 2, g.b / 2,
                                n_terms=81)
    assert abs(sol.w_centre - ana["w"]) / ana["w"] < 0.05
    assert abs(sol.sigma_centre - ana["sigma_x"]) / ana["sigma_x"] < 0.05
    # the panel stays elastic, so the linear model is the right model here
    assert sol.sigma_vm_max < verify_panel.material["sigma_y"]


def test_fem_patch_matches_navier_patch(verify_panel):
    g = verify_panel.geometry()
    q = xcheck.PATCH_TEST_PRESSURE
    box = verify_panel.patch_box(xcheck.VERIFY_PATCH_THICKNESS)
    sol = verify_panel.solve(q, patch_box=box)
    ana = analytic.navier_field(g, [box], q, g.a / 2, g.b / 2, n_terms=161)
    assert abs(sol.w_centre - ana["w"]) / ana["w"] < 0.05
    assert abs(sol.sigma_centre - ana["sigma_x"]) / ana["sigma_x"] < 0.05
    # the patch covers 8 of 192 elements; the consistent rule loads exactly it
    assert sol.loaded_area_m2 == pytest.approx(sol.patch_area_m2, rel=1e-12)


def test_mesh_convergence_towards_closed_form():
    """Deflection is converged on every mesh; the moment peak converges O(h^2)."""
    conv = xcheck.mesh_convergence()
    w_devs = [abs(r["deviation_w_pct"]) for r in conv["rows"]]
    s_devs = [abs(r["deviation_sigma_pct"]) for r in conv["rows"]]
    assert max(w_devs) < 1.0, "deflection must already be converged"
    assert all(s_devs[i] > s_devs[i + 1] for i in range(len(s_devs) - 1)), \
        "the centre-moment peak must converge monotonically with the mesh"
    assert s_devs[-1] < 5.0
    assert conv["pass"] is True


def test_element_centre_load_rule_error_is_documented():
    """The reused centre-based load rule is measurably worse when misaligned."""
    verify = xcheck.verification_cases()
    case = next(c for c in verify["cases"]
                if c["name"] == "load_rule_sensitivity_coarse_mesh")
    ec = case["deviation_pct"]["w_centre_element_centre_rule"]
    cc = case["deviation_pct"]["w_centre_consistent_rule"]
    assert abs(ec) > 10.0            # the centre rule's failure mode is visible
    assert abs(cc) < 1.0             # the consistent integration is not affected
    assert case["acceptance"] is False


# ---------------------------------------------------------------------------
# artifact / dual-chain / criteria
# ---------------------------------------------------------------------------
def test_crosscheck_criteria_all_pass(crosscheck):
    criteria = crosscheck["criteria"]
    assert criteria["fem_vs_closed_form_lt_5pct"]["pass"] is True
    assert criteria["fem_vs_closed_form_lt_5pct"]["value_pct"] < 5.0
    assert criteria["pc6_design_elastic"]["pass"] is True
    assert crosscheck["all_criteria_pass"] is True


def test_dual_chain_table_and_attribution_closure(crosscheck):
    sweep = crosscheck["dual_chain"]
    assert sweep["rows"], "the deviation table must not be empty"
    if sweep["chain_b_status"] != "AVAILABLE":
        pytest.skip("chain B not importable in this environment (declared)")
    for row in sweep["rows"]:
        assert row["deviation_pct"]["p_B_vs_A"] is not None
        assert row["deviation_pct"]["p_B_vs_A"] > 10.0, (
            "the B1 chain must over-predict; a smaller deviation would need a "
            "new physical explanation")
        attr = row["attribution"]
        assert abs(attr["closed_form_residual_pct"]) < 0.5
        factors = attr["factors"]
        prod = 1.0
        for v in factors.values():
            prod *= v
        assert prod == pytest.approx(attr["ratio_direct"], rel=1e-9)


def test_chain_b_probe_is_declared_either_way():
    probe = xcheck.probe_b1_chain()
    assert probe["status"] in ("AVAILABLE", "NOT_AVAILABLE")
    if probe["status"] == "AVAILABLE":
        assert probe["functions_used"] == ["global_ice_force", "local_line_load"]
        assert probe["module_sha256"]
        assert "merged_to_main" in probe
    else:
        assert "WITHHELD" in probe["evidence_grade"]


def test_chain_b_is_used_through_its_own_functions():
    """Where available, the chain-B pressure follows its published formula."""
    probe = xcheck.probe_b1_chain()
    mod = probe.get("_module")
    if mod is None:
        pytest.skip("chain B not importable in this environment")
    out = xcheck._b1_pressure(mod, 1.5, 3.0e6, 1.0)
    psi = out["impact_factor"]
    assert out["p_patch_Pa"] == pytest.approx(3.0e6 * psi / 0.6, rel=1e-9)
    # chain B has no area effect: the patch pressure is thickness independent
    out2 = xcheck._b1_pressure(mod, 0.5, 3.0e6, 1.0)
    assert out2["p_patch_Pa"] == pytest.approx(out["p_patch_Pa"], rel=1e-9)


def test_artifact_roundtrip_and_audit_chain(tmp_path, crosscheck):
    art = dict(crosscheck)
    art.pop("_written", None)
    del art["design_case"]["pressure"]  # keep the dump fast/small
    path = tmp_path / "xcheck.json"
    path.write_text(json.dumps(art, indent=1), encoding="utf-8")
    back = json.loads(path.read_text(encoding="utf-8"))
    assert back["schema"] == xcheck.SCHEMA
    audit = crosscheck["audit_chain"]
    assert audit["module_files"]["panel_fem.py"]["sha256"]
    assert "reused_components" in audit and audit["reused_components"]
    assert crosscheck["anchor"]["benchmarks_ice_nav_refs_present"] in (True, False)
    assert crosscheck["limitations"]


def test_design_case_is_consistent_with_the_fem():
    design = xcheck.pc6_design_case()
    sy = xcheck.DESIGN_PANEL.material["sigma_y"]
    assert design["pressure"]["ice_class"] == "PC6"
    assert 1.0e6 < design["pressure"]["pressure_Pa"] < 3.0e6
    assert design["panel_utilisation"]["sigma_vm_max_over_sigma_y"] < 1.0
    stiff = design["stiffener"]["response"]
    assert stiff["sigma_flange_Pa"] < sy
    assert 0.0 < stiff["delta_max_m"] < design["panel"]["a_m"] / 100.0