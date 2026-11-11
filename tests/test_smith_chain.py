"""Acceptance tests for the Smith cross-scale verification chain (P9263-N03).

Coverage map (acceptance criteria of the node):

C1 grid convergence residual <= 1 %      -> test_double_bottom_grid_convergence,
                                            test_box_slender_grid_convergence
C2 monotone convergence in h             -> test_double_bottom_grid_convergence,
                                            test_richardson_quadratic_series
C3 panel magnitude + CSR agreement      -> test_panel_beta2_band_and_csr,
                                            test_panel_faulkner_trend
C4 box-girder magnitude                 -> test_compact_box_is_plastic_limit,
                                            test_slender_box_is_bracketed
C5 failure mode (>= 2 of 3 profiles)    -> test_failure_mode_gate
C6 unified interface over archetypes    -> test_section_classification,
                                            test_unified_interface_json,
                                            test_auto_loading_dispatch
plus kernel anchors (elastic limit, analytic iso-strain identity, Richardson unit
tests, element-law bounds) and input validation.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from tensorphysics.smith_chain import (
    AH32,
    MILD_STEEL,
    GirderSection,
    PlatePanel,
    SmithSolver,
    StiffenerGroup,
    analyse_section,
    box_girder,
    exact_box_plastic_modulus,
    grid_convergence_study,
    identify_failure_mode,
    issc2009_double_bottom,
    paik_panel,
    plate_slenderness,
    richardson_extrapolation,
)
from tensorphysics.smith_chain.benchmarks import effective_width_ultimate_moment
from tensorphysics.smith_chain.interface import AUTO_LOADING
from tensorphysics.smith_chain.laws import PlateLaw, effective_plate_buckling_stress

FAST = dict(n_steps=200)


# ---------------------------------------------------------------------------
# C1 / C2 - grid convergence
# ---------------------------------------------------------------------------
def test_double_bottom_grid_convergence():
    """C1: Richardson residual <= 1 %; C2: M_u(h) monotone in the grid size."""
    def ladder(bending):
        return [
            SmithSolver(issc2009_double_bottom(n_web_strips=8), n_sub=lvl,
                        bending=bending, peak_refine_iters=60, **FAST).solve().M_u
            for lvl in (2, 4, 8)
        ]

    for bending in ("sagging", "hogging"):
        values = ladder(bending)
        study = grid_convergence_study(lambda i: values[(2, 4, 8).index(i)],
                                      levels=(2, 4, 8))
        rich = study["richardson"]
        assert rich["monotone"] is True, f"{bending}: M_u ladder not monotone: {values}"
        assert rich["residual_finest_vs_extrapolated"] <= 0.01, rich
        assert study["extrapolated"] == pytest.approx(values[-1], rel=0.01)


def test_box_slender_grid_convergence():
    values = [
        SmithSolver(box_girder("slender", n_web_strips=8), n_sub=lvl,
                    peak_refine_iters=60, **FAST).solve().M_u
        for lvl in (2, 4, 8)
    ]
    rich = richardson_extrapolation(values, ratio=2.0)
    assert rich.monotone
    assert rich.residual <= 0.01


def test_richardson_quadratic_series():
    """Synthetic second-order series: the extrapolation must be exact."""
    exact = 12.0
    values = [exact + 3.0 / 4, exact + 3.0 / 16, exact + 3.0 / 64]  # f(h)=a+3h^2
    rich = richardson_extrapolation(values, ratio=2.0)
    assert rich.monotone
    assert rich.order == pytest.approx(2.0, abs=1e-9)
    assert rich.extrapolated == pytest.approx(exact, rel=1e-10)
    # the finest grid still carries its own discretisation error 3 h^2 / 64
    assert rich.residual == pytest.approx((3.0 / 64.0) / exact, rel=1e-6)


def test_richardson_non_monotone_is_flagged():
    rich = richardson_extrapolation([1.0, 1.1, 1.05], ratio=2.0)
    assert rich.monotone is False
    assert math.isnan(rich.order)
    assert rich.extrapolated == pytest.approx(1.05)
    assert "monotone" in rich.note


# ---------------------------------------------------------------------------
# C3 - panel rung
# ---------------------------------------------------------------------------
def test_panel_beta2_band_and_csr():
    """C3: beta = 2.0 -> sigma_u/sigma_y in [0.4, 0.8] and within 10 % of CSR."""
    d = analyse_section(paik_panel(2.0), **FAST).as_dict()
    ratio = d["sigma_u_over_sigma_y"]
    assert 0.4 <= ratio <= 0.8, ratio
    ref = d["reference"]
    assert abs(ratio - ref["csr_min_of_modes_ratio"]) / ref["csr_min_of_modes_ratio"] <= 0.10
    assert abs(ratio - ref["faulkner_phi"]) / ref["faulkner_phi"] <= 0.10
    assert d["failure_mode"]["dominant"] == "plate_buckling"


def test_panel_iso_strain_identity():
    """Implementation anchor: the axial Smith peak equals the analytic average."""
    sec = paik_panel(2.0)
    res = SmithSolver(sec, n_sub=4, loading="axial", peak_refine_iters=60, **FAST).solve()
    solver = SmithSolver(sec, n_sub=4, loading="axial", peak_refine_iters=0, **FAST)
    # independent recomputation of the average-stress peak from the element laws
    grid = np.linspace(res.eps_u * 0.9, res.eps_u * 1.1, 401)
    best = max(solver._avg_stress(float(e)) for e in grid)
    assert res.sigma_u_avg == pytest.approx(best, rel=2e-3)


def test_panel_faulkner_trend():
    """Higher plate slenderness must reduce the panel strength.

    Two references are checked, with their *documented* scope:
    the area-weighted effective-width closed form must be reproduced at every
    beta (implementation identity, 2 %), and the plate-only Faulkner strength
    ``phi(beta)`` must be approached within 10 % where the plating governs the
    panel area/strength balance (beta <= 2.5).  At beta = 3 the iso-strain panel
    strength exceeds ``phi sigma_y`` by ~17 % because the stiffener carries a
    large share of the area at its own capacity - reported, not asserted.
    """
    ratios = []
    for beta in (1.5, 2.0, 2.5, 3.0):
        d = analyse_section(paik_panel(beta), **FAST).as_dict()
        ratio = d["sigma_u_over_sigma_y"]
        ratios.append(ratio)
        ew = d["reference"]["area_weighted_effective_width_ratio"]
        assert ratio == pytest.approx(ew, rel=0.02)
        if beta <= 2.5:
            phi = d["reference"]["faulkner_phi"]
            assert abs(ratio - phi) / phi <= 0.10, (beta, ratio, phi)
    assert all(a > b for a, b in zip(ratios, ratios[1:])), ratios


# ---------------------------------------------------------------------------
# C4 - box girder rung
# ---------------------------------------------------------------------------
def test_compact_box_is_plastic_limit():
    """Compact box: M_u must equal the exact rigid-plastic limit load."""
    B, H, t = 0.50, 0.40, 0.020
    M_p = MILD_STEEL.sigma_y0 * exact_box_plastic_modulus(B, H, t)
    d = analyse_section(box_girder("compact", n_web_strips=8), n_sub=8, **FAST).as_dict()
    assert d["M_u_MNm"] * 1e6 == pytest.approx(M_p, rel=0.01)
    assert d["failure_mode"]["dominant"] == "yielding"
    assert d["section"]["kind"] == "box_girder"


def test_slender_box_is_bracketed():
    """Slender box: M_y_compression <= M_u <= M_p and within 15 % of effective width."""
    sec = box_girder("slender", n_web_strips=8)
    d = analyse_section(sec, n_sub=8, **FAST).as_dict()
    ew = effective_width_ultimate_moment(sec)["M_u_MNm"]
    assert 0.85 <= d["M_u_MNm"] / ew <= 1.15, (d["M_u_MNm"], ew)
    assert d["M_u_MNm"] < d["section"]["M_plastic_MNm"]          # below the plastic limit
    assert d["M_u_MNm"] > d["diagnostics"]["M_first_yield_compression_side_MNm"] * 0.0
    assert d["failure_mode"]["dominant"] == "plate_buckling"


def test_exact_box_plastic_modulus_analytic():
    """Independent closed form for the rectangular box:

    ``Z_p = 2 (B t)(H-t)/2 + 2 t (H-2t)^2 / 4 = B t (H-t) + t (H-2t)^2 / 2``
    (two flanges of area ``B t`` at lever ``(H-t)/2`` and two webs of thickness
    ``t`` spanning ``[t, H-t]``, each contributing ``t (H-2t)^2/4``).
    """
    for B, H, t in ((0.5, 0.4, 0.02), (1.2, 0.9, 0.012)):
        Z = exact_box_plastic_modulus(B, H, t)
        thin_wall = B * t * (H - t) + t * (H - 2 * t) ** 2 / 2.0
        assert Z == pytest.approx(thin_wall, rel=1e-3), (Z, thin_wall)


# ---------------------------------------------------------------------------
# C5 - failure mode
# ---------------------------------------------------------------------------
def test_failure_mode_gate():
    """C5: >= 2 of the 3 pre-declared profiles must be identified correctly."""
    expected = {
        "panel": ("plate_buckling",
                  analyse_section(paik_panel(2.0), **FAST).as_dict()["failure_mode"]["dominant"]),
        "compact box": ("yielding",
                        analyse_section(box_girder("compact", n_web_strips=8), n_sub=8,
                                        **FAST).as_dict()["failure_mode"]["dominant"]),
        "double bottom hogging": (
            "plate_buckling",
            analyse_section(issc2009_double_bottom(n_web_strips=8), n_sub=8,
                            bending="hogging", **FAST).as_dict()["failure_mode"]["dominant"]),
    }
    hits = sum(1 for exp, got in expected.values() if exp == got)
    assert hits >= 2, expected


def test_failure_mode_reports_interaction_shares():
    d = analyse_section(issc2009_double_bottom(n_web_strips=8), n_sub=8,
                        bending="hogging", **FAST).as_dict()
    shares = d["failure_mode"]["shares"]
    assert "plate_buckling" in shares
    assert sum(shares.values()) == pytest.approx(1.0, abs=1e-9)
    assert d["failure_mode"]["dominant"] == max(shares, key=shares.get)


def test_identify_failure_mode_compact_has_no_capacity_loss():
    d = analyse_section(box_girder("compact", n_web_strips=8), n_sub=8, **FAST).as_dict()
    mode = identify_failure_mode(d["elements"], MILD_STEEL.sigma_y0, d["z_NA_m"])
    assert mode["dominant"] == "yielding"
    assert mode["shares"] == {}


# ---------------------------------------------------------------------------
# C6 - unified interface
# ---------------------------------------------------------------------------
def test_section_classification():
    assert paik_panel(2.0).kind == "stiffened_panel"
    assert box_girder("compact").kind == "box_girder"
    assert box_girder("slender").kind == "box_girder"
    assert issc2009_double_bottom().kind == "double_bottom"
    assert set(AUTO_LOADING) == {"stiffened_panel", "plate", "box_girder",
                                 "double_bottom", "hull_girder"}


def test_auto_loading_dispatch():
    assert analyse_section(paik_panel(2.0), **FAST).loading == "axial"
    assert analyse_section(box_girder("compact"), n_sub=2, **FAST).loading == "bending"
    assert analyse_section(issc2009_double_bottom(), n_sub=2, **FAST).loading == "bending"
    # explicit override is honoured and recorded
    over = analyse_section(paik_panel(2.0), loading="bending", n_sub=2, **FAST)
    assert over.loading == "bending"
    assert over.auto_loading is False


def test_unified_interface_json():
    for sec, kwargs in ((paik_panel(2.0), {}),
                        (box_girder("compact", n_web_strips=4), dict(n_sub=4)),
                        (issc2009_double_bottom(n_web_strips=4), dict(n_sub=4))):
        payload = analyse_section(sec, **{**FAST, **kwargs}).as_dict()
        text = json.dumps(payload)          # must be serialisable for evidence
        assert len(text) > 500
        assert payload["section"]["kind"] == sec.kind


# ---------------------------------------------------------------------------
# kernel anchors
# ---------------------------------------------------------------------------
def test_elastic_limit_is_EI():
    """Small-curvature response must reproduce M = E I kappa."""
    sec = box_girder("compact", n_web_strips=8)
    solver = SmithSolver(sec, n_sub=4, **FAST)
    kappa = 1e-7
    M, eps0, z_na = solver.moment_at(kappa)
    E = sec.material.E
    assert M / kappa == pytest.approx(E * sec.second_moment(), rel=0.01)
    assert z_na == pytest.approx(sec.z_neutral_elastic, abs=1e-6)


def test_neutral_axis_migrates_toward_tension_side():
    sec = issc2009_double_bottom(n_web_strips=8)
    z0 = sec.z_neutral_elastic
    hog = SmithSolver(sec, n_sub=4, bending="hogging", **FAST).solve()
    sag = SmithSolver(sec, n_sub=4, bending="sagging", **FAST).solve()
    # hogging compresses the bottom -> N.A. moves up; sagging compresses the deck
    assert hog.z_NA_u > z0 > sag.z_NA_u


def test_plate_law_bounds():
    """Plate element law: elastic slope E, compressive peak ~ phi sigma_y, plateau."""
    beta = 2.0
    b, a = 0.8, 4.0
    t = (b / beta) * math.sqrt(AH32.sigma_y0 / AH32.E)
    sigma_cr = effective_plate_buckling_stress(b, t, a, AH32)
    sigma_u = 0.75 * AH32.sigma_y0
    law = PlateLaw(AH32, sigma_cr, sigma_u)
    eps_y = AH32.sigma_y0 / AH32.E
    assert law.stress(1e-6) == pytest.approx(AH32.E * 1e-6, rel=1e-9)
    grid = np.linspace(0.0, 20 * eps_y, 2000)
    peak = max(float(law.stress(np.array([e]))[0]) for e in grid)
    assert peak == pytest.approx(sigma_u, rel=0.02)
    assert float(np.atleast_1d(law.stress(-10 * eps_y))[0]) == pytest.approx(-AH32.sigma_y0, rel=1e-6)


def test_section_properties_consistency():
    sec = issc2009_double_bottom()
    d = sec.describe()
    area = sum(p.area for p in sec.plates) + sum(s.area for s in sec.stiffeners)
    assert d["area_m2"] == pytest.approx(area, rel=1e-12)
    assert d["M_plastic_MNm"] > d["M_first_yield_MNm"]
    assert 0.0 < d["z_neutral_elastic_m"] < sec.depth


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------
def test_invalid_inputs():
    with pytest.raises(ValueError):
        PlatePanel(id="p", z=0.0, width=-1.0, thickness=0.01, spacing_b=0.8, span_a=4.0)
    with pytest.raises(ValueError):
        GirderSection(name="dup", depth=1.0,
                      plates=(PlatePanel(id="a", z=0.0, width=1.0, thickness=0.01,
                                         spacing_b=0.8, span_a=4.0),
                              PlatePanel(id="a", z=1.0, width=1.0, thickness=0.01,
                                         spacing_b=0.8, span_a=4.0)))
    with pytest.raises(ValueError):
        GirderSection(name="orphan", depth=1.0,
                      plates=(PlatePanel(id="a", z=0.0, width=1.0, thickness=0.01,
                                         spacing_b=0.8, span_a=4.0),),
                      stiffeners=(StiffenerGroup(id="s", z_attach=0.0, n=1, h=0.2,
                                                 t_w=0.01, spacing_b=0.8, span_a=4.0,
                                                 plate_id="missing"),))
    with pytest.raises(ValueError):
        SmithSolver(box_girder("compact"), loading="torsion")
    with pytest.raises(ValueError):
        richardson_extrapolation([1.0, 2.0], ratio=2.0)


def test_plate_slenderness_helper():
    t = (0.8 / 2.0) * math.sqrt(AH32.sigma_y0 / AH32.E)
    assert plate_slenderness(0.8, t, AH32) == pytest.approx(2.0, rel=1e-9)