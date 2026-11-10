"""Tests for :mod:`tensorphysics.hull_redundancy` (campaign node p351w1_n08).

Coverage map (acceptance criteria of the node):

(a) RIF range + intact self-check        -> test_rif_negative_control_unity,
                                            test_rif_in_unit_interval,
                                            test_every_single_damage_reduces_capacity
(b) neutral-axis negative control        -> test_negative_control_neutral_axis,
                                            test_negative_control_elastic_na_is_not_the_control
(c) monotonicity (Spearman rho <= -0.8)  -> test_sensitivity_rho_le_minus_08,
                                            test_spearman_helper
(d) published closed-form benchmark      -> test_benchmark_within_15_percent
(e) failure-mode ordering                -> test_mode_ordering_remove_debond_trip,
                                            test_tension_side_stiffener_modes_inactive
(f) discretisation convergence           -> test_discretisation_convergence
plus kernel/geometry anchors and input validation.
"""

from __future__ import annotations

import math

import pytest

from tensorphysics.hull_redundancy import (
    DamageSpec,
    SmithSectionSolver,
    apply_damage,
    bare_strut_stress,
    benchmark_vs_smith,
    convergence_study,
    double_hull_side_section,
    effective_plate_buckling_stress,
    faulkner_plate_strength_ratio,
    johnson_ostenfeld,
    most_dangerous,
    negative_control,
    p350w1_n03_combined_section,
    plate_buckling_stress,
    redundancy_set,
    rif_matrix,
    sensitivity_analysis,
    simplified_ultimate_moment,
    spearman_rho,
    tripping_stress,
    worst_case_by_component,
)
from tensorphysics.hull_redundancy.elements import plate_buckling_coefficient

SOLVER = {"n_steps": 80, "n_web_fibers": 2, "kappa_max": 6.0e-4}


@pytest.fixture(scope="module")
def section():
    return p350w1_n03_combined_section()


@pytest.fixture(scope="module")
def matrices(section):
    return {b: rif_matrix(section, solver_kwargs=dict(SOLVER), bending=b)
            for b in ("sagging", "hogging")}


# ---------------------------------------------------------------------------
# geometry / idealisation
# ---------------------------------------------------------------------------
def test_section_component_counts(section):
    """Campaign spec: <= 12 plate strips x <= 8 stiffener groups."""
    assert len(section.plates) == 12
    assert len(section.stiffeners) == 6
    assert len(set(section.component_ids())) == 18


def test_section_geometry_validation(section):
    """Independent recomputation of A, z_NA, I must reproduce the model."""
    val = section.geometry_validation()
    assert val["valid"] is True
    assert val["NA_inside_depth"] is True
    assert val["area_rel_err_vs_model"] < 1e-12
    assert val["I_rel_err_vs_model"] < 1e-12
    assert 5.0 < val["z_NA_m"] < 15.0


def test_section_section_modulus_is_consistent(section):
    """S = I / c must reproduce the elastic bending resistance."""
    sm = section.section_moduli()
    assert sm["S_deck_m3"] == pytest.approx(sm["I_m4"] / (section.depth - sm["z_NA_m"]))
    assert sm["S_bottom_m3"] == pytest.approx(sm["I_m4"] / sm["z_NA_m"])


def test_secondary_section_is_valid_and_smaller():
    sec = double_hull_side_section()
    assert len(sec.plates) == 8 and len(sec.stiffeners) == 6
    assert sec.geometry_validation()["valid"] is True
    assert sec.second_moment() < p350w1_n03_combined_section().second_moment()


# ---------------------------------------------------------------------------
# element laws
# ---------------------------------------------------------------------------
def test_plate_buckling_coefficient_long_panel():
    assert plate_buckling_coefficient(3.0) == pytest.approx(4.0)
    assert plate_buckling_coefficient(1.0) == pytest.approx(4.0)
    assert plate_buckling_coefficient(0.5) == pytest.approx((0.5 + 2.0) ** 2)


def test_plate_buckling_stress_matches_analytic(section):
    """k pi^2 E/(12(1-nu^2)) (t/b)^2 for a/b = 3."""
    m = section.material
    val = plate_buckling_stress(m, b=0.8, t=0.016, a=2.4)
    ref = 4.0 * math.pi**2 * m.E / (12.0 * (1 - m.nu**2)) * (0.016 / 0.8) ** 2
    assert val == pytest.approx(ref, rel=1e-12)


def test_effective_buckling_stress_reproduces_faulkner(section):
    """sqrt(sigma_cr,eff * sigma_y) must equal phi(beta) sigma_y (von Karman)."""
    m = section.material
    for t in (0.010, 0.012, 0.016, 0.025):
        scr = effective_plate_buckling_stress(m, b=0.8, t=t, a=2.4)
        beta = (0.8 / t) * math.sqrt(m.sigma_y0 / m.E)
        phi = faulkner_plate_strength_ratio(beta)
        assert math.sqrt(scr * m.sigma_y0) == pytest.approx(phi * m.sigma_y0, rel=1e-9)
        assert scr <= m.sigma_y0 + 1e-6


def test_faulkner_curve_limits():
    assert faulkner_plate_strength_ratio(1.0) == pytest.approx(1.0)
    assert faulkner_plate_strength_ratio(0.5) == pytest.approx(1.0)
    assert faulkner_plate_strength_ratio(2.0) == pytest.approx(0.75)
    assert faulkner_plate_strength_ratio(4.0) == pytest.approx(0.4375)


def test_tripping_stress_below_yield_and_scales_with_slenderness(section):
    """sigma_trip ~ G (t_w/h)^2: doubling t_w must roughly quadruple it."""
    m = section.material
    # thin walls keep the elastic (pre-Johnson-Ostenfeld) regime: sigma ~ (t_w/h)^2
    s1 = tripping_stress(m, h=0.35, t_w=0.005, span=2.4, plate_t=0.016)
    s2 = tripping_stress(m, h=0.35, t_w=0.010, span=2.4, plate_t=0.016)
    assert s1 < 0.5 * m.sigma_y0 and s2 < 0.5 * m.sigma_y0
    assert s2 / s1 == pytest.approx(4.0, rel=0.02)
    # and a realistic hull longitudinal trips well below yield
    assert tripping_stress(m, h=0.30, t_w=0.011, span=2.4, plate_t=0.016) < m.sigma_y0


def test_bare_strut_matches_euler_with_johnson_ostenfeld(section):
    m = section.material
    h, t_w, span = 0.30, 0.011, 2.4
    val = bare_strut_stress(m, h=h, t_w=t_w, span=span)
    r = h / math.sqrt(12.0)
    sigma_e = math.pi**2 * m.E / (span / r) ** 2
    assert val == pytest.approx(johnson_ostenfeld(sigma_e, m.sigma_y0), rel=1e-12)
    assert val <= m.sigma_y0


def test_johnson_ostenfeld_limits(section):
    m = section.material
    assert johnson_ostenfeld(0.1 * m.sigma_y0, m.sigma_y0) == pytest.approx(0.1 * m.sigma_y0)
    assert johnson_ostenfeld(50.0 * m.sigma_y0, m.sigma_y0) == pytest.approx(
        m.sigma_y0 * (1 - m.sigma_y0 / (4 * 50.0 * m.sigma_y0)))
    with pytest.raises(ValueError):
        johnson_ostenfeld(-1.0, m.sigma_y0)


# ---------------------------------------------------------------------------
# damage model
# ---------------------------------------------------------------------------
def test_invalid_damage_mode_rejected(section):
    with pytest.raises(ValueError):
        DamageSpec("PB1", "explode")
    with pytest.raises(KeyError):
        apply_damage(section, DamageSpec("NOPE", "remove"))


def test_removal_zeroes_the_element(section):
    dmg = apply_damage(section, DamageSpec("PD1", "remove"))
    plate = next(p for p in dmg.plates if p.id == "PD1")
    assert plate.removed is True
    assert plate.area == 0.0
    assert dmg.area() < section.area()


def test_tripping_caps_stress_but_keeps_plating_support(section):
    dmg = apply_damage(section, DamageSpec("SD1", "trip"))
    stiff = next(s for s in dmg.stiffeners if s.id == "SD1")
    plate = next(p for p in dmg.plates if p.id == "PD1")
    assert stiff.cap is not None and stiff.cap < section.material.sigma_y0
    assert stiff.area == pytest.approx(section.stiffener("SD1").area)
    assert plate.span_b == pytest.approx(section.plate("PD1").span_b)
    assert plate.support_lost is False


def test_debond_and_removal_double_the_plate_span(section):
    for mode in ("debond", "remove"):
        dmg = apply_damage(section, DamageSpec("SD1", mode))
        plate = next(p for p in dmg.plates if p.id == "PD1")
        assert plate.support_lost is True
        assert plate.span_b == pytest.approx(2.0 * section.plate("PD1").span_b)
        # the doubled panel must buckle at a lower stress
        intact = apply_damage(section, DamageSpec("PB1", "none"))
        sigma_intact = next(q for q in intact.plates if q.id == "PD1").sigma_cr
        assert plate.sigma_cr < sigma_intact
    dmg_d = apply_damage(section, DamageSpec("SD1", "debond"))
    dmg_r = apply_damage(section, DamageSpec("SD1", "remove"))
    stiff_d = next(s for s in dmg_d.stiffeners if s.id == "SD1")
    stiff_r = next(s for s in dmg_r.stiffeners if s.id == "SD1")
    assert stiff_d.area > 0.0 and stiff_r.area == 0.0


# ---------------------------------------------------------------------------
# Smith solver anchors
# ---------------------------------------------------------------------------
def test_smith_elastic_slope_reproduces_EI(section):
    solver = SmithSectionSolver(apply_damage(section, DamageSpec("PB1", "none")),
                                n_steps=8, n_web_fibers=2)
    slope = solver.elastic_slope(1e-9)
    assert slope == pytest.approx(section.material.E * section.second_moment(), rel=1e-3)


def test_smith_first_yield_moment(section):
    """At first yield M must be close to sigma_y * S_deck."""
    res = SmithSectionSolver(apply_damage(section, DamageSpec("PB1", "none")),
                             **SOLVER).solve()
    eps_y = section.material.sigma_y0 / section.material.E
    # curvature at first yield of the deck fibre
    kappa_y = eps_y / (section.depth - section.neutral_axis())
    M_y = next(m for k, m in res.moment_curvature if k >= kappa_y)
    M_ref = section.material.sigma_y0 * section.section_moduli()["S_deck_m3"]
    assert M_y == pytest.approx(M_ref, rel=0.10)


def test_moment_curvature_has_interior_peak(matrices):
    """The ultimate point must be found inside the sweep (not clipped at kappa_max)."""
    for m in matrices.values():
        res = m["_intact_result"]
        assert res.converged is True
        assert abs(res.kappa_u) < SOLVER["kappa_max"]


# ---------------------------------------------------------------------------
# criterion (a)
# ---------------------------------------------------------------------------
def test_rif_negative_control_unity(matrices):
    for m in matrices.values():
        assert m["rif_negative_control"] == pytest.approx(1.0, abs=1e-3)


def test_rif_in_unit_interval(matrices):
    for m in matrices.values():
        assert m["rif_in_unit_interval"] is True
        for e in m["entries"]:
            assert -1e-12 <= e["rif"] <= 1.0 + 1e-12


def test_every_single_damage_reduces_capacity(matrices):
    for m in matrices.values():
        for e in m["entries"]:
            assert e["M_u_damaged_MNm"] <= e["M_u_intact_MNm"] + 1e-9


# ---------------------------------------------------------------------------
# criterion (b)
# ---------------------------------------------------------------------------
def test_negative_control_neutral_axis(matrices):
    for bending, m in matrices.items():
        nc = negative_control(m)
        assert nc["ultimate_NA_pass"] is True, bending
        for pick in nc["ultimate_NA_picks"]:
            assert pick["rif"] >= 0.98


def test_negative_control_elastic_na_is_not_the_control(matrices):
    """Finding of the study: in sagging the plastic NA sits 3.8 m below the elastic one,
    so a mid-side strip on the elastic axis still costs ~5 % of M_u."""
    m = matrices["sagging"]
    assert m["z_NA_ultimate_m"] < m["z_NA_elastic_m"] - 2.0
    nc = negative_control(m)
    elastic_pick = nc["elastic_NA_picks"][0]
    assert elastic_pick["component_id"] in {"PO2", "PI2"}
    assert 0.90 < elastic_pick["rif"] < 0.98


# ---------------------------------------------------------------------------
# criterion (c)
# ---------------------------------------------------------------------------
def test_spearman_helper():
    assert spearman_rho([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman_rho([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    # ties are rank-averaged, so a constant vector gives rho = 0
    assert spearman_rho([1, 1, 1, 1], [1, 2, 3, 4]) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        spearman_rho([1, 2], [1, 2])


def test_sensitivity_rho_le_minus_08(matrices):
    for bending, m in matrices.items():
        sens = sensitivity_analysis(m, reference="ultimate")
        assert sens["pooled"]["spearman_rho"] <= -0.8, bending
        assert sens["plating"]["spearman_rho"] <= -0.8, bending
        assert sens["monotone_ok"] is True


# ---------------------------------------------------------------------------
# ranking / redundancy set
# ---------------------------------------------------------------------------
def test_worst_component_is_on_the_compression_flange(matrices):
    assert most_dangerous(matrices["sagging"], 1)[0]["component_id"].startswith("PD")
    assert most_dangerous(matrices["hogging"], 1)[0]["component_id"].startswith("PB")


def test_top3_and_ranking_are_ordered(matrices):
    for m in matrices.values():
        ranking = worst_case_by_component(m)
        rifs = [r["rif_min"] for r in ranking]
        assert rifs == sorted(rifs)
        assert len(most_dangerous(m, 3)) == 3


def test_redundancy_set_threshold(matrices):
    for m in matrices.values():
        rs = redundancy_set(m, threshold=0.80)
        assert rs["n_redundant"] + rs["n_critical"] == 18
        for r in rs["redundant_components"]:
            assert r["rif_min"] >= 0.80
        assert rs["all_single_failures_redundant"] is True


def test_secondary_section_has_a_critical_component():
    """The stand-alone double-hull side loses its redundancy (metric discriminates)."""
    sec = double_hull_side_section()
    m = rif_matrix(sec, solver_kwargs=dict(SOLVER), bending="sagging")
    rs = redundancy_set(m, threshold=0.80)
    assert rs["n_critical"] >= 1
    assert rs["all_single_failures_redundant"] is False


# ---------------------------------------------------------------------------
# criterion (e)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bending,cid", [("sagging", "SD1"), ("hogging", "SB1")])
def test_mode_ordering_remove_debond_trip(matrices, bending, cid):
    rows = {e["mode"]: e["rif"] for e in matrices[bending]["entries"]
            if e["component_id"] == cid}
    assert rows["remove"] < rows["debond"] < rows["trip"] < 1.0
    assert (rows["debond"] - rows["remove"]) > 1e-3
    assert (rows["trip"] - rows["debond"]) > 1e-3


def test_tension_side_stiffener_modes_inactive(matrices):
    """A stiffener in tension cannot trip: cap is compression-only, so RIF = 1."""
    m = matrices["sagging"]
    for cid in ("SB1", "SB2", "SB3"):
        for mode in ("trip", "debond"):
            rif = next(e["rif"] for e in m["entries"]
                       if e["component_id"] == cid and e["mode"] == mode)
            assert rif == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# criterion (f) and (d)
# ---------------------------------------------------------------------------
def test_discretisation_convergence(section):
    for cid, mode in (("PD1", "remove"), ("SD1", "trip")):
        c = convergence_study(section, DamageSpec(cid, mode),
                              {"n_web_fibers": 2, "n_steps": 60,
                               "kappa_max": SOLVER["kappa_max"], "bending": "sagging"})
        assert c["relative_change"] <= 0.03
        assert c["converged_at_3pct"] is True


def test_benchmark_within_15_percent(section, matrices):
    m = matrices["sagging"]
    worst = most_dangerous(m, 1)[0]["component_id"]
    for cid, mode in (("SD1", "remove"), (worst, "remove")):
        e = next(x for x in m["entries"]
                 if x["component_id"] == cid and x["mode"] == mode)
        b = benchmark_vs_smith(section, DamageSpec(cid, mode),
                               smith_M_u=e["M_u_damaged_MNm"] * 1e6)
        assert b["relative_error"] <= 0.15
        assert b["within_15pct"] is True
    # intact section against the closed-form method
    ref = simplified_ultimate_moment(section, DamageSpec("PB1", "none"))
    assert abs(m["M_u_intact_MNm"] - ref["M_u_MNm"]) / ref["M_u_MNm"] <= 0.15