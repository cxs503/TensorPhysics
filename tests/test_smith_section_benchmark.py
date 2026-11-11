"""Tests for the generalised Smith-method section analysis (IN387-FEM01).

Coverage map (each block is an invariant that would break silently if the
implementation regressed):

1. component + section geometry statics (area, neutral axis, second moment,
   section modulus, inclined plate area);
2. fiberisation conserves area and first moment for any refinement level;
3. the element load-end shortening curve is monotone, capped at the element
   strength, softens after its plateau and yields in tension;
4. the axial panel solver reproduces the element-wise (CSR-type) capacity;
5. the panel ultimate strength decreases with plate slenderness (Perry-type);
6. Richardson extrapolation recovers a known order/exact limit and flags a
   non-monotone sequence;
7. the bending solver reproduces the closed-form plastic moment of a stocky box
   and refines monotonically with the fiber count;
8. the slender box solution respects the [elastic plate-strength moment,
   full-plastic moment] bounds and the limit-analysis reference band;
9. the KCS midship section reproduces the published midship area coefficient
   (geometry gate) and has sane section properties;
10. sagging/hogging of a symmetric box are equal in magnitude and opposite in
    sign;
11. the failure-mode classifier returns the *a priori* expected mode for the
    three declared cases (plate buckling / stiffener tripping / yielding);
12. the wave-demand levels are mutually consistent (IACS formula vs design-wave
    strip model) and the load-capacity handoff gate returns a structured record;
13. input validation of the component classes.

All tests are deterministic and use no external data; the heaviest ones (7, 8,
9, 12) stay below a few seconds each.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorphysics.smith_section_benchmark import (
    PlateStrip,
    SectionModel,
    SmithSectionSolver,
    StiffenerGroup,
    axial_ultimate_strength,
    box_girder,
    box_girder_plastic_moment,
    csr_type_panel_strength,
    element_stress,
    faulkner_plate_ratio,
    grid_convergence_criterion,
    issc2009_double_bottom_unit,
    kcs_girder,
    kcs_midship_section,
    kcs_particulars,
    load_capacity_handoff,
    mesh_convergence_study,
    paik_style_panel,
    richardson_extrapolate,
    wave_demand,
)
from tensorphysics.smith_section_benchmark.benchmarks import (
    midship_area_from_outline,
    rigid_plastic_reference,
)
from tensorphysics.smith_section_benchmark.solver import build_element_arrays
from tensorphysics.smith_section_benchmark.strength_curves import (
    plate_ultimate_stress,
)

STEEL_E = 206.0e9


# ---------------------------------------------------------------------------
# 1. geometry statics
# ---------------------------------------------------------------------------
def test_section_geometry_statics():
    """Box section area / neutral axis / second moment / modulus are exact."""
    b = d = 1.0
    t = 0.010
    sy = 250.0e6
    box = box_girder(breadth=b, depth=d, t_flange=t, t_web=t, span=2.0, sigma_y=sy)
    a_expected = 2 * b * t + 2 * t * (d - 2 * t)
    assert box.area == pytest.approx(a_expected, rel=1e-12)
    assert box.neutral_axis == pytest.approx(d / 2.0, rel=1e-9)
    # the idealisation carries the parallel-axis term of the plating and the own
    # inertia of the (vertical) webs; the flange own inertia t^3/12 is dropped
    # because t << b (declared in SectionModel.second_moment)
    i_expected = (2 * (b * t * (d / 2 - t / 2) ** 2)
                  + 2 * (t * (d - 2 * t) ** 3 / 12))
    assert box.second_moment == pytest.approx(i_expected, rel=1e-9)
    assert 2 * (b * t ** 3 / 12) / box.second_moment < 1e-3
    sm = box.section_moduli
    assert sm["S_deck"] == pytest.approx(box.second_moment / (d / 2.0), rel=1e-9)
    assert sm["S_deck"] == pytest.approx(sm["S_keel"], rel=1e-9)
    gate = box.geometry_validation()
    assert gate["all_pass"] is True
    # inclined plate: area is arc length x thickness
    inc = PlateStrip(name="bilge", thickness=0.02, span_b=0.8, span_a=2.4,
                     z_low=0.0, z_high=1.0, arc_length=math.pi / 2, sigma_y=sy)
    assert inc.area == pytest.approx(math.pi / 2 * 0.02, rel=1e-12)
    assert inc.is_vertical is True
    assert inc.z_centroid == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 2. fiberisation invariance
# ---------------------------------------------------------------------------
def test_fiberisation_conserves_area_and_first_moment():
    """Fiber areas sum to the section area and reproduce the first moment."""
    sec = issc2009_double_bottom_unit()
    for n in (1, 2, 4, 8):
        fibers = sec.fibers(n_fibers=n, n_web_fibers=max(1, n // 2))
        a = sum(f.area for f in fibers)
        m1 = sum(f.area * f.z for f in fibers)
        assert a == pytest.approx(sec.area, rel=1e-12)
        assert m1 / a == pytest.approx(sec.neutral_axis, rel=1e-9)
    # sub-fiber counts also conserve the area of a stiffener web
    sec2 = paik_style_panel(beta=2.0)
    for nw in (1, 3, 7):
        fibers = sec2.fibers(n_fibers=2, n_web_fibers=nw)
        assert sum(f.area for f in fibers) == pytest.approx(sec2.area, rel=1e-12)


# ---------------------------------------------------------------------------
# 3. element curve invariants
# ---------------------------------------------------------------------------
def test_element_curve_monotone_capped_and_softening():
    """The plate element stress rises, is capped and then softens; tension EPP."""
    sec = paik_style_panel(beta=3.0, plate_law="faulkner")
    el = build_element_arrays(sec.fibers(n_fibers=1, n_web_fibers=1))
    strains = np.linspace(1e-5, 0.02, 400)
    plate = np.array([el.is_plate])[0]
    sig_all = np.array([element_stress(float(s), el, STEEL_E) for s in strains])
    assert np.all(sig_all <= el.sigma_cap * (1.0 + 1e-9))   # capped everywhere
    sig = sig_all[:, plate].max(axis=1)                     # plate element only
    cap_plate = float(el.sigma_cap[plate][0])
    assert sig[0] < sig[10]                     # rising
    assert sig.max() == pytest.approx(cap_plate, rel=1e-3)  # reaches its strength
    assert sig[-1] < 0.9 * sig.max()            # softening branch develops
    # tension side: elastic then perfectly plastic at yield, never above it
    s_ten = np.array([-1e-4, -1e-3, -2e-2])
    for s in s_ten:
        st = element_stress(float(s), el, STEEL_E)
        assert np.all(st <= 0.0)
        assert np.all(st >= -float(el.sigma_y.max()) - 1e-6)
    assert element_stress(-2e-2, el, STEEL_E).min() == pytest.approx(
        -float(el.sigma_y.max()), rel=1e-9)


# ---------------------------------------------------------------------------
# 4/5. panel axial solver
# ---------------------------------------------------------------------------
def test_axial_strength_matches_elementwise_capacity():
    """The panel axial ultimate equals the area-weighted element capacities."""
    sec = paik_style_panel(beta=2.0, plate_law="faulkner")
    res = axial_ultimate_strength(sec, n_fibers=2, n_web_fibers=2, n_steps=200)
    csr = csr_type_panel_strength(sec.plates, sec.stiffeners, sec.material,
                                  plate_law="faulkner")
    assert res.sigma_u == pytest.approx(csr["sigma_u"], rel=0.02)
    assert 0.4 <= res.sigma_u_over_sigma_y <= 0.8


def test_axial_strength_decreases_with_plate_slenderness():
    """Perry-type monotonicity: sigma_u/sigma_y falls as beta grows."""
    ratios = []
    for beta in (1.2, 1.5, 2.0, 2.5, 3.0):
        sec = paik_style_panel(beta=beta, plate_law="faulkner")
        ratios.append(axial_ultimate_strength(sec, n_fibers=2, n_web_fibers=2,
                                              n_steps=160).sigma_u_over_sigma_y)
    assert all(r1 >= r2 for r1, r2 in zip(ratios[:-1], ratios[1:]))
    assert ratios[0] > 0.9 and ratios[-1] < 0.75


# ---------------------------------------------------------------------------
# 6. Richardson machinery
# ---------------------------------------------------------------------------
def test_richardson_recovers_known_order_and_limit():
    """Second-order convergence is recovered from three synthetic meshes."""
    exact, coeff = 4.473e9, 1.25e7

    def value(h):
        return exact + coeff * h ** 2

    values = [value(0.4), value(0.2), value(0.1)]
    out = richardson_extrapolate(values, refinement_ratio=2.0)
    assert out["observed_order"] == pytest.approx(2.0, abs=1e-6)
    assert out["extrapolated"] == pytest.approx(exact, rel=1e-10)
    assert out["monotone"] is True
    non_monotone = richardson_extrapolate([1.0, 2.0, 1.5], 2.0)
    assert non_monotone["monotone"] is False
    assert non_monotone["observed_order"] is None
    assert non_monotone["evidence_type"] == "successive_difference_bound"
    # a mesh-independent quantity is reported as such, not with a fabricated order
    constant = richardson_extrapolate([5.0, 5.0, 5.0, 5.0], 2.0)
    assert constant["evidence_type"] == "mesh_independent"
    assert constant["observed_order"] is None
    assert constant["finest_vs_extrapolated"] == 0.0
    with pytest.raises(ValueError):
        richardson_extrapolate([1.0, 2.0])


def test_mesh_convergence_study_grid_criterion():
    """The convergence study returns the criterion value it gates on."""
    def evaluate(level: int) -> float:
        return 100.0 + 3.0 / level ** 2

    study = mesh_convergence_study(evaluate, levels=(1, 2, 4), tolerance=1e-2,
                                   label="synthetic")
    crit = grid_convergence_criterion(study, 1e-2)
    assert crit["observed_order"] == pytest.approx(2.0, abs=1e-6)
    assert crit["extrapolated"] == pytest.approx(100.0, rel=1e-9)
    assert crit["passes"] is True
    strict = grid_convergence_criterion(study, 1e-9)
    assert strict["passes"] is False


# ---------------------------------------------------------------------------
# 7/8. bending solver vs limit analysis
# ---------------------------------------------------------------------------
def test_stocky_box_reproduces_plastic_moment():
    """A stocky box collapses at its full-plastic moment, mesh independently."""
    box = box_girder(breadth=0.40, depth=0.40, t_flange=0.012, t_web=0.012,
                     span=1.6, sigma_y=250.0e6)
    ref = box_girder_plastic_moment(box)
    results = [abs(SmithSectionSolver(box, n_fibers=n, n_web_fibers=2,
                                      n_kappa=40).solve().M_u)
               for n in (2, 4, 8)]
    for m in results:
        assert m / ref["M_plastic_Nm"] == pytest.approx(1.0, abs=0.01)
    # the closed-form plastic moment itself is checked against hand statics
    a_f = 0.40 * 0.012
    m_hand = (2 * a_f * 250.0e6 * (0.194)
              + 2 * (0.012 * 250.0e6) * (0.188 ** 2))
    assert ref["M_plastic_Nm"] == pytest.approx(m_hand, rel=1e-3)


def test_slender_box_within_physical_bounds():
    """Slender box: M_u between the plate-strength and full-plastic moments."""
    box = box_girder(breadth=0.80, depth=0.80, t_flange=0.008, t_web=0.008,
                     span=1.6, sigma_y=300.0e6, plate_law="faulkner")
    sy = 300.0e6
    beta = box.plates[0].plate_slenderness(sy)
    assert beta > 3.0
    res = SmithSectionSolver(box, n_fibers=4, n_web_fibers=2, n_kappa=48).solve()
    m_u = abs(res.M_u)
    ref = rigid_plastic_reference(box, sigma_y=sy)
    m_lower = faulkner_plate_ratio(beta) * sy * box.section_moduli["S_deck"]
    assert m_lower < m_u < ref["M_plastic_Nm"]
    assert m_u / ref["M_reference_Nm"] == pytest.approx(1.0, abs=0.15)
    assert res.failure_mode == "plate_buckling"


def test_sagging_hogging_symmetry_of_symmetric_box():
    """A doubly symmetric box has equal sagging/hogging capacity, opposite sign."""
    box = box_girder(breadth=0.50, depth=0.50, t_flange=0.010, t_web=0.006,
                     span=1.6, sigma_y=250.0e6)
    sag = SmithSectionSolver(box, n_fibers=2, n_web_fibers=2, n_kappa=40,
                             bending="sagging").solve()
    hog = SmithSectionSolver(box, n_fibers=2, n_web_fibers=2, n_kappa=40,
                             bending="hogging").solve()
    assert sag.M_u > 0.0 > hog.M_u
    assert abs(abs(sag.M_u) - abs(hog.M_u)) / abs(sag.M_u) < 0.05


# ---------------------------------------------------------------------------
# 9. KCS geometry gate
# ---------------------------------------------------------------------------
def test_kcs_midship_section_geometry_gate():
    """The modelled KCS midship section reproduces the published Cm."""
    kcs = kcs_midship_section()
    gate = kcs.geometry_gate()
    assert gate["all_pass"] is True
    assert gate["Cm_model"] == pytest.approx(kcs.particulars["Cm"], abs=2e-3)
    p = kcs_particulars()
    assert p["Lpp"] == pytest.approx(230.0)
    assert p["B"] == pytest.approx(32.2)
    assert p["T"] == pytest.approx(10.8)
    # the outline integration (the CAD / lines-plan adapter) must reproduce the
    # closed-form area and therefore the published Cm as well
    area_outline = midship_area_from_outline(kcs.outline(), p["T"])
    assert area_outline == pytest.approx(kcs.midship_area(), rel=1e-3)
    assert gate["checks"]["outline_area_matches_closed_form"] is True
    assert midship_area_from_outline([(0.0, 0.0), (1.0, 0.0), (1.0, 2.0)], 2.0) == pytest.approx(4.0, rel=1e-6)
    sm = kcs.section.section_moduli
    assert 5.0 < sm["S_deck"] < 60.0
    assert 5.0 < sm["S_keel"] < 60.0
    assert 0.0 < sm["z_neutral_axis"] < kcs.particulars["D"] + 1.5


# ---------------------------------------------------------------------------
# 10/11. failure mode classification
# ---------------------------------------------------------------------------
def test_failure_mode_classification_cases():
    """The declared mechanism set is recovered for the three teaching cases."""
    # plate buckling
    sec = paik_style_panel(beta=3.0, plate_law="faulkner")
    r = axial_ultimate_strength(sec, n_fibers=2, n_web_fibers=2, n_steps=200)
    assert r.failure_mode == "plate_buckling"
    # stiffener tripping (slender flat bar on stocky plating)
    base = paik_style_panel(beta=0.9, plate_law="faulkner")
    sec2 = SectionModel(
        name="tripping", plates=base.plates,
        stiffeners=(StiffenerGroup(name="bar", n_long=1, h_s=0.250, t_w=0.014,
                                   b_f=0.0, t_f=0.0, spacing=0.8, span=4.0,
                                   z_base=base.plates[0].thickness, sigma_y=315.0e6),),
        material=base.material, plate_law="faulkner", stiffener_column_axis="weak")
    r2 = axial_ultimate_strength(sec2, n_fibers=2, n_web_fibers=2, n_steps=200)
    assert r2.failure_mode == "stiffener_tripping"
    # general yielding of a stocky box
    box = box_girder(breadth=0.40, depth=0.40, t_flange=0.012, t_web=0.012,
                     span=1.6, sigma_y=250.0e6)
    r3 = SmithSectionSolver(box, n_fibers=2, n_web_fibers=2, n_kappa=32).solve()
    assert r3.failure_mode == "yielding"


# ---------------------------------------------------------------------------
# 12. demand / handoff
# ---------------------------------------------------------------------------
def test_wave_demand_levels_are_consistent():
    """IACS formula and the design-wave strip model agree within 10%."""
    ship = kcs_girder(section_modulus=10.4)
    w = wave_demand(ship, n_cells=8, n_omega=40)
    iacs = max(w["IACS_M_sagging_Nm"], w["IACS_M_hogging_Nm"])
    assert w["M_design_wave_strip_Nm"] / iacs == pytest.approx(1.0, abs=0.10)
    assert w["M_wave_extreme_Nm"] == pytest.approx(
        max(iacs, w["M_design_wave_strip_Nm"]), rel=1e-12)
    assert w["M_long_term_upper_Nm"] > 0.0


def test_load_capacity_handoff_record():
    """The handoff gate returns capacity, demand, margins and pass flags."""
    ship = kcs_girder(section_modulus=10.4)
    rec = load_capacity_handoff(4.4e9, 5.2e9, ship, required_margin=1.5)
    assert rec["gate"]["passes"] is True
    assert rec["margins"]["governing"] == "sagging"
    assert rec["margins"]["governing_margin"] == pytest.approx(4.4e9 / rec["demand"]
                                                              ["M_wave_extreme_Nm"], rel=1e-12)
    assert rec["capacity"]["evidence_level"] == "REAL"
    strict = load_capacity_handoff(1.0e9, 1.0e9, ship, required_margin=1.5)
    assert strict["gate"]["passes"] is False


# ---------------------------------------------------------------------------
# 13. validation
# ---------------------------------------------------------------------------
def test_component_validation():
    """Component classes reject physically impossible input."""
    with pytest.raises(ValueError):
        PlateStrip(name="bad", thickness=0.0, span_b=0.8, span_a=2.4,
                   z_low=0.0, z_high=0.01, width=1.0)
    with pytest.raises(ValueError):
        PlateStrip(name="bad", thickness=0.01, span_b=0.8, span_a=2.4,
                   z_low=0.0, z_high=0.01, width=1.0, arc_length=2.0)
    with pytest.raises(ValueError):
        StiffenerGroup(name="bad", n_long=1, h_s=0.2, t_w=0.01, b_f=0.1, t_f=0.0,
                       spacing=0.8, span=2.4, z_base=0.0)
    with pytest.raises(ValueError):
        SectionModel(name="empty", plates=())
    sec = paik_style_panel(beta=2.0)
    with pytest.raises(ValueError):
        SectionModel(name="bad_law", plates=sec.plates, plate_law="euler")


def test_plate_ultimate_stress_laws():
    """The two plate laws bracket the plate strength at beta = 2."""
    sigma_cr = 0.9 * 315.0e6
    vk, src_vk = plate_ultimate_stress(2.0, sigma_cr, 315.0e6, law="von_karman")
    fa, src_fa = plate_ultimate_stress(2.0, sigma_cr, 315.0e6, law="faulkner")
    assert src_vk == "plate_von_karman" and src_fa == "plate_faulkner"
    assert vk > fa > 0.6 * 315.0e6
    assert faulkner_plate_ratio(0.9) == pytest.approx(1.0)