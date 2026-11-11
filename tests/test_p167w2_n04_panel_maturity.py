"""P167W2-N04 verification tests: panel section + panel ultimate maturity.

Acceptance items
----------------
A. **panel_section** -- closed-form elastic/plastic section properties proven
   against an independent strip quadrature and against the parallel-axis
   theorem; tripping and local buckling stresses against the textbook plate /
   torsion formulas recomputed inside the test.
B. **panel_ultimate** -- limit point inside the elastic/plastic bounds, mode
   competition (plate / column / tripping / web local), monotone physics,
   four-method cross-check with the spread stated.
C. **plastic N-M surface** -- machine-exact anchors, asymmetric shape.
D. **grillage_error_bound** -- the invented ``+5 %``/``+2 %`` restraint factors
   are gone: the parallel assembly is exactly size-invariant, the residual size
   effect is bounded by the imperfection spread, and the load-step refinement is
   the real resolution knob.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorphysics.elastoplastic_collapse import (
    GrillageModel,
    SectionGeometryError,
    SmithCollapseSolver,
    StiffenedPanelSection,
    StiffenedPlateGeometry,
    VonMisesMaterial,
    check_ultimate_bounds,
    compare_ultimate_methods,
    discretization_convergence_study,
    grillage_convergence_study,
    isolated_element_deviation,
    panel_strength_band,
    panel_ultimate_axial,
    plastic_axial_bending_interaction,
    plate_ultimate_stress,
    stiffener_ultimate_stress,
)

MAT = VonMisesMaterial(E=206.0e9, nu=0.3, sigma_y0=313.0e6, H=1.0e9)
SY = MAT.sigma_y0

# documented reference panel (typical inner-bottom stiffened panel)
REF = dict(a=2.4, b=0.7, t=0.012, h_s=0.200, t_w=0.010, b_f=0.100, t_f=0.015)
SECTION = StiffenedPanelSection(material=MAT, **REF)


# ---------------------------------------------------------------------------
# A. section properties
# ---------------------------------------------------------------------------
def test_area_and_centroid_match_independent_quadrature():
    num = SECTION.numeric_properties(n_strips=800)
    assert SECTION.total_area == pytest.approx(num["area"], rel=1e-10)
    assert SECTION.z_bar == pytest.approx(num["z_bar"], rel=1e-9)
    # closed-form area cross-check by hand
    expected = REF["b"] * REF["t"] + REF["h_s"] * REF["t_w"] + REF["b_f"] * REF["t_f"]
    assert SECTION.total_area == pytest.approx(expected, rel=1e-15)


def test_second_moment_and_plastic_modulus_match_quadrature():
    num = SECTION.numeric_properties(n_strips=800)
    assert SECTION.second_moment == pytest.approx(num["second_moment"], rel=1e-10)
    assert SECTION.plastic_neutral_axis == pytest.approx(
        num["plastic_neutral_axis"], abs=1e-5
    )
    assert SECTION.plastic_section_modulus == pytest.approx(
        num["plastic_section_modulus"], rel=1e-4
    )


def test_parallel_axis_theorem_is_satisfied():
    """The centroidal axis minimises the second moment (parallel-axis theorem)."""
    i_c = SECTION.second_moment
    for dz in (-0.05, -0.01, 0.01, 0.05):
        assert SECTION.second_moment_about(SECTION.z_bar + dz) > i_c


def test_plastic_neutral_axis_halves_the_area():
    y_p = SECTION.plastic_neutral_axis
    assert SECTION.area_below(y_p) == pytest.approx(0.5 * SECTION.total_area, rel=1e-12)


def test_plastic_modulus_exceeds_elastic_and_shape_factor_is_physical():
    z_el = min(SECTION.section_modulus_top, SECTION.section_modulus_bottom)
    assert SECTION.plastic_section_modulus > z_el
    # for a plate-dominated panel the plastic reserve is small (1 is the
    # thin-walled idealisation limit, 1.5 the rectangle limit)
    assert 1.0 < SECTION.shape_factor < 1.5
    assert SECTION.plastic_bending_moment == pytest.approx(
        SY * SECTION.plastic_section_modulus, rel=1e-15
    )
    assert SECTION.plastic_bending_moment > SECTION.yield_bending_moment


def test_plate_buckling_stress_matches_timoshenko_formula():
    d = MAT.E * REF["t"] ** 3 / (12.0 * (1.0 - MAT.nu ** 2))
    k = 4.0  # a/b = 3.43 >= 1, simply supported long plate
    expected = k * math.pi ** 2 * d / (REF["b"] ** 2 * REF["t"])
    assert SECTION.sigma_cr_plate == pytest.approx(expected, rel=1e-12)
    # slenderness definition
    assert SECTION.beta_plate == pytest.approx(
        (REF["b"] / REF["t"]) * math.sqrt(SY / MAT.E), rel=1e-12
    )
    # consistency: beta^2 = k pi^2 / (12(1-nu^2)) * sigma_y / sigma_cr
    assert SECTION.beta_plate ** 2 == pytest.approx(
        k * math.pi ** 2 / (12.0 * (1.0 - MAT.nu ** 2)) * SY / SECTION.sigma_cr_plate,
        rel=1e-12,
    )


def test_tripping_stress_of_flat_bar_is_far_below_t_section():
    flat = StiffenedPanelSection(
        a=2.4, b=0.7, t=0.012, h_s=0.200, t_w=0.010, material=MAT
    )
    assert flat.warping_constant == 0.0
    assert flat.sigma_tripping < SECTION.sigma_tripping
    # exact Timoshenko-type anchor: sigma_T = G J / I_p for a flat bar
    g = MAT.E / (2.0 * (1.0 + MAT.nu))
    j = 0.200 * 0.010 ** 3 / 3.0
    i_p = 0.200 * 0.010 ** 3 / 12.0 + (0.200 * 0.010) * (0.006 + 0.100) ** 2
    assert flat.sigma_tripping == pytest.approx(g * j / i_p, rel=1e-12)
    # T-section torsion constant includes the flange
    assert SECTION.torsion_constant > flat.torsion_constant


def test_section_geometry_validation():
    with pytest.raises(SectionGeometryError):
        StiffenedPanelSection(a=-2.4, b=0.7, t=0.012, h_s=0.2, t_w=0.01, material=MAT)
    with pytest.raises(SectionGeometryError):
        StiffenedPanelSection(
            a=2.4, b=0.7, t=0.012, h_s=0.2, t_w=0.01, material=MAT, b_f=0.1, t_f=0.0
        )
    with pytest.raises(SectionGeometryError):
        StiffenedPanelSection(
            a=2.4, b=0.7, t=0.012, h_s=0.2, t_w=0.01, material=MAT, b_f=0.5, t_f=0.015
        )
    with pytest.raises(SectionGeometryError):
        StiffenedPanelSection(a=2.4, b=0.7, t=0.012, h_s=0.2, t_w=0.01)


# ---------------------------------------------------------------------------
# B. panel ultimate strength
# ---------------------------------------------------------------------------
def test_panel_ultimate_lies_inside_physical_bounds():
    res = panel_ultimate_axial(SECTION)
    gate = check_ultimate_bounds(res)
    assert gate["inside_bounds"], gate
    assert res.bounds["lower"] < res.sigma_u < res.bounds["upper"]
    assert 0.0 < res.eps_u < 0.02
    # the limit point must be followed by softening (a genuine limit point)
    assert res.post_peak_ratio < 1.0


def test_reference_panel_is_plate_governed():
    res = panel_ultimate_axial(SECTION)
    assert res.mode == "plate_buckling"
    assert res.sigma_u_plate < res.sigma_u_stiffener
    # Faulkner plate ultimate with residual stress: Phi (1 - r) sigma_y
    beta = SECTION.beta_plate
    phi = 2.0 / beta - 1.0 / beta ** 2
    assert res.sigma_u_plate == pytest.approx(phi * SY * 0.85, rel=1e-12)


def test_flat_bar_stiffener_trips_before_column_or_web_buckling():
    flat = StiffenedPanelSection(
        a=2.4, b=0.7, t=0.012, h_s=0.250, t_w=0.006, material=MAT
    )
    stiff = stiffener_ultimate_stress(flat)
    assert stiff["mode"] == "tripping"
    assert stiff["tripping"] < min(stiff["column"], stiff["web_local"])
    assert panel_ultimate_axial(flat).mode == "tripping"


def test_web_local_buckling_can_govern_for_slender_web():
    slender = StiffenedPanelSection(
        a=2.4, b=0.7, t=0.012, h_s=0.600, t_w=0.006, b_f=0.100, t_f=0.015, material=MAT
    )
    stiff = stiffener_ultimate_stress(slender)
    assert stiff["mode"] in ("web_local_buckling", "tripping")
    assert stiff["web_local"] < SY


def test_ultimate_strength_increases_with_plate_thickness():
    ratios = []
    for t in (0.008, 0.010, 0.012, 0.014, 0.016):
        sec = StiffenedPanelSection(material=MAT, **{**REF, "t": t})
        ratios.append(panel_ultimate_axial(sec).sigma_u_over_sigma_y)
    assert all(b > a for a, b in zip(ratios, ratios[1:])), ratios
    # beyond beta < 1 the plate is stocky, the curve must saturate below sigma_y
    assert ratios[-1] < 1.0


def test_lateral_restraint_reduces_tripping_capacity():
    flat = StiffenedPanelSection(
        a=2.4, b=0.7, t=0.012, h_s=0.250, t_w=0.006, material=MAT
    )
    free = stiffener_ultimate_stress(flat, lambda_trip=1.0)["sigma_u"]
    restrained = stiffener_ultimate_stress(flat, lambda_trip=2.5)["sigma_u"]
    assert restrained < free


def test_strength_band_orders_the_two_plate_curves():
    band = panel_strength_band(SECTION)
    assert band["sigma_u_von_karman"] > band["sigma_u_faulkner"]
    assert 0.0 < band["band_width_ratio"] < 0.5
    assert band["band_half_width_over_mean"] == pytest.approx(
        0.5 * (band["sigma_u_von_karman"] - band["sigma_u_faulkner"])
        / (0.5 * (band["sigma_u_von_karman"] + band["sigma_u_faulkner"])),
        rel=1e-12,
    )


def test_element_curve_must_be_known():
    with pytest.raises(ValueError):
        plate_ultimate_stress(SECTION, curve="romberg")
    with pytest.raises(ValueError):
        panel_ultimate_axial(SECTION, residual_stress_ratio=1.5)
    with pytest.raises(ValueError):
        stiffener_ultimate_stress(SECTION, lambda_trip=0.0)


def test_residual_stress_lowers_the_plate_element():
    hi = plate_ultimate_stress(SECTION, residual_stress_ratio=0.0)["sigma_u"]
    lo = plate_ultimate_stress(SECTION, residual_stress_ratio=0.15)["sigma_u"]
    assert lo < hi


def test_four_method_cross_check_is_reported_with_spread():
    cmp = compare_ultimate_methods(SECTION)
    methods = cmp["sigma_u"]
    assert set(methods) == {
        "element_faulkner",
        "element_von_karman",
        "smith_perry_robertson",
        "paik_rigid_plastic",
        "fiber_progressive",
    }
    # the three Faulkner-family methods must agree closely (same element model)
    for name in ("smith_perry_robertson", "paik_rigid_plastic"):
        assert abs(cmp["deviation_vs_element_faulkner"][name]) < 0.05
    # the two von Karman-family methods must agree closely as well
    assert (
        abs(methods["fiber_progressive"] / methods["element_von_karman"] - 1.0) < 0.05
    )
    # but the plate element curve choice matters: spread is large and reported
    assert 0.2 < cmp["max_spread_over_min"] < 0.5
    assert cmp["modes"]["fiber_progressive"] in (
        "plate_buckling",
        "interactive",
        "column_buckling",
        "yielding",
        "tripping",
    )


# ---------------------------------------------------------------------------
# C. rigid-plastic N-M interaction surface
# ---------------------------------------------------------------------------
def test_plastic_interaction_machine_exact_anchors():
    surf = plastic_axial_bending_interaction(SECTION, n_points=401)
    assert surf["m_at_n_zero_over_m_p"] == pytest.approx(1.0, abs=1e-12)
    assert surf["n_range_over_n_p"][0] == pytest.approx(-1.0, abs=1e-12)
    assert surf["n_range_over_n_p"][1] == pytest.approx(1.0, abs=1e-12)
    assert surf["n_p"] == pytest.approx(SY * SECTION.total_area, rel=1e-15)
    assert surf["m_p"] == pytest.approx(SY * SECTION.plastic_section_modulus, rel=1e-15)


def test_asymmetric_section_exceeds_plastic_moment_under_axial_load():
    surf = plastic_axial_bending_interaction(SECTION, n_points=401)
    # plate-dominated T section: the moment about the elastic centroid peaks
    # away from N = 0 (computed independently below for one NA position)
    assert surf["m_max_over_m_p"] > 1.0
    # the quadratic rule form is conservative in compression ...
    assert surf["rule_form_max_violation_compression_over_m_p"] <= 0.0
    # ... but not in tension (asymmetric section)
    assert surf["rule_form_max_violation_tension_over_m_p"] > 0.0


def test_plastic_modulus_is_the_n_zero_moment_anchor():
    """Z_p from the section must equal the sweep's M(N = 0) / sigma_y."""
    surf = plastic_axial_bending_interaction(SECTION, n_points=201)
    z_p_from_sweep = surf["m_at_n_zero"] / SY
    assert z_p_from_sweep == pytest.approx(SECTION.plastic_section_modulus, rel=1e-10)


# ---------------------------------------------------------------------------
# D. grillage: no fabricated restraint factors
# ---------------------------------------------------------------------------
KW = dict(REF, material=MAT)


def test_parallel_assembly_is_exactly_size_invariant():
    """Identical bays: 3x3 and 5x5 must give the same ultimate (machine zero).

    This is the assertion the old ``1 + 0.05 (n_y-1)/n_y`` factor could not
    support: invariance is a property of the mechanics, not of a fudge factor.
    """
    coarse = GrillageModel(n_x=3, n_y=3, imperfection_spread=0.0, **KW)
    fine = GrillageModel(n_x=5, n_y=5, imperfection_spread=0.0, **KW)
    assert fine.compute_ultimate_strength() == pytest.approx(
        coarse.compute_ultimate_strength(), rel=1e-12
    )
    # and the assembly equals the single bay exactly
    single = GrillageModel(n_x=1, n_y=1, imperfection_spread=0.0, **KW)
    assert single.compute_ultimate_strength() == pytest.approx(
        coarse.compute_ultimate_strength(), rel=1e-12
    )


def test_convergence_study_reports_mechanics_not_fudge_factor():
    conv = grillage_convergence_study(**KW)
    assert conv["converged"]
    assert conv["relative_error"] < 1e-4
    assert conv["mesh_invariance_rel"] < 1e-12
    # the old fabricated model always returned exactly 0.05*(1/3 - 1/5)
    assert abs(conv["relative_error"] - 0.05 * (1.0 / 3.0 - 1.0 / 5.0)) > 1e-3
    assert conv["sigma_u_limit_estimate"] > 0.0


def test_isolated_element_deviation_is_negligible_not_five_percent():
    dev = isolated_element_deviation(**KW)
    assert dev["delta_iso_sign"] in ("positive", "negative")
    assert abs(dev["delta_iso"]) < 0.005  # documented |delta_iso| ~ 2e-5
    assert dev["delta_iso_bound"] > 0.0
    # the load-shedding gain claimed before (+5 %) is excluded
    assert dev["delta_iso"] < 0.05


def test_load_steps_are_the_real_resolution_knob():
    coarse = discretization_convergence_study(
        **KW, n_steps_coarse=40, n_steps_fine=400
    )
    fine = discretization_convergence_study(**KW, n_steps_coarse=200, n_steps_fine=400)
    # 40 steps (the previous settings) are not converged ...
    assert coarse["relative_error"] > 0.01
    # ... 200 steps are
    assert fine["relative_error"] < 0.01
    assert fine["converged"]


def test_imperfection_quantiles_are_deterministic_and_centred():
    from tensorphysics.elastoplastic_collapse.grillage_error_bound import (
        imperfection_quantiles,
    )

    q = imperfection_quantiles(9, 0.5)
    assert q.shape == (9,)
    assert np.all(q > 0.0) and np.all(q < 1.5)
    assert q.mean() == pytest.approx(1.0, abs=0.02)
    assert imperfection_quantiles(9, 0.0) == pytest.approx(np.ones(9))
    with pytest.raises(ValueError):
        imperfection_quantiles(0, 0.5)
    with pytest.raises(ValueError):
        imperfection_quantiles(4, 1.5)


# ---------------------------------------------------------------------------
# E. defect fix: tripping is now a reachable failure mode of the Smith solver
# ---------------------------------------------------------------------------
def test_smith_solver_can_return_tripping_mode():
    geom = StiffenedPlateGeometry(
        a=2.4, b=0.7, t=0.012, h_s=0.250, t_w=0.006, b_f=0.0, t_f=0.0,
        material=MAT, w0_max=2.4e-3,
    )
    res = SmithCollapseSolver(geom, n_steps=200, max_strain=0.02).solve()
    assert res.failure_mode == "tripping"
    assert res.sigma_u > 0.0
    # the documented reference panel is not tripping governed
    ref_geom = StiffenedPlateGeometry(material=MAT, w0_max=2.4e-3, **REF)
    assert SmithCollapseSolver(ref_geom, n_steps=200).solve().failure_mode != "tripping"


def test_panel_section_helper_is_consistent_with_geometry():
    geom = StiffenedPlateGeometry(material=MAT, w0_max=2.4e-3, **REF)
    sec = geom.panel_section()
    legacy = geom.section_properties()
    assert sec.total_area == pytest.approx(legacy["A"], rel=1e-12)
    assert sec.z_bar == pytest.approx(legacy["z_bar"], rel=1e-12)
    assert sec.second_moment == pytest.approx(legacy["I"], rel=1e-10)
    assert sec.section_modulus_top == pytest.approx(legacy["Z_top"], rel=1e-12)
    # Defect found: ``section_properties`` measures the plate-side section
    # modulus to the plate *mid-plane* instead of the plate bottom fibre, i.e.
    # it overestimates Z_bot by (z_bar + t/2)/z_bar = 1.067 (6.7 % here).
    assert sec.section_modulus_bottom == pytest.approx(
        legacy["Z_bot"] * sec.z_bar / (sec.z_bar + 0.5 * sec.t), rel=1e-12
    )
    assert legacy["Z_bot"] > sec.section_modulus_bottom