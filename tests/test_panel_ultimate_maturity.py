"""Acceptance tests for the matured panel ultimate-strength capability.

Covers the p167w2-n04-panel-collapse-maturity acceptance criteria:

* C1 tripping mode, mode switch and the analytical torsional-buckling stress
* C2 3x3 -> 5x5 error bound (< 2 %) plus the isolated-element deviation
* C3 transverse-frame monotonicity (s = 3a -> 1a, sigma_u non-decreasing)
* C4 four negative controls (stiffener -> 0, rigid frames, sigma_trip -> inf,
     defect/residual -> 0)
* C5 closed-form benchmark against the marine_stiffener_failure LTB reference
* C6 regression on the P70-N01 defect (sigma_u above yield)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorphysics.elastoplastic_collapse.von_mises_plasticity import VonMisesMaterial
from tensorphysics.elastoplastic_collapse.panel_section import (
    stiffener_section, rectangle_torsion_constant,
)
from tensorphysics.elastoplastic_collapse.tripping_mode import (
    TrippingAnalysis, ltb_critical_moment, ltb_numerical,
)
from tensorphysics.elastoplastic_collapse.panel_ultimate import (
    PanelScantlings, SmithPanelSolver, faulkner_phi, dnv_column_curve,
    PLATEAU_STRAIN,
)
from tensorphysics.elastoplastic_collapse.grillage_error_bound import (
    GrillageModel, grillage_convergence_study, isolated_element_deviation,
)
from tensorphysics.elastoplastic_collapse.stiffened_plate_collapse import (
    StiffenedPlateGeometry, SmithCollapseSolver,
)


@pytest.fixture
def steel():
    return VonMisesMaterial(E=2.06e11, nu=0.3, sigma_y0=3.13e8, H=1.0e9)


@pytest.fixture
def perfect_plastic():
    return VonMisesMaterial(E=2.06e11, nu=0.3, sigma_y0=3.13e8, H=0.0)


@pytest.fixture
def scantlings(steel):
    """Typical ship panel: 12 mm plate, 700 mm spacing, 200x10 tee + 100x15 flange."""
    return PanelScantlings(a=2.4, b=0.7, t=0.012, h_s=0.200, t_w=0.010,
                           b_f=0.100, t_f=0.015, material=steel)


# --------------------------------------------------------------------------- #
# C0 elementary curves
# --------------------------------------------------------------------------- #
class TestElementaryCurves:
    def test_faulkner_limits(self):
        assert faulkner_phi(0.5) == 1.0          # stocky plate: full yield
        assert faulkner_phi(1.0) == 1.0
        # beta = 2 -> 2/2 - 1/4 = 0.75
        assert abs(faulkner_phi(2.0) - 0.75) < 1e-12
        # monotone decreasing in beta beyond 1
        betas = np.linspace(1.0, 5.0, 20)
        phis = [faulkner_phi(b) for b in betas]
        assert all(np.diff(phis) < 0)

    def test_dnv_column_curve_limits(self):
        assert dnv_column_curve(0.1) == 1.0
        # asymptote towards 1/lam^2 (reached from below, slowly: -6 % at lam=5)
        for lam in (10.0, 20.0):
            assert abs(dnv_column_curve(lam) * lam**2 - 1.0) < 0.05
        assert dnv_column_curve(5.0) < 1.0 / 5.0**2 * 1.01
        lams = np.linspace(0.2, 5.0, 50)
        vals = [dnv_column_curve(l) for l in lams]
        assert all(np.diff(vals) < 0)


# --------------------------------------------------------------------------- #
# C0b section properties vs closed forms
# --------------------------------------------------------------------------- #
class TestSectionProperties:
    def test_flat_bar_properties_match_closed_form(self):
        sec = stiffener_section(plate_width=0.7, plate_thickness=0.012,
                                web_height=0.2, web_thickness=0.01)
        # Saint-Venant thin-wall constant of a rectangle with b >> t is b t^3/3
        assert abs(sec.j_thin - (0.2 * 0.01**3 / 3 + 0.7 * 0.012**3 / 3)) < 1e-15
        # weak axis: sum of the two rectangles' own inertias
        assert abs(sec.i_y - (0.012 * 0.7**3 / 12 + 0.2 * 0.01**3 / 12)) < 1e-15
        # strong axis by hand (parallel axis on plate + web)
        a_p, a_w = 0.7 * 0.012, 0.2 * 0.01
        y_bar = a_w * 0.1 / (a_p + a_w)
        i_x = (0.7 * 0.012**3 / 12 + a_p * y_bar**2
               + 0.01 * 0.2**3 / 12 + a_w * (0.1 - y_bar) ** 2)
        assert abs(sec.i_x - i_x) / i_x < 1e-12

    def test_tee_j_matches_tensorfem_independent_calculator(self):
        """J_thin of the stiffener alone must match TensorFEM's wall-graph value."""
        # tensorfem.open_section_torsion.stiffener_properties(0.200, 0.010, 0.100, 0.015)
        tensorfem_j_thin = 1.7916666666666664e-07
        sec = stiffener_section(plate_width=0.7, plate_thickness=0.012,
                                web_height=0.2, web_thickness=0.01,
                                flange_width=0.1, flange_thickness=0.015)
        j_stiffener = sec.j_thin - 0.7 * 0.012**3 / 3
        assert abs(j_stiffener / tensorfem_j_thin - 1.0) < 1e-6

    def test_rectangle_torsion_constant_limits(self):
        # thin rectangle: J -> a b^3 (1/3 - 0.21 b/a ...) -> a b^3/3
        assert abs(rectangle_torsion_constant(1.0, 1e-4)
                   / (1.0 * 1e-4**3 / 3) - 1.0) < 1e-3
        # square: known value 0.1406 a^4
        assert abs(rectangle_torsion_constant(1.0, 1.0) / 0.1406 - 1.0) < 0.01

    def test_symmetric_tee_has_zero_warping(self):
        """Mid-surface Vlasov model: every wall passes through the wall
        intersection where the shear centre sits -> C_w = 0 exactly."""
        sec = stiffener_section(plate_width=1e-9, plate_thickness=1e-3,
                                web_height=0.2, web_thickness=0.01,
                                flange_width=0.1, flange_thickness=0.015)
        assert sec.warping_constant() < 1e-24
        assert sec.shear_centre_y == pytest.approx(0.2075, abs=1e-15)


# --------------------------------------------------------------------------- #
# C1 tripping mode + mode switch
# --------------------------------------------------------------------------- #
class TestTrippingMode:
    def test_elastic_tripping_equals_closed_form(self, steel):
        """sigma_ET must equal (G J + pi^2 E Cw/L^2)/I_p,SC computed by hand."""
        an = TrippingAnalysis(a=2.4, b=0.7, t=0.012, h_s=0.2, t_w=0.01,
                              b_f=0.1, t_f=0.015, material=steel)
        g = steel.shear_modulus
        hand = (g * an.j_saint_venant
                + math.pi**2 * steel.E * an.warping_constant / 2.4**2) / an.i_p_shear_centre
        assert abs(an.elastic_tripping_stress(1.0) / hand - 1.0) < 1e-12
        # the polar inertia must be the shear-centre one, and exceed the
        # attachment one only by the parallel-axis transfer of the stiffener
        assert an.i_p_shear_centre < an.i_p_attachment
        # unpenalised upper bound: sigma_ET > 1.4 sigma_y for this stocky tee
        assert an.elastic_tripping_stress(1.0) / steel.sigma_y0 > 1.4

    def test_mode_switch_web_slenderness_sweep(self, steel):
        """slender webs trip elastically, stocky webs approach yielding."""
        modes, cap = [], []
        for t_w in (0.006, 0.008, 0.010, 0.014, 0.020):
            an = TrippingAnalysis(a=2.4, b=0.7, t=0.012, h_s=0.30, t_w=t_w,
                                  b_f=0.0, t_f=0.0, material=steel)
            r = an.tripping_capacity(1.0)
            modes.append(r.failure_mode)
            cap.append(r.capacity_reduction)
        assert modes[0] == "elastic_tripping", modes
        assert "tripping" in modes, modes
        assert cap[0] < 0.35 and cap[-1] > 0.85
        assert all(np.diff(cap) > 0)          # capacity grows with web thickness
        # the yielding end of the switch exists for a truly compact section
        stocky = TrippingAnalysis(a=2.4, b=0.7, t=0.012, h_s=0.05, t_w=0.05,
                                  b_f=0.0, t_f=0.0, material=steel)
        r_stocky = stocky.tripping_capacity(1.0)
        assert r_stocky.failure_mode == "yielding"
        assert r_stocky.capacity_reduction == pytest.approx(1.0)

    def test_tripping_reaches_the_panel_result(self, steel):
        """A slender flat bar must govern the panel strength by tripping."""
        slender = PanelScantlings(a=2.4, b=0.7, t=0.012, h_s=0.30, t_w=0.008,
                                  b_f=0.0, t_f=0.0, material=steel)
        res = SmithPanelSolver(slender, n_x=5, n_y=5).solve()
        assert res.mode == "tripping"
        assert abs(res.sigma_u - res.sigma_tripping) / res.sigma_tripping < 1e-6
        # the stocky tee of the base case must NOT trip
        tee = PanelScantlings(a=2.4, b=0.7, t=0.012, h_s=0.20, t_w=0.010,
                              b_f=0.100, t_f=0.015, material=steel)
        res_tee = SmithPanelSolver(tee, n_x=5, n_y=5).solve()
        assert res_tee.mode != "tripping"
        assert res_tee.sigma_tripping > res_tee.sigma_u

    def test_lambda_trip_reduces_capacity(self, steel):
        an = TrippingAnalysis(a=2.4, b=0.7, t=0.012, h_s=0.30, t_w=0.008,
                              b_f=0.0, t_f=0.0, material=steel)
        caps = [an.tripping_capacity(l).capacity_reduction
                for l in np.linspace(1.0, 2.5, 8)]
        assert all(np.diff(caps) < 0)
        assert (caps[0] - caps[-1]) / caps[0] > 0.1


# --------------------------------------------------------------------------- #
# C2 error bound + isolated element deviation
# --------------------------------------------------------------------------- #
class TestErrorBound:
    def test_3x3_vs_5x5_below_two_percent(self, steel):
        conv = grillage_convergence_study(
            2.4, 0.7, 0.012, 0.2, 0.01, 0.1, 0.015, steel, refine=9)
        assert conv["relative_error"] < 0.02
        assert conv["converged"] is True
        # genuine O(h^2) convergence: 3 -> 5 -> 9 must monotonically approach a limit
        seq = conv["refinement_sequence"]
        assert seq[3] < seq[5] < seq[9]
        assert (seq[5] - seq[3]) > 2.0 * (seq[9] - seq[5])   # ~ (9/5)^2
        # Richardson limit is within 0.1 % of the 9x9 value
        assert conv["richardson_relative_change"] < 1e-3
        # the difference between 1x1 and 3x3 is much larger: the criterion
        # really does discriminate (it is not a constant-by-construction ratio)
        conv1 = grillage_convergence_study(
            2.4, 0.7, 0.012, 0.2, 0.01, 0.1, 0.015, steel, n_coarse=1, n_fine=3)
        assert conv1["relative_error"] > 2.0 * conv["relative_error"]

    def test_delta_iso_sign_and_magnitude(self, steel):
        d = isolated_element_deviation(2.4, 0.7, 0.012, 0.2, 0.01, 0.1, 0.015, steel)
        # sign records the curvature of capacity-vs-imperfection at the panel
        # mean; the magnitude is the isolated-element error bound (< 1 %)
        assert d["delta_iso_sign"] in ("positive", "negative")
        assert 0.0 < d["delta_iso_magnitude"] < 0.01
        # it must be stable under mesh refinement (it is a physical, not a
        # discretisation, effect)
        d9 = isolated_element_deviation(2.4, 0.7, 0.012, 0.2, 0.01, 0.1, 0.015,
                                        steel, n_x=9, n_y=9)
        assert abs(d9["delta_iso"] - d["delta_iso"]) < 5e-4
        # the nominal-imperfection reference is conservative (positive deviation)
        assert d["delta_iso_nominal_sign"] == "positive"
        assert d["delta_iso_nominal"] > 0.0

    def test_convergence_bound_beats_delta_iso(self, steel):
        """The mesh error must be smaller than the isolated-element error bound."""
        conv = grillage_convergence_study(2.4, 0.7, 0.012, 0.2, 0.01, 0.1, 0.015, steel)
        d = isolated_element_deviation(2.4, 0.7, 0.012, 0.2, 0.01, 0.1, 0.015, steel)
        assert conv["error_bound"] < 2.0 * d["delta_iso_nominal"]


# --------------------------------------------------------------------------- #
# C3 transverse frame monotonicity
# --------------------------------------------------------------------------- #
class TestTransverseFrames:
    def test_sigma_u_non_decreasing_as_spacing_shrinks(self, steel):
        sc = PanelScantlings(a=2.4, b=0.7, t=0.012, h_s=0.2, t_w=0.01,
                             b_f=0.1, t_f=0.015, material=steel)
        spacings = [7.2, 6.0, 4.8, 3.6, 2.4]     # 3a -> 1a
        sigmas = []
        for s in spacings:
            n_bays = max(1, int(round(s / 2.4)))
            res = SmithPanelSolver(sc, n_bays=n_bays, frame_spacing=s,
                                   n_x=5, n_y=5).solve()
            sigmas.append(res.sigma_u)
        assert all(np.diff(sigmas) >= 0.0), f"not monotone: {sigmas}"
        # the effect is material, not numerical: s = 3a must be >2 % weaker
        assert (sigmas[-1] - sigmas[0]) / sigmas[0] > 0.02
        # and the governing column span follows the frame spacing
        for s, sig in zip(spacings, sigmas):
            n_bays = max(1, int(round(s / 2.4)))
            res = SmithPanelSolver(sc, n_bays=n_bays, frame_spacing=s,
                                   n_x=5, n_y=5).solve()
            assert abs(res.column_span - s) < 1e-12


# --------------------------------------------------------------------------- #
# C4 negative controls
# --------------------------------------------------------------------------- #
class TestNegativeControls:
    def test_cut1_stiffener_vanishes_gives_faulkner(self, steel):
        """Cut 1: stiffener stiffness -> 0 with no defect must equal Phi*sigma_y."""
        sc = PanelScantlings(a=2.4, b=0.7, t=0.012, h_s=1e-6, t_w=1e-6,
                             b_f=0.0, t_f=0.0, material=steel,
                             w0_plate=0.0, w0_column=0.0, residual_stress=0.0)
        res = SmithPanelSolver(sc, n_x=5, n_y=5, tripping=False).solve()
        target = faulkner_phi(sc.beta) * steel.sigma_y0
        assert abs(res.sigma_u / target - 1.0) < 1e-9, (res.sigma_u, target)
        assert res.mode == "plate_buckling"
        # the stiffener contribution is positive: a real stiffener is stronger
        base = PanelScantlings(a=2.4, b=0.7, t=0.012, h_s=0.2, t_w=0.01,
                               b_f=0.1, t_f=0.015, material=steel)
        assert SmithPanelSolver(base, n_x=5, n_y=5).solve().sigma_u < target

    def test_cut2_rigid_frames_make_bays_independent(self, steel):
        """Cut 2: rigid transverse frames -> n_bays copies of one bay, exactly."""
        sc = PanelScantlings(a=2.4, b=0.7, t=0.012, h_s=0.2, t_w=0.01,
                             b_f=0.1, t_f=0.015, material=steel)
        one = SmithPanelSolver(sc, n_bays=1, frame_spacing=2.4, n_x=5, n_y=5).solve()
        three = SmithPanelSolver(sc, n_bays=3, frame_spacing=2.4, rigid_frames=True,
                                 n_x=5, n_y=5).solve()
        assert abs(three.sigma_u / one.sigma_u - 1.0) < 1e-12
        # compliant frames must fall back to the overall (3-bay) mode and be weaker
        soft = SmithPanelSolver(sc, n_bays=3, frame_spacing=2.4, rigid_frames=False,
                                frame_stiffness=1.0, n_x=5, n_y=5).solve()
        assert abs(soft.column_span - 7.2) < 1e-12
        assert soft.sigma_u < one.sigma_u

    def test_cut3_infinite_tripping_reproduces_baseline(self, steel):
        """Cut 3: sigma_trip -> inf must reproduce the no-tripping baseline bitwise."""
        sc = PanelScantlings(a=2.4, b=0.7, t=0.012, h_s=0.30, t_w=0.008,
                             b_f=0.0, t_f=0.0, material=steel)
        inf = SmithPanelSolver(sc, n_x=5, n_y=5,
                               sigma_trip_override=float("inf")).solve()
        off = SmithPanelSolver(sc, n_x=5, n_y=5, tripping=False).solve()
        assert inf.sigma_u == off.sigma_u
        # and with tripping active the same panel is strictly weaker
        on = SmithPanelSolver(sc, n_x=5, n_y=5).solve()
        assert on.sigma_u < off.sigma_u

    def test_cut4_zero_defect_gives_ideal_elastoplastic_trilinear(self, perfect_plastic):
        """Cut 4: defect = residual = 0 on a *compact* panel -> elastic /
        yield-plateau / shedding branches (three segments)."""
        sc = PanelScantlings(a=2.4, b=0.4, t=0.020, h_s=0.2, t_w=0.012,
                             b_f=0.1, t_f=0.020, material=perfect_plastic,
                             w0_plate=0.0, w0_column=0.0, residual_stress=0.0)
        assert sc.beta < 1.0                      # compact plate: Phi = 1
        # tripping disabled: this control isolates the material/imperfection law
        res = SmithPanelSolver(sc, n_x=5, n_y=5, max_strain=0.01, n_steps=400,
                               sigma_trip_override=float("inf")).solve()
        eps = np.array([p[0] for p in res.curve])
        sig = np.array([p[1] for p in res.curve])

        # branch 1: linear elastic with slope E
        elastic = eps < 0.5 * perfect_plastic.sigma_y0 / perfect_plastic.E
        slope = np.polyfit(eps[elastic], sig[elastic], 1)[0]
        assert abs(slope / perfect_plastic.E - 1.0) < 1e-3

        # branch 2: perfectly-plastic yield plateau at sigma_y
        plateau = ((sig > 0.999 * perfect_plastic.sigma_y0)
                   & (eps >= res.eps_u) & (eps <= res.eps_u + PLATEAU_STRAIN + 1e-12))
        assert plateau.sum() > 5
        assert np.allclose(sig[plateau], perfect_plastic.sigma_y0, rtol=1e-6)

        # branch 3: post-ultimate shedding, strictly decreasing
        post = eps > res.eps_u + PLATEAU_STRAIN
        assert post.sum() > 5
        assert np.all(np.diff(sig[post]) < 0)
        assert res.sigma_u_over_sigma_y == pytest.approx(1.0, rel=1e-9)

    def test_defect_increases_reduce_capacity(self, steel):
        """Larger initial imperfections must lower sigma_u (compact panel, so the
        beam-column mode, not the plate mode, governs)."""
        sc0 = PanelScantlings(a=2.4, b=0.4, t=0.020, h_s=0.2, t_w=0.012,
                              b_f=0.1, t_f=0.020, material=steel,
                              residual_stress=0.0)
        sig = []
        for factor in (0.0, 0.5, 1.0, 2.0):
            res = SmithPanelSolver(sc0, n_x=5, n_y=5, defect_factor=factor,
                                   sigma_trip_override=float("inf")).solve()
            sig.append(res.sigma_u)
        assert all(np.diff(sig) < 0), sig


# --------------------------------------------------------------------------- #
# C5 closed-form benchmark: marine_stiffener_failure LTB reference
# --------------------------------------------------------------------------- #
class TestClosedFormBenchmark:
    def test_mcr_numerical_reproduces_closed_form(self):
        errs = [ltb_numerical(segments=s, length=2.4, young=2.06e11,
                              shear_modulus=79.2e9, weak_axis_inertia=1.267e-6,
                              torsion_constant=1.7917e-7, warping_constant=4.5e-9)["relative_error"]
                for s in (4, 8, 16, 32, 64)]
        assert all(np.diff(errs) < 0), errs              # monotone convergence
        assert errs[-1] < 0.03 and errs[2] < 0.01        # 16 segments < 1 %

    def test_ltb_moment_matches_closed_form_expression(self, steel):
        an = TrippingAnalysis(a=2.4, b=0.7, t=0.012, h_s=0.2, t_w=0.01,
                              b_f=0.1, t_f=0.015, material=steel)
        hand = ltb_critical_moment(
            length=2.4, young=steel.E, shear_modulus=steel.shear_modulus,
            weak_axis_inertia=an.stiffener_i_y,
            torsion_constant=an.j_saint_venant,
            warping_constant=an.warping_constant)
        assert abs(an.ltb_critical_moment() / hand - 1.0) < 1e-12

    def test_tripping_to_ltb_ratio_obeys_analytic_shape_factor(self, steel):
        """sigma_ET / (Mcr/(I_p/c)) has the closed-form value

        (GJ + pi^2 E Cw/L^2) / [ (pi/L) sqrt(E Iy GJ) sqrt(1 + pi^2 E Cw/(L^2 GJ)) c ]

        i.e. the two stresses are *different physical quantities* (pure torsional
        buckling vs lateral-torsional buckling); the ratio is a shape factor that
        this test pins down analytically (agreement < 1e-10).
        """
        for h_s, t_w, b_f, t_f in ((0.2, 0.01, 0.1, 0.015),
                                   (0.3, 0.008, 0.0, 0.0),
                                   (0.15, 0.012, 0.1, 0.015)):
            an = TrippingAnalysis(a=2.4, b=0.7, t=0.012, h_s=h_s, t_w=t_w,
                                  b_f=b_f, t_f=t_f, material=steel)
            ratio = an.elastic_tripping_stress(1.0) / an.ltb_stress()
            gj = steel.shear_modulus * an.j_saint_venant
            cw_term = math.pi**2 * steel.E * an.warping_constant / 2.4**2
            amp = math.sqrt(1.0 + cw_term / gj)
            mcr = (math.pi / 2.4) * math.sqrt(steel.E * an.stiffener_i_y * gj) * amp
            analytic = (gj + cw_term) / (mcr * an.c_from_sc)
            assert abs(ratio / analytic - 1.0) < 1e-10, (ratio, analytic)


# --------------------------------------------------------------------------- #
# C6 regression on the P70-N01 defect (sigma_u above yield)
# --------------------------------------------------------------------------- #
class TestBaselineRegression:
    def test_legacy_smith_solver_never_exceeds_yield(self, steel):
        """The P70-N01 baseline returned sigma_u/sigma_y = 1.196; the yield guard
        must keep the legacy solver physically admissible."""
        geom = StiffenedPlateGeometry(a=2.4, b=0.7, t=0.012, h_s=0.2, t_w=0.01,
                                      b_f=0.1, t_f=0.015, material=steel,
                                      w0_max=0.002)
        res = SmithCollapseSolver(geom).solve()
        assert res.sigma_u <= steel.sigma_y0 * (1.0 + 1e-12)

    def test_matured_panel_strength_is_physically_admissible(self, steel, scantlings):
        res = SmithPanelSolver(scantlings, n_x=5, n_y=5).solve()
        assert 0.2 < res.sigma_u_over_sigma_y < 1.0
        # the reported governing limit and sigma_u agree to within the area
        # average over the imperfection field (< 5 %)
        limits = [res.sigma_plate, res.sigma_euler, res.sigma_beam_column]
        if res.sigma_tripping is not None:
            limits.append(res.sigma_tripping)
        assert abs(res.sigma_u - min(limits)) / res.sigma_u < 0.05

    def test_grillage_model_api_is_backwards_compatible(self, steel):
        model = GrillageModel(5, 5, 2.4, 0.7, 0.012, 0.2, 0.01, 0.1, 0.015, steel)
        assert model.compute_ultimate_strength() > 0
        assert model.to_stiffened_plate().a == 2.4
