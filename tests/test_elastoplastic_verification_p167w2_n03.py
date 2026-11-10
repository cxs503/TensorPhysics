"""Verification surface for campaign node p167w2-n03-fem-elastic-plastic.

Every test below is an *anchor* check with an independent closed-form or
published reference, so that the elastoplastic progressive-collapse capability
is graded by numbers rather than by "it runs".

Anchors
-------
1. von Mises + linear isotropic hardening, uniaxial exact solution
   σ = σ_y0 + H ε̄_p, ε = σ/E + ε̄_p — 3D and plane stress.
2. Consistent tangent = central finite difference of the return mapping.
3. Plate elastic buckling σ_cr = k π²E/(12(1-ν²))(t/b)² with k = 4.
4. Plate element ultimate = von Karman sqrt(σ_cr σ_y).
5. Full-section yield: stocky panel σ_u → σ_y (and never above).
6. Slender panels: σ_u decreases monotonically with β, limit point + softening.
7. Ice-belt pressure-area law exponent -0.3 (numerical power-law fit).
8. Ice-belt strip plastic collapse p_u = 8 (1+f) M_p / L², M_p = σ_y t²/4.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorphysics.elastoplastic_collapse import (
    StiffenedPlateGeometry,
    SmithCollapseSolver,
    VonMisesMaterial,
    return_mapping,
)
from tensorphysics.elastoplastic_collapse.progressive_collapse import (
    ProgressiveCollapseSolver,
    panel_convergence_study,
    plate_load_shortening_stress,
    uniaxial_elastoplastic_stress,
)
from tensorphysics.elastoplastic_collapse.von_mises_plasticity import (
    equivalent_stress,
    yield_function,
)
from tensorphysics.elastoplastic_collapse.iacs_icebelt_pressure import (
    IACSIceBeltPressure,
    icebelt_design_ratio,
    icebelt_panel_plastic_capacity,
    icebelt_stiffened_panel_capacity,
    plastic_section_modulus,
    pressure_area_law_icebelt,
    verify_pressure_area_exponent,
)

E = 2.06e11
NU = 0.3
SY0 = 3.13e8


def steel(H: float = 1.0e9, sigma_y0: float = SY0) -> VonMisesMaterial:
    return VonMisesMaterial(E=E, nu=NU, sigma_y0=sigma_y0, H=H)


def panel(material: VonMisesMaterial, **kw) -> StiffenedPlateGeometry:
    params = dict(a=2.4, b=0.7, t=0.012, h_s=0.200, t_w=0.010,
                  b_f=0.100, t_f=0.015, material=material, w0_max=0.002)
    params.update(kw)
    return StiffenedPlateGeometry(**params)


def uniaxial_exact(eps: float, material: VonMisesMaterial):
    """Closed-form 1D elastoplastic solution (σ, ε̄_p)."""
    eps_y = material.sigma_y0 / material.E
    if eps <= eps_y:
        return material.E * eps, 0.0
    eps_p = (eps - eps_y) / (1.0 + material.H / material.E)
    return material.sigma_y0 + material.H * eps_p, eps_p


# ===========================================================================
# 1. constitutive law
# ===========================================================================
class TestVonMisesVerification:
    def test_3d_uniaxial_reproduces_closed_form(self):
        mat = steel()
        worst_sigma, worst_alpha = 0.0, 0.0
        for eps in np.linspace(1e-4, 6e-3, 40):
            sigma_ex, eps_p = uniaxial_exact(eps, mat)
            lat = -NU * sigma_ex / mat.E - 0.5 * eps_p
            strain = np.array([eps, lat, lat, 0.0, 0.0, 0.0])
            sig, _C, _ep, alpha = return_mapping(
                strain, None, np.array([0.0]), mat, plane_stress=False)
            sig = np.asarray(sig).ravel()
            worst_sigma = max(worst_sigma, abs(sig[0] - sigma_ex) / sigma_ex)
            worst_alpha = max(
                worst_alpha, abs(float(np.ravel(alpha)[0]) - eps_p) / max(eps_p, 1e-9))
        assert worst_sigma < 1e-9, f"3D uniaxial stress error {worst_sigma:.2e}"
        assert worst_alpha < 1e-9, f"3D equivalent plastic strain error {worst_alpha:.2e}"

    def test_plane_stress_uniaxial_reproduces_closed_form(self):
        """Anchor 1 (3 % criterion) — the pre-fix implementation gave 14.6 %."""
        mat = steel()
        worst_sigma, worst_spurious = 0.0, 0.0
        for eps in np.linspace(1e-4, 6e-3, 40):
            sigma_ex, eps_p = uniaxial_exact(eps, mat)
            lat = -NU * sigma_ex / mat.E - 0.5 * eps_p
            strain = np.array([eps, lat, 0.0])
            sig, _C, _ep, _alpha = return_mapping(
                strain, None, np.array([0.0]), mat, plane_stress=True)
            sig = np.asarray(sig).ravel()
            worst_sigma = max(worst_sigma, abs(sig[0] - sigma_ex) / sigma_ex)
            worst_spurious = max(worst_spurious, abs(sig[1]) / sigma_ex)
        assert worst_sigma < 0.01, f"plane stress uniaxial error {worst_sigma:.3%}"
        assert worst_sigma < 3e-8, f"should be exact, got {worst_sigma:.2e}"
        assert worst_spurious < 3e-8, f"spurious σ_yy {worst_spurious:.2e}"

    def test_plane_stress_flow_normal_and_equivalent_plastic_strain(self):
        """Plane-stress flow is isochoric and parallel to the deviator."""
        mat = steel()
        strain = np.array([3.0e-3, -1.0e-3, 5.0e-4])
        sig, _C, eps_p, alpha = return_mapping(
            strain, None, np.array([0.0]), mat, plane_stress=True)
        sig = np.asarray(sig).ravel()
        eps_p = np.asarray(eps_p).ravel()
        a = float(np.ravel(alpha)[0])
        # tensor components (engineering shear -> tensor shear)
        eps_p_t = np.array([eps_p[0], eps_p[1], -(eps_p[0] + eps_p[1]), eps_p[2] / 2.0])
        # isochoric flow
        assert abs(eps_p_t[:3].sum()) < 1e-12
        # accumulated equivalent plastic strain α = sqrt(2/3 ε_p:ε_p)
        equiv = np.sqrt(2.0 / 3.0 * (eps_p_t[0]**2 + eps_p_t[1]**2
                                     + eps_p_t[2]**2 + 2.0 * eps_p_t[3]**2))
        assert abs(equiv - a) / a < 1e-9
        # flow direction parallel to the deviatoric stress
        tr = sig[0] + sig[1]
        s_dev = np.array([sig[0] - tr / 3.0, sig[1] - tr / 3.0, -tr / 3.0])
        # compare normalised vectors (in-plane normal components)
        n_flow = eps_p_t[:3] / np.linalg.norm(eps_p_t[:3])
        n_dev = s_dev / np.linalg.norm(s_dev)
        assert np.allclose(n_flow, n_dev, atol=1e-9)

    def test_flow_is_isochoric_uniaxial(self):
        mat = steel()
        eps = 4e-3
        sigma_ex, eps_p = uniaxial_exact(eps, mat)
        strain = np.array([eps, -NU * sigma_ex / mat.E - 0.5 * eps_p, 0.0])
        sig, _C, eps_p_vec, alpha = return_mapping(
            strain, None, np.array([0.0]), mat, plane_stress=True)
        eps_p_vec = np.asarray(eps_p_vec).ravel()
        a = float(np.ravel(alpha)[0])
        assert abs(eps_p_vec[0] - a) < 1e-9
        assert abs(eps_p_vec[1] + 0.5 * a) < 1e-9  # in-plane part of the flow

    @pytest.mark.parametrize("plane_stress", [True, False])
    def test_consistent_tangent_matches_finite_difference(self, plane_stress):
        """Anchor 2: baseline deviation was 17 % (3D) / 67 % (plane stress)."""
        mat = steel()
        if plane_stress:
            strain = np.array([3.2e-3, -1.0e-3, 6.0e-4])
        else:
            strain = np.array([3.2e-3, 1.0e-3, -0.5e-3, 4.0e-4, 0.0, 0.0])
        n = strain.size
        _s, C, _ep, _a = return_mapping(
            strain, None, np.array([0.0]), mat, plane_stress=plane_stress)
        C = np.asarray(C).reshape(n, n)
        num = np.zeros((n, n))
        h = 1e-9
        for j in range(n):
            d = np.zeros(n)
            d[j] = h
            sp = np.ravel(return_mapping(
                strain + d, None, np.array([0.0]), mat, plane_stress=plane_stress)[0])
            sm = np.ravel(return_mapping(
                strain - d, None, np.array([0.0]), mat, plane_stress=plane_stress)[0])
            num[:, j] = (sp - sm) / (2 * h)
        rel = np.max(np.abs(C - num)) / np.max(np.abs(num))
        assert rel < 1e-6, f"tangent inconsistency {rel:.2e}"
        assert np.max(np.abs(C - C.T)) / np.max(np.abs(num)) < 1e-8

    @pytest.mark.parametrize("plane_stress", [True, False])
    def test_stress_returns_to_yield_surface(self, plane_stress):
        mat = steel()
        tmp = np.array([5e-3, 2e-3, 1e-3]) if plane_stress else np.array(
            [5e-3, 2e-3, 1e-3, 1e-3, 0.0, 0.0])
        sig, _C, _ep, alpha = return_mapping(
            tmp, None, np.array([0.0]), mat, plane_stress=plane_stress)
        sig = np.asarray(sig).ravel()
        sy = mat.sigma_y0 + mat.H * float(np.ravel(alpha)[0])
        assert abs(yield_function(sig, sy)) < 1e-6 * mat.sigma_y0

    def test_elastic_tangent_equals_elastic_matrix(self):
        mat = steel()
        _s, C, _ep, a = return_mapping(
            np.array([1e-4, -3e-5, 0.0]), None, np.array([0.0]), mat,
            plane_stress=True)
        assert np.allclose(np.asarray(C).reshape(3, 3),
                           mat.elastic_matrix_plane_stress, rtol=1e-12)
        assert float(np.ravel(a)[0]) == 0.0


# ===========================================================================
# 2. plate buckling / Smith-method collapse anchors
# ===========================================================================
class TestPanelCollapseVerification:
    def test_plate_buckling_stress_anchor(self):
        """Anchor 3: σ_cr = k π² E/(12(1-ν²))(t/b)², k = 4 (a/b ≥ 1)."""
        mat = steel()
        g = panel(mat, a=2.4, b=0.7, t=0.012)
        solver = ProgressiveCollapseSolver(g)
        expected = 4.0 * np.pi**2 * mat.E / (12.0 * (1 - NU**2)) * (g.t / g.b) ** 2
        assert abs(solver.sigma_cr_plate - expected) / expected < 1e-12
        # legacy API exposes the same elastic reference
        assert abs(SmithCollapseSolver(g).sigma_cr_plate - expected) / expected < 1e-12

    def test_plate_element_ultimate_is_von_karman(self):
        """Anchor 4: plate element ultimate = sqrt(σ_cr σ_y)."""
        mat = steel(H=0.0)
        g = panel(mat, b=0.7, t=0.012)
        solver = ProgressiveCollapseSolver(g)
        eps_y = mat.sigma_y0 / mat.E
        sigma = plate_load_shortening_stress(eps_y, mat, solver.sigma_cr_plate)
        expected = np.sqrt(solver.sigma_cr_plate * mat.sigma_y0)
        assert abs(sigma - expected) / expected < 1e-9
        # the post-ultimate branch decreases
        sigma2 = plate_load_shortening_stress(eps_y + 2e-3, mat, solver.sigma_cr_plate)
        assert sigma2 < sigma

    def test_stocky_panel_reaches_full_yield(self):
        """Anchor 5: stocky panel → σ_u = σ_y (baseline gave 1.13 σ_y)."""
        mat = steel(H=0.0)
        g = panel(mat, b=0.30, t=0.014, h_s=0.25, t_w=0.012,
                  b_f=0.120, t_f=0.016, w0_max=0.0)
        assert g.plate_slenderness < 1.0
        res = ProgressiveCollapseSolver(g, n_steps=100).solve()
        ratio = res.sigma_u / mat.sigma_y0
        assert abs(ratio - 1.0) < 0.03, f"stocky panel ratio {ratio:.3f}"
        # yield-surface invariant: with H = 0 the average stress cannot exceed σ_y
        assert res.sigma_u <= mat.sigma_y0 * (1.0 + 1e-9)

    def test_hardening_raises_stocky_ultimate(self):
        mat = steel(H=1.0e9)
        g = panel(mat, b=0.30, t=0.014, h_s=0.25, t_w=0.012,
                  b_f=0.120, t_f=0.016, w0_max=0.0)
        res = ProgressiveCollapseSolver(g, n_steps=100).solve()
        assert res.sigma_u > mat.sigma_y0

    def test_slender_panels_are_weaker_and_ordered(self):
        """Anchor 6: σ_u decreases monotonically with plate slenderness β."""
        mat = steel(H=0.0)
        betas, ratios = [], []
        for b in (0.30, 0.50, 0.70, 1.00, 1.40):
            g = panel(mat, b=b, t=0.012, w0_max=0.0)
            res = ProgressiveCollapseSolver(g, n_steps=100).solve()
            betas.append(g.plate_slenderness)
            ratios.append(res.sigma_u / mat.sigma_y0)
        assert all(ratios[i + 1] < ratios[i] for i in range(len(ratios) - 1)), ratios
        # a strongly slender panel must lose a large part of its capacity
        assert ratios[-1] < 0.60, f"β = {betas[-1]:.2f} gave σ_u/σ_y = {ratios[-1]:.3f}"

    def test_ultimate_between_von_karman_and_faulkner(self):
        """Model-form bracket: von Karman (upper) / Faulkner (lower) plate limits."""
        mat = steel(H=0.0)
        g = panel(mat, b=0.7, t=0.012, w0_max=0.0)
        res = ProgressiveCollapseSolver(g, n_steps=100).solve()
        vk = np.sqrt(res.sigma_cr_plate * mat.sigma_y0)
        faulk = res.diagnostics["faulkner_plate_ultimate"]
        assert faulk < res.sigma_u <= 1.05 * mat.sigma_y0
        # plate-dominated panel: ultimate close to (and not above) von Karman
        assert res.sigma_u <= 1.10 * vk
        assert res.sigma_u > 0.9 * faulk

    def test_limit_point_and_post_peak_softening(self):
        mat = steel(H=0.0)
        g = panel(mat, b=0.7, t=0.012, w0_max=0.0024)
        res = ProgressiveCollapseSolver(g, n_steps=120, max_strain=0.02).solve()
        stresses = np.array([s for _, s in res.load_displacement])
        peak_idx = int(np.argmax(stresses))
        assert peak_idx < len(stresses) - 1, "no post-ultimate branch resolved"
        assert res.post_peak_ratio() < 0.95, \
            f"post/peak = {res.post_peak_ratio():.3f} (no softening)"
        # the peak must be an interior maximum, not the last point
        assert stresses[-1] < stresses[peak_idx]

    def test_fiber_convergence(self):
        mat = steel(H=0.0)
        g = panel(mat, b=0.7, t=0.012, w0_max=0.0024)
        study = panel_convergence_study(g, fiber_counts=(2, 4, 8, 16), n_steps=80)
        assert study["finest_vs_coarsest"] < 0.02
        assert study["converged"]

    def test_imperfection_develops_bending_and_deflection(self):
        mat = steel(H=0.0)
        g = panel(mat, b=0.7, t=0.012, w0_max=0.0024)
        res = ProgressiveCollapseSolver(g, n_steps=100).solve()
        delta0 = res.diagnostics["initial_imperfection"]
        delta_u = res.diagnostics["deflection_at_ultimate"]
        # bending develops (non-zero curvature) and the mid-span deflection
        # reaches at least the imperfection amplitude somewhere on the path
        assert abs(res.diagnostics["kappa_at_ultimate"]) > 0.0
        peak = max(abs(d) for _, d in res.deflection_path)
        assert peak >= abs(delta0) - 1e-12, f"{peak:.5f} < {abs(delta0):.5f}"
        assert abs(delta_u) > 0.2 * abs(delta0)

    def test_legacy_api_delegates_to_progressive_solver(self):
        mat = steel()
        g = panel(mat)
        legacy = SmithCollapseSolver(g, n_steps=80, max_strain=0.02).solve()
        modern = ProgressiveCollapseSolver(g, n_steps=80, max_strain=0.02).solve()
        assert abs(legacy.sigma_u - modern.sigma_u) / modern.sigma_u < 1e-12
        assert legacy.failure_mode == modern.failure_mode
        assert len(legacy.load_displacement) == len(modern.load_displacement)

    def test_uniaxial_element_curve_matches_constitutive_law(self):
        mat = steel()
        for eps in (1e-3, 3e-3, 8e-3):
            sigma_ex, _ = uniaxial_exact(eps, mat)
            assert abs(uniaxial_elastoplastic_stress(eps, mat) - sigma_ex) / sigma_ex < 1e-12
            # symmetric in tension
            assert abs(uniaxial_elastoplastic_stress(-eps, mat) + sigma_ex) / sigma_ex < 1e-12


# ===========================================================================
# 3. ice belt: pressure-area law and panel capacity
# ===========================================================================
class TestIceBeltVerification:
    def test_pressure_area_exponent(self):
        """Anchor 7: P = P0 (A/A0)^-0.3, fitted exponent error < 1e-12."""
        out = pressure_area_law_icebelt(
            [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0], P0=5e6, exponent=-0.3)
        assert out["verified"]
        assert out["exponent_error"] < 1e-12
        assert abs(out["fitted_exponent"] + 0.3) < 1e-12

    def test_pressure_area_roundtrip(self):
        law = IACSIceBeltPressure(P0=5.6e6, A0=1.0, exponent=-0.3)
        for area in (0.05, 0.5, 3.0):
            p = law.pressure(area)
            assert abs(law.area_from_pressure(p) - area) / area < 1e-12

    def test_strip_plastic_collapse_anchor(self):
        """Anchor 8: p_u = 8(1+f) M_p/L², M_p = σ_y t²/4."""
        L, t, sy = 0.8, 0.025, 355e6
        m_p = sy * t**2 / 4.0
        ss = icebelt_panel_plastic_capacity(L, t, sy, 0.0)
        cl = icebelt_panel_plastic_capacity(L, t, sy, 1.0)
        assert abs(ss["p_u"] - 8.0 * m_p / L**2) / ss["p_u"] < 1e-12
        assert abs(cl["p_u"] - 16.0 * m_p / L**2) / cl["p_u"] < 1e-12
        assert abs(cl["p_u"] / ss["p_u"] - 2.0) < 1e-12

    def test_plastic_section_modulus_rectangular(self):
        mat = VonMisesMaterial(E=E, nu=NU, sigma_y0=355e6, H=0.0)
        g = StiffenedPlateGeometry(a=0.4, b=1.0, t=0.02, h_s=1e-6, t_w=1e-9,
                                   b_f=0.0, t_f=0.0, material=mat)
        sec = plastic_section_modulus(g, n_web_fibers=2, n_plate_fibers=16)
        analytic = g.b * g.t**2 / 4.0
        assert abs(sec["Z_p"] - analytic) / analytic < 1e-10

    def test_stiffened_panel_capacity_scaling(self):
        mat = VonMisesMaterial(E=E, nu=NU, sigma_y0=355e6, H=0.0)
        g = StiffenedPlateGeometry(a=0.6, b=0.7, t=0.025, h_s=0.30, t_w=0.012,
                                   b_f=0.12, t_f=0.018, material=mat)
        cap = icebelt_stiffened_panel_capacity(g)
        # p_u ∝ 1/a², ∝ σ_y, ∝ (1+f)
        g2 = StiffenedPlateGeometry(a=1.2, b=0.7, t=0.025, h_s=0.30, t_w=0.012,
                                    b_f=0.12, t_f=0.018, material=mat)
        cap2 = icebelt_stiffened_panel_capacity(g2)
        assert abs(cap2["p_u"] * 4.0 / cap["p_u"] - 1.0) < 0.02
        cap3 = icebelt_stiffened_panel_capacity(g, sigma_y=710e6)
        assert abs(cap3["p_u"] / cap["p_u"] - 2.0) < 1e-12
        capf = icebelt_stiffened_panel_capacity(g, end_fixity=1.0)
        assert abs(capf["p_u"] / cap["p_u"] - 2.0) < 1e-12

    def test_design_ratio_realistic_icebelt(self):
        mat = VonMisesMaterial(E=E, nu=NU, sigma_y0=355e6, H=0.0)
        g = StiffenedPlateGeometry(a=0.6, b=0.7, t=0.025, h_s=0.30, t_w=0.012,
                                   b_f=0.12, t_f=0.018, material=mat)
        law = IACSIceBeltPressure(P0=2.0e6, A0=1.0, exponent=-0.3)
        cap = icebelt_stiffened_panel_capacity(g)["p_u"]
        out = icebelt_design_ratio(law, g.a, g.b, g.t, mat.sigma_y0, geometry=g)
        assert abs(out["pressure_capacity"] / cap - 1.0) < 1e-12
        assert out["utilization"] < 1.0, "PC-class panel should have margin"

    def test_exponent_helper_is_honest(self):
        out = verify_pressure_area_exponent()
        assert out["exponent_used"] == -0.3
        assert out["within_bounds"] is True            # inside literature span
        out2 = verify_pressure_area_exponent(exponent_range=(-0.55, -0.45))
        assert out2["within_bounds"] is False          # outside that sub-band
        assert out2["regulatory_source_checked"] is False
        assert "interpretation" in out2