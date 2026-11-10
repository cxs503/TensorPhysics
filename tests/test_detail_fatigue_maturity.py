"""Tests for the hull detail-fatigue maturity capability (node p246w2-n09).

Coverage
--------
* FEM kernel: uniform-tension patch test (exactness), Kirsch hole Kt,
  butt-weld-toe notch Kt vs the Inglis solution, thickness-step model,
  T-joint model sanity;
* hot-spot extrapolation: exact reproduction of constant and linear fields,
  sampling-window validation;
* IIW / DNV S-N curves: FAT anchor, bi-linear continuity, log-log fitting,
  cross-standard agreement, declared scatter model;
* Palmgren-Miner: exact D = sum(n_i/N_i) identity, two-block closed form vs
  independent algebra, rainflow-based variable amplitude, sequence blindness;
* C7 honest gate: the Corten-Dolan gap is declared and the limitation is
  demonstrated numerically;
* the full C1..C7 acceptance suite (quick configuration).
"""

import math

import numpy as np
import pytest

from tensorphysics.fatigue.detail_maturity import (
    CORTEN_DOLAN_STATUS,
    IIWCurve,
    POINTS_IIW_LINEAR,
    POINTS_IIW_QUADRATIC,
    fatigue_curve,
    fat_from_log_a1,
    log_a1_from_fat,
    loglog_fit,
    reference_fit,
    standard_cross_check,
    design_to_mean_factor,
    simulate_test_series,
    block_history,
    miner_damage,
    power_law_two_block_life,
    rainflow_block_damage,
    sequence_effect_probe,
    two_block_blocks_to_failure,
    extrapolate_from_profile,
    lagrange_extrapolate,
    run_acceptance,
)
from tensorphysics.fatigue.detail_maturity import plane_stress_fem as fem
from tensorphysics.fatigue.detail_maturity.damage import scatter_band_factor


# ============================================================================
# FEM kernel
# ============================================================================

class TestFEMKernel:
    def test_patch_test_uniform_tension_is_exact(self):
        """A uniform stress field must be reproduced to machine precision."""
        L, H, s0 = 100.0, 20.0, 1.0
        xs = np.linspace(0.0, L, 11)
        ys = np.linspace(0.0, H, 4)
        grid = np.zeros((len(xs), len(ys), 2))
        grid[..., 0] = xs[:, None]
        grid[..., 1] = ys[None, :]
        builder = fem.T6MeshBuilder()
        builder.add_grid(grid)
        model = builder.build()

        fixed = {}
        for nid in fem.nodes_on_line(model, 1, 0.0):
            fixed[int(nid)] = (None, 0.0)
        corner_ids = fem.nodes_on_line(model, 0, 0.0)
        corner = int(corner_ids[np.argmin(model.nodes[corner_ids, 1])])
        fixed[corner] = (0.0, 0.0)

        loads = []
        for x0, sign in ((0.0, -1.0), (L, 1.0)):
            ids = fem.nodes_on_line(model, 0, x0)
            ids = ids[np.argsort(model.nodes[ids, 1])]
            loads.extend(fem.boundary_edge_loads(
                model, fem._corner_only(model, ids),
                fem.uniform_traction([sign * s0, 0.0])))
        sol = fem.solve_plane_stress(model, fem.assemble_loads(model, loads), fixed)

        assert np.max(np.abs(sol.nodal_stress[:, 0] - s0)) < 1e-9
        assert np.max(np.abs(sol.nodal_stress[:, 1])) < 1e-9
        ux_exact = s0 * L / model.E
        assert abs(float(np.max(sol.displacement[:, 0])) - ux_exact) / ux_exact < 1e-9

    def test_tri6_stiffness_symmetry_and_rigid_body(self):
        xy = np.array([[0.0, 0.0], [2.0, 0.3], [0.4, 1.7]])
        ke = fem.tri6_stiffness(xy, 210000.0, 0.3, 1.0)
        assert ke.shape == (12, 12)
        assert np.allclose(ke, ke.T, atol=1e-6)
        # rigid body translations give zero force
        for direction in (0, 1):
            u = np.zeros(12)
            u[direction::2] = 1.0
            assert np.max(np.abs(ke @ u)) < 1e-6 * np.max(np.abs(ke))

    def test_kirsch_hole_kt_within_10_percent(self):
        sol = fem.solve_plate_with_hole(1.0, a=10.0, R_factor=12.0, n_theta=32,
                                        first_layer_frac=0.05)
        m = sol.model
        r = np.hypot(m.nodes[:, 0], m.nodes[:, 1])
        ids = np.where((np.abs(m.nodes[:, 0]) < 1e-9)
                       & (np.abs(r - m.meta["a"]) < 1e-6))[0]
        kt = float(np.max(sol.nodal_stress[ids, 0]))
        assert abs(kt - 3.0) / 3.0 < 0.10, f"Kt={kt}"

    def test_hole_pole_stress_matches_kirsch(self):
        """At theta = 0 the hoop stress of the Kirsch field is -sigma_0."""
        sol = fem.solve_plate_with_hole(1.0, a=10.0, R_factor=12.0, n_theta=32,
                                        first_layer_frac=0.05)
        m = sol.model
        r = np.hypot(m.nodes[:, 0], m.nodes[:, 1])
        ids = np.where((np.abs(m.nodes[:, 1]) < 1e-9)
                       & (np.abs(r - m.meta["a"]) < 1e-6))[0]
        syy = float(sol.nodal_stress[ids, 1][0])
        assert abs(syy + 1.0) < 0.10

    @pytest.mark.parametrize("h,rho", [(1.0, 1.0), (0.5, 1.0)])
    def test_butt_toe_kt_vs_inglis(self, h, rho):
        sol = fem.solve_plate_with_edge_notch(1.0, L=120.0, W=40.0, h=h, rho=rho,
                                              n_arc=16, first_layer=rho / 4.0)
        m = sol.model
        peak, _ = fem.peak_surface_stress(sol, m.meta["notch_x"], rho * 1.001)
        inglis = 1.0 + 2.0 * math.sqrt(h / rho)
        assert abs(peak - inglis) / inglis < 0.10, f"Kt_fem={peak} Inglis={inglis}"

    def test_notch_kt_converges_with_arc_refinement(self):
        """Refining the toe resolution must move Kt monotonically towards Inglis."""
        vals = []
        for n_arc in (8, 24):
            sol = fem.solve_plate_with_edge_notch(1.0, L=120.0, W=40.0, h=1.0,
                                                  rho=1.0, n_arc=n_arc,
                                                  first_layer=1.0 / (2.0 * n_arc))
            m = sol.model
            peak, _ = fem.peak_surface_stress(sol, m.meta["notch_x"], 1.001)
            vals.append(peak)
        assert vals[1] < vals[0], f"Kt did not decrease on refinement: {vals}"
        assert abs(vals[1] - 3.0) / 3.0 < 0.05

    def test_thickness_step_equilibrium(self):
        """Equal membrane force at both ends: thin side carries sigma_0."""
        sol = fem.solve_plate_with_thickness_step(1.0, L=120.0, t1=12.0, t2=16.0,
                                                  rho=8.0)
        m = sol.model
        xs, sig = fem.surface_stress_profile(sol, 0)
        toe = m.meta["x_toe"]
        far_thin = sig[xs < toe - 30.0].mean()
        far_thick = sig[xs > toe + 30.0].mean()
        assert abs(far_thin - 1.0) < 0.02
        assert abs(far_thick - 12.0 / 16.0) < 0.02

    def test_t_joint_runs_and_shows_toe_concentration(self):
        sol = fem.solve_t_joint(1.0, L=200.0, t=20.0, a=60.0, t_a=10.0,
                                first_layer=2.5)
        m = sol.model
        x_toe = m.meta["x_attach"]
        xs, sig = fem.surface_stress_profile(sol, 0, x_range=(0.0, x_toe - 1e-9))
        assert xs.size > 10
        # stress decays to the nominal value away from the toe
        assert abs(sig[0] - 1.0) < 0.05
        # and is elevated approaching the toe
        assert sig[-1] > 1.2

    def test_mesh_builder_welds_shared_interfaces(self):
        """Two grids sharing an edge must produce a conforming (watertight) mesh."""
        n = 5
        xs = np.linspace(0.0, 1.0, n + 1)
        g1 = np.zeros((n + 1, n + 1, 2))
        g1[..., 0] = xs[:, None]
        g1[..., 1] = np.linspace(0.0, 0.5, n + 1)[None, :]
        g2 = np.zeros((n + 1, n + 1, 2))
        g2[..., 0] = xs[:, None]
        g2[..., 1] = np.linspace(0.5, 1.0, n + 1)[None, :]
        b = fem.T6MeshBuilder()
        b.add_grid(g1)
        b.add_grid(g2)
        nodes = np.asarray(b.nodes, dtype=float)
        # the shared interface row (y = 0.5) must not be duplicated: exactly n+1
        # corner nodes plus the n edge-midpoint nodes of that row
        on_interface = np.isclose(nodes[:, 1], 0.5, atol=1e-12)
        assert int(on_interface.sum()) == 2 * n + 1
        assert np.unique(np.round(nodes[on_interface, 0], 12)).size == 2 * n + 1
        # and no node may be duplicated anywhere (coordinate deduplication)
        keys = {(round(float(x), 9), round(float(y), 9)) for x, y in nodes}
        assert len(keys) == len(nodes)


# ============================================================================
# S-N curves and fitting
# ============================================================================

class TestSNCurves:
    def test_fat90_anchor(self):
        c = fatigue_curve(90.0)
        assert c.life(90.0) == pytest.approx(2.0e6, rel=1e-12)
        assert c.log_a1 == pytest.approx(12.164, abs=5e-4)
        assert c.m1 == 3.0

    def test_log_a_roundtrip(self):
        for fat in (36.0, 71.0, 90.0, 140.0):
            la = log_a1_from_fat(fat, 3.0)
            assert fat_from_log_a1(la, 3.0) == pytest.approx(fat, rel=1e-12)

    def test_bilinear_continuity_at_knee(self):
        c = IIWCurve(fat_class=90.0, m1=3.0, m2=5.0, knee_cycles=1e7)
        s_knee = c.stress_at_knee
        n_from_stress = c.life(s_knee)
        assert n_from_stress == pytest.approx(1e7, rel=1e-9)
        # continuity from both sides
        assert c.stress_for_life(1e7) == pytest.approx(s_knee, rel=1e-9)
        assert c.stress_for_life(1.0001e7) < s_knee

    def test_monotone_decreasing_life(self):
        c = fatigue_curve(90.0)
        s = np.array([200.0, 150.0, 120.0, 90.0, 70.0])
        n = np.array([c.life(v) for v in s])
        assert np.all(np.diff(n) > 0)

    def test_design_curve_is_exactly_linear_in_loglog(self):
        fit = reference_fit(fatigue_curve(90.0), 1e5, 1e7, 21)
        assert fit.r_squared == pytest.approx(1.0, abs=1e-12)
        assert fit.slope_m == pytest.approx(3.0, rel=1e-12)
        assert fit.fat_at_2e6 == pytest.approx(90.0, rel=1e-10)

    def test_loglog_fit_recovers_synthetic_curve(self):
        m, log_a = 3.2, 12.6
        s = np.logspace(np.log10(60.0), np.log10(200.0), 12)
        n = 10.0 ** (log_a - m * np.log10(s))
        fit = loglog_fit(s, n)
        assert fit.slope_m == pytest.approx(m, rel=1e-9)
        assert fit.log_a == pytest.approx(log_a, rel=1e-9)
        assert fit.r_squared > 0.999999

    def test_dnv_iiw_cross_standard(self):
        cross = standard_cross_check()
        assert cross["fat_relative_error_pct"] < 5.0
        assert cross["dnv_d_air_dsigma_c"] == pytest.approx(90.0, abs=0.5)

    def test_design_to_mean_factor(self):
        assert design_to_mean_factor(0.2, 2.0) == pytest.approx(10 ** 0.4, rel=1e-12)

    def test_simulated_series_mean_and_slope(self):
        c = fatigue_curve(90.0)
        levels = [float(v) for v in c.stress_for_life(np.logspace(5.0, 6.7, 4))]
        s, n = simulate_test_series(c, levels, specimens_per_level=25,
                                    sigma_log_n=0.2, seed=7, on_mean_curve=False)
        fit = loglog_fit(s, n)
        assert fit.slope_m == pytest.approx(3.0, rel=0.10)
        assert fit.fat_at_2e6 == pytest.approx(90.0, rel=0.15)


# ============================================================================
# Hot-spot extrapolation
# ============================================================================

class TestHotSpotExtrapolation:
    def test_coefficients_sum_to_one(self):
        for pts in (POINTS_IIW_LINEAR, POINTS_IIW_QUADRATIC, (0.5, 1.0, 1.5)):
            d = np.array(pts, dtype=float)
            _, coeffs = lagrange_extrapolate(d, np.ones_like(d))
            assert coeffs.sum() == pytest.approx(1.0, rel=1e-12)

    def test_constant_field_reproduced_exactly(self):
        d = np.array(POINTS_IIW_QUADRATIC)
        val, _ = lagrange_extrapolate(d, np.full(d.size, 7.5))
        assert val == pytest.approx(7.5, rel=1e-12)

    def test_linear_field_reproduced_by_two_points(self):
        d = np.array(POINTS_IIW_LINEAR)
        stresses = 1.0 + 2.0 * d
        val, _ = lagrange_extrapolate(d, stresses)
        assert val == pytest.approx(1.0, rel=1e-12)

    def test_quadratic_field_reproduced_by_three_points(self):
        d = np.array(POINTS_IIW_QUADRATIC)
        stresses = 3.0 - 0.5 * d + 2.0 * d ** 2
        val, _ = lagrange_extrapolate(d, stresses)
        assert val == pytest.approx(3.0, rel=1e-12)

    def test_profile_sampling_and_extrapolation(self):
        t = 20.0
        x = np.linspace(-60.0, 0.0, 601)           # toe at x = 0, profile left of it
        sigma = 1.0 + 0.6 * np.exp(x / 8.0)
        res = extrapolate_from_profile(x, sigma, x_toe=0.0, thickness=t)
        assert res.method == "quadratic_3point"
        assert res.distances == pytest.approx(np.array(POINTS_IIW_QUADRATIC) * t)
        # the samples must be taken on the profile side of the toe (x < toe here)
        assert np.all(res.sample_x < 0.0)
        assert res.sample_x == pytest.approx(-res.distances)
        expected = 1.0 + 0.6 * np.exp(0.0)
        assert res.hot_spot_stress > 1.0
        assert abs(res.hot_spot_stress - expected) / expected < 0.25

    def test_window_outside_profile_raises(self):
        x = np.linspace(0.0, 10.0, 50)
        with pytest.raises(ValueError):
            extrapolate_from_profile(x, np.ones_like(x), x_toe=0.0, thickness=20.0)

    def test_duplicate_distances_rejected(self):
        with pytest.raises(ValueError):
            lagrange_extrapolate([1.0, 1.0, 2.0], [1.0, 2.0, 3.0])


# ============================================================================
# Miner damage
# ============================================================================

class TestMinerDamage:
    def test_constant_amplitude_d_equals_one(self):
        c = fatigue_curve(90.0)
        n = c.life(90.0)
        d = miner_damage(c, [90.0], [n])
        assert d.total_damage == pytest.approx(1.0, rel=1e-12)
        assert d.blocks_to_failure == pytest.approx(1.0, rel=1e-12)

    def test_damage_linear_in_cycles(self):
        c = fatigue_curve(90.0)
        n = c.life(90.0)
        for frac in (0.1, 0.25, 0.5, 0.9):
            d = miner_damage(c, [90.0], [frac * n])
            assert d.total_damage == pytest.approx(frac, rel=1e-12)

    def test_two_block_matches_independent_algebra(self):
        c = fatigue_curve(90.0)
        lam_loop = two_block_blocks_to_failure(c, 140.0, 2e4, 85.0, 1.5e5)
        lam_algebra = power_law_two_block_life(140.0, 2e4, 85.0, 1.5e5,
                                               c.log_a1, c.m1)
        assert lam_loop == pytest.approx(lam_algebra, rel=1e-12)

    def test_superposition_of_blocks(self):
        c = fatigue_curve(90.0)
        d1 = miner_damage(c, [140.0], [2e4]).total_damage
        d2 = miner_damage(c, [85.0], [1.5e5]).total_damage
        dt = miner_damage(c, [140.0, 85.0], [2e4, 1.5e5]).total_damage
        assert dt == pytest.approx(d1 + d2, rel=1e-12)

    def test_rainflow_recovers_block_spectrum(self):
        c = fatigue_curve(90.0)
        ds1, n1, ds2, n2 = 140.0, 2e3, 85.0, 8e3
        history = block_history([ds1, ds2], [n1, n2], order="high_low")
        rf = rainflow_block_damage(c, history)
        d_blocks = miner_damage(c, [ds1, ds2], [n1, n2]).total_damage
        assert rf["total_damage"] == pytest.approx(d_blocks, rel=1e-3)

    def test_sequence_blindness_is_demonstrated(self):
        c = fatigue_curve(90.0)
        probe = sequence_effect_probe(c, 140.0, 2e4, 85.0, 1.5e5)
        assert probe.is_sequence_independent
        assert probe.relative_difference < 1e-12

    def test_scatter_band_factor(self):
        assert scatter_band_factor(0.2, 0.90) == pytest.approx(2.133, rel=1e-3)
        assert scatter_band_factor(0.2, 0.80) == pytest.approx(1.804, rel=1e-3)

    def test_negative_cycles_rejected(self):
        c = fatigue_curve(90.0)
        with pytest.raises(ValueError):
            miner_damage(c, [90.0], [-1.0])

    def test_mismatched_lengths_rejected(self):
        c = fatigue_curve(90.0)
        with pytest.raises(ValueError):
            miner_damage(c, [90.0, 80.0], [1.0])


# ============================================================================
# Honest gate (C7)
# ============================================================================

class TestHonestGate:
    def test_corten_dolan_declared_missing(self):
        assert CORTEN_DOLAN_STATUS["model"] == "Corten-Dolan"
        assert CORTEN_DOLAN_STATUS["status"] == "not_implemented"
        assert CORTEN_DOLAN_STATUS["action"]


# ============================================================================
# Acceptance suite
# ============================================================================

class TestAcceptanceSuite:
    @pytest.fixture(scope="class")
    def report(self):
        return run_acceptance(quick=True)

    def test_patch_test_passes(self, report):
        assert report["fem_patch_test"]["pass"]

    def test_all_criteria_pass(self, report):
        failed = report["summary"]["failed"]
        assert report["summary"]["all_pass"], f"failed criteria: {failed}"

    def test_c1_numbers(self, report):
        c1 = report["criteria"]["C1"]
        assert c1["design_curve_fit"]["r_squared"] >= 0.98
        assert c1["errors_pct"]["fat_design_vs_90"] < 5.0
        assert c1["errors_pct"]["dnv_curve_d_air_vs_90"] < 5.0

    def test_c2_numbers(self, report):
        c2 = report["criteria"]["C2"]
        assert c2["hole"]["error_pct"] <= 10.0
        assert max(c["error_pct"] for c in c2["butt_joint_toe"]) <= 10.0

    def test_c3_numbers(self, report):
        c3 = report["criteria"]["C3"]["numbers"]
        assert c3["damage_at_predicted_life"] == pytest.approx(1.0, rel=1e-12)
        assert c3["central_ratio_mean_curve"] <= 2.0

    def test_c4_numbers(self, report):
        c4 = report["criteria"]["C4"]
        assert c4["error_pct"]["quadratic_3point"] <= 5.0

    def test_c5_numbers(self, report):
        c5 = report["criteria"]["C5"]["numbers"]
        assert c5["relative_error_loop_vs_algebra"] < 1e-9
        assert c5["rainflow_relative_error"] < 1e-3
        assert c5["sequence_independent"] is True

    def test_c6_numbers(self, report):
        c6 = report["criteria"]["C6"]["changes_pct"]
        assert c6["t4_to_t8"] <= 5.0
        assert c6["t8_to_t16"] <= 5.0

    def test_c7_numbers(self, report):
        assert report["criteria"]["C7"]["corten_dolan"]["status"] == "not_implemented"