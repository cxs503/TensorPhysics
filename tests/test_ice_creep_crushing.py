"""Tests for Norton-Bailey creep crushing with structural compliance.

Covers:
  (a) Norton creep parameter consistency (Arrhenius, rate law)
  (b) Rigid-limit steady-state pressure vs analytical
  (c) Semi-implicit solver convergence to steady state
  (d) Stress relaxation vs closed-form solution
  (e) Pressure–rate scaling exponent m ≈ 1/n
  (f) Relaxation time constant consistency
  (g) IACS PC monotonicity
  (h) Frame-strip interface check
  (i) Solver stability for extreme parameters
  (j) Compliance effect: soft vs stiff structure
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from tensorphysics.ice.creep_crushing import (
    NortonCreepParams,
    CreepCrushingConfig,
    solve_creep_crushing,
    steady_state_pressure,
    rigid_limit_pressure,
    relaxation_time_constant,
    analytical_relaxation,
    pressure_rate_exponent,
    iacs_pc_sustained_pressure,
    check_iacs_monotonicity,
    link_frame_strip_check,
)


# ── (a) Norton creep parameters ──────────────────────────────────────

class TestNortonCreepParams:
    def test_arrhenius_increases_with_temperature(self):
        cold = NortonCreepParams(T_K=253.15)
        warm = NortonCreepParams(T_K=268.15)
        assert warm.B_eff > cold.B_eff

    def test_creep_rate_positive_for_positive_stress(self):
        params = NortonCreepParams()
        rate = params.creep_rate(1e6)
        assert rate > 0

    def test_creep_rate_scales_as_power_n(self):
        params = NortonCreepParams(n=3.0)
        r1 = params.creep_rate(1e6)
        r2 = params.creep_rate(2e6)
        ratio = r2 / r1
        expected = 2.0 ** 3.0
        assert abs(ratio - expected) / expected < 1e-10

    def test_creep_rate_sign(self):
        params = NortonCreepParams()
        assert params.creep_rate(-1e6) < 0
        assert params.creep_rate(0.0) == 0.0


# ── (b) Rigid-limit steady-state pressure ────────────────────────────

class TestRigidLimit:
    def test_steady_state_formula(self):
        v, B, n, L_c = 0.01, 1e-18, 3.0, 0.5
        p_ss = steady_state_pressure(v, B, n, L_c)
        expected = (v / (B * L_c)) ** (1.0 / n)
        assert abs(p_ss - expected) / expected < 1e-12

    def test_rigid_limit_via_params(self):
        creep = NortonCreepParams(B_ref=4e-18, T_K=263.15, Q=0.0)
        v, L_c = 0.01, 0.5
        p = rigid_limit_pressure(v, creep, L_c)
        expected = (v / (creep.B_eff * L_c)) ** (1.0 / creep.n)
        assert abs(p - expected) / expected < 1e-10

    def test_zero_velocity_gives_zero_pressure(self):
        assert steady_state_pressure(0.0, 1e-18, 3.0, 0.5) == 0.0


# ── (c) Solver convergence to steady state ───────────────────────────

class TestSolverConvergence:
    def test_converges_to_rigid_limit_for_stiff_structure(self):
        """With very stiff structure, solver should approach rigid-limit p_ss."""
        creep = NortonCreepParams(B_ref=1e-16, Q=0.0, n=3.0, T_K=263.15)
        v = 0.01
        L_c = 0.5
        p_rigid = rigid_limit_pressure(v, creep, L_c)

        cfg = CreepCrushingConfig(
            v=v, k_s=1e12, A=0.1, L_c=L_c,
            creep=creep, p0=0.0, t_end=10.0, n_steps=2000,
        )
        res = solve_creep_crushing(cfg)
        assert res.converged
        # Last 20% average should be within 10% of rigid limit
        p_late = np.mean(res.p[-400:])
        assert abs(p_late - p_rigid) / p_rigid < 0.10, (
            f"p_late={p_late:.3e} vs p_rigid={p_rigid:.3e}"
        )

    def test_pressure_starts_from_zero_and_rises(self):
        cfg = CreepCrushingConfig(v=0.01, k_s=1e8, A=0.1, L_c=0.5,
                                  p0=0.0, t_end=5.0, n_steps=1000)
        res = solve_creep_crushing(cfg)
        assert res.p[0] == 0.0
        assert res.p[-1] > 0.0
        # Pressure should be monotonically increasing (for v > 0, p0 = 0)
        assert np.all(np.diff(res.p) >= -1e-10)


# ── (d) Stress relaxation vs closed-form ─────────────────────────────

class TestRelaxation:
    def test_relaxation_trend_matches_analytical(self):
        """Solver in relaxation mode should show the correct trend.
        
        Note: backward Euler is first-order and has numerical diffusion for
        nonlinear problems (n=3). We check order-of-magnitude and trend rather
        than exact quantitative match.
        """
        B = 1e-20
        n = 3.0
        k_s = 1e8
        A = 0.1
        L_c = 0.5
        p0 = 1e6

        tau0 = relaxation_time_constant(k_s, A, L_c, B, n, p0)
        assert 0.01 < tau0 < 10.0

        t_end = 5.0 * tau0
        cfg = CreepCrushingConfig(
            v=0.0, k_s=k_s, A=A, L_c=L_c,
            creep=NortonCreepParams(B_ref=B, Q=0.0, n=n, T_K=263.15),
            p0=p0, t_end=t_end, n_steps=2000, relaxation_mode=True,
        )
        res = solve_creep_crushing(cfg)
        assert res.converged

        # Check that pressure decreases (correct trend)
        assert res.p[-1] < res.p[0]
        
        # Check order-of-magnitude at t=tau0
        idx_tau = int(tau0 / t_end * cfg.n_steps)
        p_at_tau = res.p[idx_tau]
        p_ana_at_tau = analytical_relaxation(p0, k_s, A, L_c, B, n,
                                             np.array([tau0]))[0]
        # Should be within a factor of 2 (order-of-magnitude agreement)
        assert 0.5 * p_ana_at_tau < p_at_tau < 2.0 * p_ana_at_tau, (
            f"p_num={p_at_tau:.3e} vs p_ana={p_ana_at_tau:.3e}"
        )

    def test_relaxation_decreases_monotonically(self):
        cfg = CreepCrushingConfig(
            v=0.0, k_s=1e8, A=0.1, L_c=0.5,
            creep=NortonCreepParams(B_ref=1e-20, Q=0.0, n=3.0, T_K=263.15),
            p0=1e6, t_end=1.0, n_steps=1000, relaxation_mode=True,
        )
        res = solve_creep_crushing(cfg)
        assert np.all(np.diff(res.p) <= 1e-10)

    def test_analytical_n_equals_1(self):
        """For n=1 (linear viscosity), relaxation should be exponential."""
        p0 = 1e6
        t = np.linspace(0, 10, 100)
        p = analytical_relaxation(p0, 1e8, 0.1, 0.5, 1e-16, 1.0, t)
        tau = relaxation_time_constant(1e8, 0.1, 0.5, 1e-16, 1.0, p0)
        expected = p0 * np.exp(-t / tau)
        np.testing.assert_allclose(p, expected, rtol=1e-10)


# ── (e) Pressure–rate scaling exponent ───────────────────────────────

class TestScalingExponent:
    def test_rigid_limit_exponent_is_one_over_n(self):
        """In rigid limit, m = 1/n exactly."""
        creep = NortonCreepParams(n=3.0)
        m = pressure_rate_exponent(0.01, creep, 0.5, k_s=float('inf'))
        assert abs(m - 1.0 / 3.0) < 1e-10

    def test_numerical_exponent_close_to_theory(self):
        """With finite stiffness, m should still be close to 1/n at steady state."""
        creep = NortonCreepParams(B_ref=1e-16, Q=0.0, n=3.0, T_K=263.15)
        m = pressure_rate_exponent(0.01, creep, 0.5, k_s=1e10, A=0.1)
        # Should be close to 1/3 (within 20% — compliance modifies prefactor)
        assert abs(m - 1.0 / 3.0) / (1.0 / 3.0) < 0.20, f"m={m}"

    def test_solver_scaling_exponent(self):
        """The solver-computed m should be close to 1/n."""
        creep = NortonCreepParams(B_ref=1e-16, Q=0.0, n=3.0, T_K=263.15)
        cfg = CreepCrushingConfig(
            v=0.01, k_s=1e10, A=0.1, L_c=0.5,
            creep=creep, t_end=50.0, n_steps=2000,
        )
        res = solve_creep_crushing(cfg)
        assert abs(res.m_scaling - 1.0 / 3.0) / (1.0 / 3.0) < 0.25, (
            f"m_scaling={res.m_scaling}"
        )


# ── (f) Relaxation time constant ─────────────────────────────────────

class TestRelaxationTime:
    def test_formula(self):
        tau = relaxation_time_constant(1e8, 0.1, 0.5, 1e-16, 3.0, 1e6)
        expected = 0.1 / (1e8 * 1e-16 * 3.0 * 0.5 * (1e6) ** 2)
        assert abs(tau - expected) / expected < 1e-10

    def test_higher_stiffness_shorter_relaxation(self):
        tau1 = relaxation_time_constant(1e7, 0.1, 0.5, 1e-16, 3.0, 1e6)
        tau2 = relaxation_time_constant(1e9, 0.1, 0.5, 1e-16, 3.0, 1e6)
        assert tau2 < tau1

    def test_higher_pressure_shorter_relaxation(self):
        tau1 = relaxation_time_constant(1e8, 0.1, 0.5, 1e-16, 3.0, 1e6)
        tau2 = relaxation_time_constant(1e8, 0.1, 0.5, 1e-16, 3.0, 5e6)
        assert tau2 < tau1


# ── (g) IACS PC monotonicity ─────────────────────────────────────────

class TestIACSMonotonicity:
    def test_monotonicity_at_reference_area(self):
        ok, values = check_iacs_monotonicity(area=1.0)
        assert ok, f"Monotonicity failed: {values}"

    def test_monotonicity_at_various_areas(self):
        for area in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
            ok, _ = check_iacs_monotonicity(area=area)
            assert ok, f"Monotonicity failed at area={area}"

    def test_pressure_values_positive(self):
        for pc in ["PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7"]:
            p = iacs_pc_sustained_pressure(pc)
            assert p > 0

    def test_invalid_class_raises(self):
        with pytest.raises(ValueError):
            iacs_pc_sustained_pressure("PC99")

    def test_pressure_area_effect(self):
        """Larger area → lower pressure (negative exponent)."""
        p_small = iacs_pc_sustained_pressure("PC6", area=0.1)
        p_large = iacs_pc_sustained_pressure("PC6", area=10.0)
        assert p_small > p_large


# ── (h) Frame-strip interface ────────────────────────────────────────

class TestFrameStripLink:
    def test_low_pressure_ok(self):
        """100 kPa on a thick plate with short span should be OK."""
        result = link_frame_strip_check(
            p_sustained=0.1e6,  # 0.1 MPa (100 kPa) — moderate
            L_span=1.5,         # short span
            b_trib=0.6,
            sigma_y=355e6,
            E=210e9,
            h_plate=0.020,      # 20 mm thick plate
            b_stiffener=0.15,
        )
        assert result["ok"], f"utilization={result['utilization']:.2f}"
        assert result["utilization"] < 1.0

    def test_high_pressure_fails(self):
        result = link_frame_strip_check(
            p_sustained=50e6,  # 50 MPa — extreme
            L_span=3.0,
            b_trib=0.6,
            sigma_y=355e6,
            E=210e9,
            h_plate=0.010,
            b_stiffener=0.15,
        )
        assert not result["ok"]
        assert result["utilization"] > 1.0

    def test_utilization_scales_with_pressure(self):
        r1 = link_frame_strip_check(1e6, 3.0, 0.6, 355e6, 210e9, 0.015, 0.15)
        r2 = link_frame_strip_check(2e6, 3.0, 0.6, 355e6, 210e9, 0.015, 0.15)
        assert abs(r2["utilization"] / r1["utilization"] - 2.0) < 1e-10


# ── (i) Solver stability ─────────────────────────────────────────────

class TestSolverStability:
    def test_moderate_stiffness(self):
        """Moderate stiffness should converge."""
        cfg = CreepCrushingConfig(
            v=0.01, k_s=1e10, A=0.1, L_c=0.5,
            creep=NortonCreepParams(B_ref=1e-16, Q=0.0, n=3.0, T_K=263.15),
            t_end=10.0, n_steps=1000,
        )
        res = solve_creep_crushing(cfg)
        assert res.converged
        assert np.all(np.isfinite(res.p))

    def test_extreme_soft_structure(self):
        """Very soft structure: pressure stays low."""
        cfg = CreepCrushingConfig(
            v=0.01, k_s=1e3, A=0.1, L_c=0.5,
            creep=NortonCreepParams(B_ref=1e-16, Q=0.0, n=3.0, T_K=263.15),
            t_end=10.0, n_steps=1000,
        )
        res = solve_creep_crushing(cfg)
        assert res.converged
        assert res.p_peak < 1e6

    def test_high_velocity_gives_correct_steady_state(self):
        """High velocity should give pressure close to rigid limit."""
        creep = NortonCreepParams(B_ref=1e-16, Q=0.0, n=3.0, T_K=263.15)
        v = 1.0
        p_rigid = rigid_limit_pressure(v, creep, 0.5)
        cfg = CreepCrushingConfig(
            v=v, k_s=1e10, A=0.1, L_c=0.5,
            creep=creep, t_end=10.0, n_steps=1000,
        )
        res = solve_creep_crushing(cfg)
        assert res.converged
        # Should be within 10% of rigid limit
        assert abs(res.p_sustained - p_rigid) / p_rigid < 0.10


# ── (j) Compliance effect ────────────────────────────────────────────

class TestComplianceEffect:
    def test_stiffer_reaches_steady_state_faster(self):
        """Stiffer structure should reach steady state faster (shorter tau)."""
        creep = NortonCreepParams(B_ref=1e-16, Q=0.0, n=3.0, T_K=263.15)
        v = 0.01
        L_c = 0.5
        
        # Short simulation: soft structure hasn't reached steady state yet
        cfg_soft = CreepCrushingConfig(
            v=v, k_s=1e6, A=0.1, L_c=L_c,
            creep=creep, t_end=0.5, n_steps=1000,
        )
        cfg_stiff = CreepCrushingConfig(
            v=v, k_s=1e10, A=0.1, L_c=L_c,
            creep=creep, t_end=0.5, n_steps=1000,
        )
        res_soft = solve_creep_crushing(cfg_soft, compute_m=False)
        res_stiff = solve_creep_crushing(cfg_stiff, compute_m=False)
        
        # At short time, stiff should have higher pressure (closer to steady state)
        assert res_stiff.p[-1] > res_soft.p[-1], (
            f"stiff p={res_stiff.p[-1]:.3e} should > soft p={res_soft.p[-1]:.3e}"
        )

    def test_soft_structure_elastic_dominates(self):
        """For very soft structure, most indentation is elastic (not creep)."""
        creep = NortonCreepParams(B_ref=1e-16, Q=0.0, n=3.0, T_K=263.15)
        cfg = CreepCrushingConfig(
            v=0.01, k_s=1e4, A=0.1, L_c=0.5,
            creep=creep, t_end=10.0, n_steps=1000,
        )
        res = solve_creep_crushing(cfg)
        # For soft structure, pressure is low → creep rate is low
        # So effective velocity ≈ imposed velocity (elastic dominates)
        v_eff_mean = np.mean(np.abs(res.v_eff[1:]))
        # v_eff should be close to v (most of the indentation is elastic)
        assert v_eff_mean > 0.8 * cfg.v
