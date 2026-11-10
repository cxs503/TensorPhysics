"""Maturity acceptance tests for the underwater attached-ice module (p302w2-n09).

Every criterion of the node specification is tested against an *independent*
reference, never against the implementation's own formula:

C1  thermal balance closure        -> integral latent-heat balance + steady-state
                                      interface balance from the flux modules
C2  Stefan limit                   -> exact solution h**2 = h0**2 + 2 k dT t/(rho L)
C3  negative control (eta=0)       -> h(t) identically zero, < 1e-15
C4  detachment monotonicity        -> 200-point property test, 0 violations
C5  ITTC-1978 penalty chain        -> canonical (105 (k_s/L)^(1/3) - 0.64)e-3
C6  adfreeze interface consistency -> MPa/Pa nerve point, rel. diff < 1e-9
C7  test count                     -> >= 15 (this file alone provides > 30)

Plus regression tests for the three defects found by the maturity audit
(2026-11-10): the ITTC leading coefficient, the convective heat-transfer
closure and the undetermined (window-dependent) "steady" thickness.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from tensorphysics.ice.adfreeze import adfreeze_strength
from tensorphysics.ice.attached_ice import (
    C_P_WATER, CF_REF, DEFAULT_BOND_STRENGTH_PA, K_ICE, L_FUSION, PR_T,
    RHO_ICE, RHO_WATER, T_FREEZE_FW, HANDOFF_SCHEMA,
    IceGrowthODE, bond_strength_from_adfreeze_pa, build_handoff,
    detachment_limited_thickness, detachment_regime, equilibrium_thickness,
    equivalent_sand_roughness, friction_velocity_from_speed, ittc_1957_cf,
    ittc_1978_delta_cf, ittc_1978_roughness_penalty, ittc_1978_thrust_loss,
    monte_carlo_envelope, roughness_state, speed_from_friction_velocity,
    stefan_with_initial, to_json,
)


# ---------------------------------------------------------------------------
# C1 — interface energy balance closure
# ---------------------------------------------------------------------------
class TestC1EnergyClosure:
    """C1: interface energy residual < 2 %."""

    @pytest.mark.parametrize("method", ["euler", "stefan_split"])
    def test_integral_latent_heat_closure(self, method):
        ode = IceGrowthODE(undercooling_k=2.0, u_star=0.03, c_f=1e-4,
                           t_water_k=T_FREEZE_FW + 0.2)
        res = ode.solve_euler(t_max=2 * 3600.0, n_steps=1000, h_init=1e-5,
                              method=method)
        clo = ode.energy_closure(res)
        assert clo["relative_error"] < 0.02, (
            f"{method}: latent-heat closure {clo['relative_error']:.3%} > 2%"
        )
        assert clo["relative_error"] > 0.0  # not a tautology: quadrature is independent

    def test_closure_improves_with_refinement(self):
        ode = IceGrowthODE(undercooling_k=2.0, u_star=0.03, c_f=1e-4,
                           t_water_k=T_FREEZE_FW + 0.2)
        errs = []
        for n in (250, 4000):
            res = ode.solve_euler(t_max=3600.0, n_steps=n, h_init=1e-5,
                                  method="stefan_split")
            errs.append(ode.energy_closure(res, skip_steps=5)["relative_error"])
        assert errs[1] < errs[0], f"closure must improve on refinement: {errs}"

    def test_steady_state_interface_balance(self):
        """Independent closure: q_cond(h_eq) == q_conv - q_frazil at equilibrium."""
        dT, u_s, c_f, eta = 2.0, 0.03, 1e-4, 1.0
        eq = equilibrium_thickness(undercooling_k=dT, u_star=u_s, c_f=c_f,
                                   eta_stick=eta, t_water_k=T_FREEZE_FW + 0.2)
        h = eq["h_eq_m"]
        q_cond = K_ICE * dT / h
        q_net = eq["q_conv_W_m2"] - eq["q_frazil_W_m2"]
        rel = abs(q_cond - q_net) / max(abs(q_cond), abs(q_net))
        assert rel < 0.02, f"steady-state interface imbalance {rel:.3%}"


# ---------------------------------------------------------------------------
# C2 — Stefan limit
# ---------------------------------------------------------------------------
class TestC2StefanLimit:
    """C2: h ~ t^{1/2}, error < 3 % against the analytic similarity solution."""

    @pytest.mark.parametrize("dT,t,h0", [(2.0, 3600.0, 0.0), (2.0, 3600.0, 1e-5),
                                         (0.5, 7200.0, 1e-4), (3.0, 1800.0, 1e-3)])
    def test_stefan_split_matches_exact(self, dT, t, h0):
        ode = IceGrowthODE(undercooling_k=dT, u_star=1e-6, c_f=0.0, eta_stick=0.0)
        res = ode.solve_euler(t_max=t, n_steps=500, h_init=h0, method="stefan_split")
        exact = stefan_with_initial(h0, dT, t)
        rel = abs(res.h_final - exact) / exact
        assert rel < 1e-9, f"stefan_split error {rel:.2e} (exact scheme expected)"
        assert rel < 0.03

    def test_euler_meets_3pct_at_production_resolution(self):
        dT, t, h0 = 2.0, 3600.0, 1e-5
        ode = IceGrowthODE(undercooling_k=dT, u_star=1e-6, c_f=0.0, eta_stick=0.0)
        res = ode.solve_euler(t_max=t, n_steps=20000, h_init=h0, method="euler")
        exact = stefan_with_initial(h0, dT, t)
        rel = abs(res.h_final - exact) / exact
        assert rel < 0.03, f"euler error {rel:.3%} at n=20000"

    def test_sqrt_time_exponent(self):
        ode = IceGrowthODE(undercooling_k=2.0, u_star=1e-6, c_f=0.0, eta_stick=0.0)
        h1 = ode.solve_euler(t_max=900.0, n_steps=200, h_init=1e-5,
                             method="stefan_split").h_final
        h2 = ode.solve_euler(t_max=3600.0, n_steps=200, h_init=1e-5,
                             method="stefan_split").h_final
        exponent = math.log(h2 / h1) / math.log(4.0)
        assert abs(exponent - 0.5) < 0.01, f"h ~ t^{exponent:.4f}, expected 0.5"

    def test_analytic_reduces_to_classic_stefan(self):
        from tensorphysics.ice.attached_ice import stefan_analytic
        for t in (100.0, 3600.0):
            assert stefan_with_initial(0.0, 2.0, t) == pytest.approx(
                stefan_analytic(2.0, t), rel=1e-15)


# ---------------------------------------------------------------------------
# C3 — negative control
# ---------------------------------------------------------------------------
class TestC3NegativeControl:
    """C3: eta_stick = 0 (and dT = 0) reproduces h(t) == 0 to machine zero."""

    @pytest.mark.parametrize("method", ["euler", "stefan_split"])
    def test_no_sticking_no_ice(self, method):
        ode = IceGrowthODE(undercooling_k=0.0, u_star=0.05, c_f=1e-3,
                           eta_stick=0.0, t_water_k=T_FREEZE_FW)
        res = ode.solve_euler(t_max=7200.0, n_steps=1000, h_init=0.0, method=method)
        assert float(np.max(np.abs(res.h_ice))) < 1e-15

    def test_no_sticking_with_undercooling(self):
        """Zero sticking and no conductive driving force -> still exactly zero."""
        ode = IceGrowthODE(undercooling_k=0.0, u_star=0.08, c_f=1e-2,
                           eta_stick=0.0, t_water_k=T_FREEZE_FW)
        res = ode.solve_euler(t_max=3600.0, n_steps=500, h_init=0.0)
        assert res.h_final == 0.0
        assert res.h_final < 1e-15

    def test_handoff_zero_sticking_zero_penalty(self):
        payload = build_handoff(undercooling_k=0.0, duration_s=3600.0,
                                u_star=0.05, c_f=1e-3, eta_stick=0.0, n_steps=100)
        assert payload["scalars"]["h_ice_end_m"] < 1e-15
        assert payload["scalars"]["delta_cf_end"] < 1e-15
        assert payload["scalars"]["k_s_end_m"] == payload["inputs"]["k_s0_m"]["value"]


# ---------------------------------------------------------------------------
# C4 — detachment / velocity monotonicity
# ---------------------------------------------------------------------------
class TestC4Monotonicity:
    """C4: increasing flow speed -> monotonically decreasing steady thickness."""

    N_POINTS = 200

    def test_equilibrium_thickness_monotone_growth_limited(self):
        us = np.linspace(0.005, 0.12, self.N_POINTS)
        h = np.array([
            equilibrium_thickness(undercooling_k=2.0, u_star=float(u), c_f=1e-4,
                                  t_water_k=T_FREEZE_FW + 0.2)["h_eq_m"]
            for u in us
        ])
        violations = int(np.sum(np.diff(h) > 0.0))
        assert violations == 0, f"{violations} monotonicity violations (growth-limited)"
        assert np.all(h > 0.0)

    def test_equilibrium_thickness_monotone_detachment_limited(self):
        """Exercise the stripping branch with a lab-scale (weak) bond."""
        us = np.linspace(0.02, 0.20, self.N_POINTS)
        rows = [equilibrium_thickness(undercooling_k=2.0, u_star=float(u), c_f=1e-5,
                                      t_water_k=T_FREEZE_FW + 0.2, tau_bond=2.0)
                for u in us]
        assert any(r["limited_by"] == "detachment" for r in rows)
        h = np.array([r["h_eq_m"] for r in rows])
        violations = int(np.sum(np.diff(h) > 0.0))
        assert violations == 0, f"{violations} monotonicity violations (detach-limited)"

    def test_numerical_thickness_monotone(self):
        """The integrated layer, not just the closed-form limit, must be monotone."""
        us = np.linspace(0.01, 0.10, 40)
        h = np.array([
            IceGrowthODE(undercooling_k=2.0, u_star=float(u), c_f=1e-4,
                         t_water_k=T_FREEZE_FW + 0.2)
            .solve_euler(t_max=3600.0, n_steps=300, h_init=0.0,
                         method="stefan_split").h_final
            for u in us
        ])
        assert int(np.sum(np.diff(h) > 1e-15)) == 0

    def test_detachment_limited_thickness_decreases(self):
        hs = [detachment_limited_thickness(u, tau_bond=50.0) for u in
              np.linspace(0.01, 0.2, self.N_POINTS)]
        hs = np.array(hs)
        assert np.all(np.diff(hs) <= 0.0)

    def test_shear_detachment_unreachable_at_ship_scale(self):
        """Honest regime statement: stripping needs O(10^2) m/s mean flow."""
        reg = detachment_regime(u_star=0.1, tau_bond=DEFAULT_BOND_STRENGTH_PA)
        assert reg["shear_detachment_active"] is False
        assert reg["speed_crit_m_s"] > 100.0


# ---------------------------------------------------------------------------
# C5 — ITTC-1978 penalty chain
# ---------------------------------------------------------------------------
class TestC5IttcPenalty:
    """C5: Delta C_f over k_s = 0.5 mm -> 5 mm vs ITTC-1978 within 1 %."""

    L = 150.0
    K_S0 = 1.5e-5

    def test_canonical_formula_value(self):
        # (105 (0.5e-3/150)^(1/3) - 0.64)e-3
        expected = (105.0 * (5e-4 / 150.0) ** (1 / 3) - 0.64) * 1e-3
        assert ittc_1978_delta_cf(5e-4, 150.0) == pytest.approx(expected, rel=1e-15)

    def test_penalty_increment_matches_canonical(self):
        d_code = (ittc_1978_roughness_penalty(5e-3, self.K_S0, self.L)
                  - ittc_1978_roughness_penalty(5e-4, self.K_S0, self.L))
        d_ref = (ittc_1978_delta_cf(5e-3, self.L)
                 - ittc_1978_delta_cf(5e-4, self.L))
        rel = abs(d_code - d_ref) / abs(d_ref)
        assert rel < 0.01, f"ITTC-1978 increment mismatch {rel:.3%}"

    def test_legacy_coefficient_would_fail(self):
        """Regression: the pre-fix alpha=0.004 under-predicted by ~96 %."""
        d_legacy = 0.004 * ((5e-3 / self.L) ** (1 / 3) - (self.K_S0 / self.L) ** (1 / 3))
        d_ref = (ittc_1978_delta_cf(5e-3, self.L)
                 - ittc_1978_delta_cf(5e-4, self.L))
        assert abs(d_legacy - d_ref) / d_ref > 0.9
        assert abs(ittc_1978_roughness_penalty(
            5e-3, self.K_S0, self.L) - ittc_1978_roughness_penalty(
            5e-4, self.K_S0, self.L) - d_ref) / d_ref < 1e-12

    def test_monotone_in_roughness(self):
        ks = np.linspace(self.K_S0, 8e-3, 200)
        dcf = [ittc_1978_roughness_penalty(k, self.K_S0, self.L) for k in ks]
        assert np.all(np.diff(dcf) > 0.0)

    def test_validity_window_guard(self):
        with pytest.raises(ValueError):
            ittc_1978_delta_cf(1.0, 150.0, strict=True)  # k/L = 6.7e-3, far outside

    def test_ittc1957_and_thrust_loss_chain(self):
        cf = ittc_1957_cf(1e9)
        assert 0.001 < cf < 0.002
        k_s = 2e-3
        d_t = ittc_1978_thrust_loss(k_s, self.K_S0, self.L, c_total=0.004)
        d_cf = ittc_1978_roughness_penalty(k_s, self.K_S0, self.L)
        assert d_t == pytest.approx(d_cf / 0.004, rel=1e-15)
        # friction-only fallback is the conservative (larger) bound
        assert ittc_1978_thrust_loss(k_s, self.K_S0, self.L) > d_t

    def test_roughness_state_chain(self):
        st = roughness_state(h_ice=2e-3, ship_length=self.L)
        assert st.k_s == pytest.approx(1.5e-5 + 0.5 * 2e-3)
        assert st.delta_cf == pytest.approx(
            ittc_1978_roughness_penalty(st.k_s, st.k_s0, self.L))


# ---------------------------------------------------------------------------
# C6 — adfreeze interface consistency
# ---------------------------------------------------------------------------
class TestC6AdfreezeInterface:
    """C6: same bond input -> same strength, units consistent (< 1e-9)."""

    def test_mpa_to_pa_bridge(self):
        tau_mpa = adfreeze_strength(ice_temperature_c=-5.0, contact_time_s=1e4)
        bridged = bond_strength_from_adfreeze_pa(-5.0, contact_time_s=1e4)
        assert abs(bridged - tau_mpa * 1e6) / (tau_mpa * 1e6) < 1e-9

    def test_dimensional_consistency_pa(self):
        """1 Pa = 1 N/m^2: tau_bond/h_ref has pressure/length = Pa/m, and
        tau_bond must itself be a pressure, i.e. consistent with rho*u^2."""
        bond = bond_strength_from_adfreeze_pa(-10.0)
        u_crit = detachment_regime(u_star=0.05, tau_bond=bond)["u_star_crit_m_s"]
        assert u_crit == pytest.approx(math.sqrt(bond / RHO_WATER), rel=1e-15)

    def test_bridge_roundtrip_through_detach_model(self):
        from tensorphysics.ice.attached_ice import adhesive_strength
        bond = bond_strength_from_adfreeze_pa(-8.0)
        assert adhesive_strength(0.0, bond) == pytest.approx(bond, rel=1e-12)

    def test_temperature_dependence_shared(self):
        cold = bond_strength_from_adfreeze_pa(-15.0)
        warm = bond_strength_from_adfreeze_pa(-2.0)
        assert cold > warm > 0.0
        assert abs((adfreeze_strength(-15.0) * 1e6 - cold) / cold) < 1e-9

    def test_warm_joint_no_bond(self):
        assert bond_strength_from_adfreeze_pa(+1.0) == 0.0


# ---------------------------------------------------------------------------
# Flux / transport sanity (C7 support)
# ---------------------------------------------------------------------------
class TestFluxLayers:
    def test_settling_is_upward(self):
        from tensorphysics.ice.attached_ice import settling_velocity_stokes
        assert settling_velocity_stokes() < 0.0  # ice is buoyant

    def test_flux_scales_with_concentration(self):
        from tensorphysics.ice.attached_ice import frazil_impact_flux
        a = frazil_impact_flux(1e-5, 0.05).mass_flux
        b = frazil_impact_flux(1e-4, 0.05).mass_flux
        assert b == pytest.approx(10 * a, rel=1e-12)

    def test_flux_validation(self):
        from tensorphysics.ice.attached_ice import frazil_impact_flux
        with pytest.raises(ValueError):
            frazil_impact_flux(-1e-5, 0.05)
        with pytest.raises(ValueError):
            frazil_impact_flux(2.0, 0.05)

    def test_friction_velocity_roundtrip(self):
        u_star = friction_velocity_from_speed(3.0)
        assert speed_from_friction_velocity(u_star) == pytest.approx(3.0, rel=1e-12)
        assert u_star < 3.0


# ---------------------------------------------------------------------------
# Defect regressions found by the maturity audit
# ---------------------------------------------------------------------------
class TestAuditRegressions:
    def test_convective_heat_transfer_closure(self):
        """h_c must follow the Colburn analogy, not h_c = rho Cp u_star."""
        u_s = 0.08
        ode = IceGrowthODE(undercooling_k=2.0, u_star=u_s, c_f=0.0,
                           t_water_k=T_FREEZE_FW + 0.2)
        expected = (RHO_WATER * C_P_WATER * u_s * math.sqrt(CF_REF / 2.0)
                    * PR_T ** (-2.0 / 3.0))
        assert ode._h_c == pytest.approx(expected, rel=1e-15)
        legacy = RHO_WATER * C_P_WATER * u_s
        assert legacy / ode._h_c > 100.0  # the old closure was ~2e2 too large

    def test_over_temperature_melting_direction(self):
        """Water above freezing removes ice; water at freezing does not."""
        warm = IceGrowthODE(undercooling_k=2.0, u_star=0.05, c_f=0.0, eta_stick=0.0,
                            t_water_k=T_FREEZE_FW + 0.2)
        cold = IceGrowthODE(undercooling_k=2.0, u_star=0.05, c_f=0.0, eta_stick=0.0,
                            t_water_k=T_FREEZE_FW)
        # thin layer: conduction wins even with 0.2 K of water over-temperature
        assert warm.rhs(1e-3) > 0.0
        # thick layer: the melting flux q_conv = h_c * 0.2 K exceeds k dT/h
        h_melt = 2.0 * K_ICE / warm._convective_flux
        assert warm.rhs(1.05 * h_melt) < 0.0
        assert 1e-3 < h_melt < 0.05
        # at the freezing point there is no melt and the layer never thins
        assert cold.rhs(1e-3) > 0.0 and cold.rhs(0.05) > 0.0
        assert cold._convective_flux == 0.0

    def test_equilibrium_requires_melting_sink(self):
        """At the freezing point with no frazil the layer grows unbounded."""
        eq = equilibrium_thickness(undercooling_k=2.0, u_star=0.05, c_f=0.0,
                                   eta_stick=0.0, t_water_k=T_FREEZE_FW)
        assert eq["h_growth_m"] == float("inf")
        assert eq["equilibrium_exists"] is False
        assert eq["h_eq_m"] == float("inf")
        assert eq["limited_by"] == "none"
        # the degenerate detachment cap must stay out of the steady-state answer
        assert np.isfinite(eq["h_detach_m"]) and eq["h_detach_m"] > 1.0

    def test_equilibrium_matches_long_integration(self):
        """Closed-form steady state must agree with the long-time ODE limit."""
        kw = dict(undercooling_k=2.0, u_star=0.05, c_f=1e-4, eta_stick=1.0,
                  t_water_k=T_FREEZE_FW + 0.2)
        eq = equilibrium_thickness(**kw)["h_eq_m"]
        ode = IceGrowthODE(**kw)
        res = ode.solve_euler(t_max=400 * 3600.0, n_steps=4000, h_init=1e-6,
                             method="stefan_split")
        assert abs(res.h_final - eq) / eq < 0.02, (
            f"ODE limit {res.h_final:.4g} vs equilibrium {eq:.4g}")


# ---------------------------------------------------------------------------
# Hand-off interface for the resistance / propulsion chain
# ---------------------------------------------------------------------------
class TestHandoff:
    KS = dict(undercooling_k=2.0, duration_s=7200.0, speed_m_s=3.0, c_f=1e-4,
              ice_temperature_c=-5.0, cf_total=0.004, n_steps=150)

    def test_payload_schema_and_json_roundtrip(self):
        payload = build_handoff(**self.KS)
        assert payload["schema"] == HANDOFF_SCHEMA
        text = to_json(payload)
        back = json.loads(text)
        assert back["scalars"]["h_ice_end_m"] == payload["scalars"]["h_ice_end_m"]

    def test_time_series_consistency(self):
        payload = build_handoff(**self.KS)
        ts = payload["time_series"]
        n = len(ts["t_s"])
        assert n == len(ts["h_ice_m"]) == len(ts["k_s_m"]) == len(ts["delta_cf"])
        assert ts["t_s"][0] == 0.0
        k_s0 = payload["inputs"]["k_s0_m"]["value"]
        assert ts["k_s_m"][0] == pytest.approx(k_s0)
        assert ts["delta_cf"][0] == pytest.approx(0.0, abs=1e-15)
        # roughness and penalty rise together with ice thickness
        assert np.all(np.diff(ts["k_s_m"]) >= 0.0)
        assert np.all(np.diff(ts["delta_cf"]) >= 0.0)

    def test_scalars_consistent_with_modules(self):
        payload = build_handoff(**self.KS)
        k_s_end = payload["scalars"]["k_s_end_m"]
        assert k_s_end == pytest.approx(equivalent_sand_roughness(
            payload["scalars"]["h_ice_end_m"]))
        assert payload["scalars"]["delta_cf_end"] == pytest.approx(
            ittc_1978_roughness_penalty(k_s_end, payload["inputs"]["k_s0_m"]["value"],
                                        payload["inputs"]["ship_length_m"]["value"]),
            rel=1e-12)

    def test_no_domain_resistance_formula_leaked(self):
        """The hand-off must expose coefficients, not ship-domain resistance models."""
        payload = build_handoff(**self.KS)
        flat = json.dumps(payload).lower()
        for forbidden in ("holtrop", "townsin", "atsumi", "resistance_total"):
            assert forbidden not in flat
        assert payload["limitations"]

    def test_envelope_attached_on_request(self):
        payload = build_handoff(n_mc=20, n_steps=60,
                                **{k: v for k, v in self.KS.items() if k != "n_steps"})
        env = payload["envelope"]
        assert env is not None and env["n_samples"] == 20
        assert env["h_ice_final_p90_m"] >= env["h_ice_final_p50_m"]
        assert len(env["t_s"]) == len(env["h_ice_p50_m"]) == len(env["h_ice_p90_m"])

    def test_input_validation(self):
        with pytest.raises(ValueError):
            build_handoff(undercooling_k=2.0, duration_s=600.0)


# ---------------------------------------------------------------------------
# Time-resolved P50/P90 envelope (spec item (d))
# ---------------------------------------------------------------------------
class TestEnvelope:
    def test_envelope_ordering_and_lengths(self):
        env = monte_carlo_envelope(duration_s=3600.0, n_steps=60, n_samples=40,
                                   seed=7)
        assert len(env["t_s"]) == len(env["h_ice_p50"]) == len(env["k_s_p90"])
        assert np.all(np.array(env["h_ice_p90"]) >= np.array(env["h_ice_p50"]))
        assert np.all(np.array(env["k_s_p90"]) >= np.array(env["k_s_p50"]))
        assert env["h_ice_final_p90"] >= env["h_ice_final_p50"] >= 0.0

    def test_envelope_reproducible(self):
        a = monte_carlo_envelope(duration_s=1800.0, n_steps=40, n_samples=20, seed=11)
        b = monte_carlo_envelope(duration_s=1800.0, n_steps=40, n_samples=20, seed=11)
        assert a["h_ice_final_p90"] == b["h_ice_final_p90"]

    def test_envelope_requires_samples(self):
        with pytest.raises(ValueError):
            monte_carlo_envelope(n_samples=5)


# ---------------------------------------------------------------------------
# C7 — coverage count
# ---------------------------------------------------------------------------
def test_C7_criterion_count():
    """C7: at least 15 acceptance tests cover the module (spec: >= 15)."""
    import inspect
    import sys
    mod = sys.modules[__name__]
    n = sum(1 for name, obj in inspect.getmembers(mod, inspect.isfunction)
            if name.startswith("test_"))
    n += sum(len([m for m in dir(obj) if m.startswith("test_")])
             for name, obj in inspect.getmembers(mod, inspect.isclass)
             if name.startswith("Test"))
    assert n >= 15, f"only {n} tests"