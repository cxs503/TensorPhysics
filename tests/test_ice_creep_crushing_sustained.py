"""Acceptance tests for P269W2-N05: rate-dependent creep crushing (A1-A7).

Criteria
--------
A1 constant-load creep closed form ``delta(t) = delta0 t**q`` reproduced to
   <= 1 % residual for q = 2, 3, 4
A2 pressure-rate exponent ``m`` within 5 % of ``1/(n+1)`` (n = 3 -> 0.25)
A3 rigid limit (``k_s -> inf``) within 3 % of the constant-rate closed form
A4 relaxation ``p = p0 exp(-t/tau)``: fit R^2 >= 0.99, ``tau`` within 5 % of
   the analytic ``eta/E_eff``
A5 energy ledger ``W = U + D`` residual <= 1 %
A6 no-creep negative control: elastic ramp and Hertz contact within 2 %
A7 >= 12 test cases (this file has 21)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorphysics.ice.creep_crushing_sustained import (
    CrushingConfig,
    NortonBailey,
    constant_load_creep,
    creep_velocity,
    energy_ledger,
    fit_power_law,
    fit_relaxation,
    hertz_delta,
    hertz_force,
    hertz_indentation_force,
    relaxation_tau,
    scaling_exponent,
    simulate_crushing,
    simulate_relaxation,
    steady_pressure_closed_form,
    sustained_design_pressure,
    theoretical_rate_exponent,
    zone_length,
)

B_ICE = 1.0e-23          # Pa^-3 s^-1 at -10 C
SIGMA_REF = 5.0e6        # Pa
H_ICE = 0.5              # m
K_S = 1.0e8              # N/m
A_CONTACT = 0.1          # m^2


def _cfg(**kw) -> CrushingConfig:
    base = dict(v=1.0e-4, k_s=K_S, A=A_CONTACT, h=H_ICE,
                sigma_ref=SIGMA_REF, sigma_f=50.0e6,
                creep=NortonBailey(B_ref=B_ICE, n=3.0),
                t_end=4000.0, n_steps=2000)
    base.update(kw)
    return CrushingConfig(**base)


# ---------------------------------------------------------------- A1
@pytest.mark.parametrize("q", [2.0, 3.0, 4.0])
def test_a1_constant_load_creep_power_law(q):
    """A1: constant-load creep is a pure t**q power law (residual <= 1 %)."""
    creep = NortonBailey(B_ref=B_ICE, n=3.0, q=q)
    t = np.geomspace(1.0e-3, 1.0e3, 400)
    delta = constant_load_creep(t, 1.0e6, creep, H_ICE)
    exp_fit, amp, rel = fit_power_law(t, delta)
    assert rel <= 0.01, f"power-law residual {rel:.2e} exceeds 1%"
    assert abs(exp_fit - q) <= 1e-6
    assert amp > 0.0


def test_a1_creep_is_proportional_to_sigma_power_n():
    """A1: the creep amplitude scales as sigma**n with n = 3."""
    creep = NortonBailey(B_ref=B_ICE, n=3.0)
    t = np.linspace(1.0, 100.0, 10)
    d1 = constant_load_creep(t, 1.0e6, creep, H_ICE)
    d2 = constant_load_creep(t, 2.0e6, creep, H_ICE)
    assert np.allclose(d2 / d1, 8.0, rtol=1e-12)      # 2**3


# ---------------------------------------------------------------- A2
def test_a2_pressure_rate_exponent_matches_theory():
    """A2: m = 1/(n+1) = 0.25 for n = 3 within 5 %."""
    cfg = _cfg()
    m, rows = scaling_exponent(cfg)
    m_theory = theoretical_rate_exponent(3.0, "pressure_scaled")
    assert m_theory == pytest.approx(0.25)
    assert abs(m - m_theory) / m_theory <= 0.05, f"m={m:.4f} vs {m_theory:.4f}"
    assert all(r["p_sustained"] > 0 for r in rows)
    # p_sustained must track the analytic steady state for every rate
    for r in rows:
        assert abs(r["p_sustained"] / r["p_steady_analytic"] - 1.0) <= 0.01


def test_a2_constant_zone_exponent_is_inverse_n():
    """A2 contrast: with a constant process zone the exponent is 1/n = 1/3."""
    cfg = _cfg(zone_model="constant")
    m, _ = scaling_exponent(cfg)
    assert abs(m - 1.0 / 3.0) / (1.0 / 3.0) <= 0.05


def test_a2_scaling_is_insensitive_to_time_window():
    """A2: the fitted exponent is a steady-state property (<= 5 % drift)."""
    m1, _ = scaling_exponent(_cfg(t_end=2000.0, n_steps=1000))
    m2, _ = scaling_exponent(_cfg(t_end=8000.0, n_steps=2000))
    assert abs(m1 - m2) / m2 <= 0.05


# ---------------------------------------------------------------- A3
@pytest.mark.parametrize("zone_model", ["pressure_scaled", "constant"])
def test_a3_rigid_limit_matches_constant_rate_closed_form(zone_model):
    """A3: k_s -> inf reproduces the constant-rate crushing closed form."""
    cfg = _cfg(k_s=float("inf"), v=1.0e-5, zone_model=zone_model)
    res = simulate_crushing(cfg)
    closed = steady_pressure_closed_form(1.0e-5, cfg)
    assert abs(res.p_sustained / closed - 1.0) <= 0.03
    # analytic value of the closed form itself
    if zone_model == "pressure_scaled":
        assert closed == pytest.approx(
            (1.0e-5 * SIGMA_REF / (H_ICE * B_ICE)) ** 0.25, rel=1e-12)


def test_a3_stiff_structure_approaches_rigid_limit():
    """A3: increasing k_s drives p_sustained monotonically to the rigid value."""
    rigid = simulate_crushing(_cfg(k_s=float("inf"), v=1.0e-5)).p_sustained
    p_soft = simulate_crushing(_cfg(k_s=1.0e7, v=1.0e-5)).p_sustained
    p_stiff = simulate_crushing(_cfg(k_s=1.0e11, v=1.0e-5)).p_sustained
    assert p_soft < p_stiff <= rigid * 1.0001
    assert abs(p_stiff / rigid - 1.0) <= 0.03


# ---------------------------------------------------------------- A4
def test_a4_relaxation_exponential_and_tau():
    """A4: p = p0 exp(-t/tau), R^2 >= 0.99, tau vs eta/E_eff within 5 %."""
    eta, k_s, A, L_c = 1.0e9, K_S, A_CONTACT, 0.1
    E_eff = k_s * L_c / A
    tau_analytic = eta / E_eff
    t, p = simulate_relaxation(1.0e5, eta, E_eff, t_end=6.0 * tau_analytic,
                               n_steps=6000)
    tau_fit, r2 = fit_relaxation(t, p, 1.0e5)
    assert r2 >= 0.99
    assert abs(tau_fit - tau_analytic) / tau_analytic <= 0.05
    assert tau_fit > 0.0


def test_a4_tau_equals_eta_over_e_eff():
    """A4: the linearised constant is tau = A/(k_s L_c B) = eta/E_eff (n = 1)."""
    eta = 1.0e9
    tau = relaxation_tau(K_S, A_CONTACT, 0.1, 1.0 / eta, 1.0, 1.0e5)
    assert tau == pytest.approx(eta * A_CONTACT / (K_S * 0.1), rel=1e-12)
    # non-linear branch reduces to the same expression at n = 1
    assert relaxation_tau(K_S, A_CONTACT, 0.1, 1.0 / eta, 1.0, 1.0e7) == \
        pytest.approx(tau, rel=1e-12)


def test_a4_relaxation_is_monotone_decaying():
    """A4: relaxation pressure decreases monotonically towards zero."""
    t, p = simulate_relaxation(1.0e5, 1.0e9, 1.0e7, t_end=300.0, n_steps=2000)
    assert np.all(np.diff(p) < 0)
    assert p[-1] < 0.05 * p[0]


# ---------------------------------------------------------------- A5
def test_a5_energy_ledger_closes():
    """A5: external work = elastic energy + creep dissipation (<= 1 %)."""
    res = simulate_crushing(_cfg(v=1.0e-4, t_end=200.0, n_steps=4000))
    led = energy_ledger(res)
    assert led["W_external_J"] > 0
    assert led["D_creep_J"] > 0
    assert led["residual_rel"] <= 0.01, led
    # energy split must be physically sensible
    assert 0.0 < led["U_end_J"] < led["W_external_J"]


def test_a5_energy_residual_decreases_with_timestep():
    """A5: the ledger closes as dt -> 0 (second-order behaviour)."""
    r_coarse = energy_ledger(
        simulate_crushing(_cfg(v=1.0e-4, t_end=200.0, n_steps=500)))
    r_fine = energy_ledger(
        simulate_crushing(_cfg(v=1.0e-4, t_end=200.0, n_steps=4000)))
    assert r_fine["residual_rel"] < r_coarse["residual_rel"]
    assert r_fine["residual_rel"] <= 0.01


# ---------------------------------------------------------------- A6
def test_a6_no_creep_reduces_to_elastic_contact():
    """A6: B -> 0 gives the elastic ramp p = k_s v t / A (<= 2 %)."""
    cfg = _cfg(creep=NortonBailey(B_ref=0.0, n=3.0), v=1.0e-4,
               t_end=10.0, n_steps=1000)
    res = simulate_crushing(cfg)
    p_analytic = K_S * cfg.v * res.t / A_CONTACT
    mask = res.t > 0
    rel = np.abs(res.p[mask] - p_analytic[mask]) / p_analytic[mask]
    assert rel.max() <= 0.02, rel.max()
    # the final pressure (not the tail mean of a ramp) matches the closed form
    assert res.p[-1] / p_analytic[-1] == pytest.approx(1.0, rel=0.02)


def test_a6_hertz_contact_limit():
    """A6: Hertz contact force matches F = (4/3) E* sqrt(R) (v t)^{3/2} (<= 2 %)."""
    E_star, R, v = 9.0e9, 0.5, 1.0e-3
    t = np.linspace(1.0e-4, 1.0, 60)
    F_num = hertz_indentation_force(v, t, E_star, R)
    F_ana = hertz_force(v * t, E_star, R)
    assert np.max(np.abs(F_num - F_ana) / F_ana) <= 0.02
    # the two Hertz relations are exact inverses
    d = np.linspace(1e-4, 1e-1, 20)
    assert np.allclose(hertz_delta(hertz_force(d, E_star, R), E_star, R), d,
                       rtol=1e-10)


# ---------------------------------------------------------------- extras
def test_a7_zero_velocity_stays_unloaded():
    """Limit: no indentation -> no crushing pressure."""
    res = simulate_crushing(_cfg(v=0.0, t_end=10.0, n_steps=100))
    assert np.allclose(res.p, 0.0)
    assert res.p_sustained == 0.0


def test_brittle_ceiling_clamps_pressure():
    """Brittle cap: p_peak equals sigma_f once the cap is active."""
    res = simulate_crushing(_cfg(v=1.0e-3, sigma_f=3.0e6, t_end=200.0,
                                 n_steps=1000))
    assert res.crushing.any()
    assert res.p_peak <= 3.0e6 * (1.0 + 1e-12)
    assert res.p_sustained <= 3.0e6 * (1.0 + 1e-12)


def test_sustained_pressure_increases_with_rate():
    """Monotonic rate dependence: faster indentation -> higher sustained load."""
    p = [simulate_crushing(_cfg(v=v)).p_sustained
         for v in (1.0e-6, 1.0e-5, 1.0e-4)]
    assert p[0] < p[1] < p[2]


def test_zone_length_models():
    """Process-zone length: constant = h, pressure-scaled -> h * p/sigma_ref."""
    cfg_c = _cfg(zone_model="constant")
    assert np.allclose(zone_length(cfg_c, np.array([1e6, 1e7])), H_ICE)
    cfg_p = _cfg()
    assert np.allclose(zone_length(cfg_p, np.array([1e6, 2.5e6])),
                       [H_ICE * 0.2, H_ICE * 0.5])
    assert np.all(zone_length(cfg_p, np.array([1e9])) <= H_ICE)


def test_arrhenius_colder_ice_creeps_slower():
    """Temperature coupling: B decreases with lower temperature."""
    creep = NortonBailey(B_ref=B_ICE, n=3.0)
    assert creep.B_at(263.15) == pytest.approx(B_ICE, rel=1e-12)
    assert creep.B_at(253.15) < creep.B_at(263.15) < creep.B_at(268.15)


def test_design_pressure_interface():
    """Interface for the m7257 chain: keys, ordering and clamping flag."""
    out = sustained_design_pressure(_cfg())
    for key in ("p_peak_Pa", "p_sustained_Pa", "p_steady_analytic_Pa",
                "tau_s", "m", "m_theory", "clamped"):
        assert key in out
    assert 0.0 < out["p_sustained_Pa"] <= out["p_peak_Pa"] * (1.0 + 1e-9)
    assert out["m_theory"] == pytest.approx(0.25)
    assert out["clamped"] is False


def test_creep_velocity_consistency():
    """G(p) is the creep branch of the crushing ODE and of the closed form."""
    cfg = _cfg(v=1.0e-4)
    res = simulate_crushing(cfg)
    # at the sustained state the creep velocity must equal the imposed rate
    assert res.v_eff[-1] == pytest.approx(cfg.v, rel=2e-2)
    assert math.isfinite(res.tau) and res.tau > 0