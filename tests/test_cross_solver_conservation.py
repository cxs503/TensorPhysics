"""Tests for the cross-solver (FEM -> DEM -> LBM) conservation audit."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tensorphysics.cross_solver_conservation import audit as audit_mod
from tensorphysics.cross_solver_conservation import cohesive_chain as cc
from tensorphysics.cross_solver_conservation import fem_beam as fb
from tensorphysics.cross_solver_conservation import lbm_wall as lbm
from tensorphysics.cross_solver_conservation import multirate as mr
from tensorphysics.cross_solver_conservation import criteria as crit

SECTION = fb.BeamSection(E=9e9, b=0.10, h=0.005)
L = 0.50
SIGMA_F = 0.7e6
G_F = 0.08


def _beam(load_factor=1.5, n_elem=200):
    P_f = fb.flexural_strength_load(SECTION, L, SIGMA_F)
    return P_f, fb.solve_cantilever(SECTION, L, load_factor * P_f, n_elem=n_elem)


# ----------------------------------------------------------------- FEM stage
def test_cantilever_matches_euler_bernoulli():
    _, sol = _beam(load_factor=1.0)
    assert sol.tip_deflection == pytest.approx(sol.analytic_tip_deflection(), rel=1e-6)
    assert sol.U == pytest.approx(sol.analytic_strain_energy(), rel=1e-6)


def test_strength_load_reaches_sigma_f():
    P_f, sol = _beam(load_factor=1.0)
    assert sol.max_stress == pytest.approx(SIGMA_F, rel=1e-4)
    assert sol.M_root == pytest.approx(P_f * L, rel=1e-6)


def test_failure_face_traction_profile_is_antisymmetric():
    _, sol = _beam()
    face = fb.extract_failure_face(sol)
    t = face.traction
    # bending traction: tension on one face, compression on the other
    assert t[0] < 0 < t[-1]
    assert t[0] == pytest.approx(-t[-1], rel=1e-12)
    # peak traction = M/Z at the extreme fibre
    assert abs(t[-1]) == pytest.approx(sol.M_root / SECTION.section_modulus, rel=1e-6)
    assert face.area == pytest.approx(SECTION.A)
    assert face.energy_release_rate > 0


def test_energy_release_rate_matches_strength_limit():
    g_strength = SIGMA_F**2 * SECTION.h / (6 * SECTION.E)
    _, sol = _beam(load_factor=1.0)
    assert sol.energy_release_rate == pytest.approx(g_strength, rel=1e-6)


# ------------------------------------------------------------------ DEM stage
def test_bilinear_cohesive_law_energy_identity():
    law = cc.CohesiveHingeLaw(
        k_theta=SECTION.EI / 0.001,
        M_peak=SIGMA_F * SECTION.I / (SECTION.h / 2),
        G_f=G_F,
        b=SECTION.b,
        l_b=0.001,
    )
    th = np.linspace(0.0, 1.2 * law.theta_c, 57)
    work = law.work_to(th)
    stored = 0.5 * (1.0 - law.damage(th)) * law.k_theta * th**2
    diss = law.dissipated(th)
    assert np.allclose(work, stored + diss, rtol=1e-12, atol=1e-18)
    # a fully broken hinge dissipates exactly G_f * crack area
    assert law.dissipated(np.array([law.theta_c]))[0] == pytest.approx(law.phi_f, rel=1e-12)


def test_chain_initialisation_maps_fem_energy():
    _, sol = _beam()
    state, info = cc.bonded_chain_from_beam(sol, SECTION, SIGMA_F, G_F, 500)
    assert info["mapping_rel_error"] < 5e-3
    assert info["peak_exceeded"] is True
    assert info["l_b_over_process_zone"] < 1.0  # process zone spans >1 bond


def test_chain_energy_ledger_closes():
    _, sol = _beam()
    state, info = cc.bonded_chain_from_beam(sol, SECTION, SIGMA_F, G_F, 500)
    law = state.law
    dt = 0.1 * law.l_b**2 / 72.4
    led = cc.run_chain(state, dt=dt, t_end=5e-6, zeta=0.02, sample_every=0)
    assert led["closure_rel_error"] < 1e-3
    assert led["diss_path_vs_analytic_rel"] < 1e-2
    assert led["n_broken"] > 0
    assert 0.0 < led["crack_front"] < L / 2
    # a fully-broken hinge dissipates at least Griffith's energy
    assert led["E_diss"] >= G_F * SECTION.b * led["crack_front"] - 1e-12


def test_crack_length_is_mesh_insensitive():
    _, sol = _beam()
    lengths = []
    for nb in (500, 1000):
        state, _ = cc.bonded_chain_from_beam(sol, SECTION, SIGMA_F, G_F, nb)
        law = state.law
        dt = 0.1 * law.l_b**2 / 72.4
        led = cc.run_chain(state, dt=dt, t_end=1e-5, zeta=0.02, sample_every=0)
        lengths.append(led["crack_front"])
    assert (max(lengths) - min(lengths)) / np.mean(lengths) < 0.10


# ------------------------------------------------------------------ LBM stage
def test_channel_poiseuille_and_wall_momentum_identity():
    out = lbm.run_channel(nx=8, ny=24, tau=0.8, force=1e-5, n_warm=4000, n_avg=1000)
    assert out["profile_rel_error"] < 0.03
    # discrete momentum exchange of the two walls equals the momentum injected by
    # the body force (rho g h), i.e. the boundary scheme is momentum consistent
    assert abs(
        (out["tau_top_ladd"] + out["tau_bottom_ladd"]) / (2 * out["tau_analytic"]) - 1.0
    ) < 0.02
    assert out["balance_error_rel"] < 1e-8
    # macroscopic wall shear agrees with the discrete exchange
    assert abs(out["tau_top_shear"] / out["tau_top_ladd"] - 1.0) < 0.03


def test_ice_wall_momentum_closure():
    out = lbm.run_ice_wall_coupling(
        nx=8, ny=24, tau=0.8, force=0.0, n_warm=1000, n_couple=800,
        m_ice=500.0, v_ice0=0.02,
    )
    assert out["dp_ice"] < 0.0  # the ice is decelerated by the quiescent fluid
    assert out["closure_ice_vs_ladd"] < 0.02
    assert out["closure_shear_vs_ladd"] < 0.02
    assert out["balance_error_rel"] < 1e-8


# ------------------------------------------------------------- multirate stage
def test_multirate_orders_and_aliasing():
    sweep = mr.multirate_sweep(
        t_end=0.02,
        rates_hz={"r1000": 1000.0, "r3000": 3000.0, "r5000": 5000.0, "r10000": 10000.0},
    )
    curves = dict(sweep["scheme_curves"])

    def _resolved(curve):
        rates, errs = curve
        keep = [(r, e) for r, e in zip(rates, errs) if r >= 3000]
        return [r for r, _ in keep], [e for _, e in keep]

    lin = mr.observed_order(*_resolved(curves["linear"]))
    cub = mr.observed_order(*_resolved(curves["cubic"]))
    assert 1.2 < lin < 2.6
    assert cub > 2.5
    assert cub > lin
    # 1 kHz cannot carry the 2.6 kHz component: error is O(1), not small
    err_1k = next(
        r for r in sweep["results"] if r["rate_hz"] == 1000.0 and r["scheme"] == "cubic"
    )
    assert err_1k["l2_rel"] > 0.3


def test_multirate_exchange_clocks():
    assert mr.RATES_HZ == {"fem": 100.0, "dem": 1000.0, "lbm": 10000.0}
    assert mr.RATE_AUDIT[0]["ratio"] == pytest.approx(10.0)
    assert mr.RATE_AUDIT[1]["ratio"] == pytest.approx(10.0)


# ------------------------------------------------------------------- criteria
def test_criteria_are_frozen_and_evaluable():
    assert len(crit.CRITERIA) == 12
    assert len(crit.CRITERIA_DIGEST) == 64
    outcomes = crit.evaluate({"closure_rel_error": 1e-3, "closure_ice_vs_ladd": 0.5})
    by = {o["cid"]: o for o in outcomes}
    assert by["C2_fem_dem_energy_closure"]["status"] == "PASS"
    assert by["C5_dem_lbm_momentum_closure"]["status"] == "FAIL"
    assert by["C8_lbm_poiseuille_profile"]["status"] == "WITHHELD"


# ------------------------------------------------------------- end-to-end audit
def test_audit_end_to_end(tmp_path):
    case = dict(audit_mod.CASE)
    case["dens_mesh"] = [500]
    case["lbm"] = dict(case["lbm"], n_warm=800, n_couple=600, m_ice=[500.0])
    audit = audit_mod.run_audit(case=case, write=True, out_dir=str(tmp_path))
    assert audit["verdict"] in {"PASS", "FAIL", "WITHHELD"}
    assert audit["metrics"]["closure_rel_error"] < 0.01
    assert audit["metrics"]["closure_ice_vs_ladd"] < 0.02
    metrics = audit["metrics"]
    for key in (
        "mapping_rel_error",
        "closure_rel_error",
        "closure_ice_vs_ladd",
        "chain_closure_rel",
        "work_rel_cubic",
        "impulse_rel_cubic",
        "crack_length_spread_across_mesh",
    ):
        assert key in metrics
    stored = json.loads((tmp_path / "audit.json").read_text())
    assert stored["criteria_digest"] == crit.CRITERIA_DIGEST
    assert (tmp_path / "failure_faces.json").exists()
    assert len(stored["stage_dem"]["runs"]) == 1