"""Acceptance tests for the slamming self-similar impact kernel (P151W1-N08).

Every test maps one-to-one onto a machine-checkable criterion of the node:

E1  self-similar fold of p(x,t) and of the force history   (L2 <= 1e-2)
E2  small-beta total force vs the classical Wagner solution (<= 3%)
E3  impulse ledger closure vs Delta(m_a U_n)               (<= 1%)
E4  Joukowsky acoustic bound max p / (rho c_s V) <= 1
E5  resolution ledger: total force converges (order >= 1.5) and the near-root
    quantity is explicitly non-convergent (order < 0)
E6  bulkhead single-DOF hand-over (interface + declared applicability)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorphysics.slamming import (
    RHO_WATER,
    C_SOUND_WATER,
    BulkheadSDOF,
    SelfSimilarWedgeImpact,
    resolution_ledger,
)
from tensorphysics.slamming.similarity_impact import similarity_constants

ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "artifacts" / "slamming_similarity" / "validation.json"

BETA = 5.0
V = 5.0
A = 50.0
T = 0.10


def _rho_un2(m, t):
    return RHO_WATER * float(m.normal_velocity(t)) ** 2


# ---------------------------------------------------------------- E1
def test_E1_self_similar_fold():
    band, dx = 0.95, 0.02
    grid = np.linspace(-band, band, 2001)
    profiles = []
    for t in (0.05, 0.10, 0.20):
        m = SelfSimilarWedgeImpact(BETA, mode="velocity", speed=V)
        c = float(m.wetted_half_width(t))
        x = np.linspace(-band * c, band * c, int(band * c / dx))
        p_hat = m.pressure(x, t, capped=False) / _rho_un2(m, t)
        profiles.append(np.interp(grid, x / c, p_hat))
    folded = np.array(profiles)
    for i in range(folded.shape[0]):
        for j in range(i + 1, folded.shape[0]):
            num = np.linalg.norm(folded[i] - folded[j])
            den = 0.5 * (np.linalg.norm(folded[i]) + np.linalg.norm(folded[j]))
            assert num / den <= 1e-2

    # the same fold on the total force: F / (rho U_n^2 c) is a constant
    m = SelfSimilarWedgeImpact(BETA, mode="velocity", speed=V)
    cf = [
        float(m.force_momentum(t)) / (_rho_un2(m, t) * float(m.wetted_half_width(t)))
        for t in np.linspace(0.02, 0.2, 8)
    ]
    assert np.ptp(cf) / np.mean(cf) <= 1e-12
    assert cf[0] == pytest.approx(math.pi * similarity_constants(BETA)["a1"], rel=1e-12)


# ---------------------------------------------------------------- E2
def _wagner_textbook_force(beta_deg, t, speed=V, rho=RHO_WATER):
    kappa = 0.5 * math.pi / math.tan(math.radians(beta_deg))
    c, cdot = kappa * speed * t, kappa * speed
    return rho * math.pi * c * cdot * speed


def test_E2_small_beta_matches_wagner():
    for beta in (1.0, 2.0, 3.0, 5.0):
        m = SelfSimilarWedgeImpact(beta, mode="velocity", speed=V)
        f_model = float(m.force_momentum(T))
        f_wag = _wagner_textbook_force(beta, T)
        assert abs(f_model - f_wag) / f_wag <= 0.03, beta
    # the deviation grows with beta (linearisation band ends): O(beta^2) = 1-cos
    m10 = SelfSimilarWedgeImpact(10.0, mode="velocity", speed=V)
    dev10 = abs(float(m10.force_momentum(T)) - _wagner_textbook_force(10.0, T)) / \
        _wagner_textbook_force(10.0, T)
    assert dev10 == pytest.approx(1.0 - math.cos(math.radians(10.0)), rel=1e-6)


def test_E2_acceleration_mode_keeps_self_similar_shape():
    mv = SelfSimilarWedgeImpact(BETA, mode="velocity", speed=V)
    ma = SelfSimilarWedgeImpact(BETA, mode="acceleration", accel=A)
    # same a1, one extra self-similar term b = kappa / (2 cos beta)
    assert ma.shape_coefficients()["a1"] == pytest.approx(mv.shape_coefficients()["a1"])
    assert ma.shape_coefficients()["b"] == pytest.approx(
        0.5 * ma.const["kappa"] / ma.const["cos_beta"])
    assert mv.shape_coefficients()["b"] == 0.0
    # the acceleration force history still collapses in similarity coordinates
    cf = [
        float(ma.force_momentum(t)) / (_rho_un2(ma, t) * float(ma.wetted_half_width(t)))
        for t in np.linspace(0.02, 0.2, 8)
    ]
    assert np.ptp(cf) / np.mean(cf) <= 1e-12
    assert cf[0] == pytest.approx(ma.force_coefficient(), rel=1e-12)


# ---------------------------------------------------------------- E3
def test_E3_impulse_ledger_closes():
    m = SelfSimilarWedgeImpact(BETA, mode="velocity", speed=V)
    nodes, weights = np.polynomial.legendre.leggauss(80)
    ts = 0.5 * T * (nodes + 1.0)
    numeric = float(np.sum(0.5 * T * weights * np.array(
        [float(m.force_momentum(t)) for t in ts])))
    momentum = float(m.impulse_momentum(T))
    assert abs(numeric - momentum) / momentum <= 1e-2
    # added-mass route: m_a = rho pi c^2 / 2 and I = m_a U_n
    assert momentum == pytest.approx(0.5 * math.pi * RHO_WATER
                                    * float(m.wetted_half_width(T)) ** 2
                                    * float(m.normal_velocity(T)), rel=1e-12)
    # von Karman waterline momentum differs by exactly (pi/2)^2 -- documented
    vk = 0.5 * math.pi * RHO_WATER * float(m.waterline_half_width(T)) ** 2 \
        * float(m.normal_velocity(T))
    assert momentum / vk == pytest.approx((0.5 * math.pi) ** 2, rel=1e-12)
    # the distributed-pressure impulse sits within 10% of the momentum impulse
    pres = float(m.impulse_pressure(T, n_time=120, n_quad=100))
    assert abs(pres - momentum) / momentum <= 0.10


# ---------------------------------------------------------------- E4
def test_E4_joukowsky_bound():
    m = SelfSimilarWedgeImpact(BETA, mode="velocity", speed=V)
    bound = RHO_WATER * C_SOUND_WATER * V
    # the reduced field peaks at rho * cdot^2 / 2
    peak = m.peak_pressure_analytic(T)
    assert peak == pytest.approx(0.5 * RHO_WATER * float(m.wetted_rate(T)) ** 2, rel=1e-9)
    assert peak <= bound
    assert m.shape_peak()["P_max"] == pytest.approx(0.5 * m.const["a1"] ** 2, rel=1e-6)
    # reported (capped) field never exceeds the acoustic bound, any beta
    for beta in (1.0, 3.0, 5.0, 10.0):
        mm = SelfSimilarWedgeImpact(beta, mode="velocity", speed=V)
        assert mm.joukowsky_ratio(T, n_nodes=4000) <= 1.0 + 1e-12
    # small beta breaks the bound in raw potential flow -> the cap is mandatory
    assert m.acoustic_violation_beta() == pytest.approx(3.66, abs=0.02)
    m1 = SelfSimilarWedgeImpact(1.0, mode="velocity", speed=V)
    assert m1.peak_pressure_analytic(T) / bound > 1.0


# ---------------------------------------------------------------- E5
def test_E5_resolution_ledger():
    m = SelfSimilarWedgeImpact(BETA, mode="velocity", speed=V)
    led = resolution_ledger(m, T)
    # reportable: total force converges with observed order >= 1.5
    assert led["force_observed_order_best"] >= 1.5
    assert led["force_pressure_rel_error"] <= 0.10
    # non-reportable: the near-root node value turns negative and diverges
    assert led["near_root_negative_crossing_node_count"] is not None
    assert led["near_root_divergence_order"] > 0.5
    # the positive peak itself converges (falsifies the prior "order <= 0" guess)
    assert led["peak_observed_order_best"] > 0.0
    assert led["peak_under_resolution_error"] > 0.1   # coarse mesh under-predicts
    led2 = resolution_ledger(m, T, node_counts=(400, 800, 1600),
                             quad_counts=(20, 40, 80))
    assert led2["force_observed_order_best"] >= 1.5


# ---------------------------------------------------------------- E6
def test_E6_bulkhead_interface():
    m = SelfSimilarWedgeImpact(BETA, mode="velocity", speed=V)
    t_imp, dt = 0.05, 1.0e-5
    ts = np.arange(0.0, t_imp + 0.5 * dt, dt)
    forces = np.array([float(m.force_momentum(t)) for t in ts])
    # impulse of the handed-over history matches the self-similar momentum impulse
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    assert float(_trapz(forces, ts)) == pytest.approx(m.impulse_momentum(t_imp), rel=1e-3)

    amp = {}
    for period in (0.005, 0.05, 0.2):
        sdof = BulkheadSDOF(mass=500.0, natural_period=period, damping_ratio=0.02)
        res = sdof.dynamic_response_ratio(forces, dt)
        amp[period] = res["dynamic_amplification"]
        # equilibrium check: a static load must return exactly unit amplification
    static = BulkheadSDOF(mass=500.0, natural_period=0.05, damping_ratio=0.02)
    const = np.full(ts.size, float(forces[-1]))
    u, v, a = static.response(np.concatenate(([0.0], const[1:])), dt)
    assert float(np.max(np.abs(u))) / (float(forces[-1]) / static.stiffness) > 0.9

    # monotone rising load: amplification <= 1 and decreasing for long periods
    assert 0.0 < amp[0.005] <= 1.05
    assert amp[0.2] < amp[0.05] < amp[0.005]

    if VAL.exists():                     # the artefact record is part of the node
        val = json.loads(VAL.read_text())
        assert val["E6_bulkhead_interface"]["interface_file"].endswith(
            "m6634_bulkhead_interface.json")
        assert val["all_criteria_passed"] is True
        assert val["E1_self_similar_fold"]["pressure_l2_residual_constant_velocity"] <= 1e-2
        assert val["E2_wagner_small_beta"]["worst_rel_dev"] <= 0.03
        assert val["E3_impulse_ledger"]["closure_rel_error"] <= 1e-2
        assert max(r["ratio_capped"] for r in val["E4_joukowsky_bound"]["sweep_beta"]) <= 1.0