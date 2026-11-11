"""Tests for the measured-data UQ face of the ice-going digital twin.

Node ``p223w2b-n03-physics-uq-digital-twin``.  The suite is organised by the
criterion it protects:

* corpus provenance / geometry convention / double-entry agreement,
* forward model identity with the reference Lindqvist implementation,
* MCMC sampler correctness against an *analytic* Gaussian posterior,
* calibration criteria (posterior vs literature, split-R-hat),
* Sobol' closure (exact quadrature vs pick-freeze Monte Carlo),
* predictive-interval coverage and its negative control,
* pseudo-BMA bookkeeping over the published literature laws.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

from tensorphysics.ice.resistance import ShipGeometry
from tensorphysics.ice_digital_twin import measured_uq as muq
from tensorphysics.ice_digital_twin.twin import resistance_components

REPO_ROOT = muq.REPO_ROOT


# ---------------------------------------------------------------------------
# corpus, provenance, geometry convention
# ---------------------------------------------------------------------------


def test_campaign_loads_registered_measured_points():
    camp = muq.load_measured_campaign()
    assert len(camp) == 10
    assert camp.series_id == "MTUIKKU-HZ2016"
    assert camp.doi == "10.1016/j.ijnaoe.2016.01.004"
    # digest must equal the registry entry, not a stale literal
    reg = json.load(open(os.path.join(REPO_ROOT, muq.REGISTRY_REL), encoding="utf-8"))
    entry = [s for s in reg["series"] if s["series_id"] == muq.SERIES_ID][0]
    assert camp.sha256 == entry["sha256"]
    assert entry["grade"] == "MEASURED-PRIMARY"


def test_campaign_physical_sanity():
    camp = muq.load_measured_campaign()
    assert np.all(camp.h_m > 0.0) and np.all(camp.h_m < 2.0)
    assert np.all(camp.v_ms > 0.0)
    assert np.all(camp.r_meas_kN > 0.0)
    assert np.all(camp.sigma_r_kN > 0.0)
    # measured ice strengths inside the cited ranges (crush 2-8 MPa, flex 0.35-0.7 MPa widened)
    assert np.all((camp.sigma_c_pa > 1.0e6) & (camp.sigma_c_pa < 6.0e6))
    assert np.all((camp.sigma_f_pa > 0.3e6) & (camp.sigma_f_pa < 1.2e6))


def test_uncertainty_imputation_is_flagged():
    camp = muq.load_measured_campaign()
    flagged = [p.point_id for p in camp.points if p.sigma_r_imputed]
    for p in camp.points:
        if p.sigma_r_imputed:
            assert p.sigma_r_kN == pytest.approx(
                muq.UNC_IMPUTATION_FRACTION * abs(p.r_meas_kN), rel=0, abs=1e-9
            )
    # the asset reports a band for most points; imputation is the exception
    assert len(flagged) == 3


def test_double_entry_against_independent_transcript():
    res = muq.double_entry_check()
    assert res["available"] is True
    assert res["n_points_checked"] == 10
    assert res["agreement"] is True
    assert res["n_mismatch"] == 0


def test_geometry_convention_is_documented_and_applied():
    camp = muq.load_measured_campaign()
    raw = json.load(open(os.path.join(REPO_ROOT, muq.SERIES_REL), encoding="utf-8"))
    half_angle = float(raw["ship"]["bow_angle_deg"])
    assert camp.ship.bow_angle_deg == pytest.approx(2.0 * half_angle)
    assert camp.ship.L_pp == pytest.approx(float(raw["ship"]["L_pp_m"]))
    assert camp.ship.B == pytest.approx(float(raw["ship"]["B_m"]))
    # the law consumes 0.5*bow_angle_rad, i.e. the documented half-angle
    assert 0.5 * camp.ship.bow_angle_rad == pytest.approx(math.radians(half_angle))


def test_published_law_matrix_is_complete():
    camp = muq.load_measured_campaign()
    keys, mat = camp.published_matrix()
    assert keys == muq.PUBLISHED_LAWS
    assert mat.shape == (10, 4)
    assert np.all(np.isfinite(mat))
    ratios = muq.load_published_laws(camp)
    # the four published laws bracket the measurement (no single law is exact)
    assert min(ratios.values()) < 1.0 < max(ratios.values())


# ---------------------------------------------------------------------------
# forward model identity + discrepancy structure
# ---------------------------------------------------------------------------


def test_vectorised_forward_matches_reference():
    """The vectorised per-point law must equal the twin's reference algebra."""
    camp = muq.load_measured_campaign()
    theta = (1.13, 0.87, 0.19, 1.07)
    rc, rb, rs = muq.forward_components_kN(camp, *theta)
    for i, point in enumerate(camp.points):
        single = camp.subset([i])
        r_c, r_b, r_s = resistance_components(
            theta[0] * single.sigma_c_pa[0] / 5.0e6,
            theta[1] * single.sigma_f_pa[0] / 5.0e5,
            theta[2],
            theta[3],
            single.h_m[0],
            single.v_ms[0],
            camp.ship,
        )
        assert rc[i] == pytest.approx(r_c / 1e3, rel=1e-12)
        assert rb[i] == pytest.approx(r_b / 1e3, rel=1e-12)
        assert rs[i] == pytest.approx(r_s / 1e3, rel=1e-12)


def test_discrepancy_factor_structure():
    h = np.array([0.5, 1.0, 2.0])
    assert np.allclose(muq.discrepancy_factor(0.0, 0.0, h, 1.0), 1.0)
    d = muq.discrepancy_factor(math.log(2.0), 1.0, h, 1.0)
    assert np.allclose(d, 2.0 * h)
    # monotone in thickness for a positive exponent: log-log structure
    assert np.all(np.diff(muq.discrepancy_factor(0.0, 2.0, h, 1.0)) > 0.0)


def test_literature_calibration_overpredicts_the_measured_points():
    """Regression guard: the known closure defect must not be silently 'fixed'.

    Node p273w3-n07 recorded a median 198 % error against this measured series
    with the bending term carrying most of the prediction.  Any change to the
    forward law that removes the overprediction has to update this test.
    """
    camp = muq.load_measured_campaign()
    theta = np.array([1.0, 1.0, 0.15, 1.0, 0.0, 0.0, 0.0])
    pred = muq.forward_kN(camp, theta, with_discrepancy=False)
    ratio = pred / camp.r_meas_kN
    assert np.median(ratio) > 2.0
    rc, rb, rs = muq.forward_components_kN(camp, 1.0, 1.0, 0.15, 1.0)
    assert float(np.median(rb / (rc + rb + rs))) > 0.5


# ---------------------------------------------------------------------------
# sampler correctness against an analytic posterior
# ---------------------------------------------------------------------------


def test_metropolis_recovers_analytic_gaussian():
    mean = np.array([0.3, -0.2])
    cov = np.array([[0.04, 0.012], [0.012, 0.01]])
    prec = np.linalg.inv(cov)
    bounds = np.array([[-1.5, 1.5], [-1.5, 1.5]], dtype=float)

    def log_post(x: np.ndarray) -> float:
        d = x - mean
        return -0.5 * float(d @ prec @ d)

    inits = np.array([[0.0, 0.0], [0.6, -0.5], [-0.5, 0.45]])
    chains = muq.run_metropolis(
        log_post, bounds, inits=inits, n_steps=8000, burn=2000, seed=11, n_adapt=1500
    )
    flat = muq._post_burn_array(chains).reshape(-1, 2)
    assert flat.shape[0] > 5000
    assert np.allclose(flat.mean(axis=0), mean, atol=0.012)
    assert np.allclose(np.cov(flat, rowvar=False), cov, rtol=0.12, atol=2e-3)
    summary = muq.posterior_summary(chains, ["a", "b"])
    assert summary["rhat_max"] < 1.05


def test_metropolis_rejects_outside_starts():
    def log_post(x):
        return -0.5 * float(np.sum(np.square(x)))

    bounds = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    with pytest.raises(ValueError):
        muq.run_metropolis(log_post, bounds, inits=np.array([[2.0, 0.0]]), n_steps=10)


# ---------------------------------------------------------------------------
# posterior construction
# ---------------------------------------------------------------------------


def test_log_posterior_bounds_and_freeze():
    camp = muq.load_measured_campaign()
    log_post, bounds, names = muq.log_posterior_factory(camp, prior="literature")
    assert names == list(muq.PARAM_NAMES_MEASURED)
    assert bounds.shape == (7, 2)
    x = np.array([1.0, 1.0, 0.15, 1.0, -1.0, 0.0, 0.25])
    assert np.isfinite(log_post(x))
    outside = x.copy()
    outside[0] = 5.0
    assert log_post(outside) == -math.inf
    # a frozen discrepancy is exactly the beta0 = beta1 = 0 model
    log_post_nc, _, _ = muq.log_posterior_factory(camp, with_discrepancy=False)
    frozen = x.copy()
    frozen[4] = 0.0
    assert log_post_nc(x) == pytest.approx(log_post_nc(frozen), rel=1e-12)


def test_literature_prior_prefers_reference_values():
    ref = np.array([1.0, 1.0, 0.15, 1.0, 0.0, 0.0, 0.25])
    moved = ref.copy()
    moved[0] = 1.5
    assert muq._prior_logpdf(ref, muq.PRIOR_LITERATURE) > muq._prior_logpdf(
        moved, muq.PRIOR_LITERATURE
    )
    # kappa is uniform: flat prior
    a, b = ref.copy(), ref.copy()
    a[6], b[6] = 0.05, 0.55
    assert muq._prior_logpdf(a, muq.PRIOR_LITERATURE) == muq._prior_logpdf(
        b, muq.PRIOR_LITERATURE
    )


# ---------------------------------------------------------------------------
# calibration criteria (a) and (b)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def calibration():
    camp = muq.load_measured_campaign()
    return muq.calibrate(camp, n_chains=3, n_steps=6000, burn=2000, seed=20261)


def test_posterior_means_agree_with_literature(calibration):
    params = calibration["summary"]["parameters"]
    for name in ("sigma_c_scale", "sigma_f_scale", "mu_hull_ice", "C_sub_scale"):
        assert params[name]["relative_error"] < 0.20, (name, params[name])
    # the discrepancy amplitude must be the quantity that absorbs the closure error
    assert params["beta0"]["mean"] < -0.5


def test_chains_converged(calibration):
    summary = calibration["summary"]
    assert summary["rhat_max"] < 1.1
    assert summary["ess_min"] > 200


def test_flat_prior_sensitivity_exposes_degeneracy():
    camp = muq.load_measured_campaign().subset([0, 1, 2, 3, 4, 5])
    ident = muq.prior_sensitivity(camp, n_steps=3000, burn=1200, seed=7)
    assert ident["prior"] == "flat"
    assert set(ident["at_bounds"]) == set(muq.PARAM_NAMES_MEASURED)
    # under flat priors the discrepancy amplitude still has to be negative
    assert ident["summary"]["parameters"]["beta0"]["mean"] < 0.0


# ---------------------------------------------------------------------------
# global sensitivity (criterion c)
# ---------------------------------------------------------------------------


def _additive(x: np.ndarray) -> np.ndarray:
    return x[:, 0] + 2.0 * x[:, 1] + 3.0 * x[:, 2]


def test_sobol_exact_is_analytic_on_an_additive_model():
    bounds = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    res = muq.sobol_exact(_additive, bounds, order=24)
    var = (1.0 + 4.0 + 9.0) / 12.0
    assert res["sum_S_i"] == pytest.approx(1.0, abs=1e-9)
    assert res["sum_S_Ti"] == pytest.approx(1.0, abs=1e-9)
    assert res["S_i"][0] == pytest.approx((1.0 / 12.0) / var, rel=1e-9)
    assert res["S_i"][1] == pytest.approx((4.0 / 12.0) / var, rel=1e-9)
    assert res["interaction_share"] == pytest.approx(0.0, abs=1e-9)


def test_sobol_exact_detects_pure_interaction():
    """``f = x*y`` on ``U[0,1]^2``: analytic indices, interaction share 1/7."""

    def product(x: np.ndarray) -> np.ndarray:
        return x[:, 0] * x[:, 1]

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    res = muq.sobol_exact(product, bounds, order=24)
    s_first = (0.25 / 12.0) / (1.0 / 9.0 - 1.0 / 16.0)  # V(E[f|x])/V(f)
    assert res["S_i"] == pytest.approx([s_first, s_first], rel=1e-9)
    s_total = 1.0 - s_first  # 1 - V(E[f|x_-i])/V(f) = 0.5714...
    assert res["sum_S_i"] == pytest.approx(2.0 * s_first, abs=1e-9)
    assert res["S_Ti"] == pytest.approx([s_total, s_total], rel=1e-9)
    assert res["sum_S_Ti"] == pytest.approx(2.0 * s_total, abs=1e-9)
    assert res["interaction_share"] == pytest.approx(2.0 * s_total - 1.0, abs=1e-9)
    # a pure product has strictly positive first-order indices and total > first
    assert 0.0 < s_first < 1.0
    assert all(t > s for s, t in zip(res["S_i"], res["S_Ti"]))


def test_twin_sobol_closure_and_mc_agreement():
    camp = muq.load_measured_campaign()
    bounds = muq.PRIOR_BOUNDS[:4].copy()
    func = muq.sensitivity_point(camp)
    exact = muq.sobol_exact(func, bounds, order=16)
    mc = muq.sobol_mc(func, bounds, n_base=1 << 13, seed=3)
    assert abs(exact["sum_S_Ti"] - 1.0) < 0.05
    assert exact["sum_S_i"] <= 1.0 + 1e-9
    assert abs(mc["sum_S_Ti"] - exact["sum_S_Ti"]) < 0.05
    for a, b in zip(exact["S_Ti"], mc["S_Ti"]):
        assert abs(a - b) < 0.05
    # the bending coefficient dominates the response variance
    assert exact["S_Ti"][1] == max(exact["S_Ti"])


# ---------------------------------------------------------------------------
# coverage (criterion d) and its negative control
# ---------------------------------------------------------------------------


def test_negative_control_undercovers():
    camp = muq.load_measured_campaign()
    nc = muq.negative_control(camp)
    assert nc["n_points"] == 10
    assert nc["coverage"] < 0.5
    assert nc["median_ratio_model_over_measured"] > 2.0


def test_loo_coverage_on_real_points():
    camp = muq.load_measured_campaign().subset([0, 1, 2, 3])
    loo = muq.loo_predictive(camp, n_steps=1200, burn=500, seed=31, with_discrepancy=True)
    assert loo["n_points"] == 4
    assert 0.0 <= loo["coverage"] <= 1.0
    for row in loo["rows"]:
        assert row["pi_low_kN"] < row["pi_high_kN"]
        assert row["pi_width_kN"] > 0.0
        assert row["inside"] == (row["pi_low_kN"] <= row["observed_kN"] <= row["pi_high_kN"])


# ---------------------------------------------------------------------------
# pseudo-BMA (criterion e bookkeeping)
# ---------------------------------------------------------------------------


def test_bma_weights_and_widths():
    camp = muq.load_measured_campaign().subset([0, 1, 2, 3, 4])
    bma = muq.bma_published_laws(
        camp, n_steps=1200, burn=500, n_steps_loo=600, burn_loo=250, seed=44, n_draw=600
    )
    assert bma["weights_sum"] == pytest.approx(1.0, abs=1e-9)
    assert set(bma["weights"]) == set(muq.PUBLISHED_LAWS)
    assert all(w > 0.0 for w in bma["weights"].values())
    # bias-corrected predictions must be inside the span of the published laws
    keys, mat = camp.published_matrix()
    for law, rec in bma["members"].items():
        assert 0.0 < rec["exp_beta_mean"] < 3.0
        assert rec["mean_pi_width_kN"] > 0.0
    assert bma["bma_mean_pi_width_kN"] > 0.0


# ---------------------------------------------------------------------------
# end-to-end report contract
# ---------------------------------------------------------------------------


def test_run_measured_uq_report_contract():
    camp = muq.load_measured_campaign().subset([0, 1, 2, 3])
    rep = muq.run_measured_uq(
        campaign=camp,
        calib_steps=800,
        calib_burn=300,
        loo_steps=500,
        loo_burn=200,
        sob_order=8,
        sob_n_base=1 << 8,
        seed=99,
    )
    assert rep["node"] == "p223w2b-n03-physics-uq-digital-twin"
    assert rep["series"]["n_points"] == 4
    assert set(rep["criteria"]) == {"a", "b", "c", "d", "e"}
    for cid, crit in rep["criteria"].items():
        assert crit["status"] in {"PASS", "FAIL"}
        assert crit["evidence"]
    assert rep["criteria"]["b"]["status"] == "PASS"  # convergence is unconditional
    assert rep["digest"]
    # the digest must be reproducible from the payload
    from tensorphysics.ice_digital_twin.bayes import digest as _digest

    assert rep["digest"] == _digest({k: v for k, v in rep.items() if k != "digest"})
