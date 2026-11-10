"""Tests for the design-sea-state statistical-convention verification surface.

Coverage map (13 machine criteria + supporting physics):

* Rosenblatt transform: inverse round trip, density/Jacobian identity, log-normal
  conditional mean definition.
* IFORM: analytic correlated-normal anchor (quadratic form, extreme H_s,
  equivalence to the chi-square ellipse), exact extreme-H_s anchor of the
  metocean contour, angle-sweep refinement.
* Reliability: p_f <-> beta round trip in both conventions, chi-square identity,
  DNV/ISO 2394 rows, lifetime conversion, F-vs-C mismatch.
* Rayleigh: scale trap (H_s/2 vs H_s/4), exceedance law, N-wave modal maximum,
  Monte-Carlo recovery.
* Adjudication: Rayleigh rule vs IFORM vs exact long-term quadrature, MC
  outside-contour probability, and the full study criteria table.
"""

from __future__ import annotations

import math

import pytest

from tensorphysics.design_sea_state import (
    BivariateNormalTransform,
    DEFAULT_CONFIG,
    MACHINE_CRITERIA_KEYS,
    ReliabilityTarget,
    analytic_max_hs,
    beta_contour_from_pout,
    beta_cornell_from_pf,
    build_model,
    contour_max_hs,
    contour_refinement_error,
    convention_report,
    elevation_sigma,
    evaluate_criteria,
    iform_contour,
    lifetime_pf,
    long_term_reference,
    mahalanobis_residual,
    max_response_on_contour,
    mc_outside_probability,
    modal_largest_wave,
    modal_largest_wave_response,
    pf_from_beta_cornell,
    points_outside_polygon,
    pout_from_beta_contour,
    quadrature_grid,
    quadratic_residual_normal,
    rayleigh_exceedance,
    rayleigh_mc_check,
    rayleigh_mean,
    rayleigh_mode,
    rayleigh_scale,
    rainflow_n_waves,
    return_period_from_lifetime_pf,
    run_study,
    wave_power,
)


@pytest.fixture(scope="module")
def model():
    return build_model()


@pytest.fixture(scope="module")
def study():
    return run_study({"mc_samples": 100_000})


# --------------------------------------------------------------------------- #
# 1. Rosenblatt transform
# --------------------------------------------------------------------------- #
def test_rosenblatt_inverse_round_trip(model):
    """(H_s, T_z) -> u -> (H_s, T_z) must be the identity to rounding."""
    worst = 0.0
    for hs, tz in model.sample(200, seed=7):
        u1, u2 = model.forward(hs, tz)
        hs2, tz2 = model.inverse(u1, u2)
        worst = max(worst, abs(hs2 - hs) / hs, abs(tz2 - tz) / tz)
    assert worst < 1e-12


def test_rosenblatt_density_jacobian_identity(model):
    """phi2(u) * |det du/dx| must equal the joint density f(H_s, T_z)."""
    assert model.max_log_residual(model.sample(200, seed=11)) < 1e-12


def test_conditional_mean_law(model):
    """The log-normal conditional is parameterised so E[T_z | H_s] is the power law."""
    for hs in (1.5, 4.0, 9.0):
        mu = model.tz_conditional.mu_ln(hs)
        sigma = model.tz_conditional.sigma_ln
        assert math.exp(mu + 0.5 * sigma * sigma) == pytest.approx(
            model.tz_conditional.mean_tz(hs), rel=1e-12
        )


def test_marginal_quantiles_match_uniform_sampling(model):
    """The Weibull ppf is the exact inverse of its cdf (guards the sampler)."""
    for p in (0.01, 0.5, 0.99, 0.999):
        assert model.hs_marginal.cdf(model.hs_marginal.ppf(p)) == pytest.approx(p, rel=1e-12)


# --------------------------------------------------------------------------- #
# 2. IFORM contours
# --------------------------------------------------------------------------- #
def test_iform_matches_analytic_bivariate_normal_anchor():
    """C1: IFORM contour vs the analytic correlated-normal anchor (<= 1 %)."""
    transform = BivariateNormalTransform(mu_x=3.0, sigma_x=1.2, mu_y=8.0, sigma_y=1.5, rho=0.6)
    beta = 2.3263478740408408
    contour = iform_contour(transform, beta, 256)
    assert quadratic_residual_normal(transform, contour) < 1e-12
    numeric = max(p.hs for p in contour)
    analytic = transform.mu_x + transform.sigma_x * beta
    assert abs(numeric - analytic) / analytic < 1e-12


def test_iform_normal_case_equals_chi_square_ellipse():
    """C12: for a correlated normal the IFORM contour *is* the chi-square ellipse."""
    transform = BivariateNormalTransform(rho=0.6)
    contour = iform_contour(transform, 3.0, 128)
    assert mahalanobis_residual(transform, contour) < 1e-12
    # ... and the two differ from the tangent-box approximation, i.e. the test
    # is not vacuous: the contour's T_z spread equals the ellipse's, both beta.
    assert max(p.tz for p in contour) - min(p.tz for p in contour) > 0.0


def test_contour_extreme_hs_analytic_anchor(model):
    """C2: max H_s on the contour equals F_H^-1(Phi(beta)) exactly."""
    beta = beta_cornell_from_pf(1e-2)
    contour = iform_contour(model, beta, 72)
    anchor = analytic_max_hs(model, beta)
    assert abs(contour_max_hs(contour) - anchor) / anchor < 1e-10
    # and this IS the 100-year marginal return value
    assert anchor == pytest.approx(model.hs_marginal.ppf(1.0 - 1e-2), rel=1e-10)


def test_contour_closure_and_physicality(model):
    """The contour is a closed curve in the physical quadrant with finite knees."""
    contour = iform_contour(model, 3.0, 72)
    assert len(contour) == 72
    assert all(p.hs > model.hs_marginal.gamma for p in contour)
    assert all(p.tz > 0.0 for p in contour)
    # u-space radius is exact by construction; physical mapping is continuous
    assert max(abs(p.u_radius2 - 9.0) for p in contour) < 1e-9
    gaps = [
        abs(contour[(i + 1) % len(contour)].hs - p.hs) for i, p in enumerate(contour)
    ]
    assert max(gaps) < 2.0  # no discontinuity between neighbouring angles


def test_contour_refinement_converges(model):
    """C10: refinement 36 -> 288 points moves the design response by <= 1 %."""
    err = contour_refinement_error(model, beta_cornell_from_pf(1e-2), 36, 288)
    assert err["response_rel_error"] < 1e-2
    assert err["max_hs_rel_error"] < 1e-2


# --------------------------------------------------------------------------- #
# 3. Reliability index and conventions
# --------------------------------------------------------------------------- #
def test_reliability_roundtrip_both_conventions():
    """C5: p_f <-> beta round trip, and the chi-square identity, are machine exact."""
    from scipy.stats import chi2

    for beta in (1.0, 2.3263478740408408, 3.71901648545568):
        assert beta_cornell_from_pf(pf_from_beta_cornell(beta)) == pytest.approx(beta, rel=1e-12)
        assert beta_contour_from_pout(pout_from_beta_contour(beta)) == pytest.approx(beta, rel=1e-12)
        assert chi2.sf(beta * beta, df=2) / math.exp(-0.5 * beta * beta) == pytest.approx(1.0, rel=1e-12)


def test_dnv_iso2394_target_rows():
    """C5: the classical target-reliability rows are reproduced from p_f alone."""
    from tensorphysics.design_sea_state import DNV_ISO2394_TARGETS

    for _name, beta_ref, pf_ref in DNV_ISO2394_TARGETS:
        assert beta_cornell_from_pf(pf_ref) == pytest.approx(beta_ref, rel=1e-2)


def test_convention_mismatch_is_quantified():
    """C6: the F-vs-C gap exists, is large, and is reported (never silently mixed)."""
    report = convention_report((1e-2, 1e-4))
    for row in report["rows"]:
        assert row["beta_contour"] > row["beta_cornell"]
    worst = report["worst_row"]
    # at p_f = 1e-4 the chi-square contour radius reports ~9.9x the target mass
    assert worst["pout_over_target"] > 5.0
    assert abs(math.log(worst["pout_over_target"])) >= 0.3


def test_return_period_and_lifetime():
    """Return period and lifetime conversions stay mutually consistent."""
    target = ReliabilityTarget(1e-2)
    assert target.return_period == pytest.approx(100.0)
    pf25 = lifetime_pf(1e-2, 25.0)
    assert 0.2 < pf25 < 0.25
    assert return_period_from_lifetime_pf(0.1, 25.0) == pytest.approx(237.0, abs=1.0)


# --------------------------------------------------------------------------- #
# 4. Rayleigh conventions
# --------------------------------------------------------------------------- #
def test_rayleigh_scale_trap():
    """Rajleigh scale of *wave heights* is H_s/2; sqrt(m0) = H_s/4 is the elevation."""
    assert rayleigh_scale(6.0) == pytest.approx(3.0)
    assert elevation_sigma(6.0) == pytest.approx(1.5)
    assert rayleigh_mode(6.0) == pytest.approx(3.0)
    assert rayleigh_mean(6.0) == pytest.approx(0.6267 * 6.0, rel=1e-3)
    # using H_s/4 as the scale would halve the mode: that would be a 50 % error
    assert abs(elevation_sigma(6.0) - rayleigh_mode(6.0)) / rayleigh_mode(6.0) == pytest.approx(0.5)


def test_rayleigh_exceedance_law():
    """Q(h) = exp(-2 h^2 / H_s^2) and the N-wave modal maximum are consistent."""
    hs, n = 6.0, 1000.0
    h_mode = modal_largest_wave(hs, n)
    assert rayleigh_exceedance(h_mode, hs) == pytest.approx(1.0 / n, rel=1e-12)
    # 3-hour wave count from T_z
    assert rainflow_n_waves(7.2) == pytest.approx(1500.0)
    # textbook: 1000-wave extreme is ~1.86 H_s
    assert h_mode / hs == pytest.approx(math.sqrt(math.log(n) / 2.0))


def test_rayleigh_monte_carlo_recovery():
    """C7: mode / mean / H_s / N-wave modal maximum recovered from 4e6 MC waves."""
    result = rayleigh_mc_check(hs=6.0, n_waves=1000.0, n_groups=4000, seed=20260101)
    assert result["mode_rel_err"] < 1e-2
    assert result["mean_rel_err"] < 1e-2
    assert result["hs_rel_err"] < 1e-2
    assert result["mode_max_rel_err"] < 1e-2
    # the smoothed-histogram estimate is documented as biased low on this skewed
    # density; keep that documented rather than hidden
    assert 0.0 < result["mode_rel_err_hist"] < 3e-2


# --------------------------------------------------------------------------- #
# 5. Exceedance calibration and adjudication
# --------------------------------------------------------------------------- #
def test_mc_outside_probability_is_calibrated(model):
    """C4: measured outside-contour probability equals exp(-beta^2/2)."""
    for beta in (1.5, 2.0):
        result = mc_outside_probability(model, beta, n=60_000, seed=4242)
        assert result["rel_error"] < 3e-2
        assert abs(result["z_score"]) < 3.0


def test_polygon_membership_sanity():
    """Ray-casting: inside/outside classification on a unit square."""
    import numpy as np

    poly_x = np.array([0.0, 1.0, 1.0, 0.0])
    poly_y = np.array([0.0, 0.0, 1.0, 1.0])
    x = np.array([0.5, 1.5, -0.5, 0.5])
    y = np.array([0.5, 0.5, 0.5, 1.5])
    outside = points_outside_polygon(x, y, poly_x, poly_y)
    assert list(outside) == [False, True, True, True]


def test_long_term_reference_self_consistency(model):
    """The quadrature reference achieves its own target probability."""
    grid = quadrature_grid(model, n=120)
    ref = long_term_reference(model, wave_power, 1e-2, grid=grid)
    assert ref["achieved_prob"] == pytest.approx(1e-2, rel=5e-2)
    assert ref["grid_prob_mass"] == pytest.approx(1.0, rel=1e-12)
    # monotone response: a stricter target gives a larger design value
    stricter = long_term_reference(model, wave_power, 1e-3, grid=grid)
    assert stricter["design_response"] > ref["design_response"]


def test_rayleigh_vs_iform_vs_longterm(study):
    """C8 + C9: the three design-value routes agree within the stated bands."""
    for name, block in study["adjudication"].items():
        assert block["rayleigh_vs_iform_rel"] <= 5e-2, name
        assert block["iform_vs_reference_rel"] <= 5e-2, name
        assert block["rayleigh_vs_reference_rel"] <= 5e-2, name
    # the wave-power design point slides off theta = 0 (joint shape matters)
    assert study["adjudication"]["wave_power"]["iform_theta_deg"] > 0.0
    # ... whereas a pure wave-height response stays on the extreme-H_s point
    assert study["adjudication"]["modal_largest_wave"]["iform_theta_deg"] == pytest.approx(0.0)


def test_convention_c_contour_is_conservative(study):
    """Documented consequence of convention C: the same annual target gives a
    much rarer (hence larger) design response — the number, not an opinion."""
    band = study["adjudication"]["wave_power"]
    assert band["conventionC_vs_reference_rel"] > 0.2
    assert band["conventionC_iform_value"] > band["iform_value"]


def test_gaussian_surrogate_is_not_equivalent(study):
    """C13: non-Gaussian marginals move the design value (why IFORM needs them)."""
    for name, block in study["adjudication"].items():
        assert block["gaussian_surrogate_rel_diff"] >= 2e-2, name


# --------------------------------------------------------------------------- #
# 6. Whole-study criteria table
# --------------------------------------------------------------------------- #
def test_all_machine_criteria_pass(study):
    """The 13 machine criteria of the node must all be PASS, with numbers."""
    rows = study["criteria"]
    assert [r["key"] for r in rows] == list(MACHINE_CRITERIA_KEYS)
    failed = [r for r in rows if r["verdict"] != "PASS"]
    assert not failed, failed
    for row in rows:
        assert isinstance(row["measured"], float)
        assert math.isfinite(row["measured"])


def test_evaluate_criteria_rejects_a_broken_study(study):
    """The criteria table is a real gate, not a rubber stamp."""
    import copy

    broken = copy.deepcopy(study)
    broken["normal_anchor"]["max_quadratic_residual"] = 0.05
    broken["adjudication"]["wave_power"]["rayleigh_vs_iform_rel"] = 0.4
    rows = {r["key"]: r for r in evaluate_criteria(broken)}
    assert rows["C1_iform_normal_anchor"]["verdict"] == "FAIL"
    assert rows["C8_rayleigh_vs_iform"]["verdict"] == "FAIL"


def test_study_is_json_serialisable(study):
    """Everything the study returns must survive a JSON dump (report pipeline)."""
    import json

    payload = json.dumps(study)
    assert len(payload) > 1000
    assert json.loads(payload)["config"]["annual_pf"] == DEFAULT_CONFIG["annual_pf"]


def test_study_config_is_honoured():
    """A different target probability moves the contour radius as documented."""
    out = run_study({"annual_pf": 1e-3, "mc_samples": 20_000, "quadrature_nodes": 100})
    assert out["contour"]["beta_convention_F"] == pytest.approx(beta_cornell_from_pf(1e-3), rel=1e-12)
    assert out["contour"]["beta_convention_C"] == pytest.approx(beta_contour_from_pout(1e-3), rel=1e-12)
    assert all(row["verdict"] == "PASS" for row in out["criteria"])