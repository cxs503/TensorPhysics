"""Tests for the hull girder ice/wave extreme bending moment module.

Coverage: correlated random fields, beam operators and analytic limits,
ice pressure model, ITTC/IACS wave bending moment, ultimate capacity,
extreme value statistics, load combination and the acceptance criteria.
"""

import math

import numpy as np
import pytest

from tensorphysics.hullgirder import (
    IceCondition,
    ShipGirder,
    correlation_length,
    design_wave_moment,
    exponential_covariance,
    fit_correlation_length_model,
    global_ice_pressure,
    ittc_design_wave_moment,
    ittc_spectrum,
    moment_batch,
    plate_buckling_reduction,
    ultimate_bending_moment,
    utilisation_check,
)
from tensorphysics.hullgirder.extremes import (
    combine_design_moments,
    gumbel_fit,
    gumbel_return_level,
    ice_bending_extremes,
    spearman_rank_correlation,
    weibull_fit,
    weibull_return_level,
)
from tensorphysics.hullgirder.girder import (
    independent_segment_variance,
    moment_matrix,
    moment_statistics,
    shear_and_moment,
    uniform_load_midspan_moment,
)
from tensorphysics.hullgirder.ice_load import (
    REFERENCE_CORRELATION_COEFFICIENTS,
    contact_patch_length,
    ice_load_scenarios,
    ice_pressure_profiles,
)
from tensorphysics.hullgirder.random_field import (
    ar1_field,
    check_uniform_grid,
    fit_exponential_length,
    karhunen_loève_field,
    sample_correlated_field,
    space_time_field,
    squared_exponential_covariance,
)
from tensorphysics.hullgirder.verification import (
    check_independent_limit,
    check_uniform_degeneracy,
    check_wave_moment,
)
from tensorphysics.hullgirder.wave import (
    long_wave_closed_form_moment,
    rule_coefficient_c,
    wave_moment_rao,
)

GRID = np.linspace(-100.0, 100.0, 401)


# ----------------------------------------------------------------- random field
def test_ar1_reproduces_exponential_covariance():
    lam = 5.0
    z = ar1_field(GRID, lam, sigma=2.0, n_real=4000, rng=np.random.default_rng(1))
    cov = np.cov(z, rowvar=False)
    targets = exponential_covariance(GRID, lam, sigma=2.0)
    for i, j in [(0, 0), (200, 200), (200, 202), (200, 206), (200, 210)]:
        assert cov[i, j] == pytest.approx(targets[i, j], rel=0.12)


def test_ar1_decorrelated_limit_is_white_noise():
    dx = GRID[1] - GRID[0]
    z = ar1_field(GRID, lam=1e-4 * dx, sigma=1.0, n_real=4000, rng=np.random.default_rng(2))
    cov = np.cov(z, rowvar=False)
    assert np.std(np.diag(cov)) < 0.05 * np.mean(np.diag(cov))
    off = cov - np.diag(np.diag(cov))
    assert np.mean(np.abs(off)) < 0.02
    assert np.max(np.abs(off)) < 0.08


def test_ar1_fully_correlated_limit():
    z = ar1_field(GRID, lam=1e6, sigma=1.0, n_real=500, rng=np.random.default_rng(3))
    # a near-constant random level: the residual spread is the accumulated
    # innovation over the 401 nodes, O(sqrt(n) * sqrt(1 - rho^2))
    assert np.max(np.std(z, axis=1)) < 0.05
    assert np.corrcoef(z[:, 0], z[:, -1])[0, 1] > 0.99


def test_karhunen_loève_field_variance():
    x = np.linspace(0.0, 50.0, 60)
    z = karhunen_loève_field(x, lam=4.0, sigma=1.5, n_real=6000, rng=np.random.default_rng(4))
    assert np.allclose(np.var(z, axis=0), 1.5 ** 2, rtol=0.12)
    cov_target = squared_exponential_covariance(x, 4.0, 1.5)
    assert np.cov(z, rowvar=False)[0, 3] == pytest.approx(cov_target[0, 3], rel=0.15)


def test_uniform_grid_required():
    with pytest.raises(ValueError):
        check_uniform_grid(np.array([0.0, 1.0, 3.0]))
    assert check_uniform_grid(np.linspace(0.0, 1.0, 11)) == pytest.approx(0.1)


def test_fit_exponential_length_recovers_lambda():
    # record length (120 m) must be long compared with the correlation length
    x = np.linspace(0.0, 120.0, 481)
    z = ar1_field(x, lam=2.0, sigma=1.0, n_real=1500, rng=np.random.default_rng(5))
    assert fit_exponential_length(x, z) == pytest.approx(2.0, rel=0.15)


def test_space_time_field_temporal_correlation():
    x = np.linspace(-20.0, 20.0, 81)
    z = space_time_field(x, lam_x=2.0, lam_t=1.0, sigma=1.0, n_steps=400, dt=0.25, seed=6)
    assert z.shape == (400, 81)
    lag1 = np.corrcoef(z[:-1, 40], z[1:, 40])[0, 1]
    # the realised lag-one correlation is O(rho_t) but not exact (the innovation
    # is spatially filtered): documented approximation
    assert 0.4 < lag1 < 0.95
    assert np.corrcoef(z[:-5, 40], z[5:, 40])[0, 1] < lag1


# ---------------------------------------------------------------------- girder
def test_uniform_load_simply_supported_midspan():
    q = 1.0e6
    _, m = shear_and_moment(GRID, np.full(GRID.size, q), support="simple")
    analytic = uniform_load_midspan_moment(q, 200.0)
    assert m[GRID.size // 2] == pytest.approx(analytic, rel=1e-6)


def test_floating_uniform_load_gives_no_bending():
    _, m = shear_and_moment(GRID, np.full(GRID.size, 1.0e6), support="free")
    assert np.max(np.abs(m)) < 1e-6 * uniform_load_midspan_moment(1.0e6, 200.0)


def test_point_load_analytical_limits():
    force = 1.0e7
    q = np.zeros(GRID.size)
    q[GRID.size // 2] = force / (GRID[1] - GRID[0])
    _, m_free = shear_and_moment(GRID, q, support="free")
    _, m_simple = shear_and_moment(GRID, q, support="simple")
    assert m_free[GRID.size // 2] == pytest.approx(-force * 200.0 / 8.0, rel=0.01)
    assert m_simple[GRID.size // 2] == pytest.approx(force * 200.0 / 4.0, rel=0.01)


def test_moment_matrix_matches_scalar_solver():
    q = np.where(np.abs(GRID) < 30.0, 2.0e6, 0.0)
    a = moment_matrix(GRID, support="free")
    reference = shear_and_moment(GRID, q, support="free")[1]
    assert np.allclose(a @ q, reference, rtol=1e-9, atol=1e-9 * np.max(np.abs(reference)))


def test_moment_statistics_matches_monte_carlo():
    sigma = 1.0e6
    cov = exponential_covariance(GRID, 1e-3, sigma=sigma)
    analytic, _ = moment_statistics(GRID, cov, support="simple")
    z = sample_correlated_field(GRID, 1e-3 * (GRID[1] - GRID[0]), sigma=sigma, n_real=20000,
                                seed=7)
    _, m = moment_batch(GRID, z, support="simple")
    mc_std = np.std(m[:, GRID.size // 2], ddof=1)
    assert mc_std == pytest.approx(analytic[GRID.size // 2], rel=0.03)


def test_independent_segment_variance_scaling():
    sigma, n = 3.0e5, 10
    res = independent_segment_variance(GRID, n, sigma, support="simple")
    scale = n * sigma ** 2 * (200.0 / n) ** 2
    assert res["scale_n_sigma2_l2"] == pytest.approx(scale)
    assert res["ratio_to_scale"] > 0.0
    # the discrete influence line of bay i, evaluated from the same operator
    a = moment_matrix(GRID, support="simple")
    l = 200.0 / n
    factors = []
    for i in range(n):
        indicator = ((GRID >= GRID[0] + i * l - 1e-9) & (GRID <= GRID[0] + (i + 1) * l + 1e-9))
        factors.append((a @ indicator.astype(float))[GRID.size // 2] / l)
    expected = float(np.mean(np.square(factors)))
    assert res["ratio_to_scale"] == pytest.approx(expected, rel=1e-6)


# -------------------------------------------------------------------- ice load
def test_correlation_length_trends():
    base = IceCondition(1.0, 0.8, 2.0)
    assert correlation_length(IceCondition(2.0, 0.8, 2.0)) > correlation_length(base)
    assert correlation_length(IceCondition(1.0, 1.0, 2.0)) > correlation_length(base)
    assert correlation_length(IceCondition(1.0, 0.8, 6.0)) < correlation_length(base)
    assert correlation_length(base) == pytest.approx(REFERENCE_CORRELATION_COEFFICIENTS["lambda_ref_m"])


def test_correlation_length_regression_recovers_coefficients():
    truth = {
        "lambda_ref_m": 2.4,
        "h_ref_m": 1.0,
        "c_ref": 0.8,
        "v_ref_m_s": 2.0,
        "thickness_exponent": 0.9,
        "concentration_exponent": 0.3,
        "speed_exponent": -0.15,
    }
    records = []
    for h in (0.5, 1.0, 1.5, 2.0, 3.0):
        for c in (0.5, 0.8, 1.0):
            for v in (1.0, 2.0, 4.0):
                lam = (
                    truth["lambda_ref_m"]
                    * (h / 1.0) ** truth["thickness_exponent"]
                    * (c / 0.8) ** truth["concentration_exponent"]
                    * (v / 2.0) ** truth["speed_exponent"]
                )
                records.append(
                    {"thickness": h, "concentration": c, "speed": v, "correlation_length": lam}
                )
    fitted = fit_correlation_length_model(records)
    assert fitted["thickness_exponent"] == pytest.approx(0.9, abs=1e-6)
    assert fitted["speed_exponent"] == pytest.approx(-0.15, abs=1e-6)
    assert fitted["lambda_ref_m"] == pytest.approx(2.4, rel=1e-6)
    assert fitted["provenance"] == "fitted"


def test_global_ice_pressure_decreases_with_thickness():
    assert global_ice_pressure(IceCondition(1.0, 0.8, 2.0)) == pytest.approx(1.2e6)
    assert global_ice_pressure(IceCondition(2.0, 0.8, 2.0)) < 1.2e6


def test_pressure_profiles_are_patch_limited_and_non_negative():
    cond = IceCondition(1.5, 0.8, 2.0)
    lam = correlation_length(cond)
    patch = contact_patch_length(cond, lam)
    p = ice_pressure_profiles(GRID, cond, lam, centre=0.0, n_real=50, seed=8)
    assert p.shape == (50, GRID.size)
    assert np.min(p) >= 0.0
    outside = np.abs(GRID) > 0.5 * patch
    assert np.all(p[:, outside] == 0.0)
    assert np.mean(p[:, ~outside]) == pytest.approx(global_ice_pressure(cond), rel=0.35)


def test_contact_patch_length_floor_and_scenarios():
    cond = IceCondition(1.5, 0.8, 2.0)
    assert contact_patch_length(cond, lam=100.0) == pytest.approx(200.0)
    assert contact_patch_length(cond, lam=0.5) == pytest.approx(9.0)
    centres = ice_load_scenarios(GRID, cond, correlation_length(cond), n_positions=5)
    assert len(centres) == 5
    assert centres[0] < 0.0 < centres[-1]


# ------------------------------------------------------------------------- wave
def test_rule_coefficient_reference_values():
    assert rule_coefficient_c(200.0) == pytest.approx(9.75)
    assert rule_coefficient_c(300.0) == pytest.approx(10.75)
    assert rule_coefficient_c(350.0) == pytest.approx(10.75)


def test_ittc_design_wave_moment_reference():
    res = ittc_design_wave_moment(200.0, 32.0, 0.8)
    assert res["hogging_Nm"] == pytest.approx(0.19 * 9.75 * 200.0 ** 2 * 32.0 * 0.8 * 1e3)
    assert res["sagging_Nm"] == pytest.approx(
        -0.11 * 9.75 * 200.0 ** 2 * 32.0 * (0.8 + 0.7) * 1e3
    )
    assert abs(res["sagging_Nm"]) / res["hogging_Nm"] == pytest.approx(1.0855, abs=1e-3)


def test_long_wave_solver_matches_closed_form():
    barge = ShipGirder(waterplane_exponent=0.0)
    wavelength = 50.0 * barge.length
    omega = math.sqrt(2.0 * math.pi * 9.81 / wavelength)
    m = wave_moment_rao(barge, omega, amplitude=1.0, smith=True)
    numeric = abs(m[m.size // 2])
    closed = long_wave_closed_form_moment(barge.length, barge.breadth, wavelength, 1.0)
    assert numeric == pytest.approx(closed, rel=0.01)


def test_design_wave_moment_within_10pct_of_rule():
    ship = ShipGirder()
    design = design_wave_moment(ship)
    rule = ittc_design_wave_moment(ship.length, ship.breadth, ship.block_coefficient)
    assert abs(design["M_max_Nm"] - rule["hogging_Nm"]) / rule["hogging_Nm"] < 0.10


def test_ittc_spectrum_moment_matches_hs():
    omega = np.linspace(1e-3, 4.0, 200001)
    for hs, t1 in [(3.0, 6.0), (6.0, 8.0), (11.0, 11.0)]:
        assert np.trapezoid(ittc_spectrum(omega, hs, t1), omega) == pytest.approx(
            hs ** 2 / 16.0, rel=0.01
        )


# ---------------------------------------------------------------- ultimate / UR
def test_plate_buckling_reduction_reference():
    assert plate_buckling_reduction(0.8) == pytest.approx(1.0)
    assert plate_buckling_reduction(1.7) == pytest.approx(2.25 / 1.7 - 1.25 / 1.7 ** 2, rel=1e-9)


def test_ultimate_moment_and_utilisation():
    ship = ShipGirder()
    ult = ultimate_bending_moment(ship)
    assert ult["M_ultimate_Nm"] == pytest.approx(
        ship.section_modulus * ship.yield_stress * ult["buckling_reduction_phi"]
    )
    ok = utilisation_check(0.5 * ult["M_ultimate_Nm"], ult["M_ultimate_Nm"])
    assert ok["passes"] and ok["utilisation"] == pytest.approx(0.5 * 1.1, rel=1e-9)
    bad = utilisation_check(ult["M_ultimate_Nm"], ult["M_ultimate_Nm"])
    assert not bad["passes"]


# -------------------------------------------------------------------- extremes
def test_gumbel_fit_and_return_level_recovery():
    rng = np.random.default_rng(9)
    samples = 3.0e8 + 1.0e8 * (-np.log(-np.log(rng.random(20000))))
    fit = gumbel_fit(samples)
    assert fit["loc"] == pytest.approx(3.0e8, rel=0.02)
    assert fit["scale"] == pytest.approx(1.0e8, rel=0.02)
    level = gumbel_return_level(fit, 1e-3)
    assert level > np.quantile(samples, 0.99)


def test_weibull_fit_recovers_shape():
    rng = np.random.default_rng(10)
    samples = 1.0e8 * rng.weibull(2.0, 20000)
    fit = weibull_fit(samples)
    assert fit["shape"] == pytest.approx(2.0, rel=0.05)
    assert fit["scale"] == pytest.approx(1.0e8, rel=0.05)
    assert weibull_return_level(fit, 1e-2) > np.quantile(samples, 0.98)


def test_spearman_rank_correlation_bounds():
    a = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
    assert spearman_rank_correlation(a, a ** 2) == pytest.approx(1.0)
    assert spearman_rank_correlation(a, -a) == pytest.approx(-1.0)


def test_moment_batch_matches_scalar_path():
    q = np.vstack([np.ones(GRID.size) * 1e6, np.linspace(0.0, 2e6, GRID.size)])
    _, m = moment_batch(GRID, q, support="free")
    for row, q_row in zip(m, q):
        assert np.allclose(row, shear_and_moment(GRID, q_row, support="free")[1])


def test_ice_bending_moment_analytic_vs_mc_std():
    ship = ShipGirder()
    cond = IceCondition(1.5, 0.8, 2.0)
    lam = correlation_length(cond)
    res = ice_bending_extremes(ship, cond, lam, n_real=6000, seed=11)
    analytic = np.array(res["analytic_std_profile_Nm"])
    # MC standard deviation of the moment over realisations (patch at midship)
    from tensorphysics.hullgirder.ice_load import pressure_coefficient_of_variation

    mu = res["mean_pressure_Pa"]
    sigma = pressure_coefficient_of_variation() * mu * cond.thickness
    x = ship.grid()
    patch = res["contact_patch_length_m"]
    mask = (np.abs(x) <= 0.5 * patch).astype(float)
    z = sample_correlated_field(x, lam, sigma=sigma, n_real=6000, seed=12)
    _, m = moment_batch(x, (mu * cond.thickness + z) * mask[None, :], support="free")
    mc_std = np.std(m[:, x.size // 2], ddof=1)
    assert mc_std == pytest.approx(analytic[x.size // 2], rel=0.06)
    assert res["M_max_abs_Nm"].mean() > 0.9 * mu * cond.thickness * patch


def test_combine_design_moments_governing_case():
    wave_leading = combine_design_moments(still_water=1.0, wave=5.0, ice=1.0)
    ice_leading = combine_design_moments(still_water=1.0, wave=5.0, ice=10.0)
    assert wave_leading["governing_case"] == "wave_leading"
    assert wave_leading["governing_Nm"] == pytest.approx(1.0 + 5.0 + 0.7 * 1.0)
    assert ice_leading["governing_case"] == "ice_leading"
    assert ice_leading["governing_Nm"] == pytest.approx(1.0 + 10.0 + 0.7 * 5.0)


# ---------------------------------------------------------------- acceptance
def test_acceptance_uniform_degeneracy():
    res = check_uniform_degeneracy(n_real=2000)
    assert res["passed"]
    assert res["relative_error_numeric"] < 0.01


def test_acceptance_independent_limit():
    res = check_independent_limit(n_real=8000)
    assert res["passed"]


def test_acceptance_wave_moment():
    res = check_wave_moment()
    assert res["passed"]
    assert res["relative_error_hogging"] < 0.10