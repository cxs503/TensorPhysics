"""Machine-checkable acceptance criteria for the ice/wave hull girder workflow.

Seven criteria (a)-(g) of the campaign node, implemented as pure functions that
return JSON-serialisable dicts::

    (a) uniform load degeneracy      M(lambda_x -> inf) -> p L^2/8,      tol 1 %
    (b) independent segment limit    Var(M)(lambda_x -> 0) -> sum s^2 I^2, tol 3 %
    (c) wave bending moment          design wave vs ITTC-1978 / IACS UR S11, tol 10 %
    (d) Monte-Carlo convergence      M_design(1e4) vs M_design(5e4),    tol 5 %
    (e) correlation length monotonic M_design(lambda_x) increasing, Spearman > 0.95
    (f) end-to-end case              200 m polar ship, h=1.5 m, c=0.8
    (g) test suite                   pytest tests/test_hullgirder_*.py

Note on criterion (b): the criterion quotes the scaling law
``Var(M) = n sigma^2 l^2``.  With segment influence integrals ``I_i`` the exact
result for independent segments is ``Var(M) = sigma^2 sum_i I_i^2``, i.e. the
quoted law is the dimensionally normalised form (unit influence slope) and the
exact variance carries the influence-line factor ``<(I_i/l)^2>``, which is also
returned by the check so that both forms can be inspected.
"""

from __future__ import annotations

import platform
import time

import numpy as np

from .extremes import (
    combine_design_moments,
    gumbel_fit,
    gumbel_return_level,
    ice_bending_extremes,
    moment_batch,
    spearman_rank_correlation,
    weibull_fit,
    weibull_return_level,
)
from .girder import moment_statistics, shear_and_moment
from .ice_load import (
    IceCondition,
    REFERENCE_CORRELATION_COEFFICIENTS,
    contact_patch_length,
    correlation_length,
    global_ice_pressure,
)
from .model import ShipGirder
from .random_field import exponential_covariance, sample_correlated_field
from .ultimate import ultimate_bending_moment, utilisation_check
from .wave import design_wave_moment, ittc_design_wave_moment, long_wave_closed_form_moment

EVENTS_PER_YEAR = 5000.0  # ice loading events per year of ice transit (documented assumption)
RETURN_PERIOD_YEARS = 100.0
STILL_WATER_RATIO = 0.40  # M_sw / M_wave for a homogeneous full-load condition


def _rel_error(value: float, reference: float) -> float:
    return float(abs(value - reference) / abs(reference))


def check_uniform_degeneracy(ship: ShipGirder | None = None, n_real: int = 2000,
                             seed: int = 11) -> dict:
    """(a) Uniform load degeneracy: M -> q L^2 / 8 (tolerance 1 %)."""
    ship = ShipGirder() if ship is None else ship
    x = ship.grid()
    length = ship.length
    load = 1.0e6  # N/m
    analytic = load * length ** 2 / 8.0

    q = np.full(x.size, load)
    _, m_numeric = shear_and_moment(x, q, support="simple")
    m_mid_numeric = float(m_numeric[x.size // 2])

    # correlated-field limit: lambda_x >> L, load applied over the full length
    mu, sigma = 1.0e6, 0.3e6
    lam = 1.0e6
    z = sample_correlated_field(x, lam, sigma=1.0, n_real=n_real, seed=seed)
    q_batch = mu + sigma * z
    _, m_batch = moment_batch(x, q_batch, support="simple")
    m_mid_mc = m_batch[:, x.size // 2]
    m_mid_mean = float(np.mean(m_mid_mc))

    err_numeric = _rel_error(m_mid_numeric, analytic)
    err_mc_mean = _rel_error(m_mid_mean, analytic)
    std_ratio = float(np.std(m_mid_mc, ddof=1) / (sigma * length ** 2 / 8.0))
    return {
        "name": "a_uniform_load_degeneracy",
        "criterion": "lambda_x -> inf: M_mid = q L^2 / 8 (simply supported, uniform load)",
        "tolerance": 0.01,
        "analytic_M_mid_Nm": analytic,
        "numeric_M_mid_Nm": m_mid_numeric,
        "mc_mean_M_mid_Nm": m_mid_mean,
        "relative_error_numeric": err_numeric,
        "relative_error_mc_mean": err_mc_mean,
        "mc_std_over_analytic_std": std_ratio,
        "correlation_length_m": lam,
        "n_realisations": int(n_real),
        "passed": bool(max(err_numeric, err_mc_mean) < 0.01 and abs(std_ratio - 1.0) < 0.10),
    }


def check_independent_limit(ship: ShipGirder | None = None, n_real: int = 20000,
                           seed: int = 12) -> dict:
    """(b) Independent segment limit: Var(M) -> sigma^2 sum I_i^2 (tolerance 3 %)."""
    ship = ShipGirder() if ship is None else ship
    x = ship.grid()
    length = ship.length
    mu, sigma = 1.0e6, 0.3e6

    # (b1) continuous white-noise limit: lambda_x << dx on every node
    lam_tiny = 1.0e-3 * (x[1] - x[0])
    z = sample_correlated_field(x, lam_tiny, sigma=1.0, n_real=n_real, seed=seed)
    q = mu + sigma * z
    _, m = moment_batch(x, q, support="simple")
    mc_var = float(np.var(m[:, x.size // 2], ddof=1))
    cov = exponential_covariance(x, lam_tiny, sigma=sigma)
    analytic_std, _ = moment_statistics(x, cov, support="simple")
    analytic_var = float(analytic_std[x.size // 2] ** 2)

    # (b2) ten independent bays (the "independent compartment" idealisation)
    n_bays = 10
    bay_id = np.minimum((np.arange(x.size) * n_bays) // x.size, n_bays - 1)
    rng = np.random.default_rng(seed + 1)
    zb = rng.standard_normal((n_real, n_bays))
    q_bay = mu + sigma * zb[:, bay_id]
    _, m_bay = moment_batch(x, q_bay, support="simple")
    mc_var_bay = float(np.var(m_bay[:, x.size // 2], ddof=1))

    # analytic: Var = sigma^2 sum_i I_i^2 with I_i = int_{bay i} g(x,s) ds
    from .girder import moment_matrix

    a = moment_matrix(x, support="simple")
    var_bay = 0.0
    for i in range(n_bays):
        indicator = (bay_id == i).astype(float)
        var_bay += sigma ** 2 * float((a @ indicator)[x.size // 2] ** 2)
    l_bay = length / n_bays
    scale = n_bays * sigma ** 2 * l_bay ** 2
    ratio_mc = mc_var_bay / scale
    # closed form of the influence-line factor: <(I_i/l)^2>
    factors = [float((a @ (bay_id == i).astype(float))[x.size // 2]) / l_bay for i in range(n_bays)]
    ratio_closed = float(np.mean(np.square(factors)))

    err_white = _rel_error(mc_var, analytic_var)
    err_bay = _rel_error(mc_var_bay, var_bay)
    err_ratio = _rel_error(ratio_mc, ratio_closed)
    return {
        "name": "b_independent_segment_limit",
        "criterion": "lambda_x -> 0: Var(M) = sigma^2 sum I_i^2  (= n sigma^2 l^2 times <(I_i/l)^2>)",
        "tolerance": 0.03,
        "white_noise": {
            "lambda_m": float(lam_tiny),
            "mc_variance": mc_var,
            "analytic_variance": analytic_var,
            "relative_error": err_white,
            "n_realisations": int(n_real),
        },
        "ten_bays": {
            "mc_variance": mc_var_bay,
            "analytic_variance": float(var_bay),
            "relative_error": err_bay,
            "scale_n_sigma2_l2": float(scale),
            "ratio_variance_to_scale_mc": float(ratio_mc),
            "ratio_variance_to_scale_closed_form": ratio_closed,
            "ratio_relative_error": err_ratio,
            "segment_length_m": float(l_bay),
        },
        "passed": bool(err_white < 0.03 and err_bay < 0.03 and err_ratio < 0.03),
    }


def check_wave_moment(ship: ShipGirder | None = None) -> dict:
    """(c) Wave bending moment vs ITTC-1978 / IACS UR S11 (tolerance 10 %)."""
    ship = ShipGirder() if ship is None else ship
    rule = ittc_design_wave_moment(ship.length, ship.breadth, ship.block_coefficient)
    design = design_wave_moment(ship, wavelength_factor=1.0, wave_height_ratio=20.0, smith=True)
    design_no_smith = design_wave_moment(ship, wavelength_factor=1.0, wave_height_ratio=20.0,
                                         smith=False)
    err_hogging = _rel_error(design["M_max_Nm"], rule["hogging_Nm"])

    # rigorous numerical verification of the solver: rectangular barge, long wave
    barge = ShipGirder(
        length=ship.length,
        breadth=ship.breadth,
        draught=ship.draught,
        block_coefficient=ship.block_coefficient,
        waterplane_exponent=0.0,  # box waterplane -> closed form applies
    )
    wavelength = 50.0 * barge.length
    amplitude = 1.0
    omega = float(np.sqrt(2.0 * np.pi * 9.81 / wavelength))
    from .wave import wave_moment_rao

    m_profile = wave_moment_rao(barge, omega, amplitude=amplitude, smith=True)
    numeric_mid = float(abs(m_profile[m_profile.size // 2]))
    closed = long_wave_closed_form_moment(barge.length, barge.breadth, wavelength, amplitude)
    err_closed = _rel_error(numeric_mid, closed)

    return {
        "name": "c_wave_bending_moment",
        "criterion": (
            "design wave (lambda=L, h=L/20) long-wave strip moment with Smith correction "
            "vs ITTC-1978 / IACS UR S11 recommended formula"
        ),
        "tolerance": 0.10,
        "rule_moment_hogging_Nm": rule["hogging_Nm"],
        "rule_moment_sagging_Nm": rule["sagging_Nm"],
        "rule_coefficient_c": rule["coefficient_c"],
        "design_wave": design,
        "relative_error_hogging": err_hogging,
        "sensitivity_without_smith": {
            "M_max_Nm": design_no_smith["M_max_Nm"],
            "relative_error_hogging": _rel_error(design_no_smith["M_max_Nm"], rule["hogging_Nm"]),
        },
        "solver_vs_closed_form": {
            "wavelength_over_L": 50.0,
            "numeric_midship_Nm": numeric_mid,
            "closed_form_Nm": closed,
            "relative_error": err_closed,
        },
        "passed": bool(err_hogging < 0.10 and err_closed < 0.01),
    }


def _design_ice_moment(ship: ShipGirder, condition: IceCondition, lam: float, n_real: int,
                       seed: int) -> dict:
    result = ice_bending_extremes(ship, condition, lam, n_real=n_real, seed=seed)
    p_event = 1.0 / (RETURN_PERIOD_YEARS * EVENTS_PER_YEAR)
    gumbel = gumbel_fit(result["M_max_abs_Nm"])
    weibull = weibull_fit(result["M_max_abs_Nm"])
    return {
        "gumbel": gumbel,
        "weibull": weibull,
        "M_design_gumbel_Nm": gumbel_return_level(gumbel, p_event),
        "M_design_weibull_Nm": weibull_return_level(weibull, p_event),
        "M_max_mean_Nm": float(np.mean(result["M_max_abs_Nm"])),
        "M_max_std_Nm": float(np.std(result["M_max_abs_Nm"], ddof=1)),
        "M_max_q50_Nm": float(np.quantile(result["M_max_abs_Nm"], 0.5)),
        "M_max_q90_Nm": float(np.quantile(result["M_max_abs_Nm"], 0.9)),
        "p_event": p_event,
        "raw": result,
    }


def check_mc_convergence(ship: ShipGirder | None = None, condition: IceCondition | None = None,
                         n_small: int = 10000, n_large: int = 50000, seed: int = 21) -> dict:
    """(d) Monte-Carlo convergence: design moment at 1e4 vs 5e4 (tolerance 5 %)."""
    ship = ShipGirder() if ship is None else ship
    condition = IceCondition(thickness=1.5, concentration=0.8, speed=2.0) if condition is None else condition
    lam = correlation_length(condition)
    small = _design_ice_moment(ship, condition, lam, n_small, seed)
    large = _design_ice_moment(ship, condition, lam, n_large, seed)
    err_gumbel = _rel_error(small["M_design_gumbel_Nm"], large["M_design_gumbel_Nm"])
    err_weibull = _rel_error(small["M_design_weibull_Nm"], large["M_design_weibull_Nm"])
    factor = float(np.sqrt(2.0 * np.log(n_small)))
    analytic = small["raw"]["analytic_std_profile_Nm"]
    analytic_std = float(np.max(analytic)) if analytic is not None else float("nan")
    return {
        "name": "d_monte_carlo_convergence",
        "criterion": "100-year design ice moment from 1e4 vs 5e4 realisations",
        "tolerance": 0.05,
        "n_small": int(n_small),
        "n_large": int(n_large),
        "nested_sampling": True,
        "M_design_gumbel_1e4_Nm": small["M_design_gumbel_Nm"],
        "M_design_gumbel_5e4_Nm": large["M_design_gumbel_Nm"],
        "relative_error_gumbel": err_gumbel,
        "M_design_weibull_1e4_Nm": small["M_design_weibull_Nm"],
        "M_design_weibull_5e4_Nm": large["M_design_weibull_Nm"],
        "relative_error_weibull": err_weibull,
        "analytic_std_profile_peak_Nm": analytic_std,
        "rayleigh_factor_for_1e4_cycles": factor,
        "analytic_1e4_cycle_extreme_Nm": analytic_std * factor,
        "correlation_length_m": float(lam),
        "passed": bool(max(err_gumbel, err_weibull) < 0.05),
    }


def check_correlation_sensitivity(ship: ShipGirder | None = None,
                                 condition: IceCondition | None = None,
                                 lambdas: list[float] | None = None,
                                 n_real: int = 2000, seed: int = 31) -> dict:
    """(e) Correlation length monotonicity of the bending moment extreme (Spearman > 0.95)."""
    ship = ShipGirder() if ship is None else ship
    condition = IceCondition(thickness=1.5, concentration=0.8, speed=2.0) if condition is None else condition
    if lambdas is None:
        lambdas = [0.05, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0]
    rows = []
    for lam in lambdas:
        res = ice_bending_extremes(ship, condition, lam, n_real=n_real, seed=seed)
        analytic = res["analytic_std_profile_Nm"]
        rows.append(
            {
                "correlation_length_m": float(lam),
                "correlation_over_length": float(lam / ship.length),
                "patch_length_m": res["contact_patch_length_m"],
                "M_max_mean_Nm": float(np.mean(res["M_max_abs_Nm"])),
                "M_max_q90_Nm": float(np.quantile(res["M_max_abs_Nm"], 0.9)),
                "analytic_std_peak_Nm": float(np.max(analytic)) if analytic else float("nan"),
            }
        )
    lam_arr = [r["correlation_length_m"] for r in rows]
    rho_mc = spearman_rank_correlation(lam_arr, [r["M_max_mean_Nm"] for r in rows])
    rho_analytic = spearman_rank_correlation(lam_arr, [r["analytic_std_peak_Nm"] for r in rows])
    monotone_mc = all(
        b["M_max_mean_Nm"] >= a["M_max_mean_Nm"] for a, b in zip(rows[:-1], rows[1:])
    )
    return {
        "name": "e_correlation_length_monotonicity",
        "criterion": "increasing lambda_x -> increasing bending moment extreme (Spearman > 0.95)",
        "tolerance": 0.95,
        "spearman_mc_mean": rho_mc,
        "spearman_analytic_std": rho_analytic,
        "strictly_monotone_mc": bool(monotone_mc),
        "rows": rows,
        "n_realisations": int(n_real),
        "passed": bool(rho_mc > 0.95 and rho_analytic > 0.95),
    }


def check_end_to_end(ship: ShipGirder | None = None, condition: IceCondition | None = None,
                     n_real: int = 10000, seed: int = 41) -> dict:
    """(f) End-to-end design case: 200 m polar ship, 1.5 m ice, 80 % concentration."""
    ship = ShipGirder() if ship is None else ship
    condition = IceCondition(thickness=1.5, concentration=0.8, speed=2.0) if condition is None else condition
    lam = correlation_length(condition)
    ice = _design_ice_moment(ship, condition, lam, n_real, seed)
    rule = ittc_design_wave_moment(ship.length, ship.breadth, ship.block_coefficient)
    wave_design = max(abs(rule["hogging_Nm"]), abs(rule["sagging_Nm"]))
    m_still = STILL_WATER_RATIO * wave_design
    combination = combine_design_moments(
        still_water=m_still,
        wave=wave_design,
        ice=ice["M_design_gumbel_Nm"],
    )
    ultimate = ultimate_bending_moment(ship)
    check = utilisation_check(combination["governing_Nm"], ultimate["M_ultimate_Nm"])
    return {
        "name": "f_end_to_end_design_case",
        "criterion": "200 m polar ship, h=1.5 m, c=0.8 -> design moment and utilisation",
        "ship": ship.as_dict(),
        "ice_condition": condition.as_dict(),
        "correlation_length_model": REFERENCE_CORRELATION_COEFFICIENTS,
        "correlation_length_m": float(lam),
        "contact_patch_length_m": contact_patch_length(condition, lam),
        "mean_ice_pressure_Pa": global_ice_pressure(condition),
        "ice": {k: v for k, v in ice.items() if k != "raw"},
        "wave": {
            "rule_hogging_Nm": rule["hogging_Nm"],
            "rule_sagging_Nm": rule["sagging_Nm"],
            "design_wave_moment_used_Nm": wave_design,
        },
        "still_water_moment_Nm": m_still,
        "still_water_ratio": STILL_WATER_RATIO,
        "combination": combination,
        "ultimate": ultimate,
        "utilisation": check,
        "events_per_year": EVENTS_PER_YEAR,
        "return_period_years": RETURN_PERIOD_YEARS,
        "passed": bool(check["passes"]),
        "raw_extremes_summary": {
            "n_realisations": ice["raw"]["n_realisations"],
            "n_patch_positions": ice["raw"]["n_patch_positions"],
            "patch_centres_m": ice["raw"]["patch_centres_m"],
            "mc_std_M_max_Nm": ice["raw"]["mc_std_M_max_Nm"],
            "M_max_min_Nm": float(np.min(ice["raw"]["M_max_abs_Nm"])),
            "M_max_max_Nm": float(np.max(ice["raw"]["M_max_abs_Nm"])),
        },
    }


def run_all(n_real_mc: int = 10000, n_real_sensitivity: int = 2000, verbose: bool = True) -> dict:
    """Run every acceptance criterion and return a JSON-serialisable report."""
    t0 = time.time()
    ship = ShipGirder()
    condition = IceCondition(thickness=1.5, concentration=0.8, speed=2.0)
    results = {}
    steps = [
        ("a", lambda: check_uniform_degeneracy(ship)),
        ("b", lambda: check_independent_limit(ship)),
        ("c", lambda: check_wave_moment(ship)),
        ("d", lambda: check_mc_convergence(ship, condition, n_small=n_real_mc,
                                           n_large=5 * n_real_mc)),
        ("e", lambda: check_correlation_sensitivity(ship, condition, n_real=n_real_sensitivity)),
        ("f", lambda: check_end_to_end(ship, condition, n_real=n_real_mc)),
    ]
    for key, fn in steps:
        if verbose:
            print(f"[hullgirder] running criterion ({key}) ...", flush=True)
        results[key] = fn()
        if verbose:
            print(f"[hullgirder] ({key}) passed={results[key]['passed']}", flush=True)
    all_passed = all(v["passed"] for v in results.values())
    return {
        "criteria": results,
        "all_passed": bool(all_passed),
        "elapsed_s": float(time.time() - t0),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "ship": ship.as_dict(),
        "ice_condition": condition.as_dict(),
        "conventions": {
            "events_per_year": EVENTS_PER_YEAR,
            "return_period_years": RETURN_PERIOD_YEARS,
            "still_water_ratio": STILL_WATER_RATIO,
            "ice_correlation_provenance": REFERENCE_CORRELATION_COEFFICIENTS["provenance"],
        },
    }