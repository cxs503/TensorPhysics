"""IN13-W4-N05 — maturation of the ISO 19906 code-calibration inverse problem.

This module turns the *bisection-only* back-calibration of
:mod:`~tensorphysics.ice_partial_factors.calibration` into a **mature,
machine-auditable** inverse problem with the four ingredients the campaign
node requires:

1. **Newton inverse** (single- and two-variable branches) of
   ``beta(gamma) = beta_T = 4.0`` with an explicit Jacobian, its condition
   number (uniqueness certificate), and convergence from three distinct
   initial values to the same ``gamma*``.
2. **Analytical sensitivity** ``d beta / d gamma`` from the first-order
   reliability-index sensitivity
   ``d beta / d theta = (partial g / partial theta) / |grad g(u*)|``
   (theta enters only through the design thickness ``t_nom(k)``), cross-checked
   against a central finite difference.
3. **Negative controls**: (i) ignoring the p-A negative correlation
   (pressure-area law) systematically *overestimates* ``gamma*`` for panels
   larger than the reference area; (ii) driving the resistance COV to zero
   drives the reliability index to the degenerate ``beta -> inf`` limit.
4. **Code adjudication**: row-by-row deviation of the calibrated factors from
   the ISO 19906 tabulated partial factors, with a "non-conservative" counter
   example for any deviation beyond 15 %.

Everything is produced by code and bound to an artifacts JSON (no hand-copying).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Callable, Sequence

import numpy as np

from ..structural.distributions import norm_cdf
from .limit_state import (
    CharacteristicValues,
    IceBeltPanel,
    build_problem,
    characteristic_values,
    design_thickness_for_factor,
)
from .pressure_area import GumbelParams

#: Mission target: annual reliability index (annual PF = Phi(-4) = 3.167e-5).
TARGET_BETA = 4.0
#: Euler-Mascheroni constant (Gumbel mean = loc + gamma_E * scale).
_EULER_GAMMA = 0.5772156649015329


# --------------------------------------------------------------------------- #
# Core: beta(k) and its analytical sensitivity
# --------------------------------------------------------------------------- #
def beta_of_product(
    panel: IceBeltPanel,
    gumbel: GumbelParams,
    char: CharacteristicValues,
    k: float,
) -> tuple[float, float, object]:
    """Achieved FORM reliability index of the structure designed with product ``k``.

    Returns ``(beta, t_nom_mm, form_result)``; ``form_result`` is ``None`` when
    the median point already fails (beta reported as 0).
    """
    t_nom = design_thickness_for_factor(panel, char, k)
    designed = panel.with_thickness(t_nom)
    problem = build_problem(designed, gumbel)
    if problem.g_u([0.0] * problem.transform.dimension) <= 0.0:
        return 0.0, t_nom, None
    form = problem.form()
    return form.beta, t_nom, form


def dbeta_dgamma(
    panel: IceBeltPanel,
    char: CharacteristicValues,
    k: float,
    form,
    gamma_f: float,
    gamma_m: float,
    t_nom_mm: float | None = None,
) -> tuple[float, float]:
    """Analytical ``(d beta / d gamma_F, d beta / d gamma_M)`` at the design point.

    First-order sensitivity of the reliability index: with the design point
    ``u*`` and ``alpha = -grad g / |grad g|``,

        d beta / d theta = (partial g_u / partial theta) / |grad_u g(u*)|

    because ``u*/|u*| = alpha`` and the constraint ``g(u*) = 0`` gives
    ``alpha . (du*/d theta) = (partial g / partial theta)/|grad g|``.  Here
    ``theta`` is either partial factor and enters only through
    ``t_nom(k) = b sqrt(alpha_b p_k k / R_k) + dt_mean`` with ``k = gamma_F gamma_M``:

        partial g / partial t_nom = 2 k_S alpha_b p b^2 / t_net^3
        partial t_nom / partial k  = b sqrt(alpha_b p_k / R_k) / (2 sqrt(k))
    """
    p_ice, _fy, dt_corr, k_s, _k_r = form_design_point(form)
    t_nom = t_nom_mm if t_nom_mm is not None else panel.t_nom_mm
    t_net = t_nom - dt_corr
    if t_net <= 0.0:
        raise ValueError("net thickness non-positive at the design point")
    dg_dt = 2.0 * k_s * panel.alpha_b * p_ice * panel.b_mm ** 2 / t_net ** 3
    dt_dk = (
        0.5
        * panel.b_mm
        * math.sqrt(panel.alpha_b * char.p_exceedance_mpa / char.r_k_mpa)
        / math.sqrt(k)
    )
    dbeta_dk = (dg_dt * dt_dk) / form.gradient_norm
    return dbeta_dk * gamma_m, dbeta_dk * gamma_f


def form_design_point(form) -> tuple[float, float, float, float, float]:
    """Physical design-point values ``(p_ice, fy, dt_corr, k_S, k_R)``."""
    return tuple(form.design_point_x)  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Newton inverse (single- and two-variable)
# --------------------------------------------------------------------------- #
@dataclass
class NewtonResult:
    gamma: float
    beta: float
    iterations: int
    converged: bool
    residual: float
    history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "gamma": self.gamma,
            "beta": self.beta,
            "iterations": self.iterations,
            "converged": self.converged,
            "residual": self.residual,
            "history": self.history,
        }


def newton_single_variable(
    panel: IceBeltPanel,
    gumbel: GumbelParams,
    char: CharacteristicValues,
    gamma_m_fixed: float,
    beta_target: float = TARGET_BETA,
    gamma_f0: float = 1.5,
    tol: float = 1e-10,
    max_iterations: int = 60,
) -> NewtonResult:
    """Newton on ``f(gamma_F) = beta(gamma_F gamma_M) - beta_T`` with ``gamma_M`` fixed.

    The Jacobian is the scalar ``f' = d beta / d gamma_F`` (analytical).
    """
    gamma_f = float(gamma_f0)
    history: list[dict] = []
    beta = 0.0
    for it in range(max_iterations):
        k = gamma_f * gamma_m_fixed
        beta, _t, form = beta_of_product(panel, gumbel, char, k)
        if form is None:
            # far below the design range: nudge up and continue
            gamma_f *= 1.5
            continue
        f = beta - beta_target
        dbeta_df, _ = dbeta_dgamma(panel, char, k, form, gamma_f, gamma_m_fixed, t_nom_mm=_t)
        history.append(
            {"iter": it, "gamma_f": gamma_f, "beta": beta, "f": f, "df_dgamma": dbeta_df}
        )
        if abs(f) < tol:
            return NewtonResult(gamma_f, beta, it, True, f, history)
        if dbeta_df == 0.0:
            break
        gamma_f -= f / dbeta_df
        if gamma_f <= 0.0:
            gamma_f = 0.5 * gamma_f + 1e-3
    return NewtonResult(gamma_f, beta, max_iterations, False, beta - beta_target, history)


def newton_two_variable(
    panel: IceBeltPanel,
    gumbel: GumbelParams,
    char: CharacteristicValues,
    beta_target: float = TARGET_BETA,
    ratio: float = 1.0,
    gamma_f0: float = 1.5,
    gamma_m0: float = 1.1,
    tol: float = 1e-10,
    max_iterations: int = 60,
) -> dict:
    """Newton on the 2x2 system

        F1(gamma_F, gamma_M) = beta(gamma_F gamma_M) - beta_T = 0
        F2(gamma_F, gamma_M) = ln gamma_F - ln gamma_M - ln(ratio) = 0

    ``ratio`` fixes the split convention (from the JCSS design-value split);
    the reliability equation fixes the product.  Returns the solution, the
    Jacobian at the solution and its condition number (uniqueness certificate).
    """
    gf, gm = float(gamma_f0), float(gamma_m0)
    history: list[dict] = []
    beta = 0.0
    jac = np.zeros((2, 2))
    for it in range(max_iterations):
        k = gf * gm
        beta, _t, form = beta_of_product(panel, gumbel, char, k)
        if form is None:
            gf *= 1.5
            continue
        f1 = beta - beta_target
        f2 = math.log(gf) - math.log(gm) - math.log(ratio)
        dbeta_df, dbeta_dm = dbeta_dgamma(panel, char, k, form, gf, gm, t_nom_mm=_t)
        jac = np.array(
            [
                [dbeta_df, dbeta_dm],
                [1.0 / gf, -1.0 / gm],
            ]
        )
        det = jac[0, 0] * jac[1, 1] - jac[0, 1] * jac[1, 0]
        history.append(
            {"iter": it, "gamma_f": gf, "gamma_m": gm, "f1": f1, "f2": f2, "det": det}
        )
        if abs(f1) < tol and abs(f2) < tol:
            break
        if det == 0.0:
            break
        # solve J [dF, dM] = -F
        dgf = (-f1 * jac[1, 1] + f2 * jac[0, 1]) / det
        dgm = (-f2 * jac[0, 0] + f1 * jac[1, 0]) / det
        gf += dgf
        gm += dgm
        if gf <= 0.0:
            gf = 0.5 * gf + 1e-3
        if gm <= 0.0:
            gm = 0.5 * gm + 1e-3
    k = gf * gm
    beta, _t, form = beta_of_product(panel, gumbel, char, k)
    dbeta_df, dbeta_dm = (
        dbeta_dgamma(panel, char, k, form, gf, gm, t_nom_mm=_t)
        if form is not None
        else (0.0, 0.0)
    )
    jac = np.array([[dbeta_df, dbeta_dm], [1.0 / gf, -1.0 / gm]])
    s = np.linalg.svd(jac, compute_uv=False)
    cond = float(s[0] / s[1]) if s[1] > 0 else float("inf")
    return {
        "gamma_f": gf,
        "gamma_m": gm,
        "k": k,
        "beta": beta,
        "residual": beta - beta_target,
        "iterations": it + 1,
        "converged": abs(f1) < tol and abs(f2) < tol,
        "jacobian": jac.tolist(),
        "condition_number": cond,
        "history": history,
    }


def multi_start_spread(
    panel: IceBeltPanel,
    gumbel: GumbelParams,
    char: CharacteristicValues,
    gamma_m_fixed: float,
    starts: Sequence[float] = (1.0, 1.5, 2.5),
    beta_target: float = TARGET_BETA,
) -> dict:
    """Solve the single-variable inverse from several starts; report the spread."""
    sols = [
        newton_single_variable(
            panel, gumbel, char, gamma_m_fixed, beta_target, gamma_f0=s
        )
        for s in starts
    ]
    gammas = [s.gamma for s in sols]
    spread = (max(gammas) - min(gammas)) / (0.5 * (max(gammas) + min(gammas)))
    return {
        "solutions": [s.to_dict() for s in sols],
        "gammas": gammas,
        "relative_spread": spread,
        "all_converged": all(s.converged for s in sols),
    }


# --------------------------------------------------------------------------- #
# Sensitivity: analytical vs finite difference
# --------------------------------------------------------------------------- #
def sensitivity_check(
    panel: IceBeltPanel,
    gumbel: GumbelParams,
    char: CharacteristicValues,
    gamma_f: float,
    gamma_m: float,
    h_rel: float = 1e-4,
) -> dict:
    """Compare analytical ``d beta / d gamma_F`` with a central finite difference."""
    k = gamma_f * gamma_m
    beta, _t, form = beta_of_product(panel, gumbel, char, k)
    if form is None:
        raise ValueError("design point does not exist at the base point")
    ana_f, ana_m = dbeta_dgamma(panel, char, k, form, gamma_f, gamma_m, t_nom_mm=_t)
    kp = beta_of_product(panel, gumbel, char, k * (1.0 + h_rel))[0]
    km = beta_of_product(panel, gumbel, char, k * (1.0 - h_rel))[0]
    fd_dk = (kp - km) / (2.0 * k * h_rel)
    fd_f = fd_dk * gamma_m
    fd_m = fd_dk * gamma_f
    rel_f = abs(ana_f - fd_f) / max(abs(fd_f), 1e-12)
    rel_m = abs(ana_m - fd_m) / max(abs(fd_m), 1e-12)
    return {
        "gamma_f": gamma_f,
        "gamma_m": gamma_m,
        "beta": beta,
        "analytic_dbeta_dgamma_f": ana_f,
        "analytic_dbeta_dgamma_m": ana_m,
        "fd_dbeta_dgamma_f": fd_f,
        "fd_dbeta_dgamma_m": fd_m,
        "rel_error_gamma_f": rel_f,
        "rel_error_gamma_m": rel_m,
    }


# --------------------------------------------------------------------------- #
# Negative control 1: p-A negative correlation
# --------------------------------------------------------------------------- #
def _gumbel_at_area(
    p_ref: float,
    a_ref: float,
    area: float,
    alpha: float,
    storm_cov: float,
    storms_per_year: int,
    spatial_averaging: bool = True,
) -> GumbelParams:
    """Annual-maximum Gumbel at ``area`` from the p-A law.

    ``p = p_ref (A/A_ref)^-alpha`` sets the mean; the storm COV is reduced by
    spatial averaging of the pressure field, ``cov(A) = cov_ref (A_ref/A)^(1/2)``
    (variance of an area average falls as 1/A).  ``spatial_averaging=False``
    reproduces the "p-A correlation ignored" case (reference statistics for all
    areas).
    """
    m = p_ref * (area / a_ref) ** (-alpha)
    cov = storm_cov * (a_ref / area) ** 0.5 if spatial_averaging else storm_cov
    s = cov * m
    loc = m - _EULER_GAMMA * s
    return GumbelParams(loc=loc + s * math.log(storms_per_year), scale=s)


def p_area_negative_correlation_control(
    panel: IceBeltPanel,
    p_ref: float,
    a_ref: float,
    alpha: float,
    areas: Sequence[float],
    storm_cov: float = 0.22,
    storms_per_year: int = 12,
    gamma_m_fixed: float = 1.10,
    beta_target: float = TARGET_BETA,
) -> list[dict]:
    """For ``A > A_ref``: ignoring the p-A law overestimates ``gamma*``.

    "With p-A" calibrates at the area-specific pressure; "without p-A" uses the
    reference-area pressure for every panel.  The relative overestimation
    ``(gamma*_noPA - gamma*_PA) / gamma*_PA`` must be >= 5 % for ``A > A_ref``.
    """
    rows = []
    for area in areas:
        g_with = _gumbel_at_area(
            p_ref, a_ref, area, alpha, storm_cov, storms_per_year, spatial_averaging=True
        )
        g_without = _gumbel_at_area(
            p_ref, a_ref, a_ref, alpha, storm_cov, storms_per_year, spatial_averaging=False
        )
        p_area = panel.with_area(area)
        ch_with = characteristic_values(p_area, g_with)
        ch_without = characteristic_values(p_area, g_without)
        sol_with = newton_single_variable(
            p_area, g_with, ch_with, gamma_m_fixed, beta_target, gamma_f0=1.5
        )
        sol_without = newton_single_variable(
            p_area, g_without, ch_without, gamma_m_fixed, beta_target, gamma_f0=1.5
        )
        over = (sol_without.gamma - sol_with.gamma) / sol_with.gamma
        rows.append(
            {
                "area_m2": area,
                "p_k_with_pa_mpa": ch_with.p_exceedance_mpa,
                "p_k_without_pa_mpa": ch_without.p_exceedance_mpa,
                "gamma_star_with_pa": sol_with.gamma,
                "gamma_star_without_pa": sol_without.gamma,
                "relative_overestimation": over,
                "detected": bool(over >= 0.05) if area > a_ref else None,
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Negative control 2: resistance COV -> 0  =>  beta -> inf
# --------------------------------------------------------------------------- #
def zero_cov_degenerate(
    panel: IceBeltPanel,
    gumbel: GumbelParams,
    cov_scales: Sequence[float] = (1.0, 0.5, 0.2, 0.1, 0.05, 0.0),
) -> dict:
    """Drive all COVs to zero and show the degenerate ``beta -> inf`` limit.

    At ``cov_scale = 0`` the limit state is deterministic; if it is safe
    (``g_det > 0``) the failure probability is exactly zero and ``beta = inf``.
    """
    rows = []
    for cs in cov_scales:
        p = replace(
            panel,
            fy_cov=panel.fy_cov * cs,
            k_r_cov=panel.k_r_cov * cs,
            k_s_cov=panel.k_s_cov * cs,
            dt_cov=panel.dt_cov * cs,
        )
        if cs == 0.0:
            # deterministic evaluation at the means
            t_net = p.t_nom_mm - p.dt_mean_mm
            g_det = (
                p.fy_mean_mpa
                - p.alpha_b * gumbel.mean() * (p.b_mm / t_net) ** 2
            )
            rows.append(
                {
                    "cov_scale": cs,
                    "beta": float("inf") if g_det > 0 else 0.0,
                    "g_deterministic": g_det,
                    "pf": 0.0 if g_det > 0 else 1.0,
                }
            )
            continue
        g_small = GumbelParams(loc=gumbel.loc, scale=max(gumbel.scale * cs, 1e-9))
        ch_small = characteristic_values(p, g_small)
        sol = newton_single_variable(p, g_small, ch_small, 1.10, TARGET_BETA, gamma_f0=1.5)
        beta = sol.beta
        rows.append(
            {
                "cov_scale": cs,
                "beta": beta,
                "g_deterministic": None,
                "pf": norm_cdf(-beta) if math.isfinite(beta) else 0.0,
            }
        )
    betas = [r["beta"] for r in rows if math.isfinite(r["beta"])]
    monotone = all(betas[i] < betas[i + 1] for i in range(len(betas) - 1))
    return {
        "rows": rows,
        "beta_tends_to_inf": bool(monotone and rows[-1]["beta"] == float("inf")),
        "monotone_increasing": bool(monotone),
    }


# --------------------------------------------------------------------------- #
# Code adjudication vs ISO 19906 tables
# --------------------------------------------------------------------------- #
def iso_table_comparison(
    calibrated: dict,
    load_rows: Sequence,
    resistance_rows: Sequence,
    tolerance: float = 0.15,
) -> dict:
    """Row-by-row deviation of calibrated factors from the ISO 19906 tables.

    Any row whose deviation exceeds ``tolerance`` is flagged and a
    non-conservative counter-example is required in the report.
    """
    gamma_f = calibrated["gamma_F"]
    gamma_m = calibrated["gamma_M"]
    load_devs = []
    for row in load_rows:
        dev = (gamma_f - row.value) / row.value
        load_devs.append(
            {
                "row": row.row_key,
                "iso_factor": row.value,
                "calibrated": gamma_f,
                "relative_deviation": dev,
                "beyond_tolerance": bool(abs(dev) > tolerance),
                "verdict": _verdict(dev, +1),
            }
        )
    res_devs = []
    for row in resistance_rows:
        dev = (gamma_m - row.value) / row.value
        res_devs.append(
            {
                "row": row.row_key,
                "iso_factor": row.value,
                "calibrated": gamma_m,
                "relative_deviation": dev,
                "beyond_tolerance": bool(abs(dev) > tolerance),
                "verdict": _verdict(dev, -1),
            }
        )
    beyond = [r for r in load_devs + res_devs if r["beyond_tolerance"]]
    return {
        "load_factor_rows": load_devs,
        "resistance_factor_rows": res_devs,
        "n_beyond_tolerance": len(beyond),
        "beyond_tolerance_rows": beyond,
        "tolerance": tolerance,
    }


def _verdict(dev: float, conservative_sign: int) -> str:
    """``dev > 0`` is conservative for a load factor, non-conservative for resistance."""
    if abs(dev) <= 0.05:
        return "CONSISTENT"
    if dev * conservative_sign > 0:
        return "CONSERVATIVE"
    return "NON_CONSERVATIVE" if abs(dev) > 0.15 else "MINOR_UNCONSERVATIVE"


# --------------------------------------------------------------------------- #
# Study driver
# --------------------------------------------------------------------------- #
def run_code_calibration_study(
    panel: IceBeltPanel,
    gumbel: GumbelParams,
    char: CharacteristicValues | None = None,
    gamma_m_fixed: float = 1.10,
    ratio: float = 1.0,
    beta_target: float = TARGET_BETA,
    samples_is: int = 200_000,
    samples_mc: int = 400_000,
    seed: int = 2026,
) -> dict:
    """Full maturation study: Newton inverse + sensitivity + negative controls."""
    ch = char if char is not None else characteristic_values(panel, gumbel)
    # -- single-variable Newton from three starts ---------------------------
    multi = multi_start_spread(panel, gumbel, ch, gamma_m_fixed)
    base = newton_single_variable(
        panel, gumbel, ch, gamma_m_fixed, beta_target, gamma_f0=1.5
    )
    k_star = base.gamma * gamma_m_fixed
    # -- two-variable Newton -------------------------------------------------
    two = newton_two_variable(
        panel, gumbel, ch, beta_target, ratio=ratio, gamma_f0=1.5, gamma_m0=1.1
    )
    # -- sensitivity ---------------------------------------------------------
    sens = sensitivity_check(panel, gumbel, ch, base.gamma, gamma_m_fixed)
    # -- simulation cross-check at the calibrated design ---------------------
    t_design = design_thickness_for_factor(panel, ch, k_star)
    designed = panel.with_thickness(t_design)
    ch_design = characteristic_values(designed, gumbel)
    problem = build_problem(designed, gumbel)
    form = problem.form()
    is_res = problem.importance_sampling_pf(samples=samples_is, seed=seed, form_result=form)
    mc_res = problem.monte_carlo_pf(samples=samples_mc, seed=seed + 1)
    beta_form = form.beta
    beta_is = -1.0 * _beta_from_pf(is_res.pf)
    beta_mc = _beta_from_pf(mc_res.pf)
    is_se = float(is_res.std_error) / is_res.pf if is_res.pf > 0 else 0.0
    return {
        "beta_target": beta_target,
        "single_variable": {
            "gamma_m_fixed": gamma_m_fixed,
            "gamma_f_star": base.gamma,
            "k_star": k_star,
            "beta_at_star": base.beta,
            "residual": base.residual,
            "iterations": base.iterations,
            "converged": base.converged,
            "multi_start": multi,
        },
        "two_variable": two,
        "sensitivity": sens,
        "simulation": {
            "beta_form": beta_form,
            "beta_importance_sampling": beta_is,
            "beta_monte_carlo": beta_mc,
            "is_std_error_rel": is_se,
            "samples_is": samples_is,
            "samples_mc": samples_mc,
            "form_vs_is_rel": abs(beta_form - beta_is) / beta_form,
            "is_covers_form": bool(
                beta_is - 1.0 * _beta_sigma(beta_is, is_res.pf, samples_is)
                <= beta_form
                <= beta_is + 1.0 * _beta_sigma(beta_is, is_res.pf, samples_is)
            ),
        },
        "characteristic": ch.to_dict(),
        "t_design_mm": t_design,
    }


def _beta_from_pf(pf: float) -> float:
    if pf <= 0.0:
        return float("inf")
    from ..structural.distributions import norm_ppf

    return float(-norm_ppf(pf))


def _beta_sigma(beta: float, pf: float, n: int) -> float:
    """Approximate 1-sigma of the beta estimate from a binomial pf estimator."""
    if pf <= 0.0 or pf >= 1.0:
        return 0.0
    se_pf = math.sqrt(pf * (1.0 - pf) / n)
    # d beta / d pf = -1 / (sqrt(2 pi) phi(beta))
    phi = math.exp(-0.5 * beta * beta) / math.sqrt(2.0 * math.pi)
    return se_pf / (math.sqrt(2.0 * math.pi) * phi)
