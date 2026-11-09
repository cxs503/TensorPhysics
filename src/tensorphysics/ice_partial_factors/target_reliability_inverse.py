"""Target-reliability inverse calibration of the ISO 19906 ice-force design format.

IN101-N09 (mature stage of the IN13-W3-N10 chain).  The design format is

    gamma_f * F_k  <=  R_k / psi        (F_k = characteristic ice force,
                                         R_k = characteristic resistance),

and the *calibration* problem is the inverse statement: find the pair
``(gamma_f, psi)`` whose as-designed member attains the target annual
reliability ``beta_T = 4.0``.  The limit state depends on the pair only through
the product ``kappa = gamma_f * psi`` (a member sized with ``R_k = kappa F_k``),
so the inverse is a two-unknown / two-equation system:

    F1(gamma_f, psi) = beta(kappa = gamma_f psi) - beta_T   = 0
    F2(gamma_f, psi) = gamma_f - F*(kappa) / F_k            = 0

``F1`` is the code-calibration condition (a structure designed by the pair
attains exactly the target reliability); ``F2`` is the JCSS/ISO 2394
design-value split (the load factor equals the ratio of the FORM design-point
action to the characteristic action).  ``F1`` fixes ``kappa*`` and ``F2`` splits
it, so the pair is unique up to the convergence of the iteration.

Because only ``kappa`` enters the limit state, ``d beta / d kappa`` follows from
the envelope (adjoint) theorem instead of finite differences: for the FORM
problem ``min ||u||  s.t.  g(u; theta) = 0``,

    d beta / d theta = (1 / ||grad_u g(u*)||) * d g / d theta |_{u*}

(the movement of the design point contributes nothing at first order), and
``kappa`` only shifts the resistance marginal in log space, ``d g/d kappa =
R*/kappa``, hence

    d beta / d kappa    = R* / (kappa ||grad_u g(u*)||)
    d beta / d gamma_f  = (d beta / d kappa) * psi
    d beta / d psi      = (d beta / d kappa) * gamma_f

Those adjoint derivatives are validated against central finite differences of
the re-solved FORM index (mission criterion (c)).
"""
from __future__ import annotations

from math import exp, inf, log, sqrt
from typing import Sequence

from ..structural.distributions import norm_cdf, norm_ppf
from ..structural.reliability import FormResult
from .p_area_joint import (
    PAreaForceModel,
    build_problem,
    characteristic_force,
    design_resistance_median,
)

__all__ = [
    "TARGET_BETA",
    "TARGET_PF",
    "INITIAL_GUESSES",
    "beta_of_kappa",
    "calibrate_kappa",
    "design_split",
    "adjoint_beta_sensitivity",
    "solve_factors",
    "solve_factors_multistart",
    "beta_derivative_check",
    "condition_number",
    "rel_error",
]

#: Mission target: annual reliability index and the corresponding annual Pf.
TARGET_BETA = 4.0
TARGET_PF = norm_cdf(-4.0)  # 3.1671e-5

#: Three deliberately dispersed starting points for the Newton inverse.
INITIAL_GUESSES: tuple[tuple[float, float], ...] = (
    (0.80, 0.80),
    (1.50, 1.50),
    (3.00, 1.00),
)


def rel_error(value: float, reference: float) -> float:
    if reference == 0.0:
        return 0.0 if value == 0.0 else inf
    return (value - reference) / reference


def beta_of_kappa(
    model: PAreaForceModel,
    kappa: float,
    ignore_correlation: bool = False,
) -> tuple[float, FormResult | None, float]:
    """Achieved FORM reliability index of a member sized with ``R_k = kappa F_k``.

    Returns ``(beta, form_result, r_median)``.  If the median point already
    violates the limit state the member has no design point and ``beta = 0`` is
    returned (used as the lower bracket of the calibration).
    """
    problem = build_problem(model, kappa, ignore_correlation=ignore_correlation)
    r_median = design_resistance_median(model, kappa, ignore_correlation)
    u0 = [0.0] * problem.transform.dimension
    if problem.g_u(u0) <= 0.0:
        return 0.0, None, r_median
    form = problem.form()
    return form.beta, form, r_median


def calibrate_kappa(
    model: PAreaForceModel,
    beta_target: float = TARGET_BETA,
    ignore_correlation: bool = False,
    tolerance: float = 1e-10,
    max_iterations: int = 120,
) -> dict:
    """Bisection on ``kappa`` such that ``beta(kappa) = beta_target``.

    ``beta(kappa)`` increases monotonically: a larger ``kappa`` is a larger
    characteristic resistance, i.e. a stronger member.  This is the one-unknown
    cross-check of the two-unknown split (:func:`solve_factors`).
    """
    lo, hi = 0.10, 8.0
    beta_lo = beta_of_kappa(model, lo, ignore_correlation)[0]
    beta_hi = beta_of_kappa(model, hi, ignore_correlation)[0]
    guard = 0
    while (beta_lo - beta_target) * (beta_hi - beta_target) > 0.0 and guard < 20:
        lo = max(lo * 0.5, 1e-3)
        hi = min(hi * 2.0, 1e6)
        beta_lo = beta_of_kappa(model, lo, ignore_correlation)[0]
        beta_hi = beta_of_kappa(model, hi, ignore_correlation)[0]
        guard += 1
    if (beta_lo - beta_target) * (beta_hi - beta_target) > 0.0:  # pragma: no cover
        raise RuntimeError("kappa calibration could not be bracketed")
    iterations = 0
    while iterations < max_iterations and (hi - lo) > tolerance * (0.5 * (lo + hi)):
        mid = 0.5 * (lo + hi)
        beta_mid = beta_of_kappa(model, mid, ignore_correlation)[0]
        iterations += 1
        if (beta_mid - beta_target) * (beta_lo - beta_target) <= 0.0:
            hi, beta_hi = mid, beta_mid
        else:
            lo, beta_lo = mid, beta_mid
    kappa = 0.5 * (lo + hi)
    beta, form, r_median = beta_of_kappa(model, kappa, ignore_correlation)
    return {
        "kappa_star": kappa,
        "beta_achieved": beta,
        "beta_residual": beta - beta_target,
        "bisection_iterations": iterations,
        "bracket": {"lo": lo, "hi": hi, "beta_lo": beta_lo, "beta_hi": beta_hi},
        "r_median_mn": r_median,
        "converged": form is not None and form.converged,
    }


def force_at_design_point(form: FormResult) -> float:
    """``F* = k_F p A`` at the FORM design point."""
    k_f, p_ice, area, _r = form.design_point_x
    return float(k_f * p_ice * area)


def design_split(
    model: PAreaForceModel,
    kappa: float,
    form: FormResult | None = None,
    ignore_correlation: bool = False,
) -> dict:
    """Design-point split ``gamma_f = F*/F_k`` and ``psi = R_k/F*`` at ``kappa``."""
    if form is None:
        _beta, form, _r = beta_of_kappa(model, kappa, ignore_correlation)
    if form is None:  # pragma: no cover - defensive
        raise RuntimeError("no design point for the requested kappa")
    f_star = force_at_design_point(form)
    f_k = characteristic_force(model, ignore_correlation)
    r_k = kappa * f_k
    return {
        "kappa": kappa,
        "F_star_mn": f_star,
        "F_k_mn": f_k,
        "R_k_mn": r_k,
        "gamma_f": f_star / f_k,
        "psi": r_k / f_star,
        "gamma_f_times_psi": kappa,
        "design_point": dict(
            zip(
                ("k_F", "p_ice", "A_contact", "R_member"),
                [float(v) for v in form.design_point_x],
            )
        ),
        "alpha": [float(a) for a in form.alpha],
        "beta": form.beta,
    }


def adjoint_beta_sensitivity(
    model: PAreaForceModel,
    kappa: float,
    form: FormResult | None = None,
    ignore_correlation: bool = False,
) -> dict:
    """Envelope-theorem ``d beta / d kappa`` at a frozen design point.

    ``kappa`` only shifts the resistance marginal in log space
    (``mu_lnR = ln(kappa F_k / q_k)``), hence ``d g / d kappa = R/kappa`` at
    fixed ``u`` and ``d beta / d kappa = R* / (kappa ||grad_u g(u*)||)``.
    """
    problem = build_problem(model, kappa, ignore_correlation=ignore_correlation)
    if form is None:
        _beta, form, _r = beta_of_kappa(model, kappa, ignore_correlation)
    if form is None:  # pragma: no cover - defensive
        raise RuntimeError("no design point for the requested kappa")
    u_star = list(form.design_point_u)
    grad = problem.gradient_u(u_star)
    grad_norm = sqrt(sum(v * v for v in grad))
    x_star = problem.transform.u_to_x(u_star)
    r_star = float(x_star[3])
    dg_dkappa = r_star / kappa
    return {
        "d_beta_d_kappa": dg_dkappa / grad_norm,
        "R_star_mn": r_star,
        "grad_norm": grad_norm,
        "dg_dkappa": dg_dkappa,
    }


def _ratio_at_kappa(
    model: PAreaForceModel, kappa: float, ignore: bool
) -> tuple[float, FormResult | None]:
    """``F*(kappa)/F_k`` -- the design-value load factor implied by ``kappa``."""
    _beta, form, _r = beta_of_kappa(model, kappa, ignore)
    if form is None:
        return 0.0, None
    return force_at_design_point(form) / characteristic_force(model, ignore), form


def _d_ratio_d_kappa(
    model: PAreaForceModel, kappa: float, ignore: bool, relative_step: float = 1e-5
) -> float:
    """Central difference of ``F*(kappa)/F_k`` (Jacobian entry of the inverse)."""
    h = relative_step * kappa
    plus = _ratio_at_kappa(model, kappa + h, ignore)[0]
    minus = _ratio_at_kappa(model, kappa - h, ignore)[0]
    return (plus - minus) / (2.0 * h)


def _residual(
    model: PAreaForceModel, gamma_f: float, psi: float, beta_target: float, ignore: bool
):
    kappa = gamma_f * psi
    beta, form, _r = beta_of_kappa(model, kappa, ignore)
    if form is None:
        return [beta - beta_target, gamma_f], kappa, beta, form, 0.0
    ratio = force_at_design_point(form) / characteristic_force(model, ignore)
    return [beta - beta_target, gamma_f - ratio], kappa, beta, form, ratio


def solve_factors(
    model: PAreaForceModel,
    beta_target: float = TARGET_BETA,
    start: tuple[float, float] = (1.0, 1.0),
    ignore_correlation: bool = False,
    tolerance: float = 1e-10,
    max_iterations: int = 60,
) -> dict:
    """Damped Newton iteration on the two-equation system for ``(gamma_f, psi)``.

    The unknowns are integrated in log space (``y = (ln gamma_f, ln psi)``) so
    both are dimensionless ``O(1)`` quantities; the Jacobian is then well scaled
    and its condition number is a meaningful uniqueness indicator.
    """
    y = [log(start[0]), log(start[1])]
    trace: list[dict] = []
    converged = False
    iterations = 0
    last_jac: list[list[float]] | None = None
    for iterations in range(1, max_iterations + 1):
        gamma_f, psi = exp(y[0]), exp(y[1])
        residual, kappa, beta, form, ratio = _residual(
            model, gamma_f, psi, beta_target, ignore_correlation
        )
        if form is None:  # pragma: no cover - start outside the design range
            raise RuntimeError("starting point outside the design range")
        sens = adjoint_beta_sensitivity(
            model, kappa, form=form, ignore_correlation=ignore_correlation
        )
        d_beta_d_kappa = sens["d_beta_d_kappa"]
        c = _d_ratio_d_kappa(model, kappa, ignore_correlation)
        jac = [
            [d_beta_d_kappa * kappa, d_beta_d_kappa * kappa],
            [gamma_f - c * kappa, -c * kappa],
        ]
        last_jac = jac
        trace.append(
            {
                "iteration": iterations,
                "gamma_f": gamma_f,
                "psi": psi,
                "kappa": kappa,
                "beta": beta,
                "residual_beta": residual[0],
                "residual_split": residual[1],
                "design_point_ratio": ratio,
            }
        )
        if abs(residual[0]) <= 1e-8 and abs(residual[1]) <= tolerance:
            converged = True
            break
        det = jac[0][0] * jac[1][1] - jac[0][1] * jac[1][0]
        if abs(det) < 1e-14:  # pragma: no cover - singular Jacobian
            raise RuntimeError("singular Jacobian in the factor inverse")
        dy = [
            (-residual[0] * jac[1][1] + jac[0][1] * residual[1]) / det,
            (-jac[0][0] * residual[1] + jac[1][0] * residual[0]) / det,
        ]
        norm_old = sqrt(residual[0] ** 2 + residual[1] ** 2)
        step = 1.0
        for _ in range(20):
            trial = [y[0] + step * dy[0], y[1] + step * dy[1]]
            res_try, _k, _b, form_try, _ra = _residual(
                model, exp(trial[0]), exp(trial[1]), beta_target, ignore_correlation
            )
            if form_try is not None and sqrt(res_try[0] ** 2 + res_try[1] ** 2) < norm_old:
                y = trial
                break
            step *= 0.5
        else:  # pragma: no cover - no descent direction found
            break
    gamma_f, psi = exp(y[0]), exp(y[1])
    kappa = gamma_f * psi
    residual, kappa, beta, form, ratio = _residual(
        model, gamma_f, psi, beta_target, ignore_correlation
    )
    sens = adjoint_beta_sensitivity(model, kappa, form=form, ignore_correlation=ignore_correlation)
    split = design_split(model, kappa, form=form, ignore_correlation=ignore_correlation)
    return {
        "gamma_f": gamma_f,
        "psi": psi,
        "kappa_star": kappa,
        "beta_achieved": beta,
        "residuals": {"beta": residual[0], "split": residual[1]},
        "iterations": iterations,
        "converged": bool(converged and form is not None),
        "jacobian": last_jac,
        "jacobian_condition_number": condition_number(last_jac) if last_jac else inf,
        "d_beta_d_kappa_adjoint": sens["d_beta_d_kappa"],
        "d_beta_d_gamma_f_adjoint": sens["d_beta_d_kappa"] * psi,
        "d_beta_d_psi_adjoint": sens["d_beta_d_kappa"] * gamma_f,
        "split": split,
        "trace": trace,
        "start": list(start),
    }


def singular_values(m: Sequence[Sequence[float]]) -> tuple[float, float]:
    """Singular values of a 2x2 matrix (eigenvalues of ``m^T m``)."""
    a, b, c, d = float(m[0][0]), float(m[0][1]), float(m[1][0]), float(m[1][1])
    a11 = a * a + c * c
    a12 = a * b + c * d
    a22 = b * b + d * d
    disc = sqrt(max((a11 - a22) ** 2 + 4.0 * a12 * a12, 0.0))
    lam_max = 0.5 * (a11 + a22 + disc)
    lam_min = 0.5 * (a11 + a22 - disc)
    if lam_min <= 0.0:
        return sqrt(max(lam_max, 0.0)), 0.0
    return sqrt(lam_max), sqrt(lam_min)


def condition_number(m: Sequence[Sequence[float]]) -> float:
    smax, smin = singular_values(m)
    return inf if smin <= 0.0 else smax / smin


def solve_factors_multistart(
    model: PAreaForceModel,
    beta_target: float = TARGET_BETA,
    starts: Sequence[tuple[float, float]] = INITIAL_GUESSES,
    ignore_correlation: bool = False,
) -> dict:
    """Run the inverse from three dispersed starts and check uniqueness."""
    runs = [
        solve_factors(
            model, beta_target=beta_target, start=tuple(s), ignore_correlation=ignore_correlation
        )
        for s in starts
    ]
    gamma_values = [r["gamma_f"] for r in runs]
    kappa_values = [r["kappa_star"] for r in runs]
    mean_gamma = sum(gamma_values) / len(gamma_values)
    mean_kappa = sum(kappa_values) / len(kappa_values)
    converged_runs = [r for r in runs if r["converged"]]
    best = min(converged_runs, key=lambda r: r["iterations"]) if converged_runs else runs[0]
    return {
        "runs": runs,
        "solution": best,
        "n_starts": len(runs),
        "all_converged": all(r["converged"] for r in runs),
        "gamma_f_spread_rel": (max(gamma_values) - min(gamma_values)) / mean_gamma,
        "kappa_spread_rel": (max(kappa_values) - min(kappa_values)) / mean_kappa,
        "condition_numbers": [r["jacobian_condition_number"] for r in runs],
    }


def beta_derivative_check(
    model: PAreaForceModel, solution: dict, relative_step: float = 1e-3
) -> dict:
    """Criterion (c): adjoint ``d beta / d gamma`` against central differences."""
    gamma_f = solution["gamma_f"]
    psi = solution["psi"]
    kappa = gamma_f * psi
    out: dict = {}

    h = relative_step * kappa
    fd_kappa = (
        beta_of_kappa(model, kappa + h)[0] - beta_of_kappa(model, kappa - h)[0]
    ) / (2.0 * h)
    adj_kappa = solution["d_beta_d_kappa_adjoint"]
    out["d_beta_d_kappa"] = {
        "adjoint": adj_kappa,
        "finite_difference": fd_kappa,
        "relative_error": rel_error(adj_kappa, fd_kappa),
        "step": h,
    }

    hg = relative_step * gamma_f
    fd_gamma = (
        beta_of_kappa(model, (gamma_f + hg) * psi)[0]
        - beta_of_kappa(model, (gamma_f - hg) * psi)[0]
    ) / (2.0 * hg)
    adj_gamma = solution["d_beta_d_gamma_f_adjoint"]
    out["d_beta_d_gamma_f"] = {
        "adjoint": adj_gamma,
        "finite_difference": fd_gamma,
        "relative_error": rel_error(adj_gamma, fd_gamma),
        "step": hg,
    }

    hp = relative_step * psi
    fd_psi = (
        beta_of_kappa(model, gamma_f * (psi + hp))[0]
        - beta_of_kappa(model, gamma_f * (psi - hp))[0]
    ) / (2.0 * hp)
    adj_psi = solution["d_beta_d_psi_adjoint"]
    out["d_beta_d_psi"] = {
        "adjoint": adj_psi,
        "finite_difference": fd_psi,
        "relative_error": rel_error(adj_psi, fd_psi),
        "step": hp,
    }
    out["max_relative_error"] = max(
        abs(out[k]["relative_error"])
        for k in ("d_beta_d_kappa", "d_beta_d_gamma_f", "d_beta_d_psi")
    )
    return out