"""Target-reliability back-calibration of ISO 19906 ice pressure partial factors.

Calibration problem
-------------------
Given (i) the p-A extreme statistics of the ice pressure (Gumbel annual
maximum), (ii) the resistance statistics of the ice-belt plate, and (iii) a
target annual reliability index ``beta_T`` (= 4.0, annual failure probability
``Phi(-4) = 3.17e-5``), find the load and resistance partial factors
``gamma_F``/``gamma_M`` of the design format

    gamma_F * S_k  <=  R_k / gamma_M

Two independent routes are implemented and cross-checked:

**Route A — code-calibration inverse (primary).**
A structure designed by the format has ``R_k = k * S_k`` with
``k = gamma_F * gamma_M``. Inverting that relation gives the nominal thickness
``t_nom(k)`` (:func:`limit_state.design_thickness_for_factor`). The achieved
reliability ``beta(k)`` of the as-designed structure increases monotonically
with ``k``, so ``k*`` is found by bisection on ``beta(k*) = beta_T``. This is
the classical code-calibration definition (achieved reliability of a structure
designed to the code equals the target) and needs no design-value
approximation.

**Route B — design-value (JCSS/ISO 2394) split.**
Once ``k*`` is known, the FORM design point of the as-designed structure gives
the design load effect ``s*`` (``= r*``). Splitting by the design values,
``gamma_F = s* / S_k`` and ``gamma_M = R_k / s*``, reproduces
``gamma_F * gamma_M = R_k / S_k = k*`` identically, so the format is satisfied
with equality at the target reliability. Route B therefore supplies the split
while Route A supplies the total factor.

Because the split is a convention (one equation, two unknowns), the module also
reports the design-point sensitivities ``alpha`` so the split can be audited,
and a design-value-only calibration (:func:`design_value_factors`) is provided
for comparison with the code-calibration inverse.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from ..structural.distributions import norm_cdf
from ..structural.reliability import FormResult, ReliabilityProblem, SormResult
from .limit_state import (
    CharacteristicValues,
    IceBeltPanel,
    build_problem,
    characteristic_values,
    design_thickness_for_factor,
    load_effect,
    resistance,
)
from .pressure_area import GumbelParams

__all__ = [
    "TARGET_BETA",
    "TARGET_PF",
    "ReliabilitySnapshot",
    "CalibrationResult",
    "analyse_problem",
    "design_value_factors",
    "calibrate_factor_product",
    "calibrate_panel",
    "beta_target_sweep",
    "scenario_sweep",
    "load_cov_sweep",
]

#: Mission target: annual reliability index and its annual failure probability.
TARGET_BETA = 4.0
TARGET_PF = 3.167124183311987e-05  # Phi(-4)


@dataclass(frozen=True)
class ReliabilitySnapshot:
    """FORM / SORM / simulation results for one reliability problem."""

    beta_form: float
    pf_form: float
    alpha: list[float]
    design_point: list[float]
    s_star: float
    r_star: float
    g_design: float
    iterations: int
    converged: bool
    beta_sorm: float
    pf_sorm: float
    sorm_valid: bool
    curvatures: list[float]
    pf_is: float
    beta_is: float
    is_cov: float
    samples_is: int
    pf_mc: float
    beta_mc: float
    samples_mc: int

    def to_dict(self) -> dict:
        return {
            "beta_form": self.beta_form,
            "pf_form": self.pf_form,
            "alpha": list(self.alpha),
            "design_point": list(self.design_point),
            "s_star_mpa": self.s_star,
            "r_star_mpa": self.r_star,
            "g_design": self.g_design,
            "iterations": self.iterations,
            "converged": self.converged,
            "beta_sorm": self.beta_sorm,
            "pf_sorm": self.pf_sorm,
            "sorm_valid": self.sorm_valid,
            "principal_curvatures": list(self.curvatures),
            "pf_importance_sampling": self.pf_is,
            "beta_importance_sampling": self.beta_is,
            "importance_sampling_cov": self.is_cov,
            "samples_importance_sampling": self.samples_is,
            "pf_monte_carlo": self.pf_mc,
            "beta_monte_carlo": self.beta_mc,
            "samples_monte_carlo": self.samples_mc,
        }


def _beta_from_pf(pf: float) -> float:
    from ..structural.distributions import norm_ppf

    if pf <= 0.0:
        return float("inf")
    return -norm_ppf(min(max(pf, 1e-300), 1.0 - 1e-16))


def analyse_problem(
    problem: ReliabilityProblem,
    panel: IceBeltPanel,
    samples_mc: int = 400_000,
    samples_is: int = 200_000,
    seed: int = 2026,
    with_simulation: bool = True,
) -> ReliabilitySnapshot:
    """Run FORM + SORM (+ MC/importance sampling) on one panel problem."""
    form: FormResult = problem.form()
    sorm: SormResult = problem.sorm(form_result=form)
    x = form.design_point_x
    p_ice, fy, dt_corr, k_s, k_r = x
    s_star = load_effect(panel, p_ice, panel.t_nom_mm - dt_corr, k_s)
    r_star = resistance(panel, fy, k_r)
    pf_is = beta_is = is_cov = 0.0
    pf_mc = beta_mc = 0.0
    do_is = with_simulation and samples_is > 0
    do_mc = with_simulation and samples_mc > 0
    if do_is:
        is_res = problem.importance_sampling_pf(samples=samples_is, seed=seed, form_result=form)
        pf_is = is_res.pf
        beta_is = _beta_from_pf(pf_is)
        is_cov = float(is_res.std_error) / pf_is if pf_is > 0 else 0.0
    if do_mc:
        mc_res = problem.monte_carlo_pf(samples=samples_mc, seed=seed + 1)
        pf_mc = mc_res.pf
        beta_mc = _beta_from_pf(pf_mc) if pf_mc > 0 else float("nan")
    return ReliabilitySnapshot(
        beta_form=form.beta,
        pf_form=form.pf,
        alpha=list(form.alpha),
        design_point=list(form.design_point_x),
        s_star=s_star,
        r_star=r_star,
        g_design=form.g_design,
        iterations=form.iterations,
        converged=form.converged,
        beta_sorm=sorm.beta_sorm,
        pf_sorm=sorm.pf_sorm,
        sorm_valid=sorm.valid,
        curvatures=list(sorm.principal_curvatures),
        pf_is=pf_is,
        beta_is=beta_is,
        is_cov=is_cov,
        samples_is=samples_is if do_is else 0,
        pf_mc=pf_mc,
        beta_mc=beta_mc,
        samples_mc=samples_mc if do_mc else 0,
    )


@dataclass(frozen=True)
class CalibrationResult:
    """Everything the report needs for one calibrated panel."""

    beta_target: float
    k_star: float
    gamma_f: float
    gamma_m: float
    t_design_mm: float
    s_k_mpa: float
    r_k_mpa: float
    s_star_mpa: float
    beta_at_k_star: float
    bisection_iterations: int
    snapshot: ReliabilitySnapshot
    characteristic: CharacteristicValues
    alpha: list[float]
    base_characteristic: CharacteristicValues | None = None

    def to_dict(self) -> dict:
        return {
            "beta_target": self.beta_target,
            "pf_target": norm_cdf(-self.beta_target),
            "k_star": self.k_star,
            "gamma_F": self.gamma_f,
            "gamma_M": self.gamma_m,
            "gamma_product": self.k_star,
            "t_design_mm": self.t_design_mm,
            "s_k_mpa": self.s_k_mpa,
            "r_k_mpa": self.r_k_mpa,
            "s_star_mpa": self.s_star_mpa,
            "r_star_mpa": self.snapshot.r_star,
            "beta_achieved": self.beta_at_k_star,
            "bisection_iterations": self.bisection_iterations,
            "alpha": list(self.alpha),
            "characteristic": self.characteristic.to_dict(),
            "base_characteristic": (
                self.base_characteristic.to_dict() if self.base_characteristic else None
            ),
            "reliability": self.snapshot.to_dict(),
        }


def _beta_of_factor_product(
    panel: IceBeltPanel,
    gumbel: GumbelParams,
    char: CharacteristicValues,
    k: float,
) -> tuple[float, float, FormResult | None]:
    t_nom = design_thickness_for_factor(panel, char, k)
    designed = panel.with_thickness(t_nom)
    problem = build_problem(designed, gumbel)
    # A structure that already fails at the median point has (by definition)
    # no design point; report beta = 0 instead of propagating the exception, so
    # the bisection can use it as the lower bracket.
    if problem.g_u([0.0] * problem.transform.dimension) <= 0.0:
        return 0.0, t_nom, None
    form = problem.form()
    return form.beta, t_nom, form


def calibrate_factor_product(
    panel: IceBeltPanel,
    gumbel: GumbelParams,
    beta_target: float = TARGET_BETA,
    char: CharacteristicValues | None = None,
    tolerance: float = 1e-6,
    max_iterations: int = 80,
) -> dict:
    """Bisection on ``k = gamma_F * gamma_M`` such that ``beta(k) = beta_target``.

    ``beta(k)`` is bracketed in ``[0.2, 6.0]`` with geometric expansion, then
    bisected to ``tolerance`` in ``k``.
    """
    ch = char if char is not None else characteristic_values(panel, gumbel)
    lo, hi = 0.2, 6.0
    beta_lo, t_lo, _ = _beta_of_factor_product(panel, gumbel, ch, lo)
    beta_hi, t_hi, _ = _beta_of_factor_product(panel, gumbel, ch, hi)
    guard = 0
    while (beta_lo - beta_target) * (beta_hi - beta_target) > 0.0 and guard < 12:  # pragma: no cover
        lo = max(lo * 0.5, 1e-3)
        hi = min(hi * 2.0, 1e6)
        beta_lo, t_lo, _ = _beta_of_factor_product(panel, gumbel, ch, lo)
        beta_hi, t_hi, _ = _beta_of_factor_product(panel, gumbel, ch, hi)
        guard += 1
    if (beta_lo - beta_target) * (beta_hi - beta_target) > 0.0:  # pragma: no cover
        raise RuntimeError("factor-product bisection could not be bracketed")
    iterations = 0
    t_mid = 0.5 * (t_lo + t_hi)
    beta_mid = 0.5 * (beta_lo + beta_hi)
    while iterations < max_iterations and (hi - lo) > tolerance * max(1.0, 0.5 * (lo + hi)):
        mid = 0.5 * (lo + hi)
        beta_mid, t_mid, _ = _beta_of_factor_product(panel, gumbel, ch, mid)
        iterations += 1
        if (beta_mid - beta_target) * (beta_lo - beta_target) <= 0.0:
            hi, beta_hi = mid, beta_mid
        else:
            lo, beta_lo = mid, beta_mid
    k_star = 0.5 * (lo + hi)
    return {
        "k_star": k_star,
        "t_design_mm": t_mid,
        "beta_at_k_star": beta_mid,
        "iterations": iterations,
        "bracket": {"lo": lo, "hi": hi, "beta_lo": beta_lo, "beta_hi": beta_hi},
        "characteristic": ch,
    }


def design_value_factors(
    panel: IceBeltPanel,
    gumbel: GumbelParams,
    char: CharacteristicValues,
    snapshot: ReliabilitySnapshot,
) -> dict:
    """JCSS/ISO 2394 design-value split of the already-calibrated problem."""
    s_k = char.s_k_mpa
    r_k = char.r_k_mpa
    gamma_f = snapshot.s_star / s_k
    gamma_m = r_k / snapshot.s_star
    return {
        "gamma_F": gamma_f,
        "gamma_M": gamma_m,
        "product": gamma_f * gamma_m,
        "s_star_mpa": snapshot.s_star,
        "r_star_mpa": snapshot.r_star,
        "s_k_mpa": s_k,
        "r_k_mpa": r_k,
        "ratio_s_star_over_s_k": gamma_f,
        "ratio_r_k_over_s_star": gamma_m,
    }


def achieved_beta_for_product(
    panel: IceBeltPanel,
    gumbel: GumbelParams,
    factor_product: float,
    char: CharacteristicValues | None = None,
) -> dict:
    """Achieved reliability index of a structure designed with ``gamma_F gamma_M``."""
    ch = char if char is not None else characteristic_values(panel, gumbel)
    beta, t_nom, form = _beta_of_factor_product(panel, gumbel, ch, factor_product)
    return {
        "factor_product": factor_product,
        "beta_achieved": beta,
        "t_design_mm": t_nom,
        "converged": bool(form is not None and form.converged),
    }


def return_period_sweep(
    panel: IceBeltPanel,
    gumbel: GumbelParams,
    return_periods: Sequence[float] = (50.0, 100.0, 200.0, 500.0, 1000.0),
    beta_target: float = TARGET_BETA,
) -> list[dict]:
    """Required factors versus the return period of the characteristic ice action.

    ``S_k`` in the design format is defined by a return period; the required
    factor product falls as the characteristic value grows, so this sweep
    identifies which characteristic-value convention a tabulated factor pair
    is consistent with.
    """
    rows = []
    for period in return_periods:
        ch = characteristic_values(panel, gumbel, return_period_years=period)
        result = calibrate_panel(
            panel, gumbel, beta_target=beta_target, char=ch, samples_mc=0, samples_is=0
        )
        rows.append(
            {
                "return_period_years": period,
                "p_k_mpa": ch.p_exceedance_mpa,
                "s_k_mpa": ch.s_k_mpa,
                "k_star": result.k_star,
                "gamma_F": result.gamma_f,
                "gamma_M": result.gamma_m,
                "t_design_mm": result.t_design_mm,
                "beta_achieved": result.beta_at_k_star,
            }
        )
    return rows


def calibrate_panel(
    panel: IceBeltPanel,
    gumbel: GumbelParams,
    beta_target: float = TARGET_BETA,
    char: CharacteristicValues | None = None,
    samples_mc: int = 400_000,
    samples_is: int = 200_000,
    seed: int = 2026,
) -> CalibrationResult:
    """Full back-calibration of one panel at ``beta_target`` (routes A + B)."""
    ch_base = char if char is not None else characteristic_values(panel, gumbel)
    inverse = calibrate_factor_product(panel, gumbel, beta_target=beta_target, char=ch_base)
    k_star = inverse["k_star"]
    # Re-derive the thickness from the bisected k* so that
    # gamma_F * gamma_M = R_k / S_k(t_design) = k* holds to machine precision.
    t_design = design_thickness_for_factor(panel, ch_base, k_star)
    designed = panel.with_thickness(t_design)
    # The format of the *as-designed* structure uses S_k of the design
    # thickness, so the characteristic values must be re-evaluated there;
    # otherwise gamma_F * gamma_M would not reproduce k* (the design-point
    # split would be internally inconsistent).
    ch = characteristic_values(
        designed,
        gumbel,
        return_period_years=ch_base.return_period_years,
        resistance_quantile=ch_base.resistance_quantile,
    )
    problem = build_problem(designed, gumbel)
    snapshot = analyse_problem(
        problem, designed, samples_mc=samples_mc, samples_is=samples_is, seed=seed
    )
    split = design_value_factors(designed, gumbel, ch, snapshot)
    return CalibrationResult(
        beta_target=beta_target,
        k_star=k_star,
        gamma_f=split["gamma_F"],
        gamma_m=split["gamma_M"],
        t_design_mm=t_design,
        s_k_mpa=ch.s_k_mpa,
        r_k_mpa=ch.r_k_mpa,
        s_star_mpa=snapshot.s_star,
        beta_at_k_star=snapshot.beta_form,
        bisection_iterations=inverse["iterations"],
        snapshot=snapshot,
        characteristic=ch,
        base_characteristic=ch_base,
        alpha=list(snapshot.alpha),
    )


def beta_target_sweep(
    panel: IceBeltPanel,
    gumbel: GumbelParams,
    betas: Sequence[float] = (3.0, 3.3, 3.5, 3.8, 4.0, 4.3, 4.5),
    char: CharacteristicValues | None = None,
) -> list[dict]:
    """``gamma_F``/``gamma_M`` as a function of the target reliability index."""
    ch = char if char is not None else characteristic_values(panel, gumbel)
    rows = []
    for beta_t in betas:
        result = calibrate_panel(
            panel,
            gumbel,
            beta_target=beta_t,
            char=ch,
            samples_mc=0,
            samples_is=0,
        )
        rows.append(
            {
                "beta_target": beta_t,
                "pf_target": norm_cdf(-beta_t),
                "k_star": result.k_star,
                "gamma_F": result.gamma_f,
                "gamma_M": result.gamma_m,
                "t_design_mm": result.t_design_mm,
                "beta_achieved": result.beta_at_k_star,
                "s_star_mpa": result.s_star_mpa,
            }
        )
    return rows


def scenario_sweep(
    base_panel: IceBeltPanel,
    gumbel_of_area: Callable[[float], GumbelParams],
    areas: Sequence[float] = (2.5, 5.0, 10.0, 20.0),
    beta_target: float = TARGET_BETA,
) -> list[dict]:
    """Calibrate the same panel geometry at several loaded areas.

    The factors of a code must cover a family of panels, so the scatter of the
    calibrated factors over the area (i.e. over the p-A law) is the practical
    envelope of the recommendation.
    """
    rows = []
    for area in areas:
        panel = base_panel.with_area(area)
        gumbel = gumbel_of_area(area)
        ch = characteristic_values(panel, gumbel)
        result = calibrate_panel(
            panel, gumbel, beta_target=beta_target, char=ch, samples_mc=0, samples_is=0
        )
        row = result.to_dict()
        row.update({"area_m2": area, "p_k_mpa": ch.p_exceedance_mpa})
        rows.append(row)
    return rows


def load_cov_sweep(
    panel: IceBeltPanel,
    gumbel_of_cov: Callable[[float], GumbelParams],
    covs: Sequence[float] = (0.10, 0.14, 0.18, 0.22, 0.26, 0.30),
    beta_target: float = TARGET_BETA,
) -> list[dict]:
    """Required factors versus the COV of the annual-maximum ice pressure.

    The calibrated load factor of an extreme-value load is controlled almost
    entirely by the COV of the annual maximum, so this sweep is what makes the
    comparison with a tabulated ``gamma_F`` interpretable.
    """
    rows = []
    for cov in covs:
        gumbel = gumbel_of_cov(cov)
        ch = characteristic_values(panel, gumbel)
        result = calibrate_panel(
            panel, gumbel, beta_target=beta_target, char=ch, samples_mc=0, samples_is=0
        )
        rows.append(
            {
                "load_cov": cov,
                "k_star": result.k_star,
                "gamma_F": result.gamma_f,
                "gamma_M": result.gamma_m,
                "t_design_mm": result.t_design_mm,
                "s_k_mpa": ch.s_k_mpa,
                "r_k_mpa": ch.r_k_mpa,
                "beta_achieved": result.beta_at_k_star,
            }
        )
    return rows


def invert_sweep(rows: Sequence[dict], key: str, value: float, column: str) -> float:
    """Linear interpolation of the ``key`` (e.g. load COV) that yields ``value``.

    Used to answer "which load COV makes the tabulated factor consistent?".
    """
    pts = sorted((row[key], row[column]) for row in rows)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if (y0 - value) * (y1 - value) <= 0.0 and abs(y1 - y0) > 1e-15:
            return x0 + (x1 - x0) * (value - y0) / (y1 - y0)
    raise ValueError(f"value {value} not bracketed by the sweep in column {column!r}")