"""Limit-state model and characteristic values for ISO 19906 ice pressure design.

The design-check format of ISO 19906 / the Polar Code, as already used in this
repository (:mod:`tensorphysics.ice_plasticity.iso19906`), is

    gamma_F * S_k  <=  R_k / gamma_M                                       (2)

with ``S_k`` the characteristic (annual exceedance 1e-2, i.e. 100-year) ice
load effect and ``R_k`` the characteristic (5 % fractile) resistance.

The probabilistic counterpart used for the calibration is the panel yield limit
state of the ice belt — the ice pressure is transferred to plate bending stress
through the same bilinear plate coefficient ``alpha_b`` used by the W3-N04
ice-belt chain::

    S(X) = k_S * alpha_b * p * (b / t_net) ** 2        [MPa]               (3)
    R(X) = k_R * f_y                                   [MPa]               (4)
    g(X) = R(X) - S(X),   t_net = t_nom - dt_corr                           (5)

with ``p`` the annual-maximum ice pressure from
:mod:`tensorphysics.ice_partial_factors.pressure_area` (Gumbel), ``f_y`` the
steel yield strength (lognormal), ``dt_corr`` the corrosion/abrasion allowance
(normal), and ``k_S``/``k_R`` the load- and resistance-model uncertainties
(lognormal, unit mean). All marginals are handled by the Nataf transformation
of :mod:`tensorphysics.structural`, so the limit state is available to FORM,
SORM, Monte Carlo and importance sampling unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, sqrt
from typing import Sequence

from ..structural.distributions import (
    GumbelMaxDistribution,
    LognormalDistribution,
    NatafTransform,
    NormalDistribution,
    norm_ppf,
)
from ..structural.reliability import ReliabilityProblem
from .pressure_area import GumbelParams

__all__ = [
    "IceBeltPanel",
    "CharacteristicValues",
    "build_transform",
    "build_limit_state",
    "build_problem",
    "load_effect",
    "resistance",
    "characteristic_values",
    "design_thickness_for_factor",
    "percentile",
    "model_uncertainty_quantile",
]

#: Order of the random vector used by every limit state in this module.
VARIABLE_NAMES = ("p_ice", "fy", "dt_corr", "k_S", "k_R")


@dataclass(frozen=True)
class IceBeltPanel:
    """Ice-belt plate panel: geometry, material and model uncertainties."""

    b_mm: float = 350.0            # transverse frame spacing (plate bending span)
    t_nom_mm: float = 26.0         # as-built nominal plate thickness
    alpha_b: float = 0.5           # plate bending coefficient alpha_b
    dt_mean_mm: float = 2.0        # mean corrosion/abrasion allowance
    dt_cov: float = 0.25
    fy_mean_mpa: float = 355.0
    fy_cov: float = 0.08
    k_s_cov: float = 0.15          # load-model uncertainty
    k_r_cov: float = 0.10          # resistance-model uncertainty
    area_m2: float = 5.0           # loaded area used for the p-A extreme
    name: str = "ice_belt_panel"

    def __post_init__(self) -> None:
        if self.t_nom_mm <= 0 or self.b_mm <= 0 or self.area_m2 <= 0:
            raise ValueError("geometry must be positive")
        if self.t_nom_mm <= self.dt_mean_mm:
            raise ValueError("nominal thickness must exceed the corrosion allowance")
        for value in (self.fy_cov, self.k_s_cov, self.k_r_cov):
            if value < 0.0:
                raise ValueError("coefficients of variation must be non-negative")

    # -- derived dimensions --------------------------------------------------
    @property
    def t_net_mm(self) -> float:
        """Mean net (corroded) thickness."""
        return self.t_nom_mm - self.dt_mean_mm

    @property
    def aspect_ratio_sq(self) -> float:
        return (self.b_mm / self.t_net_mm) ** 2

    def with_thickness(self, t_nom_mm: float) -> "IceBeltPanel":
        return IceBeltPanel(
            b_mm=self.b_mm,
            t_nom_mm=t_nom_mm,
            alpha_b=self.alpha_b,
            dt_mean_mm=self.dt_mean_mm,
            dt_cov=self.dt_cov,
            fy_mean_mpa=self.fy_mean_mpa,
            fy_cov=self.fy_cov,
            k_s_cov=self.k_s_cov,
            k_r_cov=self.k_r_cov,
            area_m2=self.area_m2,
            name=self.name,
        )

    def with_area(self, area_m2: float) -> "IceBeltPanel":
        return IceBeltPanel(
            b_mm=self.b_mm,
            t_nom_mm=self.t_nom_mm,
            alpha_b=self.alpha_b,
            dt_mean_mm=self.dt_mean_mm,
            dt_cov=self.dt_cov,
            fy_mean_mpa=self.fy_mean_mpa,
            fy_cov=self.fy_cov,
            k_s_cov=self.k_s_cov,
            k_r_cov=self.k_r_cov,
            area_m2=area_m2,
            name=self.name,
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "b_mm": self.b_mm,
            "t_nom_mm": self.t_nom_mm,
            "t_net_mm": self.t_net_mm,
            "alpha_b": self.alpha_b,
            "dt_mean_mm": self.dt_mean_mm,
            "dt_cov": self.dt_cov,
            "fy_mean_mpa": self.fy_mean_mpa,
            "fy_cov": self.fy_cov,
            "k_s_cov": self.k_s_cov,
            "k_r_cov": self.k_r_cov,
            "area_m2": self.area_m2,
        }


# --------------------------------------------------------------------------- #
# Physical limit state
# --------------------------------------------------------------------------- #
def load_effect(panel: IceBeltPanel, p_ice: float, t_net_mm: float, k_s: float = 1.0) -> float:
    """Plate bending stress [MPa] produced by ice pressure ``p_ice`` [MPa]."""
    if t_net_mm <= 0.0:
        return float("inf")
    return k_s * panel.alpha_b * p_ice * (panel.b_mm / t_net_mm) ** 2


def resistance(panel: IceBeltPanel, fy: float, k_r: float = 1.0) -> float:
    """Available plate bending capacity [MPa]."""
    return k_r * fy


def build_transform(panel: IceBeltPanel, gumbel: GumbelParams) -> NatafTransform:
    """Nataf transform of the five random variables of the panel problem.

    ``p_ice`` and ``k_S`` are positively correlated (rho = 0.25): both are driven
    by the same unmodelled contact-area / indentation effects, consistent with
    the correlation structure used by the W3-N04 ice-belt chain.
    """
    marginals = [
        GumbelMaxDistribution(loc=gumbel.loc, scale=gumbel.scale),
        LognormalDistribution(
            mu_ln=log(panel.fy_mean_mpa)
            - 0.5 * _sigma_ln(panel.fy_cov) ** 2,
            sigma_ln=_sigma_ln(panel.fy_cov),
        ) if panel.fy_cov > 0 else NormalDistribution(mu=panel.fy_mean_mpa, sigma=1e-3),
        NormalDistribution(mu=panel.dt_mean_mm, sigma=max(panel.dt_mean_mm * panel.dt_cov, 1e-6)),
        _unit_mean_lognormal(panel.k_s_cov),
        _unit_mean_lognormal(panel.k_r_cov),
    ]
    correlation = [
        [1.0, 0.0, 0.0, 0.25, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.25, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ]
    return NatafTransform(marginals, correlation)


def _sigma_ln(cov: float) -> float:
    return sqrt(log(1.0 + cov * cov))


def _unit_mean_lognormal(cov: float):
    if cov <= 0.0:
        return NormalDistribution(mu=1.0, sigma=1e-6)
    s = _sigma_ln(cov)
    return LognormalDistribution(mu_ln=-0.5 * s * s, sigma_ln=s)


def build_limit_state(panel: IceBeltPanel):
    """Return ``g(x) -> float`` with ``x = (p_ice, fy, dt_corr, k_S, k_R)``."""

    def g(x: Sequence[float]) -> float:
        p_ice, fy, dt_corr, k_s, k_r = x
        t_net = panel.t_nom_mm - dt_corr
        if t_net <= 0.0:
            return -1e6
        return k_r * fy - k_s * panel.alpha_b * p_ice * (panel.b_mm / t_net) ** 2

    return g


def build_problem(
    panel: IceBeltPanel, gumbel: GumbelParams, name: str = "ice_belt_yield"
) -> ReliabilityProblem:
    """Reliability problem of the ice-belt plate yield limit state."""
    return ReliabilityProblem(
        build_transform(panel, gumbel),
        build_limit_state(panel),
        name=name,
        variable_names=VARIABLE_NAMES,
    )


# --------------------------------------------------------------------------- #
# Characteristic values (the design-format side of equation (2))
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CharacteristicValues:
    """Characteristic load and resistance plus the format ratio ``R_k / S_k``."""

    s_k_mpa: float
    r_k_mpa: float
    p_exceedance_mpa: float
    return_period_years: float
    resistance_quantile: float

    @property
    def factor_product(self) -> float:
        """``gamma_F * gamma_M`` implied by the characteristic values."""
        return self.r_k_mpa / self.s_k_mpa

    def to_dict(self) -> dict:
        return {
            "s_k_mpa": self.s_k_mpa,
            "r_k_mpa": self.r_k_mpa,
            "p_exceedance_mpa": self.p_exceedance_mpa,
            "return_period_years": self.return_period_years,
            "resistance_quantile": self.resistance_quantile,
            "factor_product": self.factor_product,
        }


def percentile(mean: float, cov: float, p: float) -> float:
    """Lognormal percentile from mean/COV (the resistance side of the format)."""
    if cov <= 0.0:
        return mean
    s = _sigma_ln(cov)
    return mean * exp(s * norm_ppf(p) - 0.5 * s * s)


def model_uncertainty_quantile(mean: float, cov: float, p: float) -> float:
    """Percentile of a unit-mean lognormal model-uncertainty factor."""
    return percentile(mean, cov, p)


def characteristic_values(
    panel: IceBeltPanel,
    gumbel: GumbelParams,
    return_period_years: float = 100.0,
    resistance_quantile: float = 0.05,
) -> CharacteristicValues:
    """``S_k`` from the 100-year pressure, ``R_k`` from the 5 % fractile.

    The load side keeps the nominal transfer (``k_S`` at its unit mean) and the
    mean net thickness; the resistance side combines the material COV with the
    resistance-model COV (product of two lognormals).
    """
    p_k = gumbel.return_level(return_period_years)
    s_k = load_effect(panel, p_k, panel.t_net_mm, 1.0)
    cov_r = sqrt(panel.fy_cov**2 + panel.k_r_cov**2)
    r_k = percentile(panel.fy_mean_mpa, cov_r, resistance_quantile)
    return CharacteristicValues(
        s_k_mpa=s_k,
        r_k_mpa=r_k,
        p_exceedance_mpa=p_k,
        return_period_years=return_period_years,
        resistance_quantile=resistance_quantile,
    )


def design_thickness_for_factor(panel: IceBeltPanel, char: CharacteristicValues, k: float) -> float:
    """Nominal thickness satisfying ``R_k = k * S_k(t)`` exactly.

    ``S_k(t) = alpha_b p_k (b/t_net)**2`` and ``t_net = t_nom - dt_mean``, so

        t_nom(k) = b sqrt(alpha_b p_k k / R_k) + dt_mean.
    """
    if k <= 0.0:
        raise ValueError("factor product must be positive")
    inner = panel.alpha_b * char.p_exceedance_mpa * k / char.r_k_mpa
    if inner <= 0.0:
        raise ValueError("non-positive argument in the design-thickness inversion")
    return panel.b_mm * sqrt(inner) + panel.dt_mean_mm