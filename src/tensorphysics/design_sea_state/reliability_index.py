"""Target reliability index versus exceedance probability, and the two conventions.

Two *different* objects both called "beta" appear in environmental-contour work,
and mixing them is the single most common silent error in design-sea-state
studies.  This module makes both explicit and adjudicates between them.

Convention **F** (Cornell / first-order reliability, 1 limit state)
    ``p_f = Phi(-beta)``  <=>  ``beta_F = -Phi^-1(p_f)``
    Probability content is *one-sided*: ``p_f`` is the mass in the tail beyond
    the design point along the governing variable.

Convention **C** (contour / chi-square, ``n = 2`` metocean variables)
    ``p_out = P(u1**2 + u2**2 > beta_C**2) = exp(-beta_C**2 / 2)``
    ``beta_C = sqrt(-2 ln p_out)``
    This is the radius of the standard IFORM contour such that the probability
    of the joint sea state lying *outside* the whole closed contour is exactly
    ``p_out`` — which is what "100-year environmental contour" means.

The two conventions coincide only in the limit of a one-dimensional problem
(``N = 1``: ``P(chi2_1 > b^2) = 2 Phi(-b)``).  For ``p_f`` between 1e-2 and
1e-4 they differ by 1.9-2.4 in beta, i.e. the *same* annual target produces
design sea states differing by tens of percent in H_s.  :func:`convention_report`
quantifies the mismatch, and every IFORM helper in this package states which
convention it returns.

Return periods
--------------
* ``return_period = 1 / annual_exceedance_probability`` (the plain, conservative
  "expected waiting time" reading used for metocean return values);
* ``return_period_contour = 1 / (1 - exp(-beta_C**2 / 2))`` (the equivalent
  Poissonian reading, ``= 1/p_out`` to first order).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from .rosenblatt import standard_normal_cdf, standard_normal_ppf

__all__ = [
    "beta_cornell_from_pf",
    "pf_from_beta_cornell",
    "beta_contour_from_pout",
    "pout_from_beta_contour",
    "return_period_from_pf",
    "pf_from_return_period",
    "lifetime_pf",
    "return_period_from_lifetime_pf",
    "DNV_ISO2394_TARGETS",
    "convention_report",
    "calibrate_contour_beta",
    "ReliabilityTarget",
]

#: Reference target annual failure probabilities, DNV-RP-C205 / ISO 2394 style
#: (consequence classes 1..3 for "normal" relative cost of safety measures).
DNV_ISO2394_TARGETS: tuple[tuple[str, float, float], ...] = (
    ("CC1_minor", 3.30, 4.83e-4),
    ("CC2_normal", 3.70, 1.08e-4),
    ("CC3_large", 4.20, 1.33e-5),
)


def beta_cornell_from_pf(pf: float) -> float:
    """Convention F: ``beta = -Phi^-1(p_f)``."""
    if not 0.0 < pf < 1.0:
        raise ValueError(f"p_f must lie in (0, 1), got {pf!r}")
    return -standard_normal_ppf(pf)


def pf_from_beta_cornell(beta: float) -> float:
    """Convention F inverse: ``p_f = Phi(-beta)``."""
    return standard_normal_cdf(-beta)


def beta_contour_from_pout(p_out: float) -> float:
    """Convention C: ``beta = sqrt(-2 ln p_out)`` (``n = 2`` chi-square contour)."""
    if not 0.0 < p_out < 1.0:
        raise ValueError(f"p_out must lie in (0, 1), got {p_out!r}")
    return math.sqrt(-2.0 * math.log(p_out))


def pout_from_beta_contour(beta: float) -> float:
    """Convention C inverse: ``p_out = exp(-beta**2 / 2)``."""
    return math.exp(-0.5 * beta * beta)


def return_period_from_pf(pf: float) -> float:
    """``1 / p_f`` (years for an annual probability)."""
    if pf <= 0.0:
        raise ValueError("p_f must be positive")
    return 1.0 / pf


def pf_from_return_period(return_period: float) -> float:
    if return_period <= 0.0:
        raise ValueError("return period must be positive")
    return 1.0 / return_period


def lifetime_pf(pf_annual: float, years: float) -> float:
    """Probability of at least one exceedance in ``years`` (independent years)."""
    if not 0.0 < pf_annual < 1.0:
        raise ValueError("annual probability must lie in (0, 1)")
    return -math.expm1(years * math.log1p(-pf_annual))


def return_period_from_lifetime_pf(pf_lifetime: float, years: float) -> float:
    """Annual-equivalent return period matching a lifetime target."""
    if not 0.0 < pf_lifetime < 1.0:
        raise ValueError("lifetime probability must lie in (0, 1)")
    pf_annual = -math.expm1(math.log1p(-pf_lifetime) / years)
    return 1.0 / pf_annual


@dataclass(frozen=True)
class ReliabilityTarget:
    """A design target carried in *both* conventions at once (self-documenting)."""

    annual_pf: float

    @property
    def beta_cornell(self) -> float:
        return beta_cornell_from_pf(self.annual_pf)

    @property
    def beta_contour(self) -> float:
        return beta_contour_from_pout(self.annual_pf)

    @property
    def return_period(self) -> float:
        return return_period_from_pf(self.annual_pf)

    def as_dict(self) -> dict[str, float]:
        return {
            "annual_pf": self.annual_pf,
            "beta_cornell": self.beta_cornell,
            "beta_contour": self.beta_contour,
            "return_period_years": self.return_period,
            "pout_at_beta_cornell": pout_from_beta_contour(self.beta_cornell),
            "pf_at_beta_contour": pf_from_beta_cornell(self.beta_contour),
        }


def calibrate_contour_beta(annual_pf: float) -> float:
    """Convenience alias: target annual probability -> IFORM contour radius."""
    return beta_contour_from_pout(annual_pf)


def convention_report(annual_p_fs: Iterable[float] = (1e-2, 1e-3, 1e-4)) -> dict:
    """Quantify the F-versus-C mismatch for a set of annual targets.

    Returns, per target, the two beta values, the probability each convention
    would report for the *other* convention's radius (``pout_at_beta_cornell``,
    ``pf_at_beta_contour``), and the relative beta gap.  The gap is what a
    mixed-convention study silently absorbs into its design sea state.
    """
    rows = []
    for pf in annual_p_fs:
        target = ReliabilityTarget(pf)
        rows.append(
            {
                "annual_pf": pf,
                "return_period_years": target.return_period,
                "beta_cornell": target.beta_cornell,
                "beta_contour": target.beta_contour,
                "beta_gap": target.beta_contour - target.beta_cornell,
                "beta_gap_rel": (target.beta_contour - target.beta_cornell) / target.beta_cornell,
                "pout_at_beta_cornell": pout_from_beta_contour(target.beta_cornell),
                "pf_at_beta_contour": pf_from_beta_cornell(target.beta_contour),
                "pout_over_target": pout_from_beta_contour(target.beta_cornell) / pf,
            }
        )
    worst = max(rows, key=lambda r: abs(math.log(r["pout_over_target"])))
    return {
        "rows": rows,
        "max_abs_log_mismatch": abs(math.log(worst["pout_over_target"])),
        "worst_row": worst,
    }