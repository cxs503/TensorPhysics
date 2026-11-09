"""Joint pressure-area ice-force model with an explicit correlation control (IN101-N09).

Physics
-------
ISO 19906 describes the ice action through a pressure-area relationship: the
pressure that acts over a contact (or global) area ``A`` falls as ``A`` grows,

    p(A) = p_ref * (A / A_ref)**(-ex),          ex ~ 0.4 (global pressure),

so the ice force handed to the structure is

    F = k_F * p(A) * A = k_F * p_ref * A_ref**ex * A**(1 - ex).

Because ``ex > 0``, the pressure and the area that together produce the annual
extreme force are **negatively correlated**: a large contact area comes with a
low pressure.  Equivalently, in log space the deterministic trend
``ln p = ln p_ref - ex ln(A/A_ref) + ln eps`` plus a lognormal area makes
``(ln p, ln A)`` jointly normal with

    rho_ln(p, A) = -ex * sigma_lnA / sigma_lnp,
    sigma_lnp = sqrt(ex**2 sigma_lnA**2 + sigma_eps**2),

so the force scatter is

    Var_corr(ln F) = sigma_kF**2 + sigma_eps**2 + (1 - ex)**2 sigma_lnA**2,

whereas an analyst who keeps the two *marginals* but drops the correlation
("independent" model, ``rho = 0``) gets

    Var_indep(ln F) = sigma_kF**2 + sigma_eps**2 + (1 + ex**2) sigma_lnA**2.

The difference ``2 ex sigma_lnA**2`` is the physical content of the negative
control of the IN101-N09 mission: dropping the p-A correlation inflates the
load scatter and drives the back-calibrated partial factors up.

Model states
------------
The :class:`PAreaForceModel` is a *state* (annual extreme contact area,
pressure-law coefficient, epistemic/model uncertainties, resistance) and
:func:`build_problem` turns one state into a two-variable-wide reliability
problem ``g = R - k_F p A`` whose marginals are identical between the two
correlation hypotheses — only ``rho`` changes.  That is what makes the
comparison a controlled experiment rather than a comparison of two different
models.

All statistics are lognormal, so the ln-space reliability index

    beta_ln = (mu_lnR - mu_lnF) / sqrt(sigma_lnR**2 + sigma_lnF**2)

is available in closed form as an independent check of the FORM solution.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import exp, log, sqrt

from ..structural.distributions import LognormalDistribution, NatafTransform, norm_ppf
from ..structural.reliability import ReliabilityProblem

__all__ = [
    "PAreaForceModel",
    "build_problem",
    "force_moments",
    "characteristic_force",
    "design_resistance_median",
    "analytic_beta_ln",
    "log_cov_to_sigma",
]


def log_cov_to_sigma(cov: float) -> float:
    """Log-normal log-scale standard deviation from a coefficient of variation."""
    if cov <= 0.0:
        return 0.0
    return sqrt(log(1.0 + cov * cov))


def _unit_lognormal(sigma_ln: float) -> LognormalDistribution:
    """Unit-*median* lognormal (median 1), used for all multiplicative factors."""
    return LognormalDistribution(mu_ln=0.0, sigma_ln=max(sigma_ln, 1e-9))


@dataclass(frozen=True)
class PAreaForceModel:
    """One ice-action state: p-A law, extreme area statistics and resistance.

    Parameters
    ----------
    p_ref_mpa
        Pressure at the reference area ``A_ref`` (median of the pressure law).
    exponent
        ``ex`` of the ISO 19906 pressure-area relation.
    a_ref_m2, a_median_m2
        Reference area of the law and median annual extreme contact area.
    a_cov
        COV of the annual extreme contact area (lognormal).
    pressure_model_cov
        Epistemic scatter of the pressure law itself (unit median factor).
    load_model_cov
        Load-model uncertainty ``k_F`` (unit median factor).
    resistance_cov
        COV of the member resistance.
    return_period_years, resistance_quantile
        Characteristic-value conventions of the design format.
    """

    p_ref_mpa: float = 2.5
    exponent: float = 0.40
    a_ref_m2: float = 5.0
    a_median_m2: float = 5.0
    a_cov: float = 0.35
    pressure_model_cov: float = 0.15
    load_model_cov: float = 0.10
    resistance_cov: float = 0.12
    return_period_years: float = 100.0
    resistance_quantile: float = 0.05

    # -- derived log-scale scatter -----------------------------------------
    def __post_init__(self) -> None:
        if self.p_ref_mpa <= 0.0:
            raise ValueError("p_ref_mpa must be positive")
        if not -0.95 < self.exponent < 0.95:
            raise ValueError("pressure-area exponent outside the physical band")
        if self.a_ref_m2 <= 0.0 or self.a_median_m2 <= 0.0:
            raise ValueError("areas must be positive")
        if self.resistance_cov < 0.0 or self.a_cov < 0.0:
            raise ValueError("coefficients of variation must be non-negative")

    @property
    def sigma_a(self) -> float:
        return log_cov_to_sigma(self.a_cov)

    @property
    def sigma_eps(self) -> float:
        return log_cov_to_sigma(self.pressure_model_cov)

    @property
    def sigma_kf(self) -> float:
        return log_cov_to_sigma(self.load_model_cov)

    @property
    def sigma_r(self) -> float:
        return log_cov_to_sigma(self.resistance_cov)

    @property
    def sigma_p(self) -> float:
        """Log-scale scatter of the pressure *marginal* over the area ensemble."""
        return sqrt(self.exponent**2 * self.sigma_a**2 + self.sigma_eps**2)

    @property
    def median_pressure_mpa(self) -> float:
        """Median pressure of the marginal at the median extreme area."""
        return self.p_ref_mpa * (self.a_median_m2 / self.a_ref_m2) ** (-self.exponent)

    @property
    def median_force_mn(self) -> float:
        """Median annual extreme ice force (``k_F`` has unit median)."""
        return self.median_pressure_mpa * self.a_median_m2

    # -- correlation --------------------------------------------------------
    def pressure_area_correlation(self) -> float:
        """``rho(ln p, ln A)`` implied by the negative p-A exponent."""
        if self.sigma_p <= 0.0 or self.sigma_a <= 0.0:
            return 0.0
        rho = -self.exponent * self.sigma_a / self.sigma_p
        return max(-0.999, min(0.0, rho))

    def pressure_area_pearson(self) -> float:
        """Pearson correlation of ``(p, A)`` implied by the log-space correlation.

        :class:`tensorphysics.structural.distributions.NatafTransform` takes the
        correlation of the *physical* marginals, while the ISO power law fixes
        the correlation of their logarithms.  For two lognormals with
        log-sigmas ``s1, s2`` the exact relation is

            rho_phys = (exp(rho_ln s1 s2) - 1) / sqrt((exp(s1**2) - 1)(exp(s2**2) - 1)),

        which is inverted here so that the assembled reliability problem has
        exactly the log-space correlation ``pressure_area_correlation()`` -- and
        therefore exactly the closed-form index of :func:`analytic_beta_ln`.
        """
        rho_ln = self.pressure_area_correlation()
        s1, s2 = self.sigma_p, self.sigma_a
        if s1 <= 0.0 or s2 <= 0.0 or rho_ln == 0.0:
            return rho_ln
        num = exp(rho_ln * s1 * s2) - 1.0
        den = sqrt((exp(s1 * s1) - 1.0) * (exp(s2 * s2) - 1.0))
        if den <= 0.0:  # pragma: no cover - defensive
            return rho_ln
        return max(-0.999, min(0.999, num / den))

    def with_(self, **kwargs) -> "PAreaForceModel":
        return replace(self, **kwargs)


# --------------------------------------------------------------------------- #
# Force statistics
# --------------------------------------------------------------------------- #
def force_moments(model: PAreaForceModel, ignore_correlation: bool = False) -> dict:
    """Lognormal parameters of the annual extreme ice force.

    ``ignore_correlation=True`` keeps both marginals and sets ``rho = 0``
    (the negative control); the median force is unchanged, only the log-scale
    scatter grows by ``2 * ex * sigma_lnA**2`` in the variance.
    """
    # The log-space correlation is the physically meaningful one here: it is the
    # correlation of ``ln p`` and ``ln A`` fixed by the ISO power law, and the
    # one the assembled Nataf problem reproduces exactly.
    rho = 0.0 if ignore_correlation else model.pressure_area_correlation()
    var = (
        model.sigma_kf**2
        + model.sigma_p**2
        + model.sigma_a**2
        + 2.0 * rho * model.sigma_p * model.sigma_a
    )
    sigma = sqrt(max(var, 1e-18))
    mu = log(model.median_force_mn)
    return {
        "mu_ln_force": mu,
        "sigma_ln_force": sigma,
        "median_force_mn": exp(mu),
        "mean_force_mn": exp(mu + 0.5 * sigma**2),
        "cov_force": sqrt(max(exp(sigma**2) - 1.0, 0.0)),
        "rho_ln_p_a": rho,
        "var_ln_force": var,
        "variance_decomposition": {
            "load_model": model.sigma_kf**2,
            "pressure_marginal": model.sigma_p**2,
            "area": model.sigma_a**2,
            "correlation_term": 2.0 * rho * model.sigma_p * model.sigma_a,
        },
    }


def characteristic_force(
    model: PAreaForceModel, ignore_correlation: bool = False
) -> float:
    """Force exceedance level of ``model.return_period_years`` (characteristic action)."""
    mom = force_moments(model, ignore_correlation)
    p_exceedance = 1.0 / model.return_period_years
    return exp(mom["mu_ln_force"] + mom["sigma_ln_force"] * norm_ppf(1.0 - p_exceedance))


def resistance_fractile_factor(model: PAreaForceModel) -> float:
    """``R_k / R_median`` of the resistance (default 5 % fractile)."""
    return exp(model.sigma_r * norm_ppf(model.resistance_quantile))


def design_resistance_median(
    model: PAreaForceModel, kappa: float, ignore_correlation: bool = False
) -> float:
    """Median resistance of a member sized so that ``R_k = kappa * F_k``.

    ``kappa = gamma_f * psi`` is the only convention-free quantity of the
    design format ``gamma_f F_k <= R_k / psi``.
    """
    if kappa <= 0.0:
        raise ValueError("kappa must be positive")
    f_k = characteristic_force(model, ignore_correlation)
    return kappa * f_k / resistance_fractile_factor(model)


# --------------------------------------------------------------------------- #
# Reliability problem
# --------------------------------------------------------------------------- #
VARIABLE_NAMES = ("k_F", "p_ice", "A_contact", "R_member")


def _marginals(model: PAreaForceModel, r_median: float) -> list:
    a_sigma = model.sigma_a
    return [
        _unit_lognormal(model.sigma_kf),
        LognormalDistribution(
            mu_ln=log(model.median_pressure_mpa), sigma_ln=max(model.sigma_p, 1e-9)
        ),
        LognormalDistribution(
            mu_ln=log(model.a_median_m2), sigma_ln=max(a_sigma, 1e-9)
        ),
        LognormalDistribution(mu_ln=log(r_median), sigma_ln=max(model.sigma_r, 1e-9)),
    ]


def build_problem(
    model: PAreaForceModel,
    kappa: float,
    ignore_correlation: bool = False,
    name: str | None = None,
) -> ReliabilityProblem:
    """``g = R - k_F p A`` for a member designed with ``R_k = kappa F_k``.

    The marginals do not depend on ``ignore_correlation``; only the Gaussian
    copula correlation between ``p_ice`` and ``A_contact`` does, which is what
    makes the negative control a controlled experiment.
    """
    r_median = design_resistance_median(model, kappa, ignore_correlation)
    marginals = _marginals(model, r_median)
    rho = 0.0 if ignore_correlation else model.pressure_area_pearson()
    correlation = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, rho, 0.0],
        [0.0, rho, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]

    def g(x) -> float:
        k_f, p_ice, area, r_member = x
        return r_member - k_f * p_ice * area

    return ReliabilityProblem(
        NatafTransform(marginals, correlation),
        g,
        name=name or ("ice_force_independent" if ignore_correlation else "ice_force_correlated"),
        variable_names=VARIABLE_NAMES,
    )


def analytic_beta_ln(
    model: PAreaForceModel, kappa: float, ignore_correlation: bool = False
) -> float:
    """Closed-form ln-space reliability index (independent cross-check of FORM).

    For ``g = R - F`` with independent lognormal ``R`` and ``F`` the failure
    probability is exactly ``P(ln R < ln F)``, so
    ``beta_ln = (mu_lnR - mu_lnF) / sqrt(sigma_lnR**2 + sigma_lnF**2)``.
    This is a first-order-in-log index, not the Hasofer-Lind index of the same
    surface; the two agree to a few percent and the difference is reported.
    """
    mom = force_moments(model, ignore_correlation)
    r_median = design_resistance_median(model, kappa, ignore_correlation)
    mu_r = log(r_median)
    return (mu_r - mom["mu_ln_force"]) / sqrt(model.sigma_r**2 + mom["sigma_ln_force"] ** 2)