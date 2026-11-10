"""Design-sea-state statistical-convention verification surface (p229w4-n05).

The package answers one question with numbers instead of folklore: *do the
design sea states produced by IFORM, by the target reliability index, and by
the Rayleigh short-term extreme convention agree — and where do they not?*

Modules
-------
:mod:`~tensorphysics.design_sea_state.rosenblatt`
    Closed-form Rosenblatt probability-integral transform of the two-block
    ``(H_s, T_z)`` sea-state model (3-parameter Weibull marginal x log-normal
    power-law conditional), plus a linear correlated-normal anchor transform.
:mod:`~tensorphysics.design_sea_state.iform`
    Inverse first-order reliability method contours: the radius-``beta`` circle
    in standard-normal space mapped back to physical ``(H_s, T_z)``, with the
    exact extreme-``H_s`` anchor and the chi-square-ellipse anti-swap residual.
:mod:`~tensorphysics.design_sea_state.reliability_index`
    Target reliability index ``beta`` versus annual exceedance probability in
    both conventions (``Phi(-beta)`` and ``exp(-beta**2/2)``), return periods,
    lifetime targets, DNV/ISO 2394 reference rows and the mismatch report.
:mod:`~tensorphysics.design_sea_state.rayleigh_consistency`
    Rayleigh short-term conventions (mode ``H_s/4``, ``N``-wave modal maximum
    ``H_s sqrt(ln N / 2)``), Monte-Carlo audit, and the three-way adjudication
    IFORM / Rayleigh rule / exact long-term quadrature.
:mod:`~tensorphysics.design_sea_state.api`
    One-call study plus machine criteria rows.

Example
-------
>>> from tensorphysics.design_sea_state import run_study   # doctest: +SKIP
>>> out = run_study({"annual_pf": 1e-2})
>>> all(row["verdict"] == "PASS" for row in out["criteria"])
True
"""

from .api import (
    DEFAULT_CONFIG,
    MACHINE_CRITERIA_KEYS,
    build_model,
    evaluate_criteria,
    mc_outside_probability,
    points_outside_polygon,
    run_study,
)
from .iform import (
    ContourPoint,
    analytic_max_hs,
    contour_array,
    contour_max_hs,
    contour_refinement_error,
    iform_contour,
    magnitude_on_contour,
    mahalanobis_residual,
    max_response_on_contour,
    quadratic_residual_normal,
    u_circle,
)
from .rayleigh_consistency import (
    G,
    RHO_WATER,
    SEA_STATE_HOURS,
    adjudicate_rayleigh_vs_iform,
    long_term_reference,
    modal_largest_wave,
    modal_largest_wave_response,
    quadrature_grid,
    rainflow_n_waves,
    rayleigh_design_sea_state,
    rayleigh_exceedance,
    rayleigh_mc_check,
    rayleigh_mean,
    rayleigh_mode,
    rayleigh_scale,
    elevation_sigma,
    wave_power,
)
from .reliability_index import (
    DNV_ISO2394_TARGETS,
    ReliabilityTarget,
    beta_contour_from_pout,
    beta_cornell_from_pf,
    calibrate_contour_beta,
    convention_report,
    lifetime_pf,
    pf_from_beta_cornell,
    pf_from_return_period,
    pout_from_beta_contour,
    return_period_from_lifetime_pf,
    return_period_from_pf,
)
from .rosenblatt import (
    BivariateNormalTransform,
    LognormalConditional,
    RosenblattTransform,
    Weibull3,
    standard_normal_cdf,
    standard_normal_ppf,
    standard_normal_pdf,
)

__all__ = [
    "DEFAULT_CONFIG",
    "MACHINE_CRITERIA_KEYS",
    "build_model",
    "evaluate_criteria",
    "mc_outside_probability",
    "points_outside_polygon",
    "run_study",
    "ContourPoint",
    "analytic_max_hs",
    "contour_array",
    "contour_max_hs",
    "contour_refinement_error",
    "iform_contour",
    "magnitude_on_contour",
    "mahalanobis_residual",
    "max_response_on_contour",
    "quadratic_residual_normal",
    "u_circle",
    "G",
    "RHO_WATER",
    "SEA_STATE_HOURS",
    "adjudicate_rayleigh_vs_iform",
    "long_term_reference",
    "modal_largest_wave",
    "modal_largest_wave_response",
    "quadrature_grid",
    "rainflow_n_waves",
    "rayleigh_design_sea_state",
    "rayleigh_exceedance",
    "rayleigh_mc_check",
    "rayleigh_mean",
    "rayleigh_mode",
    "rayleigh_scale",
    "elevation_sigma",
    "wave_power",
    "DNV_ISO2394_TARGETS",
    "ReliabilityTarget",
    "beta_contour_from_pout",
    "beta_cornell_from_pf",
    "calibrate_contour_beta",
    "convention_report",
    "lifetime_pf",
    "pf_from_beta_cornell",
    "pf_from_return_period",
    "pout_from_beta_contour",
    "return_period_from_lifetime_pf",
    "return_period_from_pf",
    "BivariateNormalTransform",
    "LognormalConditional",
    "RosenblattTransform",
    "Weibull3",
    "standard_normal_cdf",
    "standard_normal_ppf",
    "standard_normal_pdf",
]