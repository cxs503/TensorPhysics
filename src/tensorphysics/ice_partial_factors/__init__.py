"""Back-calibration of ISO 19906 ice-pressure partial factors (IN13-W3-N10).

The package wires four ingredients into one auditable chain:

1. **Load side** — :mod:`~tensorphysics.ice_partial_factors.pressure_area`:
   the ISO 19906 pressure-area law plus Gumbel/GEV extreme statistics of the
   annual maximum pressure (method of moments, maximum likelihood, L-moments),
   fitted to a reproducible p-A ensemble whose generator plant is documented.
2. **Resistance side** — :mod:`~tensorphysics.ice_partial_factors.limit_state`:
   the ice-belt plate yield limit state ``g = k_R f_y - k_S alpha_b p
   (b/t_net)^2`` with lognormal material/model uncertainties, built on the Nataf
   transform of :mod:`tensorphysics.structural`.
3. **Calibration** — :mod:`~tensorphysics.ice_partial_factors.calibration`:
   the code-calibration inverse (bisection on ``k = gamma_F gamma_M`` such that
   the achieved reliability equals ``beta_T``) plus the JCSS/ISO 2394
   design-value split, with SORM / importance-sampling / Monte-Carlo checks.
4. **Adjudication** — :mod:`~tensorphysics.ice_partial_factors.iso_tables`:
   row-by-row comparison with ISO 19906 Tables A.1/A.2/A.3 (with explicit
   provenance tiers) and an in-repo anchor pair, judged on achieved ``beta``.

Public entry point
------------------
``run_calibration``
    Full study as a JSON-serialisable dict (used by the benchmark runner and the
    report generator).

Example
-------
>>> from tensorphysics.ice_partial_factors import run_calibration   # doctest: +SKIP
>>> out = run_calibration({"beta_targets": [3.8, 4.0, 4.3]})
>>> round(out["calibration"]["gamma_F"], 3) > 1.0
True
"""

from .api import (
    DEFAULT_CONFIG,
    MACHINE_CRITERIA_KEYS,
    evaluate_criteria,
    run_calibration,
)
from .calibration import (
    TARGET_BETA,
    TARGET_PF,
    CalibrationResult,
    ReliabilitySnapshot,
    analyse_problem,
    beta_target_sweep,
    calibrate_factor_product,
    calibrate_panel,
    design_value_factors,
    invert_sweep,
    load_cov_sweep,
    scenario_sweep,
)
from .iso_tables import (
    DESIGN_CASES,
    LOAD_FACTOR_ROWS,
    REPO_ANCHOR_ROWS,
    RESISTANCE_FACTOR_ROWS,
    TARGET_RELIABILITY_ROWS,
    adjudicate_design_cases,
    adjudicate_factor_rows,
    adjudicate_target_reliability,
    summarise_verdicts,
)
from .limit_state import (
    VARIABLE_NAMES,
    CharacteristicValues,
    IceBeltPanel,
    build_limit_state,
    build_problem,
    build_transform,
    characteristic_values,
    design_thickness_for_factor,
    load_effect,
    resistance,
)
from .pressure_area import (
    GevFit,
    GumbelFit,
    GumbelParams,
    PressureAreaTrend,
    plant_annual_params,
    fit_gev_lmoments,
    fit_gev_mle,
    fit_gumbel_mle,
    fit_gumbel_moments,
    fit_pressure_area_trend,
    ks_statistic_gumbel,
    simulate_annual_maxima,
    storm_gumbel,
)

__all__ = [
    "DEFAULT_CONFIG",
    "MACHINE_CRITERIA_KEYS",
    "evaluate_criteria",
    "run_calibration",
    "TARGET_BETA",
    "TARGET_PF",
    "CalibrationResult",
    "ReliabilitySnapshot",
    "analyse_problem",
    "beta_target_sweep",
    "calibrate_factor_product",
    "calibrate_panel",
    "design_value_factors",
    "invert_sweep",
    "load_cov_sweep",
    "scenario_sweep",
    "DESIGN_CASES",
    "LOAD_FACTOR_ROWS",
    "REPO_ANCHOR_ROWS",
    "RESISTANCE_FACTOR_ROWS",
    "TARGET_RELIABILITY_ROWS",
    "adjudicate_design_cases",
    "adjudicate_factor_rows",
    "adjudicate_target_reliability",
    "summarise_verdicts",
    "VARIABLE_NAMES",
    "CharacteristicValues",
    "IceBeltPanel",
    "build_limit_state",
    "build_problem",
    "build_transform",
    "characteristic_values",
    "design_thickness_for_factor",
    "load_effect",
    "resistance",
    "GevFit",
    "GumbelFit",
    "GumbelParams",
    "PressureAreaTrend",
    "plant_annual_params",
    "fit_gev_lmoments",
    "fit_gev_mle",
    "fit_gumbel_mle",
    "fit_gumbel_moments",
    "fit_pressure_area_trend",
    "ks_statistic_gumbel",
    "simulate_annual_maxima",
    "storm_gumbel",
]