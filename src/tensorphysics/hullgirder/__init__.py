"""Hull girder extreme bending moment under combined ice and wave loading.

Domain package (船舶总体性能 -> 结构 / 冰区航行) implementing:

* :mod:`.random_field` - exponentially correlated ice pressure fields
  (exact AR(1) sampling, Karhunen-Loeve for Gaussian covariance, correlation
  length estimation/fitting for DEM-based calibration);
* :mod:`.ice_load`    - ice pressure level, correlation-length model
  ``lambda_x = f(h, c, v)``, contact patch geometry, load cases;
* :mod:`.girder`      - Euler-Bernoulli shear/moment operators (floating hull
  and simply supported conventions), analytic variance propagation;
* :mod:`.wave`        - ITTC-1978 / IACS UR S11 wave bending moment, linear
  long-wave strip solver with Smith correction, ITTC spectrum and long-term
  extreme statistics;
* :mod:`.ultimate`    - simplified ultimate bending capacity with Falkner/John's
  plate buckling reduction and utilisation check;
* :mod:`.extremes`    - Monte-Carlo extremes, Gumbel/Weibull fits, Turkstra
  load combination;
* :mod:`.verification`- the seven machine-checkable acceptance criteria.

Evidence levels (see the mission audit vocabulary): the wave and beam modules
are REAL (closed-form / textbook physics, verified against analytic solutions);
the ultimate capacity is a simplified rule-type estimate; the ice pressure
level and the ``lambda_x`` regression coefficients are literature anchored and
carry the ``literature_anchored`` provenance flag (not re-fitted on a DEM run in
this repository).
"""

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
from .girder import (
    balance_load,
    independent_segment_variance,
    moment_matrix,
    moment_statistics,
    shear_and_moment,
    uniform_load_midspan_moment,
)
from .ice_load import (
    REFERENCE_CORRELATION_COEFFICIENTS,
    IceCondition,
    contact_patch_length,
    correlation_length,
    fit_correlation_length_model,
    global_ice_pressure,
    ice_load_scenarios,
    ice_pressure_profiles,
    pressure_coefficient_of_variation,
)
from .model import ShipGirder
from .random_field import (
    ar1_field,
    autocorrelation_length,
    exponential_covariance,
    fit_exponential_length,
    karhunen_loève_field,
    sample_correlated_field,
    space_time_field,
    squared_exponential_covariance,
)
from .ultimate import (
    elastic_bending_stress,
    plate_buckling_reduction,
    ultimate_bending_moment,
    utilisation_check,
)
from .wave import (
    design_wave_moment,
    ittc_design_wave_moment,
    ittc_spectrum,
    long_term_design_moment,
    long_wave_closed_form_moment,
    moment_rms_from_spectrum,
    north_atlantic_scatter,
    rule_coefficient_c,
    wave_moment_rao,
)

__all__ = [
    "ShipGirder",
    "IceCondition",
    "REFERENCE_CORRELATION_COEFFICIENTS",
    "correlation_length",
    "fit_correlation_length_model",
    "global_ice_pressure",
    "pressure_coefficient_of_variation",
    "contact_patch_length",
    "ice_pressure_profiles",
    "ice_load_scenarios",
    "exponential_covariance",
    "squared_exponential_covariance",
    "ar1_field",
    "karhunen_loève_field",
    "sample_correlated_field",
    "autocorrelation_length",
    "fit_exponential_length",
    "space_time_field",
    "balance_load",
    "shear_and_moment",
    "moment_matrix",
    "moment_statistics",
    "independent_segment_variance",
    "uniform_load_midspan_moment",
    "rule_coefficient_c",
    "ittc_design_wave_moment",
    "wave_moment_rao",
    "design_wave_moment",
    "long_wave_closed_form_moment",
    "ittc_spectrum",
    "moment_rms_from_spectrum",
    "north_atlantic_scatter",
    "long_term_design_moment",
    "plate_buckling_reduction",
    "ultimate_bending_moment",
    "elastic_bending_stress",
    "utilisation_check",
    "moment_batch",
    "ice_bending_extremes",
    "gumbel_fit",
    "gumbel_return_level",
    "weibull_fit",
    "weibull_return_level",
    "spearman_rank_correlation",
    "combine_design_moments",
]