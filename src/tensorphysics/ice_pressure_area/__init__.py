"""Ice pressure-area parameter library, convention algebra and calibration.

The gap ``GAP-PP-2-ICE-STRUCT-PRESSURE-AREA-CALIB`` needs the pressure-area
exponent ``ex`` to stop being a bare scan range ``0.30-0.70``.  This package
delivers three things:

* :mod:`.library` -- a source-level parameter library where every exponent
  carries citation, units, **area definition**, footprint geometry and
  conditions, with sha256 anchors and a family-conflict detector;
* :mod:`.conventions` -- the algebra that maps an exponent between area
  conventions (law ``ex_ref = 1 + (ex_src - 1) * r``), including the proof that
  a constant area ratio leaves ``ex`` invariant;
* :mod:`.calibrate` -- DerSimonian-Laird random-effects pooling per scale
  stratum, leave-one-out and process-zone sweeps, and an identification audit
  for covariates (e.g. ice temperature) that the assembled data cannot support.

Example
-------
>>> from tensorphysics.ice_pressure_area import load_library, pooled_band
>>> lib = load_library()                                  # doctest: +SKIP
>>> lib.describe()["conflicts"][0]["family"]              # doctest: +SKIP
'iacs_ur_i'
"""

from .calibrate import (
    Z90,
    Estimate,
    band_to_sigma,
    covariate_matrix,
    demand_bandwidth_study,
    dl_pool,
    estimates_from_library,
    fit_power_law,
    leave_one_out,
    meta_regression,
)
from .conventions import (
    AREA_CONVENTIONS,
    CONSTANT_RATIO_CONVENTIONS,
    DEFAULT_DAMAGE_RATIO_SWEEP,
    REFERENCE_CONVENTION,
    ConventionError,
    PressureAreaLaw,
    from_reference,
    ratio_factor,
    required_area_ratio,
    required_area_ratio_solutions,
    to_reference,
)
from .library import (
    ACCURACY_REFERENCE,
    EVIDENCE_CLASSES,
    REGRESSION_ANCHOR,
    ROLES,
    SCHEMA,
    LibraryError,
    LibraryIntegrityError,
    PressureAreaEntry,
    PressureAreaLibrary,
    default_root,
    entry_from_dict,
    load_library,
    seal,
    value_digest,
)

__all__ = [
    "SCHEMA",
    "ROLES",
    "ACCURACY_REFERENCE",
    "REGRESSION_ANCHOR",
    "EVIDENCE_CLASSES",
    "AREA_CONVENTIONS",
    "CONSTANT_RATIO_CONVENTIONS",
    "DEFAULT_DAMAGE_RATIO_SWEEP",
    "REFERENCE_CONVENTION",
    "ConventionError",
    "LibraryError",
    "LibraryIntegrityError",
    "PressureAreaEntry",
    "PressureAreaLaw",
    "PressureAreaLibrary",
    "Estimate",
    "Z90",
    "band_to_sigma",
    "covariate_matrix",
    "default_root",
    "demand_bandwidth_study",
    "dl_pool",
    "entry_from_dict",
    "estimates_from_library",
    "fit_power_law",
    "from_reference",
    "leave_one_out",
    "load_library",
    "meta_regression",
    "ratio_factor",
    "required_area_ratio",
    "required_area_ratio_solutions",
    "seal",
    "to_reference",
    "value_digest",
]
