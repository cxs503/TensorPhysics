"""Ice-structure interaction physics for TensorPhysics.

Domain module for the ice-engineering side of ship overall performance:
adfreeze (freeze-bond) mechanics of the ice/structure joint and the breakout
("escape") criterion for a beset vessel.

References
----------
* Frederking, R. (1984) / Frederking & Barker (1979): adfreeze shear strength
  of ice on steel vs temperature (calibration data set).
* Jordaan, I. (2001): review of ice-structure adhesion and compressive
  strength data.
* Frankenstein & Garner (1967): brine volume fraction of sea ice.
* Assur (1958) / Cox & Weeks (1983): seawater liquidus (freezing point
  depression), brine salinity vs temperature.
* Stefan problem: ``h = sqrt(2 k dT t / (rho L))`` for the freezing water film.
* Ice resistance (three-state, full-spectrum): Lindqvist (1989) level ice,
  brash-ice coverage decay law and Keinonen-anchored ridge ramming -- see
  :mod:`tensorphysics.ice.resistance` and ``benchmarks/ice_resistance``.
"""

from .adfreeze import (
    AdfreezeState,
    PARAMETER_PROVENANCE,
    adfreeze_strength,
    adfreeze_strength_curve,
    adfreeze_strength_state,
    brine_salinity,
    brine_volume_fraction,
    calibrate_reference_strength,
    contact_time_to_saturation,
    frozen_area_fraction,
    reference_strength_mpa,
    roughness_factor,
    salinity_factor,
    seawater_freezing_point,
    stefan_frozen_thickness,
)
from .breakout import (
    DEFAULT_FRICTION_COEFF,
    DEFAULT_NORMAL_PRESSURE_PA,
    BreakoutResult,
    ShipDefinition,
    bollard_thrust_mn,
    breakout_force_mn,
    contact_band_height,
    empirical_contact_area,
    evaluate_breakout,
    friction_force_mn,
    hull_ice_contact_area,
    required_debonded_area,
    waterline_half_breadth,
)
from .resistance import (
    DomainLimits,
    IceDomainError,
    IceState,
    ShipGeometry as IceResistanceShipGeometry,
    brash_ice_resistance,
    coefficient_sensitivity,
    coverage_decay,
    delivered_power_MW,
    ice_resistance,
    level_ice_resistance,
    ram_capacity,
    ridge_field_mean_resistance,
    ridge_penetration_work,
    ridge_resistance,
    speed_for_power,
)
from .degradation import (
    DEFAULT_INTERFACE_G_C,
    DegradationState,
    cohesive_law,
    combined_degradation,
    critical_slip,
    peel_sweep,
    peel_softening,
    salinity_degradation_factor,
    thermal_degradation_factor,
)
from .references import (
    FREDERKING_ADFREEZE,
    SALINE_TO_FRESH_RATIO,
    SHIPS,
    frederking_strengths,
    frederking_temperatures,
)

__all__ = [
    "AdfreezeState",
    "BreakoutResult",
    "DEFAULT_FRICTION_COEFF",
    "DEFAULT_INTERFACE_G_C",
    "DomainLimits",
    "IceDomainError",
    "IceResistanceShipGeometry",
    "IceState",
    "brash_ice_resistance",
    "coefficient_sensitivity",
    "coverage_decay",
    "delivered_power_MW",
    "ice_resistance",
    "level_ice_resistance",
    "ram_capacity",
    "ridge_field_mean_resistance",
    "ridge_penetration_work",
    "ridge_resistance",
    "speed_for_power",
    "DEFAULT_NORMAL_PRESSURE_PA",
    "DegradationState",
    "FREDERKING_ADFREEZE",
    "PARAMETER_PROVENANCE",
    "SALINE_TO_FRESH_RATIO",
    "SHIPS",
    "ShipDefinition",
    "adfreeze_strength",
    "adfreeze_strength_curve",
    "adfreeze_strength_state",
    "bollard_thrust_mn",
    "breakout_force_mn",
    "brine_salinity",
    "brine_volume_fraction",
    "calibrate_reference_strength",
    "cohesive_law",
    "combined_degradation",
    "contact_band_height",
    "contact_time_to_saturation",
    "critical_slip",
    "empirical_contact_area",
    "evaluate_breakout",
    "frederking_strengths",
    "frederking_temperatures",
    "friction_force_mn",
    "frozen_area_fraction",
    "hull_ice_contact_area",
    "peel_sweep",
    "peel_softening",
    "reference_strength_mpa",
    "required_debonded_area",
    "roughness_factor",
    "salinity_degradation_factor",
    "salinity_factor",
    "seawater_freezing_point",
    "stefan_frozen_thickness",
    "thermal_degradation_factor",
    "waterline_half_breadth",
]
# Cross-implementation attribution (P519-N02)
from .cross_impl_attribution import (
    COMPONENT_NAMES,
    CaliberFingerprint,
    RefShip,
    attribute_divergence,
    build_fingerprints,
    calibrate_crushing_coefficient,
    decompose_five,
    impl_ia_lindqvist,
    impl_lit_corrected,
    impl_lit_original,
    impl_tp_level,
    run_full_attribution,
    sensitivity_matrix,
    validate_fingerprints,
    write_artifacts,
)

__all__.extend([
    "COMPONENT_NAMES",
    "CaliberFingerprint",
    "RefShip",
    "attribute_divergence",
    "build_fingerprints",
    "calibrate_crushing_coefficient",
    "decompose_five",
    "impl_ia_lindqvist",
    "impl_lit_corrected",
    "impl_lit_original",
    "impl_tp_level",
    "run_full_attribution",
    "sensitivity_matrix",
    "validate_fingerprints",
    "write_artifacts",
])
