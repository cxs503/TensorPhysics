"""Wake wash propagation and environmental compliance module.

This module provides tools for analyzing ship wake wash propagation and
assessing environmental compliance with PIANC (2003) guidelines.

The module implements:

* :mod:`tensorphysics.wake_wash.kelvin_wave` -- Kelvin wake pattern geometry
  (transverse wave angle ≈19.47°, divergent wave cusp angle ≈70.53°).
* :mod:`tensorphysics.wake_wash.wave_height_decay` -- Wave height decay with
  distance (deep water H∝1/√r, shallow water H∝1/r).
* :mod:`tensorphysics.wake_wash.shoreline_impact` -- Shoreline wave height
  estimation including shoaling, breaking, and run-up.
* :mod:`tensorphysics.wake_wash.panc_compliance` -- PIANC (2003) environmental
  limit compliance assessment for various receptor types.
* :mod:`tensorphysics.wake_wash.api` -- Unified API for wake wash analysis.

Physical Background:
    When a ship moves through water, it generates a characteristic wake pattern
    known as the Kelvin wake. This pattern consists of transverse waves
    (perpendicular to ship track) and divergent waves (at an angle to the track).
    
    The transverse waves propagate at the group velocity c_g = c/2, resulting
    in a half-angle of θ_t = arcsin(1/3) ≈ 19.47° from the ship track.
    
    The divergent waves form the outer envelope of the wake, with the cusp
    line at angle θ_d = arccos(1/3) ≈ 70.53° from the track.
    
    Wave height decays with distance due to geometric spreading:
    - Deep water: H ∝ 1/√r (2D spreading)
    - Shallow water: H ∝ 1/r (1D spreading)
    
    The wake wash can impact shoreline erosion, moored vessels, swimmers,
    and marine structures. PIANC (2003) provides guidelines for acceptable
    wave heights at various receptor types.

References:
    - Kelvin, Lord (1887). "On ship waves". Proc. Inst. Mech. Engrs.
    - PIANC (2003). "Guidelines for Managing Wake Wash from High-Speed Vessels".
    - Lighthill, J. (1978). "Waves in Fluids". Cambridge University Press.
"""

from __future__ import annotations

# Kelvin wave geometry
from .kelvin_wave import (
    KelvinWaveError,
    transverse_wave_angle,
    divergent_wave_angle,
    wake_half_angle,
    wake_full_angle,
    kelvin_wake_coordinates,
    wave_wavelength,
    wave_period,
    froude_number,
)

# Wave height decay
from .wave_height_decay import (
    WaveDecayError,
    deep_water_decay,
    shallow_water_decay,
    transition_depth,
    shallow_water_criterion,
    wave_height_with_depth,
    decay_profile,
    energy_flux_conservation,
)

# Shoreline impact
from .shoreline_impact import (
    ShorelineImpactError,
    shoaling_coefficient,
    breaking_criterion,
    runup_estimation,
    shoreline_wave_height,
    wake_wash_impact_assessment,
)

# PIANC compliance
from .panc_compliance import (
    ReceptorType,
    ComplianceLevel,
    PIANCComplianceError,
    PIANC_LIMITS,
    MARGINAL_THRESHOLD,
    get_wave_height_limit,
    check_compliance,
    assess_multiple_receptors,
    risk_classification,
    mitigation_recommendations,
    compliance_report,
    speed_limit_for_compliance,
)

# Unified API
from .api import (
    WakeWashError,
    analyze_wake_pattern,
    predict_wave_height,
    assess_shoreline_impact,
    check_pianc_compliance,
    optimize_speed_for_compliance,
    multi_receptor_assessment,
)

__all__ = [
    # Kelvin wave
    "KelvinWaveError",
    "transverse_wave_angle",
    "divergent_wave_angle",
    "wake_half_angle",
    "wake_full_angle",
    "kelvin_wake_coordinates",
    "wave_wavelength",
    "wave_period",
    "froude_number",
    # Wave decay
    "WaveDecayError",
    "deep_water_decay",
    "shallow_water_decay",
    "transition_depth",
    "shallow_water_criterion",
    "wave_height_with_depth",
    "decay_profile",
    "energy_flux_conservation",
    # Shoreline impact
    "ShorelineImpactError",
    "shoaling_coefficient",
    "breaking_criterion",
    "runup_estimation",
    "shoreline_wave_height",
    "wake_wash_impact_assessment",
    # PIANC compliance
    "ReceptorType",
    "ComplianceLevel",
    "PIANCComplianceError",
    "PIANC_LIMITS",
    "MARGINAL_THRESHOLD",
    "get_wave_height_limit",
    "check_compliance",
    "assess_multiple_receptors",
    "risk_classification",
    "mitigation_recommendations",
    "compliance_report",
    "speed_limit_for_compliance",
    # API
    "WakeWashError",
    "analyze_wake_pattern",
    "predict_wave_height",
    "assess_shoreline_impact",
    "check_pianc_compliance",
    "optimize_speed_for_compliance",
    "multi_receptor_assessment",
]
