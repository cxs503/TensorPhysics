"""Rate-(in)dependent plasticity models and their verification suites.

Modules
-------
j2_flow : J2 flow theory with isotropic hardening, analytical solutions for
    uniaxial tension, thick-walled cylinder elastoplastic expansion, unloading
    paths, and energy accounting.
cowper_symonds : Cowper-Symonds strain-rate dependent flow stress, its analytic
    inverse, and a self-consistent rate-dependent 1-D return mapping
    (``eps_p_dot`` of the same step drives the overstress).
impact_daf : dynamic amplification factor of pulse-loaded plate strips -- exact
    half-sine SDOF shock spectrum, exact modal elastic reference, and a
    finite-element plate strip with rate-dependent plastic hinges.
"""

from .cowper_symonds import (
    MILD_STEEL_D,
    MILD_STEEL_P,
    CowperSymonds,
    RateDependentJ2,
    UniaxialResponse,
    analytic_uniaxial_flow_stress,
    uniaxial_tension_constant_rate,
)
from .impact_daf import (
    BeamHingeFEM,
    ImpactCase,
    beam_clamped_static_moments,
    beam_first_mode_period,
    beam_modal_elastic_daf,
    beam_rigidity,
    half_sine_pulse,
    pulse_impulse,
    sdof_half_sine_daf,
    sdof_half_sine_daf_rk4,
    sdof_half_sine_response,
)
from .j2_flow import (
    J2FlowMaterial,
    PlasticState,
    uniaxial_tension_curve,
    thick_cylinder_elastoplastic,
    thick_cylinder_tresca_rp,
    thick_cylinder_vonmises_rp,
    unloading_path,
    energy_balance,
    plane_strain_yield,
    plane_stress_yield,
)

__version__ = "0.1.0"

__all__ = [
    "MILD_STEEL_D",
    "MILD_STEEL_P",
    "CowperSymonds",
    "RateDependentJ2",
    "UniaxialResponse",
    "analytic_uniaxial_flow_stress",
    "uniaxial_tension_constant_rate",
    "BeamHingeFEM",
    "ImpactCase",
    "beam_clamped_static_moments",
    "beam_first_mode_period",
    "beam_modal_elastic_daf",
    "beam_rigidity",
    "half_sine_pulse",
    "pulse_impulse",
    "sdof_half_sine_daf",
    "sdof_half_sine_daf_rk4",
    "sdof_half_sine_response",
    "J2FlowMaterial",
    "PlasticState",
    "uniaxial_tension_curve",
    "thick_cylinder_elastoplastic",
    "thick_cylinder_tresca_rp",
    "thick_cylinder_vonmises_rp",
    "unloading_path",
    "energy_balance",
    "plane_strain_yield",
    "plane_stress_yield",
]
