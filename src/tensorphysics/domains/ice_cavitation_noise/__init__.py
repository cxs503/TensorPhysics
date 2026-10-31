"""Propeller cavitation and underwater radiated noise in ice-water mixture flow.

Reduced-order but *real* physics kernel (no hard-coded verdicts):

* ``cavitation``  -- Schnerr-Sauer / Zwart-Gerber-Belamri mass transfer plus
  Rayleigh-Plesset single bubble dynamics (nucleation, growth, collapse).
* ``particles``   -- dilute ice-particle phase: number density from volumetric
  concentration, Schiller-Naumann drag, momentum coupling and the
  particle-induced pressure-fluctuation model.
* ``flow1d``      -- quasi-1-D unsteady homogeneous-mixture solver
  (projection / pressure-correction on a staggered grid) with vapor volume
  fraction transport and particle momentum coupling.
* ``acoustics``   -- Lighthill / Ffowcs Williams-Hawkings compact-source far
  field extrapolation (monopole thickness + dipole loading), Kirchhoff sphere
  extrapolation, retardation, spectra and tone detection.

See ``docs/reports/W3N06_ice_cavitation_noise.md`` for the validation record.
"""

from .cavitation import (
    CavitationParams,
    RHO_L,
    RHO_V,
    P_V,
    SIGMA_L,
    MU_L,
    alpha_from_radius,
    radius_from_alpha,
    schnerr_sauer_source,
    zwart_gerber_belamri_source,
    rayleigh_collapse_time,
    bubble_radius_history,
    rayleigh_growth_velocity,
)
from .particles import (
    IcePhaseParams,
    number_density,
    drag_coefficient,
    particle_response_time,
    drag_force_per_volume,
    pressure_fluctuation_rms,
    fluctuating_drag_force,
)
from .flow1d import (
    Quasi1DNozzle,
    MixtureFlow1D,
    inception_sigma,
    sweep_sigma,
)
from .acoustics import (
    monopole_far_field,
    dipole_far_field,
    cavity_volume_acceleration,
    compact_source_far_field,
    kirchhoff_sphere_extrapolation,
    amplitude_spectrum,
    peak_frequency,
    blade_pass_frequency,
)

__all__ = [
    "CavitationParams",
    "RHO_L",
    "RHO_V",
    "P_V",
    "SIGMA_L",
    "MU_L",
    "alpha_from_radius",
    "radius_from_alpha",
    "schnerr_sauer_source",
    "zwart_gerber_belamri_source",
    "rayleigh_collapse_time",
    "bubble_radius_history",
    "rayleigh_growth_velocity",
    "IcePhaseParams",
    "number_density",
    "drag_coefficient",
    "particle_response_time",
    "drag_force_per_volume",
    "pressure_fluctuation_rms",
    "fluctuating_drag_force",
    "Quasi1DNozzle",
    "MixtureFlow1D",
    "inception_sigma",
    "sweep_sigma",
    "monopole_far_field",
    "dipole_far_field",
    "cavity_volume_acceleration",
    "compact_source_far_field",
    "kirchhoff_sphere_extrapolation",
    "amplitude_spectrum",
    "peak_frequency",
    "blade_pass_frequency",
]