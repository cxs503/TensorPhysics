"""Ice-coupled wave mechanics domain package (W5-N05).

Provides:

* :mod:`tensorphysics.ice_wave.flexural` -- flexural-gravity wave dispersion under
  a continuous ice cover (analytical dispersion + independent spectral
  collocation eigensolver).
* :mod:`tensorphysics.ice_wave.miz` -- marginal ice zone (MIZ) attenuation of
  wave amplitude ``A(x) = A0 exp(-alpha x)`` via a Kelvin-Voigt effective medium,
  with energy-flux cross-check and literature band comparison.
* :mod:`tensorphysics.ice_wave.dem2d` -- minimal 2-D soft-disc granular DEM shear
  cell (Lees-Edwards periodic shear) used to produce contact-force networks.
* :mod:`tensorphysics.ice_wave.homogenise` -- Love-Weber stress recovery and
  Cauchy consistency (volume average vs. per-particle virial average), including
  rotational covariance under rigid-body rotation.
* :mod:`tensorphysics.ice_wave.rheology` -- mu(I) inertial-number rheology
  calibration ``mu(I) = mu_s + (mu_2 - mu_s) / (I_0 / I + 1)``.
* :mod:`tensorphysics.ice_wave.enkf` -- ensemble Kalman analysis used to feed the
  homogenised brash-ice fields back as an assimilation background (W5-N04
  interface).
"""

from __future__ import annotations

from .flexural import (
    FlexuralWaveError,
    flexural_rigidity,
    flexural_gravity_dispersion,
    flexural_gravity_wavenumber,
    phase_speed,
    group_speed,
    spectral_dispersion,
    spectral_residual,
)
from .enkf import EnKFError, ensemble_kalman_analysis
from .homogenise import (
    HomogenisationError,
    love_weber_stress,
    per_particle_virial_stress,
    rotate_particles,
    rotational_covariance_error,
)
from .rheology import MuIError, inertial_number, mu_jop, fit_mu_i
from .miz import (
    MIZError,
    kelvin_voigt_wavenumber,
    attenuation_from_complex_wavenumber,
    exponential_decay_profile,
    fit_attenuation_coefficient,
    effective_medium_properties,
    sweep_attenuation,
    LITERATURE_ALPHA_BAND,
)

__all__ = [
    "FlexuralWaveError",
    "flexural_rigidity",
    "flexural_gravity_dispersion",
    "flexural_gravity_wavenumber",
    "phase_speed",
    "group_speed",
    "spectral_dispersion",
    "spectral_residual",
    "EnKFError",
    "ensemble_kalman_analysis",
    "HomogenisationError",
    "love_weber_stress",
    "per_particle_virial_stress",
    "rotate_particles",
    "rotational_covariance_error",
    "MuIError",
    "inertial_number",
    "mu_jop",
    "fit_mu_i",
    "MIZError",
    "kelvin_voigt_wavenumber",
    "attenuation_from_complex_wavenumber",
    "exponential_decay_profile",
    "fit_attenuation_coefficient",
    "effective_medium_properties",
    "sweep_attenuation",
    "LITERATURE_ALPHA_BAND",
]