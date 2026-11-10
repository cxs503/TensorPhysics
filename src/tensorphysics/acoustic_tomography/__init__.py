"""Under-ice acoustic tomography: sound speed, ice thickness and temperature.

Physical chain implemented here (all quantities computed, none hard-coded):

* ``sound_speed``   -- campaign Mackenzie-type relation plus the full nine-term
  Mackenzie (1981) cross-check, and the analytic inversion ``c -> T`` with its
  sensitivity ``dc/dT`` (and hence ``sigma_T``).
* ``ray_tracing``   -- Snell-law ray integration (RK4 on the eikonal ray
  equations) in a range-dependent ``c(r, z)`` field, eigenray shooting,
  closed-form linear-profile reference, and Fermat-minimised ice-underside
  reflection time.
* ``tomography``    -- bilinear path-integral matrix, ART (Kaczmarz) and SIRT
  reconstructions with Laplacian regularisation, and ensemble (Monte Carlo)
  uncertainty of the retrieved field.
* ``ice_thickness`` -- draft-mode echo inversion ``d = c t / 2`` with the
  hydrostatic draft -> thickness conversion, plus the through-ice mode and the
  first-order uncertainty budget.
* ``case_under_ice``-- the campaign case: 10-element array on a 4 km x 200 m
  section, warm Atlantic intrusion + melt-water lens, synthetic travel times
  from the ray tracer, inversion, CTD comparison, the three negative controls
  and the falsifiability check.

See ``docs/ACOUSTIC_TOMOGRAPHY_UNDER_ICE.md`` for the validation record.
"""

from .ice_thickness import (
    DRAFT_TO_THICKNESS,
    draft_from_echo_time,
    draft_from_thickness,
    echo_time,
    measure_draft_echo,
    thickness_from_draft,
    thickness_from_echo_time,
    thickness_from_ice_echo_time,
    thickness_uncertainty,
)
from .ray_tracing import (
    EigenRay,
    SoundSpeedField,
    bilinear,
    eigenray_paths,
    find_eigenray,
    find_eigenrays,
    grid_sound_speed_field,
    linear_profile_arc,
    specular_reflection_time,
    straight_leg_time,
    trace_rays,
)
from .sound_speed import (
    RHO_ICE,
    RHO_WATER,
    SOUND_SPEED_LOWER_BOUND,
    engineering_dc_dt,
    engineering_sound_speed,
    ice_sound_speed,
    invert_temperature_from_sound_speed,
    mackenzie_sound_speed,
    sound_speed_valid_mask,
    temperature_uncertainty,
)
from .tomography import (
    NodeGrid,
    art,
    bilinear_quadrature,
    interpolate_nodes,
    invert_sound_speed,
    laplacian,
    monte_carlo_uncertainty,
    path_matrix,
    sirt,
)

__all__ = [
    # sound speed
    "engineering_sound_speed",
    "mackenzie_sound_speed",
    "invert_temperature_from_sound_speed",
    "engineering_dc_dt",
    "temperature_uncertainty",
    "ice_sound_speed",
    "sound_speed_valid_mask",
    "SOUND_SPEED_LOWER_BOUND",
    "RHO_ICE",
    "RHO_WATER",
    # ray tracing
    "SoundSpeedField",
    "EigenRay",
    "trace_rays",
    "find_eigenray",
    "find_eigenrays",
    "eigenray_paths",
    "grid_sound_speed_field",
    "bilinear",
    "linear_profile_arc",
    "straight_leg_time",
    "specular_reflection_time",
    # tomography
    "NodeGrid",
    "path_matrix",
    "bilinear_quadrature",
    "interpolate_nodes",
    "laplacian",
    "art",
    "sirt",
    "invert_sound_speed",
    "monte_carlo_uncertainty",
    # ice thickness
    "echo_time",
    "draft_from_echo_time",
    "thickness_from_draft",
    "draft_from_thickness",
    "thickness_from_echo_time",
    "thickness_from_ice_echo_time",
    "thickness_uncertainty",
    "measure_draft_echo",
    "DRAFT_TO_THICKNESS",
]