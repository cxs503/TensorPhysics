"""Elastic-foundation validation surface (beams + plates on Winkler/Pasternak).

This sub-package provides an *analytical-anchor* verification surface for the
elastic-foundation members of the ice-sheet bending chain:

``beams``
    Infinite Euler-Bernoulli beam on a Winkler / Pasternak (two-parameter)
    foundation: closed-form Green functions (Hetényi 1946; Selvadurai 2000) plus
    a Hermite-beam finite element used as the independent numerical check.

``pasternak_plate``
    Infinite plate on a two-parameter (Pasternak) bed: exact closed-form Green
    function ``w = P [K_0(mu_1 r) - K_0(mu_2 r)]/[2 pi D (m_1 - m_2)]``, its
    independent Hankel-integral re-derivation, patch and moment solutions, plus
    the Vlasov identification of ``k, G_p`` with a compressible elastic layer.

``halfspace``
    Elastic half-space (Boussinesq continuum) foundation: exact spectral
    solution ``[D alpha^4 + C alpha] W = Q``, the Boussinesq closed forms that
    fix its plate-free limit, and the punch / point-load references.

``crack_spacing``
    Ring-crack spacing and first-crack load of a bending-failing ice plate:
    moment-extremum radii, ``s ~ (D/k)^(1/4)`` scaling exponent, bearing
    capacity.

``plates``
    Infinite Winkler plate under a point load (Hetényi/Kelvin ``kei`` solution),
    an independent Hankel-integral re-derivation, the circular-patch load
    solution, an axisymmetric Kirchhoff ring-element finite element, and the
    exact finite-radius plate solution (Kelvin particular + regular growing
    homogeneous pair) used to quantify rigid-boundary effects.

Conventions
-----------
* Deflection ``w`` is positive **downwards**; the transverse load ``q`` is
  positive downwards.
* ``D = E h^3 / [12 (1 - nu^2)]`` [N m] is the flexural rigidity,
  ``k`` [Pa/m] the subgrade (Winkler) modulus,
  ``G_p`` [N/m] the Pasternak shear-layer modulus.
* Characteristic length of the *plate*: ``l = (D/k)^(1/4)``.
  Characteristic length of the *beam*: ``l_b = (4 EI/k)^(1/4) = 1/beta``.
"""

from __future__ import annotations

from .beams import (  # noqa: F401
    beam_foundation_roots,
    hetenyi_beam_point_load,
    cosine_only_beam_point_load,
    pasternak_beam_point_load,
    point_load_shear_jump,
    beam_char_length,
    solve_beam_foundation_fem,
    BeamFEMResult,
)
from .pasternak_plate import (  # noqa: F401
    match_pasternak_to_halfspace,
    pasternak_plate_bilaplacian_residual,
    pasternak_plate_center_deflection,
    pasternak_plate_char_length,
    pasternak_plate_moment_r,
    pasternak_plate_patch_load,
    pasternak_plate_patch_moment_r,
    pasternak_plate_point_load,
    pasternak_plate_point_load_hankel,
    pasternak_plate_roots,
    vlasov_foundation_params,
    vlasov_layer_depth,
)
from .halfspace import (  # noqa: F401
    boussinesq_patch_deflection,
    boussinesq_point_deflection,
    boussinesq_punch_deflection,
    halfspace_foundation_modulus,
    halfspace_plate_char_length,
    halfspace_plate_center_deflection,
    halfspace_plate_moment_r,
    halfspace_plate_patch_load,
    halfspace_plate_spectral_integrand,
)
from .crack_spacing import (  # noqa: F401
    bearing_capacity_ratio,
    crack_spacing,
    first_crack_load,
    flexural_strength_moment,
    local_log_slope,
    moment_extrema_radii,
    pasternak_ring_crack_period,
    spacing_scaling_exponent,
)
from .plates import (  # noqa: F401
    winkler_plate_flexural_rigidity,
    winkler_plate_char_length,
    winkler_plate_point_load,
    winkler_plate_point_load_hankel,
    winkler_plate_patch_load,
    winkler_plate_patch_moment_r,
    kelvin_k0_derivatives,
    winkler_plate_load_recovery,
    winkler_plate_moment_r,
    winkler_plate_biharmonic_residual,
    solve_axisym_plate_fem,
    AxisymPlateFEMResult,
    finite_plate_solution,
    FinitePlateSolution,
    finite_plate_center_deflection,
)

__all__ = [
    "match_pasternak_to_halfspace",
    "pasternak_plate_bilaplacian_residual",
    "pasternak_plate_center_deflection",
    "pasternak_plate_char_length",
    "pasternak_plate_moment_r",
    "pasternak_plate_patch_load",
    "pasternak_plate_patch_moment_r",
    "pasternak_plate_point_load",
    "pasternak_plate_point_load_hankel",
    "pasternak_plate_roots",
    "vlasov_foundation_params",
    "vlasov_layer_depth",
    "boussinesq_patch_deflection",
    "boussinesq_point_deflection",
    "boussinesq_punch_deflection",
    "halfspace_foundation_modulus",
    "halfspace_plate_char_length",
    "halfspace_plate_center_deflection",
    "halfspace_plate_moment_r",
    "halfspace_plate_patch_load",
    "halfspace_plate_spectral_integrand",
    "bearing_capacity_ratio",
    "crack_spacing",
    "first_crack_load",
    "flexural_strength_moment",
    "local_log_slope",
    "moment_extrema_radii",
    "pasternak_ring_crack_period",
    "spacing_scaling_exponent",
    "beam_foundation_roots",
    "hetenyi_beam_point_load",
    "cosine_only_beam_point_load",
    "pasternak_beam_point_load",
    "point_load_shear_jump",
    "beam_char_length",
    "solve_beam_foundation_fem",
    "BeamFEMResult",
    "winkler_plate_flexural_rigidity",
    "winkler_plate_char_length",
    "winkler_plate_point_load",
    "winkler_plate_point_load_hankel",
    "winkler_plate_patch_load",
    "winkler_plate_patch_moment_r",
    "kelvin_k0_derivatives",
    "winkler_plate_load_recovery",
    "winkler_plate_moment_r",
    "winkler_plate_biharmonic_residual",
    "solve_axisym_plate_fem",
    "AxisymPlateFEMResult",
    "finite_plate_solution",
    "FinitePlateSolution",
    "finite_plate_center_deflection",
]