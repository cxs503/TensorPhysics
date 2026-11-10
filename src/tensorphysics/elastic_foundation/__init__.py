"""Elastic-foundation validation surface (beams + plates on Winkler/Pasternak).

This sub-package provides an *analytical-anchor* verification surface for the
elastic-foundation members of the ice-sheet bending chain:

``beams``
    Infinite Euler-Bernoulli beam on a Winkler / Pasternak (two-parameter)
    foundation: closed-form Green functions (Hetényi 1946; Selvadurai 2000) plus
    a Hermite-beam finite element used as the independent numerical check.

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