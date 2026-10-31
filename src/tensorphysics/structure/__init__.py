"""Structural-mechanics domain: finite-rotation (geometrically nonlinear) hull analysis.

This sub-package hosts large-rotation structural capabilities that are needed by the
ice-navigation campaign (roll / heel of the hull girder and transverse frames under
ice loads).  The central ingredient is a *co-rotational* kinematical description:
strains and section forces are measured in a frame that rotates with the element, so
the formulation remains objective (invariant under rigid-body rotation) while the
nodal rotations of the mesh may be arbitrarily large.

Modules
-------
finite_rotation
    Planar finite-rotation primitives (rotation maps, angle wrapping, polar
    decomposition of a 2x2 deformation gradient, rotational kinematics of a hull
    section about the longitudinal axis).
corotational_beam
    Co-rotational 2-node beam element with an explicit strain-energy functional.
    Internal forces and tangent stiffness are obtained as the exact gradient /
    Hessian of that energy (algorithmic differentiation), which makes the element
    energy-consistent for explicit/leapfrog time integration.
hull_roll
    Hull transverse-frame idealisation of the side shell as a co-rotational beam
    strip, the 15 deg roll load case, the analytic statics reference used to score
    it, and the conservative large-rotation transient used for the energy ledger.
"""

from .finite_rotation import (
    Rotation2D,
    polar_decomposition_2x2,
    roll_transform,
    wrap_angle,
)
from .corotational_beam import (
    CorotationalBeamModel,
    Section,
    coRotationalElementEnergy,
)
from .hull_roll import (
    HULL_SECTION,
    IceClassHullRollCase,
    build_side_shell_strip,
    analytic_root_stress_rolled,
    frozen_geometry_root_stress,
)

__all__ = [
    "Rotation2D",
    "polar_decomposition_2x2",
    "roll_transform",
    "wrap_angle",
    "CorotationalBeamModel",
    "Section",
    "coRotationalElementEnergy",
    "HULL_SECTION",
    "IceClassHullRollCase",
    "build_side_shell_strip",
    "analytic_root_stress_rolled",
    "frozen_geometry_root_stress",
]