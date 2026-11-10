"""Ice-sheet bending fracture with cohesive zones coupled to J2 elastoplasticity.

New capability of node P180W2-N04: a *numerical* (FEM) verification face for
ice-plate breaking that combines

* cohesive zone models -- bilinear and exponential traction-separation laws,
  and the moment-rotation *cohesive hinge* for bending cracks
  (:mod:`~tensorphysics.fem_ice_plate.cohesive_zone`),
* J2 flow theory with isotropic hardening, a 3-D return mapping, a genuine
  radial-return FE solution of the pressurised thick cylinder and closed-form
  reference fields (:mod:`~tensorphysics.fem_ice_plate.j2_plasticity`),
* an axisymmetric Kirchhoff plate on a Winkler foundation with a cohesive ring
  crack, whose intact limit reproduces the Hetenyi/Kelvin solution
  (:mod:`~tensorphysics.fem_ice_plate.foundation_plate`),
* a coupled bar + interface model with an exact three-way energy audit
  (:mod:`~tensorphysics.fem_ice_plate.coupled`), and
* the machine-checkable verification harness for criteria C1..C7
  (:mod:`~tensorphysics.fem_ice_plate.verification`).

Existing analytic modules (``tensorphysics.plasticity``,
``tensorphysics.fracture_mechanics``, ``tensorphysics.elastic_foundation``) are
reused as *reference* solutions; nothing here duplicates their closed forms.
"""

from .cohesive_zone import BilinearTSL, CohesiveHinge, ExponentialTSL
from .coupled import BarResult, solve_bar_with_interface
from .foundation_plate import AxisymPlateResult, solve_axisym_plate_foundation
from .j2_plasticity import (
    CylinderFEResult,
    J2Material,
    J2State,
    equivalent_stress,
    return_map,
    thick_cylinder_fe,
    uniaxial_tension_curve,
)

__version__ = "0.1.0"

__all__ = [
    "BilinearTSL",
    "ExponentialTSL",
    "CohesiveHinge",
    "BarResult",
    "solve_bar_with_interface",
    "AxisymPlateResult",
    "solve_axisym_plate_foundation",
    "CylinderFEResult",
    "J2Material",
    "J2State",
    "equivalent_stress",
    "return_map",
    "thick_cylinder_fe",
    "uniaxial_tension_curve",
]
