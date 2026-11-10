"""J2 flow theory elastoplastic constitutive model and verification suite.

Modules
-------
j2_flow : J2 flow theory with isotropic hardening, analytical solutions for
    uniaxial tension, thick-walled cylinder elastoplastic expansion, unloading
    paths, and energy accounting.
"""

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
