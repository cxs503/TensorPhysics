"""
Thin-walled structural mechanics module for TensorPhysics.

Provides:
- Saint-Venant torsion constant J (open & closed sections)
- Bredt's formula for single-cell closed sections
- Roark's series for solid rectangular sections
- Vlasov constrained torsion theory (7-DOF beam element)
- Warping stress and bimoment calculations
- Sectorial coordinate computation and normalization
"""

from .thin_walled_torsion import (
    saint_venant_J_rectangular,
    saint_venant_J_closed_box,
    bredt_J,
    roark_J_rectangular,
    warping_constant_Iw_open,
    warping_constant_Iw_box,
    characteristic_length,
)
from .warping import (
    sectorial_coordinate_closed,
    sectorial_coordinate_open,
    bimoment_decay,
    warping_stress,
    vlasov_ode_solution,
)
from .vlsov_beam import ThinWalledBeam7DOF

__all__ = [
    "saint_venant_J_rectangular",
    "saint_venant_J_closed_box",
    "bredt_J",
    "roark_J_rectangular",
    "warping_constant_Iw_open",
    "warping_constant_Iw_box",
    "characteristic_length",
    "sectorial_coordinate_closed",
    "sectorial_coordinate_open",
    "bimoment_decay",
    "warping_stress",
    "vlasov_ode_solution",
    "ThinWalledBeam7DOF",
]
