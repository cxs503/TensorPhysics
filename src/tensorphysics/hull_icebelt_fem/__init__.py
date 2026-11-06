"""Ice-belt local structural FEM cross-check (campaign node IN11-H4).

Modules
-------
iacs_pressure
    IACS Polar Class design pressure of the ice belt: class table, rule patch
    geometry and the pressure-area law, with the declared anchor provenance.
analytic
    Closed-form Navier double-sine solution of the simply supported plate under
    a piecewise constant patch load (the independent reference).
panel_fem
    Ice-belt plate panel driven through the reused H3 Hermite flat-shell
    element of :mod:`tensorphysics.plate_buckling`, with bending-stress
    recovery ``sigma = 6 M / t^2``.
stiffener
    Longitudinal stiffener beam check under the same ice pressure.
xcheck
    Dual-chain comparison against the ship platform B1 ``ice_fullscale``
    experience chain, with deviation table and closed-form attribution.

Nothing in this package re-implements the finite element, the IACS patch
geometry or the plate solver; those are imported from
:mod:`tensorphysics.plate_buckling`.
"""

from . import analytic, iacs_pressure, panel_fem, stiffener, xcheck  # noqa: F401
from .iacs_pressure import (  # noqa: F401
    ANCHOR_PROVENANCE,
    IACSPressureError,
    IcePatchDesignPressure,
    POLAR_CLASS_ICE_THICKNESS,
    design_ice_thickness,
    design_patch_pressure,
    pressure_area_law,
)
from .panel_fem import IceBeltPanel, PanelSolution, STEEL_GRADES  # noqa: F401
from .stiffener import StiffenerProfile, stiffener_response  # noqa: F401
from .xcheck import run_crosscheck  # noqa: F401

__all__ = [
    "ANCHOR_PROVENANCE",
    "IACSPressureError",
    "IceBeltPanel",
    "IcePatchDesignPressure",
    "PanelSolution",
    "POLAR_CLASS_ICE_THICKNESS",
    "STEEL_GRADES",
    "StiffenerProfile",
    "analytic",
    "design_ice_thickness",
    "design_patch_pressure",
    "iacs_pressure",
    "panel_fem",
    "pressure_area_law",
    "run_crosscheck",
    "stiffener",
    "stiffener_response",
    "xcheck",
]

__version__ = "0.1.0"