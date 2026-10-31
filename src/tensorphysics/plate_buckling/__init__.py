"""Plate-panel buckling and ultimate strength under ice loading.

Modules
-------
core
    Geometry (with validation gates), bilinear plane-stress plasticity,
    flat shell element kernels (Q4 membrane + ACM bending), analytic helpers.
nonlinear_buckling
    :class:`PlateModel` -- von Karman large-deflection FE with layered
    elastoplasticity, linearised buckling (LBA) and arc-length continuation.
imperfection
    Initial-deflection fields (including the IACS/Paik welding imperfection
    model ``w0 = w0max sin(m pi x/a) sin(n pi y/b)``) and residual stress.
loads
    IACS UR I ice patch load and lateral water pressure.
uls_check
    Ultimate limit state / IACS UR I utilisation checks and safety margins.
benchmarks
    Reference solutions (Euler/Levy buckling, Faulkner-von Karman effective
    width, DNV-RP-C201 elastic plate buckling) and the verification driver.
"""

from .core import (  # noqa: F401
    BilinearMaterial,
    GeometryError,
    PlateGeometry,
    element_kernels,
    navier_simply_supported_deflection,
    plane_stress_return_map,
)
from .nonlinear_buckling import BucklingResult, PlateModel  # noqa: F401

__all__ = [
    "BilinearMaterial",
    "BucklingResult",
    "GeometryError",
    "PlateGeometry",
    "PlateModel",
    "element_kernels",
    "navier_simply_supported_deflection",
    "plane_stress_return_map",
]