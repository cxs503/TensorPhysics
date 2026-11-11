"""Plate/shell static verification face (campaign node ``p405w1-n05``).

Modules
-------
* :mod:`.navier` -- Navier double-series exact solution of a uniformly loaded
  simply supported rectangular plate (deflection, ``Mx``, ``My``, ``Mxy``,
  ``Qx``, ``Qy``).
* :mod:`.plate_fe` -- conforming ``C^1`` bicubic-Hermite (Bogner-Fox-Schmit)
  Kirchhoff plate element with SSSS support, moment/shear recovery and an
  h-refinement study driver.
* :mod:`.shell_axisym` -- axisymmetric cylindrical shell element (``C^0`` in
  ``u``, ``C^1`` in ``w``) under internal pressure, plus the Donnell
  boundary-layer closed form used as its analytic anchor.  This is the
  membrane-bending coupling case: ``sigma_theta = pR/t`` in the membrane
  region and a clamped-edge bending boundary layer of length
  ``~sqrt(R t)``.
* :mod:`.verification` -- gate evaluation (G1..G4) with machine-checkable
  numbers, negative controls and the report payload.

Entry point for the numbers + figures: ``verify_fem_plate_shell.py`` in the
repository root.
"""

from .navier import (
    NAVIER_ALPHA_SQUARE_PUBLISHED,
    navier_alpha_coefficient,
    navier_plate_center,
    navier_plate_fields,
    series_convergence,
)
from .plate_fe import (
    SSSSPlateFE,
    convergence_study,
    element_fields,
    element_stiffness,
    element_stiffness_kernels,
    field_error,
    fit_order,
    hermite_1d,
    nodal_to_hermite,
)
from .shell_axisym import (
    CylinderPressureFE,
    beta_edge,
    cylinder_edge_solution,
    edge_decay_fit,
    fit_edge_decay,
    membrane_solution,
    shell_convergence_study,
)
from .verification import (
    PLATE_CASE,
    PLATE_MESHES,
    SHELL_CASE,
    SHELL_MESHES,
    gate_rows,
    plate_gates,
    run_verification,
    shell_gates,
)

__all__ = [
    "NAVIER_ALPHA_SQUARE_PUBLISHED",
    "navier_plate_fields", "navier_plate_center", "navier_alpha_coefficient",
    "series_convergence",
    "SSSSPlateFE", "convergence_study", "element_fields", "element_stiffness",
    "element_stiffness_kernels", "field_error", "fit_order", "hermite_1d",
    "nodal_to_hermite",
    "CylinderPressureFE", "beta_edge", "cylinder_edge_solution",
    "edge_decay_fit", "fit_edge_decay", "membrane_solution",
    "shell_convergence_study",
    "PLATE_CASE", "PLATE_MESHES", "SHELL_CASE", "SHELL_MESHES", "gate_rows",
    "plate_gates", "run_verification", "shell_gates",
]