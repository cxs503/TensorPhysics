"""LBM <-> BEM cross-validation framework for TensorPhysics.

What this package is
--------------------
A *real* two-solver cross-validation harness for bluff-body hydrodynamics:

* :mod:`~tensorphysics.lbm_bem_cross.lbm` — D2Q9 lattice-Boltzmann solver
  (regularised BGK collision, half-way bounce-back, momentum-exchange forces,
  optional Smagorinsky sub-grid model) for the Navier-Stokes side;
* :mod:`~tensorphysics.lbm_bem_cross.bem` — constant-strength source panel
  method (Hess-Smith) for the potential-flow side, plus a field evaluator used
  for the far-field comparison;
* :mod:`~tensorphysics.lbm_bem_cross.boundary_layer` — Thwaites integral
  boundary layer + displacement-thickness viscous-inviscid interaction + wake
  pressure closure, i.e. the "BEM + boundary-layer correction" side;
* :mod:`~tensorphysics.lbm_bem_cross.geometry` — the dual-format (NURBS/STL)
  geometry contract that both solvers consume bit-identically;
* :mod:`~tensorphysics.lbm_bem_cross.crossval` — shared scenarios, common
  output sampling, cross-solver metrics and machine-checkable criteria;
* :mod:`~tensorphysics.lbm_bem_cross.report` — diff-report JSON and PNG figures.

Honesty note
------------
Unlike the earlier surrogate-based ``cross_solver_adjudication`` fixtures, **no
solver here returns a fabricated field**: the LBM integrates the LBE, the BEM
solves a linear panel system, and every reported number is computed from those
fields.  Where the potential + boundary-layer route cannot reach the accuracy
target (it cannot, for a bluff body at ``Re <= 1e4`` — see
``docs/LBM_BEM_CROSS_VALIDATION.md``), the criterion is reported as FAIL rather
than tuned away.
"""

from .boundary_layer import (
    BoundaryLayerSolution,
    VPIResult,
    drag_band,
    thwaites_flat_plate_check,
    thwaites_march,
    viscous_inviscid_iteration,
)
from .bem import (
    PanelMethod2D,
    PanelSolution,
    analytic_cylinder_cp,
    analytic_cylinder_velocity,
    surface_from_polyline,
)
from .crossval import (
    CD_TOL,
    CP_FRONT_DEV_TOL,
    FARFIELD_TOL,
    SCENARIOS,
    CrossValidation,
    Scenario,
    bilinear_sample,
    evaluate_criteria,
    far_field_metrics,
    polar_samples,
    run_case,
    shipped_geometry,
    summarize_case,
)
from .geometry import (
    SurfaceGeometry,
    geometry_contract_check,
    load_geometry,
    nurbs_circle,
    polyline_from_stl,
    polyline_to_stl,
)
from .lbm import CylinderLBM, LBMConfig, lbm_reference_drag

__all__ = [
    "BoundaryLayerSolution", "VPIResult", "drag_band", "thwaites_flat_plate_check",
    "thwaites_march", "viscous_inviscid_iteration",
    "PanelMethod2D", "PanelSolution", "analytic_cylinder_cp",
    "analytic_cylinder_velocity", "surface_from_polyline",
    "CD_TOL", "CP_FRONT_DEV_TOL", "FARFIELD_TOL", "SCENARIOS", "CrossValidation",
    "Scenario", "bilinear_sample", "evaluate_criteria", "far_field_metrics",
    "polar_samples", "run_case", "shipped_geometry", "summarize_case",
    "SurfaceGeometry", "geometry_contract_check", "load_geometry", "nurbs_circle",
    "polyline_from_stl", "polyline_to_stl",
    "CylinderLBM", "LBMConfig", "lbm_reference_drag",
]

__version__ = "1.0.0"