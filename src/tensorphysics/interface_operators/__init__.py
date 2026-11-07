"""Cross-solver interface operators with conservation / order / symmetry gates.

Public API
----------
Core contract:
    :class:`~tensorphysics.interface_operators.core.InterfaceOperator`
        every exchange operator exposes ``conservation_residual``,
        ``observed_order`` and ``symmetry_flip``.
    :class:`~tensorphysics.interface_operators.core.OperatorReport`
        per-(operator, field) result row.

Operators:
    :class:`~tensorphysics.interface_operators.operators.MortarProjection`
    :class:`~tensorphysics.interface_operators.operators.ConservativeScatter`
    :class:`~tensorphysics.interface_operators.operators.LBMDEMMomentumExchange`

Acceptance (campaign node IN19-T2, gates C1..C6):
    :func:`~tensorphysics.interface_operators.acceptance.run_acceptance`
        returns the full JSON payload with every criterion number;
    :func:`...acceptance.fvm_lbm_chain`, ``...lbm_dem_chain``
        the two two-solver chains.
"""
from .core import (ANALYTIC_FIELDS, AnalyticField, InterfaceOperator,
                   OperatorReport, OrderEstimate, PointSet, SegmentMesh,
                   SymmetryFlip, conservation_residual, exact_values,
                   graded_mesh, uniform_mesh)
from .operators import (ConservativeScatter, LBMDEMMomentumExchange,
                        MortarProjection)
from .acceptance import (P_MIN, RESOLUTIONS, TOL_C1, TOL_C3, TOL_C4, TOL_C5,
                         fvm_lbm_chain, lbm_dem_chain, negative_control_suite,
                         operator_suite, run_acceptance)
from .mortar_nd import (FIELD_2D, FIELD_3D, MortarProjection2D,
                        SurfaceMortarProjection, analytic_surface_area,
                        bulged_surface, faceted_surface_area,
                        local_conservation_fix, overlap_measure_closure)
from .polygons import (PolygonMesh, SurfacePatchMesh, axisymmetric_mesh,
                       cartesian_mesh, clip_convex, convex_hull,
                       convex_intersection, fan_mesh, polygon_area,
                       polygon_centroid, triangulated_mesh)

__all__ = [
    "InterfaceOperator", "OperatorReport", "OrderEstimate", "SymmetryFlip",
    "SegmentMesh", "PointSet", "AnalyticField", "ANALYTIC_FIELDS",
    "uniform_mesh", "graded_mesh", "exact_values", "conservation_residual",
    "MortarProjection", "ConservativeScatter", "LBMDEMMomentumExchange",
    "run_acceptance", "operator_suite", "negative_control_suite",
    "fvm_lbm_chain", "lbm_dem_chain",
    "TOL_C1", "TOL_C3", "TOL_C4", "TOL_C5", "P_MIN", "RESOLUTIONS",
    # IN50-N08: 2D polygon / 3D surface mortar + local conservation fix
    "MortarProjection2D", "SurfaceMortarProjection", "FIELD_2D", "FIELD_3D",
    "PolygonMesh", "SurfacePatchMesh", "cartesian_mesh", "triangulated_mesh",
    "fan_mesh", "axisymmetric_mesh", "clip_convex", "convex_hull",
    "convex_intersection", "polygon_area", "polygon_centroid",
    "local_conservation_fix", "overlap_measure_closure", "bulged_surface",
    "analytic_surface_area", "faceted_surface_area",
]