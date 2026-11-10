"""Topology-optimization verification surfaces.

Currently hosts the SIMP (Solid Isotropic Material with Penalisation)
density-filter / sensitivity-consistency verification suite in
:mod:`~tensorphysics.topology_optimization.simp`.
"""

from .simp import (  # noqa: F401
    DensityFilter,
    Q4Mesh,
    adjoint_sensitivity,
    build_cantilever,
    build_mbb_beam,
    checkerboard_index,
    complex_step_sensitivity,
    compliance_and_displacements,
    fd_sensitivity,
    filter_sensitivity,
    run_simp,
    sensitivity_error_stats,
    simp_design,
    step_convergence_study,
)

__all__ = [
    "DensityFilter",
    "Q4Mesh",
    "adjoint_sensitivity",
    "build_cantilever",
    "build_mbb_beam",
    "checkerboard_index",
    "complex_step_sensitivity",
    "compliance_and_displacements",
    "fd_sensitivity",
    "filter_sensitivity",
    "run_simp",
    "sensitivity_error_stats",
    "simp_design",
    "step_convergence_study",
]