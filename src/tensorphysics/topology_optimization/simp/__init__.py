"""SIMP topology-optimisation verification package.

Physical model
--------------
Density-based topology optimisation interpolates the Young's modulus between a
void value ``Emin`` and a solid value ``E0``::

    E(rho) = Emin + rho**p * (E0 - Emin),      rho in [0, 1],  p = 3

and minimises the compliance ``C = F^T u`` for a prescribed volume fraction.
Without regularisation the discretised problem is ill-posed: the finite-element
mesh admits "checkerboard" (element-wise alternating density) modes and
one-node-connected hinges whose stiffness is a pure discretisation artefact.
Sigmund's linear density filter removes that length-scale pathology.

The two acceptance surfaces implemented here are

``fem.simp.density_filter``
    the filter removes the checkerboard pathology while keeping the compliant
    mechanism's load path (see :mod:`.density_filter`, :mod:`.problems`);
``fem.simp.sensitivity_consistency``
    the adjoint sensitivity ``dC/drho`` agrees with finite differences to
    ``O(drho)`` (see :mod:`.sensitivity`).
"""

from .density_filter import (
    DensityFilter,
    checkerboard_index,
    filter_sensitivity,
)
from .filter_acceptance import (
    convergence_order,
    jaccard_similarity,
    run_acceptance as run_filter_acceptance,
    sample_to_grid,
)
from .filter_chain import (
    FilterChain,
    HeavisideProjection,
    IdentityFilter,
    PdeFilter,
    adjoint_sensitivity_chain,
    fd_sensitivity_chain,
    make_filter_chain,
    pde_radius_from_hat_radius,
    run_simp_chain,
)
from .fem2d import Q4Mesh, assemble_stiffness, compliance_and_displacements
from .problems import (
    build_cantilever,
    build_mbb_beam,
    run_simp,
    simp_design,
)
from .sensitivity import (
    adjoint_sensitivity,
    complex_step_sensitivity,
    fd_sensitivity,
    sensitivity_error_stats,
    step_convergence_study,
)
from .verification import count_test_cases, run_acceptance

__all__ = [
    "DensityFilter",
    "FilterChain",
    "HeavisideProjection",
    "IdentityFilter",
    "PdeFilter",
    "adjoint_sensitivity_chain",
    "convergence_order",
    "fd_sensitivity_chain",
    "jaccard_similarity",
    "make_filter_chain",
    "pde_radius_from_hat_radius",
    "run_filter_acceptance",
    "run_simp_chain",
    "sample_to_grid",
    "checkerboard_index",
    "filter_sensitivity",
    "Q4Mesh",
    "assemble_stiffness",
    "compliance_and_displacements",
    "build_cantilever",
    "build_mbb_beam",
    "run_simp",
    "simp_design",
    "adjoint_sensitivity",
    "complex_step_sensitivity",
    "fd_sensitivity",
    "sensitivity_error_stats",
    "step_convergence_study",
    "count_test_cases",
    "run_acceptance",
]