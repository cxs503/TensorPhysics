"""Geometric multigrid solver and two-level-analysis verification for the 2-D
Poisson model problem (TensorPhysics physics-domain module).

Modules
-------
core
    Grid operators, full-weighting restriction, bilinear prolongation, damped
    Jacobi and red-black Gauss-Seidel smoothers, recursive V-/W-cycles, an
    explicit cycle-operator matrix (for spectral radii) and a Jacobi baseline.
tla
    Two-level analysis (local Fourier analysis): 4x4 symbol matrix of the
    two-grid operator, closed-form characteristic quartic, and an independent
    reference built from an explicitly assembled periodic two-grid operator.
benchmarks
    Two-grid operator spectral radius on the Dirichlet grid, Jacobi rate
    ``cos(pi h)``, low-frequency stagnation probe, complexity/speed-up study
    and the smoother parameter sweep.
verification
    Acceptance criteria C1--C8 of the campaign node, with JSON output.
"""

from .core import (  # noqa: F401
    MultigridSolver,
    analytic_solution,
    convergence_history,
    dense_direct_solve,
    discrete_operator,
    jacobi_solver,
    jacobi_sweeps,
    poisson_rhs,
    prolong_bilinear,
    red_black_gs_sweeps,
    residual,
    restrict_full_weighting,
)
from .tla import (  # noqa: F401
    periodic_lfa_max,
    periodic_two_grid_spectral_radius,
    smoothing_factor,
    tla_characteristic_coefficients,
    tla_closed_form_radius,
    tla_spectral_radius,
    tla_symbol_matrix,
    tla_two_grid_factor,
)
from .verification import run_verification  # noqa: F401

__all__ = [
    "MultigridSolver",
    "analytic_solution",
    "convergence_history",
    "dense_direct_solve",
    "discrete_operator",
    "jacobi_solver",
    "jacobi_sweeps",
    "poisson_rhs",
    "prolong_bilinear",
    "red_black_gs_sweeps",
    "residual",
    "restrict_full_weighting",
    "periodic_lfa_max",
    "periodic_two_grid_spectral_radius",
    "smoothing_factor",
    "tla_characteristic_coefficients",
    "tla_closed_form_radius",
    "tla_spectral_radius",
    "tla_symbol_matrix",
    "tla_two_grid_factor",
    "run_verification",
]