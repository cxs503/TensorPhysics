"""MPI-style domain decomposition for parallel finite-element assembly and PCG.

This module fills the ``GAP-FEM-PARALLEL`` gap: a complete, measurable parallel
FEM stack for the three verification cases of the campaign node
``p284w3-n04-fem-mpi-parallel`` (Poisson, variable-conductivity diffusion,
plane-strain/3-D elasticity).

Layers
------
``comm``
    MPI-like transport (:class:`SerialComm`, :class:`SharedMemoryComm`,
    :class:`MPIComm`) with collective timing counters.
``runtime``
    Rank launcher (``mpirun -np N`` equivalent for the shared-memory backend).
``partition``
    Cartesian blocking for structured grids, RCB/METIS graph partitioning for
    unstructured meshes, edge-cut and balance metrics.
``discretisation``
    Q1 element kernels (Poisson / heat / elasticity) integrated with Gauss
    quadrature on the reference cell.
``operator``
    Owned-row assembly on the halo box, halo plan and exchange,
    :class:`DistributedSystem` (matvec / dot / Dirichlet elimination).
``pcg``
    CG, Jacobi and additive-Schwarz (block solve) preconditioners with
    communication-aware timing.
``problems``
    Manufactured-solution verification problems.
``reference``
    Slow independent reference assembler and solution serialisation helpers.
``scaling`` / ``verification``
    Strong/weak scaling, communication-overhead and criteria drivers.
"""

from .comm import MPIComm, SerialComm, SharedMemoryComm, make_timer_stats  # noqa: F401
from .discretisation import ElementKernel, StructuredGrid  # noqa: F401
from .operator import (  # noqa: F401
    DistributedSystem,
    HaloPlan,
    assemble_owned_rows,
    build_halo_plan,
    eliminate_dirichlet,
    element_box,
)
from .partition import (  # noqa: F401
    CartesianPartition,
    balance_ratio,
    choose_factorisation,
    edge_cut,
    metis_available,
    partition_graph,
    recursive_bisection,
    refine_partition,
)
from .pcg import ASMPreconditioner, JacobiPreconditioner, SolveResult, cg  # noqa: F401
from .problems import PROBLEMS, Problem, make_problem  # noqa: F401
from .runtime import available_backends, mpi_available, run  # noqa: F401
from .scaling import grid_for_dofs, iteration_study, run_point, strong_scaling, weak_scaling  # noqa: F401

__all__ = [
    "SerialComm",
    "SharedMemoryComm",
    "MPIComm",
    "make_timer_stats",
    "run",
    "available_backends",
    "mpi_available",
    "StructuredGrid",
    "ElementKernel",
    "CartesianPartition",
    "choose_factorisation",
    "partition_graph",
    "recursive_bisection",
    "refine_partition",
    "edge_cut",
    "balance_ratio",
    "metis_available",
    "DistributedSystem",
    "HaloPlan",
    "build_halo_plan",
    "assemble_owned_rows",
    "element_box",
    "eliminate_dirichlet",
    "cg",
    "SolveResult",
    "JacobiPreconditioner",
    "ASMPreconditioner",
    "make_problem",
    "Problem",
    "PROBLEMS",
    "grid_for_dofs",
    "run_point",
    "strong_scaling",
    "weak_scaling",
    "iteration_study",
]
