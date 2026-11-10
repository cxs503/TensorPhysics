"""Benchmark problems and the optimality-criteria (OC) design update.

Two classical SIMP benchmarks are provided:

``MBB beam`` (half model)
    length : height = 2 : 1, ``nelx x nely`` element grid, symmetry rollers on
    the left edge (``u_x = 0``), vertical support at the bottom-right corner,
    unit downward load at the top-left corner.  Volume fraction 0.5.

``Cantilever``
    left edge fully clamped, unit downward load at the mid-height right-edge
    node.  Volume fraction 0.4.

The optimiser is the textbook optimality-criteria update with a move limit of
0.2; the volume constraint is imposed on the *physical* (filtered) density,
which requires the filtered volume sensitivity ``P^T 1 / n``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .density_filter import DensityFilter, checkerboard_index
from .fem2d import Q4Mesh, assemble_stiffness, compliance_and_displacements, solve_dirichlet
from .sensitivity import adjoint_sensitivity

__all__ = [
    "SIMPProblem",
    "build_cantilever",
    "build_mbb_beam",
    "evaluate_binary_design",
    "oc_update",
    "run_simp",
    "simp_design",
]


@dataclass
class SIMPProblem:
    """Mesh + boundary conditions + load vector for a 2-D compliance problem."""

    name: str
    mesh: Q4Mesh
    F: np.ndarray
    fixed_dofs: tuple[int, ...]
    volfrac: float

    @property
    def n_elems(self) -> int:
        return self.mesh.n_elems


def build_mbb_beam(nelx: int = 100, nely: int = 50, volfrac: float = 0.5) -> SIMPProblem:
    """Half MBB beam (see module docstring)."""
    mesh = Q4Mesh(nelx, nely)
    F = np.zeros(mesh.n_dof)
    F[mesh.node_dofs(0, 0)[1]] = -1.0  # downward load, top-left corner
    fixed = [mesh.node_dofs(0, iy)[0] for iy in range(nely + 1)]
    fixed.append(mesh.node_dofs(nelx, 0)[1])  # bottom-right corner, vertical support
    return SIMPProblem("mbb_beam", mesh, F, tuple(fixed), float(volfrac))


def build_cantilever(nelx: int = 50, nely: int = 25, volfrac: float = 0.4) -> SIMPProblem:
    """Cantilever: clamped left edge, tip load at mid-height of the right edge."""
    mesh = Q4Mesh(nelx, nely)
    F = np.zeros(mesh.n_dof)
    F[mesh.node_dofs(nelx, nely // 2)[1]] = -1.0
    fixed: list[int] = []
    for iy in range(nely + 1):
        ux, uy = mesh.node_dofs(0, iy)
        fixed.extend([ux, uy])
    return SIMPProblem("cantilever", mesh, F, tuple(fixed), float(volfrac))


def evaluate_binary_design(
    problem: SIMPProblem,
    rho: np.ndarray,
    *,
    level: float = 0.5,
    E0: float = 1.0,
    Emin: float = 1e-6,
) -> tuple[float | None, np.ndarray]:
    """Compliance of the black-and-white interpretation of a design.

    Grey elements are projected to solid/void at ``level`` and the structure is
    re-solved with the same FE mesh.  This is the *physical* stiffness of the
    design; one-node-connected hinges and checkerboards that look stiff under
    the SIMP interpolation disintegrate (or become extremely flexible) here.
    ``Emin`` is kept at ``1e-6*E0`` so the reduced system stays numerically
    solvable; islands that are truly disconnected still return ``None``.
    """
    rho = np.asarray(rho, dtype=float).ravel()
    solid = (rho >= level).astype(float)
    # Emin keeps void stiffness positive-definite; `solid` uses E0.
    E_elem = np.where(solid > 0.5, E0, Emin)
    K = assemble_stiffness(problem.mesh, E_elem)
    u = solve_dirichlet(K, problem.F, problem.fixed_dofs)
    if u is None:
        return None, solid
    return float(problem.F @ u), solid


def oc_update(
    rho: np.ndarray,
    dc: np.ndarray,
    dv: np.ndarray,
    volfrac: float,
    *,
    move: float = 0.2,
    tol: float = 1e-6,
    max_bisect: int = 200,
) -> np.ndarray:
    """Optimality-criteria design update (bisection on the Lagrange multiplier).

    ``dc`` is ``dC/drho`` (negative) and ``dv`` is the volume-fraction gradient
    evaluated on the *design* variables.
    """
    rho = np.asarray(rho, dtype=float).copy()
    dc = np.asarray(dc, dtype=float)
    dv = np.asarray(dv, dtype=float)
    l1, l2 = 1e-9, 1e9
    xnew = rho.copy()
    for _ in range(max_bisect):
        lmid = 0.5 * (l1 + l2)
        B = np.sqrt(np.maximum(-dc, 0.0) / np.maximum(dv * lmid, 1e-30))
        xnew = np.minimum(1.0, np.maximum(
            0.0, np.maximum(rho - move, np.minimum(rho + move, rho * B))
        ))
        if float(np.mean(xnew)) > volfrac:
            l1 = lmid
        else:
            l2 = lmid
        if abs(l2 - l1) < tol * (l1 + l2):
            break
    return xnew


def run_simp(
    problem: SIMPProblem,
    *,
    radius: float | None = None,
    penal: float = 3.0,
    iters: int = 100,
    move: float = 0.2,
    tol: float = 1e-2,
    E0: float = 1.0,
    Emin: float = 1e-9,
    rho0: np.ndarray | None = None,
) -> dict:
    """Run SIMP compliance minimisation and return the design + history.

    ``radius=None`` disables regularisation (the pathological baseline);
    ``radius=r`` enables the linear density filter with radius ``r`` elements.
    """
    mesh = problem.mesh
    n = mesh.n_elems
    dens = DensityFilter(mesh.nelx, mesh.nely, radius) if radius is not None else None
    ones = np.ones(n)

    rho = np.full(n, problem.volfrac) if rho0 is None else np.asarray(rho0, float).ravel().copy()
    dv = (dens.matrix.T @ ones) / n if dens is not None else ones / n

    history: list[dict[str, float]] = []
    converged = False
    C: float | None = None
    for it in range(1, iters + 1):
        C, _, dc = adjoint_sensitivity(
            mesh, rho, problem.F, problem.fixed_dofs,
            penal=penal, E0=E0, Emin=Emin, dens_filter=dens,
        )
        if dc is None:
            raise RuntimeError("stiffness matrix became singular during optimisation")
        rho_new = oc_update(rho, dc, dv, problem.volfrac, move=move)
        change = float(np.max(np.abs(rho_new - rho)))
        rho = rho_new
        rho_p = dens.apply(rho) if dens is not None else rho
        history.append({
            "iteration": float(it),
            "compliance": float(C),
            "change": change,
            "volume_fraction": float(np.mean(rho_p)),
            "checkerboard_index": checkerboard_index(rho_p.reshape(mesh.shape)),
        })
        if change < tol:
            converged = True
            break

    rho_p = dens.apply(rho) if dens is not None else rho
    C_final, _, dc_final = adjoint_sensitivity(
        mesh, rho, problem.F, problem.fixed_dofs,
        penal=penal, E0=E0, Emin=Emin, dens_filter=dens,
    )
    return {
        "problem": problem.name,
        "nelx": mesh.nelx,
        "nely": mesh.nely,
        "volfrac": problem.volfrac,
        "penal": penal,
        "radius": 0.0 if dens is None else float(radius),
        "iterations": len(history),
        "converged": converged,
        "compliance": float(C_final),
        "rho": rho,
        "rho_physical": rho_p,
        "dc": dc_final,
        "history": history,
        "checkerboard_index_raw": checkerboard_index(rho.reshape(mesh.shape)),
        "checkerboard_index": checkerboard_index(rho_p.reshape(mesh.shape)),
        "grey_fraction": float(np.mean((rho_p > 0.1) & (rho_p < 0.9))),
    }

def simp_design(
    name: str = "mbb_beam",
    *,
    nelx: int | None = None,
    nely: int | None = None,
    volfrac: float | None = None,
    **kwargs,
) -> dict:
    """Build one of the named benchmark problems and run :func:`run_simp`."""
    if name == "mbb_beam":
        problem = build_mbb_beam(
            nelx if nelx is not None else 100,
            nely if nely is not None else 50,
            volfrac if volfrac is not None else 0.5,
        )
    elif name == "cantilever":
        problem = build_cantilever(
            nelx if nelx is not None else 50,
            nely if nely is not None else 25,
            volfrac if volfrac is not None else 0.4,
        )
    else:
        raise ValueError(f"unknown SIMP benchmark {name!r}")
    return run_simp(problem, **kwargs)
