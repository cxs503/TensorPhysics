"""Plane-elasticity boundary-value problems assembled from geometric predicates.

Boundary conditions are expressed as *predicates on coordinates* rather than
element tags: after h-refinement the mesh changes, but the geometric predicates
(``x == 0``, ``y == H``, ``r == a`` …) automatically re-target the new boundary
entities.  That keeps the adaptive driver free of bookkeeping about hanging
nodes or inherited tags.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .elasticity2d import (
    Material2D,
    PlaneSolution,
    assemble_stiffness,
    element_stress,
    solve_dirichlet,
)
from .mesh2d import TriMesh, boundary_edges

__all__ = ["DirichletBC", "TractionBC", "PlaneProblem"]

Predicate = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class DirichletBC:
    """Fix one displacement component on the nodes selected by ``predicate``."""

    component: str  # "x" | "y"
    predicate: Predicate
    value: float = 0.0
    name: str = ""

    def __post_init__(self) -> None:
        if self.component not in ("x", "y"):
            raise ValueError("component must be 'x' or 'y'")


@dataclass(frozen=True)
class TractionBC:
    """Traction applied to every boundary edge whose midpoint matches.

    The predicate receives the two endpoint coordinate arrays ``(p0, p1)`` of
    the boundary edges and returns a mask; using endpoints (exact mesh
    coordinates) instead of midpoints makes the selection robust for graded
    meshes on curved surfaces.

    ``vector`` is either a constant ``(tx, ty)`` or a callable mapping the edge
    midpoints ``(k,2)`` to a traction field ``(k,2)`` (pressure on a curved
    surface).
    """

    predicate: Callable[[np.ndarray, np.ndarray], np.ndarray]
    vector: tuple[float, float] | Callable[[np.ndarray], np.ndarray]
    name: str = ""


@dataclass
class PlaneProblem:
    name: str
    material: Material2D
    dirichlet: tuple[DirichletBC, ...] = ()
    tractions: tuple[TractionBC, ...] = ()
    body_force: tuple[float, float] = (0.0, 0.0)
    description: dict = field(default_factory=dict)

    # -- boundary data ------------------------------------------------------
    def fixed_dofs(self, nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dofs: list[np.ndarray] = []
        vals: list[np.ndarray] = []
        for bc in self.dirichlet:
            comp = 0 if bc.component == "x" else 1
            mask = np.asarray(bc.predicate(nodes), dtype=bool)
            if mask.shape != (len(nodes),):
                raise ValueError(f"predicate of {bc.name!r} must return a boolean mask over nodes")
            ids = np.flatnonzero(mask)
            dofs.append(2 * ids + comp)
            vals.append(np.full(ids.size, float(bc.value)))
        if not dofs:
            return np.zeros(0, dtype=np.int64), np.zeros(0)
        d = np.concatenate(dofs)
        v = np.concatenate(vals)
        uniq, first = np.unique(d, return_index=True)
        if uniq.size != d.size:  # duplicated dof: the prescribed values must agree
            order = np.argsort(d, kind="stable")
            ds, vs = d[order], v[order]
            starts = np.flatnonzero(np.r_[True, ds[1:] != ds[:-1]])
            ends = np.r_[starts[1:], len(ds)]
            for s0, e0 in zip(starts, ends):
                if float(np.ptp(vs[s0:e0])) > 1e-12:
                    raise ValueError(
                        f"conflicting Dirichlet values on dof {int(ds[s0])}"
                    )
        return uniq.astype(np.int64), v[first]

    def external_forces(self, mesh: TriMesh, n_dof: int | None = None) -> np.ndarray:
        n = mesh.n_nodes if n_dof is None else n_dof // 2
        f = np.zeros(2 * n, dtype=float)
        bx, by = self.body_force
        if bx or by:
            area = np.abs(
                0.5
                * (
                    (mesh.nodes[mesh.tris[:, 1], 0] - mesh.nodes[mesh.tris[:, 0], 0])
                    * (mesh.nodes[mesh.tris[:, 2], 1] - mesh.nodes[mesh.tris[:, 0], 1])
                    - (mesh.nodes[mesh.tris[:, 2], 0] - mesh.nodes[mesh.tris[:, 0], 0])
                    * (mesh.nodes[mesh.tris[:, 1], 1] - mesh.nodes[mesh.tris[:, 0], 1])
                )
            )
            contrib = self.material.thickness * area / 3.0
            for j in range(3):
                np.add.at(f, 2 * mesh.tris[:, j], bx * contrib)
                np.add.at(f, 2 * mesh.tris[:, j] + 1, by * contrib)
        if self.tractions:
            be = boundary_edges(mesh.nodes, mesh.tris)
            if be.size:
                p0 = mesh.nodes[be[:, 0]]
                p1 = mesh.nodes[be[:, 1]]
                mid = 0.5 * (p0 + p1)
                length = np.linalg.norm(p1 - p0, axis=1)
                for tr in self.tractions:
                    sel = np.asarray(tr.predicate(p0, p1), dtype=bool)
                    if sel.shape != (len(be),):
                        raise ValueError(
                            f"traction predicate {tr.name!r} must return a mask over boundary edges"
                        )
                    e = be[sel]
                    half = 0.5 * length[sel] * self.material.thickness
                    if callable(tr.vector):
                        tvec = np.asarray(tr.vector(mid[sel]), dtype=float).reshape(-1, 2)
                        tx, ty = tvec[:, 0], tvec[:, 1]
                    else:
                        tx, ty = float(tr.vector[0]), float(tr.vector[1])
                    np.add.at(f, 2 * e[:, 0], half * tx)
                    np.add.at(f, 2 * e[:, 1], half * tx)
                    np.add.at(f, 2 * e[:, 0] + 1, half * ty)
                    np.add.at(f, 2 * e[:, 1] + 1, half * ty)
        return f

    # -- solve --------------------------------------------------------------
    def solve(self, mesh: TriMesh) -> PlaneSolution:
        K, B, area = assemble_stiffness(mesh.nodes, mesh.tris, self.material)
        f = self.external_forces(mesh)
        fixed, vals = self.fixed_dofs(mesh.nodes)
        u = solve_dirichlet(K, f, fixed, vals)
        stress = element_stress(B, u, mesh.tris, self.material)
        return PlaneSolution(mesh=mesh, material=self.material, u=u, stress=stress, area=area, B=B)

    def reaction_equilibrium_error(self, sol: PlaneSolution) -> float:
        """Relative residual ``|K u - f|`` on the *free* dofs (should be round-off).

        The residual on constrained dofs is the reaction force and is not an
        error; only the free block measures solve quality.
        """
        K, _, _ = assemble_stiffness(sol.mesh.nodes, sol.mesh.tris, self.material)
        f = self.external_forces(sol.mesh)
        fixed, _ = self.fixed_dofs(sol.mesh.nodes)
        free = np.setdiff1d(np.arange(2 * sol.mesh.n_nodes), fixed)
        r = K @ sol.u - f
        denom = max(np.linalg.norm(f[free]) if free.size else 1.0, 1e-30)
        return float(np.linalg.norm(r[free]) / denom)