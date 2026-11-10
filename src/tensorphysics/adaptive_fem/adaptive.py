"""Adaptive h-refinement driver: SOLVE → ESTIMATE → MARK → REFINE loop."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .elasticity2d import PlaneSolution
from .estimator import ZzEstimate, actual_energy_error, effectivity_index, zz_estimate
from .marking import dorfler_mark
from .mesh2d import TriMesh, mesh_summary
from .problem import PlaneProblem
from .refine import flip_to_quality, refine_longest_edge, refine_red_green

__all__ = ["AdaptiveStep", "AdaptiveRun", "adaptive_refinement", "doubling_refinement"]


@dataclass
class AdaptiveStep:
    level: int
    n_nodes: int
    n_tris: int
    n_dof: int
    n_free_dof: int
    eta_total: float
    min_angle_deg: float
    area: float
    n_marked: int
    seconds: float
    error_energy: float | None = None
    error_rel: float | None = None
    solution_norm: float | None = None
    effectivity: float | None = None

    def as_dict(self) -> dict:
        return {
            "level": self.level,
            "n_nodes": self.n_nodes,
            "n_tris": self.n_tris,
            "n_dof": self.n_dof,
            "n_free_dof": self.n_free_dof,
            "eta_total": self.eta_total,
            "min_angle_deg": self.min_angle_deg,
            "area": self.area,
            "n_marked": self.n_marked,
            "seconds": self.seconds,
            "error_energy": self.error_energy,
            "error_rel": self.error_rel,
            "solution_norm": self.solution_norm,
            "effectivity": self.effectivity,
        }


@dataclass
class AdaptiveRun:
    name: str
    steps: list[AdaptiveStep] = field(default_factory=list)
    meshes: list[TriMesh] = field(default_factory=list)
    final_solution: PlaneSolution | None = None
    final_estimate: ZzEstimate | None = None

    def table(self) -> list[dict]:
        return [s.as_dict() for s in self.steps]

    @property
    def final(self) -> AdaptiveStep:
        return self.steps[-1]


def _free_dof_count(problem: PlaneProblem, mesh: TriMesh) -> int:
    fixed, _ = problem.fixed_dofs(mesh.nodes)
    return int(2 * mesh.n_nodes - np.unique(fixed).size)


def adaptive_refinement(
    problem: PlaneProblem,
    mesh: TriMesh,
    *,
    n_steps: int = 5,
    theta: float = 0.5,
    recovery: str = "spr",
    reference: Callable[[np.ndarray], np.ndarray] | None = None,
    flip: bool = False,
    scheme: str = "leb",
    max_levels: int | None = None,
) -> AdaptiveRun:
    """Run the ZZ-driven adaptive loop for ``n_steps`` refinement cycles.

    ``reference`` maps element centroids to a reference (analytic or
    high-resolution numerical) stress, enabling measurement of the *actual*
    error and hence the effectivity index at every step.

    ``scheme`` selects the refinement: ``"leb"`` (Rivara longest-edge bisection,
    default) or ``"red-green"``.  Red-green is kept for reference/tests but
    drives the minimum angle below the 20 deg acceptance threshold after a few
    levels (its green closure is forced to bisect whatever edge the neighbour
    bisected); longest-edge bisection preserves the angle bound.
    """
    run = AdaptiveRun(name=problem.name)
    cur = mesh
    for level in range(n_steps + 1):
        t0 = time.time()
        sol = problem.solve(cur)
        est = zz_estimate(sol, recovery=recovery)
        summ = mesh_summary(cur.nodes, cur.tris)
        step = AdaptiveStep(
            level=level,
            n_nodes=summ["n_nodes"],
            n_tris=summ["n_tris"],
            n_dof=2 * summ["n_nodes"],
            n_free_dof=_free_dof_count(problem, cur),
            eta_total=est.eta_total,
            min_angle_deg=summ["min_angle_deg"],
            area=summ["area"],
            n_marked=0,
            seconds=time.time() - t0,
        )
        if reference is not None:
            ref_stress = reference(sol.mesh.centroids())
            actual = actual_energy_error(sol, ref_stress)
            sol_norm = float(
                np.sqrt(
                    np.sum(
                        np.einsum(
                            "mi,ij,mj->m",
                            ref_stress,
                            sol.material.compliance,
                            ref_stress,
                        )
                        * sol.area
                        * sol.material.thickness
                    )
                )
            )
            step.error_energy = actual
            step.solution_norm = sol_norm
            step.error_rel = actual / sol_norm if sol_norm > 0 else float("nan")
            step.effectivity = effectivity_index(est.eta_total, actual)
        step.seconds = time.time() - t0
        run.steps.append(step)
        run.meshes.append(cur)
        run.final_solution = sol
        run.final_estimate = est
        if level == n_steps:
            break
        marked = dorfler_mark(est.eta_elem, theta)
        step.n_marked = int(marked.size)
        if marked.size == 0:
            break
        if scheme == "leb":
            cur = refine_longest_edge(cur, marked)
        elif scheme == "red-green":
            cur = refine_red_green(cur, marked)
        else:
            raise ValueError("scheme must be 'leb' or 'red-green'")
        if flip:
            cur = flip_to_quality(cur)
        if max_levels is not None and level >= max_levels:
            break
    return run


def doubling_refinement(
    problem: PlaneProblem,
    mesh: TriMesh,
    *,
    n_steps: int = 3,
    recovery: str = "spr",
    reference: Callable[[np.ndarray], np.ndarray] | None = None,
) -> AdaptiveRun:
    """Uniform (global red) refinement reference curve: every element is marked."""
    run = AdaptiveRun(name=problem.name + ":uniform")
    cur = mesh
    for level in range(n_steps + 1):
        t0 = time.time()
        sol = problem.solve(cur)
        est = zz_estimate(sol, recovery=recovery)
        summ = mesh_summary(cur.nodes, cur.tris)
        step = AdaptiveStep(
            level=level,
            n_nodes=summ["n_nodes"],
            n_tris=summ["n_tris"],
            n_dof=2 * summ["n_nodes"],
            n_free_dof=_free_dof_count(problem, cur),
            eta_total=est.eta_total,
            min_angle_deg=summ["min_angle_deg"],
            area=summ["area"],
            n_marked=summ["n_tris"] if level < n_steps else 0,
            seconds=0.0,
        )
        if reference is not None:
            ref_stress = reference(sol.mesh.centroids())
            actual = actual_energy_error(sol, ref_stress)
            sol_norm = float(
                np.sqrt(
                    np.sum(
                        np.einsum("mi,ij,mj->m", ref_stress, sol.material.compliance, ref_stress)
                        * sol.area
                        * sol.material.thickness
                    )
                )
            )
            step.error_energy = actual
            step.solution_norm = sol_norm
            step.error_rel = actual / sol_norm if sol_norm > 0 else float("nan")
            step.effectivity = effectivity_index(est.eta_total, actual)
        step.seconds = time.time() - t0
        run.steps.append(step)
        run.meshes.append(cur)
        run.final_solution = sol
        run.final_estimate = est
        if level == n_steps:
            break
        cur = refine_red_green(cur, range(cur.n_tris))
    return run