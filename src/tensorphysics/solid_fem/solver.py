"""Assembly and solution drivers for the 3-D solid element kernel.

Small-strain, geometrically linear.  Two drivers:

* :func:`solve_elastic` -- one sparse solve for a linear elastic material;
* :func:`solve_plastic` -- incremental load stepping with Newton-Raphson
  iterations for J2 plasticity (state stored per Gauss point, consistent
  tangent used so the iterations converge quadratically).

Both support the volumetric `formulations` of :mod:`tensorphysics.solid_fem.element`
(``full`` / ``bbar`` / ``sri``) and Hex8 hourglass stabilisation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve

from .element import element_kinematics, element_stiffness, hourglass_energy, hourglass_stiffness
from .material import J2Material, PlasticState
from .mesh import SolidMesh
from .shape import default_rule

__all__ = [
    "solve_dirichlet",
    "assemble_stiffness",
    "solve_elastic",
    "ElasticSolution",
    "NonlinearSolution",
    "solve_plastic",
    "strain_energy",
    "hourglass_energy_ratio",
    "gauss_stresses",
    "nodal_average_stress",
    "residual_norm",
]


def solve_dirichlet(
    K: csr_matrix,
    f: np.ndarray,
    fixed_dofs: np.ndarray,
    fixed_values: np.ndarray | float = 0.0,
) -> np.ndarray:
    """Solve ``K u = f`` with Dirichlet conditions on ``fixed_dofs``."""
    n = K.shape[0]
    fixed = np.asarray(fixed_dofs, dtype=int)
    vals = np.broadcast_to(np.asarray(fixed_values, dtype=float), fixed.shape).astype(float)
    free = np.setdiff1d(np.arange(n), fixed)
    u = np.zeros(n)
    u[fixed] = vals
    if free.size == 0:
        return u
    Kff = K[free][:, free].tocsc()
    rhs = f[free] - K[free][:, fixed] @ vals
    u[free] = spsolve(Kff, rhs)
    return u


def assemble_stiffness(
    mesh: SolidMesh,
    C: np.ndarray,
    rule: str | None = None,
    formulation: str = "full",
    hourglass: float = 0.0,
    shear_modulus: float | None = None,
) -> csr_matrix:
    """Assemble the global stiffness matrix."""
    K = lil_matrix((mesh.n_dof, mesh.n_dof))
    for e in range(mesh.n_elem):
        ke = element_stiffness(
            mesh.element,
            mesh.element_coords(e),
            C,
            rule=rule,
            formulation=formulation,
            hourglass=hourglass,
            shear_modulus=shear_modulus,
        )
        dofs = mesh.element_dofs(e)
        K[np.ix_(dofs, dofs)] += ke
    return K.tocsr()


@dataclass
class ElasticSolution:
    u: np.ndarray
    K: csr_matrix
    reactions: np.ndarray
    energy: float


def solve_elastic(
    mesh: SolidMesh,
    C: np.ndarray,
    fixed_dofs: np.ndarray,
    loads: np.ndarray | None = None,
    fixed_values: np.ndarray | float = 0.0,
    rule: str | None = None,
    formulation: str = "full",
    hourglass: float = 0.0,
    shear_modulus: float | None = None,
) -> ElasticSolution:
    """Linear elastic solve."""
    K = assemble_stiffness(
        mesh, C, rule=rule, formulation=formulation, hourglass=hourglass,
        shear_modulus=shear_modulus,
    )
    loads = np.zeros(mesh.n_dof) if loads is None else np.asarray(loads, dtype=float)
    u = solve_dirichlet(K, loads, fixed_dofs, fixed_values)
    reactions = K @ u - loads
    energy = 0.5 * float(u @ (K @ u))
    return ElasticSolution(u=u, K=K, reactions=reactions, energy=energy)


def strain_energy(
    mesh: SolidMesh,
    u: np.ndarray,
    C: np.ndarray,
    rule: str | None = None,
    formulation: str = "full",
    hourglass: float = 0.0,
    shear_modulus: float | None = None,
) -> float:
    """Total (stabilised) strain energy of the mesh for displacement ``u``."""
    total = 0.0
    for e in range(mesh.n_elem):
        coords = mesh.element_coords(e)
        u_e = u[mesh.element_dofs(e)]
        ke = element_stiffness(
            mesh.element, coords, C, rule=rule, formulation=formulation,
            hourglass=hourglass, shear_modulus=shear_modulus,
        )
        total += 0.5 * float(u_e @ ke @ u_e)
    return total


def hourglass_energy_ratio(
    mesh: SolidMesh,
    u: np.ndarray,
    C: np.ndarray,
    hourglass: float,
    shear_modulus: float,
    rule: str | None = None,
    formulation: str = "full",
) -> tuple[float, float]:
    """``(hourglass energy, total strain energy)`` for a Hex8 1-point mesh."""
    if mesh.element != "hex8":
        raise ValueError("hourglass control is defined for hex8 meshes")
    e_hg = 0.0
    for e in range(mesh.n_elem):
        e_hg += hourglass_energy(
            mesh.element_coords(e), u[mesh.element_dofs(e)], hourglass, shear_modulus
        )
    total = strain_energy(
        mesh, u, C, rule=rule, formulation=formulation, hourglass=hourglass,
        shear_modulus=shear_modulus,
    )
    return e_hg, total


def gauss_stresses_plastic(
    mesh: SolidMesh,
    u: np.ndarray,
    material: J2Material,
    states: list[list[PlasticState]],
    rule: str | None = None,
    formulation: str = "full",
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per element: ``(gauss point coordinates, stress)`` from the converged states."""
    from .shape import shape as _shape

    out = []
    for e in range(mesh.n_elem):
        coords = mesh.element_coords(e)
        kin = element_kinematics(mesh.element, coords, rule, formulation)
        u_e = u[mesh.element_dofs(e)]
        pts, sig = [], []
        for g in range(kin.integration_points):
            strain = kin.B[g] @ u_e
            stress = material.elastic_matrix @ (strain - states[e][g].eps_p)
            sig.append(stress)
            N, _dN = _shape(mesh.element, *kin.points[g])
            pts.append(N @ coords)
        out.append((np.array(pts), np.array(sig)))
    return out


def gauss_stresses(
    mesh: SolidMesh,
    u: np.ndarray,
    C: np.ndarray,
    rule: str | None = None,
    formulation: str = "full",
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per element: ``(gauss point world coordinates, stress)`` (linear elastic)."""
    out = []
    for e in range(mesh.n_elem):
        coords = mesh.element_coords(e)
        kin = element_kinematics(mesh.element, coords, rule, formulation)
        u_e = u[mesh.element_dofs(e)]
        pts = []
        sig = []
        for g in range(kin.integration_points):
            strain = kin.B[g] @ u_e
            sig.append(C @ strain)
            from .shape import shape as _shape

            N, _dN = _shape(mesh.element, *kin.points[g])
            pts.append(N @ coords)
        out.append((np.array(pts), np.array(sig)))
    return out


def nodal_average_stress(
    mesh: SolidMesh,
    u: np.ndarray,
    C: np.ndarray,
    rule: str | None = None,
    formulation: str = "full",
) -> np.ndarray:
    """Nodal stress by unweighted averaging of the Gauss-point stresses.

    For B-bar / reduced formulations all Gauss points of an element share the
    same (mean) volumetic strain, so element-constant extrapolation is the
    consistent choice.
    """
    acc = np.zeros((mesh.n_nodes, 6))
    cnt = np.zeros(mesh.n_nodes)
    for e in range(mesh.n_elem):
        coords = mesh.element_coords(e)
        kin = element_kinematics(mesh.element, coords, rule, formulation)
        u_e = u[mesh.element_dofs(e)]
        stresses = np.array([C @ (kin.B[g] @ u_e) for g in range(kin.integration_points)])
        s_mean = stresses.mean(axis=0)
        for a in mesh.connectivity[e]:
            acc[a] += s_mean
            cnt[a] += 1.0
    return acc / cnt[:, None]


def residual_norm(r: np.ndarray, f_ref: float) -> float:
    return float(np.linalg.norm(r)) / max(float(f_ref), 1e-30)


@dataclass
class NonlinearSolution:
    u: np.ndarray
    states: list[list[PlasticState]]
    history: list[dict] = field(default_factory=list)
    converged: bool = True


def solve_plastic(
    mesh: SolidMesh,
    material: J2Material,
    fixed_dofs: np.ndarray,
    load_vector: np.ndarray,
    fixed_values: np.ndarray | float = 0.0,
    n_steps: int = 20,
    load_factors: np.ndarray | None = None,
    rule: str | None = None,
    formulation: str = "full",
    hourglass: float = 0.0,
    max_iter: int = 30,
    tol: float = 1e-10,
    tangent_fn: Callable[[J2Material, np.ndarray, PlasticState], tuple] | None = None,
    u0: np.ndarray | None = None,
    states0: list[list[PlasticState]] | None = None,
    reaction_dofs: np.ndarray | None = None,
) -> NonlinearSolution:
    """Incremental Newton-Raphson solution for J2 plasticity.

    Total strain is computed from the current displacement field; the
    constitutive state is always integrated from the last *converged* state,
    which makes the algorithm independent of the iteration path.
    """
    key = mesh.element
    if formulation == "bbar_stab":
        raise NotImplementedError(
            "the bbar_stab stabilisation is implemented for linear analysis only"
        )
    rule = rule or default_rule(key)
    fixed_dofs = np.asarray(fixed_dofs, dtype=int)
    fixed_vals = np.broadcast_to(
        np.asarray(fixed_values, dtype=float), fixed_dofs.shape
    ).astype(float)
    free = np.setdiff1d(np.arange(mesh.n_dof), fixed_dofs)

    # geometry-only quantities (small strain: B matrices are constant)
    kins = []
    for e in range(mesh.n_elem):
        coords = mesh.element_coords(e)
        kin = element_kinematics(key, coords, rule, formulation)
        hg = (
            hourglass_stiffness(coords, hourglass, material.mu)
            if (hourglass and key == "hex8" and rule == "1pt")
            else None
        )
        kins.append((kin, kin.weights * kin.det_j, mesh.element_dofs(e), hg))

    if states0 is None:
        states = [[PlasticState() for _ in range(k[0].integration_points)] for k in kins]
    else:
        states = [[st.copy() for st in el] for el in states0]
    u = np.zeros(mesh.n_dof) if u0 is None else np.array(u0, dtype=float)

    factors = (
        np.asarray(load_factors, dtype=float)
        if load_factors is not None
        else np.linspace(1.0 / n_steps, 1.0, n_steps)
    )

    converge = tangent_fn or (lambda m, eps, st: m.return_map(eps, st))
    history: list[dict] = []
    converged = True

    for lam in factors:
        f_ext = lam * load_vector
        u[fixed_dofs] = lam * fixed_vals
        f_ref = max(np.linalg.norm(f_ext[free]), 1e-30)
        f_ref_fixed = False
        step_converged = False
        trace: list[float] = []
        for it in range(max_iter):
            f_int = np.zeros(mesh.n_dof)
            K = lil_matrix((mesh.n_dof, mesh.n_dof))
            trial_states = []
            for e, (kin, dv, dofs, hg) in enumerate(kins):
                u_e = u[dofs]
                ke = np.zeros((kin.ndof, kin.ndof))
                fe = np.zeros(kin.ndof)
                e_states = []
                for g in range(kin.integration_points):
                    strain = kin.B[g] @ u_e
                    try:
                        sig, new_state, C_ep, _plastic = converge(
                            material, strain, states[e][g]
                        )
                    except ValueError:
                        return NonlinearSolution(
                            u=u, states=states, history=history, converged=False
                        )
                    e_states.append(new_state)
                    fe += dv[g] * (kin.B[g].T @ sig)
                    ke += dv[g] * (kin.B[g].T @ C_ep @ kin.B[g])
                if hg is not None:
                    ke += hg
                    fe += hg @ u_e
                f_int[dofs] += fe
                K[np.ix_(dofs, dofs)] += ke
                trial_states.append(e_states)

            r = f_ext - f_int
            if not f_ref_fixed:
                # displacement-controlled steps have no external load, so the
                # first internal force defines the residual reference scale
                f_ref = max(f_ref, float(np.linalg.norm(f_int[free])))
                f_ref_fixed = True
            r_free = r[free]
            rn = residual_norm(r_free, f_ref)
            trace.append(float(rn))
            if rn <= tol:
                states = trial_states
                step_converged = True
                history.append(
                    {
                        "load_factor": float(lam),
                        "iterations": it,
                        "residual": rn,
                        "residual_trace": trace,
                        # net reaction on the constrained dofs (displacement
                        # controlled steps use this as the "load")
                        "reaction_sum": (
                            None
                            if reaction_dofs is None
                            else float(np.sum(np.abs(
                                f_int[np.asarray(reaction_dofs, dtype=int)]
                                - f_ext[np.asarray(reaction_dofs, dtype=int)]
                            )))
                        ),
                    }
                )
                break
            if it == max_iter - 1:
                break
            Kcsr = K.tocsr()
            du = solve_dirichlet(Kcsr, r, fixed_dofs, 0.0)
            # simple backtracking to keep the Newton step physical
            u = u + du
        if not step_converged:
            converged = False
            history.append(
                {"load_factor": float(lam), "iterations": max_iter, "residual": None,
                 "note": "no convergence"}
            )
            break

    return NonlinearSolution(u=u, states=states, history=history, converged=converged)