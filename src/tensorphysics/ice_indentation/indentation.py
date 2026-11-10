"""Displacement-controlled indentation of a J2 ice block (FEM + penalty contact).

The driver couples ingredients that already exist elsewhere in the repository
with the two new ones of this package:

* :mod:`tensorphysics.solid_fem.element` -- Hex8 kinematics / B-matrices
  (reused, not reimplemented);
* :mod:`tensorphysics.solid_fem.material` -- J2 radial return mapping and
  consistent tangent (reused through the vectorised twin
  :func:`~tensorphysics.ice_indentation.constitutive.batch_return_map`);
* :mod:`tensorphysics.ice_indentation.contact` -- *new*: node-to-rigid-surface
  penalty contact with an active set updated inside the Newton loop.

Solution algorithm (small strain, incremental displacement control):

1. advance the rigid indenter apex to ``delta_n`` (linear or power ramp);
2. Newton iterations on the free dofs with
   ``R(u) = f_contact(u; delta_n) - f_int(u)`` and the *consistent* tangent
   ``J = df_contact/du - K_T`` (unsymmetric because of the curved-indenter
   radial coupling, solved directly);
3. the active set and the contact tangent are re-evaluated every iteration, so
   the contact area is an outcome of the solution; a hysteretic active set plus
   a residual line search make the penalty iteration robust across contact
   inception;
4. per converged step the energies are integrated: elastic strain energy,
   plastic dissipation ``int sigma_y dalpha``, penalty energy
   ``1/2 k_p g^2``, the work of the contact loads and the work of the indenter,
   which is what AC5 tests.

Two start-up options exist for the resolution-limited early contact phase
(see ``delta_offset`` / ``ramp``): the whole path can be marched from
``delta = 0`` (fully predictive but the first increments have a contact patch
smaller than one element), or the indenter can be *pre-embedded* by
``delta_offset`` so that loading starts where the contact patch already spans
several nodes.  The sensitivity to that choice is measured in the acceptance
study instead of being hidden.

All quantities are returned in SI units (m, N, Pa, J).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix

from ..solid_fem.element import element_kinematics
from ..solid_fem.material import J2Material
from ..solid_fem.shape import shape as _shape_fn
from ..solid_fem.solver import solve_dirichlet

from .constitutive import batch_return_map
from .contact import contact_state
from .contact_facets import build_surface_facets, facet_contact_state
from .mesh import face_nodes, surface_nodal_areas

__all__ = [
    "IndentationResult",
    "simulate_indentation",
    "quarter_block_bcs",
    "penalty_stiffness",
    "surface_penalty_stiffness",
    "top_surface_data",
    "load_ramp",
]


# --------------------------------------------------------------------------- #
# Boundary conditions
# --------------------------------------------------------------------------- #
def quarter_block_bcs(mesh, tol: float = 1e-9) -> np.ndarray:
    """Symmetry, roller and supported-base dofs of a quarter block.

    ``x = 0``: ``u_x = 0``; ``y = 0``: ``u_y = 0``; ``z = 0``: ``u_z = 0``;
    ``x = x_max``: ``u_x = 0``; ``y = y_max``: ``u_y = 0`` (rollers).  The
    quarter model then has no rigid-body modes and the far field is confined,
    like an ice block resting on a rigid base.
    """
    x_max = float(mesh.nodes[:, 0].max())
    y_max = float(mesh.nodes[:, 1].max())
    z_min = float(mesh.nodes[:, 2].min())
    dofs = [
        *mesh.dofs_of_nodes(face_nodes(mesh, 0, 0.0, tol), 0),
        *mesh.dofs_of_nodes(face_nodes(mesh, 1, 0.0, tol), 1),
        *mesh.dofs_of_nodes(face_nodes(mesh, 2, z_min, tol), 2),
        *mesh.dofs_of_nodes(face_nodes(mesh, 0, x_max, tol), 0),
        *mesh.dofs_of_nodes(face_nodes(mesh, 1, y_max, tol), 1),
    ]
    return np.unique(np.asarray(dofs, dtype=int))


def penalty_stiffness(material: J2Material, nodal_area, factor: float = 50.0):
    """Nodal penalty stiffness ``factor * E * sqrt(A_node)`` [N/m].

    The physically meaningful surface stiffness of one node is
    ``E A_node / h`` with the element size ``h ~ sqrt(A_node)``; the penalty
    factor makes the artificial spring ``factor`` times stiffer, so the
    residual penetration stays ``1/factor`` of the physical deformation and the
    stored penalty energy stays a small fraction of the indentation work.

    Accepts a scalar or an array of nodal areas (the graded mesh has strongly
    varying nodal areas, so the array form is the physically meaningful one).
    """
    area = np.asarray(nodal_area, dtype=float)
    if np.any(area <= 0.0):
        raise ValueError("nodal area must be positive")
    k = float(factor) * material.E * np.sqrt(area)
    return float(k) if np.ndim(nodal_area) == 0 else k


def surface_penalty_stiffness(material: J2Material, facet_length, factor: float = 30.0):
    """Surface penalty stiffness ``factor * E / h_facet`` [N/m^3].

    Used by the facet-based contact scheme, where the penalty traction is
    ``t = k_s g``; the physical stiffness of a surface layer of thickness ``h``
    is ``E / h``, and the factor controls the residual penetration
    (``1/factor`` of the elastic deformation).
    """
    h = np.asarray(facet_length, dtype=float)
    if np.any(h <= 0.0):
        raise ValueError("facet length must be positive")
    k = float(factor) * material.E / h
    return float(k) if np.ndim(facet_length) == 0 else k


def top_surface_data(mesh, tol: float = 1e-9):
    """Top-face nodes, their lumped areas and the total face area."""
    z_top = float(mesh.nodes[:, 2].max())
    return surface_nodal_areas(mesh, 2, z_top, tol)


def load_ramp(fraction: np.ndarray, kind: str = "power3") -> np.ndarray:
    """Normalised load-step fractions in ``[0, 1]``.

    ``"linear"``, ``"power2"`` (``s**2``) and ``"power3"`` (``s**3``) refine the
    start of the path, where the contact patch grows from one node to several
    elements.  The final step always reaches exactly ``1``.
    """
    s = np.asarray(fraction, dtype=float)
    if kind == "linear":
        return s
    if kind == "power2":
        return s ** 2
    if kind == "power3":
        return s ** 3
    raise ValueError(f"unknown ramp kind {kind!r}")


# --------------------------------------------------------------------------- #
# Vectorised assembly of the (linear-kinematics) Hex8 block
# --------------------------------------------------------------------------- #
class _Assembler:
    """Cached element kinematics + vectorised internal force/tangent assembly."""

    def __init__(self, mesh, rule: str | None = None, formulation: str = "full") -> None:
        self.mesh = mesh
        kin = [
            element_kinematics(mesh.element, mesh.element_coords(e), rule, formulation)
            for e in range(mesh.n_elem)
        ]
        self.B = np.stack([k.B for k in kin])                    # (ne, ngp, 6, 24)
        self.Bt = np.ascontiguousarray(self.B.transpose(0, 1, 3, 2))
        self.dv = np.stack([k.weights * k.det_j for k in kin])   # (ne, ngp)
        self.dofs = np.stack([mesh.element_dofs(e) for e in range(mesh.n_elem)])
        self.n_elem, self.ngp, _, self.nen = self.B.shape
        self.n_dof = mesh.n_dof
        self.n_gp = self.n_elem * self.ngp
        rows = np.repeat(self.dofs, self.nen, axis=1).ravel()
        cols = np.tile(self.dofs, (1, self.nen)).ravel()
        self.rows, self.cols = rows, cols
        coords = np.empty((self.n_elem, self.ngp, 3))
        for e, k in enumerate(kin):
            xyz = mesh.element_coords(e)
            for g in range(self.ngp):
                N, _ = _shape_fn(mesh.element, *k.points[g])
                coords[e, g] = N @ xyz
        self.gp_coords = coords

    def strains(self, u: np.ndarray) -> np.ndarray:
        ue = u[self.dofs]
        return np.einsum("egij,ej->egi", self.B, ue).reshape(-1, 6)

    def internal_force_tangent(self, stress: np.ndarray, tangent: np.ndarray):
        stress = stress.reshape(self.n_elem, self.ngp, 6)
        tangent = tangent.reshape(self.n_elem, self.ngp, 6, 6)
        tmp = np.einsum("egij,egj->egi", self.Bt, stress)
        f_e = np.einsum("eg,egi->ei", self.dv, tmp)
        f_int = np.zeros(self.n_dof)
        np.add.at(f_int, self.dofs.ravel(), f_e.ravel())
        BtC = np.einsum("egij,egjk->egik", self.Bt, tangent)
        ke = np.einsum("eg,egik,egkj->eij", self.dv, BtC, self.B)
        K = csr_matrix((ke.ravel(), (self.rows, self.cols)),
                       shape=(self.n_dof, self.n_dof))
        return f_int, K


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class IndentationResult:
    """History of a displacement-controlled indentation run (SI units)."""

    delta: np.ndarray
    force: np.ndarray
    elastic_energy: np.ndarray
    plastic_dissipation: np.ndarray
    penalty_energy: np.ndarray
    external_work: np.ndarray
    indenter_work: np.ndarray
    contact_area: np.ndarray
    contact_radius: np.ndarray
    mean_pressure: np.ndarray
    max_von_mises: np.ndarray
    max_alpha: np.ndarray
    newton_iterations: list
    converged: bool
    u: np.ndarray
    von_mises: np.ndarray            # (n_elem, ngp)
    alpha: np.ndarray                # (n_elem, ngp)
    gp_coords: np.ndarray            # (n_elem, ngp, 3) undeformed
    nodal_von_mises: np.ndarray
    penalty: float
    symmetry_factor: int
    nodal_area: float
    material: J2Material = field(repr=False, default=None)  # type: ignore[assignment]

    @property
    def total_force(self) -> float:
        """Symmetry-expanded indentation force at the last converged step [N]."""
        return float(self.force[-1])

    def energy_balance(self, index: int = -1) -> dict:
        """Energy bookkeeping at step ``index`` (AC5)."""
        w = float(self.external_work[index])
        u_el = float(self.elastic_energy[index])
        d_pl = float(self.plastic_dissipation[index])
        u_c = float(self.penalty_energy[index])
        w_ind = float(self.indenter_work[index])
        internal = u_el + d_pl
        return {
            "external_work": w,
            "elastic_energy": u_el,
            "plastic_dissipation": d_pl,
            "penalty_energy": u_c,
            "indenter_work": w_ind,
            "internal_energy": internal,
            "residual": w - internal,
            "relative_residual": (w - internal) / w if w else 0.0,
            "penalty_fraction": u_c / w if w else 0.0,
            "indenter_work_residual": w_ind - (internal + u_c),
            "indenter_work_relative_residual": (
                (w_ind - (internal + u_c)) / w_ind if w_ind else 0.0
            ),
        }

    def summary(self) -> dict:
        return {
            "penalty_stiffness": self.penalty,
            "symmetry_factor": self.symmetry_factor,
            "nodal_area_m2": self.nodal_area,
            "converged": self.converged,
            "delta_max_m": float(self.delta[-1]),
            "force_max_N": self.total_force,
            "contact_radius_m": float(self.contact_radius[-1]),
            "mean_pressure_Pa": float(self.mean_pressure[-1]),
            "max_von_mises_Pa": float(self.max_von_mises[-1]),
            "max_alpha": float(self.max_alpha[-1]),
            "energy_balance": self.energy_balance(),
        }


# --------------------------------------------------------------------------- #
# Residual helper
# --------------------------------------------------------------------------- #
def _residual(mesh, assembler, material, u, contact_fn, eps_p, alpha, free):
    """Return ``(relative_residual, contact_state, resp, f_int, K)`` at ``u``."""
    strain = assembler.strains(u)
    resp = batch_return_map(material, strain, eps_p, alpha)
    f_int, K = assembler.internal_force_tangent(resp.stress, resp.tangent)
    cs = contact_fn(u)
    R = cs.force_vector - f_int
    scale = max(float(np.linalg.norm(cs.force_vector)),
                float(np.linalg.norm(f_int)), 1e-30)
    return float(np.linalg.norm(R[free])) / scale, cs, resp, f_int, K


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def simulate_indentation(
    mesh,
    material: J2Material,
    indenter,
    delta_max: float,
    n_steps: int = 20,
    penalty_factor: float = 50.0,
    penalty: float | None = None,
    fixed_dofs: np.ndarray | None = None,
    contact_nodes: np.ndarray | None = None,
    nodal_area: float | None = None,
    symmetry_factor: int = 4,
    rule: str | None = None,
    formulation: str = "full",
    tol: float = 1e-6,
    max_iter: int = 30,
    line_search: bool = True,
    delta_offset: float = 0.0,
    ramp: str = "linear",
    hysteresis: float | None = None,
    penalty_continuation: int = 1,
    contact_model: str = "facet",
    max_subdivisions: int = 3,
) -> IndentationResult:
    """Run a displacement-controlled rigid-indenter indentation.

    Parameters
    ----------
    mesh : :class:`~tensorphysics.solid_fem.mesh.SolidMesh`
    material : :class:`J2Material`
    indenter : :class:`SphericalIndenter` / :class:`FlatPunchIndenter`
        Indenter positioned at the *undeformed* contact point; it is advanced
        along ``-z`` by ``delta_offset + delta``.
    delta_max, n_steps : total driven penetration and number of load steps.
    penalty_factor, penalty : nodal penalty stiffness (see
        :func:`penalty_stiffness` / :func:`surface_penalty_stiffness`);
        ``penalty`` overrides.
    fixed_dofs : Dirichlet dofs; defaults to :func:`quarter_block_bcs`.
    contact_nodes, nodal_area : candidate nodes for the node-to-surface scheme.
    symmetry_factor : mirror images represented by the model (4 for a quarter
        block); forces and energies are multiplied by it.
    tol, max_iter, line_search : Newton controls (relative force residual,
        iteration cap, residual line search).
    delta_offset : pre-embedding of the indenter (start-up option, see module
        docstring).
    ramp, hysteresis : load ramp kind and active-set hysteresis length.
    penalty_continuation : number of load steps over which the penalty stiffness
        is ramped as ``(i/n)**2`` up to its final value (needed for a
        pre-embedded start, which begins far from equilibrium).
    contact_model : ``"facet"`` (element-face integrated penalty traction,
        default) or ``"node"`` (node-to-surface penalty).
    max_subdivisions : number of times a failed load step may be bisected and
        retried (adaptive continuation).  ``0`` disables the mechanism.
    """
    if delta_max < 0.0:
        raise ValueError("delta_max must be non-negative")
    n_steps = int(n_steps)
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")

    assembler = _Assembler(mesh, rule=rule, formulation=formulation)
    if fixed_dofs is None:
        fixed_dofs = quarter_block_bcs(mesh)
    fixed_dofs = np.asarray(fixed_dofs, dtype=int)
    free = np.setdiff1d(np.arange(mesh.n_dof), fixed_dofs)

    top_nodes, top_areas, _total = top_surface_data(mesh)
    z_top = float(mesh.nodes[:, 2].max())
    if nodal_area is None:
        nodal_area = float(np.mean(top_areas))

    if contact_model == "facet":
        facets = build_surface_facets(mesh, axis=2, value=z_top)
        if penalty is None:
            k_base = np.asarray(surface_penalty_stiffness(material, facets.h,
                                                          penalty_factor),
                                dtype=float)
        else:
            k_base = np.full(facets.n_facets, float(penalty))
        if hysteresis is None:
            hysteresis = 1e-4 * float(np.mean(facets.h))
        contact_areas = facets.areas
    elif contact_model == "node":
        if contact_nodes is None:
            contact_nodes = top_nodes
        contact_nodes = np.asarray(contact_nodes, dtype=int)
        area_map = {int(n): float(a) for n, a in zip(top_nodes, top_areas)}
        contact_areas = np.array([area_map[int(n)] for n in contact_nodes])
        if penalty is None:
            k_base = np.asarray(penalty_stiffness(material, contact_areas,
                                                  penalty_factor), dtype=float)
        else:
            k_base = np.full(contact_nodes.size, float(penalty))
        if hysteresis is None:
            hysteresis = 1e-4 * float(np.sqrt(np.mean(contact_areas)))
    else:  # pragma: no cover - defensive
        raise ValueError(f"unknown contact_model {contact_model!r}")

    state = {"indenter": indenter, "k_scale": 1.0, "previous": None}

    if contact_model == "facet":
        def contact_fn(uu):
            return facet_contact_state(
                mesh, uu, state["indenter"], k_base * state["k_scale"], facets,
                previous=state["previous"], hysteresis=hysteresis)
    else:
        def contact_fn(uu):
            return contact_state(
                mesh, uu, state["indenter"], k_base * state["k_scale"],
                contact_nodes, contact_areas,
                previous=state["previous"], hysteresis=hysteresis)

    n_gp = assembler.n_gp
    eps_p = np.zeros((n_gp, 6))
    alpha = np.zeros(n_gp)

    u = np.zeros(mesh.n_dof)
    u_prev = np.zeros(mesh.n_dof)
    f_contact_prev = np.zeros(mesh.n_dof)
    force_prev = 0.0

    hist = {k: [0.0] for k in ("delta", "force", "u_el", "d_pl", "u_c", "w",
                               "w_ind", "area", "radius", "pressure", "mises",
                               "alpha")}
    iterations: list[dict] = []
    converged_all = True
    stress = np.zeros((n_gp, 6))

    # adaptive load stepping: a failed step is bisected and retried
    targets = [float(f) for f in load_ramp(np.arange(1, n_steps + 1) / n_steps, ramp)]
    targets[-1] = 1.0
    index = 0
    previous_fraction = 0.0
    subdivisions = 0
    step_no = 0

    while index < len(targets):
        fraction = targets[index]
        step_no += 1
        delta = delta_offset + delta_max * fraction
        state["indenter"] = indenter.advance(delta)
        state["previous"] = None
        state["k_scale"] = (min(1.0, (index + 1) / penalty_continuation) ** 2
                            if penalty_continuation > 1 else 1.0)
        u_step = u.copy()
        step_converged = False
        it_used = 0
        rn = float("inf")
        for it in range(max_iter):
            it_used = it + 1
            rn, cs, resp, f_int, K = _residual(
                mesh, assembler, material, u_step, contact_fn, eps_p, alpha, free)
            if rn <= tol:
                step_converged = True
                break
            J = (cs.tangent - K).tocsr()
            R = cs.force_vector - f_int
            du = solve_dirichlet(J, -R, fixed_dofs, 0.0)
            step_len = 1.0
            accepted = False
            best = None
            if line_search:
                for _ in range(6):
                    u_try = u_step + step_len * du
                    rn_try = _residual(mesh, assembler, material, u_try,
                                       contact_fn, eps_p, alpha, free)[0]
                    if best is None or rn_try < best[0]:
                        best = (rn_try, u_try)
                    if rn_try <= (1.0 - 1e-4 * step_len) * rn or rn_try <= tol:
                        u_step, accepted = u_try, True
                        break
                    step_len *= 0.5
                if not accepted:
                    # accept the least-bad damped step (keeps the iteration alive)
                    u_step = best[1]
            else:
                u_step = u_step + du
            state["previous"] = cs.active

        if not step_converged:
            if subdivisions < max_subdivisions and fraction - previous_fraction > 1e-6:
                midpoint = 0.5 * (previous_fraction + fraction)
                targets.insert(index, midpoint)
                subdivisions += 1
                iterations.append({"step": step_no, "delta": delta,
                                   "iterations": it_used, "residual": rn,
                                   "note": "step bisected"})
                continue
            converged_all = False
            iterations.append({"step": step_no, "delta": delta,
                               "iterations": it_used, "residual": rn,
                               "note": "no convergence"})
            break

        # --- converged increment: commit history and integrate energies -----
        strain = assembler.strains(u_step)
        resp = batch_return_map(material, strain, eps_p, alpha)
        stress = resp.stress
        cs = contact_fn(u_step)

        eps_e = strain - resp.eps_p
        e_density = 0.5 * (resp.stress * eps_e).sum(axis=1)
        dv_flat = assembler.dv.ravel()
        u_el = float(np.sum(dv_flat * e_density)) * symmetry_factor
        step_diss = float(np.sum(dv_flat * resp.plastic_dissipation_increment(
            material.sigma_y0, material.hardening, alpha))) * symmetry_factor
        w_step = 0.5 * float(np.dot(f_contact_prev + cs.force_vector,
                                    u_step - u_prev)) * symmetry_factor
        force = cs.total_force * symmetry_factor
        w_ind_step = 0.5 * (force_prev + force) * (delta - hist["delta"][-1])

        hist["u_el"].append(u_el)
        hist["d_pl"].append(hist["d_pl"][-1] + step_diss)
        hist["u_c"].append(cs.energy * symmetry_factor)
        hist["w"].append(hist["w"][-1] + w_step)
        hist["w_ind"].append(hist["w_ind"][-1] + w_ind_step)
        hist["delta"].append(delta)
        hist["force"].append(force)
        hist["area"].append(cs.contact_area * symmetry_factor)
        hist["radius"].append(float(np.sqrt(hist["area"][-1] / np.pi)))
        hist["pressure"].append(force / hist["area"][-1] if hist["area"][-1] > 0 else 0.0)
        hist["mises"].append(float(np.max(resp.mises)))
        hist["alpha"].append(float(np.max(resp.alpha)))
        iterations.append({"step": step_no, "delta": delta,
                           "iterations": it_used, "residual": rn,
                           "contact_nodes": int(cs.active.size), "force": force})

        eps_p = resp.eps_p
        alpha = resp.alpha
        u_prev = u_step.copy()
        u = u_step.copy()
        f_contact_prev = cs.force_vector
        force_prev = force
        previous_fraction = fraction
        subdivisions = 0
        index += 1

    alpha_field = alpha.reshape(assembler.n_elem, assembler.ngp)
    mises = _mises_from_stress(stress).reshape(assembler.n_elem, assembler.ngp)
    nodal_mises = _nodal_average(mesh, assembler, mises)
    arrays = {k: np.asarray(v, dtype=float) for k, v in hist.items()}

    return IndentationResult(
        delta=arrays["delta"],
        force=arrays["force"],
        elastic_energy=arrays["u_el"],
        plastic_dissipation=arrays["d_pl"],
        penalty_energy=arrays["u_c"],
        external_work=arrays["w"],
        indenter_work=arrays["w_ind"],
        contact_area=arrays["area"],
        contact_radius=arrays["radius"],
        mean_pressure=arrays["pressure"],
        max_von_mises=arrays["mises"],
        max_alpha=arrays["alpha"],
        newton_iterations=iterations,
        converged=bool(converged_all and previous_fraction == targets[-1]),
        u=u,
        von_mises=mises,
        alpha=alpha_field,
        gp_coords=assembler.gp_coords,
        nodal_von_mises=nodal_mises,
        penalty=float(np.mean(k_base)),
        symmetry_factor=int(symmetry_factor),
        nodal_area=float(nodal_area),
        material=material,
    )


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _mises_from_stress(stress: np.ndarray) -> np.ndarray:
    """von Mises stress of ``(n, 6)`` Voigt stress rows."""
    m = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    dev = stress - (stress.sum(axis=1) / 3.0)[:, None] * m
    return np.sqrt(1.5 * ((dev * dev).sum(axis=1)
                          + (dev[:, 3:] * dev[:, 3:]).sum(axis=1)))


def _nodal_average(mesh, assembler, gauss_field: np.ndarray) -> np.ndarray:
    """Average a Gauss-point field to the nodes (unweighted, for contour plots)."""
    acc = np.zeros(mesh.n_nodes)
    cnt = np.zeros(mesh.n_nodes)
    for e in range(assembler.n_elem):
        nodes = mesh.connectivity[e]
        acc[nodes] += gauss_field[e].mean()
        cnt[nodes] += 1.0
    return acc / np.maximum(cnt, 1.0)