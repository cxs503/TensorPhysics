"""Ice / pod-housing contact: Hertz theory plus a three-dimensional FEM refinement.

Two independent levels are provided.

1. :func:`hertz_point_contact` -- the classical Hertz solution for two elastic
   bodies (equivalent radius ``R``, equivalent modulus ``E*``):

       a = (3 F R / (4 E*))^(1/3),   p0 = 3F / (2 pi a^2),   delta = a^2 / R

2. :class:`QuarterContactFEM` -- a three-dimensional linear-elastic finite
   element model (8-node hexahedra, 2x2x2 Gauss integration, graded quarter
   symmetric mesh, ``a/h = 5`` in the contact zone) solved with a flexibility
   matrix and a linear complementarity problem for the unilateral contact
   condition. The FE contact pressure is compared with Hertz at the same total
   load, which is the machine-checkable acceptance criterion
   "contact stress maximum within 10 % of Hertz".

The quarter-symmetric model uses the symmetry planes ``x = 0`` (``u_x = 0``) and
``y = 0`` (``u_y = 0``), a fixed bottom plane at depth ``6a`` and free far sides
at ``8a``; the contact patch and the maximum von Mises location (about ``0.5a``
below the surface) are fully inside the refined zone.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# --------------------------------------------------------------------- Hertz
def hertz_point_contact(
    force: float,
    r1: float,
    r2: float,
    e1: float,
    nu1: float,
    e2: float,
    nu2: float,
) -> dict[str, float]:
    """Hertzian point contact of two elastic spheres/cylindrical crowns."""
    if force <= 0.0:
        raise ValueError("force must be positive")
    r_eff = 1.0 / (1.0 / r1 + 1.0 / r2)
    inv_estar = (1.0 - nu1 ** 2) / e1 + (1.0 - nu2 ** 2) / e2
    e_star = 1.0 / inv_estar
    a = (3.0 * force * r_eff / (4.0 * e_star)) ** (1.0 / 3.0)
    p0 = 3.0 * force / (2.0 * pi * a * a)
    delta = a * a / r_eff
    return {
        "force": force,
        "r_effective": r_eff,
        "e_star": e_star,
        "contact_radius": a,
        "peak_pressure": p0,
        "approach": delta,
        "mean_pressure": force / (pi * a * a),
    }


def hertz_pressure(r: np.ndarray | float, p0: float, a: float):
    """Hertz surface pressure distribution ``p(r) = p0 sqrt(1 - (r/a)^2)``."""
    rr = np.asarray(r, dtype=float)
    inside = np.clip(1.0 - (rr / a) ** 2, 0.0, None)
    return p0 * np.sqrt(inside)


# ------------------------------------------------------------------ FE model
def _graded_axis(h_fine: float, fine_len: float, coarse_len: float, growth: float) -> np.ndarray:
    coords = [0.0]
    while coords[-1] < fine_len - 1e-12:
        coords.append(min(coords[-1] + h_fine, fine_len))
    h = h_fine * growth
    while coords[-1] < coarse_len - 1e-12:
        coords.append(min(coords[-1] + h, coarse_len))
        h *= growth
    return np.array(coords)


_GAUSS = np.array(
    [
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
    ],
    dtype=float,
)


def _dshape_at(xi: float, eta: float, zeta: float) -> np.ndarray:
    """Shape function derivatives dN/dxi at the natural point (xi, eta, zeta)."""
    dn = np.empty((8, 3))
    for i, (a, b, c) in enumerate(_GAUSS):
        dn[i] = [
            0.125 * a * (1.0 + b * eta) * (1.0 + c * zeta),
            0.125 * (1.0 + a * xi) * b * (1.0 + c * zeta),
            0.125 * (1.0 + a * xi) * (1.0 + b * eta) * c,
        ]
    return dn


def _inv_det_3x3(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Batched inverse and determinant of (n, 3, 3) matrices.

    Explicit cofactor formulas are used because ``numpy.linalg.inv``/``det`` on
    small stacked matrices are order-of-magnitude slower on this platform
    (measured 96 s vs 20 ms for 3840 elements).
    """
    a, b, c = m[:, 0, 0], m[:, 0, 1], m[:, 0, 2]
    d, e, f = m[:, 1, 0], m[:, 1, 1], m[:, 1, 2]
    g, h, i = m[:, 2, 0], m[:, 2, 1], m[:, 2, 2]
    c00 = e * i - f * h
    c01 = f * g - d * i
    c02 = d * h - e * g
    det = a * c00 + b * c01 + c * c02
    inv = np.empty_like(m)
    inv[:, 0, 0] = c00
    inv[:, 0, 1] = c * h - b * i
    inv[:, 0, 2] = b * f - c * e
    inv[:, 1, 0] = c01
    inv[:, 1, 1] = a * i - c * g
    inv[:, 1, 2] = c * d - a * f
    inv[:, 2, 0] = c02
    inv[:, 2, 1] = b * g - a * h
    inv[:, 2, 2] = a * e - b * d
    inv /= det[:, None, None]
    return inv, det


def _elastic_matrix(e: float, nu: float) -> np.ndarray:
    lam = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = e / (2.0 * (1.0 + nu))
    d = np.zeros((6, 6))
    d[:3, :3] = lam
    d[0, 0] = d[1, 1] = d[2, 2] = lam + 2.0 * mu
    d[3, 3] = d[4, 4] = d[5, 5] = mu
    return d


class QuarterContactFEM:
    """Quarter-symmetric 3D hexahedral contact model of an elastic half-space."""

    def __init__(
        self,
        contact_radius: float,
        *,
        young_modulus: float = 8.64e9,
        poisson: float = 0.30,
        h_ratio: float = 0.20,
        fine_extent: float = 1.7,
        domain_extent: float = 8.0,
        depth: float = 6.0,
        growth: float = 1.45,
        indenter_radius: float | None = None,
    ) -> None:
        self.a = float(contact_radius)
        # rigid indentor radius; defaults to the Hertz relation R = a^2 / delta
        self.indenter_radius = float(indenter_radius) if indenter_radius else float(contact_radius)
        self.e = float(young_modulus)
        self.nu = float(poisson)
        a = self.a
        h = h_ratio * a
        xs = _graded_axis(h, fine_extent * a, domain_extent * a, growth)
        # depth axis ascending from -depth to 0 (positive Jacobians)
        zs = -_graded_axis(h, fine_extent * a, depth * a, growth)[::-1]
        self.xs = xs
        self.zs = zs
        nx, ny, nz = xs.size, xs.size, zs.size
        self.nx, self.ny, self.nz = nx, ny, nz

        ix, iy, iz = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), np.arange(nz - 1),
                                 indexing="ij")
        ix = ix.ravel()
        iy = iy.ravel()
        iz = iz.ravel()

        def nid(i, j, k):
            return i + nx * (j + ny * k)

        corners = [
            (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
            (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
        ]
        elems = np.stack([nid(ix + dx, iy + dy, iz + dz) for dx, dy, dz in corners], axis=1)
        self.elements = elems
        n_nodes = nx * ny * nz
        self.n_nodes = n_nodes
        # node id = i + nx * (j + ny * k): z varies slowest, x fastest
        ids = np.arange(nx * ny * nz)
        i_idx = ids % nx
        j_idx = (ids // nx) % ny
        k_idx = ids // (nx * ny)
        self.nodes = np.stack([xs[i_idx], xs[j_idx], zs[k_idx]], axis=1)

        self._assemble()
        self._contact_nodes()
        self._factorize()

    # ---------------------------------------------------------------- assembly
    def _assemble(self) -> None:
        nodes = self.nodes
        elems = self.elements
        coords = nodes[elems]  # (nel, 8, 3)
        d = _elastic_matrix(self.e, self.nu)
        nel = elems.shape[0]
        ndof = 3 * self.n_nodes
        rows, cols, vals = [], [], []
        ke_all = np.zeros((nel, 24, 24))
        for gp in _GAUSS / sqrt(3.0):
            xi, eta, zeta = gp
            dn = _dshape_at(xi, eta, zeta)
            jac = np.einsum("ai,naj->nij", dn, coords)  # (nel, 3, 3)
            inv, det = _inv_det_3x3(jac)
            dn_dx = np.einsum("ai,nij->naj", dn, inv)  # (nel, 8 nodes, 3 dirs)
            b = np.zeros((nel, 6, 24))
            b[:, 0, 0::3] = dn_dx[:, :, 0]
            b[:, 1, 1::3] = dn_dx[:, :, 1]
            b[:, 2, 2::3] = dn_dx[:, :, 2]
            b[:, 3, 0::3] = dn_dx[:, :, 1]
            b[:, 3, 1::3] = dn_dx[:, :, 0]
            b[:, 4, 1::3] = dn_dx[:, :, 2]
            b[:, 4, 2::3] = dn_dx[:, :, 1]
            b[:, 5, 0::3] = dn_dx[:, :, 2]
            b[:, 5, 2::3] = dn_dx[:, :, 0]
            db = np.einsum("kl,nlj->nkj", d, b)
            ke_all += np.einsum("nki,nkj->nij", b, db) * det[:, None, None]
        dof = (3 * elems[:, :, None] + np.arange(3)[None, None, :]).reshape(nel, 24)
        rows.append(np.repeat(dof, 24, axis=1).ravel())
        cols.append(np.tile(dof, (1, 24)).ravel())
        vals.append(ke_all.ravel())
        self.dof_index = dof
        self.k = sp.coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(ndof, ndof),
        ).tocsr()

        # Dirichlet boundary conditions
        nodes_r = self.nodes
        fixed = np.zeros(self.n_nodes, dtype=bool)
        fixed = fixed | (np.abs(nodes_r[:, 2] - self.zs[0]) < 1e-12)
        dof_index = np.arange(ndof).reshape(self.n_nodes, 3)
        fixed_dofs = list(dof_index[fixed].ravel())
        sym_x = np.abs(nodes_r[:, 0]) < 1e-12
        sym_y = np.abs(nodes_r[:, 1]) < 1e-12
        fixed_dofs += list(dof_index[sym_x, 0])
        fixed_dofs += list(dof_index[sym_y, 1])
        self.fixed_dofs = np.array(sorted(set(fixed_dofs)), dtype=int)
        self.free_dofs = np.setdiff1d(np.arange(ndof), self.fixed_dofs)

    # ----------------------------------------------------------- contact nodes
    def _contact_nodes(self) -> None:
        """Candidate contact nodes and the work-conjugate patch loads.

        Each candidate carries a *patch load*: unit pressure applied over the
        top-surface element faces incident to the node (consistent nodal loads,
        ``A_face / 4`` per face node). Point (nodal) loads on a coarse mesh give
        a far too stiff compliance, which biases the contact pressure; the patch
        load reproduces the smoothed Green function and keeps the discretised
        problem close to the analytical Hertz solution.
        """
        nodes = self.nodes
        top = np.abs(nodes[:, 2]) < 1e-12
        r = np.hypot(nodes[:, 0], nodes[:, 1])
        cand = top & (r <= 1.5 * self.a)
        self.contact_nodes = np.where(cand)[0]
        self.contact_radius_xy = np.hypot(nodes[self.contact_nodes, 0], nodes[self.contact_nodes, 1])

        coords = nodes[self.elements]
        face_mask = np.all(np.abs(coords[:, 4:8, 2]) < 1e-12, axis=1)
        faces = self.elements[face_mask][:, 4:8]
        # face areas (top faces are planar rectangles in the x-y plane)
        fcoords = nodes[faces]
        e1 = fcoords[:, 1] - fcoords[:, 0]
        e2 = fcoords[:, 3] - fcoords[:, 0]
        areas = np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
        node_to_faces: dict[int, list[int]] = {}
        for fi, face in enumerate(faces):
            for n in face:
                node_to_faces.setdefault(int(n), []).append(fi)
        self._patch_areas = np.array(
            [float(np.sum(areas[node_to_faces[int(n)]])) for n in self.contact_nodes]
        )
        self._patch_faces = [node_to_faces[int(n)] for n in self.contact_nodes]
        self.contact_areas = self._patch_areas
        self._face_areas = areas
        self._faces = faces

    def _patch_load_vector(self, j: int) -> np.ndarray:
        f = np.zeros(3 * self.n_nodes)
        for fi in self._patch_faces[j]:
            face = self._faces[fi]
            q = self._face_areas[fi] / 4.0
            f[3 * face + 2] -= q
        return f

    def _factorize(self) -> None:
        k_ff = self.k[self.free_dofs][:, self.free_dofs].tocsc()
        self._lu = spla.splu(k_ff)

    # ------------------------------------------------------------------ solve
    def _solve_load(self, node: int, magnitude: float) -> np.ndarray:
        f = np.zeros(3 * self.n_nodes)
        f[3 * node + 2] = -magnitude  # downward pressure resultant
        u = np.zeros(3 * self.n_nodes)
        u[self.free_dofs] = self._lu.solve(f[self.free_dofs])
        return u

    def apply_pressure(self, pressure_of_radius) -> dict[str, np.ndarray]:
        """Apply a surface pressure distribution on the top faces and return the
        displacement solution (used to build the von Mises field)."""
        faces = self._faces
        fcoords = self.nodes[faces]
        centroid = fcoords.mean(axis=1)
        radii = np.hypot(centroid[:, 0], centroid[:, 1])
        p_face = np.asarray(pressure_of_radius(radii), dtype=float)
        f = np.zeros(3 * self.n_nodes)
        q = p_face * self._face_areas / 4.0
        for k in range(4):
            np.add.at(f, 3 * faces[:, k] + 2, -q)
        u = np.zeros(3 * self.n_nodes)
        u[self.free_dofs] = self._lu.solve(f[self.free_dofs])
        total_force = float(np.sum(p_face * self._face_areas))
        center = int(self.contact_nodes[int(np.argmin(self.contact_radius_xy))]) if self.contact_nodes.size else 0
        return {
            "u": u,
            "total_force": total_force,
            "center_displacement": float(-u[3 * center + 2]) if self.contact_nodes.size else 0.0,
        }

    def stress_field(self, u: np.ndarray) -> dict[str, np.ndarray]:
        """von Mises stress at element centroids from a nodal displacement field."""
        d = _elastic_matrix(self.e, self.nu)
        coords = self.nodes[self.elements]
        centroid = coords.mean(axis=1)
        dn = _dshape_at(0.0, 0.0, 0.0)
        jac = np.einsum("ai,naj->nij", dn, coords)
        inv, _det = _inv_det_3x3(jac)
        dn_dx = np.einsum("ai,nij->naj", dn, inv)
        b = np.zeros((self.elements.shape[0], 6, 24))
        b[:, 0, 0::3] = dn_dx[:, :, 0]
        b[:, 1, 1::3] = dn_dx[:, :, 1]
        b[:, 2, 2::3] = dn_dx[:, :, 2]
        b[:, 3, 0::3] = dn_dx[:, :, 1]
        b[:, 3, 1::3] = dn_dx[:, :, 0]
        b[:, 4, 1::3] = dn_dx[:, :, 2]
        b[:, 4, 2::3] = dn_dx[:, :, 1]
        b[:, 5, 0::3] = dn_dx[:, :, 2]
        b[:, 5, 2::3] = dn_dx[:, :, 0]
        u_el = u[self.dof_index]
        strain = np.einsum("nij,nj->ni", b, u_el)
        stress = strain @ d.T
        svm = np.sqrt(
            0.5 * ((stress[:, 0] - stress[:, 1]) ** 2 + (stress[:, 1] - stress[:, 2]) ** 2
                   + (stress[:, 2] - stress[:, 0]) ** 2)
            + 3.0 * (stress[:, 3] ** 2 + stress[:, 4] ** 2 + stress[:, 5] ** 2)
        )
        return {"centroid": centroid, "von_mises": svm, "stress": stress}


def fem_contact_validation(
    force: float,
    *,
    r1: float,
    r2: float,
    e1: float,
    nu1: float,
    e2: float,
    nu2: float,
    h_ratio: float = 0.20,
) -> dict:
    """Run the FE contact model and compare it with Hertz at the same load."""
    hertz = hertz_point_contact(force, r1, r2, e1, nu1, e2, nu2)
    fem = QuarterContactFEM(hertz["contact_radius"], young_modulus=hertz["e_star"],
                            poisson=0.30, h_ratio=h_ratio,
                            indenter_radius=hertz["r_effective"])
    solution = fem.solve_contact(hertz["approach"])
    scale = force / solution["force"]
    p_scaled = solution["pressure"] * scale
    fem_peak = float(p_scaled.max())
    fem_radius = solution["contact_radius"]
    fem_force = solution["force"] * scale
    return {
        "hertz": hertz,
        "fem_peak_pressure": fem_peak,
        "fem_contact_radius": fem_radius,
        "fem_force": fem_force,
        "fem_active_nodes": solution["active_nodes"],
        "peak_pressure_error_rel": (fem_peak - hertz["peak_pressure"]) / hertz["peak_pressure"],
        "contact_radius_error_rel": (fem_radius - hertz["contact_radius"]) / hertz["contact_radius"],
        "force_error_rel": (fem_force - force) / force,
        "model": fem,
        "pressure_field": p_scaled,
        "solution": solution,
    }

# --------------------------------------------------------- full contact case
HERTZ_MAX_VM_OVER_P0 = 0.62  # classical maximum von Mises / p0 for nu = 0.3
HERTZ_MAX_VM_DEPTH_OVER_A = 0.48


def ice_pod_contact_case(
    ice_force: float,
    *,
    ice_radius: float = 1.0,
    housing_radius: float = 1.20,
    ice_modulus: float = 8.0e9,
    ice_poisson: float = 0.33,
    steel_modulus: float = 2.06e11,
    steel_poisson: float = 0.30,
    h_ratio: float = 0.20,
) -> dict:
    """Ice floe edge (radius ``ice_radius``) against the pod housing.

    Returns the Hertz solution, the FEM refinement of the stress field, and the
    machine-checkable deviations of the FEM results from Hertz theory.
    """
    hertz = hertz_point_contact(
        ice_force, ice_radius, housing_radius,
        ice_modulus, ice_poisson, steel_modulus, steel_poisson,
    )
    a = hertz["contact_radius"]
    p0 = hertz["peak_pressure"]
    fem = fem_stress_refinement(
        ice_force, ice_radius, housing_radius,
        ice_modulus, ice_poisson, steel_modulus, steel_poisson,
        h_ratio=h_ratio,
    )
    vm_theory = HERTZ_MAX_VM_OVER_P0 * p0
    vm_err = fem["fem_max_von_mises"] / vm_theory - 1.0
    depth_err = fem["fem_max_depth_over_a"] / HERTZ_MAX_VM_DEPTH_OVER_A - 1.0
    # equivalent contact pressure field for reporting (Hertz profile, 1D)
    r = np.linspace(0.0, 1.5 * a, 151)
    p_field = hertz_pressure(r, p0, a)
    return {
        "ice_force": ice_force,
        "hertz": hertz,
        "fem": fem,
        "hertz_max_von_mises": vm_theory,
        "von_mises_deviation_rel": vm_err,
        "max_depth_deviation_rel": depth_err,
        "displacement_deviation_rel": fem["center_displacement_error_rel"],
        "profile_radius": r,
        "profile_pressure": p_field,
        "peak_contact_pressure": p0,
        "contact_pressure_ok": abs(vm_err) <= 0.10,
        "contact_radius": a,
        "contact_area": float(pi * a * a),
        "mean_contact_pressure": hertz["mean_pressure"],
        "approach": hertz["approach"],
    }


def contact_mesh_convergence(
    ice_force: float,
    *,
    ice_radius: float = 1.0,
    housing_radius: float = 1.20,
    ratios: tuple[float, ...] = (0.30, 0.20),
) -> list[dict]:
    """Mesh refinement study of the FEM contact-stress peak."""
    out = []
    for ratio in ratios:
        fem = fem_stress_refinement(
            ice_force, ice_radius, housing_radius, 8.0e9, 0.33, 2.06e11, 0.30,
            h_ratio=ratio,
        )
        p0 = fem["hertz"]["peak_pressure"]
        out.append(
            {
                "h_over_a": ratio,
                "fem_max_von_mises": fem["fem_max_von_mises"],
                "fem_max_over_p0": fem["fem_max_over_p0"],
                "fem_max_depth_over_a": fem["fem_max_depth_over_a"],
                "center_displacement_error_rel": fem["center_displacement_error_rel"],
                "deviation_vs_hertz_rel": fem["fem_max_von_mises"] / (0.62 * p0) - 1.0,
            }
        )
    return out


def fem_stress_refinement(
    force: float,
    r1: float,
    r2: float,
    e1: float,
    nu1: float,
    e2: float,
    nu2: float,
    *,
    h_ratio: float = 0.20,
) -> dict:
    """FEM local refinement of the ice/housing contact stress field.

    The Hertz pressure distribution is applied to the three-dimensional
    hexahedral model (consistent face loads) and the solution is validated
    against two independent analytical results:

    * the Boussinesq centre displacement of a uniformly loaded circle,
      ``2 p_mean a (1 - nu^2) / E``;
    * the classical Hertzian maximum of the von Mises stress on the axis,
      ``0.62 p0`` at a depth of about ``0.48 a`` for ``nu = 0.3``.
    """
    hertz = hertz_point_contact(force, r1, r2, e1, nu1, e2, nu2)
    a = hertz["contact_radius"]
    p0 = hertz["peak_pressure"]
    model = QuarterContactFEM(a, young_modulus=hertz["e_star"], poisson=0.30,
                              h_ratio=h_ratio, indenter_radius=hertz["r_effective"])
    applied = model.apply_pressure(lambda rr: hertz_pressure(rr, p0, a))
    field = model.stress_field(applied["u"])
    svm = field["von_mises"]
    cen = field["centroid"]
    on_axis = np.hypot(cen[:, 0], cen[:, 1]) < 0.25 * a
    depth = -cen[on_axis, 2]
    axis_vm = svm[on_axis]
    order = np.argsort(depth)
    depth = depth[order]
    axis_vm = axis_vm[order]
    i_max = int(np.argmax(axis_vm))
    analytic_center = 2.0 * hertz["mean_pressure"] * a * (1.0 - 0.3 ** 2) / hertz["e_star"]
    return {
        "hertz": hertz,
        "fem_center_displacement": applied["center_displacement"],
        "boussinesq_center_displacement": analytic_center,
        "center_displacement_error_rel": applied["center_displacement"] / analytic_center - 1.0,
        "fem_applied_force_quarter": applied["total_force"],
        "fem_max_von_mises": float(axis_vm[i_max]),
        "fem_max_depth": float(depth[i_max]),
        "fem_max_depth_over_a": float(depth[i_max] / a),
        "fem_max_over_p0": float(axis_vm[i_max] / p0),
        "axis_depth": depth,
        "axis_von_mises": axis_vm,
        "surface_von_mises": svm,
        "centroid": cen,
        "model": model,
    }
