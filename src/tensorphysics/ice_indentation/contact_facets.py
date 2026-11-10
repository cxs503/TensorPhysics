"""Facet-based penalty contact (element-face integration) for smooth indenters.

The node-to-surface penalty scheme in :mod:`~tensorphysics.ice_indentation.contact`
applies the penalty force at the *nodes* of the contact surface.  For a smooth
indenter this makes the discrete contact patch a ring of nodes, whose resolved
area can differ from the true contact area by a factor of order ``a / h`` (the
contact radius over the element size).  Under J2 plasticity that matters a lot:
the mean contact pressure (and therefore the indentation force) is controlled by
how well the loaded area is resolved, because the hydrostatic stress is not
limited by the yield surface.

This module therefore integrates the penalty traction over the *surface facets*
of the Hex8 mesh (4-node bilinear quad, 2x2 Gauss): a facet that straddles the
contact edge contributes partially, so the discrete contact area follows the
true contact area to second order in ``h`` instead of first order.

Implementation notes
--------------------
* penalty traction ``t = k_s g`` with ``k_s`` a surface stiffness [N/m^3],
  ``k_s = beta * E / h_facet`` (``beta`` = penalty factor);
* nodal forces ``f_i = -sum_gp N_i t detJ w``; consistent tangent only in the
  ``z - z`` block (``-sum_gp N_i N_j k_s detJ w``) -- the radial coupling of a
  spherical indenter is O(r/R) and is left out of the tangent, which the
  residual line search absorbs;
* contact area is *measured* as ``sum_gp detJ w`` over the penetrating Gauss
  points, which is a much better diagnostic than counting nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from .contact import ContactState, SphericalIndenter

__all__ = ["SurfaceFacets", "build_surface_facets", "facet_contact_state"]


# Hex8 face definitions (node ordering: two CCW quads for the -z / +z faces)
_FACES_Z = ((0, 3, 2, 1), (4, 5, 6, 7))

# 2x2 Gauss rule on the reference square
_GP = np.array([-1.0, 1.0]) / sqrt(3.0)
_GW = np.array([1.0, 1.0])


def _quad_shape(xi: float, eta: float) -> np.ndarray:
    """Bilinear quad shape functions ``N_1..N_4``."""
    return 0.25 * np.array([
        (1.0 - xi) * (1.0 - eta),
        (1.0 + xi) * (1.0 - eta),
        (1.0 + xi) * (1.0 + eta),
        (1.0 - xi) * (1.0 + eta),
    ])


@dataclass
class SurfaceFacets:
    """All facets of one face of a Hex8 block, with quadrature data precomputed."""

    nodes: np.ndarray       # (nf, 4) global node ids, sorted as a quad loop
    N: np.ndarray           # (nf, ngp, 4) shape functions at the Gauss points
    jac_w: np.ndarray       # (nf, ngp) detJ * w  (undeformed area weights)
    areas: np.ndarray       # (nf,) facet area
    h: np.ndarray           # (nf,) sqrt(area) - local length scale
    z0: float               # reference plane coordinate of the face

    @property
    def n_facets(self) -> int:
        return self.nodes.shape[0]


def build_surface_facets(mesh, axis: int = 2, value: float | None = None,
                         tol: float = 1e-9) -> SurfaceFacets:
    """Collect the Hex8 facets lying on ``x_axis = value`` (default: top face)."""
    if str(mesh.element).lower() != "hex8":
        raise ValueError("facet contact is implemented for hex8 meshes")
    if value is None:
        value = float(mesh.nodes[:, axis].max())
    on_face = np.abs(mesh.nodes[:, axis] - value) <= tol
    facet_nodes = []
    for e in range(mesh.n_elem):
        el = mesh.connectivity[e]
        # pick the local face whose four nodes are all on the plane
        for local in _FACES_Z:
            if np.all(on_face[el[list(local)]]):
                facet_nodes.append(el[list(local)])
                break
    if not facet_nodes:
        raise ValueError(f"no facets found on axis {axis} = {value}")
    nodes = np.array(facet_nodes, dtype=int)

    nf = nodes.shape[0]
    ngp = 4
    N = np.zeros((nf, ngp, 4))
    jac_w = np.zeros((nf, ngp))
    areas = np.zeros(nf)
    g = 0
    for eta in _GP:
        for xi in _GP:
            N[:, g, :] = _quad_shape(xi, eta)
            g += 1
    # surface Jacobian: |dX/dxi x dX/deta| at every Gauss point
    for f in range(nf):
        X = mesh.nodes[nodes[f]]                      # (4, 3)
        xyz = np.einsum("gi,ij->gj", N[f], X)         # (4, 3) Gauss points
        for g in range(ngp):
            xi, eta = _GP[g % 2], _GP[g // 2]
            dN_dxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
            dN_deta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
            t1 = dN_dxi @ X
            t2 = dN_deta @ X
            j = float(np.linalg.norm(np.cross(t1, t2)))
            jac_w[f, g] = j * _GW[g % 2] * _GW[g // 2]
        areas[f] = float(np.sum(jac_w[f]))
    # fix the Gauss-point coordinates consistency (N order: (xi, eta) pairs below)
    return SurfaceFacets(nodes=nodes, N=N, jac_w=jac_w, areas=areas,
                         h=np.sqrt(areas), z0=float(value))


def facet_contact_state(
    mesh,
    u: np.ndarray,
    indenter,
    k_surface,
    facets: SurfaceFacets,
    previous: np.ndarray | None = None,
    hysteresis: float = 0.0,
) -> ContactState:
    """Penalty contact integrated over the surface facets at displacement ``u``.

    ``k_surface`` is a scalar or a per-facet array of penalty stiffnesses
    [N/m^3] (see :func:`tensorphysics.ice_indentation.indentation.penalty_stiffness`
    for the calibration).
    """
    nodes = facets.nodes                                  # (nf, 4)
    nf, ngp = facets.jac_w.shape
    k = (np.full(nf, float(k_surface)) if np.ndim(k_surface) == 0
         else np.asarray(k_surface, dtype=float))
    if np.any(k <= 0.0):
        raise ValueError("surface penalty stiffness must be positive")

    ux = u[3 * nodes + 0]
    uy = u[3 * nodes + 1]
    uz = u[3 * nodes + 2]
    X = mesh.nodes[nodes]
    gx = np.einsum("fgi,fi->fg", facets.N, X[:, :, 0] + ux)
    gy = np.einsum("fgi,fi->fg", facets.N, X[:, :, 1] + uy)
    gz = np.einsum("fgi,fi->fg", facets.N, X[:, :, 2] + uz)

    gaps = _gap_array(indenter, gx, gy, gz)
    active = gaps > 0.0
    if previous is not None and hysteresis > 0.0 and np.size(previous):
        # keep facets that were active in the previous iterate inside the band
        prev_nodes = np.isin(nodes, np.asarray(previous, dtype=int)).any(axis=1)
        if np.any(prev_nodes):
            band = np.zeros_like(active)
            band[prev_nodes] = gaps[prev_nodes] > -hysteresis
            active = active | band
    if not np.any(active):
        n_dof = mesh.n_dof
        return ContactState(active=np.array([], dtype=int), gaps=np.array([]),
                            nodal_force_z=np.zeros(mesh.n_nodes),
                            force_vector=np.zeros(n_dof),
                            tangent=csr_matrix((n_dof, n_dof)),
                            total_force=0.0, energy=0.0,
                            contact_area_actual=0.0,
                            nodal_area=float(np.mean(facets.areas)))

    # penetration is clamped to >= 0: the hysteresis band may keep a facet in
    # the active set with a (marginally) negative gap, but a penalty spring can
    # only push, never pull
    gap_a = np.where(active, np.maximum(gaps, 0.0), 0.0)
    traction = np.where(active, k[:, None] * gap_a, 0.0)    # N/m^2
    weight = np.where(active, facets.jac_w, 0.0)           # m^2

    # nodal forces  f_i = - sum_gp N_i t jac_w
    f_nodal = -np.einsum("fgi,fg->fi", facets.N, traction * weight)
    n_dof = mesh.n_dof
    f = np.zeros(n_dof)
    np.add.at(f, (3 * nodes + 2).ravel(), f_nodal.ravel())

    # consistent tangent in the z-z block:  - sum_gp N_i N_j k jac_w
    rows, cols, vals = [], [], []
    for i in range(4):
        for j in range(4):
            kij = -np.einsum("fg,fg->f", facets.N[:, :, i] * facets.N[:, :, j],
                             np.where(active, k[:, None] * facets.jac_w, 0.0))
            rows.append(3 * nodes[:, i] + 2)
            cols.append(3 * nodes[:, j] + 2)
            vals.append(kij)
    tangent = coo_matrix(
        (np.concatenate(vals),
         (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_dof, n_dof),
    ).tocsr()

    total = float(-f[2::3].sum())
    energy = float(0.5 * np.sum(k[:, None] * gap_a ** 2 * weight))
    area = float(np.sum(weight))
    touching = np.unique(nodes[active.any(axis=1)])
    return ContactState(
        active=touching,
        gaps=gap_a[active],
        nodal_force_z=f[2::3],
        force_vector=f,
        tangent=tangent,
        total_force=total,
        energy=energy,
        contact_area_actual=area,
        nodal_area=float(np.mean(facets.areas)),
    )


def _gap_array(indenter, gx: np.ndarray, gy: np.ndarray,
               gz: np.ndarray) -> np.ndarray:
    """Vectorised gap of an indenter over an array of Gauss points."""
    if isinstance(indenter, SphericalIndenter):
        r = np.hypot(gx, gy)
        r = np.minimum(r, indenter.radius)
        surface = indenter.apex_z + (indenter.radius
                                     - np.sqrt(np.maximum(
                                         indenter.radius ** 2 - r ** 2, 0.0)))
        return gz - surface
    # generic (slower) fallback for any indenter exposing ``gap``
    flat = gz.ravel()
    return np.array([indenter.gap(x, y, z) for x, y, z in
                     zip(gx.ravel(), gy.ravel(), flat)]).reshape(gx.shape)