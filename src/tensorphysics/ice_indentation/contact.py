"""Penalty contact between a rigid indenter and a deformable ice block.

Node-to-rigid-surface penalty contact for a *smooth* rigid indenter:

* gap function ``g(x, u)`` -- positive when the surface node has penetrated
  the indenter (this sign convention makes the contact force
  ``f = -k_p g e_z``, i.e. pushing the ice downwards);
* active set = nodes with ``g > 0`` at the current iterate, re-evaluated every
  Newton iteration (so the contact area is part of the nonlinear solution and
  not a user input); a "hysteresis" band keeps nodes that were active in the
  previous iterate while their gap is only marginally negative, which
  suppresses active-set chattering between Newton iterations;
* consistent contact tangent ``df/du = -k_p e_z (x) grad_u g`` -- the full
  ``3x3`` block including the radial coupling of a curved indenter, which is
  what keeps the global Newton iteration convergent;
* contact energy ``1/2 k_p g^2`` -- reported separately so the global energy
  balance can show that the penalty regularisation holds a negligible share
  of the work.

The representation is deliberately dependency-light (numpy + scipy sparse) and
backend-neutral: the indenter objects only expose ``gap`` and ``gap_gradient``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

__all__ = [
    "SphericalIndenter",
    "FlatPunchIndenter",
    "ContactState",
    "contact_state",
    "active_set",
]


# --------------------------------------------------------------------------- #
# Indenter geometries
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SphericalIndenter:
    """Rigid sphere, lowest point (apex) at ``apex_z`` on the axis ``x = y = 0``.

    Surface height at radius ``r``: ``z_i(r) = apex_z + (R - sqrt(R^2 - r^2))``
    -- the sphere touches the plane at ``apex_z`` and rises away from the axis.
    ``apex_z`` is advanced monotonically in a displacement-controlled test, so
    the prescribed nominal penetration is
    ``delta = z_surface_undeformed - apex_z``.
    """

    radius: float
    apex_z: float

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise ValueError("indenter radius must be positive")

    def advance(self, delta: float) -> "SphericalIndenter":
        """Return a copy whose apex is ``delta`` below the current apex height."""
        return SphericalIndenter(radius=self.radius, apex_z=self.apex_z - delta)

    def at_apex(self, z: float) -> "SphericalIndenter":
        return SphericalIndenter(radius=self.radius, apex_z=z)

    # ------------------------------------------------------------------
    def surface_z(self, r: float) -> float:
        r = min(abs(float(r)), self.radius)
        return self.apex_z + (self.radius - sqrt(max(self.radius ** 2 - r ** 2, 0.0)))

    def gap(self, x: float, y: float, z: float) -> float:
        return z - self.surface_z(sqrt(x * x + y * y))

    def gap_gradient(self, x: float, y: float) -> np.ndarray:
        """``[dg/dx, dg/dy, dg/dz]`` at a point of the *current* configuration."""
        r2 = x * x + y * y
        s2 = self.radius ** 2 - r2
        if s2 <= 1e-18:
            # on (or beyond) the equator the sphere is vertical: no radial slope
            return np.array([0.0, 0.0, 1.0])
        s = sqrt(s2)
        return np.array([-x / s, -y / s, 1.0])

    def gap_magnitude(self) -> float:
        return self.radius


@dataclass(frozen=True)
class FlatPunchIndenter:
    """Rigid flat cylindrical punch of radius ``radius`` with face at ``punch_z``."""

    radius: float
    punch_z: float

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise ValueError("punch radius must be positive")

    def advance(self, delta: float) -> "FlatPunchIndenter":
        return FlatPunchIndenter(radius=self.radius, punch_z=self.punch_z - delta)

    def at_apex(self, z: float) -> "FlatPunchIndenter":
        return FlatPunchIndenter(radius=self.radius, punch_z=z)

    def gap(self, x: float, y: float, z: float) -> float:
        if sqrt(x * x + y * y) > self.radius * (1.0 + 1e-12):
            return -1.0  # outside the punch: never in contact
        return z - self.punch_z

    def gap_gradient(self, x: float, y: float) -> np.ndarray:
        return np.array([0.0, 0.0, 1.0])

    def gap_magnitude(self) -> float:
        return float("inf")


# --------------------------------------------------------------------------- #
# Active set / force / tangent assembly
# --------------------------------------------------------------------------- #
@dataclass
class ContactState:
    """Contact forces, tangent and diagnostics at one Newton iterate."""

    active: np.ndarray          # node ids in contact
    gaps: np.ndarray            # penetration (m) at the active nodes
    nodal_force_z: np.ndarray   # z force applied to the body (negative = down)
    force_vector: np.ndarray    # (n_dof,) global contact force vector
    tangent: csr_matrix         # (n_dof, n_dof) d f_contact / d u
    total_force: float          # indentation force (positive)
    energy: float               # penalty energy 1/2 k g^2 (J)
    contact_area_actual: float  # sum of the active nodes' lumped areas (m^2)
    nodal_area: float           # mean lumped area of the candidate nodes

    @property
    def contact_area(self) -> float:
        return self.contact_area_actual

    @property
    def effective_radius(self) -> float:
        return sqrt(max(self.contact_area, 0.0) / np.pi)

    @property
    def mean_pressure(self) -> float:
        area = self.contact_area
        return 0.0 if area <= 0.0 else self.total_force / area


def active_set(
    mesh, u: np.ndarray, indenter, candidate_nodes: np.ndarray | None = None,
    previous: np.ndarray | None = None, hysteresis: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Nodes with ``g > 0`` and their penetration depths.

    ``previous`` (node ids active at the previous Newton iterate) together with
    ``hysteresis`` (a length, e.g. ``1e-9 * domain size``) keeps a node in the
    active set while its gap is only marginally negative.  This suppresses the
    classic penalty-contact active-set chattering between Newton iterations.
    """
    nodes = (np.arange(mesh.n_nodes) if candidate_nodes is None
             else np.asarray(candidate_nodes, dtype=int))
    x = mesh.nodes[nodes, 0] + u[3 * nodes + 0]
    y = mesh.nodes[nodes, 1] + u[3 * nodes + 1]
    z = mesh.nodes[nodes, 2] + u[3 * nodes + 2]
    gaps = np.array([indenter.gap(xi, yi, zi) for xi, yi, zi in zip(x, y, z)])
    keep = gaps > 0.0
    if previous is not None and hysteresis > 0.0 and np.size(previous):
        prev = np.isin(nodes, np.asarray(previous, dtype=int))
        keep = keep | (prev & (gaps > -hysteresis))
    return nodes[keep], gaps[keep]


def contact_state(
    mesh,
    u: np.ndarray,
    indenter,
    penalty,
    candidate_nodes: np.ndarray | None = None,
    nodal_areas: np.ndarray | None = None,
    previous: np.ndarray | None = None,
    hysteresis: float = 0.0,
) -> ContactState:
    """Assemble penalty contact force, tangent and energy at displacement ``u``.

    ``penalty`` is the nodal penalty stiffness [N/m]: either a scalar applied to
    every candidate node or an array aligned with ``candidate_nodes`` (the
    physically meaningful choice is ``beta * E * A_node / h`` per node, see
    :func:`tensorphysics.ice_indentation.indentation.penalty_stiffness`).
    ``nodal_areas`` (aligned with ``candidate_nodes``) are the lumped surface
    areas used for the contact area / mean pressure diagnostics.
    """
    nodes = (np.arange(mesh.n_nodes) if candidate_nodes is None
             else np.asarray(candidate_nodes, dtype=int))
    if nodal_areas is None:
        areas = np.ones(nodes.size)
    else:
        areas = np.asarray(nodal_areas, dtype=float)
        if areas.shape != nodes.shape:
            raise ValueError("nodal_areas must be aligned with candidate_nodes")
    if np.ndim(penalty) == 0:
        k_all = np.full(nodes.size, float(penalty))
    else:
        k_all = np.asarray(penalty, dtype=float)
        if k_all.shape != nodes.shape:
            raise ValueError("penalty must be a scalar or aligned with candidate_nodes")
    if np.any(k_all <= 0.0):
        raise ValueError("penalty stiffness must be positive")

    active, gaps = active_set(mesh, u, indenter, nodes, previous, hysteresis)
    idx = np.searchsorted(nodes, active)
    k_act = k_all[idx]
    a_act = areas[idx]

    n_dof = mesh.n_dof
    f = np.zeros(n_dof)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for node, gap, k in zip(active, gaps, k_act):
        # a penalty spring pushes only: clamp the (hysteretic) negative gaps
        f[3 * node + 2] -= k * max(gap, 0.0)
        x = mesh.nodes[node, 0] + u[3 * node + 0]
        y = mesh.nodes[node, 1] + u[3 * node + 1]
        grad = indenter.gap_gradient(x, y)
        for j in range(3):                       # dof of this node
            # d f_z / d u_j = -k * dg/du_j
            rows.append(3 * node + 2)
            cols.append(3 * node + j)
            vals.append(-k * grad[j])

    if rows:
        tangent = coo_matrix((vals, (rows, cols)), shape=(n_dof, n_dof)).tocsr()
    else:
        tangent = csr_matrix((n_dof, n_dof))
    total = float(-f[2::3].sum())
    energy = (float(0.5 * np.sum(k_act * np.maximum(gaps, 0.0) ** 2))
              if gaps.size else 0.0)
    return ContactState(
        active=active,
        gaps=gaps,
        nodal_force_z=f[2::3],
        force_vector=f,
        tangent=tangent,
        total_force=total,
        energy=energy,
        contact_area_actual=float(np.sum(a_act)),
        nodal_area=float(np.mean(areas)) if areas.size else 0.0,
    )
