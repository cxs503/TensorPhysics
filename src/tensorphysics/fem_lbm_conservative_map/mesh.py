"""Unstructured P1 triangular FEM heat conduction (source side of the mapping).

Why this module exists
----------------------
The cross-solver mapping in this package is only testable if the *source* field is a
real finite-element solution on a genuinely non-conforming (unstructured) mesh:

* the mesh is a perturbed triangulation of the unit square -- node coordinates are
  jittered, every quad is split along a randomly chosen diagonal, so the element
  boundaries have no relation to the uniform LBM lattice used as the target;
* the field is the P1 (linear triangle) solution of ``-div(k grad T) = f`` with
  mixed Dirichlet / natural (adiabatic) boundary conditions, assembled with the
  exact constant element gradients and a degree-2 source quadrature.

Everything downstream (``remap``) only sees ``nodes``/``tri`` + nodal values, so the
mesh object is the whole interface.

Ownership: FEM discretisation of the heat equation only -- no mapping logic here
(that lives in :mod:`~tensorphysics.fem_lbm_conservative_map.remap`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = [
    "TriMesh",
    "make_perturbed_tri_mesh",
    "make_boundary_mesh",
    "solve_heat_p1",
    "manufactured_sine",
]


# ---------------------------------------------------------------------------
# mesh container
# ---------------------------------------------------------------------------
@dataclass
class TriMesh:
    """Straight-sided triangular mesh (CCW elements), P1 unknowns at the nodes."""

    nodes: np.ndarray  # (N, 2) float
    tri: np.ndarray  # (M, 3) int, counter-clockwise
    meta: dict = field(default_factory=dict)

    # -- geometry ----------------------------------------------------------
    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def n_elements(self) -> int:
        return int(self.tri.shape[0])

    def signed_areas(self) -> np.ndarray:
        p0 = self.nodes[self.tri[:, 0]]
        p1 = self.nodes[self.tri[:, 1]]
        p2 = self.nodes[self.tri[:, 2]]
        return 0.5 * ((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
                      - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1]))

    def areas(self) -> np.ndarray:
        return np.abs(self.signed_areas())

    def centroids(self) -> np.ndarray:
        return self.nodes[self.tri].mean(axis=1)

    def gradients(self) -> np.ndarray:
        """Constant basis gradients, shape (M, 3, 2).

        ``grad[i, a]`` is the gradient of the P1 shape function of local node ``a``
        inside element ``i``:  ``grad N_a = (b_a, c_a) / (2A)``.
        """
        xy = self.nodes[self.tri]  # (M,3,2)
        x, y = xy[:, :, 0], xy[:, :, 1]
        two_a = self.signed_areas() * 2.0
        b = np.stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]], axis=1)
        c = np.stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]], axis=1)
        return np.stack([b / two_a[:, None], c / two_a[:, None]], axis=2)

    def element_gradient_of_field(self, phi: np.ndarray) -> np.ndarray:
        """``grad(phi)`` per element, exact for P1 fields -> shape (M, 2)."""
        g = self.gradients()  # (M,3,2)
        return np.einsum("mae,ma->me", g, np.asarray(phi, dtype=float)[self.tri])

    # -- data transfer helpers --------------------------------------------
    def cell_values(self, phi: np.ndarray) -> np.ndarray:
        """Element mean of a P1 field:  ``|T|^-1 int_T phi dA = mean of 3 nodes``."""
        return np.asarray(phi, dtype=float)[self.tri].mean(axis=1)

    def cell_integrals(self, phi: np.ndarray) -> np.ndarray:
        """Exact ``int_T phi dA`` for a P1 field."""
        return self.areas() * self.cell_values(phi)

    def boundary_nodes(self, tol: float = 1e-12) -> np.ndarray:
        """Node indices on the boundary of the axis-aligned bounding box."""
        xy = self.nodes
        lo = xy.min(axis=0)
        hi = xy.max(axis=0)
        on = (np.abs(xy[:, 0] - lo[0]) < tol) | (np.abs(xy[:, 0] - hi[0]) < tol) \
            | (np.abs(xy[:, 1] - lo[1]) < tol) | (np.abs(xy[:, 1] - hi[1]) < tol)
        return np.nonzero(on)[0]

    def validate(self, expected_area: float | None = None) -> dict:
        """Geometry gate: positivity, area closure, non-degenerate shape."""
        a = self.signed_areas()
        areas = np.abs(a)
        total = float(areas.sum())
        q = {
            "n_nodes": self.n_nodes,
            "n_elements": self.n_elements,
            "n_inverted": int(np.count_nonzero(a <= 0.0)),
            "total_area": total,
            "min_area": float(areas.min()),
            "max_area": float(areas.max()),
            "area_ratio": float(areas.max() / max(areas.min(), 1e-300)),
        }
        if expected_area is not None:
            q["area_error_rel"] = abs(total - expected_area) / expected_area
        q["ok"] = bool(q["n_inverted"] == 0 and q["min_area"] > 0.0
                       and (expected_area is None or q["area_error_rel"] < 1e-12))
        return q

    # -- sampling ---------------------------------------------------------
    def eval_p1(self, phi: np.ndarray, pts: np.ndarray, tri_idx: np.ndarray) -> np.ndarray:
        """Evaluate the P1 field inside known elements (barycentric)."""
        phi = np.asarray(phi, dtype=float)
        xy = self.nodes[self.tri[tri_idx]]  # (P,3,2)
        val = phi[self.tri[tri_idx]]  # (P,3)
        lam, _ = _barycentric(xy, pts)
        return np.einsum("pa,pa->p", lam, val)


def _barycentric(tri_xy: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p0 = tri_xy[:, 0, :]
    e1 = tri_xy[:, 1, :] - p0
    e2 = tri_xy[:, 2, :] - p0
    rhs = pts - p0
    det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    l1 = (rhs[:, 0] * e2[:, 1] - rhs[:, 1] * e2[:, 0]) / det
    l2 = (e1[:, 0] * rhs[:, 1] - e1[:, 1] * rhs[:, 0]) / det
    lam = np.stack([1.0 - l1 - l2, l1, l2], axis=1)
    return lam, det


# ---------------------------------------------------------------------------
# mesh generation
# ---------------------------------------------------------------------------
def make_perturbed_tri_mesh(
    nx: int,
    ny: int = None,
    jitter: float = 0.25,
    seed: int = 20261110,
    domain: Sequence[float] = (0.0, 0.0, 1.0, 1.0),
    max_attempts: int = 4,
) -> TriMesh:
    """Unstructured triangulation of a rectangle by jittered quads.

    Interior nodes are displaced by a uniform random offset up to ``jitter * h``
    (h = target spacing); boundary nodes slide *along* their edge so the domain
    boundary stays exactly the rectangle.  Each quad is cut along a pseudo-random
    diagonal.  The result is irregular (no element is aligned with the Cartesian
    directions) but watertight, which is exactly the situation that makes
    source-to-lattice remapping non-trivial.
    """
    ny = nx if ny is None else ny
    # A jitter above ~0.25 h can make neighbouring quads overlap (the union then
    # exceeds the domain area); the geometry gate below catches it, and the builder
    # then retries with a milder perturbation instead of silently producing a
    # non-watertight mesh (records the applied value in ``meta``).
    for attempt in range(max_attempts):
        jit = jitter * (0.8 ** attempt)
        try:
            return _build_perturbed(nx, ny, jit, seed + 101 * attempt, domain)
        except ValueError:
            if attempt == max_attempts - 1:
                raise


def _build_perturbed(nx: int, ny: int, jitter: float, seed: int, domain) -> TriMesh:
    x0, y0, x1, y1 = domain
    hx = (x1 - x0) / nx
    hy = (y1 - y0) / ny
    rng = np.random.default_rng(seed)

    gx = np.linspace(x0, x1, nx + 1)
    gy = np.linspace(y0, y1, ny + 1)
    X, Y = np.meshgrid(gx, gy, indexing="xy")
    pts = np.stack([X.ravel(), Y.ravel()], axis=1).astype(float)

    idx = lambda i, j: j * (nx + 1) + i  # noqa: E731  (i along x, j along y)
    for j in range(ny + 1):
        for i in range(nx + 1):
            n = idx(i, j)
            dx = rng.uniform(-jitter, jitter) * hx
            dy = rng.uniform(-jitter, jitter) * hy
            on_x = (i == 0) or (i == nx)
            on_y = (j == 0) or (j == ny)
            if on_x and on_y:
                continue
            if on_y:  # slide along the horizontal edge
                dx = np.clip(pts[n, 0] + dx, gx[i] - 0.45 * hx, gx[i] + 0.45 * hx) - pts[n, 0]
                dy = 0.0
            elif on_x:  # slide along the vertical edge
                dy = np.clip(pts[n, 1] + dy, gy[j] - 0.45 * hy, gy[j] + 0.45 * hy) - pts[n, 1]
                dx = 0.0
            pts[n] += np.array([dx, dy])

    tris = []
    for j in range(ny):
        for i in range(nx):
            n00, n10, n11, n01 = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            if rng.random() < 0.5:
                tris.append((n00, n10, n11))
                tris.append((n00, n11, n01))
            else:
                tris.append((n00, n10, n01))
                tris.append((n10, n11, n01))
    tri = np.asarray(tris, dtype=int)

    mesh = TriMesh(nodes=pts, tri=_ensure_ccw(pts, tri),
                   meta={"jitter": float(jitter), "seed": int(seed), "nx": nx, "ny": ny})
    # geometry gate: the triangulation must cover exactly the rectangle
    q = mesh.validate(expected_area=(x1 - x0) * (y1 - y0))
    if not q["ok"]:
        raise ValueError(f"generated mesh failed the geometry gate: {q}")
    return mesh


def make_boundary_mesh(
    nx: int,
    ny: int = None,
    jitter: float = 0.0,
    seed: int = 7,
    domain: Sequence[float] = (0.0, 0.0, 1.0, 1.0),
) -> TriMesh:
    """Structured right-triangle mesh (``jitter=0``) -- the limit of the family."""
    return make_perturbed_tri_mesh(nx, ny, jitter=jitter, seed=seed, domain=domain)


def _ensure_ccw(nodes: np.ndarray, tri: np.ndarray) -> np.ndarray:
    p0, p1, p2 = nodes[tri[:, 0]], nodes[tri[:, 1]], nodes[tri[:, 2]]
    area2 = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) \
        - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1])
    flip = area2 < 0
    out = tri.copy()
    out[flip] = out[flip][:, [0, 2, 1]]
    return out


# ---------------------------------------------------------------------------
# P1 heat conduction solver
# ---------------------------------------------------------------------------
def solve_heat_p1(
    mesh: TriMesh,
    k: float = 1.0,
    source: Callable[[np.ndarray], np.ndarray] | None = None,
    dirichlet_mask: np.ndarray | None = None,
    dirichlet_values: np.ndarray | float | None = None,
) -> dict:
    """Solve ``-div(k grad T) = f`` on a P1 mesh, Dirichlet + natural BCs.

    Parameters
    ----------
    k : float
        Isotropic conductivity (constant).
    source : callable, optional
        ``f(xy) -> (P,)`` volumetric source, evaluated with a degree-2 rule
        (3 interior points per element, exact for quadratic sources).
    dirichlet_mask : (N,) bool, optional
        Nodes where the temperature is prescribed.  All other boundary nodes are
        natural (zero normal flux = adiabatic), which is what the manufactured
        and the Rayleigh-Benard conduction cases need.
    dirichlet_values : array or float
        Prescribed values on ``dirichlet_mask``.

    Returns
    -------
    dict with ``phi`` (nodal field), ``A`` (sparse stiffness), ``b``, plus the
    exact ``total_integral`` ``int_T phi dA`` and the assembly norm.
    """
    m = mesh.n_elements
    areas = mesh.areas()
    g = mesh.gradients()  # (M,3,2)

    # element stiffness Ke_ab = k * A * grad_a . grad_b
    ke = k * areas[:, None, None] * np.einsum("mae,mbe->mab", g, g)
    rows = np.repeat(mesh.tri, 3, axis=1).ravel()
    cols = np.tile(mesh.tri, (1, 3)).ravel()
    K = sp.coo_matrix((ke.ravel(), (rows, cols)),
                      shape=(mesh.n_nodes, mesh.n_nodes)).tocsr()

    b = np.zeros(mesh.n_nodes)
    if source is not None:
        # three-point (degree 2) rule: points at the element vertices' midpoints
        lam = np.array([[2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
                        [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
                        [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0]])
        xy = mesh.nodes[mesh.tri]  # (M,3,2)
        fq = np.zeros((m, 3))
        for q in range(3):
            xq = np.einsum("a,mad->md", lam[q], xy)
            fq[:, q] = np.asarray(source(xq), dtype=float)
        # weights already carry A/3; multiply by the shape value to get N_a
        b_elem = np.einsum("mq,qa->ma", areas[:, None] / 3.0 * fq, lam)
        np.add.at(b, mesh.tri, b_elem)

    free = np.ones(mesh.n_nodes, dtype=bool)
    if dirichlet_mask is not None:
        mask = np.asarray(dirichlet_mask, dtype=bool)
        vals = np.zeros(mesh.n_nodes)
        if np.isscalar(dirichlet_values):
            vals[mask] = float(dirichlet_values)
        else:
            vals[mask] = np.asarray(dirichlet_values, dtype=float)[mask]
        free = ~mask
        b = b - K @ vals
        K_ff = K[free][:, free]
        rhs = b[free]
        dvals = vals
    else:
        K_ff = K
        rhs = b
        dvals = np.zeros(mesh.n_nodes)

    sol = np.zeros(mesh.n_nodes)
    resid = 0.0
    if K_ff.shape[0] > 0:
        lu = spla.splu(K_ff.tocsc())
        sol[free] = lu.solve(rhs)
        resid = float(np.linalg.norm(K_ff @ sol[free] - rhs))
    phi = sol + dvals

    return {
        "phi": phi,
        "K": K,
        "b": b,
        "stiffness_nnz": int(K.nnz),
        "solve_residual_norm": resid,
        "n_free": int(free.sum()),
        "total_integral": float(mesh.cell_integrals(phi).sum()),
        "element_gradients": g,
    }


# ---------------------------------------------------------------------------
# manufactured reference solution
# ---------------------------------------------------------------------------
def manufactured_sine(mesh: TriMesh, k: float = 1.0) -> dict:
    """Manufactured solution ``T = sin(pi x) sin(pi y)`` (zero on the boundary).

    Corresponding source term ``f = 2 k pi^2 sin(pi x) sin(pi y)``.  The exact
    volume integral is ``(4/pi^2)`` and the exact volume-weighted mean is
    ``4/pi^2`` (unit square).
    """
    def exact(xy: np.ndarray) -> np.ndarray:
        xy = np.atleast_2d(np.asarray(xy, dtype=float))
        return np.sin(np.pi * xy[:, 0]) * np.sin(np.pi * xy[:, 1])

    def src(xy: np.ndarray) -> np.ndarray:
        return 2.0 * k * np.pi ** 2 * exact(xy)

    bmask = np.zeros(mesh.n_nodes, dtype=bool)
    bmask[mesh.boundary_nodes()] = True
    res = solve_heat_p1(mesh, k=k, source=src, dirichlet_mask=bmask, dirichlet_values=0.0)
    res["exact"] = exact
    res["exact_integral"] = 4.0 / np.pi ** 2
    res["exact_nodal"] = exact(mesh.nodes)
    return res


def manufactured_linear(mesh: TriMesh, k: float = 1.0,
                        t_bottom: float = 1.0, t_top: float = 0.0) -> dict:
    """Rayleigh-Benard conduction state ``T = t_bottom + (t_top-t_bottom) y``.

    Dirichlet on the horizontal walls only, natural (adiabatic) on the vertical
    walls.  A P1 method reproduces a linear field *exactly* (up to round-off),
    which makes this a discretisation check as well as the physically relevant
    initial condition for the LBM natural-convection run.
    """
    nodes = mesh.nodes
    lo, hi = nodes.min(axis=0), nodes.max(axis=0)
    on_horizontal = (np.abs(nodes[:, 1] - lo[1]) < 1e-12) | (np.abs(nodes[:, 1] - hi[1]) < 1e-12)
    vals = np.where(nodes[:, 1] < 0.5 * (lo[1] + hi[1]), t_bottom, t_top)

    def zero_source(xy):
        return np.zeros(np.atleast_2d(xy).shape[0])

    res = solve_heat_p1(mesh, k=k, source=zero_source,
                        dirichlet_mask=on_horizontal, dirichlet_values=vals)
    span = hi[1] - lo[1]
    res["exact"] = lambda xy: t_bottom + (t_top - t_bottom) * (np.atleast_2d(xy)[:, 1] - lo[1]) / span
    res["exact_integral"] = 0.5 * (t_bottom + t_top) * (hi[0] - lo[0]) * span
    res["exact_nodal"] = res["exact"](nodes)
    return res