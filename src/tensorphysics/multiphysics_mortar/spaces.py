"""P1 trace spaces on a coupling interface (partition-of-unity by construction).

A *trace space* is the space of interface fields that one solver exposes to the
other.  Two geometries are supported:

``LineTraceSpace``
    1D interface curve ``xi in [a, b]`` with ``n`` piecewise-linear hat basis
    functions.  A ``metric`` callable supplies ``dGamma/dxi`` (arc-length
    Jacobian of a curved interface), so ``Linear``-element physics on a curved
    FSI panel is integrated with the true measure.
``TriTraceSpace``
    2D interface patch described in a chart ``(u, v)`` (e.g. ``(theta, z)`` on a
    surface of revolution) triangulated into ``n_tri`` triangles with P1 hats on
    the deduplicated vertices.  The ``metric`` callable supplies the surface
    Jacobian ``J = r sqrt(1 + r'^2)``, so every integral below is a physical
    area integral in m^2.

Both spaces are **partitions of unity** (``sum_i phi_i = 1`` everywhere).  This is
the property the conservative mortar transfer relies on (see
:mod:`tensorphysics.multiphysics_mortar.assembly`); the module exposes
``partition_of_unity_error`` so the property is measured, not assumed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from ..interface_operators.polygons import polygon_area, polygon_integral

__all__ = [
    "GAUSS5_W",
    "GAUSS5_X",
    "LineTraceSpace",
    "TriTraceSpace",
    "chart_metric",
    "cosine_nodes",
    "structured_triangles",
]

# 5-point Gauss-Legendre on [-1, 1] (exact for polynomials of degree <= 9).
GAUSS5_X = np.array([-0.9061798459386640, -0.5384693101056831, 0.0,
                     0.5384693101056831, 0.9061798459386640])
GAUSS5_W = np.array([0.2369268850561891, 0.4786286704993665,
                     0.5688888888888889, 0.4786286704993665,
                     0.2369268850561891])

_EPS = 1e-15


def chart_metric(radius: Callable[[np.ndarray], np.ndarray],
                 drdv: Callable[[np.ndarray], np.ndarray]
                 ) -> Callable[[np.ndarray], np.ndarray]:
    """Surface Jacobian ``J = r(v) sqrt(1 + r'(v)^2)`` on a revolution patch.

    ``radius``/``drdv`` act on the axial chart coordinate ``v`` (column 1 of the
    ``(m, 2)`` point array).  For a *conical* patch ``r = r0 - gamma v`` the
    Jacobian is affine in ``v``, hence ``J x (P1 products)`` stays polynomial and
    the degree-5 rules used everywhere in this package are exact -- that is the
    configuration used by the round-off-level acceptance cases.
    """

    def metric(pts: np.ndarray) -> np.ndarray:
        u = np.asarray(pts, dtype=float).reshape(-1, 2)
        r = np.asarray(radius(u[:, 1]), dtype=float)
        dr = np.asarray(drdv(u[:, 1]), dtype=float)
        return r * np.sqrt(1.0 + dr * dr)

    return metric


def cosine_nodes(n: int, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """``n + 1`` nodes clustered at both ends (structure-mesh style grading)."""
    s = np.linspace(0.0, 1.0, n + 1)
    return lo + (hi - lo) * 0.5 * (1.0 - np.cos(np.pi * s))


def structured_triangles(nu: int, nv: int, lo: Sequence[float] = (0.0, 0.0),
                         hi: Sequence[float] = (1.0, 1.0), flip: bool = False
                         ) -> np.ndarray:
    """Structured triangulation of a chart rectangle, ``(n_tri, 3, 2)``.

    ``flip`` selects the other diagonal, so two meshes with different ``nu/nv``
    *and* different ``flip`` have crossing diagonals -- i.e. they are genuinely
    non-matching (no pair of cells shares an edge), which is what the mortar
    overlap geometry has to handle.
    """
    u = np.linspace(lo[0], hi[0], nu + 1)
    v = np.linspace(lo[1], hi[1], nv + 1)
    tris = []
    for i in range(nu):
        for j in range(nv):
            p00 = (u[i], v[j])
            p10 = (u[i + 1], v[j])
            p11 = (u[i + 1], v[j + 1])
            p01 = (u[i], v[j + 1])
            if flip:
                tris.append(np.array([p00, p10, p01]))
                tris.append(np.array([p10, p11, p01]))
            else:
                tris.append(np.array([p00, p10, p11]))
                tris.append(np.array([p00, p11, p01]))
    return np.array(tris, dtype=float)


# ---------------------------------------------------------------------------
# 1D interface curve
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LineTraceSpace:
    """P1 hats on a (possibly graded) 1D interface curve.

    ``nodes`` are the ``n + 1`` curve parameters in increasing order; the
    parameter is the coordinate that ``metric`` differentiates against, so for a
    curved interface pass the *abscissa* nodes and ``metric = dGamma/dxi``.
    """

    nodes: np.ndarray
    metric: Callable[[np.ndarray], np.ndarray] | None = None
    label: str = "line"

    def __post_init__(self) -> None:
        nodes = np.asarray(self.nodes, dtype=float).ravel()
        if nodes.size < 2:
            raise ValueError("LineTraceSpace needs at least two nodes")
        if not np.all(np.diff(nodes) > 0.0):
            raise ValueError("LineTraceSpace nodes must be strictly increasing")
        object.__setattr__(self, "nodes", nodes)

    # -- topology --------------------------------------------------------
    @property
    def n(self) -> int:
        return self.nodes.size

    @property
    def n_cells(self) -> int:
        return self.nodes.size - 1

    @property
    def cells(self) -> np.ndarray:
        return np.column_stack([self.nodes[:-1], self.nodes[1:]])

    @property
    def bounds(self) -> tuple[float, float]:
        return float(self.nodes[0]), float(self.nodes[-1])

    def __len__(self) -> int:
        return self.n

    # -- quadrature helpers ---------------------------------------------
    def weights(self, xi: np.ndarray) -> np.ndarray:
        """``dGamma/dxi`` at ``xi`` (unit metric where none was given)."""
        xi = np.asarray(xi, dtype=float)
        if self.metric is None:
            return np.ones_like(xi)
        w = np.asarray(self.metric(xi), dtype=float)
        return np.full(xi.shape, float(w)) if w.ndim == 0 else w

    def locate(self, xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Cell index and local coordinate ``t in [0, 1]`` of each point."""
        xi = np.asarray(xi, dtype=float)
        idx = np.clip(np.searchsorted(self.nodes, xi, side="right") - 1,
                      0, self.n_cells - 1)
        a = self.nodes[idx]
        b = self.nodes[idx + 1]
        return idx, (xi - a) / (b - a)

    def local_ids(self, cell: int) -> np.ndarray:
        return np.array([cell, cell + 1], dtype=int)

    def local_values(self, t: np.ndarray) -> np.ndarray:
        """Local hat values ``(m, 2)`` at local coordinates ``t``."""
        t = np.asarray(t, dtype=float)
        return np.stack([1.0 - t, t], axis=-1)

    def basis_dense(self, xi: np.ndarray) -> np.ndarray:
        """Dense basis matrix ``(len(xi), n)``."""
        xi = np.asarray(xi, dtype=float).ravel()
        idx, t = self.locate(xi)
        out = np.zeros((xi.size, self.n))
        rows = np.arange(xi.size)
        out[rows, idx] += 1.0 - t
        out[rows, idx + 1] += t
        return out

    def interpolate(self, values: np.ndarray, xi: np.ndarray) -> np.ndarray:
        """Evaluate the P1 interpolant of nodal ``values`` at ``xi``."""
        vals = np.asarray(values, dtype=float)
        two_d = vals.ndim > 1
        b = self.basis_dense(xi)
        out = b @ (vals if two_d else vals[:, None])
        return out if two_d else out[:, 0]

    # -- measures (computed independently of the coupling assembly) ------
    def _cell_quadrature(self, cell: int) -> tuple[np.ndarray, np.ndarray]:
        a, b = self.nodes[cell], self.nodes[cell + 1]
        xi = 0.5 * (a + b) + 0.5 * (b - a) * GAUSS5_X
        w = 0.5 * (b - a) * GAUSS5_W * self.weights(xi)
        return xi, w

    def measures(self) -> np.ndarray:
        """``int_Gamma phi_i dGamma`` per basis function (lumped measure)."""
        out = np.zeros(self.n)
        for cell in range(self.n_cells):
            xi, w = self._cell_quadrature(cell)
            out[self.local_ids(cell)] += w @ self.local_values(
                self.locate(xi)[1])
        return out

    def centers(self) -> np.ndarray:
        """Measure-weighted centroid of each basis support."""
        num = np.zeros(self.n)
        den = np.zeros(self.n)
        for cell in range(self.n_cells):
            xi, w = self._cell_quadrature(cell)
            phi = self.local_values(self.locate(xi)[1])
            ids = self.local_ids(cell)
            num[ids] += (w * xi) @ phi
            den[ids] += w @ phi
        if np.any(np.abs(den) < _EPS):
            raise ValueError("degenerate basis support")
        return num / den

    def partition_of_unity_error(self) -> float:
        """``max |sum_i phi_i - 1|`` over a dense sample of the interface."""
        xi = np.linspace(*self.bounds, 2001)
        return float(np.max(np.abs(self.basis_dense(xi).sum(axis=1) - 1.0)))


# ---------------------------------------------------------------------------
# 2D interface patch in a chart
# ---------------------------------------------------------------------------


def _dedupe_nodes(triangles: np.ndarray, tol: float = 1e-9
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Global vertex ids (``(n_tri, 3)``) and coordinates for a triangle soup."""
    key: dict[tuple[int, int], int] = {}
    coords: list[np.ndarray] = []
    ids = np.zeros((triangles.shape[0], 3), dtype=int)
    for t, tri in enumerate(triangles):
        for l in range(3):
            p = np.asarray(tri[l], dtype=float)
            k = (int(round(p[0] / tol)), int(round(p[1] / tol)))
            got = key.get(k)
            if got is None:
                got = len(coords)
                key[k] = got
                coords.append(p)
            ids[t, l] = got
    return ids, np.array(coords, dtype=float)


@dataclass(frozen=True)
class TriTraceSpace:
    """P1 hats on a triangulated interface patch in a chart ``(u, v)``."""

    triangles: np.ndarray
    metric: Callable[[np.ndarray], np.ndarray] | None = None
    subdivisions: int = 0
    label: str = "chart"

    def __post_init__(self) -> None:
        tris = np.asarray(self.triangles, dtype=float)
        if tris.ndim != 3 or tris.shape[1:] != (3, 2):
            raise ValueError("TriTraceSpace expects an (n_tri, 3, 2) array")
        areas = np.array([polygon_area(t) for t in tris])
        if np.any(areas <= 0.0):
            raise ValueError("TriTraceSpace has degenerate / inverted triangles")
        object.__setattr__(self, "triangles", tris)
        ids, coords = _dedupe_nodes(tris)
        object.__setattr__(self, "_node_ids", ids)
        object.__setattr__(self, "_node_coords", coords)

    # -- topology --------------------------------------------------------
    @property
    def node_ids(self) -> np.ndarray:
        return self._node_ids

    @property
    def node_coords(self) -> np.ndarray:
        return self._node_coords

    @property
    def n_tri(self) -> int:
        return self.triangles.shape[0]

    @property
    def n(self) -> int:
        return self._node_coords.shape[0]

    def __len__(self) -> int:
        return self.n

    # -- quadrature helpers ---------------------------------------------
    def weights(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=float)
        if self.metric is None:
            return np.ones(pts.shape[0])
        return np.asarray(self.metric(pts), dtype=float)

    @staticmethod
    def barycentric(tri: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Area coordinates of ``pts`` w.r.t. ``tri`` (``(m, 3)``)."""
        a, b, c = (np.asarray(tri[0], float), np.asarray(tri[1], float),
                   np.asarray(tri[2], float))
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)

        def crs(u, v):
            return u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

        area2 = crs((b - a)[None, :], (c - a)[None, :])[0]
        l0 = crs(b - pts, c - pts) / area2
        l1 = crs(c - pts, a - pts) / area2
        l2 = crs(a - pts, b - pts) / area2
        return np.stack([l0, l1, l2], axis=-1)

    def basis_on(self, tri_index: int, pts: np.ndarray) -> np.ndarray:
        return self.barycentric(self.triangles[tri_index], pts)

    def _cell_integral(self, tri_index: int,
                       f: Callable[[np.ndarray], np.ndarray]):
        return polygon_integral(self.triangles[tri_index], f,
                               weight=self.weights,
                               subdivisions=self.subdivisions)

    # -- measures --------------------------------------------------------
    def areas(self) -> np.ndarray:
        """Physical area ``int_T J dA_chart`` of each triangle."""
        return np.array([float(self._cell_integral(t, lambda q: np.ones(q.shape[0])))
                         for t in range(self.n_tri)])

    def measures(self) -> np.ndarray:
        """``int_Gamma phi_i dGamma`` per vertex basis function."""
        out = np.zeros(self.n)
        for t in range(self.n_tri):
            vals = self._cell_integral(t, lambda q, t=t: self.basis_on(t, q))
            out[self.node_ids[t]] += vals
        return out

    def centers(self) -> np.ndarray:
        """Measure-weighted centroid (chart coords) of each basis support."""
        num = np.zeros((self.n, 2))
        den = np.zeros(self.n)
        for t in range(self.n_tri):
            ids = self.node_ids[t]
            den[ids] += self._cell_integral(
                t, lambda q, t=t: self.basis_on(t, q))
            num[ids, 0] += self._cell_integral(
                t, lambda q, t=t: self.basis_on(t, q) * q[:, 0:1])
            num[ids, 1] += self._cell_integral(
                t, lambda q, t=t: self.basis_on(t, q) * q[:, 1:2])
        if np.any(np.abs(den) < _EPS):
            raise ValueError("degenerate basis support")
        return num / den[:, None]

    def locate(self, pts: np.ndarray) -> np.ndarray:
        """Index of the triangle containing each point (``-1`` if outside)."""
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        lo = self.triangles.min(axis=1)
        hi = self.triangles.max(axis=1)
        out = -np.ones(pts.shape[0], dtype=int)
        for k, p in enumerate(pts):
            cand = np.nonzero(np.all((p >= lo - 1e-12) & (p <= hi + 1e-12),
                                     axis=1))[0]
            for t in cand:
                if np.all(self.basis_on(int(t), p[None, :])[0] >= -1e-10):
                    out[k] = int(t)
                    break
        return out

    def interpolate(self, values: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Evaluate the P1 interpolant of nodal ``values`` at chart points."""
        vals = np.asarray(values, dtype=float)
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        two_d = vals.ndim > 1
        vals = vals if two_d else vals[:, None]
        tris = self.locate(pts)
        out = np.full((pts.shape[0], vals.shape[1]), np.nan)
        for k, t in enumerate(tris):
            if t < 0:
                continue
            lam = self.basis_on(int(t), pts[k][None, :])[0]
            out[k] = lam @ vals[self.node_ids[int(t)]]
        return out if two_d else out[:, 0]

    def partition_of_unity_error(self) -> float:
        """``max |sum_i phi_i - 1|`` over all triangle centroids and vertices."""
        pts = np.vstack([self.node_coords,
                         self.triangles.mean(axis=1),
                         self.triangles.reshape(-1, 2)])
        err = 0.0
        for t in range(self.n_tri):
            err = max(err, float(np.max(np.abs(
                self.basis_on(t, pts).sum(axis=1) - 1.0))))
        return err