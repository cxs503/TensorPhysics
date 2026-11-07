"""Planar polygon primitives and 2D/3D interface discretisations (IN50-N08).

The 1D mortar contract (:mod:`.core`, campaign node IN19-T2) is built on exact
interval overlaps.  This module supplies the geometric primitives that lift the
same contract to surfaces:

* convex polygon clipping (Sutherland--Hodgman) with exact-ish area/moment
  integrals.  The clip window must be convex; the subject polygon may be any
  simple polygon.  Overlaps between two convex partitions of a convex domain are
  therefore exact, which is what a mortar needs;
* :class:`PolygonMesh` -- a planar control-volume partition (FVM cells,
  triangulated panels) with shoelace measures and Simpson-free cell means
  (fan triangulation + a degree-5 Dunavant rule, exact for quadratics);
* :class:`SurfacePatchMesh` -- patches living on an axisymmetric surface
  ``r = r(z)``, described in the chart ``(u, v) = (theta, z)``.  The area
  element is ``J(u, v) = r(v) sqrt(1 + r'(v)^2)``; overlaps are clipped in the
  chart (valid because a single chart covers the surface) while every measure
  and moment is evaluated with the *true* Jacobian, so the 3D patch mapping is
  area-weighted in square metres, not in chart units.

Nothing here is a solver kernel: the module only measures geometry, which is
TensorPhysics' cross-solver contract responsibility.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

__all__ = [
    "PolygonMesh",
    "SurfacePatchMesh",
    "axisymmetric_mesh",
    "cartesian_mesh",
    "fan_mesh",
    "triangulated_mesh",
    "cell_gradients",
    "clip_convex",
    "convex_hull",
    "convex_intersection",
    "polygon_area",
    "polygon_centroid",
    "polygon_integral",
    "triangulate",
]

_EPS = 1e-14

#: Degree-5 Dunavant rule on the reference triangle (barycentric points, weights
#: normalised so that ``sum(weights) = 1`` -> integral = area * weighted mean).
_TRI_BARY = np.array(
    [
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        [0.05971587178976982, 0.47014206410511505, 0.47014206410511505],
        [0.47014206410511505, 0.05971587178976982, 0.47014206410511505],
        [0.47014206410511505, 0.47014206410511505, 0.05971587178976982],
        [0.79742698535308729, 0.10128650732345633, 0.10128650732345633],
        [0.10128650732345633, 0.79742698535308729, 0.10128650732345633],
        [0.10128650732345633, 0.10128650732345633, 0.79742698535308729],
    ]
)
_TRI_W = np.array(
    [
        0.225,
        0.13239415278850616,
        0.13239415278850616,
        0.13239415278850616,
        0.12593918054482717,
        0.12593918054482717,
        0.12593918054482717,
    ]
)


# ---------------------------------------------------------------------------
# planar primitives
# ---------------------------------------------------------------------------


def _as_polygon(poly: Sequence[Sequence[float]]) -> np.ndarray:
    p = np.asarray(poly, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or p.shape[0] < 3:
        raise ValueError("a polygon is an (k>=3, 2) array of vertices")
    return p


def _signed_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polygon_area(poly: Sequence[Sequence[float]]) -> float:
    """Shoelace area (orientation independent); an empty clip has zero area."""
    p = np.asarray(poly, dtype=float)
    if p.size < 6:
        return 0.0
    return abs(_signed_area(_as_polygon(p)))


def polygon_centroid(poly: Sequence[Sequence[float]]) -> np.ndarray:
    """Area centroid of a simple polygon (falls back to the vertex mean)."""
    p = np.asarray(poly, dtype=float)
    if p.size < 6:
        return np.zeros(2)
    p = _as_polygon(p)
    a = _signed_area(p)
    if abs(a) <= _EPS:
        return p.mean(axis=0)
    x, y = p[:, 0], p[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    return np.array([float(np.dot(x + xn, cross)), float(np.dot(y + yn, cross))]) / (6.0 * a)


def triangulate(poly: Sequence[Sequence[float]]) -> list[np.ndarray]:
    """Fan triangulation about vertex 0 (valid for convex polygons)."""
    p = _as_polygon(poly)
    return [np.array([p[0], p[k], p[k + 1]]) for k in range(1, p.shape[0] - 1)]


def _subdivide(tri: np.ndarray, level: int) -> list[np.ndarray]:
    """Midpoint (barycentric) subdivision of a triangle into ``4**level`` pieces."""
    if level <= 0:
        return [tri]
    a, b, c = tri[0], tri[1], tri[2]
    ab, bc, ca = 0.5 * (a + b), 0.5 * (b + c), 0.5 * (c + a)
    out: list[np.ndarray] = []
    for sub in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)):
        out.extend(_subdivide(np.array(sub), level - 1))
    return out


def polygon_integral(
    poly: Sequence[Sequence[float]],
    f: Callable[[np.ndarray], np.ndarray],
    weight: Callable[[np.ndarray], np.ndarray] | None = None,
    subdivisions: int = 0,
) -> np.ndarray:
    """``int_poly f(x) [weight(x)] dA`` with a degree-5 rule per fan triangle.

    ``f`` maps ``(m, 2)`` points to ``(m,)`` or ``(m, k)``; ``weight`` (the
    Jacobian for surface charts) maps points to ``(m,)`` and defaults to 1.
    ``subdivisions`` midpoint-refines each fan triangle first: the rule is then
    applied on pieces of size ``h / 2**subdivisions``, which is how the curved
    surface measures are driven to round-off (needed because a constant field
    must be conserved to ``1e-14``, i.e. the two sides' area measures must agree
    to that level).
    """
    p = _as_polygon(poly)
    total = None
    tris = []
    for tri in triangulate(p):
        tris.extend(_subdivide(tri, int(subdivisions)))
    for tri in tris:
        area = polygon_area(tri)
        if area <= _EPS:
            continue
        pts = _TRI_BARY @ tri
        vals = np.asarray(f(pts), dtype=float)
        if vals.ndim == 1:
            vals = vals[:, None]
        w = _TRI_W if weight is None else _TRI_W * np.asarray(weight(pts), dtype=float)
        contrib = area * (w @ vals)
        total = contrib if total is None else total + contrib
    if total is None:
        return 0.0
    total = np.asarray(total, dtype=float)
    return float(total[0]) if total.size == 1 else total


def _dedupe(poly: list[np.ndarray], tol: float = 1e-15) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for p in poly:
        if not out or np.linalg.norm(p - out[-1]) > tol:
            out.append(p)
    if len(out) > 1 and np.linalg.norm(out[0] - out[-1]) <= tol:
        out.pop()
    return out


def clip_convex(
    subject: Sequence[Sequence[float]],
    clipper: Sequence[Sequence[float]],
) -> np.ndarray:
    """Sutherland--Hodgman clip of ``subject`` by the *convex* ``clipper``.

    Returns the intersection polygon as an ``(k, 2)`` array (``k`` may be 0).
    Vertices closer than ``1e-15`` are merged so degenerate slivers do not
    corrupt the shoelace area.  The clipper is re-oriented counter-clockwise
    internally, so the caller may pass either orientation.
    """
    s = _as_polygon(subject)
    c = _as_polygon(clipper)
    if _signed_area(c) < 0.0:
        c = c[::-1].copy()
    out = [np.array(p, dtype=float) for p in s]
    nc = c.shape[0]
    for k in range(nc):
        if not out:
            return np.zeros((0, 2))
        a = c[k]
        b = c[(k + 1) % nc]
        e = b - a
        inp = out
        out = []
        prev = inp[-1]
        prev_side = e[0] * (prev[1] - a[1]) - e[1] * (prev[0] - a[0])
        for cur in inp:
            cur_side = e[0] * (cur[1] - a[1]) - e[1] * (cur[0] - a[0])
            if cur_side >= 0.0:
                if prev_side < 0.0:
                    t = prev_side / (prev_side - cur_side)
                    out.append(prev + t * (cur - prev))
                out.append(cur)
            elif prev_side >= 0.0:
                t = prev_side / (prev_side - cur_side)
                out.append(prev + t * (cur - prev))
            prev, prev_side = cur, cur_side
        out = _dedupe(out)
    if len(out) < 3:
        return np.zeros((0, 2))
    poly = np.array(out, dtype=float)
    return poly if polygon_area(poly) > _EPS else np.zeros((0, 2))


def convex_hull(points: Sequence[Sequence[float]]) -> np.ndarray:
    """Monotone-chain convex hull (counter-clockwise, de-duplicated)."""
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 3:
        return pts.reshape(-1, 2)
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]
    uniq = [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(p - uniq[-1]) > 1e-14:
            uniq.append(p)
    if len(uniq) < 3:
        return np.array(uniq)

    def half(seq):
        out: list[np.ndarray] = []
        for p in seq:
            while len(out) >= 2:
                o, a = out[-2], out[-1]
                if (a[0] - o[0]) * (p[1] - o[1]) - (a[1] - o[1]) * (p[0] - o[0]) <= 0.0:
                    out.pop()
                else:
                    break
            out.append(p)
        return out

    lower = half(uniq)
    upper = half(uniq[::-1])
    hull = np.array(lower[:-1] + upper[:-1])
    return hull if hull.shape[0] >= 3 else np.zeros((0, 2))


def _segments_cross(p1, p2, q1, q2) -> np.ndarray | None:
    r = p2 - p1
    s = q2 - q1
    den = r[0] * s[1] - r[1] * s[0]
    if abs(den) <= 1e-15:
        return None
    t = ((q1[0] - p1[0]) * s[1] - (q1[1] - p1[1]) * s[0]) / den
    u = ((q1[0] - p1[0]) * r[1] - (q1[1] - p1[1]) * r[0]) / den
    if -1e-12 <= t <= 1.0 + 1e-12 and -1e-12 <= u <= 1.0 + 1e-12:
        return p1 + t * r
    return None


def _inside_convex(p: np.ndarray, poly: np.ndarray) -> bool:
    n = poly.shape[0]
    for k in range(n):
        a, b = poly[k], poly[(k + 1) % n]
        if (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) < -1e-12:
            return False
    return True


def convex_intersection(
    subject: Sequence[Sequence[float]],
    clipper: Sequence[Sequence[float]],
) -> np.ndarray:
    """Convex-polygon intersection by a **second, independent** construction.

    Instead of the Sutherland--Hodgman walk :func:`clip_convex` performs, this
    gathers (i) subject vertices inside the clipper, (ii) clipper vertices inside
    the subject and (iii) all edge--edge crossings, then takes their convex hull.
    For convex inputs the two algorithms must agree to round-off, which is what
    the clipping-area gate measures: an agreement test between two independent
    implementations plus an analytic partition closure, not a self-comparison.
    """
    s = _as_polygon(subject)
    c = _as_polygon(clipper)
    if _signed_area(c) < 0:
        c = c[::-1].copy()
    pts: list[np.ndarray] = [p for p in s if _inside_convex(p, c)]
    pts.extend(p for p in c if _inside_convex(p, s))
    ns, nc = s.shape[0], c.shape[0]
    for i in range(ns):
        for j in range(nc):
            hit = _segments_cross(s[i], s[(i + 1) % ns], c[j], c[(j + 1) % nc])
            if hit is not None:
                pts.append(hit)
    if len(pts) < 3:
        return np.zeros((0, 2))
    hull = convex_hull(pts)
    return hull if hull.shape[0] >= 3 and polygon_area(hull) > _EPS else np.zeros((0, 2))


def cell_gradients(
    centers: np.ndarray,
    values: np.ndarray,
    n_neighbours: int = 9,
) -> np.ndarray:
    """Weighted least-squares cell gradient from cell means.

    For a cell mean of a linear field the value at the centroid *is* the cell
    mean, so the fit is exact (to round-off) on any non-degenerate stencil; this
    is what makes the second-order mortar reproduce linear data.  The stencil is
    the ``n_neighbours`` nearest centroids with a Gaussian weight, i.e. local,
    so the operator stays sparse in structure.
    """
    xc = np.asarray(centers, dtype=float)
    u = np.asarray(values, dtype=float)
    n, dim = xc.shape
    g = np.zeros((n, dim))
    if n <= dim:
        return g
    for i in range(n):
        d = xc - xc[i]
        r = np.linalg.norm(d, axis=1)
        order = np.argsort(r)
        idx = [k for k in order[1:n_neighbours + 1] if r[k] > 1e-13]
        if len(idx) < dim:
            continue
        idx = np.array(idx, dtype=int)
        h = float(r[idx[0]])
        w = np.exp(-(r[idx] / h) ** 2)
        A = d[idx]
        b = u[idx] - u[i]
        AtA = A.T @ (w[:, None] * A)
        ridge = 1e-12 * float(np.trace(AtA)) + 1e-300
        g[i] = np.linalg.solve(AtA + ridge * np.eye(dim), A.T @ (w * b))
    return g


# ---------------------------------------------------------------------------
# 2D control-volume partition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolygonMesh:
    """A planar partition into convex polygons (interface side of a solver)."""

    polygons: tuple[np.ndarray, ...]
    label: str = "polygons"

    def __post_init__(self) -> None:
        polys = tuple(_as_polygon(p) for p in self.polygons)
        if not polys:
            raise ValueError("PolygonMesh needs at least one polygon")
        object.__setattr__(self, "polygons", polys)
        object.__setattr__(self, "_areas", np.array([polygon_area(p) for p in polys]))
        object.__setattr__(self, "_centroids",
                           np.array([polygon_centroid(p) for p in polys]))

    @property
    def n(self) -> int:
        return len(self.polygons)

    @property
    def areas(self) -> np.ndarray:
        return self._areas

    @property
    def measures(self) -> np.ndarray:
        return self._areas

    @property
    def centroids(self) -> np.ndarray:
        return self._centroids

    @property
    def centers(self) -> np.ndarray:
        return self.centroids

    def cell_averages(self, f: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """Cell mean ``(1/|P|) int_P f dA`` with a degree-5 rule."""
        vals = []
        for poly in self.polygons:
            a = polygon_area(poly)
            vals.append(np.asarray(polygon_integral(poly, f)) / a)
        out = np.array(vals)
        return out[:, 0] if out.ndim == 2 and out.shape[1] == 1 else out

    def __len__(self) -> int:
        return self.n


def cartesian_mesh(nx: int, ny: int, lo=(0.0, 0.0), hi=(1.0, 1.0),
                   label: str = "cartesian") -> PolygonMesh:
    """Axis-aligned ``nx x ny`` partition of the rectangle ``[lo, hi]``."""
    x = np.linspace(lo[0], hi[0], nx + 1)
    y = np.linspace(lo[1], hi[1], ny + 1)
    polys = []
    for i in range(nx):
        for j in range(ny):
            polys.append(np.array([[x[i], y[j]], [x[i + 1], y[j]],
                                   [x[i + 1], y[j + 1]], [x[i], y[j + 1]]]))
    return PolygonMesh(tuple(polys), label=f"{label}[{nx}x{ny}]")


def triangulated_mesh(nx: int, ny: int, lo=(0.0, 0.0), hi=(1.0, 1.0),
                      flip: bool = False, label: str = "triangles") -> PolygonMesh:
    """``nx x ny`` cells each split into two triangles (non-axis-aligned overlaps)."""
    base = cartesian_mesh(nx, ny, lo, hi, label=label)
    polys = []
    for quad in base.polygons:
        if flip:
            polys.append(np.array([quad[0], quad[1], quad[3]]))
            polys.append(np.array([quad[1], quad[2], quad[3]]))
        else:
            polys.append(np.array([quad[0], quad[1], quad[2]]))
            polys.append(np.array([quad[0], quad[2], quad[3]]))
    return PolygonMesh(tuple(polys), label=f"{label}[{nx}x{ny}]")


def fan_mesh(n_rays: int, n_rings: int, lo=(0.0, 0.0), hi=(1.0, 1.0),
             label: str = "fan") -> PolygonMesh:
    """Square split into ``n_rays`` wedges about the centre, each ``n_rings`` deep."""
    c = 0.5 * (np.asarray(lo, float) + np.asarray(hi, float))
    half = 0.5 * (np.asarray(hi, float) - np.asarray(lo, float))
    polys = []
    for r in range(n_rings):
        r0 = (hi[0] - lo[0]) * 0.5 * (r / n_rings) * 1.5
        r1 = (hi[0] - lo[0]) * 0.5 * (r + 1) / n_rings * 1.5
        for k in range(n_rays):
            t0 = 2.0 * np.pi * k / n_rays
            t1 = 2.0 * np.pi * (k + 1) / n_rays
            p = [c + r0 * np.array([np.cos(t0), np.sin(t0)]),
                 c + r1 * np.array([np.cos(t0), np.sin(t0)]),
                 c + r1 * np.array([np.cos(t1), np.sin(t1)]),
                 c + r0 * np.array([np.cos(t1), np.sin(t1)])]
            p = np.array(p)
            # keep only the part inside the rectangle
            clipped = clip_convex(p, np.array([lo, [hi[0], lo[1]], hi, [lo[0], hi[1]]], float))
            if len(clipped) >= 3 and polygon_area(clipped) > 1e-13:
                polys.append(clipped)
    return PolygonMesh(tuple(polys), label=f"{label}[{n_rays}x{n_rings}]")


# ---------------------------------------------------------------------------
# 3D surface patches on an axisymmetric surface r = r(z)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SurfacePatchMesh:
    """Patches on the axisymmetric surface ``r = radius(v)`` (``v = z``).

    ``patches`` are polygons in the chart ``(u, v) = (theta, z)``; the surface
    measure is ``J = r(v) sqrt(1 + r'(v)^2)``, so areas, moments and cell means
    are physical (m^2) quantities.
    """

    patches: tuple[np.ndarray, ...]
    radius: Callable[[np.ndarray], np.ndarray]
    drdv: Callable[[np.ndarray], np.ndarray]
    label: str = "surface-patches"
    subdivisions: int = 1

    def __post_init__(self) -> None:
        polys = tuple(_as_polygon(p) for p in self.patches)
        if not polys:
            raise ValueError("SurfacePatchMesh needs at least one patch")
        object.__setattr__(self, "patches", polys)
        areas = np.array([self.measure(p) for p in polys])
        object.__setattr__(self, "_areas", areas)
        object.__setattr__(self, "_centroids",
                           np.array([polygon_centroid(p) for p in polys]))

    # -- measure ---------------------------------------------------------
    def jacobian(self, uv: np.ndarray) -> np.ndarray:
        v = np.asarray(uv, dtype=float)[:, 1]
        r = np.asarray(self.radius(v), dtype=float)
        dr = np.asarray(self.drdv(v), dtype=float)
        return r * np.sqrt(1.0 + dr * dr)

    @property
    def n(self) -> int:
        return len(self.patches)

    @property
    def chart_areas(self) -> np.ndarray:
        return np.array([polygon_area(p) for p in self.patches])

    @property
    def areas(self) -> np.ndarray:
        return np.array([self.measure(p) for p in self.patches])

    def measure(self, poly: np.ndarray) -> float:
        """Surface area of a chart polygon, ``int_P J du dv`` (refined rule)."""
        return float(np.asarray(polygon_integral(
            poly, lambda q: np.ones(q.shape[0]), weight=self.jacobian,
            subdivisions=self.subdivisions)))

    @property
    def measures(self) -> np.ndarray:
        return self.areas

    @property
    def centroids(self) -> np.ndarray:
        return np.array([polygon_centroid(p) for p in self.patches])

    @property
    def centers(self) -> np.ndarray:
        return self.centroids

    def surface_point(self, uv: np.ndarray) -> np.ndarray:
        """3D coordinates ``(x, y, z)`` of chart points ``(theta, z)``."""
        uv = np.asarray(uv, dtype=float)
        r = np.asarray(self.radius(uv[:, 1]), dtype=float)
        return np.column_stack([r * np.cos(uv[:, 0]), r * np.sin(uv[:, 0]), uv[:, 1]])

    def area_moment(self, poly: np.ndarray, center: np.ndarray) -> np.ndarray:
        """``int_P (chi - center) J dA`` in chart coordinates."""
        c = np.asarray(center, dtype=float)
        return np.asarray(polygon_integral(poly, lambda q: q - c, weight=self.jacobian,
                                          subdivisions=self.subdivisions))

    def cell_averages(self, f: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        vals = []
        for k, poly in enumerate(self.patches):
            a = float(self._areas[k])
            vals.append(np.asarray(polygon_integral(poly, f, weight=self.jacobian,
                                                    subdivisions=self.subdivisions)) / a)
        out = np.array(vals)
        return out[:, 0] if out.ndim == 2 and out.shape[1] == 1 else out

    def __len__(self) -> int:
        return self.n


def axisymmetric_mesh(
    n_theta: int,
    n_z: int,
    radius: Callable[[np.ndarray], np.ndarray],
    drdv: Callable[[np.ndarray], np.ndarray],
    z_bounds: tuple[float, float] = (0.0, 1.0),
    theta0: float = 0.0,
    theta_span: float = 2.0 * np.pi,
    label: str = "axisym",
    subdivisions: int = 1,
) -> SurfacePatchMesh:
    """Structured ``n_theta x n_z`` chart mesh on ``r = radius(z)``.

    ``theta_span`` defaults to a full ring; a partial span is useful for the
    linear patch test, where a periodic seam would otherwise alias a jump.
    """
    th = np.linspace(theta0, theta0 + theta_span, n_theta + 1)
    zz = np.linspace(z_bounds[0], z_bounds[1], n_z + 1)
    polys = []
    for i in range(n_theta):
        for k in range(n_z):
            polys.append(np.array([[th[i], zz[k]], [th[i + 1], zz[k]],
                                   [th[i + 1], zz[k + 1]], [th[i], zz[k + 1]]]))
    return SurfacePatchMesh(tuple(polys), radius, drdv,
                            label=f"{label}[{n_theta}x{n_z}]", subdivisions=subdivisions)