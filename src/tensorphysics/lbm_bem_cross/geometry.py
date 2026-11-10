"""Unified geometry contract for the LBM <-> BEM cross-validation framework.

Both solvers must consume *bit-identical* surface data, otherwise inter-solver
discrepancies cannot be attributed to physics.  This module implements a
dual-format reader:

* **NURBS** — a rational quadratic B-spline representation of the 2-D body
  cross-section (for the circular cylinder: the standard 9-control-point
  rational quadratic circle), evaluated with the Cox-de Boor basis recursion and
  re-sampled at uniform arc length into a closed polyline;
* **STL** — the same body extruded into a prism and written as an ASCII STL
  (side quads split into triangles, triangular fan caps) with round-trip-exact
  floating point text, then parsed back into the closed polyline via the cap
  fan triangulation.

The recovered polylines must agree to machine precision (``max|Δvertex|`` and
``max|Δnormal|`` ``< 1e-9``), asserted by
``tests/test_lbm_bem_cross.py`` in the acceptance-criterion-4 test.

Everything is expressed in *physical* units with the body diameter equal to
1.0; the solvers non-dimensionalise internally.

Conventions
-----------
* vertices are ordered so that ``theta`` (see below) **increases** along the
  polyline; the loop is therefore traversed clockwise in the ``(x, y)`` plane
  as seen from ``+z`` and panel normals ``(-t_y, t_x)`` point outward;
* ``theta`` is measured from the upstream stagnation point at ``-x`` and grows
  towards ``+y``, matching the classical cylinder literature (Schlichting,
  *Boundary-Layer Theory*): ``theta=0`` front stagnation, ``theta=pi`` rear,
  ``theta in (0, pi)`` upper surface.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

__all__ = [
    "SurfaceGeometry",
    "nurbs_circle",
    "polyline_to_stl",
    "polyline_from_stl",
    "load_geometry",
    "geometry_contract_check",
]


# ---------------------------------------------------------------------------
# Surface container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SurfaceGeometry:
    """Closed 2-D body surface as a polyline (physical units).

    Attributes
    ----------
    x, y : ndarray (N,)
        Vertex coordinates, first vertex at the upstream stagnation point.
    nx, ny : ndarray (N,)
        Outward unit vertex normals.
    arclength : ndarray (N+1,)
        Cumulative arc length, ``arclength[0] == 0``, ``arclength[N] == P``.
    source_format : str
        ``"nurbs"`` or ``"stl"`` — provenance of the in-memory data.
    """

    x: np.ndarray
    y: np.ndarray
    nx: np.ndarray
    ny: np.ndarray
    arclength: np.ndarray
    source_format: str = "nurbs"

    @property
    def n(self) -> int:
        return int(self.x.size)

    @property
    def perimeter(self) -> float:
        return float(self.arclength[-1])

    @property
    def area(self) -> float:
        """Enclosed area by the shoelace formula."""
        x, y = self.x, self.y
        return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    @property
    def hydraulic_diameter(self) -> float:
        """Area-equivalent diameter ``sqrt(4A/pi)`` (geometry identity check)."""
        return float(math.sqrt(4.0 * self.area / math.pi))

    def theta(self) -> np.ndarray:
        """Angle from the front stagnation point (radians, 0..2pi, grows to +y)."""
        th = np.mod(np.arctan2(self.y, -self.x), 2.0 * np.pi)
        th[np.abs(th - 2.0 * np.pi) < 1e-9] = 0.0
        return th

    def segment_midpoints(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Panel midpoints and outward panel unit normals (for panel methods)."""
        x0, y0 = self.x, self.y
        x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
        xm, ym = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        tx, ty = x1 - x0, y1 - y0
        length = np.hypot(tx, ty)
        # outward normal for the increasing-theta (clockwise) traversal: +90 deg
        return xm, ym, np.stack([-ty / length, tx / length], axis=1)

    def segment_lengths(self) -> np.ndarray:
        return np.diff(self.arclength)

    def vertices(self) -> np.ndarray:
        return np.stack([self.x, self.y], axis=1)

    def normals(self) -> np.ndarray:
        return np.stack([self.nx, self.ny], axis=1)

    def to_dict(self) -> dict:
        return {
            "n_vertices": self.n,
            "area": self.area,
            "perimeter": self.perimeter,
            "hydraulic_diameter": self.hydraulic_diameter,
            "source_format": self.source_format,
        }


def _vertex_normals(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Outward vertex normals from adjacent-edge tangents.

    The *same* operator is applied to the NURBS polyline and to the STL-recovered
    polyline, which is what makes the dual-format contract check meaningful:
    identical vertices must yield identical normals.
    """
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    x0, y0 = np.roll(x, 1), np.roll(y, 1)
    tx, ty = x1 - x0, y1 - y0
    norm = np.hypot(tx, ty)
    nx, ny = -ty / norm, tx / norm
    # enforce outwardness (defensive: the traversal sense is a convention)
    if np.mean(nx * x + ny * y) < 0.0:
        nx, ny = -nx, -ny
    return nx, ny


def _finalise(x: np.ndarray, y: np.ndarray, fmt: str) -> SurfaceGeometry:
    x = np.asarray(x, dtype=np.float64).copy()
    y = np.asarray(y, dtype=np.float64).copy()
    # drop an explicitly repeated closing vertex, if present
    if x.size > 2 and math.hypot(x[0] - x[-1], y[0] - y[-1]) < 1e-14:
        x, y = x[:-1], y[:-1]
    # enforce theta-increasing (clockwise) traversal => negative shoelace
    area2 = float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    if area2 > 0.0:
        x, y = x[::-1].copy(), y[::-1].copy()
    # rotate so that vertex 0 is the upstream (minimum x) stagnation point
    i0 = int(np.argmin(x))
    x, y = np.roll(x, -i0), np.roll(y, -i0)
    nx, ny = _vertex_normals(x, y)
    seg = np.hypot(np.roll(x, -1) - x, np.roll(y, -1) - y)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    return SurfaceGeometry(x=x, y=y, nx=nx, ny=ny, arclength=s, source_format=fmt)


# ---------------------------------------------------------------------------
# NURBS circle (rational quadratic, 9 control points)
# ---------------------------------------------------------------------------

#: Control points / weights / knots of the unit-radius rational quadratic circle
#: (Piegl & Tiller, *The NURBS Book*, 2nd ed., §6.5 / ex. 6.3; degree p = 2).
_CIRCLE_CTRL = np.array(
    [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [-1.0, 1.0], [-1.0, 0.0],
     [-1.0, -1.0], [0.0, -1.0], [1.0, -1.0], [1.0, 0.0]]
)
_CIRCLE_W = np.array([1.0, math.sqrt(0.5), 1.0, math.sqrt(0.5), 1.0,
                      math.sqrt(0.5), 1.0, math.sqrt(0.5), 1.0])
_CIRCLE_KNOTS = np.array([0, 0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1, 1, 1],
                         dtype=np.float64)


def _bspline_basis(i: int, p: int, u: float, knots: np.ndarray) -> float:
    """Cox-de Boor recursion for a single basis function at a single parameter."""
    if p == 0:
        if u == knots[-1]:
            return 1.0 if knots[i] <= u <= knots[i + 1] else 0.0
        return 1.0 if knots[i] <= u < knots[i + 1] else 0.0
    d1 = knots[i + p] - knots[i]
    d2 = knots[i + p + 1] - knots[i + 1]
    t1 = 0.0 if d1 <= 0 else (u - knots[i]) / d1 * _bspline_basis(i, p - 1, u, knots)
    t2 = 0.0 if d2 <= 0 else (knots[i + p + 1] - u) / d2 * _bspline_basis(i + 1, p - 1, u, knots)
    return t1 + t2


def _nurbs_point(u: float, radius: float) -> tuple[float, float]:
    p = 2
    num_x = num_y = den = 0.0
    for i in range(_CIRCLE_CTRL.shape[0]):
        nbi = _bspline_basis(i, p, u, _CIRCLE_KNOTS)
        if nbi == 0.0:
            continue
        w = _CIRCLE_W[i] * nbi
        num_x += w * _CIRCLE_CTRL[i, 0]
        num_y += w * _CIRCLE_CTRL[i, 1]
        den += w
    return radius * num_x / den, radius * num_y / den


def nurbs_circle(diameter: float = 1.0, n_vertices: int = 128,
                 n_dense: int = 4097) -> SurfaceGeometry:
    """Sample the NURBS cylinder cross-section at uniform arc length."""
    radius = 0.5 * diameter
    us = np.linspace(0.0, 1.0, n_dense)
    pts = np.array([_nurbs_point(u, radius) for u in us])
    seg = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    targets = np.linspace(0.0, total, n_vertices, endpoint=False)
    x = np.interp(targets, s, pts[:, 0])
    y = np.interp(targets, s, pts[:, 1])
    # snap onto the exact circle: removes the O(h^4) arc-length resampling
    # residue and keeps the body identity exact (geometry gate: D from area)
    r = np.hypot(x, y)
    x, y = x / r * radius, y / r * radius
    return _finalise(x, y, "nurbs")


# ---------------------------------------------------------------------------
# STL prism round trip
# ---------------------------------------------------------------------------

def polyline_to_stl(geom: SurfaceGeometry, path: str | Path, thickness: float = 1.0) -> None:
    """Write the extruded body as an ASCII STL prism (round-trip exact text)."""
    x, y = geom.x, geom.y
    z0, z1 = -0.5 * thickness, 0.5 * thickness
    tri = []

    def _add(p: Sequence[float], q: Sequence[float], r: Sequence[float]) -> None:
        u = np.subtract(q, p)
        v = np.subtract(r, p)
        n = np.cross(u, v)
        nl = float(np.linalg.norm(n))
        n = n / nl if nl > 0 else n
        tri.append((n, np.asarray(p), np.asarray(q), np.asarray(r)))

    n = geom.n
    for i in range(n):
        j = (i + 1) % n
        _add((x[i], y[i], z0), (x[j], y[j], z0), (x[j], y[j], z1))
        _add((x[i], y[i], z0), (x[j], y[j], z1), (x[i], y[i], z1))
    for k in range(1, n - 1):  # caps, fan from vertex 0
        _add((x[0], y[0], z1), (x[k], y[k], z1), (x[k + 1], y[k + 1], z1))
        _add((x[0], y[0], z0), (x[k + 1], y[k + 1], z0), (x[k], y[k], z0))

    lines = ["solid cylinder"]
    for nvec, a, b, c in tri:
        lines.append(f"  facet normal {nvec[0]!r} {nvec[1]!r} {nvec[2]!r}")
        lines.append("    outer loop")
        for v in (a, b, c):
            lines.append(f"      vertex {float(v[0])!r} {float(v[1])!r} {float(v[2])!r}")
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append("endsolid cylinder")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_stl(path: str | Path) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    triangles = []
    verts = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("vertex"):
            _, xs, ys, zs = s.split()
            verts.append((float(xs), float(ys), float(zs)))
            if len(verts) == 3:
                triangles.append(tuple(np.array(v) for v in verts))
                verts = []
    return triangles


def polyline_from_stl(path: str | Path) -> SurfaceGeometry:
    """Recover the closed cross-section polyline from the prism STL.

    The cap triangulation is analysed by edge adjacency: interior (fan) edges
    appear twice, boundary edges once.  Following the boundary-edge loop yields
    the cross-section polygon in file-consistent, orientation-agnostic order.
    """
    tris = _parse_stl(path)
    if not tris:
        raise ValueError(f"no triangles parsed from {path}")
    zmin = min(float(t[0][2]) for t in tris)
    cap = [t for t in tris if max(float(v[2]) for v in t) == zmin]
    if not cap:
        raise ValueError("no cap triangle found")

    def _key(p) -> tuple[float, float]:
        return (float(p[0]), float(p[1]))

    edges: dict[tuple, int] = {}
    for t in cap:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            ka, kb = _key(a), _key(b)
            k = (ka, kb) if ka <= kb else (kb, ka)
            edges[k] = edges.get(k, 0) + 1
    boundary = [k for k, c in edges.items() if c == 1]
    if len(boundary) < 3:
        raise ValueError("degenerate cap boundary")
    adj: dict[tuple, list[tuple]] = {}
    for ka, kb in boundary:
        adj.setdefault(ka, []).append(kb)
        adj.setdefault(kb, []).append(ka)
    start = boundary[0][0]
    loop = [start]
    prev, cur = None, start
    while True:
        nxts = [n for n in adj[cur] if n != prev]
        if not nxts:
            raise ValueError("cap boundary is not a closed loop")
        nxt = nxts[0]
        if nxt == start:
            break
        loop.append(nxt)
        prev, cur = cur, nxt
    x = np.array([p[0] for p in loop], dtype=np.float64)
    y = np.array([p[1] for p in loop], dtype=np.float64)
    return _finalise(x, y, "stl")


def load_geometry(spec: str = "nurbs", diameter: float = 1.0,
                  n_vertices: int = 128, workdir: str | Path | None = None) -> SurfaceGeometry:
    """Load the shared geometry through the requested format."""
    if spec == "nurbs":
        return nurbs_circle(diameter=diameter, n_vertices=n_vertices)
    if spec == "stl":
        if workdir is None:
            raise ValueError("stl loading requires a workdir for the temporary prism file")
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        path = workdir / f"cylinder_n{n_vertices}.stl"
        polyline_to_stl(nurbs_circle(diameter=diameter, n_vertices=n_vertices), path)
        return polyline_from_stl(path)
    raise ValueError(f"unknown geometry format {spec!r}")


def geometry_contract_check(diameter: float = 1.0, n_vertices: int = 128,
                            workdir: str | Path | None = None) -> dict:
    """Dual-format consistency gate (acceptance criterion 4)."""
    a = load_geometry("nurbs", diameter, n_vertices, workdir)
    b = load_geometry("stl", diameter, n_vertices, workdir)
    if a.n != b.n:
        return {
            "n_vertices_nurbs": a.n,
            "n_vertices_stl": b.n,
            "max_vertex_deviation": float("inf"),
            "max_normal_deviation": float("inf"),
            "passed": False,
        }
    dv = float(np.max(np.abs(a.vertices() - b.vertices())))
    dn = float(np.max(np.abs(a.normals() - b.normals())))
    return {
        "n_vertices_nurbs": a.n,
        "n_vertices_stl": b.n,
        "max_vertex_deviation": dv,
        "max_normal_deviation": dn,
        "area_deviation": abs(a.area - b.area),
        "hydraulic_diameter_nurbs": a.hydraulic_diameter,
        "hydraulic_diameter_stl": b.hydraulic_diameter,
        "tolerance": 1e-9,
        "passed": bool(dv < 1e-9 and dn < 1e-9),
    }