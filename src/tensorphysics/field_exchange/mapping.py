"""Conservative cross-solver field exchange: source surface data -> FEM nodal loads.

Why this module exists
----------------------
Coupling LBM/DEM to a structural FEM code is only trustworthy if the *exchange*
step neither creates nor destroys load.  The classical requirement is the
discrete analogue of

    \\int_{Gamma} t(x) dA = \\sum_i f_i            (force conservation)
    \\int_{Gamma} r x t(x) dA = \\sum_i r_i x f_i  (moment conservation)

for every component of the traction vector ``t``.  We implement two exchange
operators on a rectangular surface target grid

* :func:`map_panels`   -- cell-centred traction panels (LBM wall data, or any
  face-averaged surface field) mapped by *overlap-area weighting*; each
  panel/target-cell intersection is clipped exactly (Sutherland-Hodgman) and
  integrated with a degree-2 rule, which reproduces bilinear shape functions
  ``N_i`` exactly (``N_i`` is a total-degree-2 polynomial inside a rectangle).
* :func:`map_contacts` -- discrete DEM contact forces lumped with the bilinear
  partition of unity ``sum_i N_i = 1`` (exact force conservation), followed by
  an optional minimum-norm **moment completion** that repairs the lever-arm
  defect of point lumping (``sum_i N_i (r_i - r) != 0`` in general).

Because both operators use a partition of unity that reproduces linear
coordinates (``sum_i N_i x_i = x``), force *and* moment conservation are exact
identities, not approximations; :func:`audit_panel_map` measures the residual
against an independently quadratured source integral so the claim is testable.

Ownership: cross-solver exchange numerics only (no solver kernels here).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

__all__ = [
    "QuadGrid",
    "clip_rect",
    "integrate_polygon",
    "map_panels",
    "map_panels_to_basis",
    "map_contacts",
    "moment_completion",
    "audit_panel_map",
    "audit_contact_map",
    "nearest_node_lumping",
]


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------
Point = tuple[float, float]
Poly = list[Point]


def clip_rect(poly: Sequence[Point], rect: tuple[float, float, float, float]) -> Poly:
    """Sutherland-Hodgman clipping of a convex polygon by an axis-aligned rect."""
    xmin, xmax, ymin, ymax = rect
    pts: Poly = [(float(p[0]), float(p[1])) for p in poly]
    if not pts:
        return []

    def clip(pts: Poly, inside: Callable[[Point], bool],
             cross: Callable[[Point, Point], Point]) -> Poly:
        if not pts:
            return []
        out: Poly = []
        n = len(pts)
        for k in range(n):
            a = pts[k]
            b = pts[(k + 1) % n]
            ia, ib = inside(a), inside(b)
            if ib:
                if not ia:
                    out.append(cross(a, b))
                out.append(b)
            elif ia:
                out.append(cross(a, b))
        return out

    def lerp_x(a: Point, b: Point, x: float) -> Point:
        if abs(b[0] - a[0]) < 1e-300:
            return (x, a[1])
        s = (x - a[0]) / (b[0] - a[0])
        return (x, a[1] + s * (b[1] - a[1]))

    def lerp_y(a: Point, b: Point, y: float) -> Point:
        if abs(b[1] - a[1]) < 1e-300:
            return (a[0], y)
        s = (y - a[1]) / (b[1] - a[1])
        return (a[0] + s * (b[0] - a[0]), y)

    pts = clip(pts, lambda p: p[0] >= xmin, lambda a, b: lerp_x(a, b, xmin))
    pts = clip(pts, lambda p: p[0] <= xmax, lambda a, b: lerp_x(a, b, xmax))
    pts = clip(pts, lambda p: p[1] >= ymin, lambda a, b: lerp_y(a, b, ymin))
    pts = clip(pts, lambda p: p[1] <= ymax, lambda a, b: lerp_y(a, b, ymax))
    return pts


def polygon_area(poly: Sequence[Point]) -> float:
    """Signed-free area of a simple polygon (shoelace formula)."""
    n = len(poly)
    if n < 3:
        return 0.0
    s = 0.0
    for k in range(n):
        x1, y1 = poly[k]
        x2, y2 = poly[(k + 1) % n]
        s += x1 * y2 - x2 * y1
    return 0.5 * abs(s)


def integrate_polygon(poly: Sequence[Point], fn: Callable[[float, float], np.ndarray]) -> np.ndarray:
    """Integrate ``fn(x, y)`` over a convex polygon with a degree-2 triangle rule.

    The polygon is fan-triangulated from its area centroid and each triangle is
    sampled at its three edge midpoints (weight ``A/3``).  This rule is exact for
    polynomials of total degree <= 2, i.e. for bilinear shape functions times a
    cell-centred constant and for their first geometric moments.
    """
    pts = list(poly)
    if len(pts) < 3:
        return np.zeros(1)
    arr = np.asarray(pts, dtype=float)
    area = polygon_area(pts)
    if area <= 0.0:
        return np.zeros(1)

    # interior point for a convex fan (area centroid)
    x = arr[:, 0]
    y = arr[:, 1]
    x1 = np.roll(x, -1)
    y1 = np.roll(y, -1)
    cr = x * y1 - x1 * y
    a2 = cr.sum()
    if abs(a2) < 1e-300:
        return np.zeros(1)
    cx = ((x + x1) * cr).sum() / (3.0 * a2)
    cy = ((y + y1) * cr).sum() / (3.0 * a2)

    acc: np.ndarray | None = None
    for k in range(len(pts)):
        p1 = pts[k]
        p2 = pts[(k + 1) % len(pts)]
        tri = np.array([[cx, cy], p1, p2], dtype=float)
        tarea = 0.5 * abs(
            (tri[1, 0] - tri[0, 0]) * (tri[2, 1] - tri[0, 1])
            - (tri[2, 0] - tri[0, 0]) * (tri[1, 1] - tri[0, 1])
        )
        if tarea <= 0.0:
            continue
        mids = [0.5 * (tri[0] + tri[1]), 0.5 * (tri[1] + tri[2]), 0.5 * (tri[2] + tri[0])]
        for m in mids:
            val = np.asarray(fn(float(m[0]), float(m[1])), dtype=float)
            if acc is None:
                acc = np.zeros_like(val)
            acc = acc + val * (tarea / 3.0)
    if acc is None:
        return np.zeros(1)
    return acc


# ---------------------------------------------------------------------------
# target discretisation: structured rectangular surface grid
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class QuadGrid:
    """Structured rectangular target grid over ``[x0, x1] x [y0, y1]``.

    Shape functions are bilinear on each cell (the standard 4-node surface
    interpolation used by FEM face loads).  Rectangularity is a deliberate
    limitation: it keeps the patch inverse affine, hence exact, and the
    production mortar path for warped faces lives in
    :mod:`tensorphysics.interface_operators.mortar_nd`.
    """

    x0: float
    x1: float
    y0: float
    y1: float
    nx: int
    ny: int

    def __post_init__(self) -> None:
        if self.nx < 1 or self.ny < 1:
            raise ValueError("nx, ny must be >= 1")
        if not (self.x1 > self.x0 and self.y1 > self.y0):
            raise ValueError("degenerate grid extent")

    @property
    def hx(self) -> float:
        return (self.x1 - self.x0) / self.nx

    @property
    def hy(self) -> float:
        return (self.y1 - self.y0) / self.ny

    @property
    def n_nodes(self) -> int:
        return (self.nx + 1) * (self.ny + 1)

    def node_index(self, i: int, j: int) -> int:
        return j * (self.nx + 1) + i

    def coords(self) -> np.ndarray:
        xs = self.x0 + self.hx * np.arange(self.nx + 1)
        ys = self.y0 + self.hy * np.arange(self.ny + 1)
        xx, yy = np.meshgrid(xs, ys, indexing="xy")
        return np.column_stack([xx.ravel(), yy.ravel()])

    def cell_rect(self, ci: int, cj: int) -> tuple[float, float, float, float]:
        return (
            self.x0 + ci * self.hx,
            self.x0 + (ci + 1) * self.hx,
            self.y0 + cj * self.hy,
            self.y0 + (cj + 1) * self.hy,
        )

    def cell_nodes(self, ci: int, cj: int) -> list[int]:
        """Node ids in CCW order: (i,j), (i+1,j), (i+1,j+1), (i,j+1)."""
        return [
            self.node_index(ci, cj),
            self.node_index(ci + 1, cj),
            self.node_index(ci + 1, cj + 1),
            self.node_index(ci, cj + 1),
        ]

    def bilinear(self, ci: int, cj: int, x: float, y: float) -> np.ndarray:
        xq0, _, yq0, _ = self.cell_rect(ci, cj)
        s = (x - xq0) / self.hx
        t = (y - yq0) / self.hy
        s = min(max(s, 0.0), 1.0)
        t = min(max(t, 0.0), 1.0)
        return np.array([
            (1.0 - s) * (1.0 - t),
            s * (1.0 - t),
            s * t,
            (1.0 - s) * t,
        ])

    def locate(self, x: float, y: float) -> tuple[int, int] | None:
        if not (self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1):
            return None
        ci = min(int((x - self.x0) / self.hx), self.nx - 1)
        cj = min(int((y - self.y0) / self.hy), self.ny - 1)
        return ci, cj

    def cells_overlapping(self, rect: tuple[float, float, float, float]):
        xmin, xmax, ymin, ymax = rect
        i0 = max(int(np.floor((xmin - self.x0) / self.hx)), 0)
        i1 = min(int(np.floor((xmax - self.x0) / self.hx)), self.nx - 1)
        j0 = max(int(np.floor((ymin - self.y0) / self.hy)), 0)
        j1 = min(int(np.floor((ymax - self.y0) / self.hy)), self.ny - 1)
        for cj in range(j0, j1 + 1):
            for ci in range(i0, i1 + 1):
                yield ci, cj


def _rect_poly(rect: tuple[float, float, float, float]) -> Poly:
    xmin, xmax, ymin, ymax = rect
    return [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]


# ---------------------------------------------------------------------------
# panel (cell-centred traction) -> nodal forces
# ---------------------------------------------------------------------------
def map_panels(
    grid: QuadGrid,
    x_edges: Sequence[float],
    y_edges: Sequence[float],
    traction: np.ndarray,
    *,
    check_partition: bool = True,
) -> np.ndarray:
    """Map cell-centred traction panels onto ``grid`` nodes by overlap weighting.

    Parameters
    ----------
    grid : QuadGrid
        Target (FEM) surface discretisation.
    x_edges, y_edges : sequences
        Source panel edges (strictly increasing); panel ``(i, j)`` spans
        ``[x_edges[i], x_edges[i+1]] x [y_edges[j], y_edges[j+1]]``.
    traction : ndarray, shape ``(len(x_edges)-1, len(y_edges)-1, k)``
        Piecewise-constant traction vector on each panel (``k`` = 2 or 3).

    Returns
    -------
    ndarray, shape ``(grid.n_nodes, k)``
        Nodal forces ``f_i = \\int_Gamma t N_i dA``.  Force and moment of the
        returned vector equal those of the panel field exactly (to round-off).
    """
    xe = np.asarray(x_edges, dtype=float)
    ye = np.asarray(y_edges, dtype=float)
    tr = np.asarray(traction, dtype=float)
    npx, npy = len(xe) - 1, len(ye) - 1
    if tr.shape[:2] != (npx, npy):
        raise ValueError("traction shape must be (n_panels_x, n_panels_y, k)")
    if check_partition:
        areas = np.outer(np.diff(xe), np.diff(ye))
        if not np.isclose(areas.sum(), (grid.x1 - grid.x0) * (grid.y1 - grid.y0), rtol=1e-9):
            raise ValueError("source panels do not cover the target grid area")

    k = tr.shape[2]
    f = np.zeros((grid.n_nodes, k))
    for pi in range(npx):
        for pj in range(npy):
            rect = (xe[pi], xe[pi + 1], ye[pj], ye[pj + 1])
            tvec = tr[pi, pj]
            for ci, cj in grid.cells_overlapping(rect):
                poly = clip_rect(_rect_poly(rect), grid.cell_rect(ci, cj))
                if polygon_area(poly) <= 0.0:
                    continue
                nodes = grid.cell_nodes(ci, cj)
                for a, nid in enumerate(nodes):
                    w = integrate_polygon(
                        poly, lambda x, y, _ci=ci, _cj=cj, _a=a: np.array(
                            [grid.bilinear(_ci, _cj, x, y)[_a]]
                        )
                    )[0]
                    if w != 0.0:
                        f[nid] += tvec * w
    return f


def map_panels_to_basis(
    grid: QuadGrid,
    x_edges: Sequence[float],
    y_edges: Sequence[float],
    traction: np.ndarray,
    basis: Callable[[int, int, float, float], np.ndarray],
    n_dof_per_node: int,
) -> np.ndarray:
    """Generalised version: distribute each panel onto an arbitrary per-cell basis.

    ``basis(ci, cj, x, y) -> weights`` returns the (partition-of-unity, linear
    reproducing) weight vector of the *local* cell DOFs.  Used for Hermite plate
    elements whose rotational DOFs also receive consistent moments.
    """
    xe = np.asarray(x_edges, dtype=float)
    ye = np.asarray(y_edges, dtype=float)
    tr = np.asarray(traction, dtype=float)
    npx, npy = len(xe) - 1, len(ye) - 1
    f = np.zeros(grid.n_nodes * n_dof_per_node)
    for pi in range(npx):
        for pj in range(npy):
            rect = (xe[pi], xe[pi + 1], ye[pj], ye[pj + 1])
            tvec = tr[pi, pj]
            scalar = float(tvec[0]) if tvec.ndim else float(tvec)
            for ci, cj in grid.cells_overlapping(rect):
                poly = clip_rect(_rect_poly(rect), grid.cell_rect(ci, cj))
                if polygon_area(poly) <= 0.0:
                    continue
                nodes = grid.cell_nodes(ci, cj)
                w = integrate_polygon(
                    poly, lambda x, y, _ci=ci, _cj=cj: basis(_ci, _cj, x, y)
                )
                for a, nid in enumerate(nodes):
                    seg = w[a * n_dof_per_node:(a + 1) * n_dof_per_node]
                    f[nid * n_dof_per_node:(nid + 1) * n_dof_per_node] += scalar * seg
    return f


# ---------------------------------------------------------------------------
# discrete contact forces -> nodal forces (+ moment completion)
# ---------------------------------------------------------------------------
def _moment_matrix(xy: np.ndarray) -> np.ndarray:
    """Constraint matrix A (6 x 3m): rows 0-2 force sums, 3-5 moments about origin."""
    m = len(xy)
    A = np.zeros((6, 3 * m))
    for a in range(m):
        x, y = xy[a]
        # force
        A[0, 3 * a + 0] = 1.0
        A[1, 3 * a + 1] = 1.0
        A[2, 3 * a + 2] = 1.0
        # r x dF with z = 0 (planar interface)
        # (r x F)_x = y Fz - z Fy = y Fz
        A[3, 3 * a + 2] = y
        # (r x F)_y = z Fx - x Fz = -x Fz
        A[4, 3 * a + 2] = -x
        # (r x F)_z = x Fy - y Fx
        A[5, 3 * a + 1] = x
        A[5, 3 * a + 0] = -y
    return A


def moment_completion(xy: np.ndarray, d_force: np.ndarray, d_moment: np.ndarray) -> np.ndarray:
    """Minimum-norm nodal corrections realising prescribed resultant force/moment.

    Solves ``A d = [d_force, d_moment]`` in the least-squares sense with the
    minimum-norm (pseudo-inverse) solution, so no spurious energy is injected
    beyond what the conservation requirement demands.
    """
    A = _moment_matrix(np.asarray(xy, dtype=float))
    b = np.concatenate([np.asarray(d_force, dtype=float), np.asarray(d_moment, dtype=float)])
    d, *_ = np.linalg.lstsq(A, b, rcond=1e-12)
    return d.reshape(-1, 3)


def map_contacts(
    grid: QuadGrid,
    points: np.ndarray,
    forces: np.ndarray,
    *,
    moments: np.ndarray | None = None,
    complete_moment: bool = True,
) -> tuple[np.ndarray, dict]:
    """Lump DEM contact forces onto ``grid`` nodes via the bilinear partition of unity.

    Returns ``(nodal_forces, info)`` where ``info`` records, per contact, the
    containing cell, the partition weights and the moment residual of the raw
    lumping (before completion).

    Notes
    -----
    For a *planar* interface and pure forces the raw partition-of-unity lumping
    already conserves the moment exactly, because the weights reproduce the
    coordinates: ``sum_i N_i (r_i x F) = (sum_i N_i r_i) x F = r x F``.  The
    moment-completion step therefore only has work to do when the source carries
    a contact couple (``moments``), when the interface is curved, or when the
    weights do not reproduce coordinates (see :func:`nearest_node_lumping`).
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    fsrc = np.asarray(forces, dtype=float).reshape(-1, 3)
    if len(pts) != len(fsrc):
        raise ValueError("points and forces must have the same length")

    if moments is None:
        mom = np.zeros_like(fsrc)
    else:
        mom = np.asarray(moments, dtype=float).reshape(-1, 3)
        if len(mom) != len(pts):
            raise ValueError("moments must have the same length as points")

    f = np.zeros((grid.n_nodes, 3))
    info = {"cells": [], "weights": [], "moment_defect_raw": 0.0, "n_contacts": len(pts)}
    coords = grid.coords()
    for n in range(len(pts)):
        loc = grid.locate(pts[n, 0], pts[n, 1])
        if loc is None:
            raise ValueError(f"contact {n} lies outside the target grid")
        ci, cj = loc
        w = grid.bilinear(ci, cj, pts[n, 0], pts[n, 1])
        nodes = grid.cell_nodes(ci, cj)
        for a, nid in enumerate(nodes):
            f[nid] += w[a] * fsrc[n]
        xy = coords[nodes]
        lumped_moment = np.zeros(3)
        for a, nid in enumerate(nodes):
            r = np.array([coords[nid, 0], coords[nid, 1], 0.0])
            lumped_moment += np.cross(r, w[a] * fsrc[n])
        exact_moment = np.cross(np.array([pts[n, 0], pts[n, 1], 0.0]), fsrc[n]) + mom[n]
        dm = exact_moment - lumped_moment
        info["cells"].append((ci, cj))
        info["weights"].append(w.copy())
        info["moment_defect_raw"] += float(np.linalg.norm(dm))
        if complete_moment:
            d = moment_completion(xy, np.zeros(3), dm)
            for a, nid in enumerate(nodes):
                f[nid] += d[a]
    return f, info


# ---------------------------------------------------------------------------
# audits
# ---------------------------------------------------------------------------
def _gauss_2d(n: int) -> tuple[np.ndarray, np.ndarray]:
    gx, gw = np.polynomial.legendre.leggauss(n)
    return gx, gw


def panel_reference_resultants(
    x_edges: Sequence[float],
    y_edges: Sequence[float],
    traction: np.ndarray,
    order: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent (high-order Gauss) source-side force and moment resultants."""
    xe = np.asarray(x_edges, dtype=float)
    ye = np.asarray(y_edges, dtype=float)
    tr = np.asarray(traction, dtype=float)
    gx, gw = _gauss_2d(order)
    tot_f = np.zeros(3)
    tot_m = np.zeros(3)
    for pi in range(len(xe) - 1):
        for pj in range(len(ye) - 1):
            a, b = xe[pi], xe[pi + 1]
            c, d = ye[pj], ye[pj + 1]
            xs = 0.5 * (a + b) + 0.5 * (b - a) * gx
            ys = 0.5 * (c + d) + 0.5 * (d - c) * gx
            jac = 0.25 * (b - a) * (d - c)
            t = tr[pi, pj]
            for m in range(len(gx)):
                for n in range(len(gx)):
                    wgt = gw[m] * gw[n] * jac
                    x, y = xs[m], ys[n]
                    tot_f += t * wgt
                    tot_m += np.cross(np.array([x, y, 0.0]), t) * wgt
    return tot_f, tot_m


def nodal_resultants(grid: QuadGrid, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resultant force and moment (about the origin) of nodal loads."""
    xy = grid.coords()
    ff = np.asarray(f, dtype=float)
    if ff.shape[1] == 2:
        ff = np.hstack([ff, np.zeros((len(ff), 1))])
    tot_f = ff.sum(axis=0)
    tot_m = np.zeros(3)
    for n in range(len(xy)):
        tot_m += np.cross(np.array([xy[n, 0], xy[n, 1], 0.0]), ff[n])
    return tot_f, tot_m


def _rel(res: np.ndarray, scale: np.ndarray) -> float:
    num = float(np.linalg.norm(res))
    den = float(np.linalg.norm(scale))
    if den == 0.0:
        return 0.0 if num == 0.0 else float("inf")
    return num / den


def audit_panel_map(
    grid: QuadGrid,
    f: np.ndarray,
    x_edges: Sequence[float],
    y_edges: Sequence[float],
    traction: np.ndarray,
) -> dict:
    """Force/moment conservation audit of a panel-to-node mapping."""
    fs, ms = panel_reference_resultants(x_edges, y_edges, traction)
    ft, mt = nodal_resultants(grid, f)
    l1 = float(np.abs(f).sum())
    m1 = 0.0
    xy = grid.coords()
    ff = np.asarray(f, dtype=float)
    if ff.shape[1] == 2:
        ff = np.hstack([ff, np.zeros((len(ff), 1))])
    for n in range(len(xy)):
        m1 += float(np.linalg.norm(np.cross(np.array([xy[n, 0], xy[n, 1], 0.0]), ff[n])))
    return {
        "force_source": fs.tolist(),
        "force_target": ft.tolist(),
        "moment_source": ms.tolist(),
        "moment_target": mt.tolist(),
        "force_residual_rel": _rel(ft - fs, np.abs(f).sum(axis=0)),
        "moment_residual_rel": _rel(mt - ms, np.array([m1, m1, m1])),
        "force_scale_L1": l1,
    }


def audit_contact_map(
    grid: QuadGrid,
    f: np.ndarray,
    points: np.ndarray,
    forces: np.ndarray,
    moments: np.ndarray | None = None,
) -> dict:
    """Force/moment conservation audit of a contact-force mapping."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    fsrc = np.asarray(forces, dtype=float).reshape(-1, 3)
    mom = np.zeros_like(fsrc) if moments is None else np.asarray(moments, dtype=float).reshape(-1, 3)
    fs = fsrc.sum(axis=0)
    ms = mom.sum(axis=0).copy()
    for n in range(len(pts)):
        ms += np.cross(np.array([pts[n, 0], pts[n, 1], 0.0]), fsrc[n])
    ft, mt = nodal_resultants(grid, f)
    xy = grid.coords()
    m1 = 0.0
    for n in range(len(xy)):
        m1 += float(np.linalg.norm(np.cross(np.array([xy[n, 0], xy[n, 1], 0.0]), f[n])))
    return {
        "force_source": fs.tolist(),
        "force_target": ft.tolist(),
        "moment_source": ms.tolist(),
        "moment_target": mt.tolist(),
        "force_residual_rel": _rel(ft - fs, np.abs(fsrc).sum(axis=0)),
        "moment_residual_rel": _rel(mt - ms, np.array([m1, m1, m1])),
        "force_scale_L1": float(np.abs(f).sum()),
    }


def nearest_node_lumping(
    grid: QuadGrid, points: np.ndarray, forces: np.ndarray
) -> np.ndarray:
    """Negative control: dump each contact force on its nearest node.

    Force is conserved, but the lever arm is snapped to the node, so the moment
    residual is O(h |F|).  Any audit that cannot reject this operator is not
    measuring conservation.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    fsrc = np.asarray(forces, dtype=float).reshape(-1, 3)
    f = np.zeros((grid.n_nodes, 3))
    x = grid.x0 + grid.hx * np.arange(grid.nx + 1)
    y = grid.y0 + grid.hy * np.arange(grid.ny + 1)
    for n in range(len(pts)):
        i = int(np.argmin(np.abs(x - pts[n, 0])))
        j = int(np.argmin(np.abs(y - pts[n, 1])))
        f[grid.node_index(i, j)] += fsrc[n]
    return f