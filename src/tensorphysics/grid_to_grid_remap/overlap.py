"""Triangle-triangle intersection and overlap table for two unstructured meshes.

The overlap table is the geometric backbone of conservative remapping: for every
pair of source/target triangles that intersect, we store the intersection polygon
and its area.  The sparse projection matrix ``M`` is then built as

    M[j, i] = |T_j cap S_i| / |T_j|

so that ``phi_tgt[j] = sum_i M[j,i] * phi_src[i]`` is conservative by construction:

    sum_j phi_tgt[j] * |T_j| = sum_j sum_i |T_j cap S_i| * phi_src[i]
                              = sum_i phi_src[i] * sum_j |T_j cap S_i|
                              = sum_i phi_src[i] * |S_i|      (partition of unity)

A quad-tree bounding-box filter avoids the O(N_s * N_t) brute-force scan.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import scipy.sparse as sp

__all__ = [
    "TriMesh",
    "OverlapTable",
    "build_overlap_table",
    "clip_triangle_by_triangle",
    "polygon_area",
    "polygon_centroid",
]

Point = tuple[float, float]
Poly = list[Point]


# ---------------------------------------------------------------------------
# mesh container (lightweight, no FEM solver)
# ---------------------------------------------------------------------------
@dataclass
class TriMesh:
    """Straight-sided triangular mesh (CCW elements)."""

    nodes: np.ndarray  # (N, 2)
    tri: np.ndarray    # (M, 3) int, CCW
    label: str = ""

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def n_elements(self) -> int:
        return int(self.tri.shape[0])

    def areas(self) -> np.ndarray:
        p0 = self.nodes[self.tri[:, 0]]
        p1 = self.nodes[self.tri[:, 1]]
        p2 = self.nodes[self.tri[:, 2]]
        return 0.5 * np.abs(
            (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
            - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1])
        )

    def centroids(self) -> np.ndarray:
        return self.nodes[self.tri].mean(axis=1)

    def bounding_boxes(self) -> np.ndarray:
        """(M, 4) = (xmin, xmax, ymin, ymax) per element."""
        xy = self.nodes[self.tri]  # (M, 3, 2)
        return np.stack([
            xy[:, :, 0].min(axis=1),
            xy[:, :, 0].max(axis=1),
            xy[:, :, 1].min(axis=1),
            xy[:, :, 1].max(axis=1),
        ], axis=1)

    def validate(self, expected_area: float | None = None) -> dict:
        a = self.areas()
        total = float(a.sum())
        q = {
            "n_nodes": self.n_nodes,
            "n_elements": self.n_elements,
            "total_area": total,
            "min_area": float(a.min()),
            "ok": bool(a.min() > 0.0),
        }
        if expected_area is not None:
            q["area_error_rel"] = abs(total - expected_area) / expected_area
        return q


# ---------------------------------------------------------------------------
# polygon geometry
# ---------------------------------------------------------------------------
def polygon_area(poly: Sequence[Point]) -> float:
    """Signed-free area of a simple polygon (shoelace)."""
    n = len(poly)
    if n < 3:
        return 0.0
    s = 0.0
    for k in range(n):
        x1, y1 = poly[k]
        x2, y2 = poly[(k + 1) % n]
        s += x1 * y2 - x2 * y1
    return 0.5 * abs(s)


def polygon_centroid(pts: np.ndarray) -> np.ndarray:
    """Area centroid of a convex polygon (fan decomposition)."""
    if len(pts) < 3:
        return pts.mean(axis=0)
    p0 = pts[0]
    total_a = 0.0
    total_c = np.zeros(2)
    for k in range(1, len(pts) - 1):
        a = pts[k] - p0
        b = pts[k + 1] - p0
        ar = 0.5 * (a[0] * b[1] - a[1] * b[0])
        total_a += ar
        total_c += ar * (p0 + pts[k] + pts[k + 1]) / 3.0
    if abs(total_a) < 1e-300:
        return pts.mean(axis=0)
    return total_c / total_a


def _lerp(a: Point, b: Point, t: float) -> Point:
    return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))


def clip_triangle_by_triangle(
    tri_a: np.ndarray, tri_b: np.ndarray
) -> list[Point]:
    """Clip convex triangle A against the three half-planes of triangle B.

    Uses Sutherland-Hodgman: for each edge of B, keep the part of the polygon
    that lies on the interior side (CCW convention: interior is to the left).

    Parameters
    ----------
    tri_a, tri_b : (3, 2) arrays of vertex coordinates (CCW order).

    Returns
    -------
    List of (x, y) vertices of the intersection polygon (empty if no overlap).
    """
    pts: list[Point] = [(float(tri_a[k, 0]), float(tri_a[k, 1])) for k in range(3)]

    for e in range(3):
        if not pts:
            return []
        p0 = (float(tri_b[e, 0]), float(tri_b[e, 1]))
        p1 = (float(tri_b[(e + 1) % 3, 0]), float(tri_b[(e + 1) % 3, 1]))
        # Edge normal pointing inward (CCW: left side)
        ex, ey = p1[0] - p0[0], p1[1] - p0[1]
        # For point p, cross product (p1-p0) x (p-p0) > 0 means left side
        def inside(p: Point) -> bool:
            return (ex * (p[1] - p0[1]) - ey * (p[0] - p0[0])) >= -1e-14

        def intersect_edge(a: Point, b: Point) -> Point:
            # Parametric: a + t*(b-a), find t where cross = 0
            da = (ex * (a[1] - p0[1]) - ey * (a[0] - p0[0]))
            db = (ex * (b[1] - p0[1]) - ey * (b[0] - p0[0]))
            denom = da - db
            if abs(denom) < 1e-300:
                return a
            t = da / denom
            return _lerp(a, b, t)

        out: list[Point] = []
        n = len(pts)
        for k in range(n):
            a = pts[k]
            b = pts[(k + 1) % n]
            ia, ib = inside(a), inside(b)
            if ib:
                if not ia:
                    out.append(intersect_edge(a, b))
                out.append(b)
            elif ia:
                out.append(intersect_edge(a, b))
        pts = out

    return pts


# ---------------------------------------------------------------------------
# spatial index: simple bounding-box filter
# ---------------------------------------------------------------------------
def _bbox_overlap(src_bb: np.ndarray, tgt_bb: np.ndarray) -> np.ndarray:
    """Return (i, j) pairs where source element i and target element j overlap in bbox.

    Uses a simple sweep: for each source element, find target elements whose
    bbox overlaps.  For moderate mesh sizes this is fast enough; for large meshes
    a quad-tree or BVH would be needed.
    """
    pairs_i, pairs_j = [], []
    ns, nt = len(src_bb), len(tgt_bb)
    # Build a simple grid index on target bboxes
    if nt == 0 or ns == 0:
        return np.empty((0, 2), dtype=int)

    # For efficiency, use vectorized bbox overlap check
    # src_bb: (ns, 4) = (xmin, xmax, ymin, ymax)
    # tgt_bb: (nt, 4)
    # Overlap condition: src.xmin < tgt.xmax AND src.xmax > tgt.xmin AND same for y
    for i in range(ns):
        sx0, sx1, sy0, sy1 = src_bb[i]
        mask = (tgt_bb[:, 0] < sx1 + 1e-14) & (tgt_bb[:, 1] > sx0 - 1e-14) & \
               (tgt_bb[:, 2] < sy1 + 1e-14) & (tgt_bb[:, 3] > sy0 - 1e-14)
        js = np.nonzero(mask)[0]
        for j in js:
            pairs_i.append(i)
            pairs_j.append(j)
    return np.column_stack([pairs_i, pairs_j]) if pairs_i else np.empty((0, 2), dtype=int)


# ---------------------------------------------------------------------------
# overlap table
# ---------------------------------------------------------------------------
@dataclass
class OverlapTable:
    """Geometry-only source/target intersection table.

    ``M`` is the sparse matrix such that ``M @ phi_src`` gives the overlap-weighted
    integral for each target cell.  For cell-averaged data the conservative
    projection is ``phi_tgt = (M @ phi_src) / tgt_areas``.
    """

    source: TriMesh
    target: TriMesh
    M: sp.csr_matrix       # (n_tgt, n_src) overlap-area matrix
    tgt_areas: np.ndarray   # (n_tgt,)
    src_areas: np.ndarray   # (n_src,)
    coverage: np.ndarray    # (n_tgt,) mapped area per target cell
    total_overlap: float    # sum of all overlap areas

    @property
    def n_src(self) -> int:
        return self.source.n_elements

    @property
    def n_tgt(self) -> int:
        return self.target.n_elements

    def coverage_fraction(self) -> np.ndarray:
        return self.coverage / np.maximum(self.tgt_areas, 1e-300)


def build_overlap_table(
    source: TriMesh,
    target: TriMesh,
    area_tol: float = 1e-14,
) -> OverlapTable:
    """Build the overlap table between two unstructured triangular meshes.

    Uses bounding-box filtering to avoid O(N_s * N_t) brute force, then exact
    triangle-triangle clipping for candidate pairs.

    Parameters
    ----------
    source, target : TriMesh
        Source (donor) and target (receiver) meshes.
    area_tol : float
        Minimum overlap area to keep (relative to mean source area).

    Returns
    -------
    OverlapTable with the sparse projection matrix.
    """
    src_areas = source.areas()
    tgt_areas = target.areas()
    mean_src = float(src_areas.mean()) if len(src_areas) > 0 else 1.0
    atol = area_tol * mean_src

    src_bb = source.bounding_boxes()
    tgt_bb = target.bounding_boxes()

    pairs = _bbox_overlap(src_bb, tgt_bb)

    rows, cols, vals = [], [], []
    coverage = np.zeros(target.n_elements)

    for idx in range(len(pairs)):
        i, j = int(pairs[idx, 0]), int(pairs[idx, 1])
        tri_s = source.nodes[source.tri[i]]
        tri_t = target.nodes[target.tri[j]]
        poly = clip_triangle_by_triangle(tri_s, tri_t)
        if len(poly) < 3:
            continue
        ap = polygon_area(poly)
        if ap <= atol:
            continue
        rows.append(j)
        cols.append(i)
        vals.append(float(ap))
        coverage[j] += ap

    M = sp.coo_matrix(
        (vals, (rows, cols)), shape=(target.n_elements, source.n_elements)
    ).tocsr()

    return OverlapTable(
        source=source,
        target=target,
        M=M,
        tgt_areas=tgt_areas,
        src_areas=src_areas,
        coverage=coverage,
        total_overlap=float(M.sum()),
    )
