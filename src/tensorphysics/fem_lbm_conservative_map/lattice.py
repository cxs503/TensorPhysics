"""Uniform Cartesian target lattice (LBM side) and exact FEM/lattice overlap.

The mapping pipeline needs, once per mesh pair, the geometry-only data

* ``M``       sparse ``(n_cells x n_nodes)`` matrix such that
  ``M @ phi_nodes`` gives, for every target cell, the *exact* integral of the P1
  field over the cell (``int_{C_j} phi dA``) -- this is linear in the nodal values
  because a linear function integrates over a polygon to area times its centroid
  value;
* ``coverage`` the mapped area of every target cell (``sum_T |C_j cap T|``);
* ``overlap``  for every target cell, the list of source elements that intersect
  it together with the area of the intersection.

Building this table once makes every further mapping O(nnz) and, more importantly,
makes global conservation an *analytic* property of the operator rather than a
numerical accident: because the target cells partition the same domain as the
elements, ``sum_j M[j,:] phi = sum_T int_T phi dA`` for every ``phi``.

Ownership: geometric overlap and the linear integral operator only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .mesh import TriMesh
from ..field_exchange.mapping import clip_rect, polygon_area

__all__ = ["CartesianLattice", "OverlapTable", "build_overlap"]


@dataclass
class CartesianLattice:
    """Cell-centred uniform lattice (the D2Q9 control volumes)."""

    nx: int
    ny: int
    x0: float = 0.0
    y0: float = 0.0
    length_x: float = 1.0
    length_y: float = 1.0

    @property
    def dx(self) -> float:
        return self.length_x / self.nx

    @property
    def dy(self) -> float:
        return self.length_y / self.ny

    @property
    def n_cells(self) -> int:
        return self.nx * self.ny

    def cell_area(self) -> float:
        return self.dx * self.dy

    def areas(self) -> np.ndarray:
        return np.full(self.n_cells, self.cell_area())

    def rect(self, j: int) -> tuple[float, float, float, float]:
        """Cell bounding box in the (xmin, xmax, ymin, ymax) order used by
        :func:`tensorphysics.field_exchange.mapping.clip_rect`."""
        jy, jx = divmod(int(j), self.nx)
        xa = self.x0 + jx * self.dx
        ya = self.y0 + jy * self.dy
        return (xa, xa + self.dx, ya, ya + self.dy)

    def centroids(self) -> np.ndarray:
        idx = np.arange(self.n_cells)
        jy, jx = np.divmod(idx, self.nx)
        return np.stack([self.x0 + (jx + 0.5) * self.dx,
                         self.y0 + (jy + 0.5) * self.dy], axis=1)

    def cell_of_point(self, xy: np.ndarray) -> np.ndarray:
        xy = np.atleast_2d(np.asarray(xy, dtype=float))
        jx = np.clip(((xy[:, 0] - self.x0) / self.dx).astype(int), 0, self.nx - 1)
        jy = np.clip(((xy[:, 1] - self.y0) / self.dy).astype(int), 0, self.ny - 1)
        return jy * self.nx + jx

    def field_to_grid(self, cells: np.ndarray) -> np.ndarray:
        """(n_cells,) -> (ny, nx) array (row = y index)."""
        return np.asarray(cells, dtype=float).reshape(self.ny, self.nx)


@dataclass
class OverlapTable:
    """Geometry-only source/target intersection table."""

    lattice: CartesianLattice
    M: sp.csr_matrix  # (n_cells, n_nodes) integral operator
    coverage: np.ndarray  # (n_cells,) mapped area
    overlap_elems: list  # per target cell: np.ndarray of element indices
    overlap_areas: list  # per target cell: np.ndarray of intersection areas
    total_source_area: float

    @property
    def n_cells(self) -> int:
        return self.lattice.n_cells

    def coverage_fraction(self) -> np.ndarray:
        return self.coverage / self.lattice.areas()


def build_overlap(mesh: TriMesh, lattice: CartesianLattice,
                  tol_rel: float = 1e-14) -> OverlapTable:
    """Clip every element against every lattice cell it touches.

    The intersection polygon of a (convex) triangle with an axis-aligned rectangle
    is computed exactly with Sutherland-Hodgman; the P1 field integrates over the
    polygon as ``|P| * phi(centroid)`` because ``phi`` is linear inside the element.
    """
    n_cells = lattice.n_cells
    rows, cols, vals = [], [], []
    coverage = np.zeros(n_cells)
    overlap_elems: list[list[np.ndarray]] = [[] for _ in range(n_cells)]
    overlap_areas: list[list[np.ndarray]] = [[] for _ in range(n_cells)]

    grad = mesh.gradients()  # (M,3,2)
    areas = mesh.areas()
    total_src = float(areas.sum())
    area_tol = tol_rel * float(areas.mean())

    for e in range(mesh.n_elements):
        tri_xy = mesh.nodes[mesh.tri[e]]
        poly = [(float(p[0]), float(p[1])) for p in tri_xy]
        xmin, ymin = poly[0]
        xmax, ymax = poly[0]
        for px, py in poly[1:]:
            xmin, xmax = min(xmin, px), max(xmax, px)
            ymin, ymax = min(ymin, py), max(ymax, py)
        jx0 = int(np.clip(np.floor((xmin - lattice.x0) / lattice.dx), 0, lattice.nx - 1))
        jx1 = int(np.clip(np.floor((xmax - lattice.x0) / lattice.dx), 0, lattice.nx - 1))
        jy0 = int(np.clip(np.floor((ymin - lattice.y0) / lattice.dy), 0, lattice.ny - 1))
        jy1 = int(np.clip(np.floor((ymax - lattice.y0) / lattice.dy), 0, lattice.ny - 1))

        # nodal shape function values at the element centroid (for gradient terms)
        for jy in range(jy0, jy1 + 1):
            for jx in range(jx0, jx1 + 1):
                j = jy * lattice.nx + jx
                rect = lattice.rect(j)
                piece = clip_rect(poly, rect)
                if len(piece) < 3:
                    continue
                ap = polygon_area(piece)
                if ap <= area_tol:
                    continue
                # centroid of the sub-polygon -> local P1 value = sum_a N_a(centroid) phi_a
                pts = np.asarray(piece, dtype=float)
                cen = _polygon_centroid(pts)
                # shape functions at cen inside element e: N_a = 1/3 + grad_a . (cen - centroid_T)
                dc = cen - tri_xy.mean(axis=0)
                na = 1.0 / 3.0 + grad[e] @ dc  # (3,)
                for a_local in range(3):
                    w = ap * na[a_local]
                    if abs(w) > 0.0:
                        rows.append(j)
                        cols.append(int(mesh.tri[e, a_local]))
                        vals.append(float(w))
                coverage[j] += ap
                overlap_elems[j].append(e)
                overlap_areas[j].append(ap)

    M = sp.coo_matrix((vals, (rows, cols)), shape=(n_cells, mesh.n_nodes)).tocsr()
    return OverlapTable(
        lattice=lattice,
        M=M,
        coverage=coverage,
        overlap_elems=[np.asarray(v, dtype=int) for v in overlap_elems],
        overlap_areas=[np.asarray(v, dtype=float) for v in overlap_areas],
        total_source_area=total_src,
    )


def _polygon_centroid(pts: np.ndarray) -> np.ndarray:
    """Area centroid of a convex polygon (fan decomposition about vertex 0)."""
    p0 = pts[0]
    total_a = 0.0
    total_c = np.zeros(2)
    for k in range(1, len(pts) - 1):
        a = pts[k] - p0
        b = pts[k + 1] - p0
        ar = 0.5 * (a[0] * b[1] - a[1] * b[0])
        total_a += ar
        total_c += ar * (p0 + np.asarray(pts[k]) + np.asarray(pts[k + 1])) / 3.0
    if abs(total_a) < 1e-300:
        return pts.mean(axis=0)
    return total_c / total_a