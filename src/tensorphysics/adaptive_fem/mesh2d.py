"""2D triangular mesh primitives for adaptive h-refinement.

Scope
-----
This module owns the *mesh* half of the adaptive-finite-element capability:

* :class:`TriMesh` — container (nodes, triangles, generation).
* geometry/quality helpers — signed area, per-triangle angles, boundary edges,
  lumped nodal weights.
* :func:`refine_red_green` — conforming red-green h-refinement:
  a marked ("red") triangle is split 1->4 using its three edge midpoints, and
  the one-ring of unmarked neighbours receives a "green" closure split 1->2 or
  1->3 so that the mesh stays conforming (no hanging nodes) without propagating
  refinement recursively.
* :func:`flip_to_quality` — guarded Delaunay edge flipping used to keep the
  minimum interior angle bounded away from zero after green closures.

Design notes
------------
All vertex numbering is CCW.  Local edge ``k`` of a triangle ``(v0, v1, v2)``
is the edge ``(v_k, v_{(k+1) mod 3})``; this convention makes the closure
splits rotation-invariant (rotations of a CCW triple stay CCW).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

__all__ = [
    "TriMesh",
    "signed_areas",
    "triangle_angles",
    "min_angles",
    "edge_records",
    "boundary_edges",
    "nodal_weights",
    "refine_red_green",
    "flip_to_quality",
    "mesh_summary",
    "quad_to_tris",
    "structured_quad_mesh",
]


# ---------------------------------------------------------------------------
# container
# ---------------------------------------------------------------------------
@dataclass
class TriMesh:
    """A conforming 2D triangulation (nodes ``(n,2)``, triangles ``(m,3)``)."""

    nodes: np.ndarray
    tris: np.ndarray
    generation: int = 0
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.nodes = np.asarray(self.nodes, dtype=float)
        self.tris = np.asarray(self.tris, dtype=np.int64)
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 2:
            raise ValueError("nodes must have shape (n, 2)")
        if self.tris.ndim != 2 or self.tris.shape[1] != 3:
            raise ValueError("tris must have shape (m, 3)")
        if self.tris.size and (self.tris.min() < 0 or self.tris.max() >= len(self.nodes)):
            raise ValueError("triangle connectivity out of range")

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def n_tris(self) -> int:
        return int(self.tris.shape[0])

    @property
    def n_dof(self) -> int:
        """Scalar unknowns of a 2-DOF-per-node (u, v) discretisation."""
        return 2 * self.n_nodes

    def centroids(self) -> np.ndarray:
        return self.nodes[self.tris].mean(axis=1)

    def copy(self) -> "TriMesh":
        return TriMesh(self.nodes.copy(), self.tris.copy(), self.generation, dict(self.meta))


# ---------------------------------------------------------------------------
# geometry / quality
# ---------------------------------------------------------------------------
def signed_areas(nodes: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Signed area of each triangle (positive for CCW orientation)."""
    p0 = nodes[tris[:, 0]]
    p1 = nodes[tris[:, 1]]
    p2 = nodes[tris[:, 2]]
    return 0.5 * (
        (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
        - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1])
    )


def triangle_angles(nodes: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Interior angles in degrees, shape ``(m, 3)``; column ``i`` is at ``v_i``."""
    p = nodes[tris]
    a = np.linalg.norm(p[:, 1] - p[:, 2], axis=1)  # opposite v0
    b = np.linalg.norm(p[:, 0] - p[:, 2], axis=1)  # opposite v1
    c = np.linalg.norm(p[:, 0] - p[:, 1], axis=1)  # opposite v2
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_a = np.clip((b * b + c * c - a * a) / (2.0 * b * c), -1.0, 1.0)
        cos_b = np.clip((a * a + c * c - b * b) / (2.0 * a * c), -1.0, 1.0)
        cos_c = np.clip((a * a + b * b - c * c) / (2.0 * a * b), -1.0, 1.0)
    return np.degrees(np.stack([np.arccos(cos_a), np.arccos(cos_b), np.arccos(cos_c)], axis=1))


def min_angles(nodes: np.ndarray, tris: np.ndarray) -> np.ndarray:
    return triangle_angles(nodes, tris).min(axis=1)


def edge_records(tris: np.ndarray):
    """Unique undirected edges and their incidence.

    Returns ``(uniq (k,2), counts (k,), incidence (3, m))`` where
    ``incidence[j, i]`` is the unique-edge id of local edge ``j`` of triangle
    ``i``.  Boundary edges have ``counts == 1``; interior edges ``counts == 2``.
    """
    tris = np.asarray(tris, dtype=np.int64)
    raw = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    key = np.sort(raw, axis=1)
    uniq, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    return uniq, counts, inv.reshape(3, -1)


def boundary_edges(nodes: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Edges belonging to exactly one triangle, as ``(k,2)`` node-id pairs."""
    uniq, counts, _ = edge_records(tris)
    return uniq[counts == 1]


def nodal_weights(nodes: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Lumped nodal area weights ``sum_K |K|/3`` (mass-lumped L2 measure)."""
    area = np.abs(signed_areas(nodes, tris))
    w = np.zeros(len(nodes), dtype=float)
    for j in range(3):
        np.add.at(w, tris[:, j], area / 3.0)
    return w


def _min_angle_pts(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Minimum interior angle (degrees) of triangles given by vertex points."""

    def _ang(p, q, r):
        u = p - q
        w = r - q
        nu = np.linalg.norm(u, axis=1)
        nw = np.linalg.norm(w, axis=1)
        cosv = np.clip(np.sum(u * w, axis=1) / np.maximum(nu * nw, 1e-300), -1.0, 1.0)
        return np.degrees(np.arccos(cosv))

    ang = np.stack([_ang(a, b, c), _ang(b, c, a), _ang(c, a, b)], axis=1)
    return ang.min(axis=1)


def quad_to_tris(nodes: np.ndarray, quads: np.ndarray) -> np.ndarray:
    """Split ``(m,4)`` CCW quads into two triangles.

    Both diagonals are evaluated on the actual geometry and the split that
    maximises the minimum interior angle is kept — this keeps the initial
    (possibly sheared / mapped) meshes away from degenerate angles.
    """
    q = np.asarray(quads, dtype=np.int64)
    nodes = np.asarray(nodes, float)
    p0, p1, p2, p3 = (nodes[q[:, k]] for k in range(4))
    a_min = np.minimum(
        _min_angle_pts(p0, p1, p2), _min_angle_pts(p0, p2, p3)
    )
    b_min = np.minimum(
        _min_angle_pts(p0, p1, p3), _min_angle_pts(p1, p2, p3)
    )
    use_a = a_min >= b_min
    tri_a = np.stack([q[:, [0, 1, 2]], q[:, [0, 2, 3]]], axis=1)
    tri_b = np.stack([q[:, [0, 1, 3]], q[:, [1, 2, 3]]], axis=1)
    chosen = np.where(use_a[:, None, None], tri_a, tri_b)
    return chosen.reshape(-1, 3)


def mesh_summary(nodes: np.ndarray, tris: np.ndarray) -> dict:
    area = np.abs(signed_areas(nodes, tris))
    ang = triangle_angles(nodes, tris)
    _, counts, _ = edge_records(tris)
    return {
        "n_nodes": int(len(nodes)),
        "n_tris": int(len(tris)),
        "n_dof": int(2 * len(nodes)),
        "area": float(area.sum()),
        "min_area": float(area.min()) if area.size else 0.0,
        "min_angle_deg": float(ang.min()) if ang.size else 0.0,
        "max_angle_deg": float(ang.max()) if ang.size else 0.0,
        "n_boundary_edges": int(np.sum(counts == 1)),
        "n_nonmanifold_edges": int(np.sum(counts > 2)),
        "n_interior_edges": int(np.sum(counts == 2)),
        "ccw": bool(np.all(signed_areas(nodes, tris) > 0.0)),
    }

def structured_quad_mesh(x: np.ndarray, y: np.ndarray, map_fn=None) -> "TriMesh":
    """Structured grid over ``x`` x ``y`` (optionally mapped), split into T3."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    nx, ny = len(x) - 1, len(y) - 1
    X, Y = np.meshgrid(x, y, indexing="ij")
    nodes = np.stack([X.ravel(), Y.ravel()], axis=1)
    if map_fn is not None:
        nodes = map_fn(nodes)
    idx = np.arange((nx + 1) * (ny + 1)).reshape(nx + 1, ny + 1)
    i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    q = np.stack([idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]], axis=-1).reshape(-1, 4)
    return TriMesh(nodes, quad_to_tris(nodes, q))
