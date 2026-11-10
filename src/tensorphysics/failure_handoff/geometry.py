"""Polygon geometry primitives for the FEM→DEM failure-face handoff.

Everything here is deliberately independent of the FEM mesh data structures and
of the DEM particle data structures: the exporter (FEM side) and the consumer
(DEM side) both recompute face geometry **from vertex coordinates only**, so the
area-conservation gate compares two independent computations rather than one
number copied through a file.

Conventions
-----------
* A face is a planar polygon given as an ordered vertex loop (right-hand rule
  about the outward normal).
* ``polygon_area`` uses the Newell area vector, which is exact for planar
  polygons of any vertex count and degrades gracefully for slightly warped
  quadrilaterals (the residual warp error is reported separately).
* ``polygon_centroid`` is the area-weighted centroid (not the vertex mean), so a
  graded mesh does not bias the fragment centre of mass.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

__all__ = [
    "as_points",
    "unit",
    "newell_area_vector",
    "polygon_area",
    "polygon_centroid",
    "planarity_error",
    "extrude_prism",
    "equivalent_sphere_radius",
    "triangulated_area",
]


def as_points(vertices: Sequence[Sequence[float]]) -> np.ndarray:
    """Return ``vertices`` as a float ``(n, 3)`` array, validating the shape."""
    pts = np.asarray(vertices, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"vertices must be (n, 3), got {pts.shape}")
    if pts.shape[0] < 3:
        raise ValueError(f"a polygon needs at least 3 vertices, got {pts.shape[0]}")
    if not np.all(np.isfinite(pts)):
        raise ValueError("vertices contain non-finite coordinates")
    return pts


def unit(vector: Sequence[float]) -> np.ndarray:
    """Unit vector; raises on a zero/near-zero input."""
    v = np.asarray(vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(v))
    if norm <= 1e-300:
        raise ValueError("cannot normalise a zero vector")
    return v / norm


def newell_area_vector(vertices: Sequence[Sequence[float]]) -> np.ndarray:
    """Newell area vector ``sum_i r_i x r_{i+1}`` / 2 (magnitude = polygon area)."""
    pts = as_points(vertices)
    r0 = pts[0]
    acc = np.zeros(3)
    for i in range(1, pts.shape[0] - 1):  # fan triangulation, exact when planar
        acc += np.cross(pts[i] - r0, pts[i + 1] - r0)
    return 0.5 * acc


def polygon_area(vertices: Sequence[Sequence[float]]) -> float:
    """Area of a planar polygon given as an ordered vertex loop."""
    return float(np.linalg.norm(newell_area_vector(vertices)))


def triangulated_area(vertices: Sequence[Sequence[float]]) -> float:
    """Area as the sum of fan-triangle areas (strictly positive contributions).

    Used as an independent second estimate: it matches :func:`polygon_area` for a
    planar loop and differs only by the warp residual otherwise.
    """
    pts = as_points(vertices)
    h = sum(
        0.5 * float(np.linalg.norm(np.cross(pts[i] - pts[0], pts[i + 1] - pts[0])))
        for i in range(1, pts.shape[0] - 1)
    )
    return float(h)


def polygon_centroid(vertices: Sequence[Sequence[float]]) -> np.ndarray:
    """Area-weighted centroid of a planar polygon."""
    pts = as_points(vertices)
    ref = pts.min(axis=0)
    acc = np.zeros(3)
    weight = 0.0
    for i in range(1, pts.shape[0] - 1):
        a = pts[i] - ref
        b = pts[i + 1] - ref
        tri = float(np.linalg.norm(np.cross(a, b))) * 0.5
        acc += tri * (a + b) / 3.0
        weight += tri
    if weight <= 0.0:
        return pts.mean(axis=0)
    return ref + acc / weight


def planarity_error(vertices: Sequence[Sequence[float]]) -> float:
    """Max distance of a vertex from the best-fit polygon plane (metres)."""
    pts = as_points(vertices)
    normal_area = newell_area_vector(pts)
    norm = float(np.linalg.norm(normal_area))
    if norm <= 1e-300:
        return float("inf")
    n = normal_area / norm
    return float(np.max(np.abs((pts - pts.mean(axis=0)) @ n)))


def extrude_prism(
    vertices: Sequence[Sequence[float]], normal: Sequence[float], thickness: float
) -> np.ndarray:
    """Extrude a polygon along ``-normal`` by ``thickness`` → prism vertices.

    The prism keeps the fracture polygon as its outer face, so the DEM fragment's
    fracture surface is geometrically the FEM face by construction; the
    consumer-side gate re-derives the area from these coordinates.
    """
    if thickness <= 0.0:
        raise ValueError(f"thickness must be positive, got {thickness}")
    pts = as_points(vertices)
    offset = -thickness * unit(normal)
    return np.vstack([pts, pts + offset])


def equivalent_sphere_radius(volume_m3: float) -> float:
    """Radius of the sphere with the given volume."""
    if volume_m3 <= 0.0:
        raise ValueError(f"volume must be positive, got {volume_m3}")
    return float((3.0 * volume_m3 / (4.0 * math.pi)) ** (1.0 / 3.0))