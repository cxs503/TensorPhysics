"""DEM side of the handoff: build initial fragments from a ``failure_faces`` packet.

Consumption rules
-----------------
1. The consumer **re-derives** every face area from the vertex coordinates in the
   packet (Newell area vector).  The declared ``area_m2`` is only used for the
   conservation gate; a packet whose declared areas disagree with its own geometry
   is rejected, so a corrupted producer cannot hide behind its own numbers.
2. Each face becomes one polyhedral fragment: the fracture polygon extruded along
   ``-normal`` by the fracture-layer thickness (the mesh element size normal to the
   crack plane).  The fracture surface of the fragment is therefore exactly the FEM
   face; the other prism walls are the "free" surfaces created by the separation.
3. The fragment inherits ``energy_j`` as initial *internal* (fracture) energy, and
   mass from ``rho * V`` with ``V = A * t_layer``.
4. Optional clump representation: each prism is filled with ``n`` rigidly bonded
   spheres whose volumes sum **exactly** to the prism volume — a mass-partition
   check that no sphere mass is invented or lost when a DEM code needs a clump
   instead of a polyhedron (rigid bonding makes overlapping spheres admissible).

Nothing here re-solves physics: the DEM side only converts a sealed packet into
initial conditions plus the bookkeeping needed by the conservation gates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from .geometry import (
    equivalent_sphere_radius,
    extrude_prism,
    polygon_area,
    polygon_centroid,
    triangulated_area,
)
from .schema import validate_packet

__all__ = [
    "Fragment",
    "FragmentSet",
    "recompute_face_area",
    "fragment_from_face",
    "fragments_from_packet",
    "clump_spheres",
    "point_in_polygon",
]


def recompute_face_area(vertices: Sequence[Sequence[float]]) -> float:
    """Area of a packet face, recomputed from coordinates only (DEM side)."""
    return polygon_area(vertices)


def point_in_polygon(point: Sequence[float], polygon: Sequence[Sequence[float]]) -> bool:
    """Crossing-number point-in-polygon test in the polygon's own 2-D projection."""
    poly = np.asarray(polygon, dtype=float)
    p = np.asarray(point, dtype=float)
    # Project onto the two dominant in-plane axes to stay robust for tilted faces.
    normal = np.cross(poly[1] - poly[0], poly[2] - poly[0])
    drop = int(np.argmax(np.abs(normal)))
    axes = [i for i in range(3) if i != drop]
    q = p[axes]
    pts = poly[:, axes]
    inside = False
    j = len(pts) - 1
    for i in range(len(pts)):
        yi, yj = pts[i, 1], pts[j, 1]
        xi, xj = pts[i, 0], pts[j, 0]
        if (yi > q[1]) != (yj > q[1]):
            x_cross = (xj - xi) * (q[1] - yi) / (yj - yi) + xi
            if q[0] < x_cross:
                inside = not inside
        j = i
    return inside


@dataclass
class Fragment:
    """One DEM fragment derived from one FEM failure face."""

    fragment_id: str
    face_id: str
    fracture_polygon_m: np.ndarray
    normal: np.ndarray
    volume_m3: float
    mass_kg: float
    centroid_m: np.ndarray
    thickness_m: float
    area_m2: float
    area_m2_triangulated: float
    fracture_energy_j: float
    specific_energy_j_m2: float
    equivalent_radius_m: float
    clump_spheres_m: list[tuple[np.ndarray, float]] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "fragment_id": self.fragment_id,
            "face_id": self.face_id,
            "area_m2": self.area_m2,
            "area_m2_triangulated": self.area_m2_triangulated,
            "thickness_m": self.thickness_m,
            "volume_m3": self.volume_m3,
            "mass_kg": self.mass_kg,
            "centroid_m": [float(c) for c in self.centroid_m],
            "normal": [float(c) for c in self.normal],
            "fracture_energy_j": self.fracture_energy_j,
            "specific_energy_j_m2": self.specific_energy_j_m2,
            "equivalent_radius_m": self.equivalent_radius_m,
            "clump_sphere_count": len(self.clump_spheres_m),
            "clump_volume_m3": float(
                sum(4.0 / 3.0 * math.pi * r**3 for _c, r in self.clump_spheres_m)
            ),
        }


@dataclass
class FragmentSet:
    """The DEM initial condition set plus the totals the gates compare against."""

    fragments: list[Fragment]
    density_kg_m3: float
    layer_thickness_m: float
    packet_digest: str

    @property
    def count(self) -> int:
        return len(self.fragments)

    @property
    def area_total_m2(self) -> float:
        return float(math.fsum(f.area_m2 for f in self.fragments))

    @property
    def area_total_triangulated_m2(self) -> float:
        return float(math.fsum(f.area_m2_triangulated for f in self.fragments))

    @property
    def volume_total_m3(self) -> float:
        return float(math.fsum(f.volume_m3 for f in self.fragments))

    @property
    def mass_total_kg(self) -> float:
        return float(math.fsum(f.mass_kg for f in self.fragments))

    @property
    def internal_energy_total_j(self) -> float:
        return float(math.fsum(f.fracture_energy_j for f in self.fragments))

    @property
    def clump_volume_total_m3(self) -> float:
        return float(
            math.fsum(
                4.0 / 3.0 * math.pi * r**3
                for f in self.fragments
                for _c, r in f.clump_spheres_m
            )
        )

    def as_dict(self) -> dict:
        return {
            "fragment_count": self.count,
            "density_kg_m3": self.density_kg_m3,
            "layer_thickness_m": self.layer_thickness_m,
            "packet_digest": self.packet_digest,
            "area_total_m2": self.area_total_m2,
            "area_total_triangulated_m2": self.area_total_triangulated_m2,
            "volume_total_m3": self.volume_total_m3,
            "mass_total_kg": self.mass_total_kg,
            "internal_energy_total_j": self.internal_energy_total_j,
            "clump_volume_total_m3": self.clump_volume_total_m3,
            "fragments": [f.as_dict() for f in self.fragments],
        }


def clump_spheres(
    polygon: np.ndarray,
    normal: np.ndarray,
    thickness_m: float,
    volume_m3: float,
    count: int,
) -> list[tuple[np.ndarray, float]]:
    """Fill the prism with ``count`` equal spheres whose volumes sum to ``volume_m3``.

    Positions are grid points inside the prism; if the grid is too coarse for the
    requested count, the remaining spheres are placed along the prism axis through
    the polygon centroid.  The radius follows from the exact volume partition, so
    the clump carries no mass error by construction.
    """
    if count < 1:
        raise ValueError("count must be >= 1")
    radius = (volume_m3 / count * 3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)
    centroid = polygon_centroid(polygon)
    lo = polygon.min(axis=0)
    hi = polygon.max(axis=0)
    step = max(2.0 * radius, 1e-12)
    positions: list[np.ndarray] = []
    nx = int(math.ceil((hi[0] - lo[0]) / step)) + 1
    ny = int(math.ceil((hi[1] - lo[1]) / step)) + 1
    nz = max(1, int(math.ceil(thickness_m / step)))
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                candidate = np.array(
                    [
                        lo[0] + (ix + 0.5) * (hi[0] - lo[0]) / nx,
                        lo[1] + (iy + 0.5) * (hi[1] - lo[1]) / ny,
                        lo[2] + (iz + 0.5) * (hi[2] - lo[2]) / nz,
                    ]
                )
                if not point_in_polygon(candidate, polygon):
                    continue
                # place the sphere centre half a layer inside the prism
                positions.append(candidate - 0.5 * thickness_m * normal)
                if len(positions) >= count:
                    break
            if len(positions) >= count:
                break
        if len(positions) >= count:
            break
    while len(positions) < count:
        offset = -thickness_m * (0.5 + 0.5 * (len(positions) - count / 2) / count)
        positions.append(centroid + offset * normal)
    return [(pos, radius) for pos in positions[:count]]


def fragment_from_face(
    face: Mapping,
    *,
    density_kg_m3: float,
    layer_thickness_m: float,
    clump_count: int = 0,
    index: int = 0,
) -> Fragment:
    """Build one polyhedral fragment (+ optional clump) from a packet face."""
    if density_kg_m3 <= 0.0:
        raise ValueError("density_kg_m3 must be positive")
    if layer_thickness_m <= 0.0:
        raise ValueError("layer_thickness_m must be positive")
    vertices = np.asarray(face["vertices_m"], dtype=float)
    normal = np.asarray(face["normal"], dtype=float)
    normal = normal / float(np.linalg.norm(normal))
    area = recompute_face_area(vertices)
    area_tri = triangulated_area(vertices)
    volume = area * layer_thickness_m
    mass = density_kg_m3 * volume
    centroid = polygon_centroid(vertices) - 0.5 * layer_thickness_m * normal
    spheres: list[tuple[np.ndarray, float]] = []
    if clump_count > 0:
        spheres = clump_spheres(vertices, normal, layer_thickness_m, volume, clump_count)
    return Fragment(
        fragment_id=f"D{index + 1:04d}",
        face_id=str(face["face_id"]),
        fracture_polygon_m=vertices,
        normal=normal,
        volume_m3=volume,
        mass_kg=mass,
        centroid_m=centroid,
        thickness_m=layer_thickness_m,
        area_m2=area,
        area_m2_triangulated=area_tri,
        fracture_energy_j=float(face["energy_j"]),
        specific_energy_j_m2=float(face["specific_energy_j_m2"]),
        equivalent_radius_m=equivalent_sphere_radius(volume),
        clump_spheres_m=spheres,
    )


def fragments_from_packet(
    packet: Mapping,
    *,
    density_kg_m3: float,
    layer_thickness_m: float | None = None,
    clump_count: int = 0,
    validate: bool = True,
) -> FragmentSet:
    """Consume a ``failure_faces`` packet into a DEM initial condition set."""
    if validate:
        validate_packet(packet)
    thickness = layer_thickness_m
    if thickness is None:
        thickness = packet.get("fracture_layer_thickness_m")
    if thickness is None:
        raise ValueError(
            "no fracture layer thickness: pass layer_thickness_m or include "
            "fracture_layer_thickness_m in the packet"
        )
    fragments = [
        fragment_from_face(
            face,
            density_kg_m3=density_kg_m3,
            layer_thickness_m=float(thickness),
            clump_count=clump_count,
            index=i,
        )
        for i, face in enumerate(packet["faces"])
    ]
    return FragmentSet(
        fragments=fragments,
        density_kg_m3=float(density_kg_m3),
        layer_thickness_m=float(thickness),
        packet_digest=str(packet.get("sha256", "")),
    )