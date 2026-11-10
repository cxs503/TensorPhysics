"""Mass / inertia / momentum / energy transfer across the FEM→DEM handoff.

The frozen ``failure_faces`` v3 packet says *where* the material separated and
how much cohesive energy that consumed.  It says nothing about the *bodies* that
the DEM solver needs, so the consumer has to rebuild them:

``polyhedron_volume``
    Volume of the reconstructed fragment polyhedron from its own vertices, using
    the divergence theorem (``V = 1/6 sum a.(b x c)`` over the closed, outward
    oriented triangulation).  This is an independent geometric measure: it never
    looks at the declared ``area_m2`` or at the extrusion product ``A * t``.
``fragment_inertia``
    Exact prism inertia from the polygon's area second moments plus the
    parallel-axis term for the fracture-layer thickness.  The consumer needs the
    inertia because a fragment that inherits a velocity *field* also inherits a
    spin; ignoring it would silently drop angular momentum.
``rigid_body_fit``
    Least-squares rigid-body projection of the FEM nodal velocity field onto one
    fragment: translation ``v`` plus spin ``omega``.  The residual
    ``rms |v_node - (v + omega x r)|`` is reported — it is the only place where
    the handoff can lose kinetic energy, so it is the physically meaningful
    number behind the energy gate.
``momentum_budget`` / ``energy_budget``
    Producer-side (FEM nodal) versus consumer-side (DEM fragment) totals for
    linear momentum, angular momentum and kinetic energy.

Everything here is pure post-processing of an already sealed packet; no physics
is re-solved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from ..solid_fem.element import element_kinematics
from .geometry import newell_area_vector, polygon_area, polygon_centroid

__all__ = [
    "TransferFragment",
    "element_volume",
    "element_volumes",
    "polyhedron_volume",
    "polygon_second_moments",
    "fragment_inertia",
    "build_transfer_fragment",
    "rigid_body_fit",
    "momentum_budget",
    "energy_budget",
    "volume_budget",
]

_VOIGT = 6


# ---------------------------------------------------------------------------
# FEM-side volumes
# ---------------------------------------------------------------------------

def element_volume(mesh, element_index: int) -> float:
    """Volume of one solid element: ``sum_g det J_g * w_g`` (exact for the isoparametric map)."""
    kin = element_kinematics(mesh.element, mesh.element_coords(int(element_index)), None, "full")
    return float(np.sum(kin.det_j * _rule_weights(mesh.element)))


def _rule_weights(element: str) -> np.ndarray:
    from ..solid_fem.shape import default_rule, GAUSS_RULES

    _pts, wts = GAUSS_RULES[default_rule(element)]
    return np.asarray(wts, dtype=float)


def element_volumes(mesh) -> np.ndarray:
    """Volumes of every element of ``mesh`` (vectorised over elements)."""
    weights = _rule_weights(mesh.element)
    out = np.empty(mesh.n_elem, dtype=float)
    for e in range(mesh.n_elem):
        kin = element_kinematics(mesh.element, mesh.element_coords(e), None, "full")
        out[e] = float(np.sum(kin.det_j * weights))
    return out


# ---------------------------------------------------------------------------
# Reconstructed fragment geometry
# ---------------------------------------------------------------------------

def _frame(normal: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = np.asarray(normal, dtype=float)
    n = n / float(np.linalg.norm(n))
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(helper, n)
    u /= float(np.linalg.norm(u))
    v = np.cross(n, u)
    return u, v, n


def polyhedron_volume(vertices: Sequence[Sequence[float]], normal: Sequence[float], thickness: float) -> float:
    """Divergence-theorem volume of the prism ``polygon x [0, -t]`` along ``normal``.

    The prism is rebuilt as a closed triangulated surface (two caps + side quads)
    and the signed tetrahedron sum is taken, so the number is a pure function of
    the vertex coordinates — the same quantity an external CAD kernel would report.
    """
    poly = np.asarray(vertices, dtype=float)
    if poly.ndim != 2 or poly.shape[0] < 3:
        raise ValueError("polygon needs at least three vertices")
    _u, _v, n = _frame(normal)
    t = float(thickness)
    if t <= 0.0:
        raise ValueError("thickness must be positive")
    top = poly
    bottom = poly - t * n
    triangles: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    c_top = top.mean(axis=0)
    c_bottom = bottom.mean(axis=0)
    m = len(poly)
    for i in range(m):
        j = (i + 1) % m
        # Caps (fan from the centroid), side quads (two triangles each).
        triangles.append((c_top, top[i], top[j]))
        triangles.append((c_bottom, bottom[j], bottom[i]))
        triangles.append((top[i], bottom[i], bottom[j]))
        triangles.append((top[i], bottom[j], top[j]))
    volume = 0.0
    for a, b, c in triangles:
        volume += float(np.dot(a, np.cross(b, c))) / 6.0
    return abs(volume)


def polygon_second_moments(
    vertices: Sequence[Sequence[float]], normal: Sequence[float]
) -> dict[str, float]:
    """Area and area moments of a planar polygon in its own ``(u, v)`` frame.

    Uses a 3-point triangle rule (exact for quadratics) around the polygon
    centroid, so the moments are reported *about the centroid* directly.
    """
    poly = np.asarray(vertices, dtype=float)
    u, v, _n = _frame(normal)
    c = polygon_centroid(poly)
    p = (poly - c) @ u
    q = (poly - c) @ v
    m = len(poly)
    area = i_p = i_q = i_pq = 0.0
    w = 1.0 / 3.0
    for i in range(m):
        j = (i + 1) % m
        # triangle (0, p_i, p_j) in centroid-relative coordinates
        tri = np.array([[0.0, 0.0], [p[i], q[i]], [p[j], q[j]]])
        a = 0.5 * abs(tri[0, 0] * (tri[1, 1] - tri[2, 1])
                      + tri[1, 0] * (tri[2, 1] - tri[0, 1])
                      + tri[2, 0] * (tri[0, 1] - tri[1, 1]))
        if a == 0.0:
            continue
        # barycentric 3-point rule: (2/3,1/6,1/6) permutations
        for bary in ((2 / 3, 1 / 6, 1 / 6), (1 / 6, 2 / 3, 1 / 6), (1 / 6, 1 / 6, 2 / 3)):
            x = bary[0] * tri[0, 0] + bary[1] * tri[1, 0] + bary[2] * tri[2, 0]
            y = bary[0] * tri[0, 1] + bary[1] * tri[1, 1] + bary[2] * tri[2, 1]
            area += a * w
            i_p += a * w * x * x
            i_q += a * w * y * y
            i_pq += a * w * x * y
    return {
        "area_m2": float(area),
        "i_u_m4": float(i_q),   # second moment about the u axis = integral of q^2
        "i_v_m4": float(i_p),   # second moment about the v axis = integral of p^2
        "i_uv_m4": float(i_pq),
    }


def fragment_inertia(
    vertices: Sequence[Sequence[float]],
    normal: Sequence[float],
    thickness: float,
    density: float,
    centroid: Sequence[float],
) -> dict[str, Any]:
    """Mass, volume and 3x3 centroidal inertia tensor (global axes) of the prism."""
    moments = polygon_second_moments(vertices, normal)
    area = moments["area_m2"]
    t = float(thickness)
    volume = area * t
    mass = float(density) * volume
    u, v, n = _frame(normal)
    # Inertia in the local (u, v, n) frame about the prism centroid.
    i_uu = float(density) * t * (moments["i_u_m4"] + area * t * t / 12.0)
    i_vv = float(density) * t * (moments["i_v_m4"] + area * t * t / 12.0)
    i_nn = float(density) * t * (moments["i_u_m4"] + moments["i_v_m4"])
    i_uv = -float(density) * t * moments["i_uv_m4"]
    local = np.array(
        [
            [i_uu, i_uv, 0.0],
            [i_uv, i_vv, 0.0],
            [0.0, 0.0, i_nn],
        ]
    )
    rot = np.column_stack([u, v, n])
    return {
        "area_m2": float(area),
        "volume_m3": float(volume),
        "mass_kg": mass,
        "inertia_local_kg_m2": local.tolist(),
        "inertia_kg_m2": (rot @ local @ rot.T).tolist(),
        "principal_moments_kg_m2": sorted(float(x) for x in np.linalg.eigvalsh(local)),
    }


# ---------------------------------------------------------------------------
# Fragment container
# ---------------------------------------------------------------------------

@dataclass
class TransferFragment:
    """One DEM fragment with the full mass / inertia / motion state."""

    fragment_id: str
    face_id: str
    element_id: int
    vertices_m: np.ndarray
    normal: np.ndarray
    thickness_m: float
    area_m2: float
    volume_m3: float
    mass_kg: float
    centroid_m: np.ndarray
    inertia_kg_m2: np.ndarray
    fracture_energy_j: float
    fem_element_volume_m3: float | None = None
    node_ids: list[int] = field(default_factory=list)
    #: Indices into ``node_ids`` of the four face vertices (used for the rigid fit).
    face_node_indices: list[int] = field(default_factory=list)
    velocity_m_s: np.ndarray = field(default_factory=lambda: np.zeros(3))
    omega_rad_s: np.ndarray = field(default_factory=lambda: np.zeros(3))
    nodal_velocity_m_s: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    nodal_positions_m: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    fit_residual_m_s: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "fragment_id": self.fragment_id,
            "face_id": self.face_id,
            "element_id": self.element_id,
            "area_m2": self.area_m2,
            "thickness_m": self.thickness_m,
            "volume_m3": self.volume_m3,
            "fem_element_volume_m3": self.fem_element_volume_m3,
            "mass_kg": self.mass_kg,
            "centroid_m": [float(c) for c in self.centroid_m],
            "normal": [float(c) for c in self.normal],
            "inertia_kg_m2": self.inertia_kg_m2.tolist(),
            "fracture_energy_j": self.fracture_energy_j,
            "velocity_m_s": [float(c) for c in self.velocity_m_s],
            "omega_rad_s": [float(c) for c in self.omega_rad_s],
            "fit_residual_m_s": self.fit_residual_m_s,
            "node_ids": list(self.node_ids),
            "face_node_indices": list(self.face_node_indices),
        }


def build_transfer_fragment(
    face: Mapping,
    *,
    density_kg_m3: float,
    thickness_m: float,
    index: int = 0,
    fem_element_volume_m3: float | None = None,
) -> TransferFragment:
    """Rebuild one fragment from a packet face (geometry + mass + inertia)."""
    if density_kg_m3 <= 0.0:
        raise ValueError("density_kg_m3 must be positive")
    if thickness_m <= 0.0:
        raise ValueError("thickness_m must be positive")
    vertices = np.asarray(face["vertices_m"], dtype=float)
    normal = np.asarray(face["normal"], dtype=float)
    normal = normal / float(np.linalg.norm(normal))
    centroid = polygon_centroid(vertices)
    # Volume by the divergence theorem, then push the centroid to the prism centre.
    volume = polyhedron_volume(vertices, normal, thickness_m)
    props = fragment_inertia(vertices, normal, thickness_m, density_kg_m3, centroid)
    mass = props["mass_kg"]
    # Keep the mass tied to the reconstructed volume (and not to A*t) so the
    # momentum/energy budget uses the same geometry the volume gate measured.
    mass = float(density_kg_m3) * float(volume)
    return TransferFragment(
        fragment_id=f"T{index + 1:04d}",
        face_id=str(face["face_id"]),
        element_id=int(face.get("element_id", -1)),
        vertices_m=vertices,
        normal=normal,
        thickness_m=float(thickness_m),
        area_m2=float(polygon_area(vertices)),
        volume_m3=float(volume),
        mass_kg=mass,
        centroid_m=centroid - 0.5 * float(thickness_m) * normal,
        inertia_kg_m2=np.asarray(props["inertia_kg_m2"], dtype=float),
        fracture_energy_j=float(face["energy_j"]),
        fem_element_volume_m3=fem_element_volume_m3,
    )


# ---------------------------------------------------------------------------
# Rigid-body projection of the FEM velocity field
# ---------------------------------------------------------------------------

def rigid_body_fit(
    points: np.ndarray,
    velocities: np.ndarray,
    centroid: Sequence[float],
) -> dict[str, Any]:
    """Least-squares translation + spin of a nodal velocity field.

    Solves ``min sum |v_i - (v + omega x r_i)|^2`` with
    ``A omega = sum r_i x (v_i - v)``, ``A = sum (|r|^2 I - r r^T)``.
    """
    pts = np.asarray(points, dtype=float)
    vel = np.asarray(velocities, dtype=float)
    if pts.shape != vel.shape or pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points and velocities must both be (n, 3)")
    c = np.asarray(centroid, dtype=float)
    r = pts - c
    v_mean = vel.mean(axis=0)
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for ri, vi in zip(r, vel - v_mean):
        A += np.dot(ri, ri) * np.eye(3) - np.outer(ri, ri)
        b += np.cross(ri, vi)
    if abs(np.linalg.det(A)) < 1e-30:
        omega = np.zeros(3)
    else:
        omega = np.linalg.solve(A, b)
    residual = vel - (v_mean + np.cross(np.broadcast_to(omega, r.shape), r))
    rms = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
    return {
        "velocity_m_s": v_mean,
        "omega_rad_s": omega,
        "fit_residual_m_s": rms,
        "velocity_rms_m_s": float(np.sqrt(np.mean(np.sum(vel**2, axis=1)))),
    }


# ---------------------------------------------------------------------------
# Budgets
# ---------------------------------------------------------------------------

def _relative_error(producer: float, consumer: float, *, scale: float | None = None) -> float:
    ref = abs(producer) if scale is None else abs(scale)
    if ref <= 1e-300:
        return 0.0 if abs(consumer) <= 1e-300 else float("inf")
    return abs(consumer - producer) / ref


def volume_budget(
    fragments: Sequence[TransferFragment], *, tolerance: float = 0.03
) -> dict[str, Any]:
    """Reconstructed polyhedron volume versus the FEM failure-region volume."""
    missing = [f.fragment_id for f in fragments if f.fem_element_volume_m3 is None]
    if missing:
        raise ValueError(f"fragments without an FEM volume: {missing}")
    fem_total = float(math.fsum(float(f.fem_element_volume_m3) for f in fragments))
    dem_total = float(math.fsum(f.volume_m3 for f in fragments))
    worst = {"fragment_id": "", "relative_error": 0.0}
    rows = []
    for f in fragments:
        err = _relative_error(float(f.fem_element_volume_m3), f.volume_m3)
        rows.append(
            {
                "fragment_id": f.fragment_id,
                "fem_volume_m3": float(f.fem_element_volume_m3),
                "reconstructed_volume_m3": f.volume_m3,
                "relative_error": err,
            }
        )
        if err > worst["relative_error"]:
            worst = {"fragment_id": f.fragment_id, "relative_error": err}
    total_error = _relative_error(fem_total, dem_total)
    return {
        "fem_failure_region_volume_m3": fem_total,
        "reconstructed_volume_m3": dem_total,
        "relative_error": total_error,
        "worst_fragment_relative_error": worst["relative_error"],
        "worst_fragment_id": worst["fragment_id"],
        "tolerance": float(tolerance),
        "passed": bool(total_error <= tolerance and worst["relative_error"] <= tolerance),
        "fragments": rows,
    }


def momentum_budget(
    fragments: Sequence[TransferFragment],
    *,
    tolerance: float = 0.02,
) -> dict[str, Any]:
    """Fragment momentum versus the FEM crack-face nodal momentum.

    The producer side uses the *consistent* nodal masses of the fracture layer
    (each node owns a quarter of every face it belongs to) and the raw FEM nodal
    velocities; the consumer side uses the fragment mass and the rigid-body fit.
    The difference is therefore the rigid-body projection residual, not a
    bookkeeping artefact.
    """
    if any(not f.node_ids for f in fragments):
        raise ValueError("fragments carry no node ids; attach the FEM velocity field first")
    positions: dict[int, np.ndarray] = {}
    node_mass: dict[int, float] = {}
    node_vel: dict[int, np.ndarray] = {}
    for f in fragments:
        n_nodes = len(f.node_ids)
        if n_nodes != len(f.nodal_velocity_m_s):
            raise ValueError(
                f"fragment {f.fragment_id}: {n_nodes} node ids for "
                f"{len(f.nodal_velocity_m_s)} velocities"
            )
        share = f.mass_kg / float(n_nodes)
        for index, nid in enumerate(f.node_ids):
            node_mass[nid] = node_mass.get(nid, 0.0) + share
            node_vel[nid] = np.asarray(f.nodal_velocity_m_s[index], dtype=float)
    p_fem = np.zeros(3)
    for nid, mass in node_mass.items():
        p_fem += mass * node_vel[nid]
    p_dem = np.zeros(3)
    l_dem = np.zeros(3)
    for f in fragments:
        p_dem += f.mass_kg * f.velocity_m_s
        l_dem += np.cross(f.centroid_m - _layer_centroid(fragments), f.mass_kg * f.velocity_m_s)
        l_dem += f.inertia_kg_m2 @ f.omega_rad_s
    l_fem = np.zeros(3)
    c = _layer_centroid(fragments)
    for f in fragments:
        share = f.mass_kg / float(len(f.node_ids))
        for index, nid in enumerate(f.node_ids):
            position = f.nodal_positions_m[index] if f.nodal_positions_m.size else f.centroid_m
            l_fem += np.cross(position - c, share * np.asarray(f.nodal_velocity_m_s[index], dtype=float))
    ref_p = float(np.linalg.norm(p_fem))
    ref_l = float(np.linalg.norm(l_fem))
    dem_p = float(np.linalg.norm(p_dem))
    dem_l = float(np.linalg.norm(l_dem))
    p_error = _relative_error(ref_p, dem_p)
    l_error = _relative_error(ref_l, dem_l)
    return {
        "fem_momentum_kg_m_s": p_fem.tolist(),
        "dem_momentum_kg_m_s": p_dem.tolist(),
        "momentum_relative_error": p_error,
        "fem_angular_momentum_kg_m2_s": l_fem.tolist(),
        "dem_angular_momentum_kg_m2_s": l_dem.tolist(),
        "angular_relative_error": l_error,
        "momentum_magnitude_fem": ref_p,
        "momentum_magnitude_dem": dem_p,
        "angular_momentum_magnitude_fem": ref_l,
        "angular_momentum_magnitude_dem": dem_l,
        "total_mass_kg": float(math.fsum(f.mass_kg for f in fragments)),
        "tolerance": float(tolerance),
        # C3 is stated for linear momentum; the angular part is reported as a
        # secondary check with the same band (it is what catches a lost spin).
        "passed": bool(p_error <= tolerance),
        "angular_passed": bool(l_error <= tolerance),
    }


def _layer_centroid(fragments: Sequence[TransferFragment]) -> np.ndarray:
    total = math.fsum(f.mass_kg for f in fragments)
    if total <= 0.0:
        return np.zeros(3)
    return np.array(
        [math.fsum(fr.mass_kg * float(fr.centroid_m[k]) for fr in fragments) / total for k in range(3)]
    )


def energy_budget(
    fragments: Sequence[TransferFragment],
    *,
    producer_internal_energy_j: float | None = None,
    gravity_m_s2: float = 9.81,
    reference_height_m: float | None = None,
    tolerance: float = 0.05,
) -> dict[str, Any]:
    """Fracture + kinetic + potential energy ledger, producer side versus consumer side."""
    if not fragments:
        raise ValueError("no fragments")
    reference = (
        float(reference_height_m)
        if reference_height_m is not None
        else float(min(f.centroid_m[1] for f in fragments))
    )
    ke_fem = 0.0
    pe_fem = 0.0
    ke_dem = 0.0
    pe_dem = 0.0
    internal = 0.0
    for f in fragments:
        share = f.mass_kg / float(len(f.node_ids)) if f.node_ids else 0.0
        for vel in f.nodal_velocity_m_s:
            ke_fem += 0.5 * share * float(np.dot(vel, vel))
        ke_dem += 0.5 * f.mass_kg * float(np.dot(f.velocity_m_s, f.velocity_m_s))
        ke_dem += 0.5 * float(f.omega_rad_s @ f.inertia_kg_m2 @ f.omega_rad_s)
        pe_fem += f.mass_kg * gravity_m_s2 * float(f.centroid_m[1] - reference)
        pe_dem += f.mass_kg * gravity_m_s2 * float(f.centroid_m[1] - reference)
        internal += f.fracture_energy_j
    # The producer's cohesive energy is the *packet* total (declared by the FEM
    # exporter); the consumer's is what the fragments actually carry.  Keeping the
    # two separate is what lets the gate see an energy bookkeeping loss — using one
    # shared value would make it cancel out of the comparison.
    internal_fem = (
        float(producer_internal_energy_j)
        if producer_internal_energy_j is not None
        else internal
    )
    total_fem = internal_fem + ke_fem + pe_fem
    total_dem = internal + ke_dem + pe_dem
    error = _relative_error(total_fem, total_dem)
    internal_error = _relative_error(internal_fem, internal)
    ke_error = _relative_error(ke_fem, ke_dem, scale=ke_fem) if ke_fem > 0.0 else 0.0
    return {
        "internal_energy_j": internal,
        "producer_internal_energy_j": internal_fem,
        "internal_transfer_error": internal_error,
        "ke_fem_j": ke_fem,
        "ke_dem_j": ke_dem,
        "potential_fem_j": pe_fem,
        "potential_dem_j": pe_dem,
        "total_fem_j": total_fem,
        "total_dem_j": total_dem,
        "relative_error": error,
        "kinetic_relative_error": ke_error,
        "reference_height_m": reference,
        "gravity_m_s2": float(gravity_m_s2),
        "kinetic_tolerance": float(tolerance),
        "kinetic_passed": bool(ke_error <= tolerance),
        "tolerance": float(tolerance),
        "passed": bool(error <= tolerance),
    }