"""FEM side of the handoff: fracture-face extraction and packet export.

The producer takes a solved FEM model, selects the element faces that sit on the
fracture plane, evaluates the traction acting on each of them and turns the
mechanical state into a ``failure_faces`` v3 packet.

What is exported per face
-------------------------
``vertices_m`` / ``normal`` / ``centroid_m`` / ``area_m2``
    Mesh-derived geometry.  Area uses the Newell area vector, so a warped
    quadrilateral is measured as the projected planar area.
``normal_traction_pa`` / ``shear_traction_pa``
    ``t = sigma . n`` split into ``sigma_n = n . t`` (mode I) and
    ``tau = |t - sigma_n n|`` (mode II/III lumped).
``mode_mixity``
    Energy-weighted mixity proxy ``psi = tau^2 / (sigma_n^2 + tau^2)``.  Squared
    tractions are the natural weight because the released energy of a cohesive
    zone scales with the traction squared.
``specific_energy_j_m2`` / ``energy_j``
    Mixed-mode toughness from the Benzeggagh-Kenane law
    ``G_c(psi) = G_Ic + (G_IIc - G_Ic) psi^eta`` times the face area.  This is the
    cohesive energy *consumed* by creating one unit of new surface — the quantity
    that a DEM fragment inherits as initial internal energy.

Failure selection
-----------------
:func:`extract_failure_faces` applies a tensile strength criterion on the crack
plane: a face fails when ``sigma_n >= sigma_t``.  Faces that do not reach the
strength stay in the packet's diagnostics (not in the packet) and their boundary
defines ``crack_front_m``.

Scope / limitations
-------------------
This is a *post-processing* extractor, not a crack-propagation solver: the stress
field is the linear-elastic FEM solution at the reported load level, and the
created area follows the strength criterion.  Crack advance, cohesive softening
and stress redistribution after each face fails are **not** modelled here; the
consequence is documented in ``docs/reports/p154w2_n03/REPORT.md`` §5.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from ..fracture_mechanics.bk_law import BKLaw
from .geometry import (
    newell_area_vector,
    planarity_error,
    polygon_area,
    polygon_centroid,
)
from .schema import SCHEMA_NAME, SCHEMA_VERSION, VERSION, seal, validate_packet

__all__ = [
    "HEX8_FACE_KEYS",
    "FractureMaterial",
    "FaceRecord",
    "FEMFractureResult",
    "element_faces",
    "faces_on_plane",
    "face_tractions",
    "mode_mixity",
    "specific_fracture_energy",
    "classify_failure_mode",
    "extract_failure_faces",
    "packet_from_result",
    "write_packet",
    "packet_to_json_text",
]

#: Local node loops of the Hex8 faces, wound right-handed about the outward normal.
HEX8_FACE_KEYS: Mapping[str, tuple[int, int, int, int]] = {
    "xi-": (0, 4, 7, 3),
    "xi+": (1, 2, 6, 5),
    "eta-": (0, 1, 5, 4),
    "eta+": (3, 7, 6, 2),
    "zeta-": (0, 3, 2, 1),
    "zeta+": (4, 5, 6, 7),
}

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2, "xi": 0, "eta": 1, "zeta": 2}


@dataclass(frozen=True)
class FractureMaterial:
    """Quasi-brittle material data for the fracture-face energy accounting."""

    E_pa: float
    nu: float
    sigma_t_pa: float
    G_Ic_j_m2: float
    G_IIc_j_m2: float
    eta_bk: float = 1.5

    def __post_init__(self) -> None:
        if self.E_pa <= 0 or self.sigma_t_pa <= 0:
            raise ValueError("E_pa and sigma_t_pa must be positive")
        if not 0.0 <= self.nu < 0.5:
            raise ValueError(f"nu must be in [0, 0.5), got {self.nu}")
        if self.G_Ic_j_m2 <= 0 or self.G_IIc_j_m2 <= 0:
            raise ValueError("fracture toughnesses must be positive")
        if self.eta_bk <= 0:
            raise ValueError("eta_bk must be positive")

    @property
    def bk_law(self) -> BKLaw:
        return BKLaw(G_Ic=self.G_Ic_j_m2, G_IIc=self.G_IIc_j_m2, eta=self.eta_bk)

    def effective_toughness(self, mixity: float) -> float:
        return float(self.bk_law.effective_toughness(min(max(mixity, 0.0), 1.0)))


@dataclass
class FaceRecord:
    """One extracted fracture face (geometry + mechanical state)."""

    face_id: str
    element_id: int
    vertices_m: np.ndarray
    normal: np.ndarray
    centroid_m: np.ndarray
    area_m2: float
    sigma_n_pa: float
    tau_pa: float
    mixity: float
    specific_energy_j_m2: float
    energy_j: float
    failed: bool = True

    def to_packet_face(self) -> dict:
        return {
            "face_id": self.face_id,
            "element_id": int(self.element_id),
            "vertices_m": [[float(c) for c in p] for p in self.vertices_m],
            "normal": [float(c) for c in self.normal],
            "centroid_m": [float(c) for c in self.centroid_m],
            "area_m2": float(self.area_m2),
            "specific_energy_j_m2": float(self.specific_energy_j_m2),
            "energy_j": float(self.energy_j),
            "failure_mode": classify_failure_mode(self.mixity),
            "mode_mixity": float(self.mixity),
            "normal_traction_pa": float(self.sigma_n_pa),
            "shear_traction_pa": float(self.tau_pa),
        }


@dataclass
class FEMFractureResult:
    """Result of the extraction step, ready to be exported as a packet."""

    source_solver: str
    timestamp: str
    faces: list[FaceRecord]
    material: FractureMaterial
    crack_front_m: list[list[float]] = field(default_factory=list)
    source_model_id: str | None = None
    fracture_layer_thickness_m: float | None = None
    diagnostics: dict = field(default_factory=dict)

    @property
    def failed_faces(self) -> list[FaceRecord]:
        return [face for face in self.faces if face.failed]

    @property
    def area_total_m2(self) -> float:
        return float(math.fsum(face.area_m2 for face in self.failed_faces))

    @property
    def energy_total_j(self) -> float:
        return float(math.fsum(face.energy_j for face in self.failed_faces))


# ---------------------------------------------------------------------------
# Mesh face enumeration
# ---------------------------------------------------------------------------

def element_faces(mesh) -> Iterable[tuple[int, str, np.ndarray]]:
    """Yield ``(element_id, key, vertices)`` for every Hex8 face of ``mesh``."""
    if str(mesh.element).lower() != "hex8":
        raise ValueError(f"only hex8 meshes are supported, got {mesh.element!r}")
    for e in range(mesh.n_elem):
        coords = mesh.element_coords(e)
        for key, loop in HEX8_FACE_KEYS.items():
            yield e, key, coords[list(loop)]


def faces_on_plane(
    mesh,
    axis: str = "x",
    coord: float | None = None,
    *,
    tol: float = 1e-9,
) -> list[tuple[int, str, np.ndarray]]:
    """Faces whose four vertices all lie on the plane ``axis = coord``.

    ``coord`` defaults to the mesh mid-plane, which is where a three-point-bend
    crack plane lives.
    """
    if axis not in _AXIS_INDEX:
        raise ValueError(f"unknown axis {axis!r}")
    index = _AXIS_INDEX[axis]
    if coord is None:
        coord = 0.5 * (mesh.nodes[:, index].min() + mesh.nodes[:, index].max())
    out = []
    for e, key, verts in element_faces(mesh):
        if np.all(np.abs(verts[:, index] - coord) <= tol):
            out.append((e, key, verts))
    return out


# ---------------------------------------------------------------------------
# Mechanics: traction split, mixity, toughness
# ---------------------------------------------------------------------------

def face_tractions(stress_voigt: Sequence[float], normal: Sequence[float]) -> tuple[float, float]:
    """Split ``t = sigma . n`` into ``(sigma_n, tau)`` in Pa.

    ``stress_voigt`` is ``[xx, yy, zz, xy, yz, zx]`` (the order used by
    :mod:`tensorphysics.solid_fem`).
    """
    sxx, syy, szz, sxy, syz, szx = (float(s) for s in stress_voigt)
    tensor = np.array(
        [[sxx, sxy, szx], [sxy, syy, syz], [szx, syz, szz]], dtype=float
    )
    n = np.asarray(normal, dtype=float)
    n = n / float(np.linalg.norm(n))
    traction = tensor @ n
    sigma_n = float(traction @ n)
    tau = float(np.linalg.norm(traction - sigma_n * n))
    return sigma_n, tau


def mode_mixity(sigma_n: float, tau: float) -> float:
    """Energy-weighted mixity ``tau^2 / (sigma_n^2 + tau^2)`` clamped to [0, 1]."""
    n_energy = max(sigma_n, 0.0) ** 2
    t_energy = max(tau, 0.0) ** 2
    total = n_energy + t_energy
    if total <= 0.0:
        return 0.0
    return float(min(max(t_energy / total, 0.0), 1.0))


def specific_fracture_energy(mixity: float, material: FractureMaterial) -> float:
    """Benzeggagh-Kenane mixed-mode toughness [J/m²] at the given mixity."""
    return material.effective_toughness(mixity)


def classify_failure_mode(mixity: float, *, mode_i_limit: float = 0.1, mode_ii_limit: float = 0.9) -> str:
    if mixity < mode_i_limit:
        return "mode_I"
    if mixity > mode_ii_limit:
        return "mode_II"
    return "mixed_mode"


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _element_average_stress(stress_by_element: np.ndarray, element_id: int) -> np.ndarray:
    return np.asarray(stress_by_element[element_id], dtype=float).reshape(6)


def extract_failure_faces(
    mesh,
    stress_by_element: np.ndarray,
    *,
    material: FractureMaterial,
    axis: str = "x",
    coord: float | None = None,
    timestamp: str,
    source_solver: str = "TensorFEM/hex8-3point-bend",
    source_model_id: str | None = None,
    fracture_layer_thickness_m: float | None = None,
    keep_negative_side: bool = True,
    ligament_min_coord: tuple[str, float] | None = None,
    tol: float = 1e-9,
) -> FEMFractureResult:
    """Extract the fracture faces on the mid-span crack plane.

    Only faces of elements on one side of the plane are taken (``keep_negative_side``)
    so each created surface is counted exactly once — the classical
    "two solids share one new pair of surfaces, take one" ledger.
    """
    index = _AXIS_INDEX[axis]
    faces = faces_on_plane(mesh, axis=axis, coord=coord, tol=tol)
    if ligament_min_coord is not None:
        lig_axis, lig_min = ligament_min_coord
        lig_index = _AXIS_INDEX[lig_axis]
        faces = [f for f in faces if float(f[2][:, lig_index].min()) >= lig_min - 1e-9]
    if not faces:
        raise ValueError("no element faces found on the requested crack plane")
    if coord is None:
        coord = 0.5 * (mesh.nodes[:, index].min() + mesh.nodes[:, index].max())

    records: list[FaceRecord] = []
    for counter, (element_id, _key, verts) in enumerate(faces):
        centroid_elem = mesh.element_coords(element_id).mean(axis=0)
        side = centroid_elem[index] - coord
        if keep_negative_side and side > 0.0:
            continue
        if (not keep_negative_side) and side < 0.0:
            continue
        area_vector = newell_area_vector(verts)
        area = float(np.linalg.norm(area_vector))
        if area <= 0.0:
            continue
        normal = area_vector / area
        # Outward from the kept solid: the Hex8 face loops are wound right-handed
        # about the outward normal, so this only re-asserts the convention.
        if keep_negative_side and normal[index] < 0.0:
            normal = -normal
        elif (not keep_negative_side) and normal[index] > 0.0:
            normal = -normal
        stress = _element_average_stress(stress_by_element, element_id)
        sigma_n, tau = face_tractions(stress, normal)
        mixity = mode_mixity(sigma_n, tau)
        specific = specific_fracture_energy(mixity, material)
        failed = bool(sigma_n >= material.sigma_t_pa)
        records.append(
            FaceRecord(
                face_id=f"F{counter + 1:04d}",
                element_id=element_id,
                vertices_m=verts,
                normal=normal,
                centroid_m=polygon_centroid(verts),
                area_m2=area,
                sigma_n_pa=sigma_n,
                tau_pa=tau,
                mixity=mixity,
                specific_energy_j_m2=specific,
                energy_j=specific * area,
                failed=failed,
            )
        )

    if not any(face.failed for face in records):
        raise ValueError(
            "no face reached the tensile strength "
            f"({material.sigma_t_pa:.3g} Pa); max sigma_n = "
            f"{max(face.sigma_n_pa for face in records):.4g} Pa"
        )

    failed = [face for face in records if face.failed]
    surviving = [face for face in records if not face.failed]
    crack_front = _crack_front(failed, surviving)

    diagnostics = {
        "candidate_face_count": len(records),
        "failed_face_count": len(failed),
        "surviving_face_count": len(records) - len(failed),
        "ligament_area_m2": float(math.fsum(face.area_m2 for face in records)),
        "ligament_failed_fraction": float(
            math.fsum(face.area_m2 for face in failed) / math.fsum(f.area_m2 for f in records)
        ),
        "max_normal_traction_pa": float(max(face.sigma_n_pa for face in records)),
        "min_normal_traction_pa": float(min(face.sigma_n_pa for face in records)),
        "mean_mode_mixity": float(np.mean([face.mixity for face in records])),
        "mean_mode_mixity_failed": float(np.mean([face.mixity for face in failed])),
        "max_mode_mixity_failed": float(max(face.mixity for face in failed)),
        "failed_specific_energy_range_j_m2": [
            float(min(face.specific_energy_j_m2 for face in failed)),
            float(max(face.specific_energy_j_m2 for face in failed)),
        ],
        "crack_front_edge_count": len(crack_front),
        "max_planarity_error_m": float(
            max(planarity_error(face.vertices_m) for face in records)
        ),
        "specific_energy_range_j_m2": [
            float(min(face.specific_energy_j_m2 for face in records)),
            float(max(face.specific_energy_j_m2 for face in records)),
        ],
    }
    return FEMFractureResult(
        source_solver=source_solver,
        timestamp=timestamp,
        faces=records,
        material=material,
        crack_front_m=[[float(c) for c in p] for p in crack_front],
        source_model_id=source_model_id,
        fracture_layer_thickness_m=fracture_layer_thickness_m,
        diagnostics=diagnostics,
    )


def _edge_key(a: np.ndarray, b: np.ndarray, *, tol: int = 9) -> tuple:
    """Canonical (order-independent) key of a mesh edge, quantised to ``tol`` digits."""
    pa = tuple(round(float(c), tol) for c in a)
    pb = tuple(round(float(c), tol) for c in b)
    return (pa, pb) if pa <= pb else (pb, pa)


def _crack_front(
    failed: Sequence["FaceRecord"], surviving: Sequence["FaceRecord"]
) -> list[np.ndarray]:
    """Crack-front polyline = edges shared by a failed and a surviving face.

    That is the mechanical definition of a front: the boundary of the newly created
    surface inside the still-intact ligament.  When the ligament is fully broken
    (no surviving face) the front degenerates to the outer boundary of the failed
    region, which is reported as such (empty list) rather than invented.
    """
    failed_edges = {}
    for face in failed:
        verts = face.vertices_m
        for i in range(len(verts)):
            key = _edge_key(verts[i], verts[(i + 1) % len(verts)])
            failed_edges.setdefault(key, (verts[i], verts[(i + 1) % len(verts)]))
    surviving_edges = set()
    for face in surviving:
        verts = face.vertices_m
        for i in range(len(verts)):
            surviving_edges.add(_edge_key(verts[i], verts[(i + 1) % len(verts)]))
    front: list[np.ndarray] = []
    seen = set()
    for key, (a, b) in failed_edges.items():
        if key not in surviving_edges:
            continue
        for point in (a, b):
            q = tuple(round(float(c), 9) for c in point)
            if q in seen:
                continue
            seen.add(q)
            front.append(np.asarray(point, dtype=float))
    return front


# ---------------------------------------------------------------------------
# Packet construction / export
# ---------------------------------------------------------------------------

def packet_from_result(result: FEMFractureResult) -> dict:
    """Build (and seal) the v3 ``failure_faces`` packet from an extraction result."""
    failed = result.failed_faces
    packet = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "version": VERSION,
        "source_solver": result.source_solver,
        "timestamp": result.timestamp,
        "units": {"length": "m", "area": "m2", "energy": "J", "traction": "Pa"},
        "face_count": len(failed),
        "area_total_m2": float(math.fsum(face.area_m2 for face in failed)),
        "energy_total_j": float(math.fsum(face.energy_j for face in failed)),
        "faces": [face.to_packet_face() for face in failed],
    }
    if result.source_model_id is not None:
        packet["source_model_id"] = result.source_model_id
    if result.fracture_layer_thickness_m is not None:
        packet["fracture_layer_thickness_m"] = float(result.fracture_layer_thickness_m)
    if result.crack_front_m:
        packet["crack_front_m"] = result.crack_front_m
    sealed = seal(packet)
    validate_packet(sealed)
    return sealed


def packet_to_json_text(packet: Mapping) -> str:
    """Deterministic JSON text of a packet (sorted keys, fixed indent, newline).

    Determinism is what makes the idempotency criterion (C5) meaningful: the same
    FEM result always produces byte-identical text, hence an identical SHA-256.
    """
    return json.dumps(packet, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def write_packet(packet: Mapping, path: str | Path) -> str:
    """Write the packet and return its ``sha256`` (also the file's audit link)."""
    validate_packet(packet)
    text = packet_to_json_text(packet)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return str(packet["sha256"])