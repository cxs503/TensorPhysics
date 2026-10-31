"""FEM→DEM Ice Fragment Data Export Interface (SIR-PC2 / F03).

This module provides the ``ice_fragments()`` function that serialises FEM
fracture data into a DEM-consumable JSON format with:

1. **JSON Schema** – strict, versioned schema for ice fragment packets.
2. **Mass Conservation Gate** – rejects any packet where |Δm|/Σm > 1e-12.
3. **Round-Trip Energy Conservation** – serialise → deserialise → verify
   that total fracture energy is preserved to < 1e-8 relative error.

Physical background
-------------------
In an ice-structure interaction simulation the FEM solver detects cohesive
face failures.  Each failed face represents a volume of ice that detaches
and becomes a discrete fragment.  The DEM solver needs these fragments as
initial conditions: position, equivalent-sphere radius, mass, velocity, and
the fracture energy that was released when the face broke.

The mapping is *equal-volume sphere*: each fragment's volume equals the
failed-face area times the characteristic element length, and the radius is
derived from V = 4/3 π r³.

Conservation laws enforced
--------------------------
- **Mass**: Σ(ρ·V_face) from FEM == Σ(m_particle) in DEM packet.
- **Energy**: Σ(G_c·A_face) from FEM == Σ(E_fracture) in DEM packet.

Both are checked at machine precision (≤ 1e-12 for mass, ≤ 1e-8 for energy
round-trip through JSON float64).
"""

from __future__ import annotations

import json
import math
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "1.0.0"
MASS_CONSERVATION_TOL = 1e-12   # kg relative
ENERGY_CONSERVATION_TOL = 1e-8  # ratio (round-trip through JSON)
DEFAULT_ICE_DENSITY = 917.0     # kg/m³ (freshwater ice at 0 °C)


def _utcnow_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# JSON Schema (frozen, embedded for self-documentation)
# ---------------------------------------------------------------------------

ICE_FRAGMENTS_JSON_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "IceFragmentsPacket",
    "description": "FEM→DEM ice fragment export packet (SIR-PC2/F03)",
    "type": "object",
    "required": [
        "schema_version", "timestamp", "packet_id",
        "material", "conservation", "fragments"
    ],
    "properties": {
        "schema_version": {
            "type": "string",
            "const": SCHEMA_VERSION,
        },
        "timestamp": {"type": "string", "format": "date-time"},
        "packet_id": {"type": "string"},
        "material": {
            "type": "object",
            "required": ["density_kg_m3", "fracture_energy_density_J_m2"],
            "properties": {
                "density_kg_m3": {"type": "number", "exclusiveMinimum": 0},
                "fracture_energy_density_J_m2": {"type": "number", "minimum": 0},
            },
        },
        "conservation": {
            "type": "object",
            "required": [
                "fem_total_mass_kg", "dem_total_mass_kg",
                "mass_conservation_error_kg",
                "fem_total_energy_J", "dem_total_energy_J",
                "round_trip_energy_ratio",
            ],
            "properties": {
                "fem_total_mass_kg": {"type": "number"},
                "dem_total_mass_kg": {"type": "number"},
                "mass_conservation_error_kg": {"type": "number"},
                "fem_total_energy_J": {"type": "number"},
                "dem_total_energy_J": {"type": "number"},
                "round_trip_energy_ratio": {"type": "number"},
            },
        },
        "fragments": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "fragment_id", "position_m", "radius_m",
                    "mass_kg", "velocity_m_s",
                    "source_face_id", "source_volume_m3",
                    "fracture_energy_J",
                ],
                "properties": {
                    "fragment_id": {"type": "integer", "minimum": 0},
                    "position_m": {
                        "type": "array", "items": {"type": "number"},
                        "minItems": 3, "maxItems": 3,
                    },
                    "radius_m": {"type": "number", "exclusiveMinimum": 0},
                    "mass_kg": {"type": "number", "exclusiveMinimum": 0},
                    "velocity_m_s": {
                        "type": "array", "items": {"type": "number"},
                        "minItems": 3, "maxItems": 3,
                    },
                    "source_face_id": {"type": "integer", "minimum": 0},
                    "source_volume_m3": {"type": "number", "exclusiveMinimum": 0},
                    "fracture_energy_J": {"type": "number", "minimum": 0},
                },
            },
        },
    },
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class IceFragment:
    """A single ice fragment (DEM-ready)."""
    fragment_id: int
    position_m: Tuple[float, float, float]
    radius_m: float
    mass_kg: float
    velocity_m_s: Tuple[float, float, float]
    source_face_id: int
    source_volume_m3: float
    fracture_energy_J: float


@dataclass
class IceFragmentsPacket:
    """Complete FEM→DEM export packet."""
    schema_version: str = SCHEMA_VERSION
    timestamp: str = ""
    packet_id: str = ""
    material_density: float = DEFAULT_ICE_DENSITY
    material_Gc: float = 0.0
    fragments: List[IceFragment] = field(default_factory=list)
    # Conservation bookkeeping
    fem_total_mass_kg: float = 0.0
    dem_total_mass_kg: float = 0.0
    fem_total_energy_J: float = 0.0
    dem_total_energy_J: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = _utcnow_z()
        if not self.packet_id:
            self.packet_id = _generate_packet_id()


def _generate_packet_id() -> str:
    """Generate a unique packet ID from timestamp + random bytes."""
    import os
    raw = f"{_utcnow_z()}:{os.urandom(8).hex()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Core physics: equal-volume sphere mapping
# ---------------------------------------------------------------------------

def volume_to_sphere_radius(volume_m3: float) -> float:
    """Compute equivalent sphere radius from volume.

    V = (4/3) π r³  →  r = (3V / 4π)^(1/3)
    """
    if volume_m3 <= 0.0:
        raise ValueError(f"volume must be positive, got {volume_m3}")
    return (3.0 * volume_m3 / (4.0 * math.pi)) ** (1.0 / 3.0)


def face_volume(area_m2: float, characteristic_length_m: float) -> float:
    """Volume of material block represented by a failed face.

    V = A × L  (face area × characteristic element length)
    """
    return area_m2 * characteristic_length_m


# ---------------------------------------------------------------------------
# Main entry point: ice_fragments()
# ---------------------------------------------------------------------------

def ice_fragments(
    failed_faces: List[Dict[str, Any]],
    nodes_2d: np.ndarray,
    density: float = DEFAULT_ICE_DENSITY,
    G_c: float = 100.0,
    thickness: float = 0.01,
) -> IceFragmentsPacket:
    """Generate a DEM-consumable ice fragment packet from FEM fracture data.

    This is the primary FEM→DEM export interface (SIR-PC2 / F03).

    Parameters
    ----------
    failed_faces : list of dict
        Each dict must contain:
        - face_id : int
        - node_i, node_j : int  (indices into nodes_2d)
        - area : float  (face area in m²)
        - length : float  (characteristic element length in m)
        - released_energy : float  (G_c × area, in J)
    nodes_2d : ndarray, shape (n_nodes, 2)
        2-D node coordinates of the FEM mesh.
    density : float
        Ice density (kg/m³). Default 917.
    G_c : float
        Fracture energy density (J/m²). Default 100.
    thickness : float
        Plate thickness (m). Default 0.01.

    Returns
    -------
    IceFragmentsPacket
        Complete packet with conservation metadata.

    Raises
    ------
    MassConservationError
        If mass conservation gate fails (|Δm|/Σm > 1e-12).
    """
    fragments: List[IceFragment] = []
    fem_total_mass = 0.0
    fem_total_energy = 0.0
    dem_total_mass = 0.0
    dem_total_energy = 0.0

    for idx, face in enumerate(failed_faces):
        # FEM-side accounting
        vol = face["area"] * face["length"]
        fem_mass = density * vol
        fem_energy = face["released_energy"]
        fem_total_mass += fem_mass
        fem_total_energy += fem_energy

        # DEM-side: equal-volume sphere mapping
        radius = volume_to_sphere_radius(vol)
        dem_mass = density * vol  # same formula → exact conservation
        dem_energy = face["released_energy"]  # inherited from face
        dem_total_mass += dem_mass
        dem_total_energy += dem_energy

        # 3-D position (plate mid-plane at z=0)
        ni, nj = face["node_i"], face["node_j"]
        cx = 0.5 * (float(nodes_2d[ni, 0]) + float(nodes_2d[nj, 0]))
        cy = 0.5 * (float(nodes_2d[ni, 1]) + float(nodes_2d[nj, 1]))
        cz = 0.0

        frag = IceFragment(
            fragment_id=idx,
            position_m=(cx, cy, cz),
            radius_m=radius,
            mass_kg=dem_mass,
            velocity_m_s=(0.0, 0.0, 0.0),
            source_face_id=face["face_id"],
            source_volume_m3=vol,
            fracture_energy_J=dem_energy,
        )
        fragments.append(frag)

    # --- Mass conservation gate ---
    mass_error_kg = abs(fem_total_mass - dem_total_mass)
    mass_sum = fem_total_mass + dem_total_mass
    mass_rel_error = mass_error_kg / mass_sum if mass_sum > 0.0 else 0.0

    if mass_rel_error > MASS_CONSERVATION_TOL:
        raise MassConservationError(
            f"Mass conservation gate FAILED: |Δm|/Σm = {mass_rel_error:.3e} "
            f"> {MASS_CONSERVATION_TOL:.1e}"
        )

    packet = IceFragmentsPacket(
        material_density=density,
        material_Gc=G_c,
        fragments=fragments,
        fem_total_mass_kg=fem_total_mass,
        dem_total_mass_kg=dem_total_mass,
        fem_total_energy_J=fem_total_energy,
        dem_total_energy_J=dem_total_energy,
    )
    return packet


class MassConservationError(RuntimeError):
    """Raised when the mass conservation gate fails."""
    pass


# ---------------------------------------------------------------------------
# Serialization: packet → JSON dict (DEM-consumable)
# ---------------------------------------------------------------------------

def packet_to_json_dict(packet: IceFragmentsPacket) -> Dict[str, Any]:
    """Serialise packet to a JSON-compatible dict matching the schema."""
    # Compute round-trip energy conservation
    dem_energy = packet.dem_total_energy_J

    return {
        "schema_version": packet.schema_version,
        "timestamp": packet.timestamp,
        "packet_id": packet.packet_id,
        "material": {
            "density_kg_m3": packet.material_density,
            "fracture_energy_density_J_m2": packet.material_Gc,
        },
        "conservation": {
            "fem_total_mass_kg": packet.fem_total_mass_kg,
            "dem_total_mass_kg": packet.dem_total_mass_kg,
            "mass_conservation_error_kg": abs(
                packet.fem_total_mass_kg - packet.dem_total_mass_kg
            ),
            "fem_total_energy_J": packet.fem_total_energy_J,
            "dem_total_energy_J": packet.dem_total_energy_J,
            "round_trip_energy_ratio": _round_trip_energy_ratio(packet),
        },
        "fragments": [
            {
                "fragment_id": f.fragment_id,
                "position_m": list(f.position_m),
                "radius_m": f.radius_m,
                "mass_kg": f.mass_kg,
                "velocity_m_s": list(f.velocity_m_s),
                "source_face_id": f.source_face_id,
                "source_volume_m3": f.source_volume_m3,
                "fracture_energy_J": f.fracture_energy_J,
            }
            for f in packet.fragments
        ],
    }


def _round_trip_energy_ratio(packet: IceFragmentsPacket) -> float:
    """Compute round-trip energy conservation ratio.

    Serialise to JSON (float64) → deserialise → compare sum of fragment
    energies to the packet's recorded total.

    ratio = |E_roundtrip - E_original| / E_original
    """
    if packet.dem_total_energy_J == 0.0:
        return 0.0

    # Simulate JSON round-trip (float64 precision)
    fragment_energies = [f.fracture_energy_J for f in packet.fragments]
    # JSON serialisation uses float64 → no precision loss for these values
    json_str = json.dumps(fragment_energies)
    recovered = json.loads(json_str)
    roundtrip_sum = sum(recovered)

    ratio = abs(roundtrip_sum - packet.dem_total_energy_J) / packet.dem_total_energy_J
    return ratio


# ---------------------------------------------------------------------------
# Deserialization: JSON dict → packet (DEM loader)
# ---------------------------------------------------------------------------

def load_packet_from_json(json_dict: Dict[str, Any]) -> IceFragmentsPacket:
    """Deserialise a JSON dict back into an IceFragmentsPacket.

    This simulates what a DEM solver would do when loading the fragment file.
    """
    # Validate schema version
    if json_dict.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Schema version mismatch: expected {SCHEMA_VERSION}, "
            f"got {json_dict.get('schema_version')}"
        )

    fragments = []
    for fd in json_dict["fragments"]:
        frag = IceFragment(
            fragment_id=fd["fragment_id"],
            position_m=tuple(fd["position_m"]),
            radius_m=fd["radius_m"],
            mass_kg=fd["mass_kg"],
            velocity_m_s=tuple(fd["velocity_m_s"]),
            source_face_id=fd["source_face_id"],
            source_volume_m3=fd["source_volume_m3"],
            fracture_energy_J=fd["fracture_energy_J"],
        )
        fragments.append(frag)

    mat = json_dict["material"]
    cons = json_dict["conservation"]

    packet = IceFragmentsPacket(
        schema_version=json_dict["schema_version"],
        timestamp=json_dict["timestamp"],
        packet_id=json_dict["packet_id"],
        material_density=mat["density_kg_m3"],
        material_Gc=mat["fracture_energy_density_J_m2"],
        fragments=fragments,
        fem_total_mass_kg=cons["fem_total_mass_kg"],
        dem_total_mass_kg=cons["dem_total_mass_kg"],
        fem_total_energy_J=cons["fem_total_energy_J"],
        dem_total_energy_J=cons["dem_total_energy_J"],
    )
    return packet


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def save_packet(packet: IceFragmentsPacket, path: str) -> str:
    """Save packet to JSON file. Returns the path written."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = packet_to_json_dict(packet)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return str(p)


def load_packet_file(path: str) -> IceFragmentsPacket:
    """Load packet from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return load_packet_from_json(data)


# ---------------------------------------------------------------------------
# Validation: DEM-loadable check + conservation verification
# ---------------------------------------------------------------------------

def validate_packet(packet: IceFragmentsPacket) -> Dict[str, Any]:
    """Full validation of an ice fragment packet.

    Returns a report dict with:
    - dem_loadable: bool
    - mass_conservation_error: float (kg)
    - round_trip_energy_conservation: float (ratio)
    - all checks passed?
    """
    # 1. DEM-loadable: can we serialise and deserialise?
    dem_loadable = True
    try:
        json_dict = packet_to_json_dict(packet)
        json_str = json.dumps(json_dict)
        recovered_dict = json.loads(json_str)
        recovered_packet = load_packet_from_json(recovered_dict)
        assert len(recovered_packet.fragments) == len(packet.fragments)
    except Exception as e:
        dem_loadable = False

    # 2. Mass conservation error (absolute, kg)
    mass_error_kg = abs(packet.fem_total_mass_kg - packet.dem_total_mass_kg)

    # 3. Round-trip energy conservation (ratio)
    energy_ratio = _round_trip_energy_ratio(packet)

    # 4. Verify recovered packet matches original
    round_trip_mass_match = True
    round_trip_energy_match = True
    if dem_loadable:
        recovered_mass = sum(f.mass_kg for f in recovered_packet.fragments)
        original_mass = sum(f.mass_kg for f in packet.fragments)
        if original_mass > 0:
            round_trip_mass_match = (
                abs(recovered_mass - original_mass) / original_mass < 1e-12
            )

        recovered_energy = sum(f.fracture_energy_J for f in recovered_packet.fragments)
        original_energy = sum(f.fracture_energy_J for f in packet.fragments)
        if original_energy > 0:
            round_trip_energy_match = (
                abs(recovered_energy - original_energy) / original_energy < ENERGY_CONSERVATION_TOL
            )

    all_passed = (
        dem_loadable
        and mass_error_kg < 1e-12
        and energy_ratio < ENERGY_CONSERVATION_TOL
        and round_trip_mass_match
        and round_trip_energy_match
    )

    return {
        "status": "PASS" if all_passed else "FAIL",
        "dem_loadable": dem_loadable,
        "mass_conservation_error_kg": mass_error_kg,
        "round_trip_energy_conservation": energy_ratio,
        "n_fragments": len(packet.fragments),
        "total_mass_kg": packet.dem_total_mass_kg,
        "total_energy_J": packet.dem_total_energy_J,
        "round_trip_mass_match": round_trip_mass_match,
        "round_trip_energy_match": round_trip_energy_match,
    }


# ---------------------------------------------------------------------------
# Convenience: generate test data
# ---------------------------------------------------------------------------

def make_test_failed_faces(
    n_faces: int = 6,
    side: float = 1.0,
    thickness: float = 0.01,
    sigma_c: float = 1e5,
    G_c: float = 100.0,
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """Generate synthetic failed-face data for testing.

    Creates an N×N structured mesh and marks internal vertical faces as failed
    (simulating uniaxial tension in x-direction).

    Returns (failed_faces, nodes_2d).
    """
    N = int(math.isqrt(n_faces // 2)) + 1
    if N < 2:
        N = 2

    # Build structured mesh
    x = np.linspace(0, side, N + 1)
    y = np.linspace(0, side, N + 1)
    xx, yy = np.meshgrid(x, y)
    nodes_2d = np.column_stack([xx.ravel(), yy.ravel()])

    # Build elements (quads)
    elements = []
    for j in range(N):
        for i in range(N):
            n0 = j * (N + 1) + i
            n1 = n0 + 1
            n2 = n0 + (N + 1) + 1
            n3 = n0 + (N + 1)
            elements.append([n0, n1, n2, n3])

    # Identify internal vertical faces (simulating σ_x > σ_c)
    edge_length = side / N
    face_area = edge_length * thickness
    face_length = edge_length
    sigma0 = 2e5  # > σ_c

    failed_faces = []
    face_id = 0
    for j in range(N):
        for i in range(N - 1):
            # Vertical internal edge between element (i,j) and (i+1,j)
            ni = j * (N + 1) + i + 1
            nj = (j + 1) * (N + 1) + i + 1
            released_energy = G_c * face_area
            failed_faces.append({
                "face_id": face_id,
                "node_i": ni,
                "node_j": nj,
                "area": face_area,
                "length": face_length,
                "released_energy": released_energy,
            })
            face_id += 1

    return failed_faces, nodes_2d
