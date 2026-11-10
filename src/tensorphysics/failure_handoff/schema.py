"""Frozen implementation-level schema for ``failure_faces.json`` (v3.0.0).

History of the interface
------------------------
``failure_faces`` v1 / v2 (see :mod:`tensorphysics.campaign.schema_defs`) are
*document-level capacity packets*: one record per face carrying
``face_id``/``normal``/``area_m2``/``capacity_n`` (plus v2's ``centroid_m``,
``failure_mode`` ...).  They were sufficient to negotiatiate an interface but they
carry no face geometry and no fracture energy, so a DEM consumer could never
initialise fragments from them — the interface was frozen on paper only
(PANORAMA §5.3).

``failure_faces`` **v3.0.0** is the implementation-level successor:

* every face carries its vertex loop, so the consumer recomputes area/centroid
  from coordinates instead of trusting a declared number;
* every face carries the specific fracture energy and the released energy, so the
  DEM fragments inherit a physical initial internal energy;
* the packet is *sealed*: a SHA-256 over the canonical JSON of the payload
  (``sha256`` key excluded) makes the handoff tamper-evident and idempotent;
* ``version`` is a semver string; any change to the field set **must** bump it,
  and :data:`SUPPORTED_VERSIONS` decides what a consumer accepts.

Version upgrade policy
----------------------
:v1/v2 → v3: admissible through :func:`upgrade_legacy_packet` (lossy: the legacy
  packet has no geometry, so a synthetic triangular patch is generated and the
  face is flagged ``geometry_synthetic``; energies from ``capacity_n`` are *not*
  invented — they stay zero unless an explicit fracture-energy density is given).
:v3 → v4: not implemented; a consumer must reject it (``SchemaError``) rather
  than silently ignoring new fields.

Validation
----------
:func:`validate_packet` runs three layers:

1. the closed JSON-Schema subset validator of :mod:`tensorphysics.campaign.schema`
   (dependency-free, used in CI); plus
2. ``jsonschema`` (draft 2020-12) when the package is importable — the same
   schema document, so the two layers cannot drift; plus
3. semantic invariants that no JSON Schema can express (totals vs. element sums,
   unit normals, winding vs. declared normal, seal digest, unique face ids).
"""

from __future__ import annotations

import copy
import hashlib
from typing import Any, Mapping

import numpy as np

from ..campaign.schema import (
    SchemaError,
    canonical_json,
    payload_digest,
    validate as _validate_subset,
)
from .geometry import newell_area_vector, polygon_area, polygon_centroid

__all__ = [
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "VERSION",
    "SUPPORTED_VERSIONS",
    "LEGACY_LEGACY_VERSIONS",
    "FAILURE_FACES_SCHEMA",
    "SCHEMA_FINGERPRINT",
    "LEGACY_MODE_MAP",
    "FailureFaceSchemaError",
    "packet_digest",
    "seal",
    "verify_seal",
    "validate_packet",
    "upgrade_legacy_packet",
]

SCHEMA_NAME = "failure_faces"
SCHEMA_VERSION = 3
VERSION = "3.0.0"
SUPPORTED_VERSIONS = ("3.0.0",)
#: Legacy packet versions that :func:`upgrade_legacy_packet` accepts.
LEGACY_LEGACY_VERSIONS = (1, 2)


class FailureFaceSchemaError(SchemaError):
    """A ``failure_faces`` packet violates the frozen v3 contract."""


_VEC3 = {
    "type": "array",
    "minItems": 3,
    "maxItems": 3,
    "items": {"type": "number"},
}

_POLYGON = {
    "type": "array",
    "minItems": 3,
    "maxItems": 8,
    "items": _VEC3,
}

_FACE = {
    "type": "object",
    "required": [
        "face_id",
        "vertices_m",
        "normal",
        "centroid_m",
        "area_m2",
        "specific_energy_j_m2",
        "energy_j",
        "failure_mode",
        "mode_mixity",
        "normal_traction_pa",
        "shear_traction_pa",
    ],
    "properties": {
        "face_id": {"type": "string", "minLength": 1},
        "element_id": {"type": "integer", "minimum": 0},
        "vertices_m": _POLYGON,
        "normal": _VEC3,
        "centroid_m": _VEC3,
        "area_m2": {"type": "number", "exclusiveMinimum": 0.0},
        "specific_energy_j_m2": {"type": "number", "minimum": 0.0},
        "energy_j": {"type": "number", "minimum": 0.0},
        "failure_mode": {"enum": ["mode_I", "mixed_mode", "mode_II"]},
        "mode_mixity": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "normal_traction_pa": {"type": "number"},
        "shear_traction_pa": {"type": "number", "minimum": 0.0},
        "geometry_synthetic": {"type": "boolean"},
        "energy_synthetic": {"type": "boolean"},
    },
    "additionalProperties": False,
}

FAILURE_FACES_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://tensor.ecosystem/schemas/failure_faces/3.0.0",
    "title": "failure_faces.v3 implementation-level fracture-face packet",
    "type": "object",
    "required": [
        "schema_name",
        "schema_version",
        "version",
        "source_solver",
        "timestamp",
        "units",
        "face_count",
        "area_total_m2",
        "energy_total_j",
        "faces",
        "sha256",
    ],
    "properties": {
        "schema_name": {"const": SCHEMA_NAME},
        "schema_version": {"const": SCHEMA_VERSION},
        "version": {"const": VERSION},
        "source_solver": {"type": "string", "minLength": 1},
        "source_model_id": {"type": "string", "minLength": 1},
        "timestamp": {
            "type": "string",
            "pattern": r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$",
        },
        "units": {
            "type": "object",
            "required": ["length", "area", "energy", "traction"],
            "properties": {
                "length": {"const": "m"},
                "area": {"const": "m2"},
                "energy": {"const": "J"},
                "traction": {"const": "Pa"},
            },
            "additionalProperties": False,
        },
        "fracture_layer_thickness_m": {"type": "number", "exclusiveMinimum": 0.0},
        "face_count": {"type": "integer", "minimum": 1},
        "area_total_m2": {"type": "number", "exclusiveMinimum": 0.0},
        "energy_total_j": {"type": "number", "minimum": 0.0},
        "crack_front_m": {"type": "array", "minItems": 2, "items": _VEC3},
        "faces": {"type": "array", "minItems": 1, "items": _FACE},
        "sha256": {"type": "string", "pattern": r"^[0-9a-f]{64}$"},
    },
    "additionalProperties": False,
}

#: SHA-256 of the canonical JSON of :data:`FAILURE_FACES_SCHEMA`.
#: ``tests/test_failure_handoff_schema.py`` pins it: editing the schema without
#: bumping ``VERSION`` and re-pinning breaks the consumer test path.
SCHEMA_FINGERPRINT = hashlib.sha256(
    canonical_json(FAILURE_FACES_SCHEMA).encode("utf-8")
).hexdigest()


def packet_digest(packet: Mapping[str, Any]) -> str:
    """SHA-256 over the canonical payload with the ``sha256`` field removed."""
    body = {k: v for k, v in packet.items() if k != "sha256"}
    return payload_digest(body)


def seal(packet: Mapping[str, Any]) -> dict[str, Any]:
    """Return a sealed copy of ``packet`` (``sha256`` recomputed)."""
    sealed = copy.deepcopy(dict(packet))
    sealed.pop("sha256", None)
    sealed["sha256"] = packet_digest(sealed)
    return sealed


def verify_seal(packet: Mapping[str, Any]) -> bool:
    """True when the stored ``sha256`` matches the payload."""
    stored = packet.get("sha256")
    if not isinstance(stored, str):
        return False
    return stored == packet_digest(packet)


def _fail(path: str, message: str) -> None:
    raise FailureFaceSchemaError(f"{path}: {message}")


def _reference_validators(schema: Mapping[str, Any], packet: Any) -> None:
    """``jsonschema`` draft-2020-12 validation when the package is available."""
    try:  # pragma: no cover - exercised only when jsonschema is installed
        import jsonschema
    except Exception:  # pragma: no cover - dependency-free fallback
        return
    try:  # pragma: no cover
        jsonschema.Draft202012Validator(schema).validate(packet)
    except jsonschema.ValidationError as exc:  # pragma: no cover
        raise FailureFaceSchemaError(
            f"jsonschema ${'/'.join(str(p) for p in exc.absolute_path)}: {exc.message}"
        ) from exc


def validate_packet(packet: Any, *, check_seal: bool = True) -> dict[str, Any]:
    """Validate a ``failure_faces`` v3 packet; returns it unchanged when valid.

    Raises :class:`FailureFaceSchemaError` on any violation.  The three layers are
    described in the module docstring.
    """
    if not isinstance(packet, Mapping):
        _fail("$", f"packet must be a mapping, got {type(packet).__name__}")

    # Layer 0: version gate before anything else (a v4 consumer contract must not
    # be silently downgraded).
    version = packet.get("version")
    if version is not None and version not in SUPPORTED_VERSIONS:
        _fail(
            "$.version",
            f"unsupported packet version {version!r}; this consumer accepts "
            f"{list(SUPPORTED_VERSIONS)} (a version bump requires an explicit "
            f"consumer upgrade)",
        )

    # Layer 1 + 2: structural JSON Schema.  The subset validator is re-raised as
    # this interface's own error type so callers only need one except clause.
    try:
        _validate_subset(FAILURE_FACES_SCHEMA, packet)
    except SchemaError as exc:
        raise FailureFaceSchemaError(str(exc)) from exc
    _reference_validators(FAILURE_FACES_SCHEMA, packet)

    # Layer 3: semantic invariants.
    if check_seal and not verify_seal(packet):
        _fail("$.sha256", "seal digest does not match the payload (tampered or stale)")

    faces = packet["faces"]
    if packet["face_count"] != len(faces):
        _fail("$.face_count", f"declared {packet['face_count']} != {len(faces)} faces")

    ids = [face["face_id"] for face in faces]
    if len(set(ids)) != len(ids):
        duplicated = sorted({i for i in ids if ids.count(i) > 1})
        _fail("$.faces", f"duplicate face_id values: {duplicated}")

    areas = []
    energies = []
    for index, face in enumerate(faces):
        path = f"$.faces[{index}]"
        normal = np.asarray(face["normal"], dtype=float)
        if abs(float(np.linalg.norm(normal)) - 1.0) > 1e-9:
            _fail(f"{path}.normal", f"normal is not unit length ({np.linalg.norm(normal):.12f})")
        verts = face["vertices_m"]
        area = polygon_area(verts)
        declared = float(face["area_m2"])
        if abs(area - declared) > 1e-9 * max(declared, 1.0):
            _fail(f"{path}.area_m2", f"declared {declared} != polygon area {area}")
        area_vector = newell_area_vector(verts)
        if float(np.dot(area_vector, normal)) <= 0.0:
            _fail(f"{path}.normal", "vertex winding is counter to the declared normal")
        centroid = polygon_centroid(verts)
        if float(np.linalg.norm(centroid - np.asarray(face["centroid_m"], dtype=float))) > 1e-9:
            _fail(f"{path}.centroid_m", "declared centroid != polygon centroid")
        areas.append(declared)
        energies.append(float(face["energy_j"]))

    area_total = float(packet["area_total_m2"])
    energy_total = float(packet["energy_total_j"])
    area_sum = float(np.sum(areas))
    energy_sum = float(np.sum(energies))
    if abs(area_sum - area_total) > 1e-9 * max(area_total, 1.0):
        _fail("$.area_total_m2", f"total {area_total} != sum of faces {area_sum}")
    if abs(energy_sum - energy_total) > 1e-9 * max(energy_total, 1.0e-12):
        _fail("$.energy_total_j", f"total {energy_total} != sum of faces {energy_sum}")

    return dict(packet)


# ---------------------------------------------------------------------------
# Version upgrade: document-level v1/v2 capacity packet → implementation v3
# ---------------------------------------------------------------------------

#: Legacy (v1/v2) failure vocabulary → v3 fracture-mode vocabulary.  The legacy
#: modes describe the *structural action* at the face, v3 describes the *crack
#: opening mode*, so the mapping is a lossy semantic downgrade (documented).
LEGACY_MODE_MAP: Mapping[str, str] = {
    "bending": "mixed_mode",
    "shear": "mode_II",
    "crushing": "mixed_mode",
    "buckling": "mixed_mode",
}


def upgrade_legacy_packet(
    legacy: Mapping[str, Any],
    *,
    timestamp: str,
    source_solver: str,
    fracture_energy_density_j_m2: float | None = None,
    synthetic_edge_m: float = 0.05,
) -> dict[str, Any]:
    """Upgrade a v1/v2 capacity packet to a sealed v3 packet (lossy, documented).

    The legacy packet carries ``area_m2``/``normal``/``capacity_n`` but **no**
    geometry, so a square patch of the declared area is synthesised in the face
    plane through the legacy ``centroid_m`` (v2) or the origin (v1).  Both the
    face and its energy are flagged ``geometry_synthetic`` / ``energy_synthetic``:
    the packet stays schema-valid and auditable, but a consumer must treat the
    geometry as a placeholder rather than a mesh-derived surface.
    """
    src_version = legacy.get("schema_version", legacy.get("version"))
    # Accept both the numeric version of the legacy schema and its semver string.
    numeric = src_version
    if isinstance(numeric, str):
        try:
            numeric = int(numeric.split(".")[0])
        except (TypeError, ValueError):
            numeric = None
    if numeric not in LEGACY_LEGACY_VERSIONS:
        raise FailureFaceSchemaError(
            f"$.schema_version: {src_version!r} is not a legacy failure_faces "
            f"(v{LEGACY_LEGACY_VERSIONS[0]}/v{LEGACY_LEGACY_VERSIONS[-1]}) packet"
        )

    faces = []
    area_total = 0.0
    energy_total = 0.0
    for index, face in enumerate(legacy.get("faces", [])):
        area = float(face["area_m2"])
        normal = np.asarray(face["normal"], dtype=float)
        normal = normal / float(np.linalg.norm(normal))
        centre = np.asarray(face.get("centroid_m", (0.0, 0.0, 0.0)), dtype=float)
        # Build a right-handed in-plane basis (u, v) for a square patch of the
        # declared area; the half-edge is sqrt(area)/2.
        helper = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(helper, normal))) > 0.9:
            helper = np.array([1.0, 0.0, 0.0])
        u = np.cross(normal, helper)
        u /= float(np.linalg.norm(u))
        v = np.cross(normal, u)
        half = 0.5 * float(np.sqrt(area))
        verts = [
            centre - half * u - half * v,
            centre + half * u - half * v,
            centre + half * u + half * v,
            centre - half * u + half * v,
        ]
        specific = 0.0 if fracture_energy_density_j_m2 is None else float(
            fracture_energy_density_j_m2
        )
        energy = specific * polygon_area(verts)
        capacity = float(face.get("capacity_n", 0.0))
        faces.append(
            {
                "face_id": str(face["face_id"]),
                "vertices_m": [list(map(float, p)) for p in verts],
                "normal": list(map(float, normal)),
                "centroid_m": list(map(float, polygon_centroid(verts))),
                "area_m2": polygon_area(verts),
                "specific_energy_j_m2": specific,
                "energy_j": energy,
                "failure_mode": LEGACY_MODE_MAP.get(
                    str(face.get("failure_mode", "mode_I")), "mode_I"
                ),
                "mode_mixity": 0.0,
                "normal_traction_pa": capacity / area if area > 0 else 0.0,
                "shear_traction_pa": 0.0,
                "geometry_synthetic": True,
                "energy_synthetic": fracture_energy_density_j_m2 is None,
            }
        )
        area_total += polygon_area(verts)
        energy_total += energy

    packet = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "version": VERSION,
        "source_solver": source_solver,
        "source_model_id": f"legacy-v{numeric}:{legacy.get('packet_id', 'unknown')}",
        "timestamp": timestamp,
        "units": {"length": "m", "area": "m2", "energy": "J", "traction": "Pa"},
        "face_count": len(faces),
        "area_total_m2": area_total,
        "energy_total_j": energy_total,
        "faces": faces,
    }
    return seal(packet)