"""Contract tests for the frozen ``failure_faces`` v3 schema and its upgrades."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from tensorphysics.failure_handoff import (
    FAILURE_FACES_SCHEMA,
    SCHEMA_FINGERPRINT,
    SCHEMA_VERSION,
    SUPPORTED_VERSIONS,
    VERSION,
    FailureFaceSchemaError,
    packet_digest,
    seal,
    upgrade_legacy_packet,
    validate_packet,
    verify_seal,
)

PINNED_SCHEMA_FINGERPRINT = "807d9eef29db7f7974ab8c1acc8546ea7659b9f69c7a64cba83f5bca8bca0db3"


def _face(face_id: str, offset: float) -> dict:
    """A planar 1 cm x 1 cm quad in the x = offset plane, normal (+1, 0, 0)."""
    verts = [
        [offset, 0.0, 0.0],
        [offset, 0.01, 0.0],
        [offset, 0.01, 0.01],
        [offset, 0.0, 0.01],
    ]
    area = 1.0e-4
    return {
        "face_id": face_id,
        "element_id": 0,
        "vertices_m": verts,
        "normal": [1.0, 0.0, 0.0],
        "centroid_m": [offset, 0.005, 0.005],
        "area_m2": area,
        "specific_energy_j_m2": 150.0,
        "energy_j": 150.0 * area,
        "failure_mode": "mode_I",
        "mode_mixity": 0.0,
        "normal_traction_pa": 4.0e6,
        "shear_traction_pa": 0.0,
    }


@pytest.fixture()
def packet() -> dict:
    faces = [_face("F0001", 0.0), _face("F0002", 0.01)]
    body = {
        "schema_name": "failure_faces",
        "schema_version": SCHEMA_VERSION,
        "version": VERSION,
        "source_solver": "pytest/synthetic",
        "source_model_id": "unit-test-model",
        "timestamp": "2026-11-10T00:00:00Z",
        "units": {"length": "m", "area": "m2", "energy": "J", "traction": "Pa"},
        "fracture_layer_thickness_m": 0.01,
        "face_count": len(faces),
        "area_total_m2": sum(f["area_m2"] for f in faces),
        "energy_total_j": sum(f["energy_j"] for f in faces),
        "crack_front_m": [[0.0, 0.01, 0.0], [0.0, 0.01, 0.01]],
        "faces": faces,
    }
    return seal(body)


def test_schema_constants_are_frozen() -> None:
    assert SCHEMA_VERSION == 3
    assert VERSION == "3.0.0"
    assert SUPPORTED_VERSIONS == ("3.0.0",)
    assert FAILURE_FACES_SCHEMA["properties"]["version"]["const"] == VERSION


def test_schema_fingerprint_is_pinned() -> None:
    """Editing the schema without a version bump breaks this pin (bilateral lock)."""
    assert SCHEMA_FINGERPRINT == PINNED_SCHEMA_FINGERPRINT


def test_valid_packet_passes(packet: dict) -> None:
    assert validate_packet(packet)["sha256"] == packet["sha256"]
    assert verify_seal(packet)


@pytest.mark.parametrize(
    "field",
    ["schema_name", "version", "source_solver", "timestamp", "units", "face_count",
     "area_total_m2", "energy_total_j", "faces", "sha256"],
)
def test_missing_required_field_rejected(packet: dict, field: str) -> None:
    broken = copy.deepcopy(packet)
    broken.pop(field)
    with pytest.raises(FailureFaceSchemaError):
        validate_packet(broken)


@pytest.mark.parametrize(
    "field", ["energy_j", "vertices_m", "normal", "area_m2", "failure_mode"]
)
def test_missing_required_face_field_rejected(packet: dict, field: str) -> None:
    broken = copy.deepcopy(packet)
    broken["faces"][0].pop(field)
    broken = seal(broken)
    with pytest.raises(FailureFaceSchemaError):
        validate_packet(broken)


def test_unknown_field_rejected(packet: dict) -> None:
    broken = copy.deepcopy(packet)
    broken["hidden_extra"] = 1
    with pytest.raises(FailureFaceSchemaError):
        validate_packet(seal(broken))


def test_tampered_area_rejected(packet: dict) -> None:
    """Declared area must match the polygon implied by the vertices."""
    broken = copy.deepcopy(packet)
    broken["faces"][0]["area_m2"] *= 1.02
    broken["area_total_m2"] = sum(f["area_m2"] for f in broken["faces"])
    with pytest.raises(FailureFaceSchemaError, match="polygon area"):
        validate_packet(seal(broken))


def test_stale_seal_rejected(packet: dict) -> None:
    broken = copy.deepcopy(packet)
    broken["faces"][1]["energy_j"] *= 1.5
    broken["energy_total_j"] = sum(f["energy_j"] for f in broken["faces"])
    with pytest.raises(FailureFaceSchemaError, match="seal digest"):
        validate_packet(broken)
    # re-sealing makes it consistent again, which is the point of the digest
    validate_packet(seal(broken))


def test_non_unit_normal_rejected(packet: dict) -> None:
    broken = copy.deepcopy(packet)
    broken["faces"][0]["normal"] = [2.0, 0.0, 0.0]
    with pytest.raises(FailureFaceSchemaError, match="unit length"):
        validate_packet(seal(broken))


def test_reversed_winding_rejected(packet: dict) -> None:
    broken = copy.deepcopy(packet)
    broken["faces"][0]["vertices_m"] = list(reversed(broken["faces"][0]["vertices_m"]))
    with pytest.raises(FailureFaceSchemaError, match="winding"):
        validate_packet(seal(broken))


def test_future_version_rejected(packet: dict) -> None:
    broken = copy.deepcopy(packet)
    broken["version"] = "4.0.0"
    with pytest.raises(FailureFaceSchemaError, match="unsupported packet version"):
        validate_packet(seal(broken))


def test_seal_is_deterministic_and_content_sensitive(packet: dict) -> None:
    assert seal(packet)["sha256"] == packet_digest(packet)
    reordered = {k: packet[k] for k in reversed(list(packet))}
    assert seal(reordered)["sha256"] == packet["sha256"]
    changed = copy.deepcopy(packet)
    changed["source_model_id"] = "another-model"
    assert seal(changed)["sha256"] != packet["sha256"]
    # the digest round-trips through JSON text without drifting
    assert packet_digest(json.loads(json.dumps(packet))) == packet["sha256"]


def test_jsonschema_reference_layer_agrees(packet: dict) -> None:
    """When ``jsonschema`` is installed the same document must accept the packet."""
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.Draft202012Validator(FAILURE_FACES_SCHEMA).validate(packet)


def _legacy_packet(schema_version: int) -> dict:
    face = {
        "face_id": "cf_1",
        "normal": [1.0, 0.0, 0.0],
        "area_m2": 2.5e-3,
        "capacity_n": 1.2e5,
    }
    if schema_version == 2:
        face["centroid_m"] = [0.1, 0.2, 0.3]
        face["failure_mode"] = "bending"
    return {
        "schema_name": "failure_faces",
        "schema_version": schema_version,
        "interface": "failure_faces",
        "faces": [face],
    }


@pytest.mark.parametrize("schema_version", [1, 2])
def test_legacy_packet_upgrades_to_v3(schema_version: int) -> None:
    legacy = _legacy_packet(schema_version)
    upgraded = upgrade_legacy_packet(
        legacy,
        timestamp="2026-11-10T00:00:00Z",
        source_solver="legacy-import",
        fracture_energy_density_j_m2=120.0,
    )
    validate_packet(upgraded)
    assert upgraded["version"] == VERSION
    assert upgraded["face_count"] == 1
    face = upgraded["faces"][0]
    assert face["geometry_synthetic"] is True
    assert face["energy_synthetic"] is False
    assert abs(face["area_m2"] - legacy["faces"][0]["area_m2"]) <= 1e-12
    assert face["specific_energy_j_m2"] == pytest.approx(120.0)
    assert face["energy_j"] == pytest.approx(120.0 * legacy["faces"][0]["area_m2"])
    # the legacy capacity becomes a normal traction (force / area)
    assert face["normal_traction_pa"] == pytest.approx(1.2e5 / 2.5e-3)


def test_legacy_upgrade_without_energy_density_is_flagged() -> None:
    upgraded = upgrade_legacy_packet(
        _legacy_packet(2), timestamp="2026-11-10T00:00:00Z", source_solver="legacy-import"
    )
    validate_packet(upgraded)
    assert upgraded["faces"][0]["energy_synthetic"] is True
    assert upgraded["energy_total_j"] == 0.0


def test_upgrade_rejects_non_legacy_version() -> None:
    with pytest.raises(FailureFaceSchemaError):
        upgrade_legacy_packet(
            {"schema_name": "failure_faces", "schema_version": 7, "faces": []},
            timestamp="2026-11-10T00:00:00Z",
            source_solver="legacy-import",
        )


def test_packet_faces_are_planar_and_centroid_consistent(packet: dict) -> None:
    for face in packet["faces"]:
        verts = np.asarray(face["vertices_m"])
        normal = np.asarray(face["normal"])
        assert abs(float(np.linalg.norm(normal)) - 1.0) < 1e-12
        assert np.allclose((verts - verts.mean(axis=0)) @ normal, 0.0, atol=1e-12)
        assert np.allclose(np.asarray(face["centroid_m"]), verts.mean(axis=0), atol=1e-12)