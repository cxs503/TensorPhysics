"""Tests for the ``geom_pc6`` geometry contract and failure-face extension.

Acceptance criterion: the geometry contract closes the block coefficient against
the displacement volume and the failure-face packet closes as a surface, both at
machine precision, and the v2 interface extension stays backward compatible with
v1 consumers.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from tensorphysics.campaign import (
    CLOSURE_RTOL,
    FailureFace,
    FailureFacePacket,
    FailureFaceV1,
    GeometryContractError,
    GeometryContractRegistry,
    PrincipalDimensions,
    tetrahedron_face_packet,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "in12_p6"
VERTICES = ((30.0, 0.0, -4.0), (38.0, -3.0, -4.0), (38.0, 3.0, -4.0), (34.0, 0.0, -1.5))


def pc6_packet(schema_version: int = 2) -> FailureFacePacket:
    return tetrahedron_face_packet(
        VERTICES,
        capacities_n=(1.2e7, 1.8e7, 1.8e7, 4.0e7),
        contact_regions=("bow_shoulder", "bow_plate", "bow_plate", "lower_icebelt"),
        failure_modes=("crushing", "shear", "shear", "buckling"),
        schema_version=schema_version,
    )


def test_principal_dimensions_close_at_machine_precision():
    dimensions = PrincipalDimensions(120.0, 20.0, 7.5, 0.72, 12960.0)
    assert dimensions.cb_closure_residual <= CLOSURE_RTOL
    assert dimensions.cb_closure_residual > 0.0, "float rounding must leave a residual"
    assert dimensions.cb_from_volume == pytest.approx(0.72, rel=1e-15)
    assert dimensions.displacement_mass_t(1025.0) == pytest.approx(13284.0, rel=1e-15)


def test_block_coefficient_must_close_against_volume():
    with pytest.raises(GeometryContractError, match="Cb closure failed"):
        PrincipalDimensions(120.0, 20.0, 7.5, 0.75, 12960.0)
    with pytest.raises(GeometryContractError, match="Lpp > B > T"):
        PrincipalDimensions(7.5, 20.0, 120.0, 0.72, 12960.0)
    with pytest.raises(GeometryContractError):
        PrincipalDimensions(120.0, 20.0, 7.5, 1.2, 12960.0)


def test_tetrahedron_packet_closes_as_a_surface_at_machine_precision():
    packet = pc6_packet()
    closure = packet.verify_closure()
    assert closure["closed"] is True
    assert 0.0 < closure["closure_ratio"] <= CLOSURE_RTOL
    assert closure["closure_ratio"] < 1e-16, "closure must sit at the round-off floor"
    assert packet.schema_version == 2
    assert len(packet.faces) == 4
    assert packet.total_capacity_n() == pytest.approx(8.8e7, rel=1e-15)


def test_open_surface_is_detected():
    packet = pc6_packet(2)
    open_packet = FailureFacePacket(faces=packet.faces[:3], schema_version=2)
    closure = open_packet.verify_closure()
    assert closure["closed"] is False
    assert closure["closure_ratio"] > 0.3, "dropping a face must open the surface"
    flipped = tuple(
        replace(face, normal=tuple(-value for value in face.normal))
        if index == 0
        else face
        for index, face in enumerate(packet.faces)
    )
    assert FailureFacePacket(faces=flipped, schema_version=2).closure_ratio() > 0.3
    contract = json.loads((ARTIFACTS / "geom_pc6.json").read_text())
    with pytest.raises(GeometryContractError, match="not a closed surface"):
        GeometryContractRegistry().register(contract, faces=open_packet)


def test_v2_extension_keeps_v1_required_fields_and_downgrades():
    v2 = pc6_packet(2)
    v1 = v2.downgrade_v1()
    assert v1.schema_version == 1
    assert all(isinstance(face, FailureFaceV1) and not isinstance(face, FailureFace)
               for face in v1.faces)
    v1_again = FailureFacePacket.from_dict(v1.to_dict())
    assert v1_again.schema_version == 1
    assert [face.face_id for face in v1_again.faces] == [face.face_id for face in v1.faces]
    assert v2.faces[0].failure_mode == "crushing"
    assert v2.faces[0].contact_region == "bow_shoulder"
    with pytest.raises(GeometryContractError, match="v1 packet cannot carry v2"):
        FailureFacePacket(faces=v2.faces, schema_version=1)


def test_unsupported_face_version_is_rejected_on_both_sides():
    payload = pc6_packet(2).to_dict()
    payload["schema_version"] = 3
    with pytest.raises(GeometryContractError, match="supported window"):
        FailureFacePacket.from_dict(payload)
    with pytest.raises(GeometryContractError):
        FailureFacePacket(faces=pc6_packet(2).faces, schema_version=3)


def test_face_normal_must_be_unit_and_ids_unique():
    with pytest.raises(GeometryContractError, match="unit vector"):
        FailureFaceV1("bad", (1.0, 1.0, 0.0), 1.0, 1.0)
    face = pc6_packet(1).faces[0]
    with pytest.raises(GeometryContractError, match="unique"):
        FailureFacePacket(faces=(face, face), schema_version=1)


def test_consumer_window_rejects_newer_packet():
    payload = json.loads((ARTIFACTS / "geom_pc6.json").read_text())
    payload["consumer_contract"]["max_schema_version"] = 1
    with pytest.raises(GeometryContractError, match="outside the declared consumer window"):
        GeometryContractRegistry().register(payload, faces=pc6_packet(2))


def test_committed_geometry_bundle_registers_and_resolves():
    registry = GeometryContractRegistry()
    handle = registry.register_file(ARTIFACTS / "geom_pc6.json")
    assert handle.geometry_id == "geom_pc6"
    assert registry.resolve("geom_pc6") is handle
    assert len(handle.checksum_sha256) == 64
    assert handle.dimensions.cb_closure_residual <= CLOSURE_RTOL
    assert handle.faces is not None and handle.faces.schema_version == 2
    assert handle.faces.verify_closure()["closed"] is True
    assert handle.displacement_mass_t == pytest.approx(13284.0, rel=1e-15)
    with pytest.raises(GeometryContractError, match="already registered"):
        registry.register_file(ARTIFACTS / "geom_pc6.json")
    with pytest.raises(GeometryContractError, match="not registered"):
        registry.resolve("geom_pc7")


def test_geometry_id_must_match_filename(tmp_path):
    contract = json.loads((ARTIFACTS / "geom_pc6.json").read_text())
    contract["failure_faces_packet"] = str(ARTIFACTS / "failure_faces_pc6.json")
    path = tmp_path / "geom_pc7.json"
    path.write_text(json.dumps(contract))
    with pytest.raises(GeometryContractError, match="does not match file"):
        GeometryContractRegistry().register_file(path)


def test_missing_declared_packet_is_rejected(tmp_path):
    contract = json.loads((ARTIFACTS / "geom_pc6.json").read_text())
    contract["failure_faces_packet"] = "missing_packet.json"
    path = tmp_path / "geom_pc6.json"
    path.write_text(json.dumps(contract))
    with pytest.raises(GeometryContractError, match="packet not found"):
        GeometryContractRegistry().register_file(path)


def test_geometry_contract_schema_is_closed():
    contract = json.loads((ARTIFACTS / "geom_pc6.json").read_text())
    contract["silent_extra_field"] = 1
    with pytest.raises(GeometryContractError, match="additional properties"):
        GeometryContractRegistry().register(contract)