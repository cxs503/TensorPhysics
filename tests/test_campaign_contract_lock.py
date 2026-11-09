"""Bilateral schema lock: producer and consumer must accept the same contract.

Each registered artifact schema is hashed into a fingerprint.  The fingerprint is
pinned below, so *both* the writer path (object -> validated dict -> JSON) and the
reader path (JSON -> validated dict -> object) are locked to the same definition:
changing a schema without updating the pin fails this test on both sides at once.
The negative controls prove the schemas are still closed (no silent extra fields,
no NaN, no unversioned payloads).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tensorphysics.campaign import (
    CouplingClockSpec,
    FailureFacePacket,
    GeometryContractRegistry,
    SchemaError,
    SideLedger,
    canonical_json,
    payload_digest,
    reconcile_books,
    schema_fingerprints,
    schema_for,
    validate_artifact,
)
from tensorphysics.campaign.ledger import (
    QuantityKind,
    ReferenceFluxes,
    build_reference_dual_ledgers,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "in12_p6"

#: Bilateral lock: both producer and consumer tests compare against these.
PINNED_SCHEMA_FINGERPRINTS = {
    "coupling_clock.v1": "4d1e49f8e9cb4a451ac6c3472263f7048ad2653afae22c889a6a246c3bf811e8",
    "side_ledger.v1": "1287cf8db3937aa9970076858f6e65cc3bd2348bf07027ab49b57cda4e3ecb6a",
    "ledger_reconciliation.v1": "d5b797c64dd24e61419b3fbe28423f078098a6937a172054da0c16dfd5bd4272",
    "geometry_contract.v1": "19f83a3be34c551011e4ea786165349b432ed07d69aff8efa86f265eb56d45e9",
    "failure_faces.v1": "e149ca6cdbd9322a4e67e96f593342e9c2c6ba8bd6eee47af39cb7cbd3135d4f",
    "failure_faces.v2": "32fb185c751f34fccb6c532df676fc7b6232e659b6b6d6fb2dd238cfe9dbe316",
    # P84-W2C gate artifacts (campaign gate + bilateral interface schemas)
    "campaign_gate_report.v1": "78929781ef79554de7acd16d91db1d19cea777412aba9da5a65b068c41c538e3",
    "aero_polar.v1": "037e737613ad61e45a9e46224b7130a834c6608339d4b2dcd458e7d4c6d5605a",
    "amp_emissions.v1": "700f0dad3cbb6d48d1c0adf194a917315b9a28c53a579b58bf3a89aa68f74993",
    "sediment_summary.v1": "243193147acbb087109315fa0ad53d076d888aa7d6183c059b09f3a7ac9bfb52",
}

PRODUCER_ARTIFACTS = {
    "clock_decimal_clock.json": "coupling_clock.v1",
    "geom_pc6.json": "geometry_contract.v1",
    "failure_faces_pc6.json": "failure_faces.v2",
    "ledger_reconciliation.json": "ledger_reconciliation.v1",
}


def test_schema_fingerprints_match_the_bilateral_pin():
    assert schema_fingerprints() == PINNED_SCHEMA_FINGERPRINTS


@pytest.mark.parametrize("name", sorted(PRODUCER_ARTIFACTS))
def test_producer_artifacts_conform_and_match_shipped_digest(name):
    payload = json.loads((ARTIFACTS / name).read_text())
    schema = PRODUCER_ARTIFACTS[name]
    assert validate_artifact(schema, payload)["schema_fingerprint"] == PINNED_SCHEMA_FINGERPRINTS[schema]
    summary = json.loads((ARTIFACTS / "campaign_run_summary.json").read_text())
    assert summary["run_digests"][name] == payload_digest(payload)


@pytest.mark.parametrize("name", sorted(PRODUCER_ARTIFACTS))
def test_consumer_round_trip_preserves_canonical_bytes(name):
    payload = json.loads((ARTIFACTS / name).read_text())
    if name.startswith("clock"):
        restored = CouplingClockSpec.from_dict(payload).to_dict(master_tick=payload["ticks"]["master"])
    elif name.startswith("failure_faces"):
        restored = FailureFacePacket.from_dict(payload).to_dict()
    elif name.startswith("geom"):
        restored = GeometryContractRegistry().register(payload).contract
    else:
        restored = payload
    assert canonical_json(restored) == canonical_json(payload)


def test_consumer_reconstructs_the_dual_ledgers_from_disk():
    books = {}
    for quantity in QuantityKind:
        producer = SideLedger.from_dict(
            json.loads((ARTIFACTS / f"ledger_decimal_clock_{quantity.value}_producer.json").read_text())
        )
        consumer = SideLedger.from_dict(
            json.loads((ARTIFACTS / f"ledger_decimal_clock_{quantity.value}_consumer.json").read_text())
        )
        books[quantity] = (producer, consumer)
    recomputed = reconcile_books(books, scenario="decimal_clock")
    shipped = json.loads((ARTIFACTS / "ledger_reconciliation.json").read_text())
    assert canonical_json(recomputed) == canonical_json(shipped)


def _minimal_clock_payload() -> dict:
    return CouplingClockSpec("rigid_body", "lbm_flow", 0.004, 0.0004, 5).to_dict()


@pytest.mark.parametrize(
    "name,payload",
    [
        ("coupling_clock.v1", {**_minimal_clock_payload(), "unexpected": 1}),
        ("coupling_clock.v1", {k: v for k, v in _minimal_clock_payload().items() if k != "ratio"}),
        ("coupling_clock.v1", {**_minimal_clock_payload(), "ratio": True}),
        ("side_ledger.v1", {"schema_name": "side_ledger", "schema_version": 1, "quantity": "momentum",
                            "units": "kg*m/s", "side": "producer", "entries": [], "total": [0.0], "entry_count": 0}),
        ("ledger_reconciliation.v1", {"schema_name": "ledger_reconciliation", "schema_version": 1,
                                      "scenario": "x", "atol_ulps": float("nan"),
                                      "quantity_books": [], "status": "pass",
                                      "max_relative_residual": 0.0, "same_batch_digest": "0" * 64}),
        ("geometry_contract.v1", {"schema_name": "geometry_contract", "schema_version": 1,
                                   "geometry_id": "geom_pc6", "vessel_class": "PC6",
                                   "principal_dimensions": {}, "frame": {"origin": "x", "x_forward": True},
                                   "design_water_density_kg_m3": 1025.0,
                                   "failure_faces_packet": "p.json",
                                   "consumer_contract": {"interface": "failure_faces",
                                                         "min_schema_version": 1, "max_schema_version": 2}}),
        ("failure_faces.v2", {"schema_name": "failure_faces", "schema_version": 2,
                              "interface": "failure_faces",
                              "faces": [{"face_id": "f", "normal": [1.0, 0.0, 0.0],
                                         "area_m2": 1.0, "capacity_n": 1.0, "bogus_field": 3}]}),
        ("failure_faces.v1", {"schema_name": "failure_faces", "schema_version": 1,
                              "interface": "failure_faces",
                              "faces": [{"face_id": "f", "normal": [1.0, 0.0],
                                         "area_m2": 1.0, "capacity_n": 1.0}]}),
    ],
)
def test_schemas_reject_malformed_payloads(name, payload):
    with pytest.raises(SchemaError):
        validate_artifact(name, payload)


def test_unknown_schema_name_is_rejected():
    with pytest.raises(SchemaError, match="unknown artifact schema"):
        schema_for("ledger_reconciliation.v2")


def test_nan_is_not_a_valid_number():
    schema = schema_for("side_ledger.v1")
    payload = SideLedger(
        quantity=QuantityKind.MASS,
        side="producer",
        entries=(),
    ).to_dict()
    payload["entries"] = [{"label": "w", "step": 0, "components": [math.nan]}]
    with pytest.raises(SchemaError):
        validate_artifact("side_ledger.v1", payload)
    assert schema["type"] == "object"


def test_producer_payloads_are_schema_clean_by_construction():
    books = build_reference_dual_ledgers(
        CouplingClockSpec("rigid_body", "lbm_flow", 0.004, 0.0004, 5),
        50,
        ReferenceFluxes(12.3, 4200.0, (1.2e5, 0.0, -3.4e4)),
    )
    for quantity, (producer, consumer) in books.items():
        validate_artifact("side_ledger.v1", producer.to_dict())
        validate_artifact("side_ledger.v1", consumer.to_dict())
    artifact = reconcile_books(books, scenario="decimal_clock")
    assert validate_artifact("ledger_reconciliation.v1", artifact)["schema_fingerprint"] == \
        PINNED_SCHEMA_FINGERPRINTS["ledger_reconciliation.v1"]