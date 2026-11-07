"""Thin orchestration driver for the IN12-P6 campaign-contract node.

This script contains *no numerics*: every quantity comes from
``tensorphysics.campaign`` (clock lattice, dual ledgers, geometry contract).  Its
only job is to wire the contracts together, write the artifacts and record the
audit chain.  Its size relative to the library it drives is the
"thin-orchestration" proof for the campaign node.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorphysics.campaign import (  # noqa: E402
    CouplingClockSpec, FailureFacePacket, GeometryContractRegistry, PrincipalDimensions,
    ReferenceFluxes, build_reference_dual_ledgers, canonical_json, clock_drift,
    ledger_totals, payload_digest, reconcile_books, tetrahedron_face_packet,
    validate_artifact, verify_alignment, verify_same_batch,
)
from tensorphysics.campaign.clock import CouplingClock  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "in12_p6"

PC6_VERTICES = ((30.0, 0.0, -4.0), (38.0, -3.0, -4.0), (38.0, 3.0, -4.0), (34.0, 0.0, -1.5))
PC6_DIMENSIONS = PrincipalDimensions(120.0, 20.0, 7.5, 0.72, 12960.0)
DECIMAL_FLUXES = ReferenceFluxes(12.3, 4200.0, (1.2e5, 0.0, -3.4e4))
BINARY_FLUXES = ReferenceFluxes(2.0, 4096.0, (8.0, 0.0, -4.0))


WRITTEN: list[str] = []


def write(filename: str, schema: str, payload: dict) -> str:
    """Validate against the named schema, write and record the artifact name."""
    digest = validate_artifact(schema, payload)["digest"]
    (OUT / filename).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    WRITTEN.append(filename)
    return digest


def geometry_artifacts() -> dict:
    packet = tetrahedron_face_packet(
        PC6_VERTICES,
        capacities_n=(1.2e7, 1.8e7, 1.8e7, 4.0e7),
        contact_regions=("bow_shoulder", "bow_plate", "bow_plate", "lower_icebelt"),
        failure_modes=("crushing", "shear", "shear", "buckling"),
        schema_version=2,
    )
    write("failure_faces_pc6.json", "failure_faces.v2", packet.to_dict())
    contract = {
        "schema_name": "geometry_contract", "schema_version": 1,
        "geometry_id": "geom_pc6", "vessel_class": "PC6",
        "principal_dimensions": PC6_DIMENSIONS.to_dict(),
        "frame": {"origin": "midship_baseline", "x_forward": True},
        "design_water_density_kg_m3": 1025.0,
        "failure_faces_packet": "failure_faces_pc6.json",
        "consumer_contract": {"interface": "failure_faces",
                              "min_schema_version": 1, "max_schema_version": 2},
    }
    write("geom_pc6.json", "geometry_contract.v1", contract)
    handle = GeometryContractRegistry().register_file(OUT / "geom_pc6.json")
    return {"handle": handle, "packet": packet}


def clock_and_ledgers(scenario: str, spec: CouplingClockSpec, ticks: int, fluxes: ReferenceFluxes) -> dict:
    clock = CouplingClock(spec)
    events = clock.run(ticks)
    books = build_reference_dual_ledgers(spec, ticks, fluxes)
    for quantity, (producer, consumer) in books.items():
        write(f"ledger_{scenario}_{quantity.value}_producer.json", "side_ledger.v1", producer.to_dict())
        write(f"ledger_{scenario}_{quantity.value}_consumer.json", "side_ledger.v1", consumer.to_dict())
    artifact = reconcile_books(books, scenario=scenario)
    filename = "ledger_reconciliation.json" if scenario == "decimal_clock" else "ledger_reconciliation_binary.json"
    write(filename, "ledger_reconciliation.v1", artifact)
    again = reconcile_books(build_reference_dual_ledgers(spec, ticks, fluxes), scenario=scenario)
    record = {
        "scenario": scenario, "spec": spec.to_dict(master_tick=ticks),
        "events": [event.to_dict() for event in events],
        "alignment": verify_alignment(spec, events),
        "drift": clock_drift(spec, ticks),
        "totals": ledger_totals(books),
        "reconciliation": artifact,
        "same_batch_diff": 0 if canonical_json(artifact) == canonical_json(again) else 1,
        "same_batch_verdict": verify_same_batch(artifact),
    }
    write(f"clock_{scenario}.json", "coupling_clock.v1", spec.to_dict(master_tick=ticks))
    (OUT / f"run_{scenario}.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    geometry = geometry_artifacts()
    decimal = clock_and_ledgers("decimal_clock", CouplingClockSpec("rigid_body", "lbm_flow", 0.004, 0.0004, 5), 50, DECIMAL_FLUXES)
    binary = clock_and_ledgers("binary_clock", CouplingClockSpec("rigid_body", "lbm_flow", 1.0 / 256.0, 1.0 / 4096.0, 4), 16, BINARY_FLUXES)
    summary = {
        "geometry_id": geometry["handle"].geometry_id,
        "geometry_checksum": geometry["handle"].checksum_sha256,
        "geometry_closure": geometry["packet"].verify_closure(),
        "cb_closure_residual": PC6_DIMENSIONS.cb_closure_residual,
        "scenarios": {name: {"status": rec["reconciliation"]["status"],
                             "max_relative_residual": rec["reconciliation"]["max_relative_residual"],
                             "same_batch_diff": rec["same_batch_diff"]}
                      for name, rec in (("decimal_clock", decimal), ("binary_clock", binary))},
        "run_digests": {
            name: payload_digest(json.loads((OUT / name).read_text()))
            for name in sorted(WRITTEN)
        },
    }
    (OUT / "campaign_run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())