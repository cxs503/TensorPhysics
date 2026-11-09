"""Registered artifact schemas for the campaign orchestration contracts.

Four artifacts cross library boundaries in an ice-region campaign:

``coupling_clock.v1``
    Master/sub step ratio and exchange cadence (M12 disposition).
``side_ledger.v1``
    One side (producer or consumer) of a mass / energy / impulse book.
``ledger_reconciliation.v1``
    Dual-book reconciliation verdict with machine-precision tolerances.
``geometry_contract.v1``
    ``geom_pc6`` principal dimensions, Cb closure, failure-face linkage.
``failure_faces.v1`` / ``failure_faces.v2``
    Closed-surface failure-face packet; v2 is the backward-compatible extension.
"""
from __future__ import annotations

from typing import Any, Mapping

from .schema import SchemaError, canonical_json, payload_digest, validate
from .p84w2c_gate.schemas import (
    AERO_POLAR_V1,
    AMP_EMISSIONS_V1,
    CAMPAIGN_GATE_REPORT_V1,
    SEDIMENT_SUMMARY_V1,
)

_VECTOR3 = {
    "type": "array",
    "minItems": 3,
    "maxItems": 3,
    "items": {"type": "number"},
}

#: Ledger totals are scalar for mass/energy and 3-vector for impulse.
_COMPONENTS = {
    "type": "array",
    "minItems": 1,
    "maxItems": 3,
    "items": {"type": "number"},
}

_SOLVER_CLOCK = {
    "type": "object",
    "required": ["name", "dt_s"],
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "dt_s": {"type": "number", "exclusiveMinimum": 0},
    },
    "additionalProperties": False,
}

COUPLING_CLOCK_V1: Mapping[str, Any] = {
    "type": "object",
    "required": [
        "schema_name",
        "schema_version",
        "disposition",
        "master",
        "sub",
        "time_base",
        "ratio",
        "exchange_every",
        "exchange_interval_s",
        "sub_steps_per_exchange",
        "max_information_age_s",
        "ticks",
    ],
    "properties": {
        "schema_name": {"const": "coupling_clock"},
        "schema_version": {"const": 1},
        "disposition": {"const": "M12"},
        "master": _SOLVER_CLOCK,
        "sub": _SOLVER_CLOCK,
        "time_base": {"const": "integer_sub_step"},
        "ratio": {"type": "integer", "minimum": 1},
        "exchange_every": {"type": "integer", "minimum": 1},
        "exchange_interval_s": {"type": "number", "exclusiveMinimum": 0},
        "sub_steps_per_exchange": {"type": "integer", "minimum": 1},
        "max_information_age_s": {"type": "number", "exclusiveMinimum": 0},
        "ticks": {
            "type": "object",
            "required": ["master", "sub_step"],
            "properties": {
                "master": {"type": "integer", "minimum": 0},
                "sub_step": {"type": "integer", "minimum": 0},
            },
            "additionalProperties": False,
        },
    },
    "additionalProperties": False,
}

_LEDGER_ENTRY = {
    "type": "object",
    "required": ["label", "step", "components"],
    "properties": {
        "label": {"type": "string", "minLength": 1},
        "step": {"type": "integer", "minimum": 0},
        "components": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {"type": "number"},
        },
    },
    "additionalProperties": False,
}

SIDE_LEDGER_V1: Mapping[str, Any] = {
    "type": "object",
    "required": [
        "schema_name",
        "schema_version",
        "quantity",
        "units",
        "side",
        "entries",
        "total",
        "entry_count",
    ],
    "properties": {
        "schema_name": {"const": "side_ledger"},
        "schema_version": {"const": 1},
        "quantity": {"enum": ["mass", "energy", "impulse"]},
        "units": {"type": "string", "minLength": 1},
        "side": {"enum": ["producer", "consumer"]},
        "entries": {"type": "array", "maxItems": 10000, "items": _LEDGER_ENTRY},
        "total": _COMPONENTS,
        "entry_count": {"type": "integer", "minimum": 0},
    },
    "additionalProperties": False,
}

_RESIDUAL = {
    "type": "object",
    "required": [
        "component",
        "producer",
        "consumer",
        "residual",
        "abs_residual",
        "scale",
        "tolerance",
        "relative",
        "grade",
    ],
    "properties": {
        "component": {"type": "integer", "minimum": 0},
        "producer": {"type": "number"},
        "consumer": {"type": "number"},
        "residual": {"type": "number"},
        "abs_residual": {"type": "number", "minimum": 0},
        "scale": {"type": "number", "minimum": 0},
        "tolerance": {"type": "number", "minimum": 0},
        "relative": {"type": "number", "minimum": 0},
        "grade": {
            "enum": ["exact", "machine_precision", "out_of_tolerance"]
        },
    },
    "additionalProperties": False,
}

QUANTITY_BOOK_V1: Mapping[str, Any] = {
    "type": "object",
    "required": [
        "quantity",
        "units",
        "component_count",
        "max_abs_residual",
        "max_relative_residual",
        "status",
        "residuals",
    ],
    "properties": {
        "quantity": {"enum": ["mass", "energy", "impulse"]},
        "units": {"type": "string", "minLength": 1},
        "component_count": {"type": "integer", "minimum": 1, "maximum": 3},
        "max_abs_residual": {"type": "number", "minimum": 0},
        "max_relative_residual": {"type": "number", "minimum": 0},
        "status": {"enum": ["pass", "fail"]},
        "residuals": {"type": "array", "minItems": 1, "items": _RESIDUAL},
    },
    "additionalProperties": False,
}

LEDGER_RECONCILIATION_V1: Mapping[str, Any] = {
    "type": "object",
    "required": [
        "schema_name",
        "schema_version",
        "scenario",
        "atol_ulps",
        "quantity_books",
        "status",
        "max_relative_residual",
        "same_batch_digest",
    ],
    "properties": {
        "schema_name": {"const": "ledger_reconciliation"},
        "schema_version": {"const": 1},
        "scenario": {"type": "string", "minLength": 1},
        "atol_ulps": {"type": "number", "exclusiveMinimum": 0},
        "quantity_books": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": QUANTITY_BOOK_V1,
        },
        "status": {"enum": ["pass", "fail"]},
        "max_relative_residual": {"type": "number", "minimum": 0},
        "same_batch_digest": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
    },
    "additionalProperties": False,
}

_DIMENSIONS = {
    "type": "object",
    "required": [
        "length_pp_m",
        "breadth_m",
        "draught_m",
        "block_coefficient",
        "displacement_volume_m3",
    ],
    "properties": {
        "length_pp_m": {"type": "number", "exclusiveMinimum": 0},
        "breadth_m": {"type": "number", "exclusiveMinimum": 0},
        "draught_m": {"type": "number", "exclusiveMinimum": 0},
        "block_coefficient": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
        "displacement_volume_m3": {"type": "number", "exclusiveMinimum": 0},
    },
    "additionalProperties": False,
}

GEOMETRY_CONTRACT_V1: Mapping[str, Any] = {
    "type": "object",
    "required": [
        "schema_name",
        "schema_version",
        "geometry_id",
        "vessel_class",
        "principal_dimensions",
        "frame",
        "design_water_density_kg_m3",
        "failure_faces_packet",
        "consumer_contract",
    ],
    "properties": {
        "schema_name": {"const": "geometry_contract"},
        "schema_version": {"const": 1},
        "geometry_id": {"type": "string", "pattern": "^geom_[a-z0-9]+$"},
        "vessel_class": {"type": "string", "minLength": 1},
        "principal_dimensions": _DIMENSIONS,
        "frame": {
            "type": "object",
            "required": ["origin", "x_forward"],
            "properties": {
                "origin": {"type": "string", "minLength": 1},
                "x_forward": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
        "design_water_density_kg_m3": {"type": "number", "exclusiveMinimum": 0},
        "failure_faces_packet": {"type": "string", "minLength": 1},
        "consumer_contract": {
            "type": "object",
            "required": ["interface", "min_schema_version", "max_schema_version"],
            "properties": {
                "interface": {"const": "failure_faces"},
                "min_schema_version": {"type": "integer", "minimum": 1},
                "max_schema_version": {"type": "integer", "minimum": 1},
            },
            "additionalProperties": False,
        },
    },
    "additionalProperties": False,
}

_FACE_V1 = {
    "type": "object",
    "required": ["face_id", "normal", "area_m2", "capacity_n"],
    "properties": {
        "face_id": {"type": "string", "minLength": 1},
        "normal": _VECTOR3,
        "area_m2": {"type": "number", "exclusiveMinimum": 0},
        "capacity_n": {"type": "number", "exclusiveMinimum": 0},
    },
    "additionalProperties": False,
}

# v2 = v1 + optional extension fields; closed packet schema stays strict.
_FACE_V2 = {
    "type": "object",
    "required": ["face_id", "normal", "area_m2", "capacity_n"],
    "properties": {
        "face_id": {"type": "string", "minLength": 1},
        "normal": _VECTOR3,
        "area_m2": {"type": "number", "exclusiveMinimum": 0},
        "capacity_n": {"type": "number", "exclusiveMinimum": 0},
        "centroid_m": _VECTOR3,
        "failure_mode": {
            "enum": ["crushing", "bending", "shear", "buckling"]
        },
        "contact_region": {"type": "string", "minLength": 1},
        "load_share": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
        "redundancy_group": {"type": "string", "minLength": 1},
    },
    "additionalProperties": False,
}


def _face_packet(schema_version: int) -> Mapping[str, Any]:
    return {
        "type": "object",
        "required": ["schema_name", "schema_version", "interface", "faces"],
        "properties": {
            "schema_name": {"const": "failure_faces"},
            "schema_version": {"const": schema_version},
            "interface": {"const": "failure_faces"},
            "faces": {
                "type": "array",
                "minItems": 1,
                "maxItems": 512,
                "items": _FACE_V2 if schema_version == 2 else _FACE_V1,
            },
        },
        "additionalProperties": False,
    }


FAILURE_FACES_V1: Mapping[str, Any] = _face_packet(1)
FAILURE_FACES_V2: Mapping[str, Any] = _face_packet(2)

ARTIFACT_SCHEMAS: dict[str, Mapping[str, Any]] = {
    "coupling_clock.v1": COUPLING_CLOCK_V1,
    "side_ledger.v1": SIDE_LEDGER_V1,
    "ledger_reconciliation.v1": LEDGER_RECONCILIATION_V1,
    "geometry_contract.v1": GEOMETRY_CONTRACT_V1,
    "failure_faces.v1": FAILURE_FACES_V1,
    "failure_faces.v2": FAILURE_FACES_V2,
    "campaign_gate_report.v1": CAMPAIGN_GATE_REPORT_V1,
    "aero_polar.v1": AERO_POLAR_V1,
    "amp_emissions.v1": AMP_EMISSIONS_V1,
    "sediment_summary.v1": SEDIMENT_SUMMARY_V1,
}


def schema_for(name: str) -> Mapping[str, Any]:
    """Return the registered schema, or raise ``SchemaError`` for unknown names."""
    try:
        return ARTIFACT_SCHEMAS[name]
    except KeyError as exc:
        raise SchemaError(
            f"unknown artifact schema {name!r}; registered: {sorted(ARTIFACT_SCHEMAS)}"
        ) from exc


def schema_fingerprint(name: str) -> str:
    """SHA-256 of the canonical schema text (the bilateral test lock value)."""
    return payload_digest(schema_for(name))


def schema_fingerprints() -> dict[str, str]:
    """Fingerprints of every registered schema, keyed by artifact name."""
    return {name: schema_fingerprint(name) for name in ARTIFACT_SCHEMAS}


def validate_quantity_book(payload: Any) -> None:
    """Validate a single reconciliation book (shared by producer and consumer)."""
    validate(QUANTITY_BOOK_V1, payload)


def validate_artifact(name: str, payload: Any) -> dict[str, Any]:
    """Validate ``payload`` and return its canonical digest (producer/consumer gate)."""
    validate(schema_for(name), payload)
    return {"artifact": name, "digest": payload_digest(payload),
            "schema_fingerprint": schema_fingerprint(name)}


def artifact_bytes(name: str, payload: Any) -> bytes:
    """Serialise a validated artifact as canonical UTF-8 JSON bytes."""
    validate_artifact(name, payload)
    return (canonical_json(payload) + "\n").encode("utf-8")