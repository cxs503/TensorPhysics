"""Registered schema for the ``campaign_readiness.v1`` gate hand-off.

The four cross-library *orchestration* artifacts (clock, ledgers, geometry,
failure faces) live in :mod:`tensorphysics.campaign.schema_defs` and are frozen
by a bilateral fingerprint lock shared with the producer/consumer repos.  A gate
verdict is a different kind of artifact: it is consumed by the dispatcher, not by
another solver, so it carries its own registry here and does not perturb that
lock.  The schema language is the same closed JSON-Schema subset, validated by
:func:`tensorphysics.campaign.schema.validate`.
"""
from __future__ import annotations

from typing import Any, Mapping

from tensorphysics.campaign.schema import SchemaError, payload_digest, validate

READINESS_SCHEMA_NAME = "campaign_readiness.v1"

_CRITERION = {
    "type": "object",
    "required": ["criterion_id", "label", "status", "detail", "metrics", "evidence"],
    "properties": {
        "criterion_id": {"enum": ["a", "b", "c", "d", "e", "f", "g"]},
        "label": {"type": "string", "minLength": 1},
        "status": {"enum": ["PASS", "FAIL", "WITHHELD"]},
        "detail": {"type": "string", "minLength": 1},
        "metrics": {"type": "object"},
        "evidence": {"type": "object"},
    },
    "additionalProperties": False,
}

_NODE_READINESS = {
    "type": "object",
    "required": ["ext_id", "mission_id", "repo", "physics_domain", "deliverable",
                 "registered", "maturity", "chain_ready", "blocking"],
    "properties": {
        "ext_id": {"type": "string", "pattern": "^P35-N[0-9]+$"},
        "mission_id": {"type": "integer", "minimum": 1},
        "repo": {"type": "string", "minLength": 1},
        "physics_domain": {"type": "string", "minLength": 1},
        "deliverable": {"enum": ["PASS", "FAIL", "WITHHELD"]},
        "registered": {"enum": ["PASS", "FAIL", "WITHHELD"]},
        "maturity": {"enum": ["PASS", "FAIL", "WITHHELD"]},
        "chain_ready": {"type": "boolean"},
        "blocking": {"type": "array", "items": {"type": "string", "minLength": 1}},
    },
    "additionalProperties": False,
}

_REGRESSION_RECORD = {
    "type": "object",
    "required": ["case", "status", "baseline", "measured", "rel_dev", "threshold",
                 "floor", "detail", "elapsed_s"],
    "properties": {
        "case": {"type": "string", "minLength": 1},
        "status": {"enum": ["PASS", "FAIL", "WITHHELD"]},
        "baseline": {"type": ["number", "null"]},
        "measured": {"type": ["number", "null"]},
        "rel_dev": {"type": ["number", "null"], "minimum": 0},
        "threshold": {"type": "number", "exclusiveMinimum": 0},
        "floor": {"type": "number", "minimum": 0},
        "detail": {"type": "string", "minLength": 1},
        "elapsed_s": {"type": ["number", "null"], "minimum": 0},
    },
    "additionalProperties": False,
}

_PROVENANCE_INPUT = {
    "type": "object",
    "required": ["path", "present", "sha256"],
    "properties": {
        "path": {"type": "string", "minLength": 1},
        "present": {"type": "boolean"},
        "sha256": {"type": ["string", "null"]},
    },
    "additionalProperties": False,
}

_PROVENANCE = {
    "type": "object",
    "required": ["inputs", "generator", "schema_fingerprint"],
    "properties": {
        "inputs": {"type": "array", "minItems": 1, "items": _PROVENANCE_INPUT},
        "generator": {"type": "string", "minLength": 1},
        "schema_fingerprint": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
    },
    "additionalProperties": False,
}

#: Machine-readable hand-off a campaign gate leaves for the dispatcher
#: (``dispatcher_ready`` is true only when every criterion is PASS).
CAMPAIGN_READINESS_V1: Mapping[str, Any] = {
    "type": "object",
    "required": [
        "schema_name", "schema_version", "campaign_id", "gate_id", "plan_id",
        "generated_utc", "overall_status", "dispatcher_ready", "criteria", "nodes",
        "blocking", "regression", "provenance", "limitations",
    ],
    "properties": {
        "schema_name": {"const": "campaign_readiness"},
        "schema_version": {"const": 1},
        "campaign_id": {"type": "string", "minLength": 1},
        "gate_id": {"type": "string", "minLength": 1},
        "plan_id": {"type": "integer", "minimum": 1},
        "generated_utc": {"type": "string", "minLength": 10},
        "overall_status": {"enum": ["PASS", "FAIL"]},
        "dispatcher_ready": {"type": "boolean"},
        "criteria": {"type": "array", "minItems": 7, "maxItems": 7, "items": _CRITERION},
        "nodes": {"type": "array", "minItems": 1, "maxItems": 64,
                  "items": _NODE_READINESS},
        "blocking": {"type": "array", "items": {"type": "string", "minLength": 1}},
        "regression": {"type": "array", "items": _REGRESSION_RECORD},
        "provenance": _PROVENANCE,
        "limitations": {"type": "array", "minItems": 1,
                        "items": {"type": "string", "minLength": 1}},
    },
    "additionalProperties": False,
}

SCHEMAS: dict[str, Mapping[str, Any]] = {READINESS_SCHEMA_NAME: CAMPAIGN_READINESS_V1}


def schema_for(name: str = READINESS_SCHEMA_NAME) -> Mapping[str, Any]:
    try:
        return SCHEMAS[name]
    except KeyError as exc:  # pragma: no cover - guard
        raise SchemaError(f"unknown readiness schema {name!r}") from exc


def schema_fingerprint(name: str = READINESS_SCHEMA_NAME) -> str:
    """SHA-256 of the canonical schema text (the pinned lock value)."""
    return payload_digest(schema_for(name))


def validate_artifact(name: str, payload: Any) -> dict:
    """Validate ``payload`` and return its canonical digest."""
    validate(schema_for(name), payload)
    return {"artifact": name, "digest": payload_digest(payload),
            "schema_fingerprint": schema_fingerprint(name)}
