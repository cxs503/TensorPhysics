"""Closed artifact schemas used by the P84-W2C campaign gate.

This module is intentionally *data only* (no imports beyond ``typing``) so that
``campaign.schema_defs`` can import the dictionaries at module import time
without creating an import cycle through the gate package ``__init__``.

Three artefacts cross a repository boundary in this campaign:

``campaign_gate_report.v1``
    The gate verdict: per-node evidence, per-criterion numbers, interface
    contract results, lineage adjudication and the gap analysis.
``aero_polar.v1``
    Producer side of the ``aero-wind`` interface: a subsonic airfoil polar
    (alpha grid, C_L, C_D) plus the air density and chord used to normalise it.
``amp_emissions.v1``
    Producer side of the ``amp-eexi`` interface: shore-power (AMP) auxiliary
    load, fuel/CO2 ledger and the innovative-energy-technology power credit
    that the EEXI consumer accepts as ``p_eff_kw``.
"""
from __future__ import annotations

from typing import Any, Mapping

# --- shared fragments -------------------------------------------------------

_NUMBER = {"type": "number"}
_STRING = {"type": "string", "minLength": 1}
_STATUS = {"type": "string", "enum": ["PASS", "FAIL", "WITHHELD", "NOT_SUPPORTED"]}

_CRITERION_RESULT = {
    "type": "object",
    "required": ["id", "statement", "status"],
    "properties": {
        "id": _STRING,
        "statement": {"type": "string"},
        "status": _STATUS,
        "value": {"type": ["number", "null"]},
        "threshold": {"type": ["number", "array", "string", "null"]},
        "comparator": {"type": "string"},
        "unit": {"type": "string"},
        "source": {"type": "string"},
        "detail": {"type": "string"},
    },
    "additionalProperties": False,
}

_ARTIFACT_RECORD = {
    "type": "object",
    "required": ["path", "kind", "exists", "parseable"],
    "properties": {
        "path": {"type": "string"},
        "kind": {"type": "string", "enum": ["json", "png"]},
        "exists": {"type": "boolean"},
        "parseable": {"type": "boolean"},
        "bytes": {"type": "integer", "minimum": 0},
        "sha256": {"type": "string"},
        "detail": {"type": "string"},
    },
    "additionalProperties": False,
}

_PYTEST_RECORD = {
    "type": "object",
    "required": ["declared", "present", "parseable", "mode", "status"],
    "properties": {
        "declared": {"type": "array", "items": _STRING},
        "present": {"type": "boolean"},
        "parseable": {"type": "boolean"},
        "mode": {"type": "string", "enum": ["executed", "collect-only", "record", "missing"]},
        "status": {"type": "string", "enum": ["PASS", "FAIL", "WITHHELD", "NOT_SUPPORTED"]},
        "collected": {"type": "integer", "minimum": 0},
        "passed": {"type": "integer", "minimum": 0},
        "failed": {"type": "integer", "minimum": 0},
        "errors": {"type": "integer", "minimum": 0},
        "skipped": {"type": "integer", "minimum": 0},
        "exit_code": {"type": "integer"},
        "duration_s": {"type": "number"},
        "record_path": {"type": "string"},
        "detail": {"type": "string"},
        "reused": {"type": "boolean"},
        "failures": {"type": "array", "items": {"type": "string"}},
    },
    "additionalProperties": False,
}

_NODE_RECORD = {
    "type": "object",
    "required": [
        "node_key", "mission_id", "repo", "title", "physics_domain",
        "root_present", "verdict", "criteria", "artifacts", "pytest",
        "key_numbers", "gaps", "lineage",
    ],
    "properties": {
        "node_key": _STRING,
        "mission_id": {"type": "integer", "minimum": 1},
        "repo": _STRING,
        "title": _STRING,
        "physics_domain": _STRING,
        "root_present": {"type": "boolean"},
        "platform_status": {"type": "string"},
        "verdict": _STATUS,
        "criteria": {"type": "array", "items": _CRITERION_RESULT},
        "artifacts": {"type": "array", "items": _ARTIFACT_RECORD},
        "pytest": _PYTEST_RECORD,
        "key_numbers": {"type": "object", "additionalProperties": {"type": ["number", "string", "null"]}},
        "gaps": {"type": "array", "items": {"type": "string"}},
        "lineage": {
            "type": "object",
            "required": ["parent_node", "parent_mission_id", "parent_status", "verdict"],
            "properties": {
                "parent_node": {"type": ["string", "null"]},
                "parent_mission_id": {"type": ["integer", "null"]},
                "parent_status": {"type": ["string", "null"]},
                "verdict": {"type": "string", "enum": ["inherited_verified", "inherited_unverified", "seeded_new"]},
                "note": {"type": "string"},
            },
            "additionalProperties": False,
        },
    },
    "additionalProperties": False,
}

_CONTRACT_CHECK = {
    "type": "object",
    "required": ["name", "status"],
    "properties": {
        "name": _STRING,
        "status": {"type": "string", "enum": ["PASS", "FAIL", "WITHHELD"]},
        "detail": {"type": "string"},
        "value": {"type": ["number", "string", "null"]},
        "threshold": {"type": ["number", "string", "null"]},
    },
    "additionalProperties": False,
}

_CONTRACT_RECORD = {
    "type": "object",
    "required": ["contract_id", "producer_node", "consumer", "status", "checks", "fingerprint"],
    "properties": {
        "contract_id": _STRING,
        "producer_node": _STRING,
        "producer_schema": _STRING,
        "consumer": _STRING,
        "status": _STATUS,
        "required": {"type": "boolean"},
        "checks": {"type": "array", "items": _CONTRACT_CHECK},
        "data_conformance": {"type": "string", "enum": ["PASS", "FAIL", "WITHHELD", "NOT_SUPPORTED"]},
        "fingerprint": {"type": "string"},
        "gaps": {"type": "array", "items": {"type": "string"}},
    },
    "additionalProperties": False,
}

_CAMPAIGN_CRITERION = {
    "type": "object",
    "required": ["id", "statement", "status", "metrics"],
    "properties": {
        "id": _STRING,
        "statement": {"type": "string"},
        "status": _STATUS,
        "metrics": {"type": "object", "additionalProperties": {"type": ["number", "string", "boolean", "null"]}},
        "gaps": {"type": "array", "items": {"type": "string"}},
        "evidence": {"type": "string"},
    },
    "additionalProperties": False,
}

CAMPAIGN_GATE_REPORT_V1: Mapping[str, Any] = {
    "type": "object",
    "required": [
        "schema_name", "schema_version", "plan_id", "plan_title", "gate_id",
        "generated_utc", "nodes", "contracts", "criteria", "lineage_summary",
        "overall_status", "blocker_count", "gap_analysis", "artifacts_index",
        "markdown", "fingerprint",
    ],
    "properties": {
        "schema_name": {"const": "campaign_gate_report.v1"},
        "schema_version": {"const": "1.0.0"},
        "plan_id": {"type": "integer", "minimum": 1},
        "plan_title": _STRING,
        "gate_id": {"const": "P84W2C-N11"},
        "generated_utc": _STRING,
        "nodes": {"type": "array", "minItems": 1, "items": _NODE_RECORD},
        "contracts": {"type": "array", "minItems": 1, "items": _CONTRACT_RECORD},
        "criteria": {"type": "array", "minItems": 4, "items": _CAMPAIGN_CRITERION},
        "lineage_summary": {
            "type": "object",
            "required": ["parent_plan_id", "inherited_verified", "inherited_unverified", "seeded_new"],
            "properties": {
                "parent_plan_id": {"type": "integer", "minimum": 1},
                "inherited_verified": {"type": "integer", "minimum": 0},
                "inherited_unverified": {"type": "integer", "minimum": 0},
                "seeded_new": {"type": "integer", "minimum": 0},
                "note": {"type": "string"},
            },
            "additionalProperties": False,
        },
        "overall_status": {"type": "string", "enum": ["PASS", "FAIL"]},
        "blocker_count": {"type": "integer", "minimum": 0},
        "gap_analysis": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["node_key", "reason", "remediation"],
                "properties": {
                    "node_key": _STRING,
                    "reason": {"type": "string"},
                    "remediation": {"type": "string"},
                    "severity": {"type": "string", "enum": ["blocker", "warning"]},
                },
                "additionalProperties": False,
            },
        },
        "artifacts_index": {
            "type": "object",
            "required": ["report_json", "report_md", "evidence_dir", "criterion_hits"],
            "properties": {
                "report_json": _STRING,
                "report_md": _STRING,
                "figure_png": {"type": ["string", "null"]},
                "evidence_dir": _STRING,
                "criterion_hits": {"type": "integer", "minimum": 0},
                "criterion_total": {"type": "integer", "minimum": 0},
                "schema_fingerprint": {"type": "string"},
            },
            "additionalProperties": False,
        },
        "markdown": {
            "type": "object",
            "required": ["path", "chars", "min_chars", "sections"],
            "properties": {
                "path": _STRING,
                "chars": {"type": "integer", "minimum": 0},
                "min_chars": {"type": "integer", "minimum": 1},
                "sections": {"type": "array", "items": _STRING},
                "figure": {"type": ["string", "null"]},
            },
            "additionalProperties": False,
        },
        "audit": {
            "type": "object",
            "properties": {
                "package_files": {"type": "array", "items": {"type": "object", "additionalProperties": {"type": "string"}}},
                "platform": {"type": "object", "additionalProperties": True},
                "tool": {"type": "string"},
                "python": {"type": "string"},
                "figure": {"type": ["string", "null"]},
            },
            "additionalProperties": False,
        },
        "fingerprint": {"type": "string", "minLength": 64, "maxLength": 64},
    },
    "additionalProperties": False,
}

AERO_POLAR_V1: Mapping[str, Any] = {
    "type": "object",
    "required": ["schema_name", "alpha_deg", "C_L", "C_D", "rho_air_kg_m3", "chord_m"],
    "properties": {
        "schema_name": {"const": "aero_polar.v1"},
        "alpha_deg": {"type": "array", "minItems": 3, "items": _NUMBER},
        "C_L": {"type": "array", "minItems": 3, "items": _NUMBER},
        "C_D": {"type": "array", "minItems": 3, "items": _NUMBER},
        "rho_air_kg_m3": {"type": "number", "exclusiveMinimum": 0},
        "chord_m": {"type": "number", "exclusiveMinimum": 0},
        "reynolds": {"type": "number", "exclusiveMinimum": 0},
        "regime": {"type": "string", "enum": ["laminar", "transitional", "turbulent"]},
        "notes": {"type": "string"},
    },
    "additionalProperties": False,
}

SEDIMENT_SUMMARY_V1: Mapping[str, Any] = {
    #: advisory interface (N04 -> ship-platform UKC endpoint): the node's own C6
    #: JSON contract, declared ahead of the consumer implementation.
    "type": "object",
    "required": ["shields_curve", "rouse_profiles", "bed_elevation",
                 "scour_depth", "mass_budget"],
    "properties": {
        "shields_curve": {"type": ["array", "object"]},
        "rouse_profiles": {"type": ["array", "object"]},
        "bed_elevation": {"type": ["array", "object"]},
        "scour_depth": {"type": ["number", "object"]},
        "mass_budget": {"type": ["number", "object"]},
        "notes": {"type": "string"},
    },
    "additionalProperties": False,
}

AMP_EMISSIONS_V1: Mapping[str, Any] = {
    "type": "object",
    "required": [
        "schema_name", "aux_power_kw", "fuel_t", "co2_t",
        "co2_reduction_t", "p_eff_kw", "grid_factor_kg_per_kwh",
    ],
    "properties": {
        "schema_name": {"const": "amp_emissions.v1"},
        "aux_power_kw": {"type": "number", "exclusiveMinimum": 0},
        "fuel_t": {"type": "number", "exclusiveMinimum": 0},
        "co2_t": {"type": "number", "exclusiveMinimum": 0},
        "co2_reduction_t": {"type": "number", "minimum": 0},
        "nox_reduction_t": {"type": "number", "minimum": 0},
        "sox_reduction_t": {"type": "number", "minimum": 0},
        "p_eff_kw": {"type": "number", "minimum": 0},
        "grid_factor_kg_per_kwh": {"type": "number", "minimum": 0},
        "notes": {"type": "string"},
    },
    "additionalProperties": False,
}