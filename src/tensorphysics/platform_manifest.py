"""Translate TensorPhysics run evidence into TensorPlatform's v1 contract.

This adapter is intentionally metadata-only: artifact bytes are uploaded through
an ArtifactStore first, and only verified references are put in the registry.
"""
from __future__ import annotations

from typing import Any, Mapping
from urllib.parse import urlsplit

from .provenance import ArtifactRef


def _artifact_record(ref: ArtifactRef, *, role: str) -> dict[str, Any]:
    if not ref.checksum_sha256 or ref.size_bytes is None:
        raise ValueError(f"artifact {ref.uri!r} must have checksum and size before registration")
    parsed = urlsplit(ref.uri)
    if not parsed.scheme or parsed.scheme == "file":
        raise ValueError(
            f"artifact {ref.uri!r} must use a shared remote URI, not a local file URI"
        )
    return {
        "uri": ref.uri,
        "media_type": ref.media_type,
        "checksum_sha256": ref.checksum_sha256,
        "size_bytes": ref.size_bytes,
        "role": role,
        **({"schema_name": ref.schema_name} if ref.schema_name else {}),
        **({"metadata": dict(ref.metadata)} if ref.metadata else {}),
    }


def build_platform_manifest(
    result: Mapping[str, Any],
    artifact_refs: list[tuple[str, ArtifactRef]],
    validation_report_ref: ArtifactRef,
    *,
    producer_revision: str = "unknown",
    producer_version: str = "0.1.0",
) -> dict[str, Any]:
    """Build a contract-v1 manifest using only verified shared artifact refs.

    artifact_refs contains (role, reference) pairs for solver-produced files.
    The validation report is always added as a first-class output artifact.
    """
    run = result.get("run_manifest")
    validation = result.get("validation_report")
    goal = result.get("goal")
    plan = result.get("plan")
    if not isinstance(run, Mapping) or not isinstance(validation, Mapping):
        raise ValueError("result must contain run_manifest and validation_report mappings")
    run_id = run.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_manifest.run_id must be a non-empty string")
    status = run.get("status")
    validation_status = run.get("validation_status")
    if hasattr(status, "value"):
        status = status.value
    if hasattr(validation_status, "value"):
        validation_status = validation_status.value
    allowed_run = {"planned", "validating", "queued", "running", "succeeded", "failed", "cancelled", "preempted"}
    allowed_validation = {"not_run", "passed", "failed", "incomplete"}
    if status not in allowed_run or validation_status not in allowed_validation:
        raise ValueError("run or validation status is outside the platform contract")

    outputs = [_artifact_record(ref, role=role) for role, ref in artifact_refs]
    report_record = _artifact_record(validation_report_ref, role="validation-report")
    if not any(item["uri"] == report_record["uri"] for item in outputs):
        outputs.append(report_record)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "status": status,
        "validation_status": validation_status,
        "goal": goal if isinstance(goal, Mapping) else {},
        "producer": {
            "name": "TensorPhysics",
            "version": producer_version,
            "revision": producer_revision,
        },
        "inputs": run.get("inputs", []),
        "configuration_sha256": run.get("config_sha256"),
        "environment": run.get("environment", {}),
        "resources": run.get("resources", {}),
        "metrics": {
            "solver_returncode": result.get("solver_returncode"),
            "validation_check_count": len(validation.get("checks", [])),
            "parameters": plan.get("parameters", {}) if isinstance(plan, Mapping) else {},
        },
        "validation": {
            "status": validation_status,
            "validator": "tensorphysics.research_loop_dem",
            "checks": validation.get("checks", []),
            "summary": validation.get("summary", ""),
            "report_uri": validation_report_ref.uri,
            "limitations": [
                "Output-integrity checks are not a physical-accuracy or icebreaking benchmark validation."
            ],
        },
        "outputs": outputs,
        "parent_run_ids": run.get("parent_run_ids", []),
        "created_at": run.get("created_at"),
        "limitations": [
            "A passed software/output-integrity gate does not establish physical correctness."
        ],
    }
    if status == "failed":
        manifest["failure"] = {"reason": run.get("failure_reason") or validation.get("summary", "run failed")}
    return {key: value for key, value in manifest.items() if value is not None}
