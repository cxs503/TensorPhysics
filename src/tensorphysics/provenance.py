"""Versioned run provenance and artifact-reference contracts.

These records contain metadata and URIs only. Bulk arrays and tensor payloads are
owned by the solver/output backend and must not be embedded in run manifests.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
import json
import re
from typing import Any, Mapping


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ManifestError(ValueError):
    """Run metadata or artifact reference violates its public contract."""


class RunStatus(str, Enum):
    PLANNED = "planned"
    VALIDATING = "validating"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PREEMPTED = "preempted"


class ValidationStatus(str, Enum):
    NOT_RUN = "not_run"
    PASSED = "passed"
    FAILED = "failed"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True)
class ArtifactRef:
    """Reference to an immutable artifact stored outside the manifest."""

    uri: str
    media_type: str
    checksum_sha256: str | None = None
    size_bytes: int | None = None
    schema_name: str | None = None
    schema_version: int = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.uri, str) or not self.uri.strip():
            raise ManifestError("artifact uri must be a non-empty string")
        if not isinstance(self.media_type, str) or not self.media_type.strip():
            raise ManifestError("artifact media_type must be a non-empty string")
        if self.checksum_sha256 is not None and not _SHA256_RE.fullmatch(self.checksum_sha256):
            raise ManifestError("checksum_sha256 must be a 64-character lowercase SHA-256 hex digest")
        if self.size_bytes is not None and (
            not isinstance(self.size_bytes, int) or isinstance(self.size_bytes, bool)
            or self.size_bytes < 0
        ):
            raise ManifestError("size_bytes must be a non-negative integer")
        if not isinstance(self.schema_version, int) or isinstance(self.schema_version, bool) or self.schema_version < 1:
            raise ManifestError("schema_version must be a positive integer")
        if not isinstance(self.metadata, Mapping):
            raise ManifestError("artifact metadata must be a mapping")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunManifest:
    """Machine-readable provenance for one solver/workflow execution."""

    run_id: str
    status: RunStatus
    created_at: str
    config: Mapping[str, Any]
    config_sha256: str
    code_revisions: Mapping[str, str]
    environment: Mapping[str, Any] = field(default_factory=dict)
    resources: Mapping[str, Any] = field(default_factory=dict)
    random_seeds: Mapping[str, int] = field(default_factory=dict)
    inputs: list[ArtifactRef] = field(default_factory=list)
    outputs: list[ArtifactRef] = field(default_factory=list)
    validation_status: ValidationStatus = ValidationStatus.NOT_RUN
    validation_report: ArtifactRef | None = None
    failure_reason: str | None = None
    parent_run_ids: list[str] = field(default_factory=list)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ManifestError("run_id must be a non-empty string")
        try:
            datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        except (AttributeError, ValueError) as exc:
            raise ManifestError("created_at must be an ISO-8601 datetime string") from exc
        if not isinstance(self.config, Mapping):
            raise ManifestError("config must be a mapping")
        if not isinstance(self.config_sha256, str) or not _SHA256_RE.fullmatch(self.config_sha256):
            raise ManifestError("config_sha256 must be a 64-character lowercase SHA-256 hex digest")
        if not isinstance(self.code_revisions, Mapping) or not self.code_revisions:
            raise ManifestError("code_revisions must identify at least one repository revision")
        if not isinstance(self.status, RunStatus):
            try:
                self.status = RunStatus(self.status)
            except ValueError as exc:
                raise ManifestError(f"unknown run status: {self.status!r}") from exc
        if not isinstance(self.validation_status, ValidationStatus):
            try:
                self.validation_status = ValidationStatus(self.validation_status)
            except ValueError as exc:
                raise ManifestError(f"unknown validation status: {self.validation_status!r}") from exc
        for name, seed in self.random_seeds.items():
            if not isinstance(name, str) or not name.strip():
                raise ManifestError("random seed names must be non-empty strings")
            if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
                raise ManifestError(f"random seed {name!r} must be a non-negative integer")
        if self.schema_version != 1:
            raise ManifestError(f"unsupported run manifest schema version: {self.schema_version}")
        if self.status is RunStatus.SUCCEEDED and self.failure_reason:
            raise ManifestError("a succeeded run cannot have failure_reason")
        if self.validation_status is ValidationStatus.PASSED and self.status is not RunStatus.SUCCEEDED:
            raise ManifestError("validation cannot pass for a run that has not succeeded")

    @staticmethod
    def digest_config(config: Mapping[str, Any]) -> str:
        """Return a stable digest for JSON-compatible normalized configuration."""
        try:
            encoded = json.dumps(
                config, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ManifestError(f"config is not canonical JSON-compatible data: {exc}") from exc
        return hashlib.sha256(encoded).hexdigest()

    def verify_config_digest(self) -> None:
        actual = self.digest_config(self.config)
        if actual != self.config_sha256:
            raise ManifestError(
                f"config digest mismatch: manifest={self.config_sha256}, computed={actual}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata to JSON-compatible values; never embeds bulk fields."""
        data = asdict(self)
        data["status"] = self.status.value
        data["validation_status"] = self.validation_status.value
        return data

    def to_json(self) -> str:
        try:
            return json.dumps(
                self.to_dict(), sort_keys=True, ensure_ascii=False, allow_nan=False,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as exc:
            raise ManifestError(f"manifest contains non-JSON metadata: {exc}") from exc
