"""Optional client for registering run provenance with TensorPlatform.

The client sends only compact metadata and artifact references; it never uploads
solver fields or artifact bytes. The target API is TensorPlatform's
/api/v1/tensorphysics/runs endpoint.
"""
from __future__ import annotations

import json
import os
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .provenance import RunManifest


# Keep this in sync with TensorPlatform's RunRegistration validator (1,000,000 bytes).
_MAX_MANIFEST_BYTES = 1_000_000
_RUN_STATUSES = frozenset({
    "planned", "validating", "queued", "running", "succeeded", "failed",
    "cancelled", "preempted",
})
_VALIDATION_STATUSES = frozenset({"not_run", "passed", "failed", "incomplete"})


class PlatformRegistryError(RuntimeError):
    """Registration failed or the platform returned an error response."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class TensorPlatformRunRegistry:
    """Small stdlib-only HTTP client for the TensorPlatform run registry."""

    def __init__(
        self,
        base_url: str,
        bearer_token: str,
        organization_id: int,
        *,
        timeout: float = 10.0,
    ) -> None:
        base_url = base_url.strip().rstrip("/")
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        if not bearer_token.strip():
            raise ValueError("bearer_token must not be empty")
        if not isinstance(organization_id, int) or isinstance(organization_id, bool) or organization_id < 1:
            raise ValueError("organization_id must be a positive integer")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.base_url = base_url
        self._bearer_token = bearer_token.strip()
        self.organization_id = organization_id
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "TensorPlatformRunRegistry":
        """Create the client from TENSORPLATFORM_URL, API_TOKEN and ORGANIZATION_ID."""
        try:
            base_url = os.environ["TENSORPLATFORM_URL"]
            token = os.environ["TENSORPLATFORM_API_TOKEN"]
            organization_id = int(os.environ["TENSORPLATFORM_ORGANIZATION_ID"])
        except (KeyError, ValueError) as exc:
            raise ValueError(
                "Set TENSORPLATFORM_URL, TENSORPLATFORM_API_TOKEN and "
                "TENSORPLATFORM_ORGANIZATION_ID to enable platform registration"
            ) from exc
        timeout = float(os.getenv("TENSORPLATFORM_TIMEOUT_SECONDS", "10"))
        return cls(base_url, token, organization_id, timeout=timeout)

    def register(self, manifest: RunManifest | Mapping[str, Any]) -> dict[str, Any]:
        """Register one manifest; exact retries are idempotent on the platform.

        Invalid status values and oversized metadata are rejected locally so a
        solver workflow can fail fast without making an HTTP request.
        """
        payload_manifest = manifest.to_dict() if isinstance(manifest, RunManifest) else dict(manifest)
        run_id = payload_manifest.get("run_id")
        status = payload_manifest.get("status")
        validation_status = payload_manifest.get("validation_status")
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("manifest.run_id must be a non-empty string")
        if run_id != run_id.strip():
            raise ValueError("manifest.run_id must not contain leading or trailing whitespace")
        if status not in _RUN_STATUSES:
            raise ValueError(f"manifest.status must be one of: {', '.join(sorted(_RUN_STATUSES))}")
        if validation_status not in _VALIDATION_STATUSES:
            raise ValueError(
                "manifest.validation_status must be one of: "
                f"{', '.join(sorted(_VALIDATION_STATUSES))}"
            )

        try:
            manifest_bytes = json.dumps(
                payload_manifest, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False, allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"manifest must be JSON-serializable without NaN or Infinity: {exc}") from exc
        if len(manifest_bytes) > _MAX_MANIFEST_BYTES:
            raise ValueError(
                f"manifest exceeds TensorPlatform's {_MAX_MANIFEST_BYTES}-byte limit"
            )

        payload = {
            "run_id": run_id,
            "status": status,
            "validation_status": validation_status,
            "organization_id": self.organization_id,
            "manifest": payload_manifest,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
        request = Request(
            f"{self.base_url}/api/v1/tensorphysics/runs",
            data=encoded,
            headers={
                "Authorization": f"Bearer {self._bearer_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Idempotency-Key": run_id,
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read(4096).decode("utf-8", errors="replace")
            raise PlatformRegistryError(
                f"TensorPlatform rejected run registration (HTTP {exc.code}): {detail}",
                status_code=exc.code,
            ) from exc
        except URLError as exc:
            raise PlatformRegistryError(f"TensorPlatform is unreachable: {exc.reason}") from exc
        except (TimeoutError, OSError) as exc:
            raise PlatformRegistryError(f"TensorPlatform request failed: {exc}") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PlatformRegistryError("TensorPlatform returned invalid JSON") from exc

        if not isinstance(result, dict) or result.get("run_id") != run_id:
            raise PlatformRegistryError("TensorPlatform returned an unexpected registration response")
        return result
