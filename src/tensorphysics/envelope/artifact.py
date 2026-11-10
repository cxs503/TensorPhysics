"""Artifact writer: deterministic JSON + digests, so a number can be re-derived.

Every artifact carries

* ``code_digest`` -- SHA-256 over the package sources of
  ``tensorphysics.envelope`` (sorted by relative path), so a result can be tied
  to the exact code that produced it;
* ``case_digest``  -- SHA-256 of the canonical case configuration;
* per-result ``source_digest`` copied from the case.

Writing is atomic (``tmp`` + ``replace``) and byte-deterministic: no timestamps,
no environment capture, no dict ordering surprises.  Anything variable
(durations, machine) is passed in by the caller and kept in a separate section
of the payload, so re-running on another machine cannot invalidate the digests.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any, Mapping

__all__ = ["artifact_digest", "canonical_json", "code_digest", "write_artifact"]

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def code_digest() -> str:
    digest = hashlib.sha256()
    for path in sorted(PACKAGE_ROOT.glob("*.py"), key=lambda p: p.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def artifact_digest(
    payload: Mapping[str, Any],
    drop: tuple[str, ...] = ("code_digest", "artifact_digest"),
) -> str:
    body = {k: v for k, v in payload.items() if k not in drop}
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


def write_artifact(path: str | pathlib.Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    enriched = dict(payload)
    enriched.setdefault("code_digest", code_digest())
    enriched["artifact_digest"] = artifact_digest(enriched)
    text = json.dumps(enriched, indent=2, sort_keys=True) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
    return enriched


def read_artifact(path: str | pathlib.Path) -> dict[str, Any]:
    payload = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    stored = payload.get("artifact_digest")
    if stored is not None and stored != artifact_digest(payload):
        raise ValueError(f"{path}: artifact digest mismatch (hand-edited payload)")
    return payload