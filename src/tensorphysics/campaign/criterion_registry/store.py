"""Durable freeze store for criterion pre-registration.

The in-memory cycle (``register_goal`` → ``audit_delivery``) only holds the
frozen manifests inside one Python process.  Pre-registration is, however, a
*cross-process* claim: the freeze happens when the mission is created (by the
dispatcher / pr-gate scheduler) and the audit happens later when the delivery
report arrives (possibly in a different process, node and day).  Without a
durable anchor a node can simply register *after* delivering and still look
"registered".

This module closes that gap with a small on-disk freeze store:

``freeze_to_store(directory, mission_id, goal_text, frozen_by)``
    Extract + freeze the goal's structural criteria and write one canonical
    JSON bundle per mission.  Fail-closed: a goal with no numeric level writes
    **nothing** (no file → nothing to audit as registered).  Idempotent: the
    same goal produces byte-identical content and is not rewritten; a
    *different* goal for an already frozen mission raises
    ``PreregistrationConflict`` (silent re-freezing would turn a loosening into
    a fresh "registered" criterion).

``load_bundle`` / ``load_registrations``
    Read the bundle back in any later process.  Each criterion carries its own
    ``sha256`` (per-field canonical JSON, from ``manifest.freeze_manifest``) and
    the bundle carries a ``bundle_sha256`` over the whole payload, so both a
    hand-edited threshold and a hand-edited bundle are detected.  A manifest
    whose hash no longer matches loads as status ``tampered`` — never
    ``registered``.

``audit_from_store(directory, mission_id, report_text, ledger)``
    One-shot: load the frozen bundle (fail-closed to ``unregistered`` when the
    mission was never frozen) and run ``audit_delivery`` against the report.

Zero domain semantics: the store only carries mission ids, actor ids,
timestamps and the six structural criterion fields.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

from .manifest import (
    CriterionManifest,
    verify_frozen_hash,
)
from .registration import Registration, audit_delivery, register_goal

STORE_SCHEMA = "criterion_freeze_store/v1"

# Mission ids become file names: keep them strictly path-safe (no traversal).
_MISSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

TAMPERED = "tampered"
UNREGISTERED = "unregistered"


class PreregistrationConflict(RuntimeError):
    """A frozen mission already exists with different criterion content."""


class StoreCorruptError(RuntimeError):
    """The stored bundle is unreadable or its bundle digest does not match."""


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _check_mission_id(mission_id: Any) -> str:
    if not isinstance(mission_id, str) or not _MISSION_ID_RE.match(mission_id):
        raise ValueError(
            f"invalid mission_id {mission_id!r}: expected [A-Za-z0-9][A-Za-z0-9._-]*"
        )
    return mission_id


def _isoformat(value: Optional[Union[str, datetime]]) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def bundle_path(directory: Union[str, Path], mission_id: str) -> Path:
    """Path of the bundle file for one mission (no side effects)."""
    return Path(directory) / f"{_check_mission_id(mission_id)}.json"


@dataclass
class FrozenBundle:
    """One mission's frozen criterion set as read from / written to disk."""

    mission_id: str
    frozen_by: str
    frozen_at: str
    criteria: list[dict[str, Any]] = field(default_factory=list)
    bundle_sha256: Optional[str] = None
    path: Optional[Path] = None
    registrations: list[Registration] = field(default_factory=list)
    schema: str = STORE_SCHEMA

    @property
    def is_registered(self) -> bool:
        """True when at least one criterion was actually frozen."""
        return bool(self.registrations) and all(
            r.is_registered for r in self.registrations
        )

    @property
    def registered_ids(self) -> list[str]:
        return [r.criterion_id for r in self.registrations if r.is_registered]

    def payload(self) -> dict[str, Any]:
        """Digest-bearing payload (bundle_sha256 excluded from its own hash)."""
        return {
            "schema": self.schema,
            "mission_id": self.mission_id,
            "frozen_by": self.frozen_by,
            "frozen_at": self.frozen_at,
            "criteria": self.criteria,
        }

    def compute_sha256(self) -> str:
        return _sha256(_canonical_json(self.payload()))

    def to_dict(self) -> dict[str, Any]:
        data = self.payload()
        data["bundle_sha256"] = self.bundle_sha256 or self.compute_sha256()
        return data

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


def _criteria_fingerprint(criteria: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Identity of a criterion set: (criterion_id, field-level sha256) pairs.

    Metadata (``frozen_at`` / ``frozen_by``) is deliberately excluded so that
    re-freezing the *same* goal is idempotent no matter when it happens.
    """
    return [(c.get("criterion_id", ""), c.get("sha256") or "") for c in criteria]


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".freeze-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def freeze_to_store(
    directory: Union[str, Path],
    mission_id: str,
    goal_text: Any,
    frozen_by: str,
    frozen_at: Optional[Union[str, datetime]] = None,
) -> FrozenBundle:
    """Freeze a mission goal into the store (idempotent, fail-closed).

    Args:
        directory: store directory (created on demand)
        mission_id: path-safe mission identifier
        goal_text: mission goal text (domain content arrives here, never below)
        frozen_by: actor that pre-registers (agent id / role)
        frozen_at: optional timestamp (ISO string or datetime)

    Returns:
        The ``FrozenBundle`` that is now on disk.  When the goal yields no
        criterion with a numeric level, nothing is written and the returned
        bundle carries a single ``unregistered`` registration.

    Raises:
        PreregistrationConflict: the mission is already frozen with different
            criterion content (no silent overwrite).
        ValueError: invalid ``mission_id``.
    """
    _check_mission_id(mission_id)
    path = bundle_path(directory, mission_id)
    stamp = _isoformat(frozen_at)

    registrations = register_goal(goal_text, frozen_by=frozen_by, frozen_at=stamp)

    if any(r.is_registered for r in registrations):
        criteria = [
            r.manifest.to_dict() for r in registrations if r.manifest is not None
        ]
    else:
        criteria = []

    candidate = FrozenBundle(
        mission_id=mission_id,
        frozen_by=frozen_by,
        frozen_at=stamp,
        criteria=criteria,
        registrations=registrations,
        path=path,
    )
    candidate.bundle_sha256 = candidate.compute_sha256()

    if path.exists():
        existing = load_bundle(directory, mission_id)
        if existing is not None and \
                _criteria_fingerprint(existing.criteria) == _criteria_fingerprint(criteria) and \
                existing.frozen_by == frozen_by:
            # Idempotent re-freeze of the identical criterion set.
            return existing
        raise PreregistrationConflict(
            f"mission {mission_id!r} is already frozen at {path} with different "
            "criterion content; refusing to overwrite a pre-registration"
        )

    if criteria:
        _atomic_write(path, candidate.to_canonical_json() + "\n")
    return candidate


def load_bundle(
    directory: Union[str, Path],
    mission_id: str,
) -> Optional[FrozenBundle]:
    """Load one mission's frozen bundle, verifying both digest layers.

    Returns ``None`` when the mission was never frozen.  Per-criterion hash
    mismatches do not raise: those criteria load with status ``tampered`` so
    the audit reports tampering instead of crashing (fail-closed either way).
    A bundle-level digest mismatch or unreadable JSON raises
    ``StoreCorruptError``.
    """
    path = bundle_path(directory, mission_id)
    if not path.exists():
        return None

    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise StoreCorruptError(f"unreadable freeze bundle {path}: {exc}") from exc

    schema = data.get("schema")
    if schema != STORE_SCHEMA:
        raise StoreCorruptError(f"unsupported freeze store schema {schema!r} in {path}")

    recorded = data.get("bundle_sha256")
    registrations: list[Registration] = []
    criteria = data.get("criteria") or []

    for cdata in criteria:
        manifest = CriterionManifest.from_dict(cdata)
        if verify_frozen_hash(manifest):
            registrations.append(Registration(
                criterion_id=manifest.criterion_id,
                status="registered",
                manifest=manifest,
            ))
        else:
            registrations.append(Registration(
                criterion_id=manifest.criterion_id,
                status=TAMPERED,
                reason="frozen manifest sha256 mismatch (criterion edited after freeze)",
                manifest=manifest,
            ))

    bundle = FrozenBundle(
        mission_id=data.get("mission_id", mission_id),
        frozen_by=data.get("frozen_by", ""),
        frozen_at=data.get("frozen_at", ""),
        criteria=criteria,
        bundle_sha256=recorded,
        path=path,
        registrations=registrations,
    )

    recomputed = bundle.compute_sha256()
    if recorded != recomputed:
        raise StoreCorruptError(
            f"bundle digest mismatch in {path}: recorded={recorded} "
            f"recomputed={recomputed}"
        )
    return bundle


def load_registrations(
    directory: Union[str, Path],
    mission_id: str,
) -> list[Registration]:
    """Load a mission's registrations for ``audit_delivery`` (fail-closed).

    A mission that was never frozen returns a single ``unregistered``
    registration, so an audit can never report "unchanged" for a criterion
    that was not pre-registered.
    """
    bundle = load_bundle(directory, mission_id)
    if bundle is None:
        return [Registration(
            criterion_id="criterion_000",
            status=UNREGISTERED,
            reason=(
                f"no frozen bundle for mission {mission_id!r} in "
                f"{Path(directory)} (fail-closed)"
            ),
        )]
    if not bundle.registrations:
        return [Registration(
            criterion_id="criterion_000",
            status=UNREGISTERED,
            reason="frozen bundle contains no criterion with a numeric level (fail-closed)",
        )]
    return list(bundle.registrations)


def audit_from_store(
    directory: Union[str, Path],
    mission_id: str,
    report_text: Any,
    ledger: Any = None,
):
    """Load the frozen bundle and audit a delivery report against it."""
    return audit_delivery(
        load_registrations(directory, mission_id),
        report_text,
        mission_id=mission_id,
        ledger=ledger,
    )