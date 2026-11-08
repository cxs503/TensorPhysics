"""Evidence collection for the P35 acceptance gate.

Two independent evidence sources are merged into one snapshot:

* **platform** -- the mission row, the ``campaign_plans`` node row, the reports
  submitted for the mission and the booked capability delta.  This is what the
  dispatcher itself knows; it is fetched over HTTP with the agent token and
  written to disk verbatim so a verdict can be re-rendered offline
  (``--snapshot``) and re-audited without the platform.
* **repository** -- the artifact families each node promised, resolved inside the
  node's checkout (``--repo-root NAME=PATH``).  JSON is parsed, PNG magic and
  IEND trailer are checked, sha256 recorded.  A repository the gate cannot see is
  reported as *unscannable*, never as empty-and-therefore-failing.

Nothing here decides anything: collection only records what exists, with the
paths and hashes that produced the record.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import spec as S

SNAPSHOT_SCHEMA = "p35_platform_snapshot.v1"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
PNG_IEND = b"IEND\xaeB`\x82"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------
# artifact scanning
# --------------------------------------------------------------------------
def _png_is_valid(path: Path) -> tuple[bool, str]:
    try:
        blob = path.read_bytes()
    except OSError as exc:
        return False, f"unreadable: {exc}"
    if not blob.startswith(PNG_MAGIC):
        return False, "bad_png_magic"
    if not blob.endswith(PNG_IEND):
        return False, "missing_iend_trailer"
    if len(blob) < 100:
        return False, "png_too_small"
    # width/height from IHDR, purely informational for the report
    try:
        width, height = struct.unpack(">II", blob[16:24])
    except struct.error:
        return False, "ihdr_unreadable"
    return True, f"{width}x{height}"


def scan_artifact_family(root: Path, requirement: S.ArtifactRequirement) -> dict:
    """Resolve one declared artifact family under ``root`` with integrity checks."""
    files = sorted(p for p in root.glob(requirement.pattern) if p.is_file())
    resolved = []
    for p in files:
        rec = {"path": str(p.relative_to(root)), "sha256": sha256_file(p),
               "bytes": p.stat().st_size, "valid": True, "reason": ""}
        if requirement.kind == "json":
            try:
                json.loads(p.read_text())
            except (OSError, ValueError) as exc:
                rec["valid"] = False
                rec["reason"] = f"json_invalid: {exc}"
        elif requirement.kind == "png":
            ok, reason = _png_is_valid(p)
            rec["valid"] = ok
            rec["reason"] = reason
        else:  # pragma: no cover - spec guard
            rec["valid"] = False
            rec["reason"] = f"unknown kind {requirement.kind}"
        resolved.append(rec)
    return {
        "kind": requirement.kind,
        "pattern": requirement.pattern,
        "note": requirement.note,
        "resolved": resolved,
        "valid_count": sum(1 for r in resolved if r["valid"]),
        "satisfied": any(r["valid"] for r in resolved),
    }


def git_head(root: Path) -> str | None:
    """HEAD of a checkout, used as checkout identity in the evidence chain."""
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def scan_node_repository(root: str | Path, node: S.NodeSpec, *,
                         authoritative: bool = False) -> dict:
    """Scan every declared artifact family of ``node`` inside one checkout.

    ``authoritative`` marks a checkout that *is* the node's delivery worktree
    (``--node-root``): there an absent artifact family is a definite failure.
    For a repository-level binding (``--repo-root``) absence is inconclusive --
    the gate may simply be looking at a different branch -- and is reported as
    ``WITHHELD`` instead of being turned into a fabricated FAIL.
    """
    base = Path(root)
    if not base.is_dir():
        return {"repo_root": str(base), "scannable": False, "authoritative": authoritative,
                "repo_head": None,
                "reason": f"repository root not available: {base}", "families": []}
    return {
        "repo_root": str(base.resolve()),
        "repo_head": git_head(base),
        "authoritative": authoritative,
        "scannable": True,
        "reason": "",
        "families": [scan_artifact_family(base, req) for req in node.artifacts],
    }


# --------------------------------------------------------------------------
# platform client
# --------------------------------------------------------------------------
class PlatformClient:
    """Minimal stdlib client for the four platform reads the gate needs."""

    def __init__(self, base_url: str, token: str, *, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def _get(self, path: str, **params) -> tuple[object, str]:
        url = self.base_url + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={
            "Authorization": f"Bearer {self.token}", "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read()), ""
        except urllib.error.HTTPError as exc:
            return None, f"HTTP {exc.code}"
        except (urllib.error.URLError, OSError, ValueError) as exc:
            return None, f"{type(exc).__name__}: {exc}"

    @staticmethod
    def _items(payload) -> list:
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("items", "data", "results"):
                if isinstance(payload.get(key), list):
                    return payload[key]
            return [payload]
        return []

    def plan(self, plan_id: int) -> tuple[dict | None, str]:
        return self._get(f"/tensor/campaigns/plans/{plan_id}")

    def plan_nodes(self, plan_id: int) -> tuple[list, str]:
        payload, err = self._get(f"/tensor/campaigns/plans/{plan_id}/nodes")
        return (self._items(payload) if payload is not None else []), err

    def mission(self, mission_id: int) -> tuple[dict | None, str]:
        return self._get(f"/tensor/missions/{mission_id}")

    def reports(self, mission_id: int) -> tuple[list, str]:
        payload, err = self._get("/tensor/reports", mission_id=mission_id, limit=50)
        return (self._items(payload) if payload is not None else []), err

    def capability_delta(self, mission_id: int) -> tuple[dict | None, str]:
        payload, err = self._get(f"/tensor/missions/{mission_id}/capability-delta")
        if payload is not None and isinstance(payload, dict) and "err" in payload:
            return None, str(payload["err"])
        return (payload if isinstance(payload, dict) else None), err


# --------------------------------------------------------------------------
# snapshot assembly
# --------------------------------------------------------------------------
def _summarise_report(report: dict) -> dict:
    figures = report.get("figures") or []
    if isinstance(figures, dict):
        figures = list(figures.keys())
    names = []
    for fig in figures:
        if isinstance(fig, dict):
            names.append(str(fig.get("name")))
        else:
            names.append(str(fig))
    markdown = report.get("markdown") or ""
    return {
        "report_id": report.get("id") or report.get("report_id"),
        "title": report.get("title"),
        "mission_id": report.get("mission_id"),
        "figures": names,
        "markdown_chars": len(markdown),
        "created_utc": report.get("created_utc") or report.get("created_at"),
    }


def _summarise_delta(delta: dict | None) -> dict | None:
    if not delta:
        return None
    evidence = delta.get("evidence")
    return {
        "category": delta.get("category"),
        "summary": delta.get("summary"),
        "mission_repo": delta.get("mission_repo"),
        "evidence": evidence if isinstance(evidence, dict) else {},
        "artifacts": delta.get("artifacts") or [],
        "created_utc": delta.get("created_utc") or delta.get("created_at"),
    }


def build_snapshot(client: PlatformClient, *, plan_id: int = S.PLAN_ID,
                   collected_utc: str | None = None) -> dict:
    """Fetch the platform half of the evidence and return a serialisable snapshot."""
    errors: list[dict] = []
    plan, err = client.plan(plan_id)
    if err:
        errors.append({"source": f"plan:{plan_id}", "error": err})
    plan_nodes, err = client.plan_nodes(plan_id)
    if err:
        errors.append({"source": f"plan_nodes:{plan_id}", "error": err})
    by_mission = {}
    for row in plan_nodes:
        if isinstance(row, dict) and row.get("mission_id"):
            by_mission[int(row["mission_id"])] = row

    nodes = []
    for node in S.NODES:
        mission, m_err = client.mission(node.mission_id)
        if m_err:
            errors.append({"source": f"mission:{node.mission_id}", "error": m_err})
        reports, r_err = client.reports(node.mission_id)
        if r_err:
            errors.append({"source": f"reports:{node.mission_id}", "error": r_err})
        delta, d_err = client.capability_delta(node.mission_id)
        if d_err:
            errors.append({"source": f"capability_delta:{node.mission_id}", "error": d_err})
        row = by_mission.get(node.mission_id) or {}
        meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        nodes.append({
            "ext_id": node.ext_id,
            "mission_id": node.mission_id,
            "declared_repo": node.repo,
            "declared_physics_domain": node.physics_domain,
            "plan_node_id": row.get("id"),
            "plan_node_name": row.get("name"),
            "plan_node_status": row.get("status"),
            "plan_dependencies": row.get("dependencies") or [],
            "platform_repo": meta.get("repo") or (mission or {}).get("repo"),
            "mission_status": (mission or {}).get("status"),
            "mission_result": (mission or {}).get("result") or {},
            "reports": [_summarise_report(r) for r in reports],
            "capability_delta": _summarise_delta(delta),
        })

    plan_meta = plan or {}
    return {
        "schema": SNAPSHOT_SCHEMA,
        "campaign_id": S.CAMPAIGN_ID,
        "gate_id": S.GATE_ID,
        "collected_utc": collected_utc or utcnow(),
        "plan": {
            "id": plan_meta.get("id", plan_id),
            "title": plan_meta.get("title"),
            "name": plan_meta.get("name"),
            "status": plan_meta.get("status"),
            "physics_scope": plan_meta.get("physics_scope") or [],
            "n_plan_nodes": len(plan_nodes),
        },
        "nodes": nodes,
        "errors": errors,
    }


def write_snapshot(snapshot: dict, path: str | Path) -> dict:
    """Write the snapshot and return its provenance record."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False, sort_keys=False))
    return {"path": str(p), "sha256": sha256_file(p), "bytes": p.stat().st_size}


def load_snapshot(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def parse_repo_roots(values) -> dict[str, str]:
    """Parse ``NAME=PATH`` repository-root bindings given on the command line."""
    roots: dict[str, str] = {}
    for item in values or ():
        if "=" not in item:
            raise ValueError(f"--repo-root expects NAME=PATH, got {item!r}")
        name, path = item.split("=", 1)
        name, path = name.strip(), path.strip()
        if not name or not path:
            raise ValueError(f"--repo-root expects NAME=PATH, got {item!r}")
        roots[name] = os.path.abspath(path)
    return roots