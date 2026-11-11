#!/usr/bin/env python3
"""Export the capability-reuse snapshot: delivered capabilities + evidence + repo manifest.

The snapshot is the *only* input of the graph builder, so the whole audit is a
pure function of one hashed JSON file.  Everything domain-specific (repository
roots, aliases, evidence limits, DSN) lives in
``src/tensorphysics/capability_reuse/configs/repos.json``.

Usage::

    python tools/capability_reuse/export_snapshot.py --out artifacts/capability_reuse/snapshot.json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.capability_reuse.model import SNAPSHOT_SCHEMA, canonical_hash  # noqa: E402

PATH_RE = re.compile(
    r"(?<![\w/.\-])((?:[\w.\-]+/)*[\w.\-]+\.(?:py|c|cc|cpp|cu|cuh|h|hpp|md|json|ya?ml|txt|png|npy))(?![\w/.\-])")
NODE_NAME_RE = re.compile(r"\b([A-Za-z][\w-]*-?N\d{1,3}|m\d{3,5}|p\d+w\d+-?N?\d*)\b")

QUERIES = {
    "nodes": """
        SELECT n.id, n.name, n.mission_id, n.metadata->>'repo', n.metadata->>'ext_id',
               n.metadata->>'physics_domain', n.metadata->>'role',
               n.metadata->>'dependencies_hint', coalesce(n.metadata->>'goal',''),
               coalesce(n.metadata->'acceptance_criteria', '[]'::jsonb)::text,
               coalesce(n.metadata->>'dependencies', ''),
               coalesce(n.actual_end, n.updated_at)::text
        FROM campaign_nodes n
        WHERE n.status = 'done'
        ORDER BY n.id
    """,
    "reports": """
        SELECT mission_id, id, markdown FROM tensor_reports
        WHERE mission_id IS NOT NULL ORDER BY mission_id, id
    """,
    "ledger": """
        SELECT slug, title, domain, status, coalesce(evidence,''), mission_id,
               coalesce(done_at, created_at)::text
        FROM feature_ledger ORDER BY slug
    """,
}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _walk_manifest(root: str, suffixes, skip_parts) -> List[str]:
    out: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root).replace(os.sep, "/")
        rel_dir = "" if rel_dir == "." else rel_dir
        probe = "/" + rel_dir + "/" if rel_dir else "/"
        dirnames[:] = sorted(d for d in dirnames
                             if not any(s in "/" + (rel_dir + "/" if rel_dir else "") + d + "/"
                                        for s in skip_parts))
        for fn in sorted(filenames):
            if not fn.endswith(tuple(suffixes)):
                continue
            rel = f"{rel_dir}/{fn}" if rel_dir else fn
            if any(s in "/" + rel for s in skip_parts):
                continue
            try:
                if os.path.getsize(os.path.join(dirpath, fn)) > 400_000:
                    continue
            except OSError:
                continue
            out.append(rel)
    return sorted(set(out))


def _git(root: str, *args: str) -> str:
    try:
        return subprocess.run(["git", "-C", root, *args], capture_output=True, text=True,
                              timeout=60).stdout.strip()
    except Exception:  # pragma: no cover - provenance only
        return ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "artifacts/capability_reuse/snapshot.json"))
    ap.add_argument("--config", default=str(ROOT / "src/tensorphysics/capability_reuse/configs/repos.json"))
    args = ap.parse_args(argv)

    cfg = json.loads(Path(args.config).read_text())
    aliases = cfg.get("repo_aliases") or {}
    limits = cfg["evidence_limits"]

    dsn = os.environ.get(cfg["db"].get("dsn_env", ""), "") or cfg["db"].get("default_dsn", "")
    import psycopg  # local import: the exporter needs a driver, the graph builder does not

    conn = psycopg.connect(dsn)
    cur = conn.cursor()
    tables = {}
    for key, sql in QUERIES.items():
        cur.execute(sql)
        tables[key] = cur.fetchall()
    conn.close()

    # ---- repository manifest -------------------------------------------------
    repos: Dict[str, dict] = {}
    for entry in cfg["repos"]:
        name, root = entry["name"], entry["root"]
        files = _walk_manifest(root, cfg["code_suffixes"], cfg["skip_path_parts"])
        repos[name] = {
            "root": root,
            "head": _git(root, "rev-parse", "HEAD")[:12],
            "dirty": bool(_git(root, "status", "--porcelain")),
            "import_roots": cfg["import_roots"],
            "files": files,
            "file_count": len(files),
        }

    manifest = {r: set(v["files"]) for r, v in repos.items()}

    # ---- report evidence lines (bounded, path-bearing only) -------------------
    per_mission: Dict[int, List[dict]] = {}
    for mission_id, report_id, markdown in tables["reports"]:
        rows = []
        for i, line in enumerate((markdown or "").splitlines(), 1):
            if PATH_RE.search(line):
                rows.append({"report_id": report_id, "line": i, "text": line.strip()[: limits["chars_per_line"]]})
        if rows:
            per_mission.setdefault(int(mission_id), []).extend(rows)

    # ---- capabilities -------------------------------------------------------
    caps = []
    unmapped_repo = 0
    name_counter = {}
    for row in tables["nodes"]:
        name_counter[row[1]] = name_counter.get(row[1], 0) + 1
    for (nid, name, mission_id, repo, ext_id, domain, role, dep_hint, goal, crit_json,
         deps, done_at) in tables["nodes"]:
        repo = aliases.get(repo or "", repo or "")
        if repo not in repos:
            unmapped_repo += 1
            repo = repo or ""
        try:
            crit = json.loads(crit_json)
        except Exception:
            crit = []
        evidence = per_mission.get(int(mission_id), [])[: limits["lines_per_capability"]] if mission_id else []
        hints = []
        for blob in (dep_hint or "", deps or ""):
            hints.extend(NODE_NAME_RE.findall(blob))
        caps.append({
            "cid": name if name_counter.get(name, 0) == 1 else f"{name}#{nid}",
            "node_id": int(nid),
            "name": name,
            "repo": repo,
            "domain": domain,
            "cap_class": domain,
            "mission_id": int(mission_id) if mission_id else None,
            "status": "done",
            "done_at": (done_at or "")[:19],
            "origin": "node",
            "ext_id": ext_id,
            "role": role,
            "goal": (goal or "")[: limits["goal_chars"]],
            "acceptance_criteria": [str(c)[: limits["criteria_chars"]] for c in (crit or [])][:12],
            "dependency_hints": sorted(set(hints))[:20],
            "evidence": evidence,
        })

    for slug, title, domain, status, evidence, mission_id, done_at in tables["ledger"]:
        if status not in ("done", "verified"):
            continue
        rows = []
        if mission_id:
            rows = per_mission.get(int(mission_id), [])[: limits["lines_per_capability"]]
        caps.append({
            "cid": f"FEAT::{slug}",
            "node_id": None,
            "name": f"FEAT::{slug}",
            "repo": "TensorPlatform",
            "domain": domain,
            "cap_class": domain,
            "mission_id": int(mission_id) if mission_id else None,
            "status": status,
            "done_at": (done_at or "")[:19],
            "origin": "feature_ledger",
            "ext_id": slug,
            "role": "feature",
            "goal": f"{title} | {evidence}"[: limits["goal_chars"]],
            "acceptance_criteria": [],
            "dependency_hints": [],
            "evidence": rows,
        })

    caps.sort(key=lambda c: c["cid"])

    snap = {
        "schema": SNAPSHOT_SCHEMA,
        "provenance": {
            "exported_at_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "database": "tensorplatform (campaign_nodes, tensor_reports, feature_ledger)",
            "queries_sha256": {k: _sha(v) for k, v in sorted(QUERIES.items())},
            "counts": {"nodes_done": len(tables["nodes"]), "reports": len(tables["reports"]),
                       "ledger_rows": len(tables["ledger"]), "capabilities": len(caps),
                       "caps_repo_unresolved": unmapped_repo,
                       "evidence_lines": sum(len(c["evidence"]) for c in caps)},
            "repos": {r: {"head": v["head"], "dirty": v["dirty"], "file_count": v["file_count"]}
                      for r, v in sorted(repos.items())},
        },
        "repos": repos,
        "capabilities": caps,
    }
    snap["provenance"]["snapshot_content_sha256"] = canonical_hash(
        {"repos": {r: {"head": v["head"], "files_sha": _sha("\n".join(v["files"]))}
                   for r, v in sorted(repos.items())},
         "capabilities": caps})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snap, ensure_ascii=False, sort_keys=True, indent=1))
    print(json.dumps({"out": str(out), "size_mb": round(out.stat().st_size / 1048576, 2),
                      "capabilities": len(caps),
                      "repos": {r: v["file_count"] for r, v in sorted(repos.items())},
                      "counts": snap["provenance"]["counts"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())