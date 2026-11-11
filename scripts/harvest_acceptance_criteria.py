#!/usr/bin/env python3
"""Freeze the campaign acceptance-criteria corpus (P106W3A2-N08).

The census in :mod:`tensorphysics.acceptance_judgeability` must be reproducible and
must not carry any hand-copied number: this script is the *only* producer of the
corpus, it reads the live campaign API (read-only), and it freezes the result as a
JSONL file plus a sha256 manifest so a downstream run can prove it saw the same bytes.

Sources (all read-only GETs on the platform API):

* ``/tensor/campaigns/plans``              -- plan inventory (latest page);
* ``/tensor/campaigns/plans/{id}/nodes``   -- node rows incl. ``metadata.acceptance_criteria``;
* ``/tensor/reports?mission_id={mid}``     -- published-report evidence per node;
* ``/tensor/missions/{mid}``               -- mission status/result (accepted flag).

Usage::

    TENSOR_AGENT_USER=... TENSOR_AGENT_PASS=... \
        python scripts/harvest_acceptance_criteria.py --out artifacts/acceptance_judgeability/corpus.jsonl

Scope is declared explicitly in the manifest (``plan_id_min``/``plan_id_max`` and
``plans_scanned``) so the report never implies more coverage than was harvested.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable

DEFAULT_BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")


# --------------------------------------------------------------------------- transport
class Client:
    """Minimal authenticated JSON client with retry (no writes, no clock dependence)."""

    def __init__(self, base: str, user: str, password: str, timeout: float = 60.0) -> None:
        self.base = base.rstrip("/")
        self.timeout = timeout
        self._token = self._login(user, password)
        self.calls = 0

    def _login(self, user: str, password: str) -> str:
        body = json.dumps({"username": user, "password": password}).encode()
        req = urllib.request.Request(
            self.base + "/auth/login", data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.load(resp)["access_token"]

    def get(self, path: str, *, allow_404: bool = True) -> Any | None:
        url = self.base + path
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers={"Authorization": "Bearer " + self._token})
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    self.calls += 1
                    return json.load(resp)
            except urllib.error.HTTPError as exc:
                if exc.code == 404 and allow_404:
                    return None
                if exc.code == 405:  # endpoint exists, wrong verb -> treat as absent
                    return None
                if attempt == 2:
                    raise
                time.sleep(0.4 * (attempt + 1))
            except (urllib.error.URLError, http.client.HTTPException, OSError):
                if attempt == 2:
                    raise
                time.sleep(0.5 * (attempt + 1))
        return None

    def get_all_pages(self, path: str, key: str = "items") -> list[dict[str, Any]]:
        data = self.get(path)
        if data is None:
            return []
        if isinstance(data, list):
            return data
        return list(data.get(key) or [])


# --------------------------------------------------------------------------- harvest
def _clean(text: Any) -> str:
    return " ".join(str(text or "").split())


def node_criteria(metadata: dict[str, Any]) -> list[dict[str, str]]:
    """Extract ``[{cid, text}]`` using the platform's single-source parser."""
    if str(Path(__file__).resolve().parents[1] / "src") not in sys.path:
        pass
    from tensorphysics.campaign.wave4_gate import parse_acceptance_criteria

    return [{"cid": cid, "text": text} for cid, text in parse_acceptance_criteria(metadata)]


def harvest(
    client: Client,
    *,
    plan_ids: Iterable[int] | None,
    report_lookup: bool = True,
    workers: int = 8,
    cache_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Collect nodes + evidence concurrently; order of the result is canonical."""
    from concurrent.futures import ThreadPoolExecutor

    plans: dict[int, dict[str, Any]] = {}
    if plan_ids is None:
        for plan in client.get_all_pages("/tensor/campaigns/plans?limit=100"):
            plans[int(plan["id"])] = plan
        plan_ids = sorted(plans)
    plan_ids = [int(p) for p in plan_ids]

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    errors: list[int] = []

    def fetch_plan_nodes(pid: int) -> tuple[int, list[dict[str, Any]]]:
        if cache_dir is not None:
            hit = cache_dir / f"{pid}.json"
            if hit.exists():
                try:
                    return pid, json.loads(hit.read_text(encoding="utf-8"))
                except Exception:  # corrupted cache entry -> refetch
                    pass
        try:
            payload = client.get(f"/tensor/campaigns/plans/{pid}/nodes")
        except Exception:  # transient API failure -> retried on the next harvest run
            errors.append(int(pid))
            return pid, []
        items = [] if payload is None else (payload.get("items") if isinstance(payload, dict) else payload)
        items = list(items or [])
        if cache_dir is not None:
            hit = cache_dir / f"{pid}.json"
            tmp = hit.with_suffix(".tmp")
            tmp.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
            tmp.replace(hit)
        return pid, items

    nodes: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for pid, items in pool.map(fetch_plan_nodes, plan_ids):
            for raw in items:
                meta = raw.get("metadata") or {}
                nodes.append(
                    {
                        "node_id": raw.get("id"),
                        "plan_id": pid,
                        "plan_name": "",
                        "milestone_id": raw.get("milestone_id"),
                        "ext_id": str(meta.get("ext_id") or ""),
                        "name": _clean(raw.get("name")),
                        "repo": str(meta.get("repo") or ""),
                        "node_type": raw.get("node_type"),
                        "status": raw.get("status"),
                        "mission_id": raw.get("mission_id"),
                        "consecutive_fails": raw.get("consecutive_fails") or 0,
                        "reproduction_verdict": raw.get("reproduction_verdict"),
                        "criteria_declared": [_clean(c) for c in (meta.get("acceptance_criteria") or [])],
                        "goal": str(meta.get("goal") or ""),
                        "is_maturity": bool(meta.get("is_maturity")),
                    }
                )

    if errors:
        print(f"[harvest] {len(errors)} plans failed transiently: {errors[:10]}", file=sys.stderr)
    seen_plans = sorted({int(n["plan_id"]) for n in nodes})

    def fetch_plan_name(pid: int) -> tuple[int, str]:
        detail = client.get(f"/tensor/campaigns/plans/{pid}") or {}
        return pid, str(detail.get("name") or "") if isinstance(detail, dict) else ""

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for pid, name in pool.map(fetch_plan_name, [p for p in seen_plans if p not in plans]):
            plans[pid] = {"name": name}
    for node in nodes:
        node["plan_name"] = str((plans.get(int(node["plan_id"])) or {}).get("name") or "")

    if report_lookup:
        mids = sorted({int(n["mission_id"]) for n in nodes if n.get("mission_id") is not None})

        def fetch_evidence(mid: int) -> tuple[int, dict[str, Any]]:
            try:
                reports = client.get_all_pages(f"/tensor/reports?mission_id={mid}&limit=50")
                mission = client.get(f"/tensor/missions/{mid}") or {}
            except Exception:
                reports, mission = [], {}
            return mid, {
                "n_reports": len(reports),
                "report_ids": [r.get("id") for r in reports],
                "mission_status": mission.get("status") if isinstance(mission, dict) else None,
                "mission_result": (mission.get("result") or {}) if isinstance(mission, dict) else {},
            }

        evidence: dict[int, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for mid, ev in pool.map(fetch_evidence, mids):
                evidence[mid] = ev
        for node in nodes:
            mid = node.get("mission_id")
            node["evidence"] = (
                dict(evidence[int(mid)])
                if mid is not None
                else {"n_reports": 0, "report_ids": [], "mission_status": None, "mission_result": {}}
            )

    for node in nodes:
        node["criteria"] = node_criteria(
            {"acceptance_criteria": node.pop("criteria_declared"), "goal": node["goal"]}
        )
    dedup: dict[int, dict[str, Any]] = {}
    for node in nodes:
        dedup[int(node["node_id"])] = node
    nodes = sorted(dedup.values(), key=lambda n: int(n["node_id"]))
    return nodes, errors


def harvest_offline(cache_dir: Path, plan_names: dict[int, str] | None = None) -> list[dict[str, Any]]:
    """Rebuild the corpus from a per-plan node cache (no API, byte-for-byte repeatable)."""
    nodes: list[dict[str, Any]] = []
    for path in sorted(cache_dir.glob("*.json"), key=lambda p: int(p.stem)):
        pid = int(path.stem)
        try:
            items = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for raw in items:
            meta = raw.get("metadata") or {}
            nodes.append(
                {
                    "node_id": raw.get("id"),
                    "plan_id": pid,
                    "plan_name": str((plan_names or {}).get(pid, "")),
                    "milestone_id": raw.get("milestone_id"),
                    "ext_id": str(meta.get("ext_id") or ""),
                    "name": _clean(raw.get("name")),
                    "repo": str(meta.get("repo") or ""),
                    "node_type": raw.get("node_type"),
                    "status": raw.get("status"),
                    "mission_id": raw.get("mission_id"),
                    "consecutive_fails": raw.get("consecutive_fails") or 0,
                    "reproduction_verdict": raw.get("reproduction_verdict"),
                    "criteria_declared": meta.get("acceptance_criteria"),
                    "goal": str(meta.get("goal") or ""),
                    "is_maturity": bool(meta.get("is_maturity")),
                }
            )
    for node in nodes:
        node["criteria"] = node_criteria(
            {"acceptance_criteria": node.pop("criteria_declared"), "goal": node["goal"]}
        )
    dedup = {int(n["node_id"]): n for n in nodes}
    return sorted(dedup.values(), key=lambda n: int(n["node_id"]))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="artifacts/acceptance_judgeability/corpus.jsonl")
    ap.add_argument("--plan-ranges", default=None, help='e.g. "1-2306,9125-9400"')
    ap.add_argument("--plan-min", type=int, default=None)
    ap.add_argument("--plan-max", type=int, default=None)
    ap.add_argument("--no-report-lookup", action="store_true")
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cache-dir", default=None, help="per-plan node cache (resumable harvest)")
    ap.add_argument("--offline", action="store_true", help="rebuild from --cache-dir, no API calls")
    ap.add_argument("--login-only", action="store_true", help="smoke test: login and exit")
    args = ap.parse_args(argv)

    if args.offline:
        if not args.cache_dir:
            print("--offline needs --cache-dir", file=sys.stderr)
            return 2
        nodes = harvest_offline(Path(args.cache_dir))
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for node in nodes:
                fh.write(json.dumps(node, ensure_ascii=False, sort_keys=True) + "\n")
        manifest = {
            "corpus": str(out),
            "sha256": sha256_file(out),
            "n_nodes": len(nodes),
            "n_nodes_with_criteria": sum(1 for n in nodes if n["criteria"]),
            "n_criteria": sum(len(n["criteria"]) for n in nodes),
            "plans_scanned": sorted({int(n["plan_id"]) for n in nodes}),
            "mode": "offline",
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (out.parent / "corpus_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps({k: v for k, v in manifest.items() if k != "plans_scanned"}, ensure_ascii=False))
        return 0

    client = Client(
        args.base,
        os.environ.get("TENSOR_AGENT_USER", "eco_admin"),
        os.environ.get("TENSOR_AGENT_PASS", ""),
    )
    if args.login_only:
        print("login ok")
        return 0

    plan_ids = None
    if args.plan_ranges:
        plan_ids = []
        for chunk in args.plan_ranges.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if "-" in chunk:
                lo, hi = (int(x) for x in chunk.split("-", 1))
            else:
                lo = hi = int(chunk)
            plan_ids.extend(range(lo, hi + 1))
    elif args.plan_min is not None or args.plan_max is not None:
        lo = args.plan_min or 1
        hi = args.plan_max or lo
        plan_ids = list(range(lo, hi + 1))

    nodes, errors = harvest(
        client,
        plan_ids=plan_ids,
        report_lookup=not args.no_report_lookup,
        workers=args.workers,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for node in nodes:
            fh.write(json.dumps(node, ensure_ascii=False, sort_keys=True) + "\n")

    manifest = {
        "corpus": str(out),
        "sha256": sha256_file(out),
        "n_nodes": len(nodes),
        "n_nodes_with_criteria": sum(1 for n in nodes if n["criteria"]),
        "n_criteria": sum(len(n["criteria"]) for n in nodes),
        "plans_scanned": sorted({int(n["plan_id"]) for n in nodes}),
        "api_calls": client.calls,
        "transient_plan_failures": len(errors),
        "base": client.base,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out.parent / "corpus_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "plans_scanned"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())