#!/usr/bin/env python3
"""Harvest a bounded local corpus of real mission reports from the TensorPlatform API.

The contribution-meter needs *real* mission reports (with mission_id) as ground truth
candidates.  The platform API caps any single page at 100 rows and ignores ``offset``
for ``/tensor/reports``, so we enumerate missions (``/tensor/missions`` paginates
correctly) and then pull ``/tensor/reports?mission_id=<id>`` for each mission inside
the requested observation window.

Output: one JSON object per line (jsonl), fields:
    mission_id, report_id, title, summary, scenario_key, tags, mission_status,
    mission_created_at, mission_finished_at, report_created_at,
    markdown_len, markdown

Usage
    python3 tools/contribution_meter/harvest_corpus.py --hours 96 \
        --out data/corpus_reports.jsonl
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time
import urllib.error
import urllib.request

API_BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")


def _login(base: str) -> str:
    user = os.environ["TENSOR_AGENT_USER"]
    pwd = os.environ["TENSOR_AGENT_PASS"]
    body = json.dumps({"username": user, "password": pwd}).encode()
    req = urllib.request.Request(
        base + "/login", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read())
    return payload["access_token"]


class Client:
    def __init__(self, base: str = API_BASE, token: str | None = None) -> None:
        self.base = base
        self.token = token or _login(base)

    def get(self, path: str, retries: int = 3) -> dict:
        url = self.base + path
        for attempt in range(retries):
            req = urllib.request.Request(
                url, headers={"Authorization": "Bearer " + self.token}
            )
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as exc:  # pragma: no cover - network path
                if exc.code in (429, 500, 502, 503) and attempt + 1 < retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise
        raise RuntimeError("unreachable")  # pragma: no cover


def _parse_ts(text: str | None) -> _dt.datetime | None:
    if not text:
        return None
    try:
        return _dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:  # pragma: no cover
        return None


def harvest(client: Client, hours: float, max_missions: int = 600) -> list[dict]:
    now = _dt.datetime.now(_dt.timezone.utc)
    cutoff = now - _dt.timedelta(hours=hours)

    missions: list[dict] = []
    offset = 0
    while offset < max_missions:
        page = client.get(f"/tensor/missions?limit=200&offset={offset}")
        items = page.get("items") or []
        if not items:
            break
        missions.extend(items)
        offset += len(items)
        if len(items) < 200:
            break

    rows: list[dict] = []
    for m in missions:
        ts = _parse_ts(m.get("finished_at") or m.get("created_at"))
        if ts is not None and ts < cutoff:
            continue
        try:
            reports = client.get(f"/tensor/reports?mission_id={m['id']}&limit=100")
        except urllib.error.HTTPError:
            continue
        for r in reports.get("items") or []:
            detail = r
            if "markdown" not in r or not r.get("markdown"):
                try:
                    detail = client.get(f"/tensor/reports/{r['id']}")
                except urllib.error.HTTPError:  # pragma: no cover
                    detail = r
            md = detail.get("markdown") or ""
            rows.append(
                {
                    "mission_id": m["id"],
                    "report_id": str(r.get("id")),
                    "title": detail.get("title") or "",
                    "summary": detail.get("summary") or "",
                    "scenario_key": detail.get("scenario_key") or "",
                    "tags": detail.get("tags") or "",
                    "mission_status": m.get("status"),
                    "mission_created_at": m.get("created_at"),
                    "mission_finished_at": m.get("finished_at"),
                    "report_created_at": detail.get("created_at"),
                    "markdown_len": len(md),
                    "markdown": md,
                }
            )
    return rows


def dump_missions(client: Client, hours: float, out: str, max_missions: int = 600) -> int:
    """Dump mission metadata for every mission inside the window (needed for the
    zero-contribution denominator: missions that never produced a report)."""
    now = _dt.datetime.now(_dt.timezone.utc)
    cutoff = now - _dt.timedelta(hours=hours)
    kept: list[dict] = []
    offset = 0
    while offset < max_missions:
        page = client.get(f"/tensor/missions?limit=200&offset={offset}")
        items = page.get("items") or []
        if not items:
            break
        for m in items:
            ts = _parse_ts(m.get("finished_at") or m.get("created_at"))
            if ts is not None and ts < cutoff:
                continue
            kept.append(
                {
                    "id": m["id"],
                    "status": m.get("status"),
                    "created_at": m.get("created_at"),
                    "finished_at": m.get("finished_at"),
                    "budget": m.get("budget"),
                    "repo": m.get("repo"),
                    "goal_head": (m.get("goal") or "")[:160],
                }
            )
        offset += len(items)
        if len(items) < 200:
            break
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for row in kept:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"dumped {len(kept)} missions (window={hours}h) -> {out}", file=sys.stderr)
    return len(kept)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hours", type=float, default=96.0)
    ap.add_argument("--out", default="data/corpus_reports.jsonl")
    ap.add_argument("--mission-limit", type=int, default=600)
    ap.add_argument("--missions-out", default=None,
                    help="also dump mission metadata (id/status/created_at/budget) as jsonl")
    args = ap.parse_args(argv)

    client = Client()
    if args.missions_out:
        dump_missions(client, args.hours, args.missions_out, args.mission_limit)
    rows = harvest(client, args.hours, args.mission_limit)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    missions = {r["mission_id"] for r in rows}
    print(
        f"harvested {len(rows)} reports from {len(missions)} missions "
        f"(window={args.hours}h) -> {args.out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())