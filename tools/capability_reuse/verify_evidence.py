"""Resolve review-worksheet evidence to raw context (report lines / source lines)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import psycopg  # noqa: E402

DSN = "postgresql://foundation:uGTK-unused@127.0.0.1:5432/tensorplatform"


def dsn() -> str:
    cfg = json.loads((ROOT / "src/tensorphysics/capability_reuse/configs/repos.json").read_text())
    import os
    return os.environ.get(cfg["db"]["dsn_env"], "") or cfg["db"]["default_dsn"]


def main() -> int:
    ws = json.loads((ROOT / "artifacts/capability_reuse/edge_sample_worksheet.json").read_text())
    conn = psycopg.connect(dsn())
    cur = conn.cursor()
    for i, row in enumerate(ws["rows"], 1):
        print(f"\n=== [{i}] {row['src'][:70]}\n        -> {row['dst'][:70]}")
        for ev in row["evidence"]:
            print("   EV:", ev)
            loc = ev.split(" [")[0]
            if loc.startswith("report:"):
                rid, line = loc.split(":")[1], int(loc.split(":")[2])
                cur.execute("select title, mission_id, markdown from tensor_reports where id=%s", (int(rid),))
                got = cur.fetchone()
                if not got:
                    print("    !! report missing")
                    continue
                title, mid, md = got
                lines = md.splitlines()
                print(f"    report title: {title[:80]} (mission {mid})")
                for k in range(max(0, line - 2), min(len(lines), line + 1)):
                    print(f"    {k+1:5d}| {lines[k][:200]}")
            else:
                repo, rest = loc.split("::", 1)
                rel, _ln = rest.rsplit(":", 1)
                cfg = json.loads((ROOT / "src/tensorphysics/capability_reuse/configs/repos.json").read_text())
                root = next(r["root"] for r in cfg["repos"] if r["name"] == repo)
                line = int(_ln)
                try:
                    lines = (Path(root) / rel).read_text(errors="replace").splitlines()
                except OSError:
                    print("    !! file missing")
                    continue
                for k in range(max(0, line - 2), min(len(lines), line + 1)):
                    print(f"    {k+1:5d}| {lines[k][:200]}")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())