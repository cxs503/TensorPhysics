#!/usr/bin/env python3
"""Build compact, human-readable digests of mission reports for gold-set annotation.

Reading full reports (5-20 KB each) would flood an annotator's context, so this tool prints a
bounded digest per report: title, summary head, the lines that carry contribution evidence
(误差 / PASS / 基线 / 解析解 / MOCK→REAL / 修复 / 不可行 / 反例) and the extracted evidence
field bundle.  The annotator (human or reviewing agent) assigns the label from the digest and
records the rationale; the classifier is *not* consulted for the label.

Usage
    python3 tools/contribution_meter/make_digests.py --corpus data/corpus_reports.jsonl \
        --stratify --n-c1 18 --n-c2 12 --n-c3 12 --n-c4 10 --n-c5 10 --n-c6 12 \
        --out artifacts/gold_digests.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evidence import extract, open_text, resolve_corpus  # noqa: E402
import classifier as cm  # noqa: E402

EVIDENCE_LINE = re.compile(
    r"误差|PASS|FAIL|基线|解析解|analytic|参考值|MOCK|STUB|REAL|修复|不可行|反例|对拍|"
    r"回归|判据|benchmark|golden|verified|混淆|保守|失效|no benefit",
    re.IGNORECASE,
)


def digest(report: dict, ev: dict, max_lines: int = 6) -> dict:
    md = report.get("markdown") or ""
    lines, seen = [], set()
    for raw in md.splitlines():
        line = raw.strip()
        if len(line) < 8 or not EVIDENCE_LINE.search(line):
            continue
        if line[:24] in seen:
            continue
        seen.add(line[:24])
        lines.append(line[:150])
        if len(lines) >= max_lines:
            break
    return {
        "mission_id": report["mission_id"],
        "report_id": report["report_id"],
        "title": (report.get("title") or "")[:110],
        "summary": (report.get("summary") or "")[:260],
        "evidence_lines": lines,
        "fields": cm.summarise_fields(ev),
        "n_error_rows": ev["n_error_rows"],
        "ref_kinds": ev["reference_kinds"],
        "n_units": ev["n_units"],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="data/corpus_reports.jsonl")
    ap.add_argument("--out", default="artifacts/gold_digests.json")
    ap.add_argument("--seed", type=int, default=20261107)
    ap.add_argument("--report-ids", default=None, help="comma list to digest instead of strategising")
    ap.add_argument("--n-c1", type=int, default=12)
    ap.add_argument("--n-c2", type=int, default=9)
    ap.add_argument("--n-c3", type=int, default=9)
    ap.add_argument("--n-c4", type=int, default=8)
    ap.add_argument("--n-c5", type=int, default=8)
    ap.add_argument("--n-c6", type=int, default=14)
    args = ap.parse_args(argv)

    rows = [json.loads(line) for line in open_text(resolve_corpus(args.corpus))]
    # one report per mission (cars: a mission may publish several near-duplicate reports; counting
    # all of them would inflate the gold set with the same underlying contribution)
    best_by_mission: dict[str, dict] = {}
    for r in rows:
        mid = str(r["mission_id"])
        cur = best_by_mission.get(mid)
        if cur is None or (r.get("markdown_len") or 0) > (cur.get("markdown_len") or 0):
            best_by_mission[mid] = r
    rows = list(best_by_mission.values())
    evs = {str(r["report_id"]): (r, extract(r)) for r in rows}
    rng = random.Random(args.seed)

    if args.report_ids:
        ids = [s.strip() for s in args.report_ids.split(",") if s.strip()]
        picked = [digest(evs[i][0], evs[i][1]) for i in ids if i in evs]
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(picked, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        for d in picked:
            print(f"### {d['mission_id']}/{d['report_id']} {d['title']}")
            print(f"  summary: {d['summary']}")
            for ln in d["evidence_lines"]:
                print(f"   | {ln}")
            print(f"  fields: {' '.join(d['fields'])}")
        return 0

    # preliminary rule pass, used only for stratification (not for labelling)
    prelim: dict[str, list[str]] = {c: [] for c in cm.CLASSES + ["NONE"]}
    for rid, (r, ev) in evs.items():
        res = cm.classify_evidence(ev)
        prelim[res["label"]].append(rid)

    budget = {"C1": args.n_c1, "C2": args.n_c2, "C3": args.n_c3,
              "C4": args.n_c4, "C5": args.n_c5, "C6": args.n_c6}
    chosen: list[str] = []
    for cls, n in budget.items():
        pool = sorted(prelim[cls])
        rng.shuffle(pool)
        # half from strongest-signal head, half random, so lookalikes and misses both appear
        head = sorted(pool, key=lambda rid: -evs[rid][1]["n_error_rows"])[: max(1, n // 2)]
        rest = [rid for rid in pool if rid not in head]
        chosen.extend((head + rest)[:n])

    # adversarial pool: error-table-bearing reports that look quantitative but are infrastructure
    adv = [
        rid for rid, (r, ev) in evs.items()
        if ev["n_error_rows"] >= 1 and ev["n_units"] < 30 and rid not in chosen
    ]
    rng.shuffle(adv)
    chosen.extend(adv[:8])
    chosen = sorted(set(chosen))

    picked = [digest(evs[rid][0], evs[rid][1]) for rid in chosen]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(picked, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"wrote {len(picked)} digests -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())