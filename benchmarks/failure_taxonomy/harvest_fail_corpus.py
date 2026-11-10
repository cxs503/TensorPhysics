"""Harvest candidate honest-FAIL prose segments from in-repo evidence.

The P186-W1-N08 failure taxonomy is back-tested on a *frozen* set of real
limitation / next-step passages that already exist in this repository's own
reports (``docs/**/*.md`` and ``artifacts/**/*.json``).  Nothing is invented
here: this script only

1. walks the evidence tree deterministically (sorted paths),
2. splits each file into sentence-like segments,
3. keeps segments that carry a limitation / failed-criterion cue,
4. records provenance for every kept segment (path, line, file sha256,
   segment digest).

The output ``/tmp/fail_candidates.jsonl`` is a *candidate* pool only.  Class
assignment is a separate manual annotation step (see
``frozen_set.py``); the harvester never labels anything.

Usage::

    PYTHONPATH=src python benchmarks/failure_taxonomy/harvest_fail_corpus.py \
        --out /tmp/fail_candidates.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Cue lexemes are *epistemic* (limit / absence / non-convergence), deliberately
# free of any physics or ship-domain noun so that the harvester itself stays
# domain-neutral as required by criterion C5.
CUES = (
    "未完成", "未验证", "未覆盖", "未能", "未收敛", "未标定", "未校准",
    "局限", "限制", "缺口", "不足", "缺少", "缺乏", "无法", "不适用",
    "超出", "外推", "假设", "残差", "收敛", "误差", "偏差", "退化为",
    "fail", "limitation", "limitations", "not validated", "no reference",
    "unresolved", "stalled", "diverg", "tolerance", "workaround",
)

SENT_SPLIT = re.compile(r"(?<=[。；;!?！？])|\n+|(?<=\.)\s+(?=[A-Z])")
MIN_LEN, MAX_LEN = 40, 320


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def iter_files(root: Path):
    for pattern in ("docs/**/*.md", "artifacts/**/*.json", "benchmarks/artifacts/**/*.json"):
        for path in sorted(root.glob(pattern)):
            if path.is_file():
                yield path


def segments(text: str):
    """Yield ``(line_no, segment)`` for sentence-like segments of *text*."""
    offset = 0
    for raw in SENT_SPLIT.split(text):
        if raw is None:
            continue
        seg = " ".join(raw.split())
        line_no = text[: offset].count("\n") + 1
        offset += len(raw or "")
        if not seg:
            continue
        yield line_no, seg


def harvest(root: Path, require_cue: bool = True) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for path in iter_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        file_digest = sha256_text(text)
        rel = str(path.relative_to(root))
        for line_no, seg in segments(text):
            low = seg.lower()
            if not (MIN_LEN <= len(seg) <= MAX_LEN):
                continue
            if require_cue and not any(cue in low for cue in CUES):
                continue
            digest = sha256_text(seg)
            if digest in seen:
                continue
            seen.add(digest)
            out.append(
                {
                    "segment_digest": digest,
                    "source_path": rel,
                    "source_line": line_no,
                    "source_sha256": file_digest,
                    "body": seg,
                }
            )
    out.sort(key=lambda r: (r["source_path"], r["source_line"]))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(REPO))
    ap.add_argument("--out", default="/tmp/fail_candidates.jsonl")
    args = ap.parse_args(argv)
    rows = harvest(Path(args.root))
    with open(args.out, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"candidates": len(rows), "out": args.out}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())