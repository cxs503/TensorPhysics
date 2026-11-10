"""Build the frozen FAIL corpus used by the P186-W1-N08 back-test.

Protocol
--------
1. ``harvest_fail_corpus.harvest`` retrieves every sentence-like segment of this
   repository's own reports that carries an epistemic cue (limit / absence /
   non-convergence / defect).  This is **retrieval**, not annotation.
2. ``SELECTION`` below is the *analyst adjudication* (``agent:m9338``): for each
   class it names the segments that were read and judged to be a genuine
   failure statement of that class, by a unique substring key.  Segments that
   read as PASS statements, as schema documentation, or that negate the failure
   were rejected at this step; nothing was rephrased (the stored body is the
   verbatim harvested text).
3. The payload digest and per-class counts are pinned in
   ``frozen_manifest.json``; later edits raise in
   ``dataset.load_frozen_set``.

Re-run::

    PYTHONPATH=src python benchmarks/failure_taxonomy/build_frozen_set.py
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks" / "failure_taxonomy"))

from harvest_fail_corpus import CUES, harvest  # noqa: E402
from tensorphysics.failure_taxonomy.dataset import DATA_DIR, digest_of  # noqa: E402

# (class, [key, optional second key]) -- keys are unique substrings of real
# report segments; a two-key entry joins two adjacent sentences of the same file.
SELECTION: list[tuple[str, list[str]]] = [
    ("BASELINE_ABSENT", ["所有求解器适配器（DEM/LBM/FEM/FVM）均为烟道测试级别"]),
    ("BASELINE_ABSENT", ["组合极限在渐近域外失效"]),
    ("BASELINE_ABSENT", ["Pasternak 地基下冰板首裂载荷本身无实验/文献对照值"]),
    ("BASELINE_ABSENT", ["无数字化基准"]),
    ("BASELINE_ABSENT", ["该文非弯曲实验，原文 403 不可取"]),
    ("BASELINE_ABSENT", ["无可靠公开值随本节点入库"]),
    ("BASELINE_ABSENT", ["无冰+浪组合疲劳实验/实船实测数据"]),
    ("BASELINE_ABSENT", ["σ_qs 非高斯"]),
    ("TOLERANCE_EXCEEDED", ["耦合弯矩的绝对量级不可信"]),
    ("TOLERANCE_EXCEEDED", ["被继承的欠账误差"]),
    ("TOLERANCE_EXCEEDED", ["分离频带（ratio = 10）"]),
    ("TOLERANCE_EXCEEDED", ["加和律前提"]),
    ("TOLERANCE_EXCEEDED", ["线性叠加假设被否证"]),
    ("TOLERANCE_EXCEEDED", ["板格大挠度（塑性区）"]),
    ("TOLERANCE_EXCEEDED", ["383 %"]),
    ("TOLERANCE_EXCEEDED", ["99.41 pp"]),
    ("CONVERGENCE_STALL", ["500 次迭代不收敛"]),
    ("CONVERGENCE_STALL", ["任何松弛因子都不收敛"]),
    ("CONVERGENCE_STALL", ["蒙特卡洛窗口只有 ~10 次冲击无法收敛"]),
    ("CONVERGENCE_STALL", ["蒙特卡洛无法收敛"]),
    ("CONVERGENCE_STALL", ["m4 对数发散"]),
    ("CONVERGENCE_STALL", ["the explicit scheme diverges for every time step"]),
    ("CONVERGENCE_STALL", ["0.75 … 0.89"]),
    ("CONVERGENCE_STALL", ["three-grid residual decay"]),
    ("MODEL_FIDELITY_LIMIT", ["h_ridge 在刚性冰脊假设下"]),
    ("MODEL_FIDELITY_LIMIT", ["全局冰压量级为参数级假设"]),
    ("MODEL_FIDELITY_LIMIT", ["船体梁理想化"]),
    ("MODEL_FIDELITY_LIMIT", ["能量法 ram 容量忽略推力限制"]),
    ("MODEL_FIDELITY_LIMIT", ["无界外推"]),
    ("MODEL_FIDELITY_LIMIT", ["有理式闭包在角点"]),
    ("MODEL_FIDELITY_LIMIT", ["wide_bimodal_rp3"]),
    ("MODEL_FIDELITY_LIMIT", ["各向同性假设"]),
    ("DATA_INSUFFICIENCY", ["缺任一交接面产物判 WITHHELD"]),
    ("DATA_INSUFFICIENCY", ["没有可采纳的 UQ 记录"]),
    ("DATA_INSUFFICIENCY", ["(a) 证据缺失"]),
    ("DATA_INSUFFICIENCY", ["8 WITHHELD"]),
    ("DATA_INSUFFICIENCY", ["未取得复跑值"]),
    ("DATA_INSUFFICIENCY", ["无数字证据（判据 WITHHELD）"]),
    ("DATA_INSUFFICIENCY", ["5 个节点在飞"]),
    ("DATA_INSUFFICIENCY", ["文献腿仍为 WITHHELD"]),
    ("IMPLEMENTATION_DEFECT", ["谱半径估计 bug"]),
    ("IMPLEMENTATION_DEFECT", ["两套圆盘实现现吻合"]),
    ("IMPLEMENTATION_DEFECT", ["使真实接口缺陷无法藏在验收指标的离散噪声里"]),
    ("IMPLEMENTATION_DEFECT", ["这个自检本身是抓 bug 的关键工具"]),
    ("IMPLEMENTATION_DEFECT", ["早期实现忽略了 `x_toe`"]),
    ("IMPLEMENTATION_DEFECT", ["初版未使用 `x_toe`"]),
    ("IMPLEMENTATION_DEFECT", ["幂律斜率锚的解析式一开始写错"]),
    ("IMPLEMENTATION_DEFECT", ["2 contract checks run (0 PASS / 2 FAIL)"]),
]

MIN_LEN, MAX_LEN = 30, 320
DECOY_TARGET = 120


def _resolve(rows: list[dict], key: str) -> dict:
    hits = [r for r in rows if key in r["body"]]
    if not hits:
        raise SystemExit(f"selection key not found in harvest: {key!r}")
    hits.sort(key=lambda r: (r["source_path"], r["source_line"]))
    return hits[0]


def _join(a: dict, b: dict) -> dict:
    body = f"{a['body']} {b['body']}".strip()
    return {
        "body": body,
        "source_path": a["source_path"],
        "source_line": a["source_line"],
        "source_sha256": a["source_sha256"],
        "segment_sha256": a["segment_digest"] + "+" + b["segment_digest"],
    }


def build_records(rows: list[dict]) -> list[dict]:
    records: list[dict] = []
    seen_bodies: set[str] = set()
    for code, keys in SELECTION:
        first = _resolve(rows, keys[0])
        row = first if len(keys) == 1 else _join(first, _resolve(rows, keys[1]))
        body = row["body"]
        if not (MIN_LEN <= len(body) <= MAX_LEN):
            raise SystemExit(f"segment length {len(body)} out of window for {keys[0]!r}")
        if body in seen_bodies:
            raise SystemExit(f"duplicate segment selected: {keys[0]!r}")
        seen_bodies.add(body)
        records.append(
            {
                "record_id": "",
                "body": body,
                "gold_class": code,
                "annotator": "agent:m9338",
                "provenance": {
                    "source_path": row["source_path"],
                    "source_line": row["source_line"],
                    "source_sha256": row["source_sha256"],
                    "segment_sha256": row.get("segment_sha256") or row["segment_digest"],
                },
            }
        )
    records.sort(key=lambda r: (r["gold_class"], r["provenance"]["source_path"], r["provenance"]["source_line"]))
    for i, rec in enumerate(records, start=1):
        rec["record_id"] = f"F{i:03d}"
    return records


def build_decoys(rows: list[dict], chosen: set[str], target: int = DECOY_TARGET) -> list[dict]:
    """Non-cue segments of comparable length: the negative-control corpus."""
    rows = harvest(REPO, require_cue=False)
    pool = [
        r
        for r in rows
        if r["body"] not in chosen
        and 60 <= len(r["body"]) <= 300
        and not any(cue in r["body"].lower() for cue in CUES)
    ]
    pool.sort(key=lambda r: (r["source_path"], r["source_line"]))
    rng = random.Random(20261110)
    sample = pool if len(pool) <= target else rng.sample(pool, target)
    sample.sort(key=lambda r: (r["source_path"], r["source_line"]))
    return [
        {
            "body": r["body"],
            "source_path": r["source_path"],
            "source_line": r["source_line"],
            "segment_sha256": r["segment_digest"],
        }
        for r in sample
    ]


def main() -> int:
    rows = harvest(REPO)
    records = build_records(rows)
    decoys = build_decoys(rows, {r["body"] for r in records})

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with (DATA_DIR / "frozen_fail_set.jsonl").open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
    with (DATA_DIR / "decoy_pool.jsonl").open("w", encoding="utf-8") as fh:
        for dec in decoys:
            fh.write(json.dumps(dec, ensure_ascii=False, sort_keys=True) + "\n")

    counts: dict[str, int] = {}
    for rec in records:
        counts[rec["gold_class"]] = counts.get(rec["gold_class"], 0) + 1
    manifest = {
        "n_records": len(records),
        "sha256": digest_of(records),
        "per_class": counts,
        "n_decoys": len(decoys),
        "annotator": "agent:m9338",
        "protocol": (
            "epistemic-cue retrieval over in-repo reports, then analyst adjudication "
            "by unique substring key; verbatim segments, digest pinned"
        ),
        "source_scope": ["docs/**/*.md", "artifacts/**/*.json", "benchmarks/artifacts/**/*.json"],
    }
    (DATA_DIR / "frozen_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for rec in records:
        print(f"{rec['record_id']} {rec['gold_class']:<22} {rec['body'][:100]}")
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())