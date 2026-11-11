#!/usr/bin/env python3
"""Census runner: corpus -> class distribution, KS table, hollow-PASS ledger, figures.

Offline and deterministic: it reads only the frozen corpus JSONL
(``artifacts/acceptance_judgeability/corpus.jsonl``), the 100-clause gold labels and
the two control sets, and writes every number the report quotes. Nothing here
touches the network, so the census can be re-run and its digests compared.

Outputs (all under ``artifacts/acceptance_judgeability``)::

    census.json                  aggregate + per-node scorecard + hollow ledger
    census_sensitivity.json      ks_min sweep / weight variant / strict policy
    judge_criteria.json          this node's own machine acceptance criteria, measured
    gold_kappa.json              machine-vs-human agreement on the 100-clause sample
    controls.json                negative control (20 forged PASS) + false-positive test
    figs/*.png                   four figures

Usage::

    PYTHONPATH=src python scripts/acceptance_judgeability_census.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.acceptance_judgeability import (  # noqa: E402
    DEFAULT_POLICY,
    IMPLICIT_POLICY,
    STRICT_POLICY,
    TOLERANCE_WEIGHTS,
    UNIFORM_WEIGHTS,
    adjudicate,
    classify,
    classify_all,
    cohen_kappa,
    corpus_digest,
    judgeability_support,
    load_corpus,
    run_census,
)
from tensorphysics.acceptance_judgeability.agreement import per_class_agreement  # noqa: E402
from tensorphysics.acceptance_judgeability.support import Evidence, gate_flag  # noqa: E402

ART = ROOT / "artifacts" / "acceptance_judgeability"
LABELS = ("JUDGEABLE", "THRESHOLD_ONLY", "RECORD_ONLY", "WITHHELD_OK")


# --------------------------------------------------------------------------- controls
def negative_control() -> dict[str, Any]:
    """20 clauses that only ask for a record/declaration, forged into a PASS.

    Each is wrapped in a node row marked ``done`` (i.e. the campaign says PASS). A
    correct classifier must call **all 20** hollow: a PASS backed only by prose
    cannot be supported.
    """
    texts = [
        "记录峰值压力曲线",
        "报告中给出时程图",
        "运行不崩溃且无异常日志",
        "补充适用域声明",
        "列出全部假设与局限",
        "输出结果文件并归档",
        "在报告中说明参数取值",
        "给出与上一版结果的对比说明",
        "结果趋势单调",
        "给出网格无关性讨论",
        "存档求解器日志与版本号",
        "递交截图证明端点可调用",
        "说明各符号的物理含义",
        "列出未纳入本验证的效应",
        "运行端到端流程并保留日志",
        "在报告附录给出原始数据清单",
        "曲线形状符合预期",
        "补充术语表",
        "记录算例清单与耗时",
        "说明与上游模块的接口",
    ]
    rows = []
    detected = 0
    for idx, text in enumerate(texts, start=1):
        node = {
            "node_id": 900000 + idx,
            "plan_id": 0,
            "ext_id": f"NC-{idx:02d}",
            "repo": "synthetic",
            "status": "done",
            "consecutive_fails": 0,
            "criteria": [{"cid": "C1", "text": text}],
            "evidence": {"n_reports": 1},
        }
        adj = adjudicate(node["criteria"], Evidence.from_node(node))
        hollow = not adj.supports_pass
        detected += int(hollow)
        rows.append(
            {
                "node_id": node["node_id"],
                "text": text,
                "machine_class": classify(text).judgeability.value,
                "measured": Evidence.from_node(node).measured,
                "forged_verdict": "PASS",
                "adjudication": adj.as_dict(),
                "flagged_hollow": hollow,
            }
        )
    return {
        "requirement": "20 pure RECORD_ONLY clauses wrapped in a forged PASS",
        "n": len(rows),
        "flagged": detected,
        "rate": detected / len(rows),
        "passed": detected == len(rows),
        "rows": rows,
    }


def false_positive_control(gold: dict[str, Any]) -> dict[str, Any]:
    """20 clauses a human confirmed carry a reference+tolerance must not be RECORD_ONLY."""
    judgeable = [i for i in gold["items"] if i["human"] == "JUDGEABLE"]
    picked = judgeable[:20]
    rows = []
    misses = 0
    for item in picked:
        cls = classify(item["text"], cid=item["cid"])
        wrong = cls.judgeability.value == "RECORD_ONLY" or cls.judgeability.value == "WITHHELD_OK"
        misses += int(wrong)
        rows.append(
            {
                "text": item["text"],
                "machine_class": cls.judgeability.value,
                "subtype": cls.subtype.value,
                "false_record_only": wrong,
            }
        )
    return {
        "requirement": "20 human-confirmed measurable clauses must not be called RECORD_ONLY",
        "n": len(rows),
        "misreported": misses,
        "tolerance": 1,
        "passed": len(rows) == 20 and misses <= 1,
        "rows": rows,
    }


# --------------------------------------------------------------------------- kappa
def gold_kappa_report(gold: dict[str, Any]) -> dict[str, Any]:
    items = gold["items"]
    human = [i["human"] for i in items]
    machine = [classify(i["text"], cid=i["cid"]).judgeability.value for i in items]
    hold = [i for i, it in enumerate(items) if it.get("split") == "holdout"]
    dev = [i for i, it in enumerate(items) if it.get("split") == "dev"]
    mismatches = [
        {"text": items[i]["text"], "human": human[i], "machine": machine[i], "split": items[i].get("split")}
        for i in range(len(items))
        if human[i] != machine[i]
    ]
    return {
        "requirement": "kappa >= 0.80 on 100 stratified human-labelled clauses",
        "n": len(items),
        "kappa_all": cohen_kappa(human, machine, LABELS),
        "kappa_holdout": cohen_kappa([human[i] for i in hold], [machine[i] for i in hold], LABELS),
        "kappa_dev": cohen_kappa([human[i] for i in dev], [machine[i] for i in dev], LABELS),
        "agreement_all": sum(1 for h, m in zip(human, machine) if h == m) / len(items),
        "per_class": per_class_agreement(human, machine),
        "mismatches": mismatches,
        "passed": cohen_kappa(human, machine, LABELS) >= 0.80,
        "provenance": "artifacts/acceptance_judgeability/gold_labels_100.json",
    }


# --------------------------------------------------------------------------- figures
def figures(census, sensitivity: dict[str, Any], out_dir: Path) -> list[dict[str, str]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    made: list[dict[str, str]] = []
    d = census.as_dict(include_rows=False)

    # 1 class distribution
    labels = list(d["classes"]["counts"].keys())
    counts = [d["classes"]["counts"][k] for k in labels]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    bars = ax.bar(labels, counts, color=["#2b7bba", "#f0a202", "#b0b0b0", "#6a4c93"])
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{c}\n{c / d['corpus']['n_criteria']:.1%}",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("clauses")
    ax.set_title(f"Acceptance-clause judgeability classes (n={d['corpus']['n_criteria']})")
    ax.set_ylim(0, max(counts) * 1.2)
    path = out_dir / "fig1_class_distribution.png"
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    made.append({"name": path.name, "caption": "Judgeability class distribution over all harvested acceptance clauses."})

    # 2 KS histogram with threshold
    hist = [h for h in d["ks"]["histogram"] if h.get("lo") is not None]
    undefined = d["ks"]["histogram"][-1]["count"]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    xs = [f"{h['lo']:.1f}-{h['hi']:.1f}" for h in hist]
    ax.bar(xs, [h["count"] for h in hist], color="#2b7bba")
    ax.axvline(census.policy.ks_min * len(hist) - 0.5, color="crimson", ls="--",
               label=f"ks_min={census.policy.ks_min}")
    ax.set_xlabel("KS (judgeability support)")
    ax.set_ylabel("nodes")
    ax.set_title(f"Per-node support (median={d['ks']['median']}, undefined={undefined})")
    ax.tick_params(axis="x", rotation=60, labelsize=7)
    ax.legend()
    path = out_dir / "fig2_ks_histogram.png"
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    made.append({"name": path.name, "caption": "KS histogram over campaign nodes with the PASS-support threshold."})

    # 3 threshold sweep
    sweep = sensitivity["ks_min_sweep"]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot([s["ks_min"] for s in sweep], [s["hollow_ratio"] for s in sweep], "o-", color="#b02a37")
    ax.set_xlabel("ks_min")
    ax.set_ylabel("hollow-PASS ratio among done nodes")
    ax.set_title("Sensitivity of the hollow-PASS population to the threshold")
    ax.grid(alpha=0.3)
    path = out_dir / "fig3_threshold_sweep.png"
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    made.append({"name": path.name, "caption": "Hollow-PASS ratio vs support threshold (no single threshold is privileged)."})

    # 4 per-repo hollow share
    repos = d["pass_dependency"]["by_repo"]
    names = list(repos)[:10]
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.barh(names[::-1], [repos[n]["hollow_ratio"] or 0.0 for n in names[::-1]], color="#6a4c93")
    ax.set_xlabel("hollow-PASS ratio")
    ax.set_title("Hollow-PASS ratio per repository (top 10 by node count)")
    ax.grid(axis="x", alpha=0.3)
    path = out_dir / "fig4_repo_hollow.png"
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    made.append({"name": path.name, "caption": "Hollow-PASS ratio by repository: the defect is systemic, not local."})
    return made


# --------------------------------------------------------------------------- judge criteria
def judge_criteria(census, controls: dict[str, Any], kappa: dict[str, Any], sensitivity: dict[str, Any]) -> dict[str, Any]:
    """This node's own machine acceptance criteria, each with its measured value."""
    d = census.as_dict(include_rows=False)
    return {
        "node": "P106W3A2-N08",
        "criteria": [
            {
                "cid": "A1",
                "text": "金标κ：分层抽样100条判据人工标注，机器κ≥0.80",
                "measured": round(kappa["kappa_all"], 4),
                "threshold": 0.80,
                "passed": kappa["kappa_all"] >= 0.80,
                "artifact": "artifacts/acceptance_judgeability/gold_kappa.json",
            },
            {
                "cid": "A2",
                "text": "负控：注入20条纯RECORD_ONLY判据并伪造PASS，必须100%判空心PASS",
                "measured": controls["negative_control"]["flagged"],
                "threshold": 20,
                "passed": controls["negative_control"]["passed"],
                "artifact": "artifacts/acceptance_judgeability/controls.json",
            },
            {
                "cid": "A3",
                "text": "假阳性：20条确认已实测判据不得被判RECORD_ONLY(误报≤1)",
                "measured": controls["false_positive"]["misreported"],
                "threshold": 1,
                "passed": controls["false_positive"]["passed"],
                "artifact": "artifacts/acceptance_judgeability/controls.json",
            },
            {
                "cid": "A4",
                "text": "覆盖率：节点判据抽取率≥85%",
                "measured": round(d["corpus"]["extraction_rate"], 4),
                "threshold": 0.85,
                "passed": d["corpus"]["extraction_rate"] >= 0.85,
                "artifact": "artifacts/acceptance_judgeability/census.json",
            },
            {
                "cid": "A5",
                "text": "可复算：两次运行KS表sha256一致",
                "measured": d["ks"]["table_sha256"],
                "threshold": "identical",
                "passed": sensitivity["recompute"]["identical"],
                "artifact": "artifacts/acceptance_judgeability/census_sensitivity.json",
            },
        ],
        "census_headline": {
            "n_nodes": d["corpus"]["n_nodes"],
            "n_criteria": d["corpus"]["n_criteria"],
            "judgeable_share": d["ks"]["judgeable_clause_share"],
            "ks_median": d["ks"]["median"],
            "hollow_pass_ratio": d["pass_dependency"]["hollow_ratio"],
            "hollow_pass_nodes": d["pass_dependency"]["hollow_pass_nodes"],
            "done_nodes": d["pass_dependency"]["done_nodes"],
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", default=str(ART / "corpus.jsonl"))
    ap.add_argument("--gold", default=str(ART / "gold_labels_100.json"))
    ap.add_argument("--out-dir", default=str(ART))
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    corpus_path = Path(args.corpus)
    nodes = load_corpus([corpus_path])
    digest = corpus_digest(corpus_path)

    def run(**kw):
        return run_census(nodes, corpus_paths=[str(corpus_path)], corpus_sha256=digest, **kw)

    census = run()
    census_again = run()
    recompute = {
        "first": census.ks_table_sha256,
        "second": census_again.ks_table_sha256,
        "identical": census.ks_table_sha256 == census_again.ks_table_sha256,
        "corpus_sha256": digest,
    }

    sweep = []
    for ks_min in (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.75):
        pol = type(DEFAULT_POLICY)(ks_min=ks_min)
        res = run(policy=pol)
        sweep.append(
            {
                "ks_min": ks_min,
                "hollow_pass_nodes": len(res.hollow),
                "hollow_ratio": round(res.hollow_ratio or 0.0, 4),
                "done_nodes": len(res.done_rows),
            }
        )

    strict = run(policy=STRICT_POLICY)
    implicit = run(policy=IMPLICIT_POLICY)
    weighted = run(weights=TOLERANCE_WEIGHTS)
    sensitivity = {
        "ks_min_sweep": sweep,
        "recompute": recompute,
        "policy_strict_no_self_baseline": {
            "ks_min": strict.policy.ks_min,
            "hollow_ratio": strict.hollow_ratio,
            "hollow_nodes": len(strict.hollow),
            "ks_median": strict.as_dict(include_rows=False)["ks"]["median"],
        },
        "policy_implicit_pair_optimistic": {
            "ks_min": implicit.policy.ks_min,
            "hollow_ratio": implicit.hollow_ratio,
            "hollow_nodes": len(implicit.hollow),
            "ks_median": implicit.as_dict(include_rows=False)["ks"]["median"],
            "note": "upper bound: unnamed error/deviation thresholds counted as judgeable",
        },
        "weights_tolerance_weighted": {
            "hollow_ratio": weighted.hollow_ratio,
            "hollow_nodes": len(weighted.hollow),
            "table_sha256": weighted.ks_table_sha256,
        },
    }

    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    kappa = gold_kappa_report(gold)
    controls = {"negative_control": negative_control(), "false_positive": false_positive_control(gold)}
    criteria = judge_criteria(census, controls, kappa, sensitivity)

    (out / "census.json").write_text(
        json.dumps(
            {
                "generated_by": "scripts/acceptance_judgeability_census.py",
                "census": census.as_dict(include_rows=True),
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    (out / "census_sensitivity.json").write_text(json.dumps(sensitivity, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "gold_kappa.json").write_text(json.dumps(kappa, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "controls.json").write_text(json.dumps(controls, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "judge_criteria.json").write_text(json.dumps(criteria, ensure_ascii=False, indent=1), encoding="utf-8")
    figs = [] if args.no_figures else figures(census, sensitivity, out / "figs")

    summary = {
        "corpus_sha256": digest,
        "n_nodes": census.n_nodes,
        "n_criteria": census.n_criteria,
        "extraction_rate": round(census.n_nodes_with_criteria / census.n_nodes, 4),
        "class_counts": dict(census.class_counts),
        "judgeable_share": round(census.judgeable_share or 0.0, 4),
        "ks_median": census.as_dict(include_rows=False)["ks"]["median"],
        "hollow_pass_nodes": len(census.hollow),
        "hollow_ratio": round(census.hollow_ratio or 0.0, 4),
        "ks_table_sha256": census.ks_table_sha256,
        "kappa_all": round(kappa["kappa_all"], 4),
        "kappa_holdout": round(kappa["kappa_holdout"], 4),
        "negative_control_rate": controls["negative_control"]["rate"],
        "false_positive_misses": controls["false_positive"]["misreported"],
        "recompute_identical": recompute["identical"],
        "all_criteria_passed": all(c["passed"] for c in criteria["criteria"]),
        "figures": [f["name"] for f in figs],
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())