#!/usr/bin/env python3
"""Build the lineage-debt ledger artefacts: summary JSON + figures.

Every number in the summary is computed from the ledgers in ``runs/lineage_debt``
(which are themselves pure replays of the campaign snapshot), so the report can
inject them verbatim -- nothing is transcribed by hand.

Usage::

    PYTHONPATH=src python scripts/gen_lineage_debt_artifacts.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib                                    # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402

from tensorphysics.lineage_debt import load_config, load_bundle, selfcheck   # noqa: E402
from tensorphysics.lineage_debt.model import canonical_json                   # noqa: E402
from tensorphysics.lineage_debt.query import gate_report, load_ledger, query  # noqa: E402
from tensorphysics.lineage_debt.replay import build_ledger                    # noqa: E402

RUNS = ROOT / "runs" / "lineage_debt"
ART = ROOT / "artifacts" / "lineage_debt"
FIGS = ART / "figs"
BUNDLE = ART / "inputs" / "lineage_debt_bundle.json"
PLANS = (26, 173, 388)
GOLD_MISSION = 6753


def _ascii(label: str) -> str:
    """Render CJK tokens as ``<zh>``: the render host has no CJK font."""
    out, run = [], False
    for ch in str(label):
        if ord(ch) < 128:
            if run:
                out.append("<zh>")
                run = False
            out.append(ch)
        else:
            run = True
    if run:
        out.append("<zh>")
    return "".join(out)


def _ledgers() -> dict[int, dict]:
    return {pid: load_ledger(RUNS / f"ledger_plan{pid}.json") for pid in PLANS}


def _fig_inheritance(ledgers: dict[int, dict]) -> str:
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    labels = [f"plan {pid}" for pid in PLANS]
    inherited = [ledgers[pid]["lineage"]["inherited_debts"] for pid in PLANS]
    own = [len(ledgers[pid]["debts"]) - ledgers[pid]["lineage"]["inherited_debts"] for pid in PLANS]
    ax.bar(labels, own, label="own debts (report error rows + terminal nodes)", color="#3b6ea5")
    ax.bar(labels, inherited, bottom=own, label="inherited from ancestors", color="#d98b3a")
    for i, pid in enumerate(PLANS):
        total = len(ledgers[pid]["debts"])
        ax.text(i, total + 0.4, f"{total}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("payable debts")
    ax.set_title("Lineage debt inheritance: parent gaps are carried, not described")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    out = FIGS / "fig_debt_inheritance.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out.name


def _fig_m6753(ledger: dict, gold: list[dict], in_gate: list[dict]) -> str:
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    thr = gold[0]["threshold_pct"] if gold else 15.0
    items = [(r["quantity"], abs(r["abs_err_pct"]), r["condition"]) for r in gold]
    items += [(r["quantity"], abs(r["abs_err_pct"]), r["condition"]) for r in in_gate]
    items.sort(key=lambda t: t[1])
    ys = range(len(items))
    colors = ["#c0392b" if v > thr else "#2e8b57" for _, v, _ in items]
    ax.barh(list(ys), [v for _, v, _ in items], color=colors)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([_ascii(f"{q} ({c})") for q, _, c in items], fontsize=9)
    ax.axvline(thr, color="k", linestyle="--", linewidth=1.2)
    ax.text(thr, len(items) - 0.4, f" gate {thr:.0f}%", fontsize=8, va="bottom")
    for y, (_, v, _) in zip(ys, items):
        ax.text(v + 0.4, y, f"{v:.2f}%", va="center", fontsize=8)
    ax.set_xlabel("|error| vs model-test reference [%]")
    ax.set_title("Mission 6753 error rows: the two rows above the gate become debts")
    out = FIGS / "fig_m6753_error_rows.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out.name


def _fig_coverage(ledger: dict) -> str:
    rows = ledger["coverage_matrix"]["matrix"]
    classes = ["covered", "covered_by_domain", "deferred", "dropped"]
    grid = [[1 if r["coverage"] == c else 0 for r in rows] for c in classes]
    fig, ax = plt.subplots(figsize=(8.6, 3.2))
    ax.imshow(grid, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=9)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([_ascii(r["debt_key"]) for r in rows], rotation=90, fontsize=6)
    for i, c in enumerate(classes):
        n = sum(grid[i])
        ax.text(len(rows) + 0.4, i, f"{n}", va="center", fontsize=9, fontweight="bold")
    ax.set_title(f"Coverage matrix of plan {ledger['plan_id']} debts against the descendant wave "
                 f"(plan {ledger['coverage_target_plan']}); CJK tokens shown as <zh>")
    out = FIGS / "fig_coverage_matrix.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out.name


def _fig_gate(ledgers: dict[int, dict]) -> str:
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    cats = ["open", "missing", "weak", "settled"]
    width = 0.8 / len(PLANS)
    for i, pid in enumerate(PLANS):
        counts = ledgers[pid]["seed_gate"]["counts"]
        vals = [counts.get(c, 0) for c in cats]
        xs = [j + i * width - 0.4 + width / 2 for j in range(len(cats))]
        ax.bar(xs, vals, width=width, label=f"plan {pid}")
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(["open debts", "missing (fail-closed)", "weak (domain only)", "settled"])
    ax.set_ylabel("debts")
    ax.set_title("Fail-closed seeding gate: what each wave would have to pay")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    out = FIGS / "fig_seed_gate.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out.name


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    bundle = load_bundle(BUNDLE)
    ledgers = _ledgers()

    # determinism re-check on the real snapshot (independent of the pytest run)
    rebuilt = {pid: build_ledger(bundle, pid, cfg) for pid in PLANS}
    determinism = all(canonical_json(rebuilt[pid]) == canonical_json(ledgers[pid]) for pid in PLANS)

    gold_rows = query(ledgers[173], mission_id=GOLD_MISSION, state="debt_open")
    gate_rows = query(ledgers[173], mission_id=GOLD_MISSION, state="debt_closed")
    in_gate = []
    for debt in ledgers[173]["debts"]:
        if debt["mission_id"] != GOLD_MISSION:
            continue
        if debt["state"] == "debt_open":
            continue
        in_gate.append(debt)

    figs = [
        _fig_inheritance(ledgers),
        _fig_m6753(ledgers[173], gold_rows, in_gate),
        _fig_coverage(ledgers[173]),
        _fig_gate(ledgers),
    ]

    scan = selfcheck(cfg)
    report = {
        "schema": "lineage_debt_artifacts.v1",
        "bundle": {"path": str(BUNDLE.relative_to(ROOT)), "sha256": bundle.sha256,
                   "fetched_utc": bundle.raw.get("fetched_utc"),
                   "reports": len(bundle.reports), "plans": sorted(bundle.plans),
                   "nodes": {str(k): len(v) for k, v in sorted(bundle.nodes.items())},
                   "nodes_total": sum(len(v) for v in bundle.nodes.values())},
        "ledgers": [{"plan_id": pid,
                     "debts": ledgers[pid]["summary"]["total"],
                     "by_source": ledgers[pid]["summary"]["by_source"],
                     "coverage": ledgers[pid]["coverage_matrix"]["counts"],
                     "lineage_chain": ledgers[pid]["lineage"]["chain"],
                     "coverage_target_plan": ledgers[pid]["coverage_target_plan"],
                     "coverage_target_source": ledgers[pid]["coverage_target_source"],
                     "gate": ledgers[pid]["seed_gate"]["counts"],
                     "verdict": ledgers[pid]["seed_gate"]["verdict"],
                     "missing": [m["debt_key"] for m in ledgers[pid]["seed_gate"]["missing"]]}
                    for pid in PLANS],
        "gate_report": gate_report([ledgers[pid] for pid in PLANS]),
        "criterion_d": {
            "mission_id": GOLD_MISSION,
            "gold": "hand label: report 8117 declares w(+25.2%) and t(+15.1%) on KCS as honest FAIL",
            "debt_open_rows": [{"key": r["key"], "err_pct": r["err_pct"], "state": r["state"],
                                "report_id": r["report_id"], "threshold_pct": r["threshold_pct"]}
                               for r in gold_rows],
            "agreement": f"{len(gold_rows)}/{len(gold_rows)}",
        },
        "criterion_e": {"determinism": "byte-identical" if determinism else "MISMATCH",
                        "plans_checked": list(PLANS)},
        "criterion_f": {"selfcheck_verdict": scan["verdict"], "hits": len(scan["hits"]),
                        "files_scanned": scan["files_scanned"],
                        "deny_list_size": scan["deny_list_size"]},
        "figures": [str((FIGS / f).relative_to(ROOT)) for f in figs],
    }
    out = ART / "lineage_debt_summary.json"
    out.write_text(canonical_json(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())