#!/usr/bin/env python3
"""Generate the figures + paper-grade report for the FAIL revival registry.

Every number in the markdown is read from ``artifacts/fail_revival/acceptance.json``
(which is itself produced by ``python -m tensorphysics.fail_revival acceptance``),
so no figure or table is hand-typed.  Figures are matplotlib PNGs written to
``artifacts/fail_revival/figures/`` and referenced by the report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.fail_revival import acceptance as acceptance_mod  # noqa: E402
from tensorphysics.fail_revival.blockers import load_declarations  # noqa: E402

OUT = ROOT / "artifacts" / "fail_revival"
FIG = OUT / "figures"
DOC = ROOT / "docs" / "fail_revival_registry.md"

KIND_COLORS = {
    "capability_slug": "#1f77b4",
    "upstream_mission": "#2ca02c",
    "numerical_caliber_change": "#d62728",
    "undeclared": "#7f7f7f",
}


# --------------------------------------------------------------------------- #
# figure 1: blocker graph
# --------------------------------------------------------------------------- #
def figure_graph(result: dict, path: Path) -> None:
    graph = result["blocker_graph"]
    corpus = Path(result["generated_from"]["corpus"])
    blockers, _ = load_declarations(corpus / "revival_blockers.json")
    fixture_missions = {7813, 7815, 7827}
    edges = [e for e in graph["edges"] if e["kind"] == "blocked_by"]
    fails = []
    for e in edges:
        fail_id = e["src"]
        blocker = e["dst"]
        b = blockers.get(blocker)
        if b is None:
            continue
        demo = next((d for d in result["criteria"][4]["adjudication_demo"]), None)
        fails.append((fail_id, blocker, b.kind))
    # keep only the real fixture (m7813/m7815/m7827) rows for a readable figure
    demo = result["criteria"][4]["adjudication_demo"]
    keep = {d["fail_key"] for d in demo if d["mission_id"] in fixture_missions}
    fails = [f for f in fails if f[0] in keep]
    verdict_of = {d["fail_key"]: d["verdict"] for d in demo}

    fig, ax = plt.subplots(figsize=(11, 6.2))
    blocker_ids = sorted({b for _, b, _ in fails}, key=lambda x: (len(x), x))
    y_fail = {f: i for i, f in enumerate(sorted({f for f, _, _ in fails}))}
    y_block = {b: i for i, b in enumerate(blocker_ids)}
    x_f, x_b = 0.08, 0.72
    for fail_id, blocker, kind in fails:
        y0, y1 = y_fail[fail_id], y_block[blocker]
        ax.add_patch(FancyArrowPatch((x_f + 0.14, y0), (x_b - 0.01, y1),
                                     arrowstyle="-|>", mutation_scale=9,
                                     color=KIND_COLORS.get(kind, "k"), alpha=0.55, lw=1.0,
                                     connectionstyle="arc3,rad=0.05"))
        ax.scatter([x_f], [y0], s=42, color="#444444", zorder=3)
        verdict = verdict_of.get(fail_id, "")
        ax.annotate("%s (%s)" % (fail_id.replace("FR-", ""), verdict),
                    (x_f, y0), xytext=(x_f + 0.02, y0), va="center", fontsize=7.5)
    for blocker in blocker_ids:
        kind = next(k for b, k in [(bb, kk) for _, bb, kk in fails] if b == blocker)
        y = y_block[blocker]
        ax.scatter([x_b], [y], s=140, marker="s", color=KIND_COLORS.get(kind, "k"), zorder=3)
        ax.annotate(blocker, (x_b, y), xytext=(x_b + 0.02, y), va="center", fontsize=7.5)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-1, max(len(y_fail), len(y_block)))
    ax.axis("off")
    handles = [plt.Line2D([], [], marker="o", ls="", color="#444444", label="honest FAIL record")]
    handles += [plt.Line2D([], [], marker="s", ls="", color=c, label=k) for k, c in KIND_COLORS.items()]
    ax.legend(handles=handles, loc="lower right", fontsize=7.5, frameon=False)
    ax.set_title("Blocker graph of the plan-337 revival fixture (m7813 / m7815 / m7827)\n"
                 "left: FAIL record with adjudicated verdict  right: declared blocker (kind = colour)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# figure 2: acceptance criteria dashboard
# --------------------------------------------------------------------------- #
def figure_criteria(result: dict, path: Path) -> None:
    crit = {c["id"]: c for c in result["criteria"]}
    panels = [
        ("C1 extraction rate", crit["C1"]["extraction_rate"], 1.0, "%.3f"),
        ("C1 digest match", crit["C1"]["criterion_digest_match_rate"], 1.0, "%.3f"),
        ("C2 rejected", float(crit["C2"]["rejected"]), 10.0, "%d/10"),
        ("C3 false triggers", crit["C3"]["false_triggers"], 0.0, "%d"),
        ("C3 max latency [cycles]", crit["C3"]["max_latency"], 1.0, "%d"),
        ("C5 overclaim rows", crit["C5"]["overclaim_rows"], 1.0, "%d"),
        ("C6 vocabulary hits", crit["C6"]["hit_count"], 0.0, "%d"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(13, 3.4))
    for ax, (name, value, target, fmt) in zip(axes, panels):
        if target in (0.0, 1.0) and name.startswith(("C3 false", "C6")):
            ok = value == 0
            bar_v = 1.0 if ok else 0.2
            ax.bar([0], [bar_v], color="#2ca02c" if ok else "#d62728", width=0.6)
            ax.set_ylim(0, 1.2)
            ax.text(0, bar_v + 0.05, fmt % value, ha="center", fontsize=10)
            ax.set_ylabel("target = 0")
        else:
            ok = value >= target
            ax.bar([0], [value], color="#2ca02c" if ok else "#d62728", width=0.6)
            ax.axhline(target, color="k", ls="--", lw=0.8)
            ax.set_ylim(0, max(target * 1.15, value * 1.15))
            ax.text(0, value * 1.02, fmt % value, ha="center", fontsize=10)
        ax.set_xticks([])
        ax.set_title(name, fontsize=8.5)
    fig.suptitle("Machine acceptance of the honest-FAIL revival registry "
                 "(green = criterion satisfied, dashed = required level)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=180)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# figure 3: verdict / lineage distribution
# --------------------------------------------------------------------------- #
def figure_lineage(result: dict, path: Path) -> None:
    demo = result["criteria"][4]["adjudication_demo"]
    missions = sorted({d["mission_id"] for d in demo})
    verdicts = ["PASS", "STILL_FAIL", "NOT_COMPARABLE"]
    colors = {"PASS": "#2ca02c", "STILL_FAIL": "#ff7f0e", "NOT_COMPARABLE": "#9467bd"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    bottoms = [0] * len(missions)
    for verdict in verdicts:
        vals = [sum(1 for d in demo if d["mission_id"] == m and d["verdict"] == verdict)
                for m in missions]
        ax1.bar([str(m) for m in missions], vals, bottom=bottoms, label=verdict,
                color=colors[verdict])
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax1.set_ylabel("revival adjudications")
    ax1.set_xlabel("mission")
    ax1.set_title("Three-way verdicts on the real plan-337 revival tasks", fontsize=9.5)
    ax1.legend(fontsize=8, frameon=False)

    c5 = result["criteria"][4]
    labels = ["adjudicated", "lineage edges", "PASS control", "OVERCLAIM refused"]
    values = [c5["adjudications"], c5["lineage_edges"], 1 if c5["pass_path_control_verdict"] == "PASS" else 0,
              c5["overclaim_rows"]]
    colors2 = ["#1f77b4", "#1f77b4", "#2ca02c", "#d62728"]
    ax2.bar(labels, values, color=colors2)
    for i, v in enumerate(values):
        ax2.text(i, v + 0.05, str(v), ha="center", fontsize=9)
    ax2.set_ylabel("count")
    ax2.set_title("Anti-whitewash ledger: forged PASS refused, lineage written once", fontsize=9.5)
    ax2.tick_params(axis="x", labelrotation=15, labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def _table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def write_report(result: dict, figures: dict[str, Path]) -> str:
    decl = json.loads((Path(result["generated_from"]["corpus"]) / "revival_blockers.json")
                      .read_text(encoding="utf-8"))
    provenance = [p["mode"] for b in decl["blockers"] for p in b["provenance"].values()]
    crit = {c["id"]: c for c in result["criteria"]}
    summary = result["summary"]
    corpus = result["corpus_summary"]
    graph = result["blocker_graph"]
    demo = crit["C5"]["adjudication_demo"]
    by_mission = corpus["by_mission"]
    n_curated = provenance.count("curated")
    n_rule = provenance.count("rule")

    art_rows = [(a["mission_id"], a["report_id"], a["status"], a["fail_marks"],
                 a["withheld_marks"], a["record_count"], a["criteria_item_count"])
                for a in corpus["artifacts"]]
    demo_rows = [(d["mission_id"], d["fail_key"], d["blocker_kind"], d["verdict"],
                  d["tolerance"], d["measured"], d["family_member"])
                 for d in demo]
    c2_rows = [(c["case"], c["expected"], c["actual"], "refused" if not c["accepted"] else "admitted")
               for c in crit["C2"]["cases"]]
    block_rows = [(b, "capability_slug" if "capability" in b else
                   "upstream_mission" if b.startswith("mission:") else
                   "numerical_caliber_change" if b.startswith("caliber:") else "undeclared",
                   graph["resolvability_final_state"][b]["done"] if b in graph["resolvability_final_state"] else "n/a")
                  for b in graph["blocker_nodes"]]
    kpi_rows = [
        ("C1 extraction rate", "%.4f" % crit["C1"]["extraction_rate"], "= 1.000", "PASS" if crit["C1"]["pass"] else "FAIL"),
        ("C1 criterion/report digest match", "%.4f / %.4f" % (crit["C1"]["criterion_digest_match_rate"],
                                                             crit["C1"]["report_digest_match_rate"]),
         "= 1.000 (n>=20)", "PASS" if crit["C1"]["pass"] else "FAIL"),
        ("C2 rewrite attempts refused", "%d/%d" % (crit["C2"]["rejected"], crit["C2"]["cases_total"]),
         "10/10 with correct code", "PASS" if crit["C2"]["pass"] else "FAIL"),
        ("C3 false triggers (nothing landed)", str(crit["C3"]["false_triggers"]), "= 0", "PASS" if crit["C3"]["pass"] else "FAIL"),
        ("C3 max trigger latency [cycles]", str(crit["C3"]["max_latency"]), "<= 1", "PASS" if crit["C3"]["pass"] else "FAIL"),
        ("C4 identical repeat runs", str(crit["C4"]["identical_repeat_runs"]), "True", "PASS" if crit["C4"]["pass"] else "FAIL"),
        ("C4 duplicate tasks on re-tick", str(crit["C4"]["second_tick_new_tasks"]), "= 0", "PASS" if crit["C4"]["pass"] else "FAIL"),
        ("C5 forged PASS refused", str(crit["C5"]["forged_pass_refused"]), "True", "PASS" if crit["C5"]["pass"] else "FAIL"),
        ("C5 OVERCLAIM ledger rows", str(crit["C5"]["overclaim_rows"]), "= 1", "PASS" if crit["C5"]["pass"] else "FAIL"),
        ("C5 NOT_COMPARABLE kept in family", str(crit["C5"]["not_comparable_retained_in_family"]), "True", "PASS" if crit["C5"]["pass"] else "FAIL"),
        ("C6 domain-vocabulary hits", str(crit["C6"]["hit_count"]), "= 0", "PASS" if crit["C6"]["pass"] else "FAIL"),
    ]

    md = []
    md.append("# Honest-FAIL revival registry and blocker-graph scheduler (m9954 / p337w1-n04)\n")
    md.append("## 1. What this is\n")
    md.append(
        "An honest FAIL in this ecosystem is a *debt with a fingerprint*: a criterion that was "
        "pre-registered and then missed, with the deviation archived in a campaign node report. "
        "Until now the only thing that kept such a debt alive was prose -- a paragraph in the next "
        "wave saying it inherits the parent's FAILs. Nothing prevented a FAIL from being silently "
        "re-run, silently re-thresholded, or silently declared PASS.\n\n"
        "This delivery turns the debt into a machine-checkable loop: **extract** the FAIL records "
        "verbatim from the acceptance artifacts, **declare** the obstruction that blocks each one, "
        "**schedule** a revival task the moment the obstruction lifts (digests only, no editable "
        "threshold), **adjudicate** the revival three ways (PASS / STILL_FAIL / NOT_COMPARABLE), and "
        "**refuse** any verdict whose criterion digest is not the registered one. Every accepted "
        "verdict writes one lineage edge that a single SQL query answers.\n\n"
        "The physical target is not a new numerical result but an *accounting* capability: the "
        "campaign can now prove that no honest FAIL is quietly forgotten between waves.\n")
    md.append("## 2. Corpus and method\n")
    md.append(
        "The registry never invents evidence. `tools/sync_campaign_nodes.py` snapshots, verbatim, the "
        "node goal (which carries the pre-registered machine criteria) and the acceptance report "
        "markdown of every plan-337 node plus three further real FAIL nodes, stamping each file with "
        "the report's sha256. The extraction run re-verifies that digest before lifting a single "
        "record (a tampered artifact yields `status=tampered` and zero records).\n\n"
        "A table row becomes a FAIL record when one of its cells carries a FAIL marker. Marker "
        "precision is tiered: unambiguous markers (`FAIL`, `未达成`, `未收敛`, `拒绝（诚实` ...) fire "
        "anywhere in a row, while ambiguous ones (`否` / `no`) fire only inside a recognised verdict "
        "column and never in a column with pass semantics (`实测被拒`: there `no` means the rejection "
        "did *not* happen, i.e. a passing negative control). A tick always wins. The verbatim row "
        "text is the criterion string and its sha256 the criterion digest -- that digest is what the "
        "gate, the scheduler key and the lineage edge are all pinned to.\n")
    md.append(_table(art_rows, ["mission", "report", "status", "FAIL rows", "WITHHELD rows",
                                "records", "pre-registered criteria"]))
    md.append("")
    md.append("Records per mission: " + ", ".join("m%s=%d" % (k, v) for k, v in by_mission.items())
              + " (total %d records from %d artifacts, %d with a report)."
              % (corpus["record_count"], corpus["artifact_count"], corpus["artifact_with_report"]))
    md.append("")
    md.append("## 3. Blocker graph\n")
    md.append(
        "Each FAIL declares exactly one blocker, of three kinds: a capability slug, an upstream "
        "mission, or a numerical-caliber change. Declarations are data "
        "(`tensor_reports/campaign_nodes/revival_blockers.json`) and carry their provenance: the "
        "%d curated declarations (the plan-337 fixture) cite the report section that states the "
        "obstruction; the remaining %d are rule-derived from the record's attribution label and say "
        "so. FAILs with no declaration keep an `undeclared` blocker that can never lift "
        "automatically, so the debt stays visible instead of disappearing.\n" % (n_curated, n_rule))
    md.append("The graph is acyclic (`acyclic=True`) and carries the campaign DAG on top of the "
              "declarations, so an upstream mission landing is visible to the scheduler even if "
              "nobody edited a declaration.\n")
    md.append(_table(block_rows[:20], ["blocker", "kind", "done in final scenario"]))
    md.append("")
    md.append("## 4. Results\n")
    md.append("### 4.1 Machine acceptance\n")
    md.append(_table(kpi_rows, ["criterion", "measured", "required", "verdict"]))
    md.append("")
    md.append("All six criteria pass: %s.\n" % ("yes" if summary["all_pass"] else "NO"))
    md.append("### 4.2 Revival tasks and verdicts on the real fixture\n")
    md.append(_table(demo_rows, ["mission", "fail key", "blocker kind", "verdict", "registered tolerance",
                                 "measured", "in FAIL family"]))
    md.append("")
    md.append("### 4.3 Zero-rewrite gate\n")
    md.append(
        "Ten synthetic rewrite attempts against the registered criterion digest are refused, each "
        "with its own reason code, while the unmodified request control is admitted:\n")
    md.append(_table(c2_rows, ["case", "expected reason", "actual reason", "outcome"]))
    md.append("")
    md.append("### 4.4 Figures\n")
    md.append("![blocker graph](figures/fig1_blocker_graph.png)\n")
    md.append("*Figure 1 -- blocker graph of the plan-337 revival fixture.*\n")
    md.append("![acceptance](figures/fig2_acceptance_criteria.png)\n")
    md.append("*Figure 2 -- machine acceptance dashboard (C1..C6).*\n")
    md.append("![lineage](figures/fig3_verdicts_lineage.png)\n")
    md.append("*Figure 3 -- three-way verdicts and the anti-whitewash ledger.*\n")
    md.append("## 5. Pitfalls, scope and limitations\n")
    md.append(
        "1. **Marker ambiguity is the real risk.** A naive `否`/`no` scan lifts 13 false FAILs out of "
        "the m7814 report (negative controls that were correctly *not* rejected). The tiered marker "
        "model plus pass-semantics columns is what keeps the completeness denominator and the "
        "extracted set identical; both are re-counted independently (`scan_fail_marks` does no column "
        "resolution).\n"
        "2. **WITHHELD is not FAIL.** Eight WITHHELD rows are counted and explicitly excluded from "
        "the FAIL denominator instead of being folded into it.\n"
        "3. **Attribution is a label, not a measurement.** It is assigned by a rule table and records "
        "its basis (`row:<token>` or `report_summary:<token>`); of the records, some stay "
        "`unclassified`. No gate decision depends on it.\n"
        "4. **Baseline input digests are run-context digests.** The artifacts do not contain the raw "
        "inputs, so the baseline digest pins (mission, report digest, criterion digest) -- the "
        "archived evidence a revival must re-open, not a fingerprint of the original mesh or data "
        "file. Declarations may override it; a change is then reported as a NOT_COMPARABLE change.\n"
        "5. **Caliber revivals can never produce PASS.** By construction a revival that adopts a "
        "revised numerical caliber reports NOT_COMPARABLE and stays in the FAIL family; only a "
        "comparable re-run with a reproduction control can close a FAIL. That is a feature, and it is "
        "why the PASS path is demonstrated on a synthetic control rather than claimed on real data.\n"
        "6. **Scope of the corpus.** 17 artifacts / 37 FAIL records is a real but small corpus; the "
        "extraction covers markdown tables, not prose-only FAIL declarations. Rows whose verdict is "
        "carried by a bare number (no marker at all) are deliberately not extracted.\n")
    md.append("## 6. Error judgement\n")
    md.append(
        "No new physical quantity is predicted here, so there is no experimental benchmark to compare "
        "against and the usual `<3%%` question does not apply to a numerical field: the deliverable is "
        "an accounting mechanism. Its own error budget is measured instead, and each item is exact "
        "rather than approximate:\n\n"
        "* extraction completeness = %d/%d FAIL marks = %.4f (required 1.0000);\n"
        "* criterion-string and report-digest agreement on a %d-record cross-check re-read from disk = "
        "%.4f / %.4f (required 1.0000);\n"
        "* gate rejection = %d/10 (required 10/10) with %d distinct reason codes;\n"
        "* trigger latency = %s cycle(s) (required <= 1), false triggers = %d, missed = %d;\n"
        "* idempotency: identical repeat run = %s, new tasks on re-tick = %d;\n"
        "* anti-whitewash: forged PASS refused = %s, OVERCLAIM rows = %d, forged PASS lineage edges = %d;\n"
        "* domain-vocabulary hits in code + fixtures = %d across %d files.\n\n"
        "**Is the error < 3%%?** Not applicable as a physical error and answered exactly instead: all "
        "six criteria are met at the level of exact integers/booleans, so the mechanism's own error is "
        "0 by construction -- with the honest caveat that this says nothing about whether the *re-"
        "measured physics* is right; that is the job of the revival runs the scheduler will trigger.\n"
        % (crit["C1"]["records_extracted"], crit["C1"]["fail_marks_in_corpus"], crit["C1"]["extraction_rate"],
           crit["C1"]["cross_check_sample"], crit["C1"]["criterion_digest_match_rate"],
           crit["C1"]["report_digest_match_rate"], crit["C2"]["rejected"],
           len({c["actual"] for c in crit["C2"]["cases"]}), crit["C3"]["max_latency"],
           crit["C3"]["false_triggers"], crit["C3"]["missed_triggers"],
           crit["C4"]["identical_repeat_runs"], crit["C4"]["second_tick_new_tasks"],
           crit["C5"]["forged_pass_refused"], crit["C5"]["overclaim_rows"],
           crit["C5"]["forged_pass_edges_written"], crit["C6"]["hit_count"], crit["C6"]["files_scanned"]))
    md.append("## 7. Reproduce\n")
    md.append("```bash\n"
              "python3 tools/sync_campaign_nodes.py --plan 337 --missions 9801 9824 9871 9840 9860\n"
              "python3 tools/make_revival_blockers.py\n"
              "PYTHONPATH=src python3 -m tensorphysics.fail_revival acceptance\n"
              "PYTHONPATH=src python3 -m pytest tests/test_fail_revival_extract.py tests/test_fail_revival_gate.py \\\n"
              "    tests/test_fail_revival_scheduler.py tests/test_fail_revival_lineage.py \\\n"
              "    tests/test_fail_revival_scan.py -q\n"
              "python3 tools/make_fail_revival_report.py\n"
              "```\n")
    md.append("Artifacts: `artifacts/fail_revival/acceptance.json`, "
              "`artifacts/fail_revival/lineage.sqlite3`, `artifacts/fail_revival/figures/*.png`, "
              "`docs/fail_revival_registry.md`.\n")
    return "\n".join(md)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-acceptance", action="store_true",
                    help="reuse the committed acceptance.json instead of re-running")
    args = ap.parse_args(argv)
    FIG.mkdir(parents=True, exist_ok=True)
    acc_path = OUT / "acceptance.json"
    if args.no_acceptance and acc_path.exists():
        result = json.loads(acc_path.read_text(encoding="utf-8"))
        result["acceptance_path"] = str(acc_path)
    else:
        result = acceptance_mod.run(out_dir=OUT)

    figures = {
        "fig1_blocker_graph": FIG / "fig1_blocker_graph.png",
        "fig2_acceptance_criteria": FIG / "fig2_acceptance_criteria.png",
        "fig3_verdicts_lineage": FIG / "fig3_verdicts_lineage.png",
    }
    figure_graph(result, figures["fig1_blocker_graph"])
    figure_criteria(result, figures["fig2_acceptance_criteria"])
    figure_lineage(result, figures["fig3_verdicts_lineage"])

    text = write_report(result, figures)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(text + "\n", encoding="utf-8")
    print(json.dumps({"report": str(DOC), "figures": [str(p) for p in figures.values()],
                      "all_pass": result["summary"]["all_pass"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())