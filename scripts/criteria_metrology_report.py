#!/usr/bin/env python3
"""P266W1-N09 — build the report artefacts (numbers + figures + platform payload).

Everything is computed from the frozen artifacts (corpus, label set, case studies, controls),
so no number in the report is typed by hand.  Outputs:

* ``artifacts/criteria_metrology/report_data.json`` — every acceptance number with its source
* ``docs/figures/p266w1n09_*.png`` — four figures
* ``/tmp/report.json`` — the platform report payload (title/summary/markdown/figures)

Usage::

    python3 scripts/criteria_metrology_report.py [--publish]
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorphysics.metrology.criteria_semantics import corpus as C  # noqa: E402
from tensorphysics.metrology.criteria_semantics import labeling, replay as R  # noqa: E402
from tensorphysics.metrology.criteria_semantics.adaptive import evaluate  # noqa: E402
from tensorphysics.metrology.criteria_semantics.controls import run_controls  # noqa: E402
from tensorphysics.metrology.criteria_semantics.model import (  # noqa: E402
    Criterion, Degeneracy, Measurement, Verdict, relative_error,
)

ART = "artifacts/criteria_metrology"
FIGDIR = "docs/figures"
CORPUS = os.path.join(ART, "historical_criteria.jsonl")
CASE = os.path.join(ART, "case_studies.json")
SUMMARY = os.path.join(ART, "corpus_summary.json")


def invariance_probe(n: int = 1500, seed: int = 20261110) -> dict:
    """Property probe: y_ref -> 0 must not produce an unbounded relative error."""
    rng = random.Random(seed)
    baseline_unbounded = 0          # rows where the *baseline* had no defined number
    adaptive_withheld = 0
    max_adaptive_error = 0.0
    for _ in range(n):
        exp = rng.uniform(-300, 3)
        y_ref = 0.0 if rng.random() < 0.1 else math.copysign(10 ** exp, rng.choice((-1, 1)))
        y = math.copysign(10 ** rng.uniform(-300, 3), rng.choice((-1, 1)))
        crit = Criterion(measurement=Measurement(y=y, y_ref=y_ref), tau=0.03)
        if relative_error(y, y_ref) is None:
            baseline_unbounded += 1
        ev = evaluate(crit)
        assert ev.verdict in (Verdict.PASS, Verdict.FAIL, Verdict.WITHHELD)
        assert ev.error is None or math.isfinite(ev.error)
        if ev.verdict is Verdict.WITHHELD:
            adaptive_withheld += 1
        if ev.error is not None:
            max_adaptive_error = max(max_adaptive_error, ev.error)
    return {"n_samples": n, "baseline_undefined": baseline_unbounded,
            "adaptive_withheld": adaptive_withheld,
            "max_adaptive_error": max_adaptive_error,
            "no_unbounded_number": True}


def fig_invariance(path: str) -> None:
    """Fig 2: relative error vs reference magnitude for a fixed absolute deviation."""
    delta = 1e-3
    refs = [10 ** (e / 4) for e in range(-32, 9)]
    base, adapt = [], []
    for yr in refs:
        y = yr + delta
        b = relative_error(y, yr)
        base.append(b if b is not None and b < 1e4 else (1e4 if b is None else b))
        ev = evaluate(Criterion(measurement=Measurement(y=y, y_ref=yr), tau=0.03))
        adapt.append(ev.error)
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.loglog(refs, base, "o-", label="baseline: |Δ|/|y_ref|  (unbounded as y_ref → 0)")
    ax.loglog(refs, adapt, "s-", label="adaptive: |Δ|/scale_y  (bounded)")
    ax.axhline(0.03, color="k", ls="--", lw=1, label="τ = 3%")
    ax.set_xlabel("reference magnitude |y_ref|  (Δ = 1e-3 fixed)")
    ax.set_ylabel("error under each criterion")
    ax.set_title("Fig 2  T1 remedy: the unbounded relative error is replaced by a bounded one")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_degeneracy(path: str, ledger: dict) -> None:
    by_type = {k: v for k, v in ledger["by_type"].items()}
    witness: dict[str, int] = {}
    for e in ledger["entries"]:
        if e.degeneracy is not Degeneracy.NONE:
            w = str(e.degeneracy_witnesses.get("why") or "unspecified")
            witness[w] = witness.get(w, 0) + 1
        elif e.self_referential:
            witness["self_referential"] = witness.get("self_referential", 0) + 1
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    labels = ["NONE", "T1_REFERENCE\n_NEAR_ZERO", "T2_SIGN\n_CROSSING",
              "T3_MULTI_METRIC\n_NO_PRIORITY"]
    vals = [by_type.get("NONE", 0), by_type.get("T1_REFERENCE_NEAR_ZERO", 0),
            by_type.get("T2_SIGN_CROSSING", 0), by_type.get("T3_MULTI_METRIC_NO_PRIORITY", 0)]
    axes[0].bar(labels, vals, color=["#9aa5b1", "#e08c3a", "#c4534b", "#4b7bec"])
    axes[0].set_ylabel("criteria")
    axes[0].set_title(f"Fig 1a  degeneracy classes over {ledger['n_criteria']} replay criteria")
    axes[0].tick_params(axis="x", labelsize=7)
    for i, v in enumerate(vals):
        axes[0].text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    top = sorted(witness.items(), key=lambda kv: -kv[1])[:8]
    axes[1].barh([k for k, _ in top][::-1], [v for _, v in top][::-1], color="#4b7bec")
    axes[1].set_xlabel("criteria")
    axes[1].set_title("Fig 1b  witnesses that produced the labels")
    axes[1].tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_cases(path: str, art: dict) -> None:
    import re as _re
    rows = []
    for m, cs in art["cases"].items():
        for k, c in enumerate(cs):
            ascii_name = _re.sub(r"[^\x20-\x7F]+", " ", c["name"]).strip()
            ascii_name = " ".join(ascii_name.split())[:14] or f"case{k}"
            rows.append((f"{m}\n{ascii_name}", c))
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    xs = range(len(rows))
    ratios = [(c["alternative_criterion_ratio"] or 0.0) for _, c in rows]
    colors = ["#c4534b" if c["adaptive"]["verdict"] == "FAIL" else
              ("#4b7bec" if c["adaptive"]["verdict"] == "PASS" else "#9aa5b1")
              for _, c in rows]
    ax.bar(xs, ratios, color=colors)
    ax.axhline(1.0, color="k", ls="--", lw=1, label="error = τ")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([k for k, _ in rows], rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("error / τ  under the adaptive criterion")
    ax.set_title("Fig 3  four case studies re-judged (red FAIL, blue PASS, grey WITHHELD)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_labels(path: str, res: dict, controls: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    pairs = [("A vs classifier", res["annotator_A_vs_classifier"]),
             ("A vs B (text-only)", res["annotator_A_vs_B"])]
    axes[0].bar([p[0] for p in pairs], [p[1]["agreement"] * 100 for p in pairs],
                color=["#4b7bec", "#e08c3a"])
    for i, (_, d) in enumerate(pairs):
        axes[0].text(i, d["agreement"] * 100 + 1,
                     f"{d['agreement']*100:.1f}%\nκ={d['kappa']:.2f}", ha="center", fontsize=8)
    axes[0].axhline(90, color="k", ls="--", lw=1, label="acceptance 90 %")
    axes[0].set_ylim(0, 110)
    axes[0].set_ylabel("agreement on the 40 labelled rows (%)")
    axes[0].set_title("Fig 4a  labelling agreement")
    axes[0].legend(fontsize=8)
    cats = ["degenerate\ndetected /12", "benign\nfalse alarms /12", "benign\nverdict ok /12"]
    vals = [controls["degenerate_detected"], controls["benign_false_positive"],
            controls["benign_verdict_ok"]]
    axes[1].bar(cats, vals, color=["#4b7bec", "#c4534b", "#41a05e"])
    for i, v in enumerate(vals):
        axes[1].text(i, v + 0.2, f"{v}/12", ha="center", fontsize=9)
    axes[1].set_ylim(0, 14)
    axes[1].set_title("Fig 4b  negative-control set")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args()
    os.makedirs(FIGDIR, exist_ok=True)

    rows = C.load_corpus(CORPUS)
    ledger = R.replay_corpus(rows)
    with open(SUMMARY, encoding="utf-8") as fh:
        corpus_summary = json.load(fh)
    with open(CASE, encoding="utf-8") as fh:
        cases = json.load(fh)
    controls = run_controls()
    controls_json = {k: v for k, v in controls.items() if k != "results"}
    controls_json["results"] = [r.as_dict() for r in controls["results"]]
    ls = labeling.load_label_set()
    labels = labeling.score_labels(ls)
    inv = invariance_probe()

    figs = {
        "p266w1n09_degeneracy.png": ("Fig 1 degeneracy classes and their witnesses",
                                     lambda p: fig_degeneracy(p, ledger)),
        "p266w1n09_invariance.png": ("Fig 2 unbounded vs bounded error as the reference vanishes",
                                     lambda p: fig_invariance(p)),
        "p266w1n09_cases.png": ("Fig 3 case studies: error/tau under the adaptive criterion",
                                lambda p: fig_cases(p, cases)),
        "p266w1n09_labels.png": ("Fig 4 labelling agreement and the negative-control set",
                                 lambda p: fig_labels(p, labels, controls)),
    }
    for name, (_cap, fn) in figs.items():
        fn(os.path.join(FIGDIR, name))

    data = {
        "mission": 9554,
        "node": "P266W1-N09",
        "corpus": {"n_criteria": ledger["n_criteria"], "n_reports": ledger["n_reports"],
                   "sha256": corpus_summary["manifest"]["sha256"],
                   "path": CORPUS},
        "degeneracy": {"rate": ledger["degeneracy_rate"],
                       "rate_strict": ledger["degeneracy_rate_strict"],
                       "n_degenerate": ledger["n_degenerate"],
                       "n_degenerate_strict": ledger["n_degenerate_strict"],
                       "n_self_referential": ledger["n_self_referential"],
                       "by_type": ledger["by_type"]},
        "replay": {"flips": ledger["flips"], "recorded_flips": ledger["recorded_flips"],
                   "n_softened_to_fail": ledger["n_recorded_softened_to_fail"],
                   "baseline_withheld": ledger["baseline_withheld"]},
        "controls": controls_json,
        "labels": {k: labels[k] for k in
                   ("n_labels", "annotator_A_vs_classifier", "annotator_A_vs_B",
                    "classifier_vs_B")},
        "invariance": inv,
        "cases": {"n_cases": cases["n_cases"],
                  "n_direction_consistent": cases["n_direction_consistent"],
                  "m7241_explicit_fail": cases["m7241_explicit_fail"],
                  "m7241": cases["cases"]["m7241"],
                  "all": cases["cases"]},
        "figures": list(figs),
    }
    with open(os.path.join(ART, "report_data.json"), "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)

    criteria = {
        "(a) 40条标注一致率≥90%": {"value": labels["annotator_A_vs_classifier"]["agreement"],
                              "kappa": labels["annotator_A_vs_classifier"]["kappa"],
                              "pass": labels["annotator_A_vs_classifier"]["agreement"] >= 0.90},
        "(b) 回放≥200条且退化率≥5%": {"n_criteria": ledger["n_criteria"],
                                "rate": ledger["degeneracy_rate"],
                                "pass": ledger["n_criteria"] >= 200 and
                                        ledger["degeneracy_rate"] >= 0.05},
        "(c) 四案例替代判据+方向一致4/4": {"n_cases": cases["n_cases"],
                                   "consistent": cases["n_direction_consistent"],
                                   "m7241_fail": cases["m7241_explicit_fail"],
                                   "pass": cases["n_direction_consistent"] == cases["n_cases"]
                                           and cases["m7241_explicit_fail"]},
        "(d) 负控 12/12 检出 & 0/12 误报": {"detected": controls["degenerate_detected"],
                                     "false_positive": controls["benign_false_positive"],
                                     "pass": controls["degenerate_detected"] == 12
                                             and controls["benign_false_positive"] == 0},
        "(e) y_ref→0 有界性属性测试≥1000": {"n_samples": inv["n_samples"],
                                     "pass": inv["n_samples"] >= 1000 and inv["no_unbounded_number"]},
    }
    data["acceptance"] = criteria
    with open(os.path.join(ART, "acceptance.json"), "w", encoding="utf-8") as fh:
        json.dump(criteria, fh, ensure_ascii=False, indent=1)

    for k, v in criteria.items():
        print(("PASS " if v["pass"] else "FAIL ") + k, json.dumps(
            {kk: vv for kk, vv in v.items() if kk != "pass"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())