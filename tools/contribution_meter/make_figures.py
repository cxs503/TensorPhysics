#!/usr/bin/env python3
"""Figures for the contribution-meter report, generated from the JSON artifacts only.

No number in the figures is typed by hand: everything is read from
  data/golden_eval.json              (gold-set metrics / confusion)
  data/contribution_report_48h.json  (window unit counts / conversion table)
  data/contribution_report_96h.json  (stability check)

Outputs (artifacts/):
  fig1_confusion.png      gold-set confusion matrix (rows = gold, cols = predicted)
  fig2_class_f1.png       per-class F1 with the 0.70 class bar and 0.85 macro bar
  fig3_window_units.png   48 h window: mission funnel + unit composition
  fig4_conversion.png     M_mission -> SCU conversion
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(ROOT, "artifacts")
CLASSES = ["C1", "C2", "C3", "C4", "C5", "C6"]
NAMES = {
    "C1": "C1 verified-case", "C2": "C2 capability-delta", "C3": "C3 defect-fix",
    "C4": "C4 method/doc", "C5": "C5 negative-result", "C6": "C6 infra", "NONE": "NONE",
}


def load(name: str) -> dict:
    with open(os.path.join(ROOT, "data", name), encoding="utf-8") as fh:
        return json.load(fh)


def fig_confusion(ev: dict, path: str) -> None:
    conf = ev["metrics"]["confusion"]
    labels = CLASSES + ["NONE"]
    mat = np.array([[conf.get(g, {}).get(p, 0) for p in labels] for g in labels], dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted (rule + field classifier)")
    ax.set_ylabel("gold (hand annotation)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = int(mat[i, j])
            if v:
                ax.text(j, i, str(v), ha="center", va="center",
                        color="white" if v > mat.max() * 0.5 else "black", fontsize=10)
    ax.set_title(f"Gold-set confusion (n={ev['metrics']['n']}, macro F1={ev['metrics']['macro_f1']:.3f})")
    fig.colorbar(im, ax=ax, shrink=0.8, label="count")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_class_f1(ev: dict, path: str) -> None:
    per = ev["metrics"]["per_class"]
    f1 = [per[c]["f1"] for c in CLASSES]
    prec = [per[c]["precision"] for c in CLASSES]
    rec = [per[c]["recall"] for c in CLASSES]
    x = np.arange(len(CLASSES))
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.bar(x - 0.26, prec, width=0.25, label="precision")
    ax.bar(x, rec, width=0.25, label="recall")
    ax.bar(x + 0.26, f1, width=0.25, label="F1")
    ax.axhline(0.70, color="tab:red", ls="--", lw=1.2, label="per-class bar 0.70")
    ax.axhline(ev["metrics"]["macro_f1"], color="k", ls=":", lw=1.2,
               label=f"macro F1 {ev['metrics']['macro_f1']:.3f}")
    ax.axhline(0.85, color="tab:green", ls="--", lw=1.2, label="acceptance 0.85")
    ax.set_xticks(x, CLASSES)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("score")
    ax.set_title("Per-class precision / recall / F1 on the 60-item gold set "
                 f"(support: " + ", ".join(f"{c}:{per[c]['support']}" for c in CLASSES) + ")")
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_window(rep48: dict, path: str) -> None:
    conv = rep48["conversion"]
    counts = conv["units_by_class"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3))
    ax = axes[0]
    stages = ["missions in window", "published a report", "scientific unit (C1-C5)"]
    values = [conv["n_missions"], conv["n_missions_with_report"], conv["n_scientific_units"]]
    ax.bar(stages, values, color=["#8fbcd4", "#5b8db8", "#2f5d80"])
    for i, v in enumerate(values):
        ax.text(i, v * 1.02, f"{v}", ha="center", fontsize=10)
    ax.set_ylabel("count")
    ax.set_title(f"48 h window funnel (zero-contribution ratio {conv['zero_contribution_ratio']:.3f})")
    ax.tick_params(axis="x", labelsize=8)

    ax = axes[1]
    vals = [counts.get(c, 0) for c in CLASSES] + [counts.get("NONE", 0)]
    labels = CLASSES + ["NONE"]
    ax.bar(labels, vals, color=["#2f5d80"] * len(CLASSES) + ["#bbbbbb"])
    for i, v in enumerate(vals):
        ax.text(i, v * 1.02 if v else 0.5, f"{v}", ha="center", fontsize=9)
    ax.set_ylabel("units (missions)")
    ax.set_title("Primary unit per mission (one label per mission)")
    fig.suptitle("48 h observation window: unit composition", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_conversion(rep48: dict, path: str) -> None:
    conv = rep48["conversion"]
    counts = conv["units_by_class"]
    weights = conv["weights"]
    scus = {c: counts.get(c, 0) * weights[c] for c in CLASSES}
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    bottom = 0.0
    for c in CLASSES:
        ax.bar(["Σ contribution (SCU)"], [scus[c]], bottom=bottom, label=f"{c} x{weights[c]}")
        if scus[c] > 3:
            ax.text(0, bottom + scus[c] / 2, f"{c} {scus[c]:.1f}", ha="center", va="center",
                    color="white", fontsize=9)
        bottom += scus[c]
    ax.bar(["M_mission (missions)"], [conv["n_missions"]], color="#dddddd")
    ax.text(1, conv["n_missions"] * 0.5, f"{conv['n_missions']} missions\n"
            f"{conv['mission_units_per_scientific_contribution']:.2f} missions per "
            f"scientific unit", ha="center", va="center", fontsize=9)
    ax.text(0, bottom * 1.02, f"Σ {conv['sum_weighted_scu']:.1f} SCU", ha="center", fontsize=10)
    ax.set_yscale("log")
    ax.set_ylabel("count / SCU (log scale)")
    ax.set_title("M_mission -> Σ contribution conversion, same 48 h window")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_stability(rep48: dict, rep96: dict, path: str) -> None:
    keys = ["scientific_units_per_mission", "zero_contribution_ratio", "scu_per_mission"]
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    x = np.arange(len(keys))
    v48 = [rep48["conversion"][k] for k in keys]
    v96 = [rep96["conversion"][k] for k in keys]
    ax.bar(x - 0.2, v48, width=0.38, label="48 h window")
    ax.bar(x + 0.2, v96, width=0.38, label="96 h window")
    for i, (a, b) in enumerate(zip(v48, v96)):
        ax.text(i - 0.2, a * 1.02 if a else 0.004, f"{a:.3f}", ha="center", fontsize=8)
        ax.text(i + 0.2, b * 1.02 if b else 0.004, f"{b:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x, ["scientific units / mission", "zero-contribution ratio", "SCU / mission"])
    ax.set_ylim(0, max(max(v48), max(v96)) * 1.25)
    ax.set_ylabel("value")
    ax.set_title("Window-size stability of the meter (48 h vs 96 h)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    ev = load("golden_eval.json")
    rep48 = load("contribution_summary_48h.json")
    rep96 = load("contribution_summary_96h.json")
    fig_confusion(ev, os.path.join(OUT, "fig1_confusion.png"))
    fig_class_f1(ev, os.path.join(OUT, "fig2_class_f1.png"))
    fig_window(rep48, os.path.join(OUT, "fig3_window_units.png"))
    fig_conversion(rep48, os.path.join(OUT, "fig4_conversion.png"))
    fig_stability(rep48, rep96, os.path.join(OUT, "fig5_stability.png"))
    print("figures ->", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())