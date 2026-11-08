#!/usr/bin/env python3
"""Generate P27W2-N10 integration-acceptance figures (programmatic, zero
hand-typed numbers: everything is read from the gate artifact bundle)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "runs/p27w2_n10"
OUT = BUNDLE / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 140,
})

NODE_POS = {
    961: (0.0, 2.0), 962: (1.0, 2.0), 963: (2.0, 2.0),
    964: (0.0, 1.0), 965: (1.0, 1.0),
    966: (0.0, 0.0), 967: (1.0, 0.0),
    968: (2.0, 1.0), 969: (3.0, 1.0),
    970: (4.0, 1.0),
}
NODE_LABEL = {
    961: "N01 BEM\nradiation/\ndiffraction",
    962: "N02 seakeeping\nRAO",
    963: "N03 roll damping\nzeta_eq",
    964: "N04 spectral\nfatigue",
    965: "N05 buoyancy-\ncoupled plate",
    966: "N06 DEM rolling\nfriction",
    967: "N07 DEM\nnon-spherical",
    968: "N08 turbulence\nu+ UQ",
    969: "N09 AMR\nbackflow",
    970: "N10 integration\nacceptance",
}
COLOR = {
    "done": "#2e7d32", "in-flight": "#f9a825", "failed": "#c62828",
    "honest-fail": "#6a1b9a",
}


def fig1_dag() -> Path:
    nodes = json.loads((BUNDLE / "campaign_nodes.json").read_text())
    verdict = json.loads((BUNDLE / "gate_verdict.json").read_text())
    dag_ev = next(c for c in verdict["criteria"] if c["criterion_id"] == "a")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for n in nodes:
        nid = n["id"]
        x, y = NODE_POS[nid]
        color = COLOR.get(n["status"], "#9e9e9e")
        ax.add_patch(plt.Rectangle((x - 0.38, y - 0.28), 0.76, 0.56,
                                   facecolor=color, alpha=0.85,
                                   edgecolor="black", lw=0.8, zorder=3))
        ax.text(x, y, NODE_LABEL[nid], ha="center", va="center",
                fontsize=6.8, zorder=4, color="white"
                if n["status"] != "in-flight" else "black")
    for n in nodes:
        for dep in (n.get("dependencies") or []):
            x1, y1 = NODE_POS[dep]
            x2, y2 = NODE_POS[n["id"]]
            ax.annotate("", xy=(x2 - 0.38, y2), xytext=(x1 + 0.38, y1),
                      arrowprops=dict(arrowstyle="-|>", color="#455a64",
                                      lw=1.1), zorder=2)
    ax.set_xlim(-0.7, 4.8)
    ax.set_ylim(-0.6, 2.7)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"P27-W2 campaign DAG (plan 153) -- acyclic: "
        f"{dag_ev['evidence']['acyclic']}, terminal "
        f"{dag_ev['evidence']['done_count']}/"
        f"{dag_ev['evidence']['total_count']}, blocked "
        f"{dag_ev['evidence']['blocked_nodes']}",
        fontsize=9,
    )
    fig.tight_layout()
    p = OUT / "fig1_dag_convergence.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def fig2_contracts() -> Path:
    results = json.loads((BUNDLE / "contract_results.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.4))
    titles = ["C1 BEM->ship RAO", "C2 DEM->Physics granular",
              "C3 LBM->platform GCI/UQ"]
    for ax, res, title in zip(axes, results, titles):
        checks = res["checks"]
        ok = [c["ok"] for c in checks]
        labels = [c["check_id"].split(".", 1)[1] for c in checks]
        colors = ["#2e7d32" if o else "#c62828" for o in ok]
        ax.barh(range(len(checks)), [1] * len(checks), color=colors,
                height=0.7)
        ax.set_yticks(range(len(checks)))
        ax.set_yticklabels(labels, fontsize=6.5)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.set_title(f"{title}\n{sum(ok)}/{len(ok)} checks pass",
                     fontsize=8.5)
    fig.suptitle("P27-W2 cross-repo interface contracts (real node data)",
                 fontsize=10)
    fig.tight_layout()
    p = OUT / "fig2_contract_checks.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def fig3_evidence_capability() -> Path:
    ledger = json.loads((BUNDLE / "evidence_ledger.json").read_text())
    caps = json.loads((BUNDLE / "capability_ledger.json").read_text())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.6))
    entries = ledger["entries"]
    labels = [f"N{e['node_id'] - 960}" for e in entries]
    complete = [1 if e["complete"] else 0 for e in entries]
    ax1.bar(range(len(entries)), complete,
             color=["#2e7d32" if c else "#f9a825" for c in complete])
    ax1.set_xticks(range(len(entries)))
    ax1.set_xticklabels(labels, fontsize=7)
    ax1.set_ylim(0, 1.4)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(["incomplete", "complete"])
    ax1.set_title(f"Evidence ledger completeness "
                  f"({ledger['completeness_rate']:.0%})", fontsize=9)
    caps = caps["capabilities"]
    levels = [c["evidence_level"] for c in caps]
    names = [c["capability"][:18] for c in caps]
    cmap = {"REAL": "#2e7d32", "WITHHELD": "#f9a825",
             "MOCK": "#c62828", "STUB": "#c62828"}
    ax2.barh(range(len(caps)), [1] * len(caps),
              color=[cmap.get(l, "#9e9e9e") for l in levels], height=0.7)
    ax2.set_yticks(range(len(caps)))
    ax2.set_yticklabels(names, fontsize=6.5)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 1)
    ax2.set_xticks([])
    n_real = levels.count("REAL")
    ax2.set_title(f"Capability evidence levels "
                  f"({n_real}/{len(caps)} REAL)", fontsize=9)
    fig.tight_layout()
    p = OUT / "fig3_evidence_capability.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


if __name__ == "__main__":
    paths = [fig1_dag(), fig2_contracts(), fig3_evidence_capability()]
    for p in paths:
        print(p)
