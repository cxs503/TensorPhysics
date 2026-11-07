#!/usr/bin/env python3
"""Figures for the P16-G1 campaign integration gate report.

All figures are generated from ``artifacts/p16_gate/gate_verdict.json`` -- the
image content is therefore a rendering of the recorded numbers, never a hand
drawn summary.

    python3 scripts/p16_gate_figures.py [--verdict ...] [--out artifacts/p16_gate]
"""
from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

STATUS_COLOR = {"PASS": "#2e7d32", "FAIL": "#c62828", "WITHHELD": "#9e9e9e"}
STATUS_VALUE = {"PASS": 2.0, "WITHHELD": 1.0, "FAIL": 0.0}


def _criteria(verdict, cid):
    return next(c for c in verdict["criteria"] if c["criterion_id"] == cid)


def fig_node_matrix(verdict, path):
    """(a) node x criterion acceptance matrix."""
    crit_a = _criteria(verdict, "a")
    nodes = crit_a["evidence"]["nodes"]
    maxc = max(len(n["criteria"]) for n in nodes)
    mat = np.full((len(nodes), maxc), np.nan)
    labels = [["" for _ in range(maxc)] for _ in nodes]
    for i, n in enumerate(nodes):
        for j, c in enumerate(n["criteria"]):
            mat[i, j] = STATUS_VALUE[c["status"]]
            labels[i][j] = c["criterion_id"]
    fig, ax = plt.subplots(figsize=(maxc * 0.62 + 2.2, len(nodes) * 0.5 + 1.6))
    cmap = matplotlib.colors.ListedColormap([STATUS_COLOR["FAIL"], STATUS_COLOR["WITHHELD"],
                                             STATUS_COLOR["PASS"]])
    ax.imshow(mat, cmap=cmap, vmin=0, vmax=2, aspect="auto")
    for i in range(len(nodes)):
        for j in range(maxc):
            if np.isnan(mat[i, j]):
                continue
            ax.text(j, i, labels[i][j], ha="center", va="center", color="white",
                    fontsize=7, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks(range(len(nodes)))
    ax.set_yticklabels([f"{n['ext_id']}  ({n['verdict']})" for n in nodes], fontsize=8)
    ax.set_title("(a) P16 N1-N8 acceptance criteria — no delivered artifact, no pass\n"
                 "green=PASS  grey=WITHHELD  red=FAIL", fontsize=9)
    ax.set_xlabel("acceptance criteria of each node (C1, C2, …)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=88)
    plt.close(fig)


def fig_regression(verdict, path):
    """(c) P15 -> P16 regression: measured pass rate and |delta| vs the 1% band."""
    crit_c = _criteria(verdict, "c")
    cases = crit_c["evidence"]["cases"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.0, 6.2), sharex=True,
                                   gridspec_kw={"height_ratios": [2.2, 1.4]})
    x = np.arange(len(cases))
    for i, c in enumerate(cases):
        col = STATUS_COLOR[c["status"]]
        rate = c["rate_now"]
        if rate is None:
            ax1.plot([i], [0.0], marker="x", color=col, markersize=9, mew=2)
            ax1.annotate("no probe", (i, 0.02), ha="center", fontsize=6.5, color=col)
        else:
            ax1.plot([i, i], [c["rate_p15"], rate], color=col, lw=1.2, alpha=0.7)
            ax1.plot([i], [c["rate_p15"]], marker="_", color="#37474f", markersize=12)
            ax1.plot([i], [rate], marker="o", color=col, markersize=6)
    ax1.set_ylabel("probe pass rate", fontsize=8.5)
    ax1.set_ylim(-0.03, 1.06)
    ax1.axhline(1.0, color="#37474f", lw=0.8, ls=":")
    ax1.set_title("(c) P15 regression baseline re-executed on the P16 code base\n"
                  "black dash = P15 recorded rate, dot = measured rate", fontsize=9)
    ax1.grid(alpha=0.25)
    ax1.tick_params(labelsize=8)

    deltas = [0.0 if c["delta"] is None else abs(c["delta"]) * 100.0 for c in cases]
    colors = [STATUS_COLOR[c["status"]] for c in cases]
    ax2.bar(x, deltas, color=colors, width=0.62)
    ax2.axhline(1.0, color="#c62828", lw=1.0, ls="--", label="criterion |Δ| < 1%")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{c['ext_id']}\n{c['repo'] or ''}" for c in cases],
                        fontsize=6.5, rotation=0)
    ax2.set_ylabel("|Δ| pass rate  [%]", fontsize=8.5)
    ax2.legend(fontsize=7.5, loc="upper left")
    ax2.tick_params(labelsize=7.5)
    ax2.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=88)
    plt.close(fig)


def fig_contracts(verdict, path):
    """(b) cross-domain contract checks as a pass/fail matrix."""
    crit_b = _criteria(verdict, "b")
    contracts = crit_b["evidence"]["contracts"]
    rows = [(f"{c['contract_id']}", chk["check_id"], chk["status"])
            for c in contracts for chk in c["checks"]]
    ids = sorted({r[1].split("_")[0] for r in rows},
                 key=lambda k: (k[0], int("".join(ch for ch in k if ch.isdigit()) or 0)))
    fig, ax = plt.subplots(figsize=(9.4, 0.34 * len(rows) + 1.9))
    for i, (ct, chk, st) in enumerate(rows):
        ax.barh(i, 1.0, color=STATUS_COLOR[st], alpha=0.85, height=0.72)
        ax.text(0.02, i, chk, va="center", ha="left", color="white", fontsize=7.2,
                fontweight="bold")
        ax.text(0.99, i, st, va="center", ha="right", color="white", fontsize=7.0)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.2)
    ax.set_xticks([])
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_title("(b) cross-domain interface contract checks\n"
                 "wind table (LBM→ship-platform) · wet-modal handoff (FEM→BEM) · consumer endpoints",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=88)
    plt.close(fig)


def fig_topology(verdict, path):
    """Data-flow topology of the two P16 cross-domain couplings."""
    crit_b = _criteria(verdict, "b")
    status = {c["contract_id"]: c["verdict"] for c in crit_b["evidence"]["contracts"]}
    fig, ax = plt.subplots(figsize=(9.2, 3.5))
    ax.set_axis_off()

    def box(x, y, w, h, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                    fc=color, ec="#263238", lw=0.9, alpha=0.9))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.8,
                color="white", wrap=True)

    def arrow(x1, y1, x2, y2, label):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                                     lw=1.1, color="#37474f"))
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.06, label, ha="center", fontsize=6.8,
                color="#37474f")

    def col(st):
        return STATUS_COLOR[st]

    box(0.02, 0.62, 0.20, 0.24, "TensorLBM N7\nwind coefficients\nC_X,C_Y,C_N(β)", "#1565c0")
    box(0.32, 0.62, 0.22, 0.24,
        f"contract wind_table_v1\n{status.get('wind_table_v1', 'WITHHELD')}",
        col(status.get("wind_table_v1", "WITHHELD")))
    box(0.66, 0.62, 0.30, 0.24,
        "ship-platform\n/calculations/wind_resistance\n"
        f"{status.get('wind_endpoint_v1', 'WITHHELD')}",
        col(status.get("wind_endpoint_v1", "WITHHELD")))
    arrow(0.22, 0.74, 0.32, 0.74, "table JSON")
    arrow(0.54, 0.74, 0.66, 0.74, "HTTP POST")

    box(0.02, 0.14, 0.20, 0.24, "TensorFEM N8\nwet modes f, ω\nadded mass", "#1565c0")
    box(0.32, 0.14, 0.22, 0.24,
        f"contract wet_modal_v1\n{status.get('wet_modal_v1', 'WITHHELD')}",
        col(status.get("wet_modal_v1", "WITHHELD")))
    box(0.66, 0.14, 0.30, 0.24,
        "TensorBEM\nsolve_helmholtz_diffraction\nk = ω/c",
        col(status.get("wet_modal_v1", "WITHHELD")))
    arrow(0.22, 0.26, 0.32, 0.26, "modal handoff")
    arrow(0.54, 0.26, 0.66, 0.26, "k [1/m]")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("P16 cross-domain coupling topology and contract verdicts", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(path, dpi=88)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verdict", default="artifacts/p16_gate/gate_verdict.json")
    ap.add_argument("--out", default="artifacts/p16_gate")
    args = ap.parse_args()
    verdict = json.loads(pathlib.Path(args.verdict).read_text())
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    made = {}
    for name, fn in (("fig1_node_acceptance_matrix.png", fig_node_matrix),
                     ("fig2_p15_regression.png", fig_regression),
                     ("fig3_contract_checks.png", fig_contracts),
                     ("fig4_coupling_topology.png", fig_topology)):
        p = out / name
        fn(verdict, p)
        made[name] = p.stat().st_size
    print(json.dumps(made, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())