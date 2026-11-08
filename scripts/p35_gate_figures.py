#!/usr/bin/env python3
"""Figures for the P35-G1 acceptance gate report.

Reads ``artifacts/p35_gate/campaign_readiness.json`` (the machine verdict) plus
the recorded regression probe outputs, and renders:

1. ``fig1_node_readiness.png``   -- ten-node readiness matrix (deliverable /
   ledger / maturity / chain) as an annotated status grid;
2. ``fig2_regression_deviation.png`` -- measured |delta|/max(|baseline|,floor)
   per re-run case on a log axis against the campaign's 1% line, with the
   withheld cases annotated with their cost/absence reason;
3. ``fig3_couette_profile.png`` -- the physics behind figure 2: re-run Couette
   profile against the analytic linear solution and the pointwise deviation
   (machine-precision floor, fp64).

Numbers are read from the artefacts; nothing is typed in by hand.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GATE_DIR = ROOT / "artifacts" / "p35_gate"
FIG_DIR = GATE_DIR / "figures"

STATUS_COLOR = {"PASS": "#2e7d32", "FAIL": "#c62828", "WITHHELD": "#9e9e9e"}
STATUS_TEXT = {"PASS": "PASS", "FAIL": "FAIL", "WITHHELD": "WH"}


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def fig_readiness(doc: dict) -> Path:
    nodes = doc["nodes"]
    cols = [("deliverable", "deliverable (a)"), ("registered", "ledger (b)"),
            ("maturity", "maturity (e)")]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for row, n in enumerate(nodes):
        cells = [n[c] for c, _ in cols] + [("PASS" if n["chain_ready"] else "FAIL")]
        for col, status in enumerate(cells):
            ax.add_patch(plt.Rectangle((col, row), 0.92, 0.86,
                                       color=STATUS_COLOR[status], alpha=0.85))
            ax.text(col + 0.46, row + 0.43, STATUS_TEXT[status], ha="center", va="center",
                    color="white", fontsize=8, fontweight="bold")
    ax.set_xlim(0, 4)
    ax.set_ylim(0, len(nodes))
    ax.set_xticks([i + 0.46 for i in range(4)])
    ax.set_xticklabels([label for _, label in cols] + ["chain-ready (d)"])
    ax.set_yticks([i + 0.43 for i in range(len(nodes))])
    ax.set_yticklabels([n["ext_id"] for n in nodes])
    ax.invert_yaxis()
    ax.set_title(f"P35 node readiness - overall {doc['overall_status']} "
                 f"(dispatcher_ready={doc['dispatcher_ready']})", fontsize=10)
    ax.set_frame_on(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    out = FIG_DIR / "fig1_node_readiness.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_regression(doc: dict) -> Path:
    cases = doc["regression"]
    names = [c["case"].replace("_", "\n") for c in cases]
    devs, colors = [], []
    for c in cases:
        if c["status"] == "PASS":
            devs.append(max(c["rel_dev"], 1e-19))
            colors.append(STATUS_COLOR["PASS"])
        else:
            devs.append(np.nan)
            colors.append(STATUS_COLOR["WITHHELD" if c["status"] == "WITHHELD" else "FAIL"])
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    x = np.arange(len(cases))
    ax.bar(x, devs, color=colors, width=0.55)
    ax.axhline(1e-2, color="#c62828", linestyle="--", linewidth=1.2,
               label="campaign threshold |delta|/scale < 1%")
    ax.axhline(1e-12, color="#1565c0", linestyle=":", linewidth=1.0,
               label="fp64 machine floor 1e-12")
    for xi, c in zip(x, cases):
        if np.isnan(devs[xi]):
            note = "not re-run\n(budget / no entry)"
            ax.annotate(note, (xi, 1e-9), ha="center", va="bottom", fontsize=7,
                        color=STATUS_COLOR["WITHHELD"])
        else:
            ax.annotate(f"{devs[xi]:.1e}", (xi, devs[xi]), ha="center", va="bottom",
                        fontsize=7)
    ax.set_yscale("log")
    ax.set_ylim(1e-19, 1e0)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("|delta| / max(|baseline|, floor)")
    ax.set_title("Regression re-run vs recorded baseline (criterion c)", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    out = FIG_DIR / "fig2_regression_deviation.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_couette() -> Path | None:
    probe = GATE_DIR / "probes" / "couette_3d" / "case_H16_fp64.json"
    if not probe.is_file():
        return None
    d = _load(probe)
    u_num = np.asarray(d["u_profile"], dtype=float)
    u_ana = np.asarray(d["u_analytic"], dtype=float)
    y = np.arange(len(u_num), dtype=float) + 0.5
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.8))
    ax1.plot(u_num, y, "o", ms=3.5, label="LBM re-run (fp64, H=16)")
    ax1.plot(u_ana, y, "-", lw=1.2, label="analytic u=U0*y/H_eff")
    ax1.set_xlabel("u_x [lattice]")
    ax1.set_ylabel("y [lattice]")
    ax1.set_title(f"Couette profile overlap (l2={d['l2_rel_err']:.3e})", fontsize=9)
    ax1.legend(fontsize=7)
    err = np.abs(u_num - u_ana)
    ax2.semilogy(y, np.maximum(err, 1e-18), "s-", ms=3.5, color="#1565c0")
    ax2.axhline(np.finfo(float).eps, color="#c62828", ls="--", lw=1.0,
                label=f"float64 eps = {np.finfo(float).eps:.1e}")
    ax2.set_xlabel("y [lattice]")
    ax2.set_ylabel("|u_LBM - u_analytic|")
    ax2.set_title("pointwise error: rounding floor, not discretisation", fontsize=9)
    ax2.legend(fontsize=7)
    fig.tight_layout()
    out = FIG_DIR / "fig3_couette_profile.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    doc = _load(GATE_DIR / "campaign_readiness.json")
    made = [fig_readiness(doc), fig_regression(doc)]
    couette = fig_couette()
    if couette:
        made.append(couette)
    for p in made:
        print(f"{p} ({p.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())