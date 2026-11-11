#!/usr/bin/env python3
"""Figure for the m15005 audit verdict: evidence strength vs maturity rung.

Left panel  — every quantitative check the audit re-derived, on a log axis, with the
              3 % (acceptance) and 15 % (reject) thresholds drawn in.
Right panel — the cumulative maturity rungs L1..L5 with pass/fail from
              ``runs/m15005_audit_cross_clock/audit_checks.json``.

Reads only machine-produced JSON (no hand-typed numbers).
"""
from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "m15005_audit_cross_clock"
CHK = json.loads((OUT / "audit_checks.json").read_text())


def short(q: str) -> str:
    return (q.replace("tensorlbm ", "lbm ").replace("tensordem ", "dem ")
             .replace("coupled particle velocity vs exact ZOH", "coupled v vs ZOH")
             .replace("momentum bookkeeping residual", "momentum residual")
             .replace("Taylor-Green decay rate (measured/analytic 2*nu*k^2)", "Taylor-Green decay")
             .replace("Stokes relaxation v at t=0.2 s [m/s]", "Stokes relaxation"))


def main() -> int:
    rows = [r for r in CHK["benchmark"]["rows"] if isinstance(r["rel_error"], (int, float))]
    rows = [r for r in rows if r["rel_error"] > 0]
    labels = [short(r["quantity"]) for r in rows]
    vals = [max(r["rel_error"], 1e-18) for r in rows]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=150,
                                  gridspec_kw={"width_ratios": [1.35, 1.0]})
    y = range(len(vals))
    ax.barh(list(y), vals, color="#1f77b4", height=0.55)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.axvline(0.03, color="#2ca02c", ls="--", lw=1.2, label="3% acceptance")
    ax.axvline(0.15, color="#d62728", ls="--", lw=1.2, label="15% reject")
    for yi, v in zip(y, vals):
        ax.annotate(f"{v:.1e}", (v, yi), textcoords="offset points", xytext=(4, 0),
                    fontsize=7, va="center")
    ax.set_xlabel("relative error (log)")
    ax.set_title("m14792 numbers re-derived by m15005 (all far below 3%)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, axis="x", which="both")

    rung = CHK["level"]["rung_ok"]
    names = {"1": "L1 tests", "2": "L2 real impl + entry + run artifacts",
             "3": "L3 system integration (solver/platform caller)",
             "4": "L4 benchmark ref-vs-got + repro_cmd",
             "5": "L5 external literature/experimental anchor"}
    keys = ["1", "2", "3", "4", "5"]
    colors = ["#2ca02c" if rung[k] else "#d62728" for k in keys]
    ax2.barh(range(5), [1] * 5, color=colors, height=0.6)
    ax2.set_yticks(range(5))
    ax2.set_yticklabels([names[k] for k in keys], fontsize=8)
    ax2.invert_yaxis()
    for i, k in enumerate(keys):
        ax2.text(0.5, i, "PASS" if rung[k] else "FAIL", ha="center", va="center",
                 color="white", fontsize=9, fontweight="bold")
    ax2.set_xticks([])
    lvl = CHK["level"]["high_water_level"]
    ax2.set_title(f"maturity scheme (cumulative) → declared L{lvl}\n"
                  f"validation rungs pass, integration rung fails", fontsize=10)
    fig.suptitle("m15005 · GAP-CROSS-CLOCK (m14792) cross-clock binding audit", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    p = OUT / "figures" / "audit_evidence.png"
    fig.savefig(p)
    print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())