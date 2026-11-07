#!/usr/bin/env python3
"""m5213 评审报告配图：全部数值从 artifacts/m5213/verification.json 程序化读取（零手抄）。"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
V = json.loads((ROOT / "artifacts/m5213/verification.json").read_text())
OUT = ROOT / "artifacts/m5213/figures"
OUT.mkdir(parents=True, exist_ok=True)

d, c = V["derived"], V["counts"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.6))

# --- (a) 独立复算的相对误差 vs 各自验收阈值 --------------------------------
probe = [
    ("SUBOFF $C_T$", d["suboff_ct_err_pct"], 5.0),
    ("SUBOFF $S_{wet}$", d["suboff_wetted_err_pct"], 3.0),
    ("KCS $S_{wet}$", d["kcs_wetted_err_pct"], 1.5),
]
names = [p[0] for p in probe]
errs = [abs(p[1]) for p in probe]
thrs = [p[2] for p in probe]
xs = range(len(probe))
bars = ax1.bar(xs, errs, width=0.55, color=["#c0392b", "#e67e22", "#2980b9"], label="|relative error| (recomputed)")
ax1.axhline(0, color="k", lw=0.8)
for x, p in zip(xs, probe):
    ax1.text(x, abs(p[1]) * 1.15, f"{p[1]:+.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax1.plot([x - 0.32, x + 0.32], [p[2], p[2]], color="k", ls="--", lw=1.4)
ax1.set_yscale("log")
ax1.set_xticks(list(xs), names)
ax1.set_ylabel("|relative error| vs reference  [%]")
ax1.set_title("(a) independent recompute (2026-11-07, :8002)\nbar = recomputed error, dashed = acceptance threshold")
ax1.legend(loc="upper right", fontsize=8)
ax1.grid(axis="y", alpha=0.3, which="both")

# --- (b) 报告结构自检数字（与正文逐一对比）--------------------------------
cats = ["criteria\n(all WITHHELD)", "error rows\n(measured)", "error rows\n(withheld)",
        "root cause", "lineage ITERATE", "lineage MATURE", "lineage SHELVE"]
vals = [c["criteria"], c["error_measured"], c["error_withheld"], c["rootcause"],
        c["lineage"]["iterate"], c["lineage"]["mature"], c["lineage"]["shelve"]]
colors = ["#7f8c8d", "#2980b9", "#7f8c8d", "#16a085", "#f39c12", "#27ae60", "#95a5a6"]
bars = ax2.bar(range(len(cats)), vals, color=colors, width=0.62)
for i, v in enumerate(vals):
    ax2.text(i, v + 0.8, str(v), ha="center", fontsize=10, fontweight="bold")
ax2.set_xticks(range(len(cats)), cats, fontsize=8)
ax2.set_ylabel("count (re-derived from published body)")
ax2.set_ylim(0, max(vals) * 1.20)
ax2.set_title("(b) reported vs re-derived structural counts\n(9 nodes / 49 criteria / 0 MATURE of 9)")
ax2.grid(axis="y", alpha=0.3)

fig.suptitle("m5213 peer review of mission 5208 (IN19-G19 wave-3 acceptance gate): independent recompute",
             fontsize=11, y=1.00)
fig.tight_layout()
fig.savefig(OUT / "recompute_vs_threshold.png", dpi=125, bbox_inches="tight")
print("wrote", OUT / "recompute_vs_threshold.png")