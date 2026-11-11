#!/usr/bin/env python3
"""Figure for the m15004 audit of m14679: convergence claims vs independent re-fit.

Numbers are read from runs/m15004_audit_m14679/checks.json (no hand copying).
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
CHECKS = os.path.join(ROOT, "runs", "m15004_audit_m14679", "checks.json")
OUT_DIR = os.path.join(ROOT, "artifacts", "m15004")
os.makedirs(OUT_DIR, exist_ok=True)

d = json.load(open(CHECKS))["C_convergence"]
rows = d["rows"]
h = np.array([r["h"] for r in rows])
avg = np.array([r["avg_l2_pointref"] for r in rows])
bnd = np.array([r["bnd_l2_pointref"] for r in rows])
bnd_e = np.array([r["bnd_l2_exactref"] for r in rows])
pollution = d["reference_point_eval_l2_vs_exact_cell_average"]
shipped = d["global_fit"]["shipped_levels_pointref"]
four = d["global_fit"]["four_levels_8_32_pointref"]
local_bnd = d["local_orders"]["bnd"]

fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))

ax[0].loglog(h, avg, "o-", label="average mode (1st order)")
ax[0].loglog(h, bnd, "s-", label="bounded mode (BJ limiter + repair)")
ax[0].loglog(h, bnd_e, "^--", ms=4, alpha=0.75,
             label="bounded vs near-exact cell-average reference")
ax[0].axhline(pollution, color="crimson", ls=":", lw=1.4,
              label=f"reference error floor = {pollution:.2e}")
ax[0].axvline(1 / 64, color="grey", ls="-.", lw=0.9)
ax[0].text(1 / 64 * 0.9, 3e-2, " n_src = n_tgt = 64\n (commensurate)", fontsize=7, color="grey")
for lv, mark in ((h[3], "8-32"), (h[4], "8-48")):
    pass
ax[0].set_xlabel("source spacing h = 1/n_src (target fixed at 64x64)")
ax[0].set_ylabel(r"$L_2$ error vs analytic field")
ax[0].set_title("m14679 convergence, audit re-run over denser level set")
ax[0].legend(fontsize=7, loc="lower right")
ax[0].grid(alpha=0.3, which="both")

idx = np.arange(len(local_bnd))
ax[1].plot(idx, local_bnd, "s-", color="tab:blue", label="local order, bounded mode")
ax[1].axhline(2.0, color="darkgreen", ls="--", lw=1.2, label="theoretical 2nd order")
ax[1].axhline(shipped["bnd"], color="tab:red", ls=":", lw=1.2,
              label=f"shipped global fit (8..48): {shipped['bnd']:.2f}")
ax[1].axhline(four["bnd"], color="tab:orange", ls=":", lw=1.2,
              label=f"global fit (8..32): {four['bnd']:.2f}")
ax[1].set_xticks(idx)
ax[1].set_xticklabels([f"{int(rows[i]['n_src'])}->{int(rows[i+1]['n_src'])}"
                       for i in idx], rotation=45, fontsize=7)
ax[1].set_xlabel("refinement pair (n_src -> n_src)")
ax[1].set_ylabel("observed local order")
ax[1].set_title("local order is non-monotone: 2.11 is not asymptotic")
ax[1].legend(fontsize=7)
ax[1].grid(alpha=0.3)

fig.tight_layout()
out = os.path.join(OUT_DIR, "m14679_convergence_audit.png")
fig.savefig(out, dpi=150)
print("wrote", out)