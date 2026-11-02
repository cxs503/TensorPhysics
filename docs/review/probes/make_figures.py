import sys, math
sys.path.insert(0, "src")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorphysics.physics_kernels_resistance import (
    shared_core_resistance, axisymmetric_wetted_surface, ittc_1957_cf, schoenherr_1932_cf)
from tensorphysics import physics_contracts as pc

c = pc.get_contract("resistance.suboff_shared_core")
d = [r for r in c.references if r.provenance == "DERIVED"][0]
e = [r for r in c.references if r.provenance == "EXPERIMENTAL"][0]
L, B, cb, V, rho, nu = 4.356, 0.508, 0.542, 3.0, 998.2, 1.004e-6
S = axisymmetric_wetted_surface(L, B, cb)
base = shared_core_resistance(L, B, cb, V, rho, nu, S)

# ---------- F1: anchor band map ----------
fig, ax = plt.subplots(figsize=(9, 4.2))
lo, hi = 0.0029, 0.0044
ax.axvspan(e.value*(1-e.tolerance_rel), e.value*(1+e.tolerance_rel), color="#9fd3a0", alpha=.55,
           label=f"EXPERIMENTAL 0.0038 ±15% = [{e.value*0.85:.5f}, {e.value*1.15:.5f}]")
ax.axvspan(e.value*0.97, e.value*1.03, color="#2e7d32", alpha=.45, label="3% discipline zone around 0.0038")
ax.axvline(d.value, color="#c62828", lw=2,
           label=f"DERIVED anchor {d.value:.9f} (tol {d.tolerance_rel:g} — pixel-width line)")
ax.axvline(base.ct, color="#1565c0", ls="--", lw=2,
           label=f"reference kernel ct = {base.ct:.9f}")
ax.annotate(f"kernel vs EXPERIMENTAL\nrel err = {abs(base.ct-e.value)/e.value*100:.2f}%",
            xy=(base.ct, 0.55), xytext=(0.00316, 0.62),
            arrowprops=dict(arrowstyle="->", color="#333"), fontsize=9)
ax.set_xlim(lo, hi); ax.set_ylim(0, 1)
ax.set_yticks([]); ax.set_xlabel("total resistance coefficient $C_t$ [-]")
ax.set_title("SUBOFF-01 anchor map: DERIVED self-consistency anchor vs EXPERIMENTAL band\n"
             "(reference kernel reproduces the DERIVED value to 1.3e-16 — tautological by construction)")
ax.legend(loc="upper left", fontsize=8); ax.grid(axis="x", alpha=.3)
fig.tight_layout(); fig.savefig("docs/review/figures/fig1_suboff01_anchor_map.png", dpi=150)

# ---------- F2: Cb sensitivity vs anchor tolerance ----------
cbs = np.linspace(0.40, 0.85, 200)
cts = np.array([shared_core_resistance(L, B, float(k), V, rho, nu, S).ct for k in cbs])
cb_box = 0.7008 / (L*B*B)          # published displaced volume 0.7008 m^3, box base L*B*T
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
axs[0].plot(cbs, cts, color="#1565c0")
for k, lab, col in ((cb, "contract Cb=0.542 (anchor input)", "#c62828"),
                    (cb_box, f"Cb={cb_box:.4f} from published\nvolume 0.7008 m$^3$ (box base)", "#ef6c00")):
    axs[0].axvline(k, color=col, ls="--")
    axs[0].plot([k], [shared_core_resistance(L, B, k, V, rho, nu, S).ct], "o", color=col, label=lab)
axs[0].set_xlabel("block coefficient $C_b$ [-]"); axs[0].set_ylabel("$C_t$ [-]")
axs[0].set_title("$C_t$ sensitivity to the unvalidated geometry input $C_b$")
axs[0].legend(fontsize=8); axs[0].grid(alpha=.3)

rel = np.abs(cts - base.ct) / base.ct
axs[1].semilogy(cbs, rel, color="#6a1b9a", label="|Δ$C_t$|/$C_t$ vs contract $C_b$=0.542")
axs[1].axhline(d.tolerance_rel, color="#c62828", ls="--", label=f"DERIVED anchor tol {d.tolerance_rel:g}")
axs[1].axhline(0.03, color="#2e7d32", ls=":", label="3% platform discipline")
axs[1].axhline(0.15, color="#e65100", ls="-.", label="15% acceptance ceiling")
axs[1].set_xlabel("block coefficient $C_b$ [-]"); axs[1].set_ylabel("relative deviation [-]")
axs[1].set_title("Anchor tolerance (1e-9) is ~6 orders tighter\nthan the geometry uncertainty it depends on")
axs[1].legend(fontsize=7.5); axs[1].grid(alpha=.3, which="both")
fig.tight_layout(); fig.savefig("docs/review/figures/fig2_cb_sensitivity.png", dpi=150)

# ---------- F3: ITTC-1957 vs Schoenherr independence ----------
res = np.logspace(5.5, 9.5, 200)
relg = (np.array([ittc_1957_cf(float(r)) for r in res]) -
        np.array([schoenherr_1932_cf(float(r)) for r in res])) / \
        np.array([ittc_1957_cf(float(r)) for r in res])
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.semilogx(res, relg*100, color="#00695c", label="(Cf_ITTC1957 - Cf_Schoenherr)/Cf_ITTC1957")
ax.axvline(base.reynolds, color="#c62828", ls="--",
           label=f"SUBOFF-01 Re = {base.reynolds:.3e}")
ax.axhline(0, color="k", lw=.8)
ax.set_xlabel("Re [-]"); ax.set_ylabel("relative gap [%]")
ax.set_title("Reference-kernel independence is real: ITTC-1957 vs Schoenherr are\n"
             "distinct correlations (gap shrinks with Re, crossing near Re≈6e8)")
ax.legend(fontsize=8); ax.grid(alpha=.3, which="both")
fig.tight_layout(); fig.savefig("docs/review/figures/fig3_ittc_vs_schoenherr.png", dpi=150)

print("ct_base", base.ct, "ct_cbbox", shared_core_resistance(L, B, cb_box, V, rho, nu, S).ct,
      "rel", abs(shared_core_resistance(L, B, cb_box, V, rho, nu, S).ct - base.ct)/base.ct,
      "cb_box", cb_box, "S", S, "gap_at_Re", ittc_1957_cf(base.reynolds)-schoenherr_1932_cf(base.reynolds))
