"""Figures for the measured-data UQ study (node p223w2b-n03).

Every number is read from ``artifacts/p223w2b_n03_measured_uq/report.json``
(no hand-typed values); each figure is written as PNG with its sha256 printed so
the report can cite a digest.
"""
from __future__ import annotations

import hashlib
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join("artifacts", "p223w2b_n03_measured_uq")


def _sha(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def fig1_baseline(rep, figdir):
    base = rep["baseline"]
    y = np.array(base["observed_kN"])
    pred = np.array(base["predicted_kN"])
    rc, rb, rs = (np.array(base["component_share"][k]) for k in ("crush", "break", "submersion"))
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.4))
    ax[0].loglog(y, pred, "o", ms=7, label="twin, literature coefficients")
    lim = [80, 5000]
    ax[0].plot(lim, lim, "k--", lw=1, label="1:1")
    ax[0].fill_between(lim, [v / 1.15 for v in lim], [v * 1.15 for v in lim],
                       color="0.8", alpha=0.5, label=r"$\pm$15% band")
    pub = rep["baseline"]["published_ratio"]
    for k, r in pub.items():
        ax[0].axhline(0, alpha=0)  # keep autoscale
    ax[0].set_xlim(lim); ax[0].set_ylim(lim)
    ax[0].set_xlabel("measured resistance [kN]")
    ax[0].set_ylabel("predicted resistance [kN]")
    ax[0].set_title(f"literature twin vs measured (median ratio "
                    f"{base['median_ratio_model_over_measured']:.2f})")
    ax[0].legend(fontsize=8)
    order = np.argsort(y)
    idx = np.arange(len(y))
    ax[1].bar(idx - 0.27, rc[order], width=0.27, label="crush")
    ax[1].bar(idx, rb[order], width=0.27, label="break")
    ax[1].bar(idx + 0.27, rs[order], width=0.27, label="submersion")
    ax[1].set_xticks(idx)
    ax[1].set_xticklabels([f"{v:.0f}" for v in y[order]], fontsize=7, rotation=60)
    ax[1].set_xlabel("measured resistance [kN] (points sorted)")
    ax[1].set_ylabel("share of predicted resistance [-]")
    ax[1].set_title(f"component share (median break share {base['median_break_share']:.2f})")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(figdir, "fig1_baseline_vs_measured.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig2_posterior(rep, figdir):
    params = rep["calibration"]["summary"]["parameters"]
    names = [n for n in rep["calibration"]["summary"]["names"]]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6))
    axes = axes.ravel()
    for ax, name in zip(axes, names):
        rec = params[name]
        mean, lo, hi = rec["mean"], rec["ci_low"], rec["ci_high"]
        ax.barh([0], [hi - lo], left=[lo], height=0.5, color="#9ecae1")
        ax.plot([mean], [0], "k|", ms=16)
        if "literature" in rec:
            ax.plot([rec["literature"]], [0], "r*", ms=12, label="literature")
            ax.set_title(f"{name}\nrel.err={rec['relative_error']*100:.1f}%", fontsize=9)
        else:
            ax.set_title(f"{name}\n(mean {mean:.2f})", fontsize=9)
        ax.set_yticks([])
        ax.tick_params(labelsize=7)
    axes[-1].axis("off")
    axes[-1].text(0.0, 0.7, f"split-$\\hat R$ max = {rep['calibration']['summary']['rhat_max']:.3f}\n"
                            f"ESS min = {rep['calibration']['summary']['ess_min']}\n"
                            f"discrepancy amplitude\n$\\exp(\\beta_0)$ = "
                            f"{np.exp(params['beta0']['mean']):.3f}",
                  fontsize=9, va="top")
    fig.suptitle("posterior marginals, 95% credible intervals (measured campaign, n=10)", fontsize=11)
    fig.tight_layout()
    path = os.path.join(figdir, "fig2_posterior_marginals.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig3_coverage(rep, figdir):
    loo = rep["coverage"]["loo"]
    nc = rep["coverage"]["negative_control"]
    rows = sorted(loo["rows"], key=lambda r: r["observed_kN"])
    x = np.arange(len(rows))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for i, r in enumerate(rows):
        col = "#2ca02c" if r["inside"] else "#d62728"
        ax[0].plot([i, i], [r["pi_low_kN"], r["pi_high_kN"]], color=col, lw=4, solid_capstyle="butt")
        ax[0].plot([i], [r["pred_mean_kN"]], "o", color=col, ms=5)
        ax[0].plot([i], [r["observed_kN"]], "k*", ms=9)
    ax[0].plot([], [], "k*", ms=9, label="measured")
    ax[0].plot([], [], "o", color="0.2", ms=5, label="LOO predictive mean")
    ax[0].plot([], [], "-", color="#2ca02c", lw=4, label="95% PI, covered")
    ax[0].plot([], [], "-", color="#d62728", lw=4, label="95% PI, missed")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([r["point_id"] for r in rows], fontsize=7)
    ax[0].set_ylabel("resistance [kN]")
    ax[0].set_title(f"LOO coverage = {loo['inside']}/{loo['n_points']} "
                    f"({loo['coverage']*100:.0f}%), mean width {loo['mean_pi_width_kN']:.0f} kN")
    ax[0].legend(fontsize=8, loc="upper left")
    rows_nc = sorted(nc["rows"], key=lambda r: r["observed_kN"])
    for i, r in enumerate(rows_nc):
        col = "#2ca02c" if r["inside"] else "#d62728"
        ax[1].plot([i, i], [r["pi_low_kN"], r["pi_high_kN"]], color=col, lw=4, solid_capstyle="butt")
        ax[1].plot([i], [r["predicted_kN"]], "o", color=col, ms=5)
        ax[1].plot([i], [r["observed_kN"]], "k*", ms=9)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([r["point_id"] for r in rows_nc], fontsize=7)
    ax[1].set_ylabel("resistance [kN]")
    ax[1].set_title(f"negative control (no discrepancy): coverage "
                    f"{nc['inside']}/{nc['n_points']} ({nc['coverage']*100:.0f}%)")
    fig.tight_layout()
    path = os.path.join(figdir, "fig3_coverage.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig4_sobol(rep, figdir):
    ex = rep["sobol"]["exact"]
    mc = rep["sobol"]["monte_carlo"]
    labels = [n.replace("_", " ") for n in rep["sobol"]["reference_condition"]["inputs"]]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.bar(x - 0.2, ex["S_i"], width=0.4, label="first order $S_i$ (exact quadrature)")
    ax.bar(x + 0.2, ex["S_Ti"], width=0.4, label="total order $S_{Ti}$ (exact quadrature)")
    ax.plot(x - 0.2, mc["S_i"], "k+", ms=10, label="MC (Saltelli/Jansen)")
    ax.plot(x + 0.2, mc["S_Ti"], "kx", ms=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Sobol' index [-]")
    ax.set_title(f"closure: $\\sum S_{{Ti}}$={ex['sum_S_Ti']:.4f} (exact), "
                 f"{mc['sum_S_Ti']:.4f} (MC); interaction {ex['interaction_share']:.4f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(figdir, "fig4_sobol.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig5_bma(rep, figdir):
    bma = rep["bma"]
    keys = sorted(bma["weights"])
    w = [bma["weights"][k] for k in keys]
    widths = [bma["members"][k]["mean_pi_width_kN"] for k in keys]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].barh(keys, w, color="#4c72b0")
    ax[0].set_xlabel("pseudo-BMA weight [-]")
    ax[0].set_title(f"weights (sum = {bma['weights_sum']:.6f})")
    labs = keys + ["BMA", "single twin"]
    vals = widths + [bma["bma_mean_pi_width_kN"],
                     bma.get("reference_single_model_width_kN", float("nan"))]
    colors = ["#4c72b0"] * len(keys) + ["#55a868", "#c44e52"]
    ax[1].barh(labs, vals, color=colors)
    for i, v in enumerate(vals):
        ax[1].text(v, i, f" {v:.0f}", va="center", fontsize=8)
    ratio = bma.get("width_ratio_vs_reference", float("nan"))
    ax[1].set_xlabel("mean 95% predictive-interval width [kN]")
    ax[1].set_title(f"BMA / single twin width = {ratio:.3f} "
                    f"(narrowing {(1-ratio)*100:.1f}%, target $\\geq$20%)")
    fig.tight_layout()
    path = os.path.join(figdir, "fig5_bma.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> int:
    with open(os.path.join(OUT, "report.json"), encoding="utf-8") as fh:
        rep = json.load(fh)
    figdir = os.path.join(OUT, "figures")
    os.makedirs(figdir, exist_ok=True)
    made = [fig1_baseline(rep, figdir), fig2_posterior(rep, figdir),
            fig3_coverage(rep, figdir), fig4_sobol(rep, figdir), fig5_bma(rep, figdir)]
    manifest = {os.path.basename(p): {"sha256": _sha(p), "bytes": os.path.getsize(p)} for p in made}
    with open(os.path.join(figdir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)
    for name, rec in manifest.items():
        print(f"{name}: sha256={rec['sha256'][:16]}... {rec['bytes']} B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
