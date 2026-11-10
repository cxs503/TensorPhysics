"""Campaign driver: run the digital-twin UQ study and emit report + figures.

Usage
-----
``python -m tensorphysics.ice_digital_twin [--out DIR] [--fast]``

writes

``<out>/report.json``
    every number of the study (calibration, sensitivity, BMA, coverage, online
    timing, pass/fail per acceptance criterion, seeds, payload sha256),
``<out>/figures/*.png``
    the report figures, each with its sha256 recorded in
``<out>/figures/index.json``.

Every number in the figures is read from the report dict produced by
:func:`tensorphysics.ice_digital_twin.validation.run_digital_twin_uq`, i.e. the
figures cannot drift away from the audited numbers (no hand transcription).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

__all__ = [
    "build_figures",
    "run_and_write",
    "sha256_file",
    "attach_figures",
    "attach_pytest",
]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def build_figures(report: Mapping[str, Any], outdir: str, *, dpi: int = 140) -> List[Dict[str, str]]:
    """Render the report figures from the report payload (returns index entries)."""
    plt = _plt()
    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)
    figures: List[Dict[str, str]] = []

    def _save(fig, name: str, caption: str) -> None:
        path = os.path.join(figdir, name)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        figures.append({"name": name, "caption": caption, "sha256": sha256_file(path)})

    # ---- Fig 1: posterior marginals vs literature ------------------------
    cal = report["calibration"]
    refs = {row["name"]: row["reference"] for row in report["parameter_table"]}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, key in zip(axes, ("recovery", "inadequacy")):
        rows = [r for r in cal[key]["parameters"] if r["name"] in refs]
        names = [r["name"] for r in rows]
        y = np.arange(len(rows))[::-1]
        mean = np.array([r["mean"] for r in rows])
        lo = np.array([r["q05"] for r in rows])
        hi = np.array([r["q95"] for r in rows])
        ref = np.array([refs[n] for n in names])
        ax.errorbar(mean, y, xerr=[mean - lo, hi - mean], fmt="o", color="#1f4e79",
                    capsize=4, label="posterior mean, 90% credible interval")
        ax.plot(ref, y, "x", color="#c00000", markersize=9, label="literature reference")
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.set_title(f"{key} campaign (N={cal[key]['n_records']})")
        ax.set_xlabel("parameter value")
        ax.grid(alpha=0.3)
    axes[0].legend(loc="lower right", fontsize=8)
    sup = (
        f"split-R̂ max {cal['recovery']['max_rhat_split']:.3f}, "
        f"min ESS {cal['recovery']['min_ess']:.0f} "
        f"({cal['criteria']['n_chains']} chains x {cal['criteria']['n_steps_per_chain']} steps)"
    )
    fig.suptitle("Fig. 1  Bayesian calibration of the ice-resistance twin — " + sup, fontsize=10)
    _save(fig, "fig1_posterior_calibration.png",
          "Posterior mean and 90% credible interval of the four twin coefficients "
          "against their literature reference values, for the recovery campaign "
          "(truth = literature) and the model-inadequacy campaign (truth shifted).")

    # ---- Fig 2: convergence diagnostics ---------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    rows = cal["recovery"]["parameters"]
    names = [r["name"] for r in rows]
    x = np.arange(len(rows))
    axes[0].bar(x, [r["ess"] for r in rows], color="#2e75b6")
    axes[0].axhline(1000, color="#c00000", ls="--", lw=1.2, label="acceptance threshold 1000")
    axes[0].axhline(cal["criteria"]["min_ess_at_10000_step_budget"], color="grey", ls=":",
                    lw=1.0, label="ESS at the 10000-step budget")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=30, ha="right")
    axes[0].set_ylabel("effective sample size (summed over chains)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].bar(x, [r["rhat_split"] for r in rows], color="#548235")
    axes[1].axhline(1.1, color="#c00000", ls="--", lw=1.2, label="R̂ = 1.1 limit")
    axes[1].set_ylim(0.99, max(1.12, 1.02 * max(r["rhat_split"] for r in rows)))
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=30, ha="right")
    axes[1].set_ylabel("split R̂")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.suptitle("Fig. 2  MCMC convergence of the twin calibration", fontsize=10)
    _save(fig, "fig2_convergence_diagnostics.png",
          "Per-parameter effective sample size and split-R̂ of the production chains; "
          "both acceptance thresholds are drawn.")

    # ---- Fig 3: Sobol' sensitivity --------------------------------------
    sen = report["sensitivity"]["twin"]
    names = list(sen["first"])
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    w = 0.38
    axes[0].bar(x - w / 2, [sen["first"][n] for n in names], w, label="first order $S_i$",
                color="#2e75b6")
    axes[0].bar(x + w / 2, [sen["total"][n] for n in names], w, label="total $S_{T,i}$",
                color="#ed7d31")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=30, ha="right")
    axes[0].set_ylabel("share of output variance")
    axes[0].set_title(
        f"Sobol' indices at {sen['reference']['label']} "
        f"($\\sum S_T$ = {sen['sum_total']:.3f})"
    )
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    clo = report["sensitivity"]["closure"]
    per = clo["per_parameter"]
    pn = list(per)
    xx = np.arange(len(pn))
    axes[1].bar(xx - w / 2, [per[n]["exact"] for n in pn], w, label="exact (analytic)",
                color="#7f7f7f")
    axes[1].bar(xx + w / 2, [per[n]["total"] for n in pn], w, label="estimated $S_{T,i}$",
                color="#ffc000")
    axes[1].set_xticks(xx)
    axes[1].set_xticklabels(pn, rotation=30, ha="right")
    axes[1].set_title(
        f"Additive submodel closure: $\\sum S_T$ = {clo['sum_total_estimate']:.3f} "
        f"(error {clo['sum_total_error_vs_one']*100:.2f}%)"
    )
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.suptitle("Fig. 3  Global sensitivity of the twin resistance", fontsize=10)
    _save(fig, "fig3_sobol_sensitivity.png",
          "First/total Sobol' indices of the twin resistance at the reference ice "
          "condition, and the analytic-closure check on the additive submodel.")

    # ---- Fig 4: coverage of the 95% prediction intervals ----------------
    cov = report["coverage"]
    labels = ["fixed literature\n(no calibration)", "full form\n(calibrated)", "BMA\n(model-averaged)"]
    rates = [
        cov["baseline_uncalibrated"]["coverage"],
        cov["per_form_loo"]["full"]["coverage"],
        cov["bma_loo"]["coverage"],
    ]
    widths = [
        cov["baseline_uncalibrated"]["mean_pi_width_kN"],
        cov["per_form_loo"]["full"]["mean_pi_width_kN"],
        cov["bma_loo"]["mean_pi_width_kN"],
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    xx = np.arange(len(labels))
    axes[0].bar(xx, rates, color=["#a6a6a6", "#2e75b6", "#548235"])
    axes[0].axhspan(0.90, 1.0, color="#c6e0b4", alpha=0.5, label="acceptance band [0.90, 1.00]")
    axes[0].set_xticks(xx)
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel(f"LOO coverage of the 95% interval (N = {cov['folds']})")
    for i, r in enumerate(rates):
        axes[0].text(i, r + 0.02, f"{r*100:.0f}%", ha="center", fontsize=9)
    axes[0].legend(fontsize=8, loc="lower right")
    axes[0].grid(alpha=0.3)
    axes[1].bar(xx, widths, color=["#a6a6a6", "#2e75b6", "#548235"])
    axes[1].set_xticks(xx)
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylabel("mean 95% PI width [kN]")
    axes[1].grid(alpha=0.3)
    folds = cov.get("folds_bma") or []
    if folds:
        obs = np.array([f["observed_kN"] for f in folds])
        mu = np.array([f["pred_mean_kN"] for f in folds])
        lo = np.array([f["pi_lower_kN"] for f in folds])
        hi = np.array([f["pi_upper_kN"] for f in folds])
        order = np.argsort(obs)
        fig2, ax2 = plt.subplots(figsize=(8.5, 4.2))
        idx = np.arange(len(obs))
        ax2.errorbar(idx, mu[order], yerr=[mu[order] - lo[order], hi[order] - mu[order]],
                     fmt="o", ms=4, color="#1f4e79", ecolor="#8faadc", capsize=2,
                     label="BMA LOO prediction, 95% PI")
        inside = (obs[order] >= lo[order]) & (obs[order] <= hi[order])
        ax2.plot(idx[inside], obs[order][inside], "o", color="#2e7d32", ms=5,
                 label="observation inside PI")
        ax2.plot(idx[~inside], obs[order][~inside], "x", color="#c00000", ms=8,
                 label="observation outside PI")
        ax2.set_yscale("log")
        ax2.set_xlabel("held-out record (sorted by observed resistance)")
        ax2.set_ylabel("ice resistance [kN] (log scale)")
        ax2.set_title(
            f"Fig. 4b  LOO coverage of the model-averaged 95% interval: "
            f"{cov['bma_loo']['n_inside']}/{cov['bma_loo']['n_records']} inside",
            fontsize=10,
        )
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3, which="both")
        _save(fig2, "fig4b_loo_predictions.png",
              "Held-out BMA prediction intervals against the observations, with the "
              "misses marked; log resistance axis to span the 0.5–8 MN range.")
    fig.suptitle("Fig. 4  Prediction-interval coverage validation (leave-one-out)", fontsize=10)
    _save(fig, "fig4_coverage_validation.png",
          "LOO coverage and width of the 95% prediction interval for the uncalibrated "
          "law, the calibrated full form and the model-averaged predictor.")

    # ---- Fig 5: BMA weights and width comparison ------------------------
    bma = cov["bma_full_data"]
    keys = sorted(bma["weights"])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    xx = np.arange(len(keys))
    axes[0].bar(xx, [bma["weights"][k] for k in keys], color="#7030a0")
    axes[0].set_xticks(xx)
    axes[0].set_xticklabels([f"{k}\n(w={bma['weights'][k]:.3f})" for k in keys], fontsize=8)
    axes[0].set_ylabel("pseudo-BMA weight")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("model-form weights from LOO predictive scores")
    axes[0].grid(alpha=0.3)
    widths = [bma["members"][k]["pi_mean_width_kN"] for k in keys]
    axes[1].bar(xx, widths, color="#8faadc", label="single form")
    axes[1].axhline(bma["bma"]["pi_mean_width_kN"], color="#c00000", ls="--",
                    label=f"BMA ({bma['bma']['pi_mean_width_kN']:.0f} kN)")
    axes[1].axhline(bma["comparison"]["mean_single_pi_width_kN"], color="grey", ls=":",
                    label=f"mean single ({bma['comparison']['mean_single_pi_width_kN']:.0f} kN)")
    axes[1].set_xticks(xx)
    axes[1].set_xticklabels(keys, fontsize=8)
    axes[1].set_ylabel("mean 95% PI width [kN]")
    axes[1].legend(fontsize=8)
    axes[1].set_title(
        f"BMA PI is {bma['comparison']['narrower_fraction_vs_mean_single']*100:.0f}% narrower "
        "than the mean single model"
    )
    axes[1].grid(alpha=0.3)
    fig.suptitle("Fig. 5  Model-form uncertainty (Bayesian model averaging)", fontsize=10)
    _save(fig, "fig5_bma_model_form.png",
          "Pseudo-BMA weights of the four structural members and the resulting "
          "prediction-interval width compared with the individual members.")

    # ---- Fig 6: online calibration --------------------------------------
    online = report["online"]
    traj = online["mean_trajectory"]
    names = [n for n in traj if n in refs]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    steps = np.arange(1, online["n_steps"] + 1)
    for n in names:
        axes[0].plot(steps, traj[n], lw=1.2, label=n)
        axes[0].axhline(report["calibration"]["inadequacy"]["parameters"][
            [p["name"] for p in report["calibration"]["inadequacy"]["parameters"]].index(n)
        ]["mean"], ls=":", lw=1.0, color=axes[0].lines[-1].get_color())
    axes[0].set_xlabel("record folded in (increasing ice thickness)")
    axes[0].set_ylabel("online posterior mean")
    axes[0].set_title("streaming posterior mean (dotted: batch MCMC mean)")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)
    times = online.get("time_series_s") or []
    if times:
        axes[1].plot(np.arange(1, len(times) + 1), np.array(times) * 1e3, "o-", ms=4,
                     color="#2e75b6")
        axes[1].axhline(1000.0, color="#c00000", ls="--", label="1 s budget")
        axes[1].set_yscale("log")
        axes[1].set_xlabel("update step")
        axes[1].set_ylabel("wall time per update [ms]")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3, which="both")
    fig.suptitle("Fig. 6  Online (streaming) re-calibration of the twin", fontsize=10)
    _save(fig, "fig6_online_calibration.png",
          "Left: streaming posterior mean as records are folded in, against the batch "
          "MCMC mean; right: measured wall time per update against the 1 s budget.")

    index = {"figures": figures}
    with open(os.path.join(figdir, "index.json"), "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2)
        fh.write("\n")
    return figures


def attach_figures(report: Dict[str, Any], outdir: str) -> Dict[str, Any]:
    """Add the figure index (name/caption/sha256) to the report payload."""
    index_path = os.path.join(outdir, "figures", "index.json")
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as fh:
            report["figures"] = json.load(fh)["figures"]
    return report


def attach_pytest(report: Dict[str, Any], log_path: str) -> Dict[str, Any]:
    """Record the acceptance-test evidence (criterion g) from a pytest log."""
    if not os.path.exists(log_path):
        return report
    import re

    with open(log_path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    match = re.search(r"(\d+) passed", text)
    if match is None:
        report.setdefault("verification", {})["pytest"] = {
            "log": log_path,
            "status": "no-pass-count-found",
        }
        return report
    report.setdefault("criteria", {})["g_pytest_passed"] = int(match.group(1))
    report.setdefault("verification", {})["pytest"] = {
        "n_passed": int(match.group(1)),
        "failed": int(m.group(1)) if (m := re.search(r"(\d+) failed", text)) else 0,
        "command": "python -m pytest tests/test_p223w3_n09_digital_twin_uq.py -q",
        "log": log_path,
        "log_sha256": sha256_file(log_path),
    }
    return report


def run_and_write(
    outdir: str,
    *,
    fast: bool = False,
    refit_loo_check: bool = False,
) -> Dict[str, Any]:
    """Run the campaign, write report.json + figures, and return the report."""
    from .validation import run_digital_twin_uq, write_report

    if fast:
        report = run_digital_twin_uq(
            n_calib_steps=4000, calib_burn=1000, bma_steps=1500, bma_burn=500,
            loo_steps=1200, loo_burn=600, sobol_n_base=1 << 9, online_particles=150,
            n_records=32,
        )
    else:
        report = run_digital_twin_uq(refit_loo_check=refit_loo_check)
    os.makedirs(outdir, exist_ok=True)
    write_report(report, os.path.join(outdir, "report.json"))
    figures = build_figures(report, outdir)
    attach_figures(report, outdir)
    attach_pytest(report, os.path.join(outdir, "pytest.log"))
    # re-write so report.json carries the figure hashes and the test evidence
    write_report(report, os.path.join(outdir, "report.json"))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="artifacts/p223w3_n09_digital_twin_uq",
        help="output directory for report.json and figures/",
    )
    parser.add_argument("--fast", action="store_true", help="reduced sampling budget")
    parser.add_argument(
        "--refit-loo-check",
        action="store_true",
        help="also run cold-start fold refits to test the warm-start LOO",
    )
    args = parser.parse_args(argv)
    report = run_and_write(args.out, fast=args.fast, refit_loo_check=args.refit_loo_check)
    crit = report["criteria"]
    print(json.dumps({k: crit[k] for k in sorted(crit)}, indent=2, default=float))
    print(f"\nreport -> {os.path.join(args.out, 'report.json')}")
    print(f"figures -> {os.path.join(args.out, 'figures')}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())