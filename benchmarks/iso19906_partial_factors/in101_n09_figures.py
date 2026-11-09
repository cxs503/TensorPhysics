"""Figures for the IN101-N09 target-reliability inverse report.

Every figure is produced from the study JSON alone -- no numbers are typed in
here, so a rerun with different parameters regenerates consistent plots.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tensorphysics.ice_partial_factors.p_area_joint import PAreaForceModel  # noqa: E402


def _save(fig, out_dir: str, name: str, caption: str, figures: list) -> None:
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figures.append((path, caption))


def fig_pressure_area(results: dict, out_dir: str, figures: list) -> None:
    """p-A law with the marginal pressure/area distributions and the correlation."""
    model = PAreaForceModel(**results["model"])
    rng = np.random.default_rng(20261109)
    n = 4000
    z_p = rng.standard_normal(n)
    z_a = rng.standard_normal(n)
    rho_ln = model.pressure_area_correlation()
    z_p_corr = rho_ln * z_a + np.sqrt(max(1.0 - rho_ln**2, 0.0)) * z_p
    area = model.a_median_m2 * np.exp(model.sigma_a * z_a)
    press_corr = model.median_pressure_mpa * np.exp(model.sigma_p * z_p_corr)
    press_indep = model.median_pressure_mpa * np.exp(model.sigma_p * z_p)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    ax = axes[0]
    grid = np.logspace(np.log10(0.6), np.log10(60.0), 200)
    ax.loglog(grid, model.p_ref_mpa * (grid / model.a_ref_m2) ** (-model.exponent), "k-",
              label=r"ISO law $p=p_0(A/A_0)^{-ex}$, ex=%.2f" % model.exponent)
    ax.loglog(area, press_corr, "o", ms=2, alpha=0.25, color="C0",
              label=r"with $\rho_{\ln p,\ln A}$ = %.3f" % rho_ln)
    ax.loglog(area, press_indep, "o", ms=2, alpha=0.25, color="C3",
              label=r"independent ($\rho$=0, same marginals)")
    ax.set_xlabel("contact area A [m$^2$]")
    ax.set_ylabel("ice pressure p [MPa]")
    ax.set_title("joint p-A state (median force %.2f MN)" % results["force_statistics"]["median_force_mn"])
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1]
    fs_c = results["force_statistics"]
    fs_i = results["force_statistics_independent"]
    f = np.linspace(1.0, 70.0, 400)
    for mom, style, label, colour in (
        (fs_c, "-", "correlated force", "C0"),
        (fs_i, "--", "independent force", "C3"),
    ):
        mu, sig = mom["mu_ln_force"], mom["sigma_ln_force"]
        pdf = np.exp(-((np.log(f) - mu) ** 2) / (2 * sig**2)) / (f * sig * np.sqrt(2 * np.pi))
        ax.plot(f, pdf, style, color=colour,
                label="%s: $\\sigma_{\\ln F}$=%.3f, CoV=%.2f" % (label, sig, mom["cov_force"]))
    ax.axvline(results["characteristic_values"]["F_k_mn"], color="k", ls=":", lw=1,
               label="$F_k$ (100-yr) = %.2f MN" % results["characteristic_values"]["F_k_mn"])
    ax.set_xlabel("annual extreme ice force F [MN]")
    ax.set_ylabel("probability density")
    ax.set_title("annual extreme force: effect of the p-A correlation")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    _save(fig, out_dir, "fig1_pa_joint.png",
          "Joint p-A state: both correlation hypotheses keep identical marginals and differ only "
          "in $\\rho_{\\ln p,\\ln A}$; the force density shows the resulting scatter inflation.",
          figures)


def fig_beta_curve(results: dict, out_dir: str, figures: list) -> None:
    """Achieved beta versus the factor product, with every calibration route."""
    curve = results["beta_curve"]
    ks = [r["kappa"] for r in curve]
    betas = [r["beta"] for r in curve]
    sol = results["inverse"]["solution"]
    cc = results["crosscheck"]
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.plot(ks, betas, "-", color="C0", label=r"FORM $\beta(\kappa)$, $\kappa=\gamma_f\psi$")
    ax.axhline(results["study"]["target_beta"], color="C3", ls="--",
               label=r"$\beta_T$ = %.2f" % results["study"]["target_beta"])
    ax.plot([sol["kappa_star"]], [sol["beta_achieved"]], "o", ms=9, color="C3",
            label=r"Newton solution $\kappa^*$ = %.4f" % sol["kappa_star"])
    ax.plot([results["kappa_crosscheck"]["kappa_star"]], [results["kappa_crosscheck"]["beta_achieved"]],
            "x", ms=9, color="k", label="bisection cross-check on $\\kappa$")
    ax.axvline(sol["kappa_star"], color="C3", lw=0.8, ls=":")
    ax.errorbar([sol["kappa_star"]], [cc["monte_carlo"]["beta"]],
                yerr=[[cc["monte_carlo"]["beta"] - cc["monte_carlo"]["beta_one_sigma_band"][0]],
                      [cc["monte_carlo"]["beta_one_sigma_band"][1] - cc["monte_carlo"]["beta"]]],
                fmt="s", ms=6, color="C2", capsize=4,
                label=r"crude MC $\pm1\sigma$ (%.1e samples)" % cc["monte_carlo"]["samples"])
    ax.errorbar([sol["kappa_star"]], [cc["conditional_monte_carlo"]["beta"]],
                yerr=[[cc["conditional_monte_carlo"]["beta"]
                       - cc["conditional_monte_carlo"]["beta_one_sigma_band"][0]],
                      [cc["conditional_monte_carlo"]["beta_one_sigma_band"][1]
                       - cc["conditional_monte_carlo"]["beta"]]],
                fmt="^", ms=6, color="C4", capsize=4,
                label="crude MC, resistance integrated (Rao-Blackwell)")
    ax.plot([sol["kappa_star"]], [cc["importance_sampling"]["beta"]], "D", ms=6, color="C5",
            label=r"design-point IS ($\pm$%.1e of Pf; band too tight to be visible)"
            % cc["importance_sampling"]["cov_pf"])
    ax.set_xlabel(r"factor product $\kappa$")
    ax.set_ylabel(r"achieved annual reliability index $\beta$")
    ax.set_title("code-calibration inverse: FORM curve, target and four independent checks")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="upper left")
    ax.set_ylim(1.0, 7.0)
    _save(fig, out_dir, "fig2_beta_vs_kappa.png",
          "Achieved reliability index versus the factor product: the Newton solution, the "
          "independent bisection on $\\kappa$, and the FORM/IS/MC/Rao-Blackwell estimates of the "
          "same point.", figures)


def fig_newton(results: dict, out_dir: str, figures: list) -> None:
    """Newton convergence and the residual system behind the uniqueness claim."""
    runs = results["inverse"]["runs"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    ax = axes[0]
    for run in runs:
        trace = run["trace"]
        it = [t["iteration"] for t in trace]
        res = [max(abs(t["residual_beta"]), abs(t["residual_split"]), 1e-16) for t in trace]
        ax.semilogy(it, res, "o-", ms=3,
                    label=r"start ($\gamma_f$,$\psi$)=(%.2f,%.2f)" % tuple(run["start"]))
    ax.axhline(1e-8, color="k", ls=":", lw=1, label="beta residual tolerance 1e-8")
    ax.set_xlabel("Newton iteration")
    ax.set_ylabel("max |residual|")
    ax.set_title("quadratic convergence from three dispersed starts")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1]
    from tensorphysics.ice_partial_factors.target_reliability_inverse import design_split

    sol = results["inverse"]["solution"]
    model = PAreaForceModel(**results["model"])
    kappas = np.linspace(0.9, 2.2, 40)
    pairs = [design_split(model, float(k)) for k in kappas]
    ax.plot(kappas, [p["gamma_f"] for p in pairs], "-", color="C0",
            label=r"$\gamma_f(\kappa)=F^*/F_k$ (design-value split)")
    ax.plot(kappas, [p["psi"] for p in pairs], "-", color="C1",
            label=r"$\psi(\kappa)=\kappa/\gamma_f$")
    ax.plot([sol["kappa_star"]], [sol["gamma_f"]], "o", ms=8, color="C0",
            label=r"$\gamma_f^*$ = %.4f" % sol["gamma_f"])
    ax.plot([sol["kappa_star"]], [sol["psi"]], "s", ms=7, color="C1",
            label=r"$\psi^*$ = %.4f" % sol["psi"])
    ax.axvline(sol["kappa_star"], color="k", ls=":", lw=0.8)
    ax.set_xlabel(r"$\kappa=\gamma_f\psi$")
    ax.set_ylabel("factor value")
    ax.set_title("split convention; Jacobian cond = %.2f"
                 % sol["jacobian_condition_number"])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    _save(fig, out_dir, "fig3_newton.png",
          "Left: residual norm of the two-equation system from three dispersed starting points "
          "(quadratic convergence, same root). Right: the design-value split curve that makes the "
          "second equation non-degenerate.", figures)


def fig_iso_deviations(results: dict, out_dir: str, figures: list) -> None:
    """Deviation map against the ISO 19906 table pairs."""
    rows = results["iso_deviations"]["rows"]
    labels = [r["id"].replace(" x ", "\nx ").replace(" ", "\n", 1) for r in rows]
    dev_g = [100.0 * r["deviation_gamma_f_rel"] for r in rows]
    dev_p = [100.0 * r["deviation_psi_rel"] for r in rows]
    dev_k = [100.0 * r["deviation_kappa_rel"] for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 6.6), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1.2]})
    ax = axes[0]
    w = 0.27
    ax.bar(x - w, dev_g, w, label=r"$\gamma_f$ deviation", color="C0")
    ax.bar(x, dev_p, w, label=r"$\psi$ deviation", color="C1")
    ax.bar(x + w, dev_k, w, label=r"$\kappa=\gamma_f\psi$ deviation", color="C2")
    ax.axhspan(-15.0, 15.0, color="0.85", zorder=0, label="15 % acceptance band")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_ylabel("deviation of table value from calibration [%]")
    ax.set_title("ISO 19906 A.2 x A.3 pairs versus the calibrated pair "
                 "(gamma_f=%.4f, psi=%.4f)" % (results["inverse"]["solution"]["gamma_f"],
                                               results["inverse"]["solution"]["psi"]))
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1]
    beta_ach = []
    for row in rows:
        if row["non_conservative_example"]:
            beta_ach.append(row["non_conservative_example"]["beta_achieved"])
        else:
            beta_ach.append(np.nan)
    ax.bar(x, beta_ach, 0.55, color=["C3" if (r["non_conservative_example"]) else "0.7" for r in rows])
    ax.axhline(results["study"]["target_beta"], color="k", ls="--",
               label=r"$\beta_T$ = %.2f" % results["study"]["target_beta"])
    ax.set_ylabel("achieved $\\beta$\n(non-conservative rows)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.5, rotation=90)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=7)
    _save(fig, out_dir, "fig4_iso_deviations.png",
          "Deviation map of every ISO 19906 load/resistance factor pair against the calibrated "
          "pair, with the achieved reliability index of the rows that fall more than 15 % short.",
          figures)


def fig_negative_control(results: dict, out_dir: str, figures: list) -> None:
    """The negative control: calibrated pair with and without the p-A correlation."""
    nc = results["negative_control"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    ax = axes[0]
    names = ["correlated\n(physical)", "independent\n($\\rho$=0 control)"]
    gammas = [nc["correlated"]["gamma_f"], nc["independent"]["gamma_f"]]
    psis = [nc["correlated"]["psi"], nc["independent"]["psi"]]
    x = np.arange(2)
    ax.bar(x - 0.18, gammas, 0.36, color="C0", label=r"$\gamma_f^*$")
    ax.bar(x + 0.18, psis, 0.36, color="C1", label=r"$\psi^*$")
    for i, (g, p) in enumerate(zip(gammas, psis)):
        ax.text(i - 0.18, g, "%.4f" % g, ha="center", va="bottom", fontsize=8)
        ax.text(i + 0.18, p, "%.4f" % p, ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("calibrated factor")
    ax.set_title(r"negative control: $\Delta\gamma_f$ = %+.1f %% (threshold 5 %%)"
                 % (100.0 * nc["delta_gamma_f_rel"]))
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    model = PAreaForceModel(**results["model"])
    fig_ax = ax
    corr = [nc["correlated"]["sigma_ln_force"], nc["independent"]["sigma_ln_force"]]
    covs = [nc["correlated"]["cov_force"], nc["independent"]["cov_force"]]
    x = np.arange(2)
    fig_ax.bar(x - 0.18, corr, 0.36, color="C0", label=r"$\sigma_{\ln F}$")
    fig_ax.bar(x + 0.18, covs, 0.36, color="C4", label="force CoV")
    for i, (a, b) in enumerate(zip(corr, covs)):
        fig_ax.text(i - 0.18, a, "%.3f" % a, ha="center", va="bottom", fontsize=8)
        fig_ax.text(i + 0.18, b, "%.3f" % b, ha="center", va="bottom", fontsize=8)
    fig_ax.set_xticks(x)
    fig_ax.set_xticklabels(names)
    fig_ax.set_title("load scatter inflation %.1f %% ($\\rho_{\\ln p,\\ln A}$=%.3f)"
                     % (100.0 * nc["sigma_ln_force_inflation_rel"], model.pressure_area_correlation()))
    fig_ax.grid(True, axis="y", alpha=0.3)
    fig_ax.legend(fontsize=8)
    _save(fig, out_dir, "fig5_negative_control.png",
          "Negative control. Dropping the negative p-A correlation inflates the load scatter and "
          "the calibrated load factor; the calibrated member is then over-designed by the reported "
          "beta margin when applied to the physical model.", figures)


def fig_sensitivity(results: dict, out_dir: str, figures: list) -> None:
    """Adjoint versus finite-difference sensitivities and the beta_T sweep."""
    sens = results["sensitivity"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    ax = axes[0]
    keys = ["d_beta_d_kappa", "d_beta_d_gamma_f", "d_beta_d_psi"]
    labels = [r"$\partial\beta/\partial\kappa$", r"$\partial\beta/\partial\gamma_f$",
              r"$\partial\beta/\partial\psi$"]
    adj = [sens[k]["adjoint"] for k in keys]
    fd = [sens[k]["finite_difference"] for k in keys]
    x = np.arange(len(keys))
    ax.bar(x - 0.18, adj, 0.36, color="C0", label="adjoint (envelope theorem)")
    ax.bar(x + 0.18, fd, 0.36, color="C3", label="central finite difference")
    for i, (a, b) in enumerate(zip(adj, fd)):
        ax.text(i, max(a, b) * 1.01, "%.5f / %.5f" % (a, b), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"$\partial\beta/\partial(\cdot)$")
    ax.set_ylim(0.0, max(adj + fd) * 1.18)
    ax.set_title("adjoint vs finite difference: max rel. error %.1e" % sens["max_relative_error"])
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    sweep = results["beta_target_sweep"]
    bt = [r["beta_target"] for r in sweep]
    ax.plot(bt, [r["gamma_f"] for r in sweep], "o-", color="C0", label=r"$\gamma_f^*(\beta_T)$")
    ax.plot(bt, [r["psi"] for r in sweep], "s-", color="C1", label=r"$\psi^*(\beta_T)$")
    ax.plot(bt, [r["kappa_star"] for r in sweep], "^-", color="C2", label=r"$\kappa^*(\beta_T)$")
    ax.axvline(results["study"]["target_beta"], color="k", ls=":", lw=1, label=r"mission $\beta_T$")
    ax.set_xlabel(r"target reliability index $\beta_T$")
    ax.set_ylabel("calibrated factor")
    ax.set_title("target-reliability sensitivity of the calibrated pair")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, out_dir, "fig6_sensitivity.png",
          "Left: adjoint (envelope-theorem) derivatives against central finite differences. Right: "
          "the calibrated pair as a function of the target reliability index.", figures)


def make_all(results: dict, out_dir: str) -> list:
    os.makedirs(out_dir, exist_ok=True)
    figures: list = []
    fig_pressure_area(results, out_dir, figures)
    fig_beta_curve(results, out_dir, figures)
    fig_newton(results, out_dir, figures)
    fig_iso_deviations(results, out_dir, figures)
    fig_negative_control(results, out_dir, figures)
    fig_sensitivity(results, out_dir, figures)
    return figures