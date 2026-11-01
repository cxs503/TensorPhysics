#!/usr/bin/env python3
"""End-to-end artefact run: ice load spatial correlation -> hull girder extreme moment.

Produces (all under ``artifacts/hullgirder_ice_extremes``):

* ``acceptance.json``      - every acceptance criterion (a)-(f) with numbers,
                             tolerances and pass flags;
* ``arrays.npz``           - raw arrays behind the figures;
* ``fig1..fig7`` (PNG)     - space-time pressure contour, bending moment
                             envelope, correlation sensitivity, wave RAO,
                             extreme value fits, utilisation, criteria errors.

Usage::

    python examples/hullgirder_ice_bending_extreme.py [--n-mc 10000] [--n-sens 2000]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorphysics.hullgirder import (  # noqa: E402
    IceCondition,
    ShipGirder,
    contact_patch_length,
    correlation_length,
    design_wave_moment,
    exponential_covariance,
    gumbel_fit,
    gumbel_return_level,
    ice_bending_extremes,
    ittc_design_wave_moment,
    moment_batch,
    moment_statistics,
    space_time_field,
    wave_moment_rao,
    weibull_fit,
    weibull_return_level,
)
from tensorphysics.hullgirder.verification import (  # noqa: E402
    EVENTS_PER_YEAR,
    RETURN_PERIOD_YEARS,
    run_all,
)

PLOT_STYLE = {
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 9,
    "axes.titlesize": 10,
    "figure.autolayout": True,
}


def _save(fig, path: Path, caption: str) -> dict:
    fig.savefig(path)
    plt.close(fig)
    return {"name": path.stem, "path": path.name, "caption": caption}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-mc", type=int, default=10_000)
    parser.add_argument("--n-sens", type=int, default=2_000)
    parser.add_argument("--outdir", default="artifacts/hullgirder_ice_extremes")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    plt.rcParams.update(PLOT_STYLE)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ship = ShipGirder()
    condition = IceCondition(thickness=1.5, concentration=0.8, speed=2.0)
    lam = correlation_length(condition)
    x = ship.grid()
    patch = contact_patch_length(condition, lam)

    report = run_all(n_real_mc=args.n_mc, n_real_sensitivity=args.n_sens, verbose=not args.quiet)
    report["provenance"] = {
        "workflow": "examples/hullgirder_ice_bending_extreme.py",
        "correlation_length_m": lam,
        "contact_patch_length_m": patch,
        "events_per_year": EVENTS_PER_YEAR,
        "return_period_years": RETURN_PERIOD_YEARS,
    }

    # ------------------------------------------------------------- raw arrays
    mu = report["criteria"]["f"]["mean_ice_pressure_Pa"]
    sigma = 0.30 * mu
    tau = lam / condition.speed
    z_st = space_time_field(x, lam, tau, sigma=1.0, n_steps=240, dt=0.25, seed=7)
    p_st = np.clip(mu + sigma * z_st, 0.0, None)

    ic = report["criteria"]["f"]["ice"]
    mc = ice_bending_extremes(ship, condition, lam, n_real=args.n_mc, seed=41)
    m_max = mc["M_max_abs_Nm"]
    gumbel = gumbel_fit(m_max)
    weibull = weibull_fit(m_max)
    p_event = 1.0 / (RETURN_PERIOD_YEARS * EVENTS_PER_YEAR)
    m_design_g = gumbel_return_level(gumbel, p_event)
    m_design_w = weibull_return_level(weibull, p_event)

    mask = (np.abs(x - 0.0) <= 0.5 * patch).astype(float)
    q_field = p_st * (condition.thickness * mask[None, :])
    _, m_st = moment_batch(x, q_field, support="free")
    cov = exponential_covariance(x, lam, sigma=sigma * condition.thickness) * np.outer(mask, mask)
    std_profile, _ = moment_statistics(x, cov, support="free")
    _, m_mean = moment_batch(x, (p_st.mean(axis=0) * condition.thickness * mask)[None, :])
    m_mean = m_mean[0]

    rule = ittc_design_wave_moment(ship.length, ship.breadth, ship.block_coefficient)
    omega_grid = np.linspace(0.2, 1.6, 60)
    rao = np.array([abs(wave_moment_rao(ship, float(w))).max() for w in omega_grid])
    lam_wave = 2.0 * np.pi * 9.81 / omega_grid ** 2
    design = design_wave_moment(ship)

    np.savez_compressed(
        outdir / "arrays.npz",
        x=x,
        pressure_space_time=p_st,
        moment_space_time=m_st,
        moment_mean=m_mean,
        moment_std=std_profile,
        m_max_samples=m_max,
        omega=omega_grid,
        rao_max_moment=rao,
        wavelength_over_L=lam_wave / ship.length,
        correlations=np.array(
            [r["correlation_length_m"] for r in report["criteria"]["e"]["rows"]]
        ),
        m_max_mean=np.array([r["M_max_mean_Nm"] for r in report["criteria"]["e"]["rows"]]),
        m_max_q90=np.array([r["M_max_q90_Nm"] for r in report["criteria"]["e"]["rows"]]),
        analytic_std=np.array([r["analytic_std_peak_Nm"] for r in report["criteria"]["e"]["rows"]]),
    )

    figs = []

    # ---------------------------------------------------------- fig 1: p(x,t)
    fig, ax = plt.subplots(1, 1, figsize=(8.0, 4.2))
    extent = [x[0], x[-1], 0.0, p_st.shape[0] * 0.25]
    im = ax.imshow(
        p_st / 1e6, aspect="auto", origin="lower", extent=extent, cmap="viridis", vmin=0.0
    )
    ax.axvline(-0.5 * patch, color="w", ls="--", lw=0.8)
    ax.axvline(0.5 * patch, color="w", ls="--", lw=0.8)
    ax.set_xlabel("x [m] (AP .. FP)")
    ax.set_ylabel("time [s]")
    ax.set_title(
        rf"Correlated ice pressure $p(x,t)=\mu_p+\sigma_p Z(x,t)$, "
        rf"$\lambda_x={lam:.2f}$ m, $\lambda_t={tau:.2f}$ s (dashed: contact patch)"
    )
    fig.colorbar(im, ax=ax, label="p [MPa]")
    figs.append(_save(fig, outdir / "fig1_space_time_pressure.png",
                      "Space-time ice pressure field; dashed lines delimit the contact patch."))

    # -------------------------------------------------- fig 2: M(x) realisations
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    ax = axes[0]
    for row in m_st[:40] / 1e6:
        ax.plot(x, row, color="0.6", lw=0.6)
    ax.plot(x, m_mean / 1e6, "k-", lw=1.6, label=r"mean $M(x)$")
    ax.plot(x, (m_mean + 2 * std_profile) / 1e6, "C3--", lw=1.2, label=r"mean $\pm 2\sigma$")
    ax.plot(x, (m_mean - 2 * std_profile) / 1e6, "C3--", lw=1.2)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("M [MN·m]")
    ax.set_title(rf"Ice-induced bending moment, $\lambda_x={lam:.2f}$ m (patch {patch:.1f} m at midship)")
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot(x, std_profile / 1e6, "C0-", lw=1.5, label="analytic std (covariance propagation)")
    ax.scatter(x[::10], np.std(m_st[:, ::10], axis=0) / 1e6, s=6, color="C1",
               label=f"MC std ({m_st.shape[0]} snapshots)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel(r"$\sigma_M$ [MN·m]")
    ax.set_title("Moment standard deviation: Monte-Carlo vs analytic")
    ax.legend(fontsize=8)
    figs.append(_save(fig, outdir / "fig2_moment_envelope.png",
                      "Left: bending moment realisations and envelope. Right: MC vs analytic std."))

    # --------------------------------------------- fig 3: correlation sensitivity
    rows = report["criteria"]["e"]["rows"]
    lam_list = np.array([r["correlation_length_m"] for r in rows])
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.2))
    ax.semilogx(lam_list, np.array([r["M_max_mean_Nm"] for r in rows]) / 1e6, "o-",
                label=r"MC mean $M_{max}$")
    ax.semilogx(lam_list, np.array([r["M_max_q90_Nm"] for r in rows]) / 1e6, "s--",
                label=r"MC 90% quantile $M_{max}$")
    ax.semilogx(lam_list, np.array([r["analytic_std_peak_Nm"] for r in rows]) / 1e6, "^-",
                label=r"analytic $\sigma_M$ (peak)")
    ax.axvline(lam, color="0.4", ls=":", lw=1.0)
    ax.annotate(rf"design $\lambda_x={lam:.2f}$ m", (lam, ax.get_ylim()[0]), fontsize=8,
                rotation=90, va="bottom", ha="right")
    ax.set_xlabel(r"correlation length $\lambda_x$ [m]")
    ax.set_ylabel("[MN·m]")
    ax.set_title(rf"Correlation-length sensitivity (Spearman $\rho$={report['criteria']['e']['spearman_mc_mean']:.3f})")
    ax.legend(fontsize=8)
    figs.append(_save(fig, outdir / "fig3_correlation_sensitivity.png",
                      "Bending moment extreme statistics vs longitudinal correlation length."))

    # --------------------------------------------------------- fig 4: wave RAO
    fig, ax = plt.subplots(1, 1, figsize=(6.6, 4.2))
    ax.plot(lam_wave / ship.length, rao / 1e6, "C0-", lw=1.6,
            label="long-wave strip RAO, Smith correction")
    ax.axhline(rule["hogging_Nm"] / 1e6, color="C3", ls="--", lw=1.2,
               label="ITTC-1978 / IACS UR S11 hogging")
    ax.axhline(abs(rule["sagging_Nm"]) / 1e6, color="C3", ls=":", lw=1.2,
               label="ITTC-1978 / IACS UR S11 sagging")
    ax.plot([design["wavelength_m"] / ship.length], [design["M_max_Nm"] / 1e6], "k*", ms=14,
            label=rf"design wave $\lambda=L$, $h=L/20$: {design['M_max_Nm']/1e9:.3f} GN·m")
    ax.set_xlim(0.4, 3.0)
    ax.set_xlabel(r"$\lambda_w / L$")
    ax.set_ylabel("|M| amplitude [MN·m]")
    ax.set_title("Regular-wave bending moment vs recommended design value")
    ax.legend(fontsize=8)
    figs.append(_save(fig, outdir / "fig4_wave_rao.png",
                      "Regular-wave moment amplitude vs wavelength, compared with the ITTC/IACS design value."))

    # --------------------------------------------------- fig 5: extreme value fit
    fig, ax = plt.subplots(1, 1, figsize=(6.6, 4.2))
    ax.hist(m_max / 1e6, bins=60, density=True, alpha=0.45, color="C0",
            label=rf"MC $M_{{max}}$ (n={args.n_mc})")
    grid = np.linspace(m_max.min(), 1.6 * m_design_g, 400)
    g_pdf = (1.0 / gumbel["scale"]) * np.exp(
        -(grid - gumbel["loc"]) / gumbel["scale"]
        - np.exp(-(grid - gumbel["loc"]) / gumbel["scale"])
    )
    ax.plot(grid / 1e6, g_pdf * 1e6, "C1-", lw=1.5, label="Gumbel fit")
    k, s_w = weibull["shape"], weibull["scale"]
    w_pdf = (k / s_w) * (grid / s_w) ** (k - 1) * np.exp(-(grid / s_w) ** k)
    ax.plot(grid / 1e6, w_pdf * 1e6, "C2--", lw=1.5, label="Weibull fit")
    ax.axvline(m_design_g / 1e6, color="C1", ls=":", lw=1.4,
               label=rf"100-yr Gumbel {m_design_g/1e9:.3f} GN·m")
    ax.axvline(m_design_w / 1e6, color="C2", ls=":", lw=1.4,
               label=rf"100-yr Weibull {m_design_w/1e9:.3f} GN·m")
    ax.set_xlabel("$M_{max}$ [MN·m]")
    ax.set_ylabel("pdf [1/(MN·m)]")
    ax.set_title(
        "Ice bending moment extremes, per-event exceedance "
        rf"$p={p_event:.2e}$ (1/{RETURN_PERIOD_YEARS:.0f} y x {EVENTS_PER_YEAR:.0f} events/y)"
    )
    ax.legend(fontsize=8)
    figs.append(_save(fig, outdir / "fig5_extreme_fit.png",
                      "Monte-Carlo extremes of the ice bending moment with Gumbel/Weibull fits."))

    # ------------------------------------------------------- fig 6: utilisation
    comb = report["criteria"]["f"]["combination"]
    ult = report["criteria"]["f"]["ultimate"]
    util = report["criteria"]["f"]["utilisation"]
    fig, ax = plt.subplots(1, 1, figsize=(6.0, 4.2))
    parts = [
        ("still water", comb["still_water_Nm"] / 1e9, "C0"),
        (r"wave (ITTC/IACS)", comb["wave_Nm"] / 1e9, "C2"),
        (r"$\psi_{ice}\cdot$ ice (100-yr)", comb["psi_ice"] * comb["ice_Nm"] / 1e9, "C4"),
    ]
    bottom = 0.0
    for label, value, colour in parts:
        ax.bar(["design moment"], [value], bottom=bottom, color=colour, label=label, width=0.45)
        bottom += value
    ax.bar(["ultimate capacity"], [ult["M_ultimate_Nm"] / 1e9], color="0.65", width=0.45,
           label=r"$M_u$ (simplified, $\Phi$ reduced)")
    ax.axhline(util["M_capacity_factored_Nm"] / 1e9, color="C3", ls="--", lw=1.2,
               label=rf"$M_u/\gamma_M$ = {util['M_capacity_factored_Nm']/1e9:.3f} GN·m")
    ax.set_ylabel("[GN·m]")
    ax.set_title(rf"UR I ultimate moment check: utilisation = {util['utilisation']:.3f} "
                 rf"({'PASS' if util['passes'] else 'FAIL'})")
    ax.legend(fontsize=8)
    figs.append(_save(fig, outdir / "fig6_utilisation.png",
                      "Composition of the design bending moment against the factored ultimate capacity."))

    # ------------------------------------------------------ fig 7: criteria errors
    a = report["criteria"]["a"]
    b = report["criteria"]["b"]
    c = report["criteria"]["c"]
    d = report["criteria"]["d"]
    e = report["criteria"]["e"]
    labels = [
        "(a) uniform limit\n(numeric)",
        "(a) uniform limit\n(MC mean)",
        "(b) white-noise\nvariance",
        "(b) 10 bays\nvariance",
        "(c) wave vs ITTC\n(hogging)",
        "(c) solver vs\nclosed form",
        "(d) MC 1e4 vs 5e4\n(Gumbel)",
    ]
    errors = np.array([
        a["relative_error_numeric"],
        a["relative_error_mc_mean"],
        b["white_noise"]["relative_error"],
        b["ten_bays"]["relative_error"],
        c["relative_error_hogging"],
        c["solver_vs_closed_form"]["relative_error"],
        d["relative_error_gumbel"],
    ])
    tolerances = np.array([0.01, 0.01, 0.03, 0.03, 0.10, 0.01, 0.05])
    fig, ax = plt.subplots(1, 1, figsize=(8.4, 4.0))
    idx = np.arange(len(errors))
    ax.bar(idx - 0.2, np.maximum(errors, 1e-6), width=0.4, color="C0", label="measured relative error")
    ax.bar(idx + 0.2, tolerances, width=0.4, color="C7", label="tolerance")
    ax.set_yscale("log")
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("relative error")
    ax.set_title(f"Acceptance criteria (a)-(d): errors vs tolerances "
                 f"(Spearman monotonicity \\rho={e['spearman_mc_mean']:.3f} > 0.95)")
    ax.legend(fontsize=8)
    figs.append(_save(fig, outdir / "fig7_criteria_errors.png",
                      "Measured relative errors of criteria (a)-(d) against their tolerances."))

    report["figures"] = [{"name": f["name"], "file": f["path"], "caption": f["caption"]} for f in figs]
    (outdir / "acceptance.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"[hullgirder] acceptance.json written to {outdir/'acceptance.json'}")
    print(f"[hullgirder] all_passed = {report['all_passed']}")
    for key, value in report["criteria"].items():
        print(f"  ({key}) passed={value['passed']}")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())