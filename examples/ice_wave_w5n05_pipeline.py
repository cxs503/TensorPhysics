#!/usr/bin/env python3
"""W5-N05 pipeline: flexural-gravity dispersion, MIZ attenuation, Love-Weber
homogenisation of a DEM brash-ice cell, mu(I) rheology calibration and the
EnKF background interface.

Every number that appears in the report is produced here and written to
``artifacts/w5n05/results.json``; figures are written to
``artifacts/w5n05/figs/`` and the markdown report is rendered from the JSON
(no hand-copied digits).

Usage::

    PYTHONPATH=src python3 examples/ice_wave_w5n05_pipeline.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from tensorphysics.ice_wave import dem2d, flexural, homogenise, miz, rheology
from tensorphysics.ice_wave.enkf import ensemble_kalman_analysis

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "w5n05"
FIGS = OUT / "figs"


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# --------------------------------------------------------------------------- 1
def dispersion_stage() -> dict:
    _log("stage 1: flexural-gravity dispersion")
    rigidities = {f"{t:.1f}": flexural.flexural_rigidity(t) for t in (0.5, 1.0, 1.5, 2.0)}
    ks = np.logspace(-3.3, 0.3, 60)
    curves = {}
    for depth in (50.0, 500.0, 5000.0):
        for label, D in rigidities.items():
            omega = np.sqrt(
                [flexural.flexural_gravity_dispersion(k, depth, D) for k in ks]
            )
            cph = omega / ks
            curves[f"{label}|{depth:g}"] = {
                "k": ks.tolist(),
                "omega": omega.tolist(),
                "phase_speed": cph.tolist(),
            }

    # (a) analytical vs spectral residual, plus the depth-truncation error.
    cases = [
        (0.001, 2.0),
        (0.005, 5.0),
        (0.02, 100.0),
        (0.05, 500.0),
        (0.1, 1000.0),
        (0.5, 1000.0),
        (1.0, 3000.0),
        (2.0, 5000.0),
    ]
    D1 = flexural.flexural_rigidity(1.5)
    residual_rows = []
    for k, h in cases:
        for n in (16, 24, 32, 40):
            residual_rows.append(
                {
                    "k": k,
                    "h": h,
                    "n_modes": n,
                    "residual": flexural.spectral_residual(k, h, D1, n_modes=n),
                }
            )
    worst = max(row["residual"] for row in residual_rows if row["n_modes"] == 32)
    truncation = max(flexural.depth_truncation_error(k, h, D1) for k, h in cases)

    # Limits and dispersion invariants.
    g, rho = flexural.GRAVITY, flexural.RHO_WATER
    k_shallow, h_shallow = 1e-3, 50.0
    w2 = flexural.flexural_gravity_dispersion(k_shallow, h_shallow, D1)
    shallow_rel = abs(w2 - g * k_shallow**2 * h_shallow) / (g * k_shallow**2 * h_shallow)
    k_deep, h_deep = 0.05, 5000.0
    w2_deep = flexural.flexural_gravity_dispersion(k_deep, h_deep, D1)
    deep_expected = g * k_deep + D1 * k_deep**5 / rho
    deep_rel = abs(w2_deep - deep_expected) / deep_expected

    # Minimum phase speed of the flexural branch (deep water): c^2 = g/k + D k^3 / rho
    k_min_analytic = (rho * g / (3.0 * D1)) ** 0.25
    k_grid = np.logspace(-4, 0.5, 200001)
    cph_grid = np.sqrt([flexural.flexural_gravity_dispersion(k, 1.0e5, D1) for k in k_grid]) / k_grid
    k_min_numeric = float(k_grid[int(np.argmin(cph_grid))])
    group_check = []
    for k in (0.005, 0.02, 0.05, 0.2, 0.5):
        dk = 1e-6 * k
        numeric = (
            np.sqrt(flexural.flexural_gravity_dispersion(k + dk, 500.0, D1))
            - np.sqrt(flexural.flexural_gravity_dispersion(k - dk, 500.0, D1))
        ) / (2 * dk)
        analytic = flexural.group_speed(k, 500.0, D1)
        group_check.append(
            {"k": k, "group_speed_analytic": analytic, "group_speed_fd": float(numeric),
             "rel_error": abs(float(numeric) - analytic) / analytic}
        )

    return {
        "rigidities": rigidities,
        "curves": curves,
        "residuals": residual_rows,
        "criterion_a_residual_max": worst,
        "criterion_a_truncation_max": truncation,
        "limits": {
            "shallow_water_rel_error": shallow_rel,
            "deep_water_flexural_rel_error": deep_rel,
            "flexural_k_min_analytic": float(k_min_analytic),
            "flexural_k_min_numeric": k_min_numeric,
            "flexural_k_min_rel_error": abs(k_min_numeric - k_min_analytic) / k_min_analytic,
        },
        "group_speed_checks": group_check,
    }


# --------------------------------------------------------------------------- 2
def miz_stage() -> dict:
    _log("stage 2: MIZ attenuation")
    concentrations = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    thicknesses = [0.5, 1.0, 1.5, 2.0]
    sweep = miz.sweep_attenuation(concentrations, thicknesses, period=10.0, depth=1000.0)

    checks = []
    for n_c in (0.3, 0.5, 0.7, 0.9):
        for h_i in (1.0, 2.0):
            props = miz.effective_medium_properties(n_c, h_i, loss_factor=0.02)
            k = miz.kelvin_voigt_wavenumber(
                10.0, 1000.0, props["D_real"], props["D_imag"]
            )
            alpha = miz.attenuation_from_complex_wavenumber(k)
            temporal = miz.temporal_attenuation(
                10.0, 1000.0, props["D_real"], props["D_imag"]
            )
            checks.append(
                {
                    "concentration": n_c,
                    "ice_thickness": h_i,
                    "k_real": float(k.real),
                    "alpha": alpha,
                    "alpha_temporal": temporal["alpha"],
                    "rel_error": abs(temporal["alpha"] - alpha) / alpha,
                }
            )
    worst_rel = max(row["rel_error"] for row in checks)

    # Exponential-decay profile + fit (validates the A(x) = A0 exp(-alpha x) fitter).
    x = np.linspace(0.0, 40000.0, 401)
    alpha_ref = checks[0]["alpha"]
    profile = miz.exponential_decay_profile(x, 1.0, alpha_ref)
    fit = miz.fit_attenuation_coefficient(x, profile)

    band = miz.LITERATURE_ALPHA_BAND
    alpha_10s = {
        f"{row['concentration']}|{row['ice_thickness']}": row["alpha"] for row in checks
    }
    inside = [v for v in alpha_10s.values() if band["alpha_min"] <= v <= band["alpha_max"]]
    return {
        "criterion_d_rel_error_max": worst_rel,
        "cross_checks": checks,
        "sweep": {
            "concentrations": sweep["concentrations"],
            "ice_thicknesses": sweep["ice_thicknesses"],
            "alpha_grid": sweep["alpha_grid"].tolist(),
            "alpha_vs_concentration_exponent": sweep["alpha_vs_concentration_exponent"],
            "alpha_vs_thickness_exponent": sweep["alpha_vs_thickness_exponent"],
            "period": sweep["period"],
            "depth": sweep["depth"],
            "loss_factor": sweep["loss_factor"],
        },
        "profile_fit": {
            "alpha_ref": alpha_ref,
            "alpha_fit": fit["alpha"],
            "amplitude0_fit": fit["amplitude0"],
            "r2": fit["r2"],
            "rel_error": abs(fit["alpha"] - alpha_ref) / alpha_ref,
        },
        "literature_band": band,
        "literature_check": {
            "n_values_in_band": len(inside),
            "n_values": len(alpha_10s),
            "min_alpha": min(alpha_10s.values()),
            "max_alpha": max(alpha_10s.values()),
            "status": "WITHHELD",
            "reason": band["note"],
        },
    }


# --------------------------------------------------------------------------- 3
def dem_stage() -> dict:
    _log("stage 3: DEM shear cell + mu(I) calibration")
    # Constant-volume Lees-Edwards shear at a fixed, freshly jammed preparation
    # (area fraction 0.845). The cell pressure is set by the preparation, so the
    # inertial number I = gammadot d / sqrt(p / rho) is swept through gammadot.
    shear_rates = [0.05, 0.1, 0.25, 0.6, 1.2]
    strain_target = 2.5
    cell = dem2d.ShearCell(
        n_particles=150,
        area_fraction=0.845,
        shear_rate=0.0,
        kn=1.0e3,
        seed=11,
    )
    cell.vel *= 0.5
    for _ in range(500):
        cell.step()
    rows = []
    for index, gamma in enumerate(shear_rates):
        # One preparation, several rates: preparation history is held fixed so the
        # rate dependence of mu is not confounded with the packing realisation.
        cell.shear_rate = gamma
        n_steps = int(min(strain_target / (gamma * cell.dt), 60000))
        run = cell.run(n_steps, measure_every=25)
        inertial = rheology.inertial_number(gamma, 1.0, run["p"], 1.0)
        rows.append(
            {
                "shear_rate": gamma,
                "inertial_number": inertial,
                "mu": run["mu_xy"],
                "mu_flux_symmetric": run["mu_flux"],
                "mu_std": run["mu_xy_std"],
                "pressure": run["p"],
                "coordination": run["z"],
                "area_fraction": run["area_fraction"],
                "strain": run["strain_total"],
                "n_samples": run["n_samples"],
                "n_steps": n_steps,
            }
        )
        _log(
            "  gdot={:.2f} I={:.4f} mu_xy={:.4f} mu_sym={:.4f} p={:.2f} Z={:.2f} strain={:.1f}".format(
                gamma, inertial, run["mu_xy"], run["mu_flux"], run["p"], run["z"],
                run["strain_total"],
            )
        )
    snapshot = cell.snapshot()
    snapshot_cell = cell

    I = np.array([row["inertial_number"] for row in rows])
    mu = np.array([row["mu"] for row in rows])
    try:
        fit = rheology.fit_mu_i(I, mu)
        model = rheology.mu_jop(I, mu_s=fit["mu_s"], mu_2=fit["mu_2"], I_0=fit["I_0"])
        fit_valid = True
        fit_error = None
    except rheology.MuIError as exc:
        # Honest failure: no admissible mu(I) parameter set for the measured data.
        fit = {"mu_s": float("nan"), "mu_2": float("nan"), "I_0": float("nan"),
               "r2": float("nan"), "rmse": float("nan"), "n_points": int(I.size),
               "residuals": np.full(I.size, np.nan)}
        model = np.full(I.size, np.nan)
        fit_valid = False
        fit_error = str(exc)
        _log("  mu(I) fit refused: " + fit_error)

    # --- Love-Weber homogenisation of the recorded contact network
    contacts = snapshot["contacts"]
    box = float(snapshot["box_length"][0])
    volume = box**2
    sigma = homogenise.love_weber_stress(contacts["force"], contacts["branch"], volume)
    sigma_virial = homogenise.per_particle_virial_stress(
        snapshot["positions"],
        np.column_stack([contacts["i"], contacts["j"]]),
        contacts["force"],
        volume,
        box_length=box,
    )
    scale = float(np.abs(sigma).max())
    covariance = []
    for angle_deg in (1.0, 17.0, 45.0, 90.0, 137.0, 231.0):
        angle = np.deg2rad(angle_deg)
        covariance.append(
            {
                "angle_deg": angle_deg,
                "error": homogenise.rotational_covariance_error(
                    contacts["force"], contacts["branch"], volume, angle
                ),
            }
        )
    # Negative control: rotating forces and branch vectors by *different* angles
    # must break the covariance by O(1) -- it shows the covariance test discriminates.
    angle_a, angle_b = np.deg2rad(37.0), np.deg2rad(52.0)
    Ra = np.array([[np.cos(angle_a), -np.sin(angle_a)], [np.sin(angle_a), np.cos(angle_a)]])
    Rb = np.array([[np.cos(angle_b), -np.sin(angle_b)], [np.sin(angle_b), np.cos(angle_b)]])
    sigma_ctrl = homogenise.love_weber_stress(
        contacts["force"] @ Ra.T, contacts["branch"] @ Rb.T, volume
    )
    broken_error = float(
        np.abs(sigma_ctrl - Ra @ sigma @ Ra.T).max() / float(np.abs(sigma).max())
    )

    homogenised = {
        "sigma": sigma.tolist(),
        "sigma_virial": sigma_virial.tolist(),
        "pressure": float(0.5 * np.trace(sigma)),
        "shear_stress": float(0.5 * (sigma[0, 1] + sigma[1, 0])),
        "shear_stress_xy": float(sigma[0, 1]),
        "asymmetry": float(abs(sigma[0, 1] - sigma[1, 0]) / scale),
        "n_contacts": int(contacts["force"].shape[0]),
        "coordination": float(2.0 * contacts["force"].shape[0] / snapshot["positions"].shape[0]),
        "volume": volume,
        "criterion_e_rel_difference": float(np.abs(sigma - sigma_virial).max() / scale),
        "criterion_b_covariance_error_max": max(row["error"] for row in covariance),
        "covariance_scan": covariance,
        "inconsistent_rotation_covariance_error": broken_error,
    }
    _log(
        "  Love-Weber sigma={} criterion_e={:.2e} criterion_b={:.2e} corrupted={:.2e}".format(
            np.array2string(sigma, precision=4),
            homogenised["criterion_e_rel_difference"],
            homogenised["criterion_b_covariance_error_max"],
            broken_error,
        )
    )
    return {
        "calibration_rows": rows,
        "fit": {k: v for k, v in fit.items() if k != "residuals"},
        "fit_valid": fit_valid,
        "fit_error": fit_error,
        "fit_model_values": np.asarray(model, dtype=float).tolist(),
        "criterion_c_r2": float(fit["r2"]),
        "homogenised": homogenised,
        "cell": {
            "n_particles": snapshot_cell.n_particles,
            "box_length": box,
            "kinetic_stress_note": "streaming (kinetic) stress neglected; dense regime, Z ~ 3.4",
            "protocol": "constant-volume Lees-Edwards shear, area fraction 0.845, kn=1e3",
        },
    }


# --------------------------------------------------------------------------- 4
def enkf_stage(homogenised_mu: float) -> dict:
    _log("stage 4: EnKF background interface")
    rng = np.random.default_rng(7)
    n_ens, n_state = 120, 9
    # The state IS an ensemble of homogenised mu fields, so the synthetic truth is
    # centred on the DEM-homogenised value with a zero-mean spatial pattern; the
    # pseudo-observation of the field mean is then exactly consistent.
    pattern = 0.04 * np.sin(2.0 * np.pi * np.arange(n_state) / n_state)
    truth = homogenised_mu + pattern
    background = truth + rng.normal(scale=0.04, size=(n_ens, n_state))
    obs_index = np.array([0, 4, 8])
    H = np.zeros((obs_index.size, n_state))
    for row, col in enumerate(obs_index):
        H[row, col] = 1.0
    observations = truth[obs_index] + rng.normal(scale=0.01, size=obs_index.size)
    # The homogenised DEM mu enters as an extra pseudo-observation of the field mean.
    H_mean = np.full((1, n_state), 1.0 / n_state)
    result = ensemble_kalman_analysis(
        background,
        np.concatenate([observations, [homogenised_mu]]),
        operator=np.vstack([H, H_mean]),
        observation_error_std=np.concatenate([np.full(obs_index.size, 0.01), [0.01]]),
        seed=3,
    )
    analysis = result["analysis"]
    rmse_bg = float(np.sqrt(np.mean((background.mean(axis=0) - truth) ** 2)))
    rmse_an = float(np.sqrt(np.mean((analysis.mean(axis=0) - truth) ** 2)))
    return {
        "n_ensemble": n_ens,
        "n_state": n_state,
        "observation_index": obs_index.tolist(),
        "homogenised_mu_pseudo_observation": float(homogenised_mu),
        "rmse_background": rmse_bg,
        "rmse_analysis": rmse_an,
        "rmse_reduction": 1.0 - rmse_an / rmse_bg,
        "spread_before": result["spread_before"],
        "spread_after": result["spread_after"],
        "truth": truth.tolist(),
        "background_mean": background.mean(axis=0).tolist(),
        "analysis_mean": analysis.mean(axis=0).tolist(),
    }


# --------------------------------------------------------------------------- 5
def figures(results: dict) -> list[str]:
    _log("stage 5: figures")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGS.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    # fig1: dispersion curves (omega and phase speed)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for key, data in results["dispersion"]["curves"].items():
        label, depth = key.split("|")
        if float(depth) != 5000.0:
            continue
        axes[0].loglog(data["k"], data["omega"], label=f"h_i={label} m")
        axes[1].loglog(data["k"], data["phase_speed"], label=f"h_i={label} m")
    axes[0].set_xlabel("k [1/m]")
    axes[0].set_ylabel("omega [rad/s]")
    axes[0].set_title("Flexural-gravity dispersion (deep water)")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()
    axes[1].set_xlabel("k [1/m]")
    axes[1].set_ylabel("phase speed [m/s]")
    axes[1].set_title("Phase speed with flexural minimum")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    path = FIGS / "fig1_dispersion.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append("fig1_dispersion.png")

    # fig2: spectral residual convergence
    fig, ax = plt.subplots(figsize=(6, 4))
    rows = results["dispersion"]["residuals"]
    for (k, h) in sorted({(row["k"], row["h"]) for row in rows}):
        subset = [row for row in rows if row["k"] == k and row["h"] == h]
        ax.semilogy(
            [row["n_modes"] for row in subset],
            [max(row["residual"], 1e-16) for row in subset],
            marker="o",
            label=f"k={k}, h={h:g} m",
        )
    ax.axhline(1e-12, color="k", ls="--", lw=1, label="1e-12 acceptance")
    ax.set_xlabel("Chebyshev modes N")
    ax.set_ylabel("|omega_num^2 - omega_ana^2| / omega_ana^2")
    ax.set_title("Spectral collocation residual vs analytical dispersion")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path = FIGS / "fig2_spectral_residual.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append("fig2_spectral_residual.png")

    # fig3: MIZ attenuation vs concentration with the literature band
    sweep = results["miz"]["sweep"]
    band = results["miz"]["literature_band"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.axhspan(band["alpha_min"], band["alpha_max"], color="0.85",
               label="literature band (order of magnitude)")
    for i, h_i in enumerate(sweep["ice_thicknesses"]):
        ax.semilogx(sweep["concentrations"], sweep["alpha_grid"][i], marker="o",
                    label=f"h_i={h_i} m")
    ax.set_xlabel("ice concentration n_c [-]")
    ax.set_ylabel("alpha [1/m]")
    ax.set_title("MIZ attenuation, T=10 s, depth 1000 m, tan_delta=0.02")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIGS / "fig3_miz_attenuation.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append("fig3_miz_attenuation.png")

    # fig4: mu(I) calibration
    rows = results["dem"]["calibration_rows"]
    I = np.array([row["inertial_number"] for row in rows])
    mu = np.array([row["mu"] for row in rows])
    mu_std = np.array([row["mu_std"] for row in rows])
    fit = results["dem"]["fit"]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.errorbar(I, mu, yerr=mu_std, fmt="o", color="C3", label="DEM shear cell")
    if results["dem"].get("fit_valid", False):
        grid = np.logspace(np.log10(I.min() * 0.7), np.log10(I.max() * 1.4), 200)
        ax.semilogx(grid, rheology.mu_jop(grid, mu_s=fit["mu_s"], mu_2=fit["mu_2"],
                                          I_0=fit["I_0"]),
                    "-", color="C0",
                    label=("mu(I) fit: mu_s={:.3f}, mu_2={:.3f}, I_0={:.3f}, R2={:.4f}"
                           .format(fit["mu_s"], fit["mu_2"], fit["I_0"], fit["r2"])))
    ax.set_xlabel("inertial number I [-]")
    ax.set_ylabel("effective friction mu = tau/p")
    ax.set_title("mu(I) rheology calibration from the 2-D DEM cell")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIGS / "fig4_mu_i_calibration.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append("fig4_mu_i_calibration.png")

    # fig5: stress recovery + covariance scan
    hom = results["dem"]["homogenised"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    labels = ["xx", "xy", "yx", "yy"]
    lw = np.array(hom["sigma"]).ravel()
    vir = np.array(hom["sigma_virial"]).ravel()
    idx = np.arange(4)
    axes[0].bar(idx - 0.18, lw, width=0.36, label="Love-Weber")
    axes[0].bar(idx + 0.18, vir, width=0.36, label="per-particle virial")
    axes[0].set_xticks(idx, [f"sigma_{l}" for l in labels])
    axes[0].set_ylabel("stress [N/m]")
    axes[0].set_title("Homogenised stress: two independent paths")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, axis="y", alpha=0.3)
    scan = hom["covariance_scan"]
    axes[1].semilogy([row["angle_deg"] for row in scan],
                     [max(row["error"], 1e-18) for row in scan], "o-",
                     label="rotational covariance error")
    axes[1].axhline(1e-10, color="k", ls="--", lw=1, label="1e-10 acceptance")
    axes[1].set_xlabel("rigid rotation angle [deg]")
    axes[1].set_ylabel("relative error")
    axes[1].set_title("Rotational covariance of the recovered stress")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    path = FIGS / "fig5_stress_recovery.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append("fig5_stress_recovery.png")

    # fig6: EnKF background vs analysis
    enkf = results["enkf"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    x = np.arange(enkf["n_state"])
    ax.plot(x, enkf["truth"], "k-", lw=2, label="truth")
    ax.plot(x, enkf["background_mean"], "C0o--", label="background mean")
    ax.plot(x, enkf["analysis_mean"], "C3s-", label="analysis mean")
    ax.set_xlabel("homogenised field node index")
    ax.set_ylabel("effective friction mu [-]")
    ax.set_title("EnKF analysis: RMSE {:.4f} -> {:.4f}".format(
        enkf["rmse_background"], enkf["rmse_analysis"]))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIGS / "fig6_enkf_analysis.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append("fig6_enkf_analysis.png")
    return written


# --------------------------------------------------------------------------- 6
def criteria(results: dict) -> dict:
    d = results["dispersion"]
    h = results["dem"]["homogenised"]
    return {
        "a_dispersion_residual_lt_1e-12": {
            "value": d["criterion_a_residual_max"],
            "threshold": 1e-12,
            "pass": d["criterion_a_residual_max"] < 1e-12,
            "note": "max over the (k, h) sweep at N=32 modes; depth truncation error "
                    f"{d['criterion_a_truncation_max']:.2e} excluded from the residual",
        },
        "b_love_weber_covariance_lt_1e-10": {
            "value": h["criterion_b_covariance_error_max"],
            "threshold": 1e-10,
            "pass": h["criterion_b_covariance_error_max"] < 1e-10,
        },
        "c_mu_i_r2_gt_0.99": {
            "value": results["dem"]["criterion_c_r2"],
            "threshold": 0.99,
            "pass": bool(results["dem"]["fit_valid"]) and results["dem"]["criterion_c_r2"] > 0.99,
        },
        "d_miz_alpha_error_lt_15pct": {
            "value": results["miz"]["criterion_d_rel_error_max"],
            "threshold": 0.15,
            "pass": results["miz"]["criterion_d_rel_error_max"] < 0.15,
            "note": "complex-k Newton vs spectral temporal-decay route (internal "
                    "numerical validation); the literature leg is WITHHELD, see "
                    "miz.literature_check",
        },
        "e_homogenised_vs_dem_lt_10pct": {
            "value": h["criterion_e_rel_difference"],
            "threshold": 0.10,
            "pass": h["criterion_e_rel_difference"] < 0.10,
        },
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    started = time.time()
    results: dict = {"mission": 994, "node": "W5-N05"}
    results["dispersion"] = dispersion_stage()
    results["miz"] = miz_stage()
    results["dem"] = dem_stage()
    results["enkf"] = enkf_stage(results["dem"]["calibration_rows"][2]["mu"])
    results["figures"] = figures(results)
    results["criteria"] = criteria(results)
    results["runtime_seconds"] = time.time() - started
    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=float))
    _log(f"done in {results['runtime_seconds']:.0f} s")
    for name, item in results["criteria"].items():
        _log(f"  {name}: value={item['value']:.3e} pass={item['pass']}")


if __name__ == "__main__":
    main()