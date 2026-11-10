#!/usr/bin/env python3
"""Figures for campaign node p380w2-n07 (superstructure icing).

Runs the acceptance battery (``tensorphysics.superstructure_icing.case_runner``)
and plots the report figures into ``artifacts/p380w2_n07/figures``.

    python generate_p380w2_n07_figures.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorphysics.superstructure_icing.anti_icing import (  # noqa: E402
    ElectrothermalAntiIcing,
    power_saving,
)
from tensorphysics.superstructure_icing.case_runner import (  # noqa: E402
    CaseConfig,
    run_acceptance,
)
from tensorphysics.superstructure_icing.droplets import (  # noqa: E402
    impingement,
    trajectory_paths,
)
from tensorphysics.superstructure_icing.flowfield import CylinderFlow  # noqa: E402
from tensorphysics.superstructure_icing.geometry import reference_superstructure  # noqa: E402
from tensorphysics.superstructure_icing.messinger import (  # noqa: E402
    heat_transfer_coefficient,
    ice_growth_1d,
    messinger_stagnation_reference,
)
from tensorphysics.superstructure_icing.sensitivity import (  # noqa: E402
    LWC_GRID_G_M3,
    MVD_GRID_UM,
    T_GRID_C,
)

OUT = ROOT / "artifacts" / "p380w2_n07"
FIG = OUT / "figures"
CFG = CaseConfig()


def _save(fig, name):
    path = FIG / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", path.relative_to(ROOT))
    return path


def fig_trajectories_and_beta(body, res):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for y0 in (0.004, 0.02, 0.05, 0.08, 0.11, 0.14, 0.2):
        for sign in (1, -1):
            path = trajectory_paths(body, CFG.d_m, [y0 * sign], U=CFG.U, t_air=CFG.T_air)[0]
            ax[0].plot(path[:, 0], path[:, 1], lw=1.0,
                       color="tab:blue" if sign > 0 else "tab:green")
    th = np.linspace(0, 2 * np.pi, 400)
    ax[0].fill(body.R * np.cos(th), body.R * np.sin(th), color="0.35", zorder=3)
    ax[0].set_aspect("equal")
    ax[0].set_xlim(-0.45, 0.3)
    ax[0].set_ylim(-0.22, 0.22)
    ax[0].set_xlabel("x [m]")
    ax[0].set_ylabel("y [m]")
    ax[0].set_title(
        f"Lagrangian droplet paths, MVD={CFG.mvd_um:.0f} um, U={CFG.U:.0f} m/s\n"
        f"K={res.K:.3f}, Re_d={res.Re_d:.0f}, E={res.E_total:.3f}"
    )
    theta = np.degrees(res.s / body.R)
    ax[1].plot(theta, res.beta, "o-", ms=3, lw=1.2, label="trajectory method")
    ax[1].plot(theta, np.sqrt(np.clip(1 - (res.s / body.R) ** 2, 0, 1)), "--", lw=1.2,
               label="ballistic limit cos(theta)")
    for d_um, style in ((20, ":"), (100, "-.")):
        r2 = impingement(body, d_um * 1e-6, U=CFG.U, t_air=CFG.T_air, n_traj=61)
        ax[1].plot(np.degrees(r2.s / body.R), r2.beta, style, lw=1.1, label=f"MVD {d_um} um")
    ax[1].set_xlabel("arc angle from stagnation point [deg]")
    ax[1].set_ylabel("local collection efficiency beta [-]")
    ax[1].set_title("Collection efficiency beta(theta)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    return _save(fig, "fig1_trajectories_and_beta.png")


def fig_collection_efficiency(body):
    mvd = np.array([10, 20, 30, 50, 70, 100, 150, 200, 300, 500], dtype=float)
    E, K = [], []
    for d in mvd:
        r = impingement(body, d * 1e-6, U=CFG.U, t_air=CFG.T_air, n_traj=61)
        E.append(r.E_total)
        K.append(r.K)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.semilogx(K, E, "o-", label="E from trajectory integration")
    ax.axhline(1.0, ls="--", color="0.5", lw=1.0, label="ballistic limit E=1")
    ax.set_xlabel("inertia parameter K = rho_w d^2 U / (18 mu D)")
    ax.set_ylabel("total collection efficiency E")
    ax.set_title("Cylinder (D=0.2 m) collection efficiency vs inertia parameter")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    return _save(fig, "fig2_collection_efficiency.png")


def fig_ice_growth(body, res):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for (Ta, lwc_g, mvd_um, col) in (
        (-15.0, 1.0, 50.0, "tab:blue"),
        (-15.0, 2.0, 50.0, "tab:red"),
        (-25.0, 0.5, 20.0, "tab:green"),
        (-5.0, 1.0, 100.0, "tab:orange"),
    ):
        r = impingement(body, mvd_um * 1e-6, U=CFG.U, t_air=Ta, n_traj=61)
        s = np.linspace(0.0, r.s_limit, 21)
        h_c = heat_transfer_coefficient(CFG.U, CFG.diameter, Ta)
        g = ice_growth_1d(s, r.beta_at(s), CFG.U, Ta, lwc_g * 1e-3, 600.0, dt=5.0,
                          h_c=h_c, L_char=CFG.diameter)
        ax[0].plot(g.t, g.h_ice[:, 0] * 1e3, col, lw=1.6,
                   label=f"T={Ta:.0f}C, LWC={lwc_g:.1f} g/m3, MVD={mvd_um:.0f}um")
        t_ref, h_ref = messinger_stagnation_reference(
            float(r.beta[0]), CFG.U, Ta, lwc_g * 1e-3, h_c, 600.0)
        ax[0].plot(t_ref, h_ref * 1e3, "k--", lw=0.8, alpha=0.6)
    ax[0].plot([], [], "k--", lw=0.9, label="independent Messinger reference")
    ax[0].set_xlabel("time [s]")
    ax[0].set_ylabel("ice thickness at stagnation [mm]")
    ax[0].set_title("Ice growth h_ice(t) (transient Messinger, 1-D runback)")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)

    s = np.linspace(0.0, res.s_limit, 41)
    h_c = heat_transfer_coefficient(CFG.U, CFG.diameter, CFG.T_air)
    g = ice_growth_1d(s, res.beta_at(s), CFG.U, CFG.T_air, CFG.lwc, 600.0, dt=5.0,
                      h_c=h_c, L_char=CFG.diameter)
    ax[1].plot(s * 1e3, g.h_final * 1e3, "o-", ms=3, color="tab:red",
               label="ice thickness at t=600 s")
    ax2 = ax[1].twinx()
    ax2.plot(s * 1e3, g.freezing_fraction, "s--", ms=3, color="tab:blue", label="freezing fraction n")
    ax2.plot(s * 1e3, g.T_surface, "^:", ms=3, color="tab:green", label="surface temperature [C]")
    ax2.set_ylabel("n [-], T_s [C]")
    ax[1].set_xlabel("arc length from stagnation point [mm]")
    ax[1].set_ylabel("h [mm]")
    ax[1].set_title(f"Design case T={CFG.T_air:.0f}C LWC={CFG.lwc_g_m3:.1f} g/m3: strip state")
    h1, l1 = ax[1].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax[1].legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    ax[1].grid(alpha=0.3)
    return _save(fig, "fig3_ice_growth.png")


def fig_messinger_validation(rec):
    b = rec["criteria"]["b_ice_growth"]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
    labels = ["h at stagnation\n(design case)", "rime closed form\n(T=-40C, low LWC)"]
    vals = [b["result"]["rel_error_pct"], b["rime_limit_check"]["rel_error_pct"]]
    ax[0].bar(labels, vals, color=["tab:green", "tab:olive"])
    ax[0].axhline(15.0, ls="--", color="tab:red", label="acceptance limit 15 %")
    ax[0].set_ylabel("relative error [%]")
    ax[0].set_title("Messinger solver verification")
    for i, v in enumerate(vals):
        ax[0].text(i, v * 1.02 + 1e-3, f"{v:.3f} %", ha="center", va="bottom", fontsize=9)
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, axis="y")
    comp = ["Messinger\n(no conduction)", "Stefan\n(conduction limited)"]
    vals2 = [b["result"]["h_stagnation_mm"], b["stefan_conduction_limited_mm"]]
    ax[1].bar(comp, vals2, color=["tab:blue", "tab:orange"])
    ax[1].set_ylabel("ice thickness at 600 s [mm]")
    ax[1].set_title("Regime of the design case (impingement- vs conduction-limited)")
    for i, v in enumerate(vals2):
        ax[1].text(i, v * 1.01, f"{v:.2f} mm", ha="center", va="bottom", fontsize=9)
    ax[1].grid(alpha=0.3, axis="y")
    return _save(fig, "fig4_messinger_validation.png")


def fig_anti_icing(body, res):
    s = np.linspace(0.0, res.s_limit, CFG.n_cell_anti)
    h_c = heat_transfer_coefficient(CFG.U, CFG.diameter, CFG.T_air)
    anti = ElectrothermalAntiIcing(s, res.beta_at(s), CFG.U, CFG.T_air, CFG.lwc,
                                   t_end=CFG.t_anti_icing, dt=10.0, h_c=h_c,
                                   L_char=CFG.diameter)
    unheated = anti.ice_thickness(np.zeros(s.size))
    uniform = anti.uniform_design(CFG.h_allow)
    optimised = anti.optimize(CFG.h_allow)
    simp = anti.simp_design(CFG.h_allow)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.0))
    xmm = s * 1e3
    ax[0].step(xmm, uniform.q, where="mid", label=f"uniform ({uniform.power_w:.0f} W/m)")
    ax[0].step(xmm, optimised.q, where="mid", label=f"SLSQP optimum ({optimised.power_w:.0f} W/m)")
    ax[0].step(xmm, simp.q, where="mid", ls="--", label=f"SIMP 0/1 ({simp.power_w:.0f} W/m)")
    ax[0].set_xlabel("arc length [mm]")
    ax[0].set_ylabel("heater power density [W/m2]")
    ax[0].set_title(f"Heater layout, h_allow={CFG.h_allow*1e3:.0f} mm, t={CFG.t_anti_icing:.0f} s")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[1].step(xmm, unheated * 1e3, where="mid", label="unheated")
    ax[1].step(xmm, uniform.h_ice * 1e3, where="mid", label="uniform")
    ax[1].step(xmm, optimised.h_ice * 1e3, where="mid", label="SLSQP optimum")
    ax[1].step(xmm, simp.h_ice * 1e3, where="mid", ls="--", label="SIMP 0/1")
    ax[1].axhline(CFG.h_allow * 1e3, ls=":", color="k", label="constraint")
    ax[1].set_xlabel("arc length [mm]")
    ax[1].set_ylabel("ice thickness at t_end [mm]")
    ax[1].set_title("Residual ice thickness")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    names = ["uniform", "SLSQP optimum", "SIMP 0/1"]
    powers = [uniform.power_w, optimised.power_w, simp.power_w]
    sav = [0.0, power_saving(optimised, uniform), power_saving(simp, uniform)]
    bars = ax[2].bar(names, powers, color=["tab:gray", "tab:blue", "tab:cyan"])
    for bar, pv, sv in zip(bars, powers, sav):
        ax[2].text(bar.get_x() + bar.get_width() / 2, pv * 1.01,
                   f"{pv:.0f} W/m\n({sv:+.1f} %)", ha="center", va="bottom", fontsize=8)
    ax[2].axhline(powers[0] * 0.7, ls="--", color="tab:red", label="30 % saving target")
    ax[2].set_ylabel("electric power per metre [W/m]")
    ax[2].set_title("Anti-icing power: uniform vs optimised")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3, axis="y")
    return _save(fig, "fig5_anti_icing.png")


def fig_stability(rec):
    d = rec["criteria"]["d_stability"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for hull, style in (("coaster", "o-"), ("fishing", "s-")):
        rows = d["hulls"][hull]["load_sweep"]
        masses = [r["ice_mass_t"] for r in rows]
        ax[0].plot(masses, [r["dKG_m"] for r in rows], style,
                   label=f"{hull} (disp. {d['hulls'][hull]['ship']['displacement_t']:.0f} t)")
        ax[1].plot(masses, [r["dT_roll_pct"] for r in rows], style, label=hull)
    ax[0].axhline(0.1, ls="--", color="tab:red", label="criterion dKG >= 0.1 m")
    ax[1].axhline(10.0, ls="--", color="tab:red", label="criterion dT_roll >= 10 %")
    ax[0].set_xlabel("total ice mass on the superstructure [t]")
    ax[0].set_ylabel("dKG [m]")
    ax[0].set_title("Centre-of-gravity rise due to topside icing")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[1].set_xlabel("total ice mass on the superstructure [t]")
    ax[1].set_ylabel("dT_roll [%]")
    ax[1].set_title("Roll-period increase due to topside icing")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    return _save(fig, "fig6_stability.png")


def fig_matrix(rec):
    cases = rec["criteria"]["e_sensitivity_matrix"]["cases"]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.0), sharey=True)
    im = None
    for k, Ta in enumerate(T_GRID_C):
        sub = [c for c in cases if c["T_air"] == Ta]
        M = np.zeros((len(LWC_GRID_G_M3), len(MVD_GRID_UM)))
        for c in sub:
            i = LWC_GRID_G_M3.index(c["lwc_g_m3"])
            j = MVD_GRID_UM.index(c["mvd_um"])
            M[i, j] = c["h_stag_mm"]
        im = ax[k].imshow(M, origin="lower", cmap="viridis", aspect="auto")
        ax[k].set_xticks(range(len(MVD_GRID_UM)), [f"{v:.0f}" for v in MVD_GRID_UM])
        ax[k].set_yticks(range(len(LWC_GRID_G_M3)), [f"{v:.1f}" for v in LWC_GRID_G_M3])
        ax[k].set_xlabel("MVD [um]")
        if k == 0:
            ax[k].set_ylabel("LWC [g/m3]")
        ax[k].set_title(f"T_air = {Ta:.0f} C, U = {CFG.U:.0f} m/s")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax[k].text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                           color="w" if M[i, j] < 0.6 * M.max() else "k", fontsize=7)
    fig.colorbar(im, ax=ax, label="h_ice at stagnation at t=600 s [mm]", shrink=0.9)
    fig.suptitle("27-case sensitivity matrix of superstructure icing (0.2 m cylinder section)")
    return _save(fig, "fig7_sensitivity_matrix.png")


def fig_convergence(rec):
    sv = rec["criteria"]["a_impingement"]["substitute_verification"]
    tc = rec["criteria"]["f_time_convergence"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
    names = ["time step\n(2x -> 0.25x)", "launch grid\n(81 vs 241)", "contact tol.\n(4e-4 vs 1e-4)"]
    vals = [sv["convergence_in_step"]["max_rel_change_E_pct"],
            sv["convergence_in_launch_grid"]["max_rel_change_pct"],
            sv["convergence_in_contact_tolerance"]["max_rel_change_pct"]]
    ax[0].bar(names, vals, color="tab:purple")
    ax[0].set_ylabel("relative change of E [%]")
    ax[0].set_title("Impingement: numerical convergence (target < 2 %)")
    ax[0].axhline(2.0, ls="--", color="tab:red")
    for i, v in enumerate(vals):
        ax[0].text(i, v + 0.01, f"{v:.2f} %", ha="center", fontsize=9)
    ax[0].grid(alpha=0.3, axis="y")
    dts = tc["dt_values_s"]
    ax[1].plot(dts, [tc["h_max_mm"][f"dt_{dt:g}"] for dt in dts], "o-", label="max ice thickness")
    ax[1].plot(dts, [tc["h_stag_mm"][f"dt_{dt:g}"] for dt in dts], "s--",
               label="stagnation thickness")
    ax[1].set_xlabel("time step [s]")
    ax[1].set_ylabel("h_ice at t=600 s [mm]")
    ax[1].set_title("Time-step independence of the ice growth\n"
                    f"(max change {tc['max_rel_change_pct']:.3f} %, limit 10 %)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    return _save(fig, "fig8_convergence.png")


def fig_geometry(hull="coaster"):
    sup = reference_superstructure(hull)
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    names = [c.name for c in sup.components]
    areas = [c.area for c in sup.components]
    zs = [c.z for c in sup.components]
    y = np.arange(len(names))[::-1]
    ax.barh(y, areas, color="tab:blue", alpha=0.75)
    for yi, a, z in zip(y, areas, zs):
        ax.text(a + 8, yi, f"A={a:.0f} m2, z={z:.1f} m", va="center", fontsize=8)
    ax.set_yticks(y, names, fontsize=8)
    ax.set_xlabel("exposed (iced) area [m2]")
    ax.set_title(f"Reference superstructure '{hull}': total exposed area {sup.area:.0f} m2")
    ax.set_xlim(0, max(areas) * 1.5)
    ax.grid(alpha=0.3, axis="x")
    return _save(fig, "fig9_superstructure.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    rec = run_acceptance(cfg=CFG, out_dir=OUT, verbose=True)
    body = CylinderFlow(R=CFG.diameter / 2.0, U=CFG.U)
    res = impingement(body, CFG.d_m, U=CFG.U, t_air=CFG.T_air, n_traj=CFG.n_traj)
    print("[figures]")
    fig_trajectories_and_beta(body, res)
    fig_collection_efficiency(body)
    fig_ice_growth(body, res)
    fig_messinger_validation(rec)
    fig_anti_icing(body, res)
    fig_stability(rec)
    fig_matrix(rec)
    fig_convergence(rec)
    fig_geometry("coaster")
    (OUT / "figure_manifest.json").write_text(json.dumps(
        sorted(p.name for p in FIG.glob("*.png")), indent=2))
    print("done")


if __name__ == "__main__":
    main()