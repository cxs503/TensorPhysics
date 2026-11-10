"""Payload + figures for campaign node ``p227w2-N08`` (QZS isolation maturity).

Runs the maturity chain once and writes

* ``artifacts/m10169/ice_vibration_transfer_mitigation_n08.json`` - the
  machine-checked payload (single source of every number in the report);
* ``docs/ice_vibration_transfer_n08_maturity/fig*.png`` - the report figures.

Usage: ``PYTHONPATH=src python scripts/m10169_n08_figures.py``
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorphysics.ice_vibration_transfer.maturity import (
    DEFAULT_SEVERITIES,
    MitigationConfig,
    run_mitigation,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "m10169"
FIG = ROOT / "docs" / "ice_vibration_transfer_n08_maturity"


def main() -> pathlib.Path:
    result = run_mitigation(MitigationConfig())
    payload = result.to_json_dict()

    ART.mkdir(parents=True, exist_ok=True)
    out = ART / "ice_vibration_transfer_mitigation_n08.json"
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    FIG.mkdir(parents=True, exist_ok=True)

    crit = {c["id"]: c for c in result.criteria}
    design = result.design
    cfg = result.config
    basis = result.basis
    base_mount = result.baseline
    sev = result.severities
    curve = result.curves["design"]
    sn = payload["sn_curve"]
    f = np.asarray(curve["freqs_hz"])
    ice_lo, ice_hi = cfg.ice_band_hz
    labels = list(DEFAULT_SEVERITIES)

    # ------------------------------------------------------------- fig 1: modes
    modes = np.arange(1, basis.n_modes + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    axes[0].plot(modes, basis.dry_frequencies_hz, "o-", label="dry $f_{dry}$")
    axes[0].plot(modes, basis.wet_frequencies_hz, "s-", label="wet $f_{wet}$ (added mass)")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("mode number")
    axes[0].set_ylabel("natural frequency [Hz]")
    axes[0].set_title("KCS hull-girder modes: dry vs wet")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()
    axes[1].bar(modes, basis.frequency_ratio, color="#2b6cb0")
    axes[1].axhline(1.0, color="k", ls="--", lw=1)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_xlabel("mode number")
    axes[1].set_ylabel("$f_{wet}/f_{dry}$")
    axes[1].set_title("wet/dry ratio (max %.3f < 1)" % np.max(basis.frequency_ratio))
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_wet_dry_modes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------- fig 2: Bode
    fig, axes = plt.subplots(2, 1, figsize=(9.6, 7.0), sharex=True)
    for mount, label, color in (
        (base_mount, "conventional mount f_n=%.2f Hz" % base_mount.f_n_hz, "#c53030"),
        (design, "QZS mount f_n=%.3f Hz (alpha=%.3f)" % (design.f_n_hz, design.negative_stiffness_fraction), "#2f855a"),
    ):
        mag = np.maximum(mount.magnitude(f), 1e-12)
        axes[0].semilogx(f, 20.0 * np.log10(mag), color=color, lw=1.6, label=label)
        axes[1].semilogx(f, np.degrees(np.angle(mount.H(f))), color=color, lw=1.4)
        axes[0].axvline(mount.isolation_frequency_hz, color=color, ls=":", lw=1.0)
    for ax in axes:
        ax.axvspan(ice_lo, ice_hi, color="#f6e05e", alpha=0.25, label="ice band %.1f-%.1f Hz" % (ice_lo, ice_hi))
        ax.grid(True, which="both", alpha=0.3)
        ax.set_xlim(0.1, 100.0)
    axes[0].axhline(0.0, color="k", lw=0.8, ls="--")
    axes[0].set_ylabel("|H| [dB]")
    axes[0].set_title("Bode: equipment base acceleration / hull acceleration")
    axes[0].legend(loc="lower left", fontsize=8)
    axes[1].set_ylabel("phase [deg]")
    axes[1].set_xlabel("frequency [Hz]")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_bode_transfer.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------- fig 3: accel PSD
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    ax.loglog(f, np.maximum(curve["hull_psd"], 1e-30), "k-", lw=1.6, label="hull girder (foundation station)")
    ax.loglog(f, np.maximum(curve["support_psd"], 1e-30), color="grey", lw=1.0, ls="-.", label="bracket / support")
    ax.loglog(f, np.maximum(curve["base_psd_conventional"], 1e-30), "#c53030", lw=1.4, label="base acc (conventional)")
    ax.loglog(f, np.maximum(curve["base_psd_qzs"], 1e-30), "#2f855a", lw=1.8, label="base acc (QZS)")
    ax.axvspan(ice_lo, ice_hi, color="#f6e05e", alpha=0.25)
    ax.set_xlim(0.1, 100.0)
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("acceleration PSD [m$^2$/s$^4$/Hz]")
    ax.set_title("design ice class: acceleration PSD along the chain")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_base_accel_psd.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------- fig 4: design map
    rows = result.design_map_rows
    f_vals = sorted({round(r["f_n_hz"], 9) for r in rows})
    z_vals = sorted({round(r["zeta"], 9) for r in rows})
    grid = np.full((len(z_vals), len(f_vals)), np.nan)
    feasible = np.zeros_like(grid)
    for r in rows:
        i = z_vals.index(round(r["zeta"], 9))
        j = f_vals.index(round(r["f_n_hz"], 9))
        grid[i, j] = r["rms_ratio_base_over_hull"]
        feasible[i, j] = float(
            r["static_ok"] and r["stiffness_ok"] and r["zeta_ok"] and r["stroke_ok"] and r["ice_linear_ok"]
        )
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    mesh = ax.pcolormesh(
        np.asarray(f_vals),
        np.asarray(z_vals),
        np.log10(np.maximum(grid, 1e-3)),
        cmap="viridis",
        shading="nearest",
    )
    cs = ax.contour(np.asarray(f_vals), np.asarray(z_vals), grid, levels=[1.0], colors="red", linewidths=1.6)
    ax.clabel(cs, fmt="ratio = 1", fontsize=8)
    ax.contourf(
        np.asarray(f_vals),
        np.asarray(z_vals),
        feasible,
        levels=[0.5, 1.5],
        colors="none",
        hatches=["", "////"],
    )
    ax.plot([design.f_n_hz], [design.zeta], "w*", ms=18, mec="k", label="selected design")
    ax.axhline(cfg.qzs_zeta_min, color="w", ls="--", lw=1.0)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("log10( band RMS ratio base/hull )")
    ax.set_xlabel("dynamic natural frequency f_n,dyn [Hz]")
    ax.set_ylabel("damping ratio zeta")
    ax.set_title("QZS design map (ice band %.1f-%.1f Hz); hatched = feasible" % (ice_lo, ice_hi))
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_design_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------- fig 5: life vs ice load
    life_conv = [sev[s]["conventional"]["fatigue"]["life_years"] for s in labels]
    life_qzs = [sev[s]["qzs"]["fatigue"]["life_years"] for s in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.semilogy(x, life_conv, "o--", color="#c53030", label="conventional mount")
    ax.semilogy(x, life_qzs, "s-", color="#2f855a", label="QZS mount")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("pipe fatigue life [years]")
    ax.set_xlabel("ice severity (increasing ice load)")
    ax.grid(True, which="both", alpha=0.3)
    ax2 = ax.twinx()
    ax2.semilogy(x, [sev[s]["qzs"]["stress_rms_MPa"] for s in labels], "^:", color="#2b6cb0", label="QZS pipe stress RMS")
    ax2.semilogy(x, [sev[s]["conventional"]["stress_rms_MPa"] for s in labels], "v:", color="#dd6b20", label="conv. pipe stress RMS")
    ax2.set_ylabel("pipe stress RMS [MPa]")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower left")
    ax.set_title("fatigue life vs ice load (monotone decreasing)")
    fig.tight_layout()
    fig.savefig(FIG / "fig5_life_vs_ice_load.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------- fig 6: S-N curve
    m1, m2 = sn["m1"], sn["m2"]
    n1 = np.logspace(3.0, np.log10(sn["n_transition"]), 200)
    n2 = np.logspace(np.log10(sn["n_transition"]), 12.0, 200)
    s1 = 10 ** ((sn["log_K1"] - np.log10(n1)) / m1)
    s2 = 10 ** ((sn["log_K2"] - np.log10(n2)) / m2)
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.loglog(n1, s1, "k-", lw=1.8, label="DNV-RP-C203 D (repo) m1=%g m2=%g" % (m1, m2))
    ax.loglog(n2, s2, "k-", lw=1.8)
    ax.axhline(
        sn["stress_at_1e7_cycles_MPa"],
        color="grey",
        ls=":",
        lw=1.0,
        label="S at 1e7 = %.1f MPa" % sn["stress_at_1e7_cycles_MPa"],
    )
    colors = {"light": "#4299e1", "design": "#2f855a", "heavy": "#c53030"}
    for s in labels:
        fat = sev[s]["qzs"]["fatigue"]
        seq = fat["equivalent_stress_range_1e7_MPa"]
        if seq and seq > 0:
            ax.plot([1e7], [seq], "o", color=colors[s], ms=8, label="%s: seq,1e7=%.2e MPa (QZS)" % (s, seq))
        smax = fat["max_stress_range_MPa"]
        if smax > 0:
            ax.plot([max(fat["rainflow_cycles"], 1.0)], [smax], "x", color=colors[s], ms=7)
    ax.set_xlabel("cycles to failure N")
    ax.set_ylabel("stress range [MPa]")
    ax.set_title("DNV-RP-C203 D curve vs operating stress ranges (QZS)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "fig6_sn_curve_ranges.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --------------------------------------------------- fig 7: tolerance/stroke
    tol = result.tolerance
    stroke = result.stroke
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))
    axes[0].plot(
        [r["delta_alpha"] for r in tol["rows"]],
        [r["rms_ratio_base_over_hull"] for r in tol["rows"]],
        "o-",
        color="#2f855a",
    )
    axes[0].axhline(1.0, color="r", ls="--", lw=1.2, label="criterion: ratio = 1")
    axes[0].set_xlabel("delta alpha (stiffness-cancellation error)")
    axes[0].set_ylabel("band RMS ratio base/hull")
    axes[0].set_title("stiffness-cancellation tolerance")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    names = [r["label"] for r in stroke["rows"]]
    vals = [r["relative_displacement_rms_m"] * 1e3 for r in stroke["rows"]]
    axes[1].bar(names, vals)
    axes[1].axhline(design.stroke_m * 1e3, color="k", ls="--", lw=1.2, label="stroke %.0f mm" % (design.stroke_m * 1e3))
    axes[1].axhline(
        design.linearity_amplitude_m() * 1e3,
        color="grey",
        ls=":",
        lw=1.2,
        label="NS linear range %.3f mm" % (design.linearity_amplitude_m() * 1e3),
    )
    axes[1].set_yscale("log")
    axes[1].set_ylabel("relative displacement RMS [mm]")
    axes[1].set_title("stroke audit: ice band vs wave/whipping envelope")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, which="both", axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig7_tolerance_stroke.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("payload:", out)
    print("figures:", sorted(p.name for p in FIG.glob("*.png")))
    print(result.summary())
    for c in result.criteria:
        print(c["id"], "PASS" if c["passed"] else "FAIL")
    return out


if __name__ == "__main__":
    main()