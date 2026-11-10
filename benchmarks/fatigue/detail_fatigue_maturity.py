#!/usr/bin/env python3
"""Detail-fatigue maturity benchmark for node p246w2-n09 (TensorPhysics).

Runs the C1..C7 acceptance suite, generates the report figures and writes the
machine-readable artifacts:

    artifacts/p246w2_n09/detail_fatigue_acceptance.json
    artifacts/p246w2_n09/figures/*.png
    artifacts/p246w2_n09/figure_manifest.json   (path + sha256 of every figure)

Usage
-----
    python benchmarks/fatigue/detail_fatigue_maturity.py [--quick] [--out DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from tensorphysics.fatigue.detail_maturity import (        # noqa: E402
    fatigue_curve,
    loglog_fit,
    reference_fit,
    simulate_test_series,
    miner_damage,
    sequence_effect_probe,
    run_acceptance,
)
from tensorphysics.fatigue.detail_maturity import plane_stress_fem as fem   # noqa: E402
from tensorphysics.fatigue.detail_maturity import hotspot_extrapolation as hsx  # noqa: E402


# ----------------------------------------------------------------------------
def fig_sn_curve(path: Path) -> dict:
    """S-N curve, log-log fit and simulated test series."""
    curve = fatigue_curve(90.0)
    stress, life = curve.sample(1e5, 1e7, 80)
    fit = reference_fit(curve, 1e5, 1e7, 41)
    levels = [float(v) for v in curve.stress_for_life(np.logspace(5.0, 7.0, 5))]
    s_hi, n_hi = simulate_test_series(curve, levels, specimens_per_level=12,
                                      sigma_log_n=0.2, seed=20260909)
    fit_hi = loglog_fit(s_hi, n_hi)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.loglog(stress, life, "k-", lw=2, label="IIW FAT90 design curve (m=3)")
    s_fit = np.logspace(np.log10(stress.min()), np.log10(stress.max()), 50)
    ax.loglog(s_fit, fit.life(s_fit), "r--", lw=1.2,
              label=f"log-log fit: m={fit.slope_m:.3f}, R²={fit.r_squared:.4f}")
    ax.loglog(s_hi, n_hi, "o", ms=3.5, alpha=0.55, color="tab:blue",
              label=f"simulated weld tests (sd logN=0.2): m={fit_hi.slope_m:.2f}")
    ax.axvline(90.0, color="gray", ls=":", lw=1)
    ax.annotate("FAT90\n(Δσ=90 MPa @ 2×10⁶)", xy=(90.0, 2e6), xytext=(112, 6e5),
                fontsize=9, arrowprops=dict(arrowstyle="->", lw=0.8, color="gray"))
    ax.axhline(1e7, color="green", ls=":", lw=1)
    ax.text(60, 1.1e7, "knee N=10⁷ (m→5 beyond)", fontsize=8, color="green")
    ax.set_xlabel("stress range Δσ [MPa]")
    ax.set_ylabel("cycles to failure N")
    ax.set_title("C1: S-N curve and log-log linearity (verified band 10⁵–10⁷ cycles)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"m_design": fit.slope_m, "r2_design": fit.r_squared,
            "m_tests": fit_hi.slope_m, "r2_tests": fit_hi.r_squared}


def fig_kt_validation(path: Path, report: dict) -> dict:
    """FEM vs analytical Kt for the hole and the butt-weld toe."""
    c2 = report["criteria"]["C2"]
    labels, fem_v, ana_v = [], [], []
    labels.append("open hole\n(Kirsch)")
    fem_v.append(c2["hole"]["kt_fem"])
    ana_v.append(3.0)
    for row in c2["butt_joint_toe"]:
        labels.append(f"butt toe\nh={row['h_mm']:g}, ρ={row['rho_mm']:g}")
        fem_v.append(row["kt_fem"])
        ana_v.append(row["kt_inglis"])

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    w = 0.36
    ax.bar(x - w / 2, ana_v, w, label="analytical (Kirsch / Inglis)",
           color="tab:gray", alpha=0.85)
    ax.bar(x + w / 2, fem_v, w, label="T6 plane-stress FEM",
           color="tab:orange", alpha=0.9)
    for xi, a, f in zip(x, ana_v, fem_v):
        err = abs(f - a) / a * 100.0
        ax.text(xi + w / 2, f + 0.03, f"{f:.3f}\n({err:+.1f}%)", ha="center",
                fontsize=8)
        ax.text(xi - w / 2, a + 0.03, f"{a:.3f}", ha="center", fontsize=8)
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xticks(x, labels, fontsize=9)
    ax.set_ylabel("stress concentration factor K$_t$")
    ax.set_title("C2: K$_t$ from real FEM vs analytical anchors (acceptance ≤10%)")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"kt_fem": fem_v, "kt_analytical": ana_v}


def fig_hotspot(path: Path, report: dict) -> dict:
    """Surface-stress profile, hot-spot sampling points and extrapolation."""
    t = 20.0
    profiles = {}
    for fl in (t / 4.0, t / 16.0):
        sol = fem.solve_t_joint(1.0, L=200.0, t=t, a=60.0, t_a=10.0, first_layer=fl)
        m = sol.model
        x_toe = m.meta["x_attach"]
        xs, sig = fem.surface_stress_profile(sol, 0, x_range=(0.0, x_toe - 1e-9))
        res = hsx.extrapolate_from_profile(xs, sig, x_toe, t)
        profiles[fl] = (xs - x_toe, sig, res)

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    colors = {t / 4.0: "tab:blue", t / 16.0: "tab:red"}
    for fl, (dx, sig, res) in profiles.items():
        ax.plot(dx, sig, "-", color=colors[fl], lw=1.6,
                label=f"FEM surface stress, element t/{int(t/fl)}")
        ax.plot(res.sample_x - x_toe, res.stresses, "o", color=colors[fl], ms=5)
        ax.axhline(res.hot_spot_stress, color=colors[fl], ls="--", lw=1.0, alpha=0.8)
        ax.text(-52, res.hot_spot_stress + 0.05,
                f"hot-spot (extrap.) = {res.hot_spot_stress:.3f} MPa",
                color=colors[fl], fontsize=8)
    for p in hsx.POINTS_IIW_QUADRATIC:
        ax.axvline(-p * t, color="gray", ls=":", lw=0.8)
        ax.text(-p * t + 0.6, 2.05, f"{p}t", fontsize=8, color="gray")
    ax.axvline(0.0, color="k", lw=1.2)
    ax.text(1.5, 2.05, "weld toe", fontsize=8)
    ax.set_xlim(-60, 30)
    ax.set_ylim(0.85, 2.25)
    ax.set_xlabel("distance from weld toe [mm]")
    ax.set_ylabel("σ$_{xx}$ on plate surface / σ$_{nom}$")
    ax.set_title("C4/C6: 0.4t/0.9t/1.4t hot-spot extrapolation (T-joint, main-plate tension)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {f"hot_spot_t{int(t/fl)}": profiles[fl][2].hot_spot_stress for fl in profiles}


def fig_mesh_convergence(path: Path, report: dict) -> dict:
    c6 = report["criteria"]["C6"]
    sizes = [row["element_size_mm"] for row in c6["levels"]]
    values = c6["hot_spot_values"]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot(sizes, values, "o-", color="tab:purple", lw=1.6, ms=7)
    for s, v in zip(sizes, values):
        ax.annotate(f"{v:.4f}", xy=(s, v), xytext=(4, 6),
                    textcoords="offset points", fontsize=8)
    ch = c6["changes_pct"]
    ax.set_xlabel("first element size at the toe [mm]")
    ax.set_ylabel("extrapolated hot-spot stress / σ$_{nom}$")
    ax.set_title("C6: mesh sensitivity t/4 → t/8 → t/16  "
                 f"(Δ = {ch['t4_to_t16']:.2f} %, limit 5 %)")
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"sizes_mm": sizes, "hot_spot": values, "changes_pct": ch}


def fig_miner(path: Path) -> dict:
    """Two-block damage accumulation and the sequence-invariance probe."""
    curve = fatigue_curve(90.0)
    ds1, n1, ds2, n2 = 140.0, 2e4, 85.0, 1.5e5
    blocks = np.arange(0, 13)
    d_block = miner_damage(curve, [ds1, ds2], [n1, n2]).total_damage
    dmg_curve = blocks * d_block
    probe = sequence_effect_probe(curve, ds1, n1, ds2, n2)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    ax = axes[0]
    ax.plot(blocks, dmg_curve, "o-", color="tab:blue", label="D = Σ n$_i$/N$_i$")
    ax.axhline(1.0, color="r", ls="--", lw=1.2, label="D = 1 (failure)")
    lam = 1.0 / d_block
    ax.axvline(lam, color="k", ls=":", lw=1.2,
               label=f"predicted failure at {lam:.2f} blocks")
    ax.set_xlabel("block repetitions")
    ax.set_ylabel("Miner damage D")
    ax.set_title(f"C3/C5: two-level block spectrum\n"
                 f"({ds1:g} MPa×{n1:.0e} + {ds2:g} MPa×{n2:.0e})")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.bar(["high→low", "low→high"], [probe.damage_high_low, probe.damage_low_high],
           color=["tab:orange", "tab:green"], alpha=0.9)
    for i, v in enumerate([probe.damage_high_low, probe.damage_low_high]):
        ax.text(i, v + 0.002, f"{v:.6f}", ha="center", fontsize=9)
    ax.set_ylabel("damage per block D")
    ax.set_title("C7: Miner is load-sequence independent\n"
                 f"(Δ = {probe.relative_difference:.1e} — documented limitation)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"damage_per_block": d_block, "blocks_to_failure": float(lam),
            "sequence_relative_difference": probe.relative_difference}


def fig_stress_field(path: Path) -> dict:
    """σ_xx field of the T-joint (coarse mesh) around the weld toe."""
    sol = fem.solve_t_joint(1.0, L=200.0, t=20.0, a=60.0, t_a=10.0, first_layer=2.5)
    m = sol.model
    tri = m.elements[:, :3]
    x, y = m.nodes[:, 0], m.nodes[:, 1]
    sxx = sol.nodal_stress[:, 0]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    tp = ax.tripcolor(x, y, tri, sxx, shading="gouraud", cmap="viridis")
    fig.colorbar(tp, ax=ax, label="σ$_{xx}$ / σ$_{nom}$")
    x_toe = m.meta["x_attach"]
    ax.plot([x_toe], [m.meta["t"] / 2], "r^", ms=9, label="weld toe")
    ax.set_aspect("equal")
    ax.set_xlim(x_toe - 60, x_toe + 60)
    ax.set_ylim(-2, 50)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title("T-joint σ$_{xx}$ field: load path around the transverse attachment")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"peak_sxx": float(np.max(sxx)), "node_count": int(len(x))}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="artifacts/p246w2_n09")
    args = ap.parse_args()

    out = REPO / args.out
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    print("[1/4] running acceptance suite ...")
    report = run_acceptance(quick=args.quick)
    acc_path = out / "detail_fatigue_acceptance.json"
    acc_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"      all_pass={report['summary']['all_pass']} "
          f"({report['summary']['n_pass']}/{report['summary']['n_criteria']})")

    print("[2/4] generating figures ...")
    manifest = {}
    manifest["fig_sn_curve"] = fig_sn_curve(figs / "fig1_sn_curve.png")
    manifest["fig_kt"] = fig_kt_validation(figs / "fig2_kt_validation.png", report)
    manifest["fig_hotspot"] = fig_hotspot(figs / "fig3_hotspot_extrapolation.png", report)
    manifest["fig_mesh"] = fig_mesh_convergence(figs / "fig4_mesh_convergence.png", report)
    manifest["fig_miner"] = fig_miner(figs / "fig5_miner_two_block.png")
    manifest["fig_field"] = fig_stress_field(figs / "fig6_t_joint_field.png")

    print("[3/4] hashing artifacts ...")
    hashes = {p.name: sha256(p) for p in sorted(figs.glob("*.png"))}
    (out / "figure_manifest.json").write_text(
        json.dumps({"figures": hashes, "derived": manifest,
                    "acceptance_all_pass": report["summary"]["all_pass"]},
                   indent=2), encoding="utf-8")

    print("[4/4] summary")
    for k, v in report["criteria"].items():
        print(f"      {k}: pass={v['pass']}")
    print(f"      artifacts -> {out}")
    return 0 if report["summary"]["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())