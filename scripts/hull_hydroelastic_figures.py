"""Figures for the springing / whipping report (reads artifacts/m5549/results.json)."""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _load(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def fig_modes(data: dict, out: str) -> str:
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.4), dpi=95)
    for name, color in (("KCS", "C0"), ("KVLCC2", "C3")):
        case = data["cases"][name]
        dry = [m["frequency_hz"] for m in case["dry_modes"]]
        wet = [m["wet_hz"] for m in case["wet_modes"]]
        ax[0].plot(range(1, len(dry) + 1), dry, "o--", color=color, label=f"{name} dry")
        ax[0].plot(range(1, len(wet) + 1), wet, "s-", color=color, alpha=0.65, label=f"{name} wet")
    ax[0].set_xlabel("mode number")
    ax[0].set_ylabel("frequency [Hz]")
    ax[0].set_title("dry vs added-mass corrected modes")
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    dx = data["cases"]["KCS"]["diagnostics"]["dry_shapes"]
    wx = data["cases"]["KCS"]["diagnostics"]["wet_shapes"]
    x = np.asarray(dx["x"])
    for i in range(3):
        ax[1].plot(x, np.asarray(dx["modes"][i]) / np.abs(dx["modes"][i]).max(), "--", lw=0.9,
                   label=f"dry {i+1}")
        ax[1].plot(x, np.asarray(wx["modes"][i]) / np.abs(wx["modes"][i]).max(), lw=1.3,
                   label=f"wet {i+1}")
    ax[1].set_xlabel("x [m]")
    ax[1].set_ylabel("normalised mode shape")
    ax[1].set_title("KCS free-free mode shapes")
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_added_mass(data: dict, out: str) -> str:
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.2), dpi=95)
    prof = data["cases"]["KCS"]["diagnostics"]["a33_profile"]
    x = np.asarray(prof["x"])
    ax[0].plot(x, np.asarray(prof["a33"]) / 1e3, lw=1.2, label="a33 (2-D BEM, phi=0)")
    ax[0].plot(x, np.asarray(prof["mass_per_length"]) / 1e3, lw=1.2, label="hull mass / length")
    ax[0].set_xlabel("x [m]")
    ax[0].set_ylabel("10^3 kg/m")
    ax[0].set_title("KCS sectional added mass vs structural mass")
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    rows = data["gates"]["added_mass"]["circle_rows"]
    n = [r["n_panels"] for r in rows]
    err = [abs(r["error_heave_pct"]) for r in rows]
    ax[1].loglog(n, err, "o-", label="|error| circle heave")
    ax[1].axhline(5.0, ls="--", c="r", lw=1, label="5 % gate")
    ell = abs(data["gates"]["added_mass"]["ellipse"]["error_heave_pct"])
    ax[1].plot([max(n)], [ell], "s", c="C2", label="ellipse (rho pi a^2)")
    ax[1].set_xlabel("panels")
    ax[1].set_ylabel("relative error [%]")
    ax[1].set_title("2-D added mass vs analytic rho pi a^2")
    ax[1].grid(alpha=0.3, which="both")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_springing(data: dict, out: str) -> str:
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.2), dpi=95)
    sp = data["cases"]["KCS"]["diagnostics"]["springing"]
    om = np.asarray(sp["omega"])
    ss = np.asarray(sp["S_sigma_MPa2s"])
    ax[0].loglog(om, np.maximum(ss, 1e-20), lw=1.0, label="S_sigma (flexible hull)")
    sz = np.asarray(sp["S_zeta"])
    ax[0].loglog(om, np.maximum(sz, 1e-20) * 1e3, lw=0.8, ls="--", label="ITTC spectrum x1000")
    for w in data["cases"]["KCS"]["wet_modes"][:3]:
        ax[0].axvline(2 * np.pi * w["wet_hz"], ls=":", lw=0.7)
    ax[0].set_xlabel("omega [rad/s]")
    ax[0].set_ylabel("density")
    ax[0].set_title("KCS springing stress spectrum (design speed)")
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    for name, color in (("KCS", "C0"), ("KVLCC2", "C3")):
        rows = data["cases"][name]["coupling"]["rows"]
        fn = [r["Fn"] for r in rows]
        ax[1].plot(fn, [r["delta_pct"] for r in rows], "o-", color=color, label=f"{name} coupled")
        ax[1].plot(fn, [100 * (r["elastic_amplification"] - 1) for r in rows], "s--", color=color,
                   alpha=0.6, label=f"{name} flexibility only")
    ax[1].set_xlabel("Froude number Fn")
    ax[1].set_ylabel("midship moment increase [%]")
    ax[1].set_title("decoupled -> coupled (rigid vs flexible+slamming)")
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_whipping(data: dict, out: str) -> str:
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.2), dpi=95)
    for name, color in (("KCS", "C0"), ("KVLCC2", "C3")):
        h = data["cases"][name]["diagnostics"]["whipping_history"]
        ax[0].plot(h["t"], h["stress_MPa"], lw=0.9, color=color, label=name)
    ax[0].set_xlim(0, 5)
    ax[0].set_xlabel("time [s]")
    ax[0].set_ylabel("midship deck stress [MPa]")
    ax[0].set_title("whipping transient (one slam, 5 MN peak)")
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    for name, color in (("KCS", "C0"), ("KVLCC2", "C3")):
        rows = data["cases"][name]["whipping_sensitivity"]
        f = sorted({r["peak_force_MN"] for r in rows})
        ax[1].plot(f, [r["peak_stress_MPa"] for r in rows if r["duration_s"] == 0.25], "o-",
                   color=color, label=f"{name} 0.25 s")
        ax[1].plot(f, [r["peak_stress_MPa"] for r in rows if r["duration_s"] == 0.12], "s--",
                   color=color, alpha=0.6, label=f"{name} 0.12 s")
    ax[1].set_xlabel("slamming peak force [MN]")
    ax[1].set_ylabel("peak whipping stress [MPa]")
    ax[1].set_title("whipping severity sensitivity")
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_gates(data: dict, out: str) -> str:
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.2), dpi=95)
    g = data["gates"]["dry_modes"]
    names = ["EB roots", "shear sign", f"L/h=20 m1"]
    vals = [g["max_relative_error_pct"], 0.0, g["slender"]["20"]["drop_pct"][0]]
    tols = [3.0, 0.0, 1.0]
    ax[0].bar(np.arange(3) - 0.2, vals, width=0.4, label="value [%]")
    ax[0].bar(np.arange(3) + 0.2, tols, width=0.4, label="gate [%]")
    ax[0].set_xticks(range(3))
    ax[0].set_xticklabels(names, fontsize=7)
    ax[0].set_ylabel("[%]")
    ax[0].set_title("dry-mode gate (gate 1)")
    ax[0].grid(alpha=0.3, axis="y")
    ax[0].legend(fontsize=7)
    for name, color in (("KCS", "C0"), ("KVLCC2", "C3")):
        led = data["cases"][name]["utilisation_ledger"]
        ax[1].bar([0.0, 1.0], [led["utilisation_decoupled"], led["utilisation_coupled"]],
                  width=0.35, color=color, alpha=0.75,
                  label=f"{name}: {led['utilisation_decoupled']:.2f}->{led['utilisation_coupled']:.2f}")
    ax[1].axhline(1.0, ls="--", c="k", lw=1, label="utilisation limit")
    ax[1].set_xticks([0.0, 1.0])
    ax[1].set_xticklabels(["decoupled", "coupled"], fontsize=8)
    ax[1].set_ylabel("utilisation M/(M_u/gamma_m)")
    ax[1].set_title("elastic correction in the section-modulus ledger (gate 6)")
    ax[1].grid(alpha=0.3, axis="y")
    ax[1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=os.path.join("artifacts", "m5549", "results.json"))
    ap.add_argument("--out", dest="outdir", default=os.path.join("artifacts", "m5549"))
    args = ap.parse_args()
    data = _load(args.inp)
    made = []
    for fn, name in (
        (fig_modes, "fig1_modes.png"),
        (fig_added_mass, "fig2_added_mass.png"),
        (fig_springing, "fig3_springing.png"),
        (fig_whipping, "fig4_whipping.png"),
        (fig_gates, "fig5_gates.png"),
    ):
        path = os.path.join(args.outdir, name)
        made.append(fn(data, path))
        print("wrote", path, os.path.getsize(path) // 1024, "kB")


if __name__ == "__main__":
    main()