"""Reproduce the p360w1-n05 moving-load / FNV node: JSON payload + figures.

Usage::

    PYTHONPATH=src python3 scripts/p360w1_n05_moving_load_fnv.py

Writes
------
``artifacts/p360w1_n05/moving_load_fnv_report.json``
    every acceptance number (A1 - A4, geometry, natural frequencies, studies)
``artifacts/p360w1_n05/fig*.png``
    the report figures
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tensorphysics.ice_moving_load import (  # noqa: E402
    FREE_FREE,
    SIMPLY_SUPPORTED,
    analytic_point_load_response,
    band_amplitude,
    build_contact_history,
    build_reference_case,
    dominant_frequency_hz,
    evaluate_acceptance,
    run_moving_point_load,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "artifacts", "p360w1_n05")
FIG_DPI = 130


def _savefig(fig, name: str, attempts: int = 12) -> str:
    """Save a figure, retrying while the shared disk is transiently full."""
    path = os.path.abspath(os.path.join(OUT_DIR, name))
    fig.tight_layout()
    for attempt in range(attempts):
        try:
            fig.savefig(path, dpi=FIG_DPI)
            break
        except OSError as exc:  # ENOSPC on a shared, heavily used volume
            print(f"  retry {name}: {exc} (attempt {attempt + 1}/{attempts})", flush=True)
            time.sleep(20.0)
    else:
        raise RuntimeError(f"could not write {path}: disk stayed full")
    plt.close(fig)
    print(f"  wrote {path}")
    return path


def fig_mode_shapes(case, payload):
    beam = case.beam
    basis = beam.modal_basis(support=FREE_FREE, n_modes=3)
    x = np.linspace(0.0, beam.length_m, 600)
    shapes = basis.shapes(x)
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    for i in range(3):
        ax.plot(x, shapes[i], label=f"mode {i+1}: {basis.frequencies_hz[i]:.3f} Hz")
    ax.set_xlabel("x [m] (0 = AP, 150 = FP)")
    ax.set_ylabel(r"$\phi_n(x)$ [-] (unit max)")
    ax.set_title(
        "Free-free hull-girder modes, EI = 7.0e12 N$\\cdot$m$^2$, "
        r"$\rho A$ = 1.218e5 kg/m"
    )
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    ax.axhline(0.0, color="k", lw=0.6)
    fem = payload["natural_frequencies"]
    ax.text(
        0.02,
        0.97,
        "analytic vs Hermite FE: max rel. err "
        f"{100*max(fem['relative_error']):.4f} %",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
    )
    return _savefig(fig, "fig1_mode_shapes.png")


def fig_classic_moving_load(case, payload):
    a2 = payload["A2"]
    c = a2["critical_speed_m_s"]
    load = case.static_midspan_load_n()
    ratios = [0.5, 0.8, 1.0, 1.2]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for ratio in ratios:
        v = ratio * c
        resp = run_moving_point_load(
            case.beam, load, v, n_modes=case.n_modes, zeta=0.0,
            support=SIMPLY_SUPPORTED, monitor_x_m=case.monitor_x,
        )
        ref = analytic_point_load_response(
            case.beam, load, v, resp.time_s, n_modes=300, monitor_x_m=case.monitor_x
        )["displacement_m"]
        axes[0].plot(resp.time_s * 1e3, resp.displacement_m * 1e3, "-", lw=1.6,
                     label=f"V/Vcr={ratio:g} (num)")
        axes[0].plot(resp.time_s * 1e3, ref * 1e3, "--", lw=0.9, color=axes[0].lines[-1].get_color())
    static_u = a2["static_midspan_m"]
    axes[0].set_xlabel("t [ms]")
    axes[0].set_ylabel("midspan displacement [mm]")
    axes[0].set_title("A2: traversing force, simply supported beam\nnumeric (solid) vs closed form (dashed)")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)
    pts = a2["points"]
    axes[1].plot([p["speed_ratio"] for p in pts], [p["dynamic_amplification"] for p in pts], "o-")
    axes[1].axhline(1.0, color="k", lw=0.6, ls=":")
    axes[1].set_xlabel(r"$V/V_{cr}$, $V_{cr}=\omega_1 L/\pi$")
    axes[1].set_ylabel(r"dynamic amplification $u_{max}/u_{static}$")
    axes[1].set_title("A2: amplification (Timoshenko/Fryba maximum $\\approx$ 1.7)")
    axes[1].grid(alpha=0.3)
    return _savefig(fig, "fig2_classic_moving_load.png")


def fig_fnv_scan(payload):
    a3 = payload["A3"]
    res = a3["results"]
    ratios = [r["speed_ratio_V_over_Vcr"] for r in res]
    fnv = [r["fnv"] for r in res]
    fnv_raw = [r["fnv_raw"] for r in res]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    ax = axes[0]
    ax.axhspan(0.0, 0.5, color="tab:green", alpha=0.12, label="habitable (< 0.5)")
    ax.axhspan(0.5, 1.0, color="tab:orange", alpha=0.12, label="tolerable (0.5 - 1.0)")
    ax.axhspan(1.0, max(max(fnv), max(fnv_raw)) * 1.15, color="tab:red", alpha=0.10,
               label="unacceptable (> 1.0)")
    ax.plot(ratios, fnv, "o-", color="tab:blue", label="FNV (narrowband, used)")
    ax.plot(ratios, fnv_raw, "s--", color="tab:gray", label="FNV from max|a(t)| (raw)")
    for r, f in zip(ratios, fnv):
        ax.annotate(f"{f:.2f}", (r, f), textcoords="offset points", xytext=(4, 6), fontsize=8)
    ax.axvline(1.0, color="k", lw=0.6, ls=":")
    ax.set_xlabel(r"$V/V_{cr}$ with $V_{cr} = s\,f_1$")
    ax.set_ylabel("FNV = a / (g f)")
    ax.set_title("A3: FNV speed scan, 1 m broken ice, $\\zeta$ = 0.5 %")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(ratios, [r["f_dominant_hz"] for r in res], "o-", label="dominant response $f$")
    ax.plot(ratios, [r["f_ice_hz"] for r in res], "s--", label="$f_{ice}=V/s$")
    ax.axhline(a3["first_natural_frequency_hz"], color="tab:red", lw=1.0, ls=":",
               label=f"$f_1$ = {a3['first_natural_frequency_hz']:.3f} Hz")
    ax.set_xlabel(r"$V/V_{cr}$")
    ax.set_ylabel("frequency [Hz]")
    ax.set_title("A3: excitation vs response frequency")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    return _savefig(fig, "fig3_fnv_scan.png")


def fig_ice_response(case, collected):
    responses = collected["scan"]
    resp, history = responses[1.0]
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 8.6), sharex=False)
    ax = axes[0]
    ax.plot(history.time_s, history.force_n / 1e6, lw=0.8, color="tab:brown")
    ax.set_ylabel("contact force [MN]")
    ax.set_xlim(0, min(20.0, history.time_s[-1]))
    ax.set_title("Ice contact force history at $V_{cr}$ (floe-by-floe crushing, contact sweep)")
    ax.grid(alpha=0.3)
    ax = axes[1]
    mask = resp.time_s <= 20.0
    ax.plot(resp.time_s[mask], resp.displacement_m[mask] * 1e3, lw=0.9, color="tab:blue")
    ax.set_ylabel("midspan $u$ [mm]")
    ax.grid(alpha=0.3)
    ax = axes[2]
    ax.plot(resp.time_s, resp.acceleration_m_s2, lw=0.6, color="tab:red")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("midspan $\\ddot u$ [m/s$^2$]")
    ax.grid(alpha=0.3)
    return _savefig(fig, "fig4_ice_response.png")


def fig_spectrum(case, collected, payload):
    responses = collected["scan"]
    resp, history = responses[1.0]
    mask = resp.time_s >= case.steady_start_s
    t, a = resp.time_s[mask], resp.acceleration_m_s2[mask]
    dt = float(np.mean(np.diff(t)))
    y = (a - a.mean()) * np.hanning(a.size)
    spec = 2.0 * np.abs(np.fft.rfft(y)) / (np.sum(np.hanning(a.size)) / 2.0)
    freqs = np.fft.rfftfreq(a.size, dt)
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.semilogy(freqs, np.maximum(spec, 1e-6), lw=0.8)
    f1 = payload["A3"]["first_natural_frequency_hz"]
    ax.axvline(f1, color="tab:red", ls=":", lw=1.0, label=f"$f_1$ = {f1:.3f} Hz")
    for h, lab in ((2, "$2f_1$"), (3, "$3f_1$")):
        ax.axvline(h * f1, color="tab:gray", ls=":", lw=0.8)
        ax.text(h * f1, spec.max() * 0.5, lab, fontsize=8, color="tab:gray")
    ax.set_xlim(0, 8.0)
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("acceleration amplitude [m/s$^2$]")
    ax.set_title("Midspan acceleration spectrum at $V_{cr}$ (steady window)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _savefig(fig, "fig5_acceleration_spectrum.png")


def fig_energy(case, collected, payload):
    responses = collected["scan"]
    resp, _ = responses[1.0]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]
    ax.plot(resp.time_s, resp.work_input_j / 1e6, label="input work $W$")
    ax.plot(resp.time_s, resp.damping_dissipated_j / 1e6, label="damping $D$")
    ax.plot(resp.time_s, resp.strain_energy_j / 1e6, label="strain $SE$")
    ax.plot(resp.time_s, resp.kinetic_energy_j / 1e6, label="kinetic $KE$")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("energy [MJ]")
    ax.set_title("A4: energy ledger at $V_{cr}$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[1]
    for ratio in sorted(responses):
        r, _ = responses[ratio]
        ax.semilogy(r.time_s, np.maximum(np.abs(r.energy_residual_j), 1e-3), lw=0.7,
                    label=f"V/Vcr={ratio:g}")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("$|W - D - SE - KE|$ [J]")
    ax.set_title(
        "A4: absolute residual (worst relative %.2e)"
        % payload["A4"]["worst_relative_residual"]
    )
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    return _savefig(fig, "fig6_energy_balance.png")


def fig_studies(payload):
    reg = payload["studies"]["floe_regularity"]["rows"]
    swp = payload["studies"]["sweep_vs_stationary"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]
    ax.plot([r["floe_length_cv"] for r in reg], [r["fnv"] for r in reg], "o-")
    ax.set_xlabel("floe-length coefficient of variation [-]")
    ax.set_ylabel("FNV at $V_{cr}$")
    ax.set_title("Resonance quality vs floe-spacing regularity")
    ax.grid(alpha=0.3)
    ax = axes[1]
    labels = ["sweeping\ncontact", "stationary\ncontact", "single force\ntransit"]
    vals = [
        swp["sweeping_contact"]["fnv"],
        swp["stationary_contact"]["fnv"],
        float("nan"),
    ]
    bars = ax.bar(labels, [v for v in vals], color=["tab:blue", "tab:orange", "tab:gray"])
    for b, v in zip(bars, vals):
        if not math.isnan(v):
            ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=8)
    daf = swp["single_force_transit"]["dynamic_amplification"]
    ax.annotate(
        f"DAF = {daf:.3f}\n(quasi-static)",
        (2, 0.05),
        ha="center",
        fontsize=8,
    )
    ax.set_ylabel("FNV (band)")
    ax.set_title("Moving-load effect isolated at $V_{cr}$")
    ax.grid(alpha=0.3, axis="y")
    return _savefig(fig, "fig7_studies.png")


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    case = build_reference_case()
    collected: dict = {}
    payload = evaluate_acceptance(case, verbose=True, collect=collected)
    payload["artifacts"] = {"directory": os.path.relpath(OUT_DIR)}
    payload["environment"] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
    }

    json_path = os.path.abspath(os.path.join(OUT_DIR, "moving_load_fnv_report.json"))
    with open(json_path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
    print(f"wrote {json_path}")

    print("generating figures ...")
    figs = {}
    figs["mode_shapes"] = fig_mode_shapes(case, payload)
    figs["classic_moving_load"] = fig_classic_moving_load(case, payload)
    figs["fnv_scan"] = fig_fnv_scan(payload)
    figs["ice_response"] = fig_ice_response(case, collected)
    figs["acceleration_spectrum"] = fig_spectrum(case, collected, payload)
    figs["energy_balance"] = fig_energy(case, collected, payload)
    figs["studies"] = fig_studies(payload)
    payload["figures"] = {k: os.path.basename(v) for k, v in figs.items()}

    with open(json_path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
    print(f"updated {json_path} with figure names")

    print("\n=== criteria ===")
    for key, value in payload["criteria"].items():
        print(f"  {key:9s} {'PASS' if value else 'FAIL'}")
    print("all_passed:", payload["all_passed"])
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())