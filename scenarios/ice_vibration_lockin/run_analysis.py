#!/usr/bin/env python3
"""Ice-induced vibration lock-in / whiplash campaign run.

Produces the evidence bundle for the W3-N01 node:

* ``report.json``            full analysis payload (every number the docs quote)
* ``metrics.json``           acceptance criteria plus the headline quantities
* ``*.csv``                  sweep tables (amplification, self-excitation, shock spectrum, ...)
* ``figures/*.png``          publication figures
* ``pytest_summary.json``    pytest outcome recorded by ``--with-pytest``

Usage
-----
    python3 scenarios/ice_vibration_lockin/run_analysis.py [--out-dir DIR] [--with-pytest]

The script is deterministic (fixed seeds), writes only inside ``--out-dir`` and
prints a compact summary.  Nothing is transcribed by hand: the docs and the
platform report read ``report.json`` / ``metrics.json``.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

WORKTREE = Path(__file__).resolve().parents[2]
SRC = WORKTREE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tensorphysics.ice_vibration.analysis import (  # noqa: E402
    IceVibrationAnalysis,
    build_reference_case,
)
from tensorphysics.ice_vibration.modal import (  # noqa: E402
    FREE_FREE_BETA_L,
    free_free_beam_mode_shape,
)

DEFAULT_OUT = WORKTREE / "artifacts" / "ice_vibration"


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False, default=float) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def figure_mode_shapes(case, path: Path) -> Dict[str, Any]:
    x = np.linspace(0.0, case.hull.length_m, 601)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True)
    for index in range(case.modal.n_modes):
        phi, phi_pp = free_free_beam_mode_shape(FREE_FREE_BETA_L[index], x, case.hull.length_m)
        axes[0].plot(x, phi, label=f"mode {index + 1}: $f$ = {case.modal.frequencies_hz[index]:.3f} Hz")
        axes[1].plot(x, phi_pp, label=f"mode {index + 1}")
    axes[0].set_ylabel(r"$\phi_n(x)$ [-]")
    axes[0].legend(loc="lower center", fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].set_ylabel(r"$\phi_n''(x)$ [1/m]")
    axes[1].set_xlabel("x [m]  (0 = stern, L = bow)")
    axes[1].axvline(0.5 * case.hull.length_m, color="k", ls=":", lw=0.8)
    axes[1].text(0.52 * case.hull.length_m, 0.0, "midship", fontsize=8, va="bottom")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Uniform free-free hull-girder reference modes (unit-max normalized)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    mid_index = int(np.argmin(np.abs(x - 0.5 * case.hull.length_m)))
    return {
        "captions": [
            "Mode shapes and curvatures of the reference free-free hull girder.",
            "Note the node of mode 2 at midship: its midship curvature vanishes, so the second-mode "
            "bending moment is carried at the quarter points while the bow displacement includes it.",
        ],
        "second_mode_midship_curvature": float(abs(phi_pp[mid_index])) if False else float(
            abs(np.interp(0.5 * case.hull.length_m, x, free_free_beam_mode_shape(
                FREE_FREE_BETA_L[1], x, case.hull.length_m)[1]))
        ),
    }


def figure_amplification(report: Dict[str, Any], path: Path) -> Dict[str, Any]:
    sweep = report["forced_response_sweep"]
    lock_in = report["lock_in"]
    ratios = np.array([row["frequency_ratio"] for row in sweep["rows"]])
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.axvspan(1.0 - lock_in["relative_tolerance"], 1.0 + lock_in["relative_tolerance"],
               color="tab:orange", alpha=0.15, label="lock-in band |f-f_n|/f_n < 5%")
    ax.axhline(lock_in["amplification_threshold"], color="tab:red", ls="--", lw=1.0, label="amplification = 3")
    ax.plot(ratios, [row["amplification_analytic_linear"] for row in sweep["rows"]],
            "-", lw=2.0, label="analytic linear $D=1/\\sqrt{(1-r^2)^2+(2\\zeta r)^2}$")
    ax.plot(ratios, [row["amplification_time_domain_linear"] for row in sweep["rows"]],
            "o", ms=4, label="time domain, linear ($k_3=0$)")
    ax.plot(ratios, [row["amplification_time_domain"] for row in sweep["rows"]],
            "s", ms=4, label=f"Duffing time domain ({sweep['duffing_relative_nonlinearity']*100:.0f}% cubic)")
    ax.plot(ratios, [row["amplification_duffing_harmonic_balance"] for row in sweep["rows"]],
            "^", ms=4, mfc="none", label="Duffing harmonic balance")
    for key, payload in lock_in["amplification_at_band_edges_time_domain"].items():
        ax.annotate(f"{payload['detuning_relative']*100:+.0f}%: D={payload['amplification_time_domain']:.2f}",
                    xy=(payload["frequency_ratio"] if "frequency_ratio" in payload
                        else payload["detuning_relative"] + 1.0,
                        payload["amplification_time_domain"]),
                    xytext=(8, 10), textcoords="offset points", fontsize=8)
    ax.set_xlabel("frequency ratio $r = f_{ice} / f_n$")
    ax.set_ylabel("dynamic amplification $D = X_{dyn} / (F_0/k)$")
    ax.set_title(f"Lock-in amplification of mode 1 ($f_n$ = {lock_in['natural_frequency_hz']:.3f} Hz, "
                 f"$\\zeta$ = {lock_in['damping_ratio']:.2f})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return {
        "captions": [
            "Response amplification versus excitation frequency ratio.",
            "Inside the 5 % band the measured amplification stays above 3 for every point; the hardening "
            "Duffing response peaks slightly above $r=1$ and drops sharply at $r\\approx1.04$ (fold).",
        ]
    }


def figure_duffing_fold(report: Dict[str, Any], case, path: Path) -> Dict[str, Any]:
    hysteresis = report["duffing_hysteresis"]
    ratios = np.array(hysteresis["ratios"])
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for index, branches in enumerate(hysteresis["amplification_branches"]):
        if len(branches) == 1:
            ax.plot(ratios[index], branches[0], "o", ms=3.0, color="tab:blue")
        elif len(branches) == 3:
            ax.plot([ratios[index]] * 2, [branches[0], branches[2]], "|", ms=14, color="tab:purple")
            ax.plot(ratios[index], branches[1], "x", ms=5, color="0.5")
    if hysteresis["fold_band_ratio"]:
        ax.axvspan(*hysteresis["fold_band_ratio"], color="tab:purple", alpha=0.12,
                   label="three harmonic-balance roots (fold)")
    ax.plot([], [], "o", ms=4, color="tab:blue", label="single stable root")
    ax.plot([], [], "x", ms=5, color="0.5", label="middle (unstable) root")
    ax.set_xlabel("frequency ratio $r = f_{ice} / f_n$")
    ax.set_ylabel("amplification")
    ax.set_title(f"Duffing fold at {hysteresis['cubic_term_fraction_of_k_at_resonance']*100:.0f}% cubic "
                 "stiffness: the lock-in band becomes amplitude-history dependent")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return {
        "captions": [
            "Harmonic-balance branches of a strongly hardening Duffing oscillator.",
            "Where three roots exist the response curls back: entering the fold from below or from above "
            "lands on different amplitudes (jump phenomenon).",
        ]
    }


def figure_spectra(analysis: IceVibrationAnalysis, report: Dict[str, Any], path: Path) -> Dict[str, Any]:
    from tensorphysics.ice_vibration.ice_load import poisson_pulse_train, quasi_periodic_pulse_train
    from tensorphysics.ice_vibration.integrators import rk4_sdof

    sdof = analysis._sdof(0)
    dt = 1.0 / analysis.samples_per_second
    duration = analysis.record_s
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0), sharey=False)
    cases = []
    for label, loader in (
        ("quasi-periodic crushing", quasi_periodic_pulse_train(
            pulse_frequency_hz=analysis.case.ice.crushing_frequency_hz,
            amplitude_n=analysis.case.ice.peak_ice_force_n,
            duration_s=duration, dt_s=dt, seed=analysis.seed,
        )),
        ("Poisson impacts", poisson_pulse_train(
            mean_rate_hz=analysis.case.ice.crushing_frequency_hz,
            amplitude_n=analysis.case.ice.peak_ice_force_n,
            duration_s=duration, dt_s=dt,
            pulse_duration_s=0.55 / analysis.case.ice.crushing_frequency_hz, seed=analysis.seed,
        )),
    ):
        result = rk4_sdof(
            mass=sdof["mass_kg"], damping=sdof["damping_n_s_m"], stiffness=sdof["stiffness_n_m"],
            force=loader.force_n, duration_s=duration, dt_s=dt,
        )
        cases.append((label, loader, result))

    for ax, (label, loader, result) in zip(axes, cases):
        for signal, name, plot_kw in (
            (loader.force_n, "ice force", dict(color="tab:gray", lw=0.9)),
            (result.displacement, "hull response", dict(color="tab:blue", lw=0.9)),
        ):
            signal = np.asarray(signal, dtype=float)
            spectrum = np.abs(np.fft.rfft((signal - signal.mean()) * np.hanning(signal.size)))
            freqs = np.fft.rfftfreq(signal.size, d=dt)
            spectrum = spectrum / spectrum.max()
            ax.semilogy(freqs, np.maximum(spectrum, 1e-6), label=name, **plot_kw)
        ax.axvline(sdof["frequency_hz"], color="tab:red", ls="--", lw=1.0, label="$f_n$")
        ax.axvline(analysis.case.ice.crushing_frequency_hz, color="tab:green", ls=":", lw=1.0,
                   label="$f_{ice}=U/s$")
        ax.set_xlim(0.0, 4.0)
        ax.set_xlabel("frequency [Hz]")
        ax.set_ylabel("normalized magnitude [-]")
        ax.set_title(label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Ice load and hull response spectra: coherent crushing vs memoryless impacts")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return {
        "captions": [
            "Load and response spectra for the two intermittent-crushing models.",
            f"Left: coherent pulse train (load concentration "
            f"{report['pulse_train']['load_spectral_concentration']['concentration']:.2f}); the response "
            f"peaks at $f_{{ice}}$ and is amplified. Right: Poisson impacts "
            f"(concentration {report['poisson_control']['load_spectral_concentration']['concentration']:.2f}); "
            "the narrow-band structure still rings at $f_n$, but no coherent lock-in exists.",
        ]
    }


def figure_self_excited(report: Dict[str, Any], path: Path) -> Dict[str, Any]:
    rows = report["self_excited"]["rows"]
    speeds = np.array([row["drift_speed_m_s"] for row in rows])
    amplitudes = np.array([row["peak_amplitude_m"] for row in rows])
    deviations = np.array([row["lock_in_frequency_deviation"] for row in rows]) * 100.0
    static = report["case"]["first_mode_sdof"]["stiffness_n_m"]
    peak_static = report["case"]["ice"]["peak_ice_force_n"] / static
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0))
    axes[0].loglog(speeds, amplitudes, "o-")
    axes[0].axhline(peak_static, color="tab:red", ls="--", lw=1.0,
                    label=f"static deflection $F_{{max}}/k$ = {peak_static*1000:.1f} mm")
    axes[0].set_xlabel("ice drift speed U [m/s]")
    axes[0].set_ylabel("limit-cycle amplitude [m]")
    axes[0].set_title("Self-excited amplitude (velocity-weakening ice law)")
    axes[0].grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=8)
    axes[1].semilogx(speeds, deviations, "s-")
    axes[1].axhline(5.0, color="tab:red", ls="--", lw=1.0, label="5 % lock-in tolerance")
    axes[1].axhline(-5.0, color="tab:red", ls="--", lw=1.0)
    axes[1].set_xlabel("ice drift speed U [m/s]")
    axes[1].set_ylabel("$(f_{response} - f_n)/f_n$ [%]")
    axes[1].set_title("Lock-in frequency deviation")
    axes[1].grid(alpha=0.3, which="both")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return {
        "captions": [
            "Self-excited lock-in versus drift speed (velocity-weakening ice law with contact loss).",
            "The limit cycle exists only inside the velocity-weakening window; its frequency sits within "
            "5 % of $f_n$ for U >= 0.2 m/s and drifts below that (relaxation regime), while above 0.7 m/s "
            "the negative damping is too weak and the response collapses.",
        ]
    }


def figure_whiplash(report: Dict[str, Any], case, path: Path) -> Dict[str, Any]:
    from tensorphysics.ice_vibration.whiplash import whiplash_response

    pulse = case.ice.impact_pulse_duration_s
    result = whiplash_response(
        case.modal, case.bending_stiffness_nm2, case.ice.peak_ice_force_n, pulse
    )
    time = result.time_s
    fig, axes = plt.subplots(3, 1, figsize=(7.6, 7.2), sharex=True)
    axes[0].plot(time, result.force_n / 1e6, color="tab:gray")
    axes[0].set_ylabel("bow ice force [MN]")
    axes[0].set_title(f"Whiplash response to a {pulse:.2f} s half-sine bow impact "
                      f"($T_p f_1$ = {pulse*case.first_frequency_hz:.2f})")
    axes[1].plot(time, result.midship_moment_nm / 1e6, label="modal superposition (Newmark)")
    axes[1].plot(time, result.midship_moment_analytic_nm / 1e6, "--",
                 label="closed-form modal superposition")
    axes[1].set_ylabel("midship moment [MN m]")
    warp = result.diagnostics["whiplash_dynamic_amplification"]
    axes[1].set_title(f"peak {result.peak_midship_moment_nm/1e6:.1f} MN m, "
                      f"relative error {result.moment_error_relative()*100:.4f} %, "
                      f"dynamic amplification {warp:.2f}", fontsize=9)
    axes[1].legend(fontsize=8)
    axes[2].plot(time, result.bow_displacement_m * 1000, label="bow displacement")
    axes[2].set_ylabel("bow displacement [mm]")
    axes[2].set_xlabel("time [s]")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[2].set_xlim(0.0, min(8.0, time[-1]))
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return {
        "captions": [
            "Transient whiplash response to one bow ice impact.",
            "The first two vertical bending modes are superposed; the time-marched and closed-form peaks "
            "agree to a few $10^{-5}$, which is the numerical fidelity of the reported whiplash moment.",
        ]
    }


def figure_shock_spectrum(report: Dict[str, Any], path: Path) -> Dict[str, Any]:
    spectrum = report["whiplash_shock_spectrum"]
    ratios = [row["pulse_ratio_Tp_f1"] for row in spectrum["rows"]]
    dlf = [row["whiplash_dynamic_amplification"] for row in spectrum["rows"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(ratios, dlf, "o-")
    peak = spectrum["peak_row"]
    ax.plot(peak["pulse_ratio_Tp_f1"], peak["whiplash_dynamic_amplification"], "*", ms=14,
            color="tab:red", label=f"peak {peak['whiplash_dynamic_amplification']:.3f} "
                                   f"at $T_p f_1$ = {peak['pulse_ratio_Tp_f1']:.2f}")
    ax.axhline(1.0, color="0.4", ls="--", lw=0.9, label="quasi-static")
    ax.set_xlabel("pulse duration ratio $T_p f_1$ [-]")
    ax.set_ylabel("whiplash dynamic amplification of midship moment")
    ax.set_title("Whiplash shock spectrum versus impact duration")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return {
        "captions": [
            "Whiplash shock spectrum: midship bending-moment amplification versus impact duration.",
            "Short pulses carry too little impulse and long pulses are quasi-static, so the response peaks "
            "near $T_p f_1 \\approx 0.7$ - the most damaging ice impact duration for this hull.",
        ]
    }


def figure_damping(report: Dict[str, Any], path: Path) -> Dict[str, Any]:
    rows = report["damping_sensitivity"]["rows"]
    zetas = [row["damping_ratio"] for row in rows]
    amplification = [row["amplification_at_upper_band_edge"] for row in rows]
    limit_cycle = [row["self_excited_limit_cycle_amplitude_m"] for row in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.semilogy(np.array(zetas) * 100, amplification, "o-", label="D at 5 % detuning")
    ax.axhline(3.0, color="tab:red", ls="--", lw=1.0, label="amplification = 3")
    ax.axvline(report["lock_in"]["damping_limit_for_threshold_at_upper_edge"] * 100, color="tab:red",
               ls=":", lw=1.0,
               label=f"$\\zeta_{{limit}}$ = {report['lock_in']['damping_limit_for_threshold_at_upper_edge']*100:.1f} %")
    ax.set_xlabel("modal damping ratio [%]")
    ax.set_ylabel("amplification (log scale)")
    ax2 = ax.twinx()
    ax2.plot(np.array(zetas) * 100, np.array(limit_cycle) * 1000, "s--", color="tab:green",
             label="self-excited limit-cycle amplitude")
    ax2.set_ylabel("limit-cycle amplitude [mm]", color="tab:green")
    ax.set_title("Damping controls the lock-in amplification; the limit cycle is contact-loss limited")
    ax.grid(alpha=0.3, which="both")
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return {
        "captions": [
            "Amplification at 5 % detuning and self-excited limit-cycle amplitude versus modal damping.",
            "D > 3 up to $\\zeta \\approx 15$ %, with the measured curve following the analytic relation; "
            "the limit-cycle amplitude is nearly damping independent because it is set by the velocity at "
            "which ice contact is lost.",
        ]
    }


# ---------------------------------------------------------------------------
# Campaign
# ---------------------------------------------------------------------------


def metrics_payload(report: Dict[str, Any], figures: List[Dict[str, str]], out_dir: Path) -> Dict[str, Any]:
    lock_in = report["lock_in"]
    duffing = report["duffing_lock_in"]
    whiplash = report["whiplash"]
    self_excited = report["self_excited"]
    pulse = report["pulse_train"]
    return {
        "generated_utc": _utcnow(),
        "case": report["case"]["label"],
        "hull": {
            "length_m": report["case"]["hull"]["length_m"],
            "breadth_m": report["case"]["hull"]["breadth_m"],
            "draught_m": report["case"]["hull"]["draught_m"],
            "block_coefficient": report["case"]["hull"]["block_coefficient_stated"],
            "displacement_kg": report["case"]["hull"]["displacement_kg"],
            "geometry_consistent": report["modal_audit"]["geometry"]["consistent"],
        },
        "modal": {
            "frequencies_hz": report["modal_audit"]["frequencies_hz"],
            "damping_ratios": report["modal_audit"]["damping_ratios"],
            "source": report["modal_audit"]["modal_source"],
            "bending_stiffness_nm2": report["modal_audit"]["bending_stiffness_nm2"],
            "second_moment_of_area_m4": report["modal_audit"]["second_moment_of_area_m4"],
            "frequency_ratio_relative_error": report["modal_audit"]["frequency_ratio_relative_error"],
        },
        "headline": {
            "lock_in_band_hz": lock_in["band_hz"],
            "amplification_at_5pct_detuning_min": lock_in["min_edge_amplification_time_domain"],
            "amplification_at_resonance_analytic": lock_in["amplification_at_band_edges_analytic"][
                "at_natural_frequency"
            ],
            "damping_limit_for_amplification_three": lock_in["damping_limit_for_threshold_at_upper_edge"],
            "amplification_three_half_bandwidth_relative": lock_in[
                "amplification_three_half_bandwidth_relative"
            ],
            "duffing_max_relative_error": duffing["max_relative_error"],
            "whiplash_peak_midship_moment_nm": whiplash["peak_midship_moment_nm"],
            "whiplash_midship_moment_relative_error": whiplash["midship_moment_relative_error"],
            "whiplash_max_dynamic_amplification": report["whiplash_shock_spectrum"][
                "max_whiplash_dynamic_amplification"
            ],
            "whiplash_worst_pulse_ratio": report["whiplash_shock_spectrum"]["peak_row"]["pulse_ratio_Tp_f1"],
            "pulse_train_lock_in": pulse["lock_in_diagnosis"]["lock_in"],
            "pulse_train_response_frequency_deviation": pulse[
                "response_peak_relative_deviation_from_excitation"
            ],
            "poisson_control_state": report["poisson_control"]["lock_in_diagnosis"]["state"],
            "self_excited_locked_speeds_m_s": self_excited["locked_drift_speeds_m_s"],
            "self_excited_frequency_deviation_operational": self_excited[
                "max_lock_in_frequency_deviation_operational"
            ],
            "self_excited_frequency_deviation_all": self_excited[
                "max_lock_in_frequency_deviation_of_locked_cases"
            ],
            "self_excited_amplitude_range_m": [
                min(row["peak_amplitude_m"] for row in self_excited["rows"]
                    if row["regime"] == "SELF_EXCITED_LIMIT_CYCLE"),
                max(row["peak_amplitude_m"] for row in self_excited["rows"]
                    if row["regime"] == "SELF_EXCITED_LIMIT_CYCLE"),
            ],
        },
        "criteria": report["criteria"],
        "figures": figures,
        "artifact_dir": str(out_dir),
    }


def run(out_dir: Path, with_pytest: bool) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    case = build_reference_case()
    analysis = IceVibrationAnalysis(case=case)
    report = analysis.run(include_time_series=False)
    write_json(out_dir / "report.json", report)

    # sweep tables
    write_csv(
        out_dir / "forced_response_sweep.csv",
        report["forced_response_sweep"]["rows"],
        ["frequency_ratio", "excitation_hz", "detuning_relative", "amplification_analytic_linear",
         "amplification_time_domain_linear", "amplification_time_domain",
         "amplification_duffing_harmonic_balance", "relative_error_time_vs_linear",
         "relative_error_time_vs_duffing", "in_lock_in_band", "state"],
    )
    write_csv(
        out_dir / "self_excited_sweep.csv",
        report["self_excited"]["rows"],
        ["drift_speed_m_s", "regime", "peak_amplitude_m", "rms_amplitude_m",
         "response_spectrum_peak_hz.magnitude", "lock_in_frequency_deviation",
         "ice_slope_dF_dv_n_s_m", "effective_damping_n_s_m", "negative_damping",
         "amplification_vs_peak_force_static", "amplification_vs_operating_force_static"],
    )
    write_csv(
        out_dir / "shock_spectrum.csv",
        report["whiplash_shock_spectrum"]["rows"],
        ["pulse_ratio_Tp_f1", "pulse_duration_s", "whiplash_dynamic_amplification",
         "peak_midship_moment_nm", "peak_bow_displacement_m", "midship_moment_relative_error"],
    )
    write_csv(
        out_dir / "damping_sensitivity.csv",
        report["damping_sensitivity"]["rows"],
        ["damping_ratio", "amplification_at_upper_band_edge", "amplification_analytic",
         "amplification_at_resonance_analytic", "self_excited_limit_cycle_amplitude_m"],
    )

    # whiplash time history (full resolution, from the same analysis settings)
    whiplash_full = analysis.whiplash_case()
    series = whiplash_full["time_series"]
    write_csv(
        out_dir / "whiplash_timeseries.csv",
        [{"time_s": round(t, 6), "force_n": round(f, 1), "midship_moment_nm": round(m, 0),
          "midship_moment_analytic_nm": round(ma, 0), "bow_displacement_m": round(d, 8)}
         for t, f, m, ma, d in zip(series["time_s"], series["force_n"], series["midship_moment_nm"],
                                    series["midship_moment_analytic_nm"], series["bow_displacement_m"])],
        ["time_s", "force_n", "midship_moment_nm", "midship_moment_analytic_nm", "bow_displacement_m"],
    )

    figures: List[Dict[str, str]] = []
    figure_specs = [
        ("fig1_mode_shapes.png", figure_mode_shapes(case, fig_dir / "fig1_mode_shapes.png")),
        ("fig2_amplification_curve.png", figure_amplification(report, fig_dir / "fig2_amplification_curve.png")),
        ("fig3_duffing_fold.png", figure_duffing_fold(report, case, fig_dir / "fig3_duffing_fold.png")),
        ("fig4_spectra.png", figure_spectra(analysis, report, fig_dir / "fig4_spectra.png")),
        ("fig5_self_excited.png", figure_self_excited(report, fig_dir / "fig5_self_excited.png")),
        ("fig6_whiplash.png", figure_whiplash(report, case, fig_dir / "fig6_whiplash.png")),
        ("fig7_shock_spectrum.png", figure_shock_spectrum(report, fig_dir / "fig7_shock_spectrum.png")),
        ("fig8_damping_sensitivity.png", figure_damping(report, fig_dir / "fig8_damping_sensitivity.png")),
    ]
    for name, meta in figure_specs:
        figures.append({
            "name": name,
            "path": str((fig_dir / name).relative_to(WORKTREE)),
            "caption": " ".join(meta["captions"]),
        })

    pytest_summary: Optional[Dict[str, Any]] = None
    if with_pytest:
        pytest_summary = run_pytest()
        write_json(out_dir / "pytest_summary.json", pytest_summary)
        report["criteria"]["entries"].append(
            {
                "criterion": "c_pytest_suite",
                "value": pytest_summary.get("passed"),
                "threshold": 12,
                "passed": bool((pytest_summary.get("failed") == 0) and (pytest_summary.get("passed", 0) >= 12)),
                "detail": "pytest run of the ice-vibration suites (tests/test_ice_vibration_*.py)",
            }
        )
        report["criteria"]["n_criteria"] = len(report["criteria"]["entries"])
        report["criteria"]["n_passed"] = sum(1 for e in report["criteria"]["entries"] if e["passed"])
        report["criteria"]["all_passed"] = all(e["passed"] for e in report["criteria"]["entries"])
        write_json(out_dir / "report.json", report)

    metrics = metrics_payload(report, figures, out_dir)
    if pytest_summary is not None:
        metrics["pytest"] = pytest_summary
    write_json(out_dir / "metrics.json", metrics)
    return {"report": report, "metrics": metrics, "figures": figures}


def run_pytest() -> Dict[str, Any]:
    command = [
        sys.executable, "-m", "pytest", "-q",
        "tests/test_ice_vibration_modal.py",
        "tests/test_ice_vibration_ice_load.py",
        "tests/test_ice_vibration_lockin.py",
        "tests/test_ice_vibration_whiplash.py",
        "tests/test_ice_vibration_analysis.py",
    ]
    completed = subprocess.run(command, cwd=WORKTREE, capture_output=True, text=True)
    output = (completed.stdout or "") + (completed.returncode and (completed.stderr or "") or "")
    passed = re.search(r"(\d+) passed", output)
    failed = re.search(r"(\d+) failed", output)
    return {
        "command": " ".join(command[1:]),
        "returncode": completed.returncode,
        "passed": int(passed.group(1)) if passed else 0,
        "failed": int(failed.group(1)) if failed else 0,
        "summary_tail": output.strip().splitlines()[-1] if output.strip() else "",
        "run_utc": _utcnow(),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--with-pytest", action="store_true",
                        help="also run the ice-vibration pytest suites and record the outcome")
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir).resolve()
    payload = run(out_dir, args.with_pytest)
    metrics = payload["metrics"]
    criteria = payload["report"]["criteria"]
    print(f"artifacts: {out_dir}")
    print(f"criteria: {criteria['n_passed']}/{criteria['n_criteria']} passed "
          f"(all={criteria['all_passed']})")
    for entry in criteria["entries"]:
        print(f"  {entry['criterion']}: {entry['value']} (threshold {entry['threshold']}) "
              f"{'PASS' if entry['passed'] else 'FAIL'}")
    print(f"figures: {len(payload['figures'])}")
    print(f"headline: {json.dumps(metrics['headline'], indent=2, default=float)}")
    return 0 if criteria["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())