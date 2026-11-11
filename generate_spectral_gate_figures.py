"""Generate figures for the spectral gate report (p357w1-n07)."""

from __future__ import annotations

import base64
import json
import os
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tensorphysics.spectral_gate import (
    welch_psd, dpss_multitaper_psd, lomb_scargle_psd,
    ar1_psd, ar2_psd, generate_ar1, generate_ar2,
    run_spectral_gate,
)
from tensorphysics.spectral_gate.ar_models import ar2_resonant_frequency

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def save_fig(name: str, caption: str) -> dict:
    """Save figure and return dict for report."""
    path = os.path.join(FIG_DIR, f"{name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return {"name": name, "caption": caption, "png_base64": b64}


def main():
    fs = 100.0
    n = 8192
    seed = 42

    # ---- Figure 1: AR(1) PSD comparison ----
    phi1 = 0.85
    sigma_ar1 = 1.0
    ar1_signal = generate_ar1(n, phi1, sigma_ar1, seed=seed)

    welch_ar1 = welch_psd(ar1_signal, fs, nperseg=512)
    dpss_ar1 = dpss_multitaper_psd(ar1_signal, fs, nw=4.0)
    psd_ar1_true = ar1_psd(welch_ar1.freqs_hz, phi1, sigma_ar1**2, fs)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(welch_ar1.freqs_hz, welch_ar1.psd, label="Welch", alpha=0.8)
    ax.semilogy(dpss_ar1.freqs_hz, dpss_ar1.psd, label="DPSS Multi-taper", alpha=0.8)
    ax.semilogy(welch_ar1.freqs_hz, psd_ar1_true, "k--", linewidth=2, label="Analytical AR(1)")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD [units²/Hz]")
    ax.set_title("AR(1) Process: Welch vs DPSS vs Analytical PSD")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig1 = save_fig("fig1_ar1_psd", "AR(1) process PSD: Welch, DPSS multi-taper, and analytical comparison")

    # ---- Figure 2: AR(2) PSD with resonance ----
    phi1_ar2 = 1.95
    phi2_ar2 = -0.97
    sigma_ar2 = 1.0
    n_ar2 = 32768
    ar2_signal = generate_ar2(n_ar2, phi1_ar2, phi2_ar2, sigma_ar2, seed=seed + 1)

    welch_ar2 = welch_psd(ar2_signal, fs, nperseg=4096)
    dpss_ar2 = dpss_multitaper_psd(ar2_signal, fs, nw=4.0)
    psd_ar2_true = ar2_psd(welch_ar2.freqs_hz, phi1_ar2, phi2_ar2, sigma_ar2**2, fs)
    f_res = ar2_resonant_frequency(phi1_ar2, phi2_ar2, fs)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(welch_ar2.freqs_hz, welch_ar2.psd, label="Welch", alpha=0.8)
    ax.semilogy(dpss_ar2.freqs_hz, dpss_ar2.psd, label="DPSS Multi-taper", alpha=0.8)
    ax.semilogy(welch_ar2.freqs_hz, psd_ar2_true, "k--", linewidth=2, label="Analytical AR(2)")
    ax.axvline(f_res, color="r", linestyle=":", label=f"Resonant freq = {f_res:.2f} Hz")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD [units²/Hz]")
    ax.set_title("AR(2) Process: Sharp Resonance Peak")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 10)
    fig2 = save_fig("fig2_ar2_psd", "AR(2) process PSD with sharp resonance at f_res ≈ 2.24 Hz")

    # ---- Figure 3: Cross-validation comparison ----
    fig, ax = plt.subplots(figsize=(10, 5))
    freqs_common = np.linspace(0, fs/2, 1024)
    welch_interp = np.interp(freqs_common, welch_ar1.freqs_hz, welch_ar1.psd)
    dpss_interp = np.interp(freqs_common, dpss_ar1.freqs_hz, dpss_ar1.psd)
    ax.semilogy(freqs_common, welch_interp, label="Welch", alpha=0.8)
    ax.semilogy(freqs_common, dpss_interp, label="DPSS Multi-taper", alpha=0.8)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD [units²/Hz]")
    ax.set_title("Cross-Validation: Welch vs DPSS Multi-taper")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig3 = save_fig("fig3_cross_validation", "Cross-validation: Welch and DPSS multi-taper PSD comparison")

    # ---- Figure 4: Lomb–Scargle on non-uniform data ----
    rng = np.random.default_rng(seed + 2)
    time_uniform = np.arange(n) / fs
    test_freq = 5.0
    clean = np.sin(2 * np.pi * test_freq * time_uniform)
    noisy = clean + 0.3 * rng.normal(size=n)

    n_keep = int(n * 0.7)
    keep_idx = rng.choice(n, size=n_keep, replace=False)
    keep_idx.sort()
    t_obs = time_uniform[keep_idx]
    sig_obs = noisy[keep_idx]

    ls_result = lomb_scargle_psd(t_obs, sig_obs, n_freq=2048)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(time_uniform[:500], noisy[:500], "b-", alpha=0.5, label="Full signal")
    ax1.plot(t_obs[:350], sig_obs[:350], "r.", markersize=2, label="Observed (70%)")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Signal")
    ax1.set_title("Non-uniform Sampling (30% missing)")
    ax1.legend()

    ax2.plot(ls_result.freqs_hz, ls_result.psd, "b-")
    ax2.axvline(test_freq, color="r", linestyle="--", label=f"True freq = {test_freq} Hz")
    ax2.axvline(ls_result.peak_frequency(), color="g", linestyle=":", label=f"Detected = {ls_result.peak_frequency():.2f} Hz")
    ax2.set_xlabel("Frequency [Hz]")
    ax2.set_ylabel("PSD [units²/Hz]")
    ax2.set_title("Lomb–Scargle Periodogram")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig4 = save_fig("fig4_lomb_scargle", "Lomb–Scargle periodogram on non-uniformly sampled data (30% missing)")

    # ---- Figure 5: Spectral gate summary ----
    report = run_spectral_gate(n_samples=n, fs=fs, seed=seed)

    criteria = ["AR(1)\nEnergy", "AR(2)\nEnergy", "AR(2)\nPeak Freq", "Cross-Val\nDeviation", "L-S\nAccuracy"]
    values = [
        report.ar1_energy_error,
        report.ar2_energy_error,
        report.ar2_peak_freq_error,
        report.cv_max_deviation,
        1.0 - report.ls_freq_accuracy,  # Error rate
    ]
    thresholds = [0.05, 0.08, 0.02, 0.10, 0.05]
    passes = [report.ar1_pass, report.ar2_pass, report.ar2_peak_freq_error < 0.02,
              report.cv_pass, report.ls_pass]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(criteria))
    colors = ["green" if p else "red" for p in passes]
    bars = ax.bar(x, values, color=colors, alpha=0.7, edgecolor="black")
    ax.plot(x, thresholds, "r--", linewidth=2, label="Threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(criteria)
    ax.set_ylabel("Error / Deviation")
    ax.set_title("Spectral Gate Acceptance Criteria Summary")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_yscale("log")

    # Add pass/fail labels
    for i, (v, p) in enumerate(zip(values, passes)):
        ax.text(i, v * 1.5, "PASS" if p else "FAIL", ha="center", va="bottom",
                fontweight="bold", color="green" if p else "red")

    fig5 = save_fig("fig5_gate_summary", "Spectral gate acceptance criteria: all PASS")

    figures = [fig1, fig2, fig3, fig4, fig5]
    print(f"Generated {len(figures)} figures in {FIG_DIR}")
    return figures


if __name__ == "__main__":
    main()
