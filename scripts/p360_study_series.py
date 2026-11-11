"""Series and figures for the P360W1-N04 moving-load / FNV study.

Kept separate from the driver so the report builder stays readable: this module
only computes the time histories the figures need and draws them.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorphysics.moving_load import (  # noqa: E402
    LoadPath,
    analytic_undamped_response,
    fd_beam_response,
    ice_field_case,
    moving_point_load,
    reference_load_n,
    solve_modal_response,
    static_anchor,
)
from tensorphysics.moving_load.fnv import dominant_frequency  # noqa: E402


def build_series(beam) -> dict:
    """Time histories and profiles required by the figures."""
    load = reference_load_n()
    v_cr = beam.critical_velocity()
    x_mid = 0.5 * beam.length_m

    path = moving_point_load(beam, load, v_cr, x_entry_m=0.0)
    crossing = beam.length_m / v_cr
    modal = solve_modal_response(beam, [path], crossing, dt_s=2.0e-4, n_modes=10, damping_ratio=0.0)
    analytic = analytic_undamped_response(beam, load, v_cr, 0.0, modal.time_s, x_mid, n_modes=100)
    fd = fd_beam_response(beam, [path], crossing + 0.2, n_elements=200, probe_positions_m=(x_mid,))
    trunc_high = solve_modal_response(beam, [path], crossing, dt_s=2.0e-4, n_modes=40)
    damped = solve_modal_response(
        beam, [path], crossing + 1.0, dt_s=2.0e-4, n_modes=10, damping_ratio=0.015
    )

    static = static_anchor(beam)
    hold = LoadPath(
        magnitude_n=load,
        x_entry_m=x_mid,
        x_exit_m=x_mid,
        t_entry_s=0.0,
        duration_s=static["settle_time_s"],
        shape="constant",
    )
    settle = solve_modal_response(
        beam,
        [hold],
        static["settle_time_s"],
        dt_s=static["dt_s"],
        n_modes=10,
        damping_ratio=static["damping_ratio"],
    )

    v_slow = 0.01 * v_cr
    slow_path = moving_point_load(beam, load, v_slow, x_entry_m=0.0)
    slow = solve_modal_response(
        beam, [slow_path], beam.length_m / v_slow, dt_s=4.0e-3, n_modes=10, damping_ratio=0.0
    )
    idx = int(np.argmin(np.abs(slow.time_s - x_mid / v_slow)))
    x_prof, u_prof = slow.deflection_profile(idx, n_points=201)

    ice = ice_field_case(beam, dt_s=2.0e-3, tail_s=8.0)
    load_series = ice["load_total_n"]
    mask = ice["time_s"] > 0.0
    f_load = dominant_frequency(ice["time_s"][mask], load_series[mask])

    return {
        "v_cr": v_cr,
        "crossing": crossing,
        "modal": modal,
        "analytic": analytic,
        "fd": fd,
        "trunc_high": trunc_high,
        "damped": damped,
        "static": static,
        "settle": settle,
        "slow": slow,
        "x_prof": x_prof,
        "u_prof": u_prof,
        "ice": ice,
        "f_load": f_load,
        "f1": beam.frequency(1),
    }


def _style(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)


def make_figures(beam, series: dict, payload: dict, fig_dir: Path) -> list:
    """Draw the eight figures and return ``[{name, caption, path}]``."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    figs = []
    x = np.linspace(0.0, beam.length_m, 401)
    load = reference_load_n()

    # -- fig 1: modes and frequencies --------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for n in (1, 2, 3):
        axes[0].plot(x, np.sin(n * math.pi * x / beam.length_m), label=f"mode {n}: {beam.frequency(n):.3f} Hz")
    _style(axes[0], "x [m]", "phi_n(x) [-]", "Simply supported hull-girder modes")
    axes[0].legend(fontsize=8)
    n_modes = np.arange(1, 11)
    f = beam.natural_frequencies(10)
    axes[1].vlines(n_modes, 0.0, f, lw=2)
    axes[1].plot(n_modes, f, "o", ms=5)
    _style(axes[1], "mode number n", "frequency [Hz]", "Natural frequencies (f_n = n^2 f_1)")
    fig.tight_layout()
    figs.append(
        (
            "fig1_modes_frequencies.png",
            "图 1 船体梁模态：前 3 阶振型与前 10 阶固有频率（f_n=n²f₁）",
            fig,
        )
    )

    # -- fig 2: static anchor ----------------------------------------------
    static = series["static"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    exact_prof = beam.static_deflection_profile(load, x)
    modal_prof = beam.static_modal_solution(load, 0.5 * beam.length_m, n_modes=10)
    axes[0].plot(x, exact_prof, "k-", lw=2, label="exact $PL^3/48EI$")
    axes[0].plot(
        np.linspace(0.0, beam.length_m, modal_prof.size), modal_prof, "r--", lw=1.2, label="10-mode sum"
    )
    _style(axes[0], "x [m]", "deflection [m]", "Static load at midspan")
    axes[0].legend(fontsize=8)
    t = series["settle"].time_s
    u = series["settle"].displacement_at(0.5 * beam.length_m)
    axes[1].plot(t, u, lw=1.2, label="damped transient (zeta=%.2f)" % static["damping_ratio"])
    axes[1].axhline(
        static["exact_static_midspan_m"],
        color="k",
        ls="--",
        lw=1,
        label=f"exact = {static['exact_static_midspan_m']:.5f} m",
    )
    _style(axes[1], "time [s]", "midspan deflection [m]", "Stationary load: settling onto static value")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    figs.append(("fig2_static_anchor.png", "图 2 静止载荷基准：静挠曲线与恒载阶跃下的阻尼瞬态（判据 a）", fig))

    # -- fig 3: V_cr time histories ----------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(series["modal"].time_s, series["modal"].displacement_at(0.5 * beam.length_m), "C0-", lw=1.6, label="10-mode Newmark")
    axes[0].plot(series["analytic"]["time_s"], series["analytic"]["displacement_m"], "C1--", lw=1.2, label="100-term closed form")
    axes[0].plot(
        series["fd"]["time_s"],
        series["fd"]["probes"][0.5 * beam.length_m]["displacement_m"],
        "C2:",
        lw=1.4,
        label="finite-difference reference",
    )
    _style(axes[0], "time [s]", "midspan deflection [m]", f"V = V_cr = {series['v_cr']:.1f} m/s")
    axes[0].legend(fontsize=8)
    axes[1].plot(series["modal"].time_s, series["modal"].displacement_at(0.5 * beam.length_m), "C0-", lw=1.4, label="10 modes")
    axes[1].plot(series["trunc_high"].time_s, series["trunc_high"].displacement_at(0.5 * beam.length_m), "C3--", lw=1.2, label="40 modes")
    _style(axes[1], "time [s]", "midspan deflection [m]", "Modal truncation at V_cr")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    figs.append(
        (
            "fig3_vcr_time_history.png",
            "图 3 V=V_cr 中跨挠度时程：10 模态 Newmark vs 100 项解析级数 vs 有限差分（判据 b），右图模态截断（判据 e）",
            fig,
        )
    )

    # -- fig 4: DAF and FNV versus V/V_cr ----------------------------------
    sweep = payload["fnv_sweep"]
    fine, spec = sweep["fine_rows"], sweep["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot([r["v_over_vcr"] for r in fine], [r["daf"] for r in fine], "o-", lw=1.4, label="DAF (12-point sweep)")
    axes[0].plot([r["v_over_vcr"] for r in spec], [r["daf"] for r in spec], "s", ms=6, label="specified points")
    axes[0].axvline(1.0, color="grey", ls=":", lw=1)
    axes[0].axvline(sweep["daf_peak_ratio"], color="C3", ls="--", lw=1, label=f"DAF peak at {sweep['daf_peak_ratio']:.1f} V_cr")
    _style(axes[0], "V / V_cr [-]", "DAF = u_max/u_static [-]", "Dynamic amplification factor")
    axes[0].legend(fontsize=8)
    axes[1].plot([r["v_over_vcr"] for r in fine], [r["fnv"] for r in fine], "o-", lw=1.4, label="FNV (12-point sweep)")
    axes[1].plot([r["v_over_vcr"] for r in spec], [r["fnv"] for r in spec], "s", ms=6, label="specified points")
    axes[1].axhline(0.5, color="orange", ls="--", lw=1, label="habitable / tolerable (0.5)")
    axes[1].axhline(1.0, color="red", ls="--", lw=1, label="tolerable / unacceptable (1.0)")
    _style(axes[1], "V / V_cr [-]", "FNV = a_max/(g f) [-]", "FNV over the resonance sweep")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    figs.append(
        (
            "fig4_daf_fnv_sweep.png",
            f"图 4 V/V_cr 扫描：DAF 与 FNV（判据 c）；FNV 峰值位于 {sweep['fnv_peak_ratio']:.1f} V_cr，方点为任务指定工况",
            fig,
        )
    )

    # -- fig 5: energy balance ---------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    undamped, damped = series["modal"], series["damped"]
    axes[0].plot(undamped.time_s, undamped.energy_j - undamped.work_j, lw=1.4, label="undamped: E - W")
    axes[0].plot(
        damped.time_s, damped.energy_j + damped.dissipated_j - damped.work_j, lw=1.2, label="damped: E + D - W"
    )
    _style(axes[0], "time [s]", "power-balance residual [J]", "Energy conservation residual")
    axes[0].legend(fontsize=8)
    axes[1].plot(damped.time_s, damped.work_j, lw=1.4, label="injected work W")
    axes[1].plot(damped.time_s, damped.dissipated_j, lw=1.4, label="dissipated D (zeta=0.015)")
    axes[1].plot(damped.time_s, damped.energy_j, lw=1.4, label="mechanical energy E")
    _style(axes[1], "time [s]", "energy [J]", "Work / dissipation / energy balance")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    figs.append(("fig5_energy_balance.png", "图 5 功率平衡与能量守恒残差（判据 d）", fig))

    # -- fig 6: ice load history and spectrum -------------------------------
    ice = series["ice"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(ice["time_s"], ice["load_total_n"] / 1e6, "k-", lw=1.0)
    axes[0].set_ylim(0.0, 1.05 * ice["peak_contact_force_n"] / 1e6)
    _style(
        axes[0],
        "time [s]",
        "ice contact force [MN]",
        "Floe stream at V = %.1f m/s (peak %.1f MN)" % (ice["speed_m_s"], ice["peak_contact_force_n"] / 1e6),
    )
    mask = ice["time_s"] > 0.0
    sig = ice["load_total_n"][mask]
    spec_amp = np.abs(np.fft.rfft(sig - sig.mean()))
    freqs = np.fft.rfftfreq(int(mask.sum()), d=float(np.mean(np.diff(ice["time_s"]))))
    axes[1].semilogy(freqs, spec_amp / max(spec_amp.max(), 1e-30), lw=1.0, label="load spectrum")
    axes[1].axvline(series["f1"], color="C3", ls="--", lw=1.2, label=f"f1 = {series['f1']:.3f} Hz")
    axes[1].axvline(
        ice["encounter_frequency_hz"], color="C2", ls=":", lw=1.4, label=f"f_enc = V/s = {ice['encounter_frequency_hz']:.3f} Hz"
    )
    axes[1].set_xlim(0.0, 2.0)
    _style(axes[1], "frequency [Hz]", "normalised |F(f)| [-]", "Ice-load spectrum vs structural frequency")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    figs.append(("fig6_ice_load_spectrum.png", "图 6 碎冰流接触力时程与频谱：遭遇频率 f_enc=V/s 与一阶固有频率 f₁ 的关系", fig))

    # -- fig 7: acceleration and FNV ---------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(ice["time_s"], ice["acceleration_mid_series_m_s2"], lw=0.9, label="midspan (FNV = %.3f)" % ice["fnv_midspan"])
    axes[0].plot(
        ice["time_s"],
        ice["acceleration_bow_series_m_s2"],
        lw=0.7,
        alpha=0.8,
        label="bow shoulder 0.92L (FNV = %.3f)" % ice["stations"]["0.92"]["fnv"],
    )
    _style(axes[0], "time [s]", "hull-girder acceleration [m/s^2]", "Ice-induced acceleration vs time")
    axes[0].legend(fontsize=8)
    a = ice["acceleration_mid_series_m_s2"][mask]
    spec_a = np.abs(np.fft.rfft((a - a.mean()) * np.hanning(a.size) * 2.0 / a.size))
    freqs_a = np.fft.rfftfreq(a.size, d=float(np.mean(np.diff(ice["time_s"]))))
    axes[1].semilogy(freqs_a, spec_a, lw=1.1, label="midspan response spectrum")
    axes[1].axvline(
        ice["stations"]["0.50"]["dominant_frequency_hz"],
        color="C3",
        ls="--",
        lw=1.0,
        label="f_dom = %.3f Hz" % ice["stations"]["0.50"]["dominant_frequency_hz"],
    )
    axes[1].set_xlim(0.0, 2.0)
    _style(axes[1], "frequency [Hz]", "|a(f)| [m/s^2]", "Response spectrum (FNV input)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    figs.append(
        (
            "fig7_ice_response_fnv.png",
            "图 7 5 m/s 冰流工况：中跨/首肩加速度时程与响应谱（FNV=%.3f，%s）" % (ice["fnv_midspan"], ice["fnv_level_midspan"]),
            fig,
        )
    )

    # -- fig 8: ship speed sweep -------------------------------------------
    scan = payload["ice_speed_sweep"]
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    speeds = [r["speed_m_s"] for r in scan["rows"]]
    fnvs = [r["fnv"] for r in scan["rows"]]
    ax.plot(speeds, fnvs, "o-", lw=1.5, label="FNV (midspan)")
    ax.axhline(0.5, color="orange", ls="--", lw=1.0, label="habitable / tolerable")
    ax.axhline(1.0, color="red", ls="--", lw=1.0, label="tolerable / unacceptable")
    ax.axvline(
        scan["resonant_speed_estimate_m_s"], color="C3", ls=":", lw=1.2, label="V* = f1*s = %.2f m/s" % scan["resonant_speed_estimate_m_s"]
    )
    ax.plot(
        [scan["fnv_peak_speed_m_s"]], [scan["fnv_peak_value"]], "k*", ms=12, label="FNV peak at %.1f m/s" % scan["fnv_peak_speed_m_s"]
    )
    _style(ax, "ship speed V [m/s]", "FNV [-]", "Ice-class habitability vs ship speed")
    ax.legend(fontsize=7)
    fig.tight_layout()
    figs.append(
        (
            "fig8_ship_speed_sweep.png",
            "图 8 航速扫描：FNV 在遭遇频率与一阶固有频率重合处（V*≈%.2f m/s）取得峰值" % scan["resonant_speed_estimate_m_s"],
            fig,
        )
    )

    out = []
    for name, caption, fig_obj in figs:
        path = fig_dir / name
        fig_obj.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig_obj)
        out.append({"name": name, "caption": caption})
    return out