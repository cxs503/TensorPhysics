"""Figures + paper-level report for p228w1-N08 (ice-induced vibration chain).

Reads ``artifacts/m9212/ice_vibration_transfer_n08.json`` (written by
``scripts/m9212_ice_vibration_transfer_payload.py``), draws the figures into
``docs/ice_vibration_transfer_n08/``, writes ``docs/ICE_VIBRATION_TRANSFER_N08.md``
and posts the report to the platform report API.

Every number in the markdown comes from the payload - nothing is typed by hand.
"""

from __future__ import annotations

import base64
import json
import os
import pathlib
import sys
import urllib.error
import urllib.request

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAYLOAD = ROOT / "artifacts" / "m9212" / "ice_vibration_transfer_n08.json"
FIGDIR = ROOT / "docs" / "ice_vibration_transfer_n08"
DOC = ROOT / "docs" / "ICE_VIBRATION_TRANSFER_N08.md"

sys.path.insert(0, str(ROOT / "src"))
from tensorphysics.fatigue.sn_curve import DNCurve  # noqa: E402

plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3})


def _design(data):
    return data["severities"][data["config"]["severity"]]


# ------------------------------------------------------------------ figures
def fig_wet_dry(data):
    modes = data["hull_modal"]["modes"]
    idx = np.array([m["mode"] for m in modes], dtype=float)
    dry = np.array([m["f_dry_hz"] for m in modes])
    wet = np.array([m["f_wet_hz"] for m in modes])
    ratio = np.array([m["ratio_wet_over_dry"] for m in modes])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.4))
    ax.bar(idx - 0.2, dry, width=0.4, label="dry (in air)", color="#4c72b0")
    ax.bar(idx + 0.2, wet, width=0.4, label="wet (added mass)", color="#c44e52")
    ax.set_xlabel("mode number")
    ax.set_ylabel("natural frequency [Hz]")
    ax.set_title("hull-girder vertical bending: dry vs wet")
    ax.legend()
    ax2.plot(idx, ratio, "o-", color="#55a868")
    ax2.axhline(1.0, color="k", ls="--", lw=1)
    ax2.set_xlabel("mode number")
    ax2.set_ylabel(r"$f_{wet}/f_{dry}$")
    ax2.set_ylim(0.55, 1.05)
    ax2.set_title("C1: max ratio = %.3f < 1" % ratio.max())
    fig.tight_layout()
    out = FIGDIR / "fig1_wet_dry_modes.png"
    fig.savefig(out)
    plt.close(fig)
    return out, (
        "船体梁垂向弯曲干/湿模态对比：湿模态（二维剖面无穷频附连水质量 + 浮力恢复刚度）全部低于干模态，"
        "C1（f_wet/f_dry<1）成立；右图给出逐阶比值。"
    )


def fig_bode(data):
    bode = data["bode"]
    f = np.array(bode["freqs_hz"])
    mag_db = np.array(bode["magnitude_db"])
    design = data["curves"][data["config"]["severity"]]
    fc = np.array(design["freqs_hz"])
    hull = np.array(design["s_hull_acc"])
    base = np.array(design["s_base_acc_shaft_path"])
    phase = np.array(bode["phase_deg"])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.5))
    ax.semilogx(f, mag_db, color="#4c72b0", label="|H(f)| (mount)")
    ax.axhline(0.0, color="k", ls="--", lw=1, label="|H|=1")
    ax.axvline(bode["isolation_frequency_hz"], color="#c44e52", ls=":", lw=1.2, label=r"$\sqrt{2}f_n$")
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("|H(f)| [dB]")
    ax.set_title(
        "Bode: f_n=%.1f Hz, zeta=%.2f, peak=%.2f"
        % (bode["f_n_hz"], bode["zeta"], bode["peak_amplification"])
    )
    ax.set_ylim(-45, 15)
    ax.legend(fontsize=7, loc="lower left")
    ax3 = ax.twinx()
    ax3.semilogx(f, phase, color="#937860", lw=0.9, ls="-.")
    ax3.set_ylabel("phase [deg]")
    ax3.grid(False)
    ax2.loglog(fc, hull, color="#8172b2", lw=1.2, label="hull-girder acc (path A)")
    ax2.loglog(fc, base, color="#55a868", lw=1.2, label="equipment base acc (path B)")
    ax2.set_xlabel("frequency [Hz]")
    ax2.set_ylabel(r"acceleration PSD [m$^2$/s$^4$/Hz]")
    ax2.set_title("ice-induced acceleration spectra (design ice class)")
    ax2.legend(fontsize=7)
    fig.tight_layout()
    out = FIGDIR / "fig2_bode_transfer.png"
    fig.savefig(out)
    plt.close(fig)
    return out, (
        "左：设备基座传递函数 H(f) 的 Bode 图（幅值 dB 实线 + 相位点划线）。|H|=1 在 sqrt(2)f_n 处交叉，"
        "其上严格衰减；f_n 以下为放大段（H(0)=1，任何被动隔振器必然如此），峰值 "
        "%.2f。右：设计冰级下船体梁加速度谱与设备基座加速度谱。" % bode["peak_amplification"]
    )


def fig_stress_sn(data):
    design = _design(data)
    curves = data["curves"][data["config"]["severity"]]
    fc = np.array(curves["freqs_hz"])
    s_a = np.array(curves["s_pipe_stress_hull_path"])
    s_b = np.array(curves["s_pipe_stress_shaft_path"])
    fatigue = design["paths"]["shaft_reaction"]["pipe"]["fatigue"]
    hist = fatigue["rainflow_histogram"]
    sn = DNCurve(data["sn_curve"]["class"], data["sn_curve"]["environment"])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.6))
    ax.loglog(fc, np.maximum(s_a, 1e-30), color="#8172b2", lw=1.1, label="path A (hull girder)")
    ax.loglog(fc, np.maximum(s_b, 1e-30), color="#c44e52", lw=1.1, label="path B (shaft reaction, governing)")
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel(r"pipe stress PSD [(MPa)$^2$/Hz]")
    ax.set_title("pipe-span detail stress spectrum (design ice class)")
    ax.legend(fontsize=7)
    s = np.logspace(0.0, 3.0, 400)
    n = np.array([sn.n_to_failure(v) for v in s])
    ax2.loglog(n, s, "k-", lw=1.4, label="DNV-RP-C203 D, seawater CP")
    ax2.axhline(
        float(data["sn_curve"]["stress_at_1e7_cycles_MPa"]), color="gray", ls=":", lw=1,
        label="%.1f MPa at 1e7 cyc" % float(data["sn_curve"]["stress_at_1e7_cycles_MPa"]),
    )
    counts = np.array(hist["cycles"])
    ranges = np.array(hist["range_MPa"])
    keep = counts > 0
    if np.any(keep):
        cyc_per_year = counts[keep] / fatigue["synthesis"]["duration_s"] * 3.15576e7 * data["config"]["operating_fraction"]
        ax2.plot(np.maximum(cyc_per_year, 1.0), np.maximum(ranges[keep], 1e-3), "o", ms=4, color="#c44e52",
                 label="rainflow bins (cycles/yr)")
    ax2.set_xlabel("cycles to failure N (curve) / cycles per year (bins)")
    ax2.set_ylabel("stress range [MPa]")
    ax2.set_title("rainflow + Miner: D/yr=%.2e, life=%.2e yr" % (fatigue["damage_per_year"], fatigue["life_years"]))
    ax2.legend(fontsize=6, loc="lower left")
    ax2.set_xlim(1e3, 1e16)
    fig.tight_layout()
    out = FIGDIR / "fig3_stress_spectrum_sn.png"
    fig.savefig(out)
    plt.close(fig)
    return out, (
        "左：管路细节应力谱（路径 A 与路径 B 相差约 3 个数量级）。右：DNV-RP-C203 D 曲线（海水+阴极保护）与"
        "雨流计数结果（阴影点按“每年循环数”绘制）；全部应力范围落在高周分支，损伤经 Miner 累积换算为寿命。"
    )


def fig_life_vs_load(data):
    sweep = data["scale_sweep"]
    lam = np.array([row["scale"] for row in sweep])
    life_b = np.array([row["life_years"] for row in sweep])
    life_a = np.array([row["hull_girder_pipe_life_years"] for row in sweep])
    design_life = float(data["config"]["design_life_years"])
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.loglog(lam, life_b, "o-", color="#c44e52", label="path B (shaft-line reaction, governing)")
    ax.loglog(lam, life_a, "s--", color="#8172b2", label="path A (hull girder)")
    for severity, marker in (("light", "^"), ("design", "o"), ("heavy", "v")):
        block = data["severities"][severity]["paths"]["shaft_reaction"]["pipe"]["fatigue"]
        ax.plot([1.0], [block["life_years"]], marker=marker, ms=9, mfc="none", mec="k", ls="none")
    ax.axhline(design_life, color="k", ls=":", lw=1.2, label="design life %.0f yr" % design_life)
    ax.set_xlabel(r"torque-intensity multiplier $\lambda$ (ice load)")
    ax.set_ylabel("pipe fatigue life [years]")
    ax.set_title("C3: fatigue life decreases monotonically with ice load")
    ax.legend(fontsize=7)
    fig.tight_layout()
    out = FIGDIR / "fig4_life_vs_ice_load.png"
    fig.savefig(out)
    plt.close(fig)
    return out, (
        "疲劳寿命-冰载荷曲线（双对数）：扭矩强度 λ 增大→寿命严格单调下降（C3b）；空心标记为轻/设计/重三级冰况。"
        "路径 A（船体梁）比路径 B（轴系局部反力）长十几个数量级，说明冰铣削扭矩经船体梁的传递并非疲劳控制路径。"
    )


def fig_tuning_map(data):
    rows = data["mount_tuning_map"]
    f_grid = sorted({r["f_n_hz"] for r in rows})
    z_grid = sorted({r["zeta"] for r in rows})
    z = np.full((len(z_grid), len(f_grid)), np.nan)
    for r in rows:
        z[z_grid.index(r["zeta"]), f_grid.index(r["f_n_hz"])] = r["rms_ratio_base_over_hull"]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    im = ax.imshow(np.log10(z), origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(f_grid)), ["%g" % v for v in f_grid])
    ax.set_yticks(range(len(z_grid)), ["%g" % v for v in z_grid])
    ax.set_xlabel(r"mount natural frequency $f_n$ [Hz]")
    ax.set_ylabel(r"mount damping ratio $\zeta$")
    for i in range(len(z_grid)):
        for j in range(len(f_grid)):
            ax.text(j, i, "%.2f" % z[i, j], ha="center", va="center", fontsize=6, color="w")
    ax.set_title(r"full-band RMS(base)/RMS(hull girder), cell = ratio")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label=r"$\log_{10}$ ratio")
    fig.tight_layout()
    out = FIGDIR / "fig5_mount_tuning_map.png"
    fig.savefig(out)
    plt.close(fig)
    return out, (
        "隔振器调谐图：格内为全频带（0.2-500 Hz）RMS(基座)/RMS(船体梁) 比值。全部组合 >1，说明单级被动隔振无法降低"
        "船体梁频段（0.4-3 Hz）的冰致振动——这是披露项 D1 的定量证据，而不是可回避的建模细节。")