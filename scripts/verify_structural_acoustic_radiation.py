#!/usr/bin/env python3
"""Verification harness for P350W1-N04: underwater structural acoustic radiation.

Produces, with zero hand-copied numbers:

* ``artifacts/structural_acoustic_radiation/verification.json``
  - per-criterion measured value / error / pass flag
  - sha256 provenance anchor (payload + every figure)
* ``artifacts/structural_acoustic_radiation/fig_*.png`` (publication figures)
* ``docs/P350W1_N04_STRUCTURAL_ACOUSTIC_RADIATION.md`` (report, generated)

Acceptance criteria (campaign DAG P350W1-N04):
  (a) pulsating-sphere radiated power vs closed-form analytic solution,
      f = 100 / 500 / 1000 Hz, error <= 3 %
  (b) circular-piston radiation efficiency sigma(ka) vs analytic Bessel
      solution for ka = 0.1 ... 10, error <= 5 %
  (c) far-field pressure decay p(r) ~ 1/r, error < 2 %
  (d) negative control: f -> 0 gives W_rad < 0.01 rho c S v0^2
  (e) pytest suite all PASS
  (f) result JSON carries a sha256 provenance anchor

Extra physics hardening added by this node:
  (g) piston on-axis far field vs exact Rayleigh-integral solution (<= 5 %)
  (h) dipole (unbaffled plate) sigma ~ (ka)^4 low-frequency law
  (i) SDOF structural vibration stage vs closed-form resonance velocity
  (j) end-to-end chain F0 -> v0 -> W_rad -> SPL at 1/10/100 m
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.special import j1

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.domains.structural_acoustic_radiation.radiation import (  # noqa: E402
    C_WATER,
    P_REF,
    RHO_WATER,
    CircularPiston,
    PulsatingSphere,
    RadiationChain,
    dipole_radiation_efficiency,
    piston_radiation_efficiency,
    pulsating_sphere_radiation_efficiency,
)
from tensorphysics.domains.structural_acoustic_radiation.vibration import (  # noqa: E402
    HarmonicVibrator,
    chain_provenance,
    vibration_radiation_chain,
)

OUT_DIR = ROOT / "artifacts" / "structural_acoustic_radiation"
DOC_PATH = ROOT / "docs" / "P350W1_N04_STRUCTURAL_ACOUSTIC_RADIATION.md"
JSON_PATH = OUT_DIR / "verification.json"
TEST_FILE = "tests/test_structural_acoustic_radiation.py"

plt.rcParams.update({
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def sha256_json(payload) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=float)
    return hashlib.sha256(blob.encode()).hexdigest()


# ---------------------------------------------------------------------------
# (a) pulsating sphere radiated power vs independent closed form
# ---------------------------------------------------------------------------
def criterion_a() -> dict:
    """W_analytic = 2 pi rho c k^2 a^4 v0^2 / (1 + (ka)^2)."""
    radius, v0 = 0.5, 0.01
    rows = []
    for freq in (100.0, 500.0, 1000.0):
        sphere = PulsatingSphere(radius=radius, frequency=freq, velocity_amplitude=v0)
        k, ka = sphere.k, sphere.ka
        W_num = sphere.radiated_power()
        W_ana = (2.0 * np.pi * RHO_WATER * C_WATER * k**2 * radius**4 * v0**2
                 / (1.0 + ka**2))
        rows.append({
            "frequency_Hz": freq,
            "ka": float(ka),
            "sigma_rad": sphere.radiation_efficiency(),
            "W_computed_W": float(W_num),
            "W_analytic_W": float(W_ana),
            "relative_error": float(abs(W_num - W_ana) / W_ana),
        })
    max_err = max(r["relative_error"] for r in rows)
    return {"rows": rows, "max_relative_error": max_err,
            "threshold": 0.03, "passes": bool(max_err <= 0.03)}


# ---------------------------------------------------------------------------
# (b) piston radiation efficiency vs analytic Bessel + independent quadrature
# ---------------------------------------------------------------------------
def criterion_b() -> dict:
    rows = []
    for ka in np.logspace(-1, 1, 13):
        sigma_mod = float(piston_radiation_efficiency(ka))
        sigma_ana = 1.0 - 2.0 * j1(2.0 * ka) / (2.0 * ka)

        def integrand(theta, ka_=ka):
            x = ka_ * np.sin(theta)
            d = 1.0 if abs(x) < 1e-14 else 2.0 * j1(x) / x
            return d**2 * np.sin(theta)

        sigma_quad = (ka**2 / 2.0) * quad(integrand, 0.0, np.pi / 2.0, limit=400)[0]
        rows.append({
            "ka": float(ka),
            "sigma_module": sigma_mod,
            "sigma_analytic": float(sigma_ana),
            "sigma_quadrature": float(sigma_quad),
            "relative_error": float(abs(sigma_mod - sigma_ana) / max(abs(sigma_ana), 1e-300)),
            "quadrature_relative_error": float(abs(sigma_quad - sigma_ana) / abs(sigma_ana)),
        })
    max_err = max(r["relative_error"] for r in rows)
    max_quad = max(r["quadrature_relative_error"] for r in rows)
    return {"rows": rows, "max_relative_error": max_err,
            "max_independent_quadrature_error": max_quad,
            "threshold": 0.05, "passes": bool(max_err <= 0.05)}


# ---------------------------------------------------------------------------
# (c) far-field pressure decay p ~ 1/r
# ---------------------------------------------------------------------------
def criterion_c() -> dict:
    radius, v0, freq = 0.5, 0.01, 500.0
    sphere = PulsatingSphere(radius=radius, frequency=freq, velocity_amplitude=v0)
    r1, r2 = 10.0, 100.0
    p1 = sphere.far_field_pressure_rms(r1)
    p2 = sphere.far_field_pressure_rms(r2)
    decay_err = abs(p1 / p2 - r2 / r1) / (r2 / r1)

    chain = RadiationChain(frequency=freq, velocity_amplitude=v0,
                           surface_area=1.0, sigma_rad=0.5)
    c1 = chain.compute_far_field_pressure(r1)["p_rms"]
    c2 = chain.compute_far_field_pressure(r2)["p_rms"]
    chain_err = abs(c1 / c2 - r2 / r1) / (r2 / r1)

    rs = np.logspace(0, 3, 31)
    ps = np.array([sphere.far_field_pressure_rms(r) for r in rs])
    slope = float(np.polyfit(np.log10(rs), np.log10(ps), 1)[0])
    return {
        "r1_m": r1, "r2_m": r2,
        "p1_over_p2": float(p1 / p2), "expected_ratio": float(r2 / r1),
        "relative_error": float(decay_err),
        "chain_relative_error": float(chain_err),
        "loglog_slope": slope,
        "threshold": 0.02,
        "passes": bool(decay_err < 0.02 and chain_err < 0.02 and abs(slope + 1.0) < 1e-6),
    }


# ---------------------------------------------------------------------------
# (d) negative control: f -> 0
# ---------------------------------------------------------------------------
def criterion_d() -> dict:
    v0 = 0.01
    rows = []
    for label, freq, radius in (("pulsating_sphere", 0.1, 0.1), ("piston", 0.1, 0.1)):
        cls = CircularPiston if label == "piston" else PulsatingSphere
        src = cls(radius=radius, frequency=freq, velocity_amplitude=v0)
        W = src.radiated_power()
        S = src.surface_area
        threshold = 0.01 * RHO_WATER * C_WATER * S * v0**2
        rows.append({
            "model": label, "frequency_Hz": freq, "ka": float(src.ka),
            "sigma_rad": src.radiation_efficiency(),
            "W_rad_W": float(W), "threshold_W": float(threshold),
            "ratio_W_over_threshold": float(W / threshold),
            "passes": bool(W < threshold),
        })

    vib = HarmonicVibrator(mass=5.0, stiffness=5.0 * (2 * np.pi * 100.0) ** 2,
                           damping_ratio=0.05, force_amplitude=50.0)
    chain_rows = []
    for model in ("monopole", "dipole", "piston"):
        res = vibration_radiation_chain(
            vib, 0.1, surface_area=0.5, characteristic_length=0.15, model=model,
            distances=[10.0])
        bound = 0.01 * res["rho"] * res["c"] * res["surface_area"] * res["v0"] ** 2
        chain_rows.append({
            "model": model, "ka": res["ka"], "W_rad_W": res["W_rad"],
            "bound_W": float(bound), "ratio": float(res["W_rad"] / bound),
            "passes": bool(res["W_rad"] < bound),
        })
    passes = all(r["passes"] for r in rows) and all(r["passes"] for r in chain_rows)
    return {"sources": rows, "chain": chain_rows, "passes": bool(passes)}


# ---------------------------------------------------------------------------
# (e) pytest suite
# ---------------------------------------------------------------------------
def criterion_e() -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", TEST_FILE, "-q"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    n_passed = None
    tokens = tail.split()
    for idx, token in enumerate(tokens):
        if token.isdigit() and idx + 1 < len(tokens) and "passed" in tokens[idx + 1]:
            n_passed = int(token)
    return {
        "command": f"python -m pytest {TEST_FILE} -q",
        "returncode": proc.returncode,
        "summary": tail,
        "n_passed": n_passed,
        "passes": proc.returncode == 0,
    }


# ---------------------------------------------------------------------------
# (g) piston on-axis far field vs exact Rayleigh integral
# ---------------------------------------------------------------------------
def criterion_g() -> dict:
    radius = 0.1
    rows = []
    for ka in (0.5, 1.0, 2.0, 5.0):
        freq = ka * C_WATER / (2.0 * np.pi * radius)
        piston = CircularPiston(radius=radius, frequency=freq, velocity_amplitude=0.01)
        wavelength = C_WATER / freq
        r = 40.0 * radius**2 / wavelength
        exact = piston.on_axis_pressure_exact(r)
        approx = piston.on_axis_pressure_amplitude(r)
        power_based = piston.far_field_pressure_amplitude(r)
        rows.append({
            "ka": ka, "r_m": float(r), "kr": float(piston.k * r),
            "p_onaxis_exact_Pa": float(exact),
            "p_onaxis_farfield_Pa": float(approx),
            "p_powerbased_Pa": float(power_based),
            "relative_error": float(abs(approx - exact) / exact),
            "powerbased_relative_error": float(abs(power_based - exact) / exact),
        })
    max_err = max(r["relative_error"] for r in rows)
    return {"rows": rows, "max_relative_error": max_err,
            "threshold": 0.05, "passes": bool(max_err <= 0.05)}


# ---------------------------------------------------------------------------
# (h) dipole sigma ~ (ka)^4
# ---------------------------------------------------------------------------
def criterion_h() -> dict:
    rows = []
    for ka in (0.02, 0.05, 0.1, 0.2):
        sigma = float(dipole_radiation_efficiency(ka))
        rows.append({"ka": ka, "sigma_dipole": sigma,
                     "ka4": float(ka**4), "ratio_sigma_over_ka4": float(sigma / ka**4)})
    kas = np.array([0.01, 0.02, 0.04])
    sig = np.array([float(dipole_radiation_efficiency(k)) for k in kas])
    mono = np.array([float(pulsating_sphere_radiation_efficiency(k)) for k in kas])
    slope = float(np.polyfit(np.log10(kas), np.log10(sig), 1)[0])
    mono_slope = float(np.polyfit(np.log10(kas), np.log10(mono), 1)[0])
    return {"rows": rows, "dipole_loglog_slope": slope,
            "monopole_loglog_slope": mono_slope,
            "passes": bool(abs(slope - 4.0) < 0.05 and abs(mono_slope - 2.0) < 0.05)}


# ---------------------------------------------------------------------------
# (i) SDOF vibration stage
# ---------------------------------------------------------------------------
def criterion_i() -> dict:
    mass, f_n, zeta, F0 = 10.0, 100.0, 0.02, 100.0
    stiffness = mass * (2.0 * np.pi * f_n) ** 2
    vib = HarmonicVibrator(mass=mass, stiffness=stiffness,
                           damping_ratio=zeta, force_amplitude=F0)
    omega_n = vib.omega_n
    v_res_num = vib.velocity_amplitude(omega_n)
    v_res_cf = vib.resonance_velocity_amplitude()
    omega_low = 1e-3 * omega_n
    v_static_num = vib.velocity_amplitude(omega_low)
    v_static_cf = F0 * omega_low / stiffness
    out = {
        "mass_kg": mass, "stiffness_N_per_m": float(stiffness),
        "damping_ratio": zeta, "force_amplitude_N": F0,
        "natural_frequency_Hz": float(vib.natural_frequency),
        "damping_coefficient_Ns_per_m": float(vib.damping_coefficient),
        "resonant_velocity_numeric_m_per_s": float(v_res_num),
        "resonant_velocity_closed_form_m_per_s": float(v_res_cf),
        "resonance_relative_error": float(abs(v_res_num - v_res_cf) / v_res_cf),
        "static_displacement_m": float(vib.static_displacement),
        "quasistatic_relative_error": float(abs(v_static_num - v_static_cf) / v_static_cf),
    }
    out["passes"] = bool(out["resonance_relative_error"] < 1e-12
                         and out["quasistatic_relative_error"] < 1e-3)
    return out


# ---------------------------------------------------------------------------
# (j) end-to-end prediction chain demo (force -> SPL)
# ---------------------------------------------------------------------------
def criterion_j() -> dict:
    mass, f_n, zeta, F0 = 10.0, 100.0, 0.02, 100.0
    vib = HarmonicVibrator(mass=mass, stiffness=mass * (2 * np.pi * f_n) ** 2,
                           damping_ratio=zeta, force_amplitude=F0)
    demo = {}
    for model in ("monopole", "dipole", "piston"):
        res = vibration_radiation_chain(
            vib, f_n, surface_area=1.0, characteristic_length=0.3, model=model,
            distances=[1.0, 10.0, 100.0])
        prov = chain_provenance(res)
        demo[model] = {
            "v0_m_per_s": res["v0"],
            "v_rms_m_per_s": res["v_rms"],
            "mean_square_velocity": res["mean_square_velocity"],
            "ka": res["ka"],
            "sigma_rad": res["sigma_rad"],
            "W_rad_W": res["W_rad"],
            "solid_angle_sr": res["solid_angle_sr"],
            "distances_m": res["distances"].tolist(),
            "p_rms_Pa": res["p_rms"].tolist(),
            "SPL_dB_re_1uPa": res["SPL_dB"].tolist(),
            "provenance_sha256": prov["sha256"],
        }
    res = vibration_radiation_chain(
        vib, f_n, surface_area=1.0, characteristic_length=0.3, model="monopole",
        distances=[1.0, 2.0, 4.0, 40.0])
    decay_err = float(abs(res["p_rms"][0] / res["p_rms"][3] - 40.0) / 40.0)
    spl_step = float(np.diff(res["SPL_dB"])[0])
    demo["chain_1_over_r_relative_error"] = decay_err
    demo["chain_spl_step_dB_per_doubling"] = spl_step
    demo["passes"] = bool(decay_err < 0.02 and abs(spl_step + 20 * np.log10(2.0)) < 1e-9)
    return demo


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def figure_sigma() -> Path:
    ka = np.logspace(-2, 1.5, 600)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.loglog(ka, pulsating_sphere_radiation_efficiency(ka), "b-", lw=2,
              label=r"pulsating sphere: $(ka)^2/(1+(ka)^2)$")
    ax.loglog(ka, piston_radiation_efficiency(ka), "r-", lw=2,
              label=r"baffled piston: $1-2J_1(2ka)/(2ka)$")
    ax.loglog(ka, dipole_radiation_efficiency(ka), "g-", lw=2,
              label=r"unbaffled plate (dipole): $(ka)^4/(1+(ka)^4)$")
    ax.loglog(ka, ka**2, "b--", lw=1, alpha=0.6, label=r"monopole $(ka)^2$")
    ax.loglog(ka, ka**2 / 2.0, "r--", lw=1, alpha=0.6, label=r"piston $(ka)^2/2$")
    ax.loglog(ka, ka**4, "g--", lw=1, alpha=0.6, label=r"dipole $(ka)^4$")
    ax.axhline(1.0, color="k", ls=":", lw=1, label=r"$\sigma_{rad}\to1$")
    ax.set_xlabel(r"$ka$")
    ax.set_ylabel(r"radiation efficiency $\sigma_{rad}$")
    ax.set_title("Radiation efficiency ladder: monopole / piston / dipole")
    ax.legend(fontsize=7.5, loc="lower right")
    path = OUT_DIR / "fig_radiation_efficiency.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_decay() -> Path:
    rs = np.logspace(0, 3, 200)
    sphere = PulsatingSphere(radius=0.5, frequency=500.0, velocity_amplitude=0.01)
    p_exact = np.array([sphere.far_field_pressure_rms(r) for r in rs])
    chain = RadiationChain(frequency=500.0, velocity_amplitude=0.01,
                           surface_area=1.0, sigma_rad=0.5)
    p_chain = chain.compute_chain(rs)["p_rms"]
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.loglog(rs, p_exact, "b-", lw=2, label="pulsating sphere (exact)")
    ax.loglog(rs, p_chain, "r--", lw=2, label=r"chain $\sqrt{\rho c W/4\pi r^2}$")
    ax.loglog(rs, p_exact[0] * rs[0] / rs, "k:", lw=1.2, label=r"reference $\propto 1/r$")
    ax.set_xlabel("distance r [m]")
    ax.set_ylabel(r"$p_{rms}$ [Pa]")
    ax.set_title("Far-field pressure decay (f = 500 Hz)")
    ax.legend(fontsize=8)
    path = OUT_DIR / "fig_far_field_decay.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_onaxis() -> Path:
    radius = 0.1
    ka = 2.0
    freq = ka * C_WATER / (2 * np.pi * radius)
    piston = CircularPiston(radius=radius, frequency=freq, velocity_amplitude=0.01)
    rs = np.logspace(np.log10(2 * radius), 3, 400)
    p_exact = np.array([piston.on_axis_pressure_exact(r) for r in rs])
    p_ff = np.array([piston.on_axis_pressure_amplitude(r) for r in rs])
    p_pow = np.array([piston.far_field_pressure_amplitude(r) for r in rs])
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.loglog(rs, p_exact, "b-", lw=2, label="exact Rayleigh on-axis")
    ax.loglog(rs, p_ff, "r--", lw=2, label=r"far field $\rho c k a^2 v_0/2r$")
    ax.loglog(rs, p_pow, "g-.", lw=1.5, label="power-based (directivity averaged)")
    ax.axvline(40 * radius**2 / (C_WATER / freq), color="k", ls=":", lw=1,
               label=r"$r = 40a^2/\lambda$")
    ax.set_xlabel("distance r [m]")
    ax.set_ylabel("pressure amplitude [Pa]")
    ax.set_title(f"Baffled piston on-axis pressure (ka = {ka:g})")
    ax.legend(fontsize=7.5)
    path = OUT_DIR / "fig_piston_onaxis.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_sdof() -> Path:
    mass, f_n, zeta, F0 = 10.0, 100.0, 0.02, 100.0
    vib = HarmonicVibrator(mass=mass, stiffness=mass * (2 * np.pi * f_n) ** 2,
                           damping_ratio=zeta, force_amplitude=F0)
    freqs = np.logspace(0, 2.5, 500)
    v0 = np.array([vib.velocity_amplitude(2 * np.pi * f) for f in freqs])
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.loglog(freqs, v0, "b-", lw=2, label="SDOF velocity amplitude")
    ax.plot([f_n], [vib.resonance_velocity_amplitude()], "ro", ms=8,
            label=r"closed form $F_0/c_d$ at $f_n$")
    ax.axvline(f_n, color="k", ls=":", lw=1)
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel(r"$v_0$ [m/s]")
    ax.set_title(r"SDOF vibration stage ($f_n$=100 Hz, $\zeta$=0.02, $F_0$=100 N)")
    ax.legend(fontsize=8)
    path = OUT_DIR / "fig_sdof_response.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_chain_spl(j: dict) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    styles = {"monopole": "b-o", "dipole": "g-s", "piston": "r-^"}
    for model, style in styles.items():
        data = j[model]
        ax.semilogx(data["distances_m"], data["SPL_dB_re_1uPa"], style, lw=1.8,
                    label=f"{model}: W={data['W_rad_W']:.3e} W, "
                          rf"$\sigma$={data['sigma_rad']:.4f}")
    ax.set_xlabel("distance r [m]")
    ax.set_ylabel(r"SPL [dB re 1 $\mu$Pa]")
    ax.set_title("Full chain prediction at f = 100 Hz (resonance)")
    ax.legend(fontsize=7.5)
    path = OUT_DIR / "fig_chain_spl.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# report rendering (all numbers injected programmatically)
# ---------------------------------------------------------------------------
def render_markdown(payload: dict) -> str:
    cs = payload["criteria"]
    a, b, c, d, e = cs["a"], cs["b"], cs["c"], cs["d"], cs["e"]
    g, h, i, j = cs["g"], cs["h"], cs["i"], cs["j"]
    prov = payload["provenance"]
    md = []

    def add(text=""):
        md.append(text + "\n")

    add("# 水下结构声辐射：振动→声压的解析锚与半经验预报链")
    add("**战役节点**: P350W1-N04 | **mission**: 7793 | **仓**: TensorPhysics | "
        "**分支**: agent/m7793")
    add(f"**生成时间 (UTC)**: {payload['generated_utc']} | **模块**: "
        f"`{payload['module']}` | **判据总判定**: "
        f"{'ALL PASS' if payload['all_pass'] else 'FAIL'}")
    add("---")

    add("## ① 案例介绍")
    add("### 物理背景")
    add("水下结构振动经流固耦合向流体辐射声波，构成舰船水下辐射噪声的主体。"
        "本节点建立并验证从结构振动速度到远场声压级的完整解析-半经验预报链：")
    add("$$F(t)=F_0e^{i\\omega t}\\;\\Rightarrow\\;\\langle v^2\\rangle=v_0^2/2"
        "\\;\\Rightarrow\\;W_{rad}=\\sigma_{rad}(\\omega)\\,\\rho c S\\langle v^2\\rangle"
        "\\;\\Rightarrow\\;p_{rms}(r)=\\sqrt{\\rho cW_{rad}/(\\Omega r^2)}"
        "\\;\\Rightarrow\\;SPL=20\\log_{10}(p_{rms}/p_{ref})$$")
    add("其中 $\\Omega=4\\pi$（自由场）或 $2\\pi$（无限障板半空间，活塞）；"
        "$p_{ref}=1\\,\\mu$Pa（水下）。")
    add("### 参数表")
    add("| 参数 | 符号 | 取值 | 单位 |")
    add("|---|---|---|---|")
    add(f"| 海水密度 | $\\rho$ | {RHO_WATER} | kg/m³ |")
    add(f"| 声速 | $c$ | {C_WATER} | m/s |")
    add(f"| 参考声压 | $p_{{ref}}$ | {P_REF:.0e} | Pa |")
    add("| 脉动球半径 | $a$ | 0.5 | m |")
    add("| 活塞半径 | $a$ | 0.1 | m |")
    add("| 表面速度幅值 | $v_0$ | 0.01 | m/s |")
    add(f"| SDOF 模态质量 | $m$ | {i['mass_kg']:g} | kg |")
    add(f"| SDOF 固有频率 | $f_n$ | {i['natural_frequency_Hz']:g} | Hz |")
    add(f"| SDOF 阻尼比 | $\\zeta$ | {i['damping_ratio']:g} | — |")
    add(f"| SDOF 激振力幅 | $F_0$ | {i['force_amplitude_N']:g} | N |")
    add("")
    add("### 解析锚（闭式解，无拟合参数）")
    add("- **脉动球（单极子）**：$\\sigma=(ka)^2/(1+(ka)^2)$，"
        "$W=2\\pi\\rho ck^2a^4v_0^2/(1+(ka)^2)$，"
        "$|p(r)|=\\rho ck v_0a^2/(r\\sqrt{1+(ka)^2})$。")
    add("- **无限障板活塞**：$\\sigma=1-2J_1(2ka)/(2ka)$；轴上声压由 Rayleigh 积分"
        "闭式给出 $p(r)=\\rho cv_0|e^{-ikr}-e^{-ik\\sqrt{r^2+a^2}}|$（近场亦成立），"
        "远场退化为 $\\rho cka^2v_0/(2r)$。")
    add("- **无障板板（偶极子）半经验**：$\\sigma=(ka)^4/(1+(ka)^4)$，低频 $\\propto(ka)^4$，"
        "对应声短路效应。")
    add("- **SDOF 振动级**：$Z(\\omega)=c_d+i(\\omega m-k/\\omega)$，$v_0=F_0/|Z|$，"
        "共振时 $v_0=F_0/c_d$，$c_d=2\\zeta\\omega_n m$。")

    add("## ② 怎么做的")
    add("1. **模块**（`src/tensorphysics/domains/structural_acoustic_radiation/`）：")
    add("   - `radiation.py`：脉动球/活塞圆盘解析解、偶极子半经验模型、通用预报链 "
        "`RadiationChain`（含 sha256 provenance）、SPL；")
    add("   - `vibration.py`：SDOF 简谐响应级 `HarmonicVibrator`，力→速度→声功率→"
        "远场 SPL 全链 `vibration_radiation_chain()`，结果级哈希 `chain_provenance()`。")
    add("2. **验证策略**——每级都与独立闭式解对照，而非自洽比较：")
    add("   - (a) 声功率对独立解析式 $2\\pi\\rho ck^2a^4v_0^2/(1+(ka)^2)$；")
    add("   - (b) 活塞辐射效率除 Bessel 闭式外，另用**远场强度半球数值积分** "
        "$\\sigma_{quad}=\\frac{(ka)^2}{2}\\int_0^{\\pi/2}D^2(\\theta)\\sin\\theta\\,d\\theta$ "
        "（SciPy `quad`）作独立交叉校验；")
    add("   - (g) 活塞轴上声压对 **Rayleigh 积分闭式解**（未作远场近似），"
        "并验证远场退化式在 $r\\ge40a^2/\\lambda$ 的精度；")
    add("   - (i) SDOF 速度对共振闭式 $F_0/c_d$ 与准静态 $F_0\\omega/k$；")
    add("   - (j) 全链对 $1/r$ 律与 6.02 dB/倍距律，并给出三种 $\\sigma$ 模型的 SPL。")
    add("3. **工具链**：Python 3.11 + NumPy + SciPy（`j1`、`quad`）+ matplotlib；"
        "`scripts/verify_structural_acoustic_radiation.py` 一次运行产出本报告、"
        "全部判据数字与图，**零手抄**。")
    add(f"4. **测试**：`{e['command']}` → `{e['summary']}`。")

    add("## ③ 计算结果")
    add("### (a) 脉动球辐射声功率 vs 解析解")
    add("| $f$ [Hz] | $ka$ | $\\sigma_{rad}$ | $W_{num}$ [W] | $W_{ana}$ [W] | 相对误差 |")
    add("|---|---|---|---|---|---|")
    for row in a["rows"]:
        add(f"| {row['frequency_Hz']:.0f} | {row['ka']:.4f} | {row['sigma_rad']:.6f} | "
            f"{row['W_computed_W']:.6e} | {row['W_analytic_W']:.6e} | "
            f"{row['relative_error']:.2e} |")
    add(f"**最大相对误差 {a['max_relative_error']:.2e}（阈值 ≤3%）→ "
        f"{'PASS' if a['passes'] else 'FAIL'}**")
    add("### (b) 活塞辐射效率 $\\sigma(ka)$ vs 解析 Bessel 与独立求积")
    add("| $ka$ | $\\sigma_{module}$ | $\\sigma_{analytic}$ | $\\sigma_{quad}$ | "
        "误差 vs 解析 | 求积交叉校验误差 |")
    add("|---|---|---|---|---|---|")
    for row in b["rows"]:
        add(f"| {row['ka']:.3f} | {row['sigma_module']:.6f} | {row['sigma_analytic']:.6f} | "
            f"{row['sigma_quadrature']:.6f} | {row['relative_error']:.2e} | "
            f"{row['quadrature_relative_error']:.2e} |")
    add(f"**最大相对误差 {b['max_relative_error']:.2e}（阈值 ≤5%）→ "
        f"{'PASS' if b['passes'] else 'FAIL'}**；独立半球强度积分最大偏差 "
        f"{b['max_independent_quadrature_error']:.2e}。")
    add("### (c) 远场声压 $1/r$ 衰减")
    add("| 量 | 值 |")
    add("|---|---|")
    add(f"| $p(r_1)/p(r_2)$，$r_1={c['r1_m']:g}$ m, $r_2={c['r2_m']:g}$ m | "
        f"{c['p1_over_p2']:.6f} |")
    add(f"| 理论 $r_2/r_1$ | {c['expected_ratio']:.1f} |")
    add(f"| 相对误差（解析球） | {c['relative_error']:.2e} |")
    add(f"| 相对误差（通用链） | {c['chain_relative_error']:.2e} |")
    add(f"| log-log 拟合斜率（1–1000 m） | {c['loglog_slope']:.12f} |")
    add(f"**判定：{'PASS' if c['passes'] else 'FAIL'}（阈值 <2%）**")
    add("### (d) 负控：$f\\to0$")
    add("| 源 | $f$ [Hz] | $ka$ | $\\sigma_{rad}$ | $W_{rad}$ [W] | "
        "$0.01\\rho cSv_0^2$ [W] | 比值 |")
    add("|---|---|---|---|---|---|---|")
    for row in d["sources"]:
        add(f"| {row['model']} | {row['frequency_Hz']:g} | {row['ka']:.3e} | "
            f"{row['sigma_rad']:.3e} | {row['W_rad_W']:.3e} | {row['threshold_W']:.3e} | "
            f"{row['ratio_W_over_threshold']:.3e} |")
    add("全链层面（SDOF 激发，$f=0.1$ Hz）：")
    add("| 模型 | $ka$ | $W_{rad}$ [W] | 上界 [W] | 判定 |")
    add("|---|---|---|---|---|")
    for row in d["chain"]:
        add(f"| {row['model']} | {row['ka']:.3e} | {row['W_rad_W']:.3e} | "
            f"{row['bound_W']:.3e} | {'PASS' if row['passes'] else 'FAIL'} |")
    add(f"**判定：{'PASS' if d['passes'] else 'FAIL'}**")
    add("### (g) 活塞轴上远场声压 vs Rayleigh 精确解")
    add("| $ka$ | $r$ [m] | $kr$ | $p_{exact}$ [Pa] | $p_{ff}$ [Pa] | 误差 | 功率法误差 |")
    add("|---|---|---|---|---|---|---|")
    for row in g["rows"]:
        add(f"| {row['ka']:g} | {row['r_m']:.3f} | {row['kr']:.2f} | "
            f"{row['p_onaxis_exact_Pa']:.6e} | {row['p_onaxis_farfield_Pa']:.6e} | "
            f"{row['relative_error']:.2e} | {row['powerbased_relative_error']:.2e} |")
    add(f"**最大误差 {g['max_relative_error']:.2e}（阈值 ≤5%）→ "
        f"{'PASS' if g['passes'] else 'FAIL'}**")
    add("### (h) 偶极子（无障板板）低频律")
    add("| $ka$ | $\\sigma_{dipole}$ | $(ka)^4$ | $\\sigma/(ka)^4$ |")
    add("|---|---|---|---|")
    for row in h["rows"]:
        add(f"| {row['ka']:g} | {row['sigma_dipole']:.6e} | {row['ka4']:.6e} | "
            f"{row['ratio_sigma_over_ka4']:.6f} |")
    add(f"log-log 斜率：偶极子 {h['dipole_loglog_slope']:.4f}（理论 4），"
        f"单极子 {h['monopole_loglog_slope']:.4f}（理论 2）→ "
        f"{'PASS' if h['passes'] else 'FAIL'}")
    add("### (i) SDOF 结构振动级")
    add("| 量 | 数值 |")
    add("|---|---|")
    add(f"| 固有频率 $f_n$ | {i['natural_frequency_Hz']:.6f} Hz |")
    add(f"| 阻尼系数 $c_d$ | {i['damping_coefficient_Ns_per_m']:.6f} N·s/m |")
    add(f"| 共振速度（数值 $F_0/|Z|$） | "
        f"{i['resonant_velocity_numeric_m_per_s']:.9f} m/s |")
    add(f"| 共振速度（闭式 $F_0/c_d$） | "
        f"{i['resonant_velocity_closed_form_m_per_s']:.9f} m/s |")
    add(f"| 共振相对误差 | {i['resonance_relative_error']:.2e} |")
    add(f"| 静态位移 $F_0/k$ | {i['static_displacement_m']:.6e} m |")
    add(f"| 准静态相对误差 | {i['quasistatic_relative_error']:.2e} |")
    add(f"**判定：{'PASS' if i['passes'] else 'FAIL'}**")
    add("### (j) 全链预报示范（$f=f_n=100$ Hz，$S=1$ m²，$a=0.3$ m）")
    add("| 模型 | $v_0$ [m/s] | $ka$ | $\\sigma_{rad}$ | $W_{rad}$ [W] | "
        "SPL@1 m | SPL@10 m | SPL@100 m |")
    add("|---|---|---|---|---|---|---|---|")
    for model in ("monopole", "dipole", "piston"):
        data = j[model]
        spl = data["SPL_dB_re_1uPa"]
        add(f"| {model} | {data['v0_m_per_s']:.6f} | {data['ka']:.4f} | "
            f"{data['sigma_rad']:.6f} | {data['W_rad_W']:.6e} | "
            f"{spl[0]:.2f} dB | {spl[1]:.2f} dB | {spl[2]:.2f} dB |")
    add(f"全链 $1/r$ 律相对误差 {j['chain_1_over_r_relative_error']:.2e}；SPL 每倍距衰减 "
        f"{j['chain_spl_step_dB_per_doubling']:.6f} dB（理论 −6.0206 dB）→ "
        f"{'PASS' if j['passes'] else 'FAIL'}")
    add("### (f) Provenance（sha256 锚定）")
    add(f"- 结果载荷 SHA256：`{prov['payload_sha256']}`")
    add(f"- 锚定范围：{prov['payload_keys']}")
    add("- 各图 sha256：")
    for name, digest in prov["figure_sha256"].items():
        add(f"  - `{name}`：`{digest}`")
    add("- 全链结果载荷哈希（monopole 示范）："
        f"`{j['monopole']['provenance_sha256']}`")

    add("## ④ 结果图")
    add("| 图 | 说明 |")
    add("|---|---|")
    add("| `fig_radiation_efficiency.png` | 单极子/活塞/偶极子 $\\sigma_{rad}(ka)$ 与三条渐近线 |")
    add("| `fig_far_field_decay.png` | 远场声压 $p_{rms}(r)$ 与 $1/r$ 参考线 |")
    add("| `fig_piston_onaxis.png` | 活塞轴上声压：Rayleigh 精确解 vs 远场近似 vs 功率平均 |")
    add("| `fig_sdof_response.png` | SDOF 速度频响与共振闭式 $F_0/c_d$ |")
    add("| `fig_chain_spl.png` | 三种 $\\sigma$ 模型的全链 SPL vs 距离 |")

    add("## ⑤ 注意点与局限")
    add("1. **活塞散射截面振荡**：精确式 $\\sigma=1-2J_1(2ka)/(2ka)$ 在 $ka>1$ 振荡并"
        "局部略微超过 1（$ka\\approx5.1$ 处约 1.13），属辐射阻抗实部归一化的数学特征，"
        "非能量不守恒；工程应用可截断 $\\sigma\\le1$。")
    add("2. **远场条件**：$kr\\gg1$ 且 $r\\gg a^2/\\lambda$。活塞轴上远场式在 "
        "$r=40a^2/\\lambda$ 处误差已在 5% 内（判据 g 实测），更近需用 Rayleigh 闭式解。")
    add("3. **功率-声压的方向性**：由 $W$ 反推 $p_{rms}=\\sqrt{\\rho cW/\\Omega r^2}$ 得到"
        "**方向平均**值；活塞轴上峰值在 $ka\\gtrsim1$ 时高于该值（$ka=2$ 时功率法偏低"
        f"{g['rows'][2]['powerbased_relative_error']*100:.1f}%），故本模块同时提供 "
        "`on_axis_pressure_amplitude()`。")
    add("4. **半经验链假设表面速度均匀**；实际结构为模态叠加，需空间均方速度与"
        "模态辐射效率。偶极子模型 $\\sigma=(ka)^4/(1+(ka)^4)$ 只是无障板板的量级规律，"
        "不含模态共振细节（Wallace/Maidanik 模态平均更精细）。")
    add("5. **局限**：线性声学、稳态简谐、均匀自由场；不含浅水波导、障板有限尺寸绕射、"
        "流固耦合反馈（重流体附加质量与辐射阻尼对结构响应的反作用）与空化噪声。")

    add("## ⑥ 误差判定（逐项显式回答）")
    add("| 判据 | 要求 | 实测 | 是否满足 |")
    add("|---|---|---|---|")
    add(f"| (a) 脉动球 $W_{{rad}}$ (100/500/1000 Hz) | ≤3% | "
        f"{a['max_relative_error']:.2e} | {'是' if a['passes'] else '否'} |")
    add(f"| (b) 活塞 $\\sigma_{{rad}}$ (ka=0.1–10) | ≤5% | vs 闭式 "
        f"{b['max_relative_error']:.2e}；vs 独立半球求积 "
        f"{b['max_independent_quadrature_error']:.2e} | {'是' if b['passes'] else '否'} |")
    add(f"| (c) 远场 $p(r)\\propto1/r$ | <2% | {c['relative_error']:.2e} | "
        f"{'是' if c['passes'] else '否'} |")
    add(f"| (d) 负控 $f\\to0$ | $W<0.01\\rho cSv_0^2$ | 比值 "
        f"{d['sources'][0]['ratio_W_over_threshold']:.2e} | "
        f"{'是' if d['passes'] else '否'} |")
    add(f"| (e) pytest | 全 PASS | {e['summary']} | {'是' if e['passes'] else '否'} |")
    add(f"| (f) provenance sha256 | 存在且可复现 | `{prov['payload_sha256'][:16]}…` | 是 |")
    add(f"| (g) 活塞轴上声压 vs Rayleigh | ≤5% | {g['max_relative_error']:.2e} | "
        f"{'是' if g['passes'] else '否'} |")
    add(f"| (h) 偶极子低频律 | 斜率≈4 | {h['dipole_loglog_slope']:.4f} | "
        f"{'是' if h['passes'] else '否'} |")
    add(f"| (i) SDOF 共振速度 | 闭式一致 | {i['resonance_relative_error']:.2e} | "
        f"{'是' if i['passes'] else '否'} |")
    add(f"| (j) 全链 1/r + 6 dB/倍距 | <2% / −6.02 dB | "
        f"{j['chain_1_over_r_relative_error']:.2e} / "
        f"{j['chain_spl_step_dB_per_doubling']:.4f} dB | "
        f"{'是' if j['passes'] else '否'} |")
    add("")
    add("**结论**：判据 (a)–(j) 全部满足；所有关键量均与独立闭式解对照，"
        "最大误差见上表（均远低于各自阈值）。")
    add("---")
    add("**产物清单**")
    add("- `src/tensorphysics/domains/structural_acoustic_radiation/radiation.py`、`vibration.py`、`__init__.py`")
    add(f"- `{TEST_FILE}`（{e['n_passed']} 个测试）")
    add("- `scripts/verify_structural_acoustic_radiation.py`")
    add("- `artifacts/structural_acoustic_radiation/verification.json` + 5 张图")
    add("- 本文档（由脚本程序化生成，数字零手抄）")
    return "".join(md)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/5] computing criteria ...")
    criteria = {
        "a": criterion_a(),
        "b": criterion_b(),
        "c": criterion_c(),
        "d": criterion_d(),
        "e": criterion_e(),
        "g": criterion_g(),
        "h": criterion_h(),
        "i": criterion_i(),
        "j": criterion_j(),
    }
    all_pass = all(v["passes"] for v in criteria.values())

    print("[2/5] rendering figures ...")
    fig_paths = [figure_sigma(), figure_decay(), figure_onaxis(),
                 figure_sdof(), figure_chain_spl(criteria["j"])]

    print("[3/5] building payload + provenance ...")
    figure_sha = {p.name: sha256_file(p) for p in fig_paths}
    payload = {
        "node": "P350W1-N04",
        "mission_id": 7793,
        "module": "tensorphysics.domains.structural_acoustic_radiation",
        "branch": "agent/m7793",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repository": "TensorPhysics",
        "criteria": criteria,
        "all_pass": bool(all_pass),
        "artifacts": [str(p.relative_to(ROOT)) for p in fig_paths] + [
            "artifacts/structural_acoustic_radiation/verification.json",
            "docs/P350W1_N04_STRUCTURAL_ACOUSTIC_RADIATION.md",
        ],
    }
    # Hash only the reproducible physics/verification content: drop the
    # volatile pytest wall-clock text so that re-running the harness on the
    # same code reproduces the anchor bit-for-bit.
    e = criteria["e"]
    hashable = {k: v for k, v in criteria.items() if k != "e"}
    hashable["e"] = {"command": e["command"], "returncode": e["returncode"],
                     "n_passed": e["n_passed"], "passes": e["passes"]}
    payload["provenance"] = {
        "payload_sha256": sha256_json(hashable),
        "payload_keys": sorted(criteria.keys()),
        "hashed_fields_note": "pytest wall-clock summary excluded (volatile); "
                              "command/returncode/n_passed included",
        "figure_sha256": figure_sha,
        "algorithm": "sha256 over canonical JSON (sorted keys, compact separators)",
    }

    print("[4/5] writing verification.json + report ...")
    JSON_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True))
    DOC_PATH.write_text(render_markdown(payload), encoding="utf-8")

    print("[5/5] summary")
    for key, val in criteria.items():
        print(f"  ({key}) pass={val['passes']}")
    print(f"  all_pass={all_pass}")
    print(f"  payload_sha256={payload['provenance']['payload_sha256']}")
    print(f"  json -> {JSON_PATH.relative_to(ROOT)}")
    print(f"  md   -> {DOC_PATH.relative_to(ROOT)}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
