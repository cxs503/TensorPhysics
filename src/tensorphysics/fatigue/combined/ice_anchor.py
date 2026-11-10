"""Ice-induced stress anchor: reduced-order DEM ice load -> hull-girder modal stress.

This module carries the **physical ice branch** of the ice+wave combination-law
study.  It is a faithful port of the calibrated ice-fatigue chain of the
ship-platform mission m5528 (DAG IN32-N07, "冰激振动疲劳累积损伤闭环"), so that
switching the wave branch off reproduces that reference result within the
pre-declared 2 % negative-control tolerance.  Provenance of every constant is
recorded in ``ANCHOR_CASE`` / ``ICE_PARAMS_CALIBRATED`` below.

Chain
-----
1. **Ice load** ``F(t)`` -- event-driven reduced-order DEM contact/crushing model
   (per zone: Hertz loading ``F = k_c (s-s_0)^n`` while the ice structure is
   indented at ``v_indent``; at ``F_cr = sigma_c A Lambda`` with
   ``Lambda ~ LogNormal(1, CoV)`` the zone unloads to ``beta F_cr`` and advances
   the indentation by ``delta_crush``).  The three zones are phase-shifted and
   their indentation rates are slowly modulated (floe encounter).  The
   equivalent contact parameters ``(k_c, sigma_c)`` are the *calibrated* values
   of m5528 (``calibrate`` there scored 12 candidate pairs against full-scale
   load statistics and the observed 20-40 % whipping share); they are imported
   here as data, not re-fitted.
2. **Modal superposition** -- free-free Euler-Bernoulli hull-girder modes
   (TensorFEM-free analytical reference, 2-node upward), exact ZOH discretisation
   of each modal SDOF ``q'' + 2 zeta w q' + w^2 q = phi(a) F / m_n``, 10 modes,
   zeta = 1.5 %.
3. **Stress recovery** at the detail (x_data, z_deck):
   ``sigma(t) = -E z sum_k q_k(t) phi_k''(x_data)`` (nominal), hot-spot = SCF x
   nominal.
4. **Whipping share** (m5528 convention): ``sigma_whip = sigma - sigma_qs`` with
   ``sigma_qs`` the modal-static limit of the *same* load;
   ``share_peak = max|sigma_whip| / max|sigma|``,
   ``share_rms = rms(sigma_whip - mean) / rms(sigma - mean)``.

Units: SI (m, s, N, Pa) internally; stresses are reported in MPa.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import signal

from ..rainflow import rainflow_counting_stack
from .sn import BASELINE_SN, SNCurveF3, SNParams

#: The reference implementation (m5528) derives its bilinear knee from the
#: *CP-shifted* log a1, which places it at 23.48 MPa instead of the DNV value
#: 31.86 MPa.  Reproducing the published damage requires that inherited rule;
#: the DNV-standard rule is reported alongside (see ``knee_rule_note``).
LEGACY_CURVE = SNCurveF3(SNParams(knee_rule="legacy_m5528"))

__all__ = [
    "ANCHOR_CASE",
    "ANCHOR_REFERENCE_RESULT",
    "LEGACY_CURVE",
    "ICE_PARAMS_CALIBRATED",
    "IceContactParams",
    "generate_ice_force",
    "load_statistics",
    "FreeFreeBeamModes",
    "modal_stress_history",
    "whipping_share",
    "slow_varying_ice_params",
    "run_anchor_case",
]

# ---------------------------------------------------------------------------
# Reference configuration (ship-platform mission m5528, artifacts/ice_fatigue/
# results.json -- 13 万吨双壳油船/散货船典型冰区焊接接头).
# ---------------------------------------------------------------------------
ANCHOR_CASE: Dict[str, float] = {
    "length_m": 250.0,
    "EI_Nm2": 5.0e13,
    "mass_per_length_kg_m": 5.2e5,
    "z_deck_m": 10.5,          # detail height above the neutral axis
    "Z_deck_m3": 23.1,         # deck section modulus (consistency check)
    "E_Pa": 2.06e11,
    "n_modes": 10,
    "zeta": 0.015,
    "scf": 2.5,
    "thickness_mm": 28.0,
    "s_uts_MPa": 490.0,
    "ice_hours_per_year": 2500.0,
    "duration_s": 300.0,
    "dt_s": 0.002,
    "x_load_m": 12.5,          # bow impact station
    "x_detail_m": 75.0,        # detail station (midbody fore end)
    "settle_time_s": 60.0,     # start-up transient window discarded
}

# Calibrated equivalent DEM parameters (m5528 calibration.best_params).
ICE_PARAMS_CALIBRATED: Dict[str, float] = {
    "n_zones": 3,
    "v_indent": 0.12,
    "k_contact": 1.0e8,
    "exp_contact": 1.5,
    "sigma_crush": 2.0e6,
    "area_zone": 2.5,
    "crush_depth": 0.8,
    "residual_ratio": 0.7,
    "decay_time": 0.15,
    "strength_cov": 0.3,
    "modulation": 0.25,
    "mod_freq": 0.03,
    "seed": 20261106,
}

# Published m5528 result (the negative-control target, criterion (e)).
ANCHOR_REFERENCE_RESULT: Dict[str, float] = {
    "damage_annual_goodman": 0.37643677999163916,
    "damage_annual_no_goodman": 0.47919402222805674,
    "life_years": 2.656488561033304,
    "whipping_share_peak": 0.35108652124027145,
    "whipping_share_rms": 0.7591470731209626,
    "nominal_rms_MPa": 3.7256474569628737,
    "nominal_peak_MPa": 29.587849671171814,
    "hotspot_rms_MPa": 9.314118642407184,
}


# ---------------------------------------------------------------------------
# 1. Ice load
# ---------------------------------------------------------------------------
@dataclass
class IceContactParams:
    """Equivalent per-zone DEM contact/crushing parameters."""

    n_zones: int = 3
    v_indent: float = 0.12
    k_contact: float = 1.0e8
    exp_contact: float = 1.5
    sigma_crush: float = 2.0e6
    area_zone: float = 2.5
    crush_depth: float = 0.8
    residual_ratio: float = 0.7
    decay_time: float = 0.15
    strength_cov: float = 0.3
    modulation: float = 0.25
    mod_freq: float = 0.03
    seed: int = 20261106

    @classmethod
    def calibrated(cls, **overrides) -> "IceContactParams":
        params = dict(ICE_PARAMS_CALIBRATED)
        params.update(overrides)
        return cls(**params)


def generate_ice_force(
    params: IceContactParams, duration: float = 300.0, dt: float = 2.0e-3
) -> Tuple[np.ndarray, np.ndarray]:
    """Total ice force F(t) [N] from the event-driven reduced-order DEM model.

    Per zone: rise phase ``F = k_c (s0 + v t)^n`` clipped at ``F_cr``, then a
    finite-time decay ``F = F_cr [beta + (1-beta) e^{-t/tau}]`` over a crushing
    window ``delta_crush/v``.  Crushing strength is redrawn log-normally per
    event (spatial randomness of the ice sheet), and the indentation rate is
    modulated by the floe-encounter sinusoid.
    """
    n = int(round(duration / dt))
    t = np.arange(n) * dt
    total = np.zeros(n)
    beta = params.residual_ratio
    expo = params.exp_contact
    ln_sig = math.sqrt(math.log(1.0 + params.strength_cov ** 2))
    ln_mu = math.log(max(1e-9, 1.0 - params.strength_cov ** 2)) * 0.5

    for zone in range(params.n_zones):
        rng = np.random.default_rng(params.seed + 101 * zone)
        area = params.area_zone * (1.0 + 0.15 * rng.standard_normal())
        area = max(area, 0.2 * params.area_zone)
        phase = rng.uniform(0.0, 2.0 * math.pi)
        f_mod = params.mod_freq * (0.7 + 0.6 * rng.random())
        fzone = np.zeros(n)
        t_cur = float(rng.uniform(0.0, 0.5))
        while t_cur < duration:
            v = params.v_indent * (
                1.0 + params.modulation * math.sin(2.0 * math.pi * f_mod * t_cur + phase)
            )
            v = max(v, 0.15 * params.v_indent)
            f_cr = params.sigma_crush * area * math.exp(
                ln_mu + ln_sig * rng.standard_normal()
            )
            s0 = (beta * f_cr / params.k_contact) ** (1.0 / expo)
            s1 = (f_cr / params.k_contact) ** (1.0 / expo)
            t_r = max((s1 - s0) / v, dt)
            i0 = int(round(t_cur / dt))
            i1 = min(n, i0 + max(2, int(round(t_r / dt))))
            if i0 < n:
                tt = t[i0:i1] - t[i0]
                rise = np.maximum(params.k_contact * (s0 + v * tt) ** expo, 0.0)[: i1 - i0]
                fzone[i0:i1] = np.minimum(rise, f_cr)
            t_cur = t[i0] + t_r if i0 < n else duration
            t_c = max(params.crush_depth * (0.7 + 0.6 * rng.random()) / v, dt)
            j0 = int(round(t_cur / dt))
            j1 = min(n, j0 + max(2, int(round(t_c / dt))))
            if j0 < n:
                tau = max(params.decay_time, dt)
                tt = t[j0:j1] - t[j0]
                fzone[j0:j1] = f_cr * (beta + (1.0 - beta) * np.exp(-tt / tau))[: j1 - j0]
            t_cur = t[j0] + t_c if j0 < n else duration
        total += fzone
    return t, total


def load_statistics(F: np.ndarray, dt: float, f_cut: float = 1.0) -> Dict[str, float]:
    """Full-scale load statistics used by the m5528 calibration (mean/peak, rise time)."""
    F = np.asarray(F, dtype=float)
    n = F.size
    peak = float(np.max(F)) if n else 0.0
    mean = float(np.mean(F)) if n else 0.0
    out = {
        "mean_N": mean,
        "peak_N": peak,
        "mean_peak": mean / peak if peak > 0.0 else 0.0,
        "std_N": float(np.std(F)) if n else 0.0,
    }
    if n > 3:
        d = np.diff(F)
        sign = np.sign(d)
        sign[sign == 0] = 1
        turns = np.where(np.diff(sign) != 0)[0] + 1
        rise = [float((turns[j + 1] - turns[j]) * dt)
                for j in range(len(turns) - 1) if F[turns[j + 1]] > F[turns[j]]]
        if rise:
            out["rise_median_s"] = float(np.median(rise))
            out["rise_count"] = float(len(rise))
    return out


# ---------------------------------------------------------------------------
# 2./3. Free-free Euler-Bernoulli modal superposition
# ---------------------------------------------------------------------------
_BETA_L_FREE = np.array(
    [4.730040744862704, 7.853204624095838, 10.995607838001671, 14.137165491257464,
     17.278759657399481, 20.420352248333657, 23.561944902040446, 26.703537555508192,
     29.845130209103256, 32.986722862692826]
)


class FreeFreeBeamModes:
    """Uniform free-free Euler-Bernoulli modes (no rigid-body modes)."""

    def __init__(self, length: float, EI: float, mass_per_length: float,
                 n_modes: int = 10, zeta: float = 0.015) -> None:
        if n_modes < 1:
            raise ValueError("n_modes >= 1")
        self.L = float(length)
        self.n_modes = int(n_modes)
        if self.n_modes <= _BETA_L_FREE.size:
            beta = _BETA_L_FREE[: self.n_modes] / self.L
        else:
            extra = (2.0 * np.arange(_BETA_L_FREE.size, self.n_modes) + 1.0) * math.pi / 2.0
            beta = np.concatenate([_BETA_L_FREE, extra]) / self.L
        self.beta = beta
        self.omega = beta ** 2 * math.sqrt(float(EI) / float(mass_per_length))
        self.freq = self.omega / (2.0 * math.pi)
        self.zeta = np.full(self.n_modes, float(zeta))
        xs = np.linspace(0.0, self.L, 2001)
        self.modal_mass = np.array(
            [float(mass_per_length) * np.trapezoid(self.shape(xs, k) ** 2, xs)
             for k in range(self.n_modes)]
        )

    def _sigma(self, k: int) -> float:
        b = self.beta[k]
        return (math.cosh(b * self.L) - math.cos(b * self.L)) / (
            math.sinh(b * self.L) - math.sin(b * self.L))

    def shape(self, x, k: int) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        b = self.beta[k]
        s = self._sigma(k)
        return np.cosh(b * x) + np.cos(b * x) - s * (np.sinh(b * x) + np.sin(b * x))

    def curvature_shape(self, x, k: int) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        b = self.beta[k]
        s = self._sigma(k)
        return b ** 2 * (np.cosh(b * x) - np.cos(b * x) - s * (np.sinh(b * x) - np.sin(b * x)))

    def stress_from_modal(self, q: np.ndarray, x_detail: float, z: float, E: float) -> np.ndarray:
        kap = np.zeros(q.shape[1])
        for k in range(q.shape[0]):
            kap += q[k] * float(self.curvature_shape(x_detail, k))
        return -float(E) * float(z) * kap


def modal_stress_history(
    modes: FreeFreeBeamModes, force: np.ndarray, dt: float,
    x_load: float, x_detail: float, z: float, E: float,
) -> Dict[str, np.ndarray]:
    """Nominal detail stress ``sigma(t)`` and its modal-static limit ``sigma_qs(t)``."""
    force = np.asarray(force, dtype=float)
    nt = force.size
    kdyn = modes.n_modes
    q = np.zeros((kdyn, nt))
    for k in range(kdyn):
        m_n, w, zeta = modes.modal_mass[k], modes.omega[k], modes.zeta[k]
        dnum, dden, _ = signal.cont2discrete(
            ([1.0], [m_n, 2.0 * zeta * w * m_n, w * w * m_n]), dt, method="zoh")
        Qn = force * float(modes.shape(np.array([x_load]), k)[0])
        q[k] = signal.lfilter(np.asarray(dnum).ravel(), np.asarray(dden).ravel(), Qn)
    sigma = modes.stress_from_modal(q, x_detail, z, E)

    q_static = np.zeros((kdyn, nt))
    for k in range(kdyn):
        k_n = modes.omega[k] ** 2 * modes.modal_mass[k]
        q_static[k] = force * float(modes.shape(np.array([x_load]), k)[0]) / k_n
    sigma_qs = modes.stress_from_modal(q_static, x_detail, z, E)
    return {"sigma": sigma, "sigma_qs": sigma_qs, "q": q}


def whipping_share(sigma: np.ndarray, sigma_qs: np.ndarray) -> Dict[str, float]:
    """m5528 whipping-share convention (peak and RMS measures)."""
    s = np.asarray(sigma, dtype=float)
    q = np.asarray(sigma_qs, dtype=float)
    s0 = s - s.mean()
    w0 = (s - q) - (s - q).mean()
    rms_s = float(np.sqrt(np.mean(s0 ** 2)))
    rms_w = float(np.sqrt(np.mean(w0 ** 2)))
    peak_s = float(np.max(np.abs(s)))
    peak_w = float(np.max(np.abs(s - q)))
    return {
        "share_rms": rms_w / rms_s if rms_s > 0.0 else 0.0,
        "share_peak": peak_w / peak_s if peak_s > 0.0 else 0.0,
        "rms_total": rms_s,
        "rms_whip": rms_w,
        "peak_total": peak_s,
        "peak_whip": peak_w,
    }


def slow_varying_ice_params(v_indent: float = 0.02, seed: int = 20261106) -> IceContactParams:
    """Slow-varying (quasi-static) ice crushing: long rise times, low whipping share.

    ``v_indent`` is reduced by 6x w.r.t. the calibrated lock-in value, which
    stretches the rise time of every crushing pulse far beyond the first
    hull-girder period (1/f1 = 1.79 s) -- the regime in which m5528 reports a
    whipping share below 8 %.
    """
    return IceContactParams.calibrated(v_indent=v_indent, seed=seed)


# ---------------------------------------------------------------------------
# 4. Anchor case (criterion (e) negative control)
# ---------------------------------------------------------------------------
def run_anchor_case(
    case: Optional[Dict[str, float]] = None,
    params: Optional[IceContactParams] = None,
    with_wave: Optional[np.ndarray] = None,
    dt_override: Optional[float] = None,
    duration_override: Optional[float] = None,
    curve: Optional[SNCurveF3] = None,
) -> Dict[str, object]:
    """Ice-only (optionally +wave) fatigue damage of the m5528 reference case.

    ``with_wave`` may inject an externally generated wave-stress history [Pa]
    of the same length as the ice history; with ``None`` the result is the pure
    ice chain and must reproduce ``ANCHOR_REFERENCE_RESULT``.
    """
    cfg = dict(ANCHOR_CASE)
    if case:
        cfg.update(case)
    dt = float(cfg["dt_s"] if dt_override is None else dt_override)
    duration = float(cfg["duration_s"] if duration_override is None else duration_override)
    prm = params or IceContactParams.calibrated()

    modes = FreeFreeBeamModes(cfg["length_m"], cfg["EI_Nm2"], cfg["mass_per_length_kg_m"],
                              n_modes=int(cfg["n_modes"]), zeta=cfg["zeta"])
    f1 = float(modes.freq[0])
    f_cut = 0.5 * f1

    _, F = generate_ice_force(prm, duration=duration, dt=dt)
    skip = min(int(cfg["settle_time_s"] / dt), max(0, F.size // 2))

    resp = modal_stress_history(modes, F, dt, cfg["x_load_m"], cfg["x_detail_m"],
                                cfg["z_deck_m"], cfg["E_Pa"])
    if with_wave is not None:
        wave = np.asarray(with_wave, dtype=float)
        if wave.size != resp["sigma"].size:
            raise ValueError("wave history length must match the ice history")
        resp["sigma"] = resp["sigma"] + wave

    sigma_nom = resp["sigma"][skip:]
    sigma_qs = resp["sigma_qs"][skip:]
    ws = whipping_share(sigma_nom, sigma_qs)
    sigma_hot = cfg["scf"] * sigma_nom / 1.0e6

    rf = rainflow_counting_stack(sigma_hot)
    curve = curve or LEGACY_CURVE
    window_hours = (duration - cfg["settle_time_s"]) / 3600.0
    annual_scale = cfg["ice_hours_per_year"] / window_hours if window_hours > 0 else 1.0

    D_no_goodman = curve.damage(rf.ranges, rf.counts, thickness_mm=cfg["thickness_mm"],
                               exposure_scale=annual_scale)
    D_goodman = curve.damage(rf.ranges, rf.counts, means=rf.means, s_uts=cfg["s_uts_MPa"],
                             thickness_mm=cfg["thickness_mm"], exposure_scale=annual_scale)
    D_standard = SNCurveF3().damage(
        rf.ranges, rf.counts, means=rf.means, s_uts=cfg["s_uts_MPa"],
        thickness_mm=cfg["thickness_mm"], exposure_scale=annual_scale)
    D_dnv = SNCurveF3(SNParams(thickness_rule="dnv")).damage(
        rf.ranges, rf.counts, means=rf.means, s_uts=cfg["s_uts_MPa"],
        thickness_mm=cfg["thickness_mm"], exposure_scale=annual_scale)

    return {
        "case": cfg,
        "ice_params": prm.__dict__.copy(),
        "modal": {
            "f1_Hz": f1,
            "f2_Hz": float(modes.freq[1]),
            "f_cut_Hz": f_cut,
            "modal_mass_kg": modes.modal_mass.tolist(),
        },
        "load": load_statistics(F[skip:], dt, f_cut=f_cut),
        "whipping": ws,
        "stress": {
            "nominal_rms_MPa": float(np.std(sigma_nom / 1.0e6)),
            "nominal_peak_MPa": float(np.max(np.abs(sigma_nom / 1.0e6))),
            "hotspot_rms_MPa": float(np.std(sigma_hot)),
            "hotspot_peak_MPa": float(np.max(np.abs(sigma_hot))),
        },
        "rainflow": {
            "n_cycles_window": float(rf.counts.sum()),
            "n_cycles_annual": float(rf.counts.sum() * annual_scale),
            "range_max_MPa": float(np.max(rf.ranges)) if rf.ranges.size else 0.0,
        },
        "fatigue": {
            "damage_annual_no_goodman": float(D_no_goodman),
            "damage_annual_goodman": float(D_goodman),
            "life_years": float(1.0 / D_goodman) if D_goodman > 0 else math.inf,
            "damage_annual_goodman_standard_knee": float(D_standard),
            "damage_annual_goodman_dnv_thickness": float(D_dnv),
            "life_years_standard_knee": float(1.0 / D_standard) if D_standard > 0 else math.inf,
            "annual_scale": float(annual_scale),
        },
        "sn": {
            "curve": "DNV-RP-C203 F3 seawater+CP",
            "knee_rule": curve.params.knee_rule,
            "log_a1": curve.params.log_a1,
            "log_a2": curve.params.log_a2,
            "range_knee_MPa": curve.params.range_knee_effective,
            "range_knee_standard_MPa": BASELINE_SN.range_knee_effective,
            "thickness_factor": curve.params.thickness_factor(cfg["thickness_mm"]),
        },
    }