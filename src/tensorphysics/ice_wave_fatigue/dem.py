"""Reduced-order DEM ice-crushing load generator (m5528 calibrated parameter set).

Provenance
----------
This is a **faithful port** of the reduced-order event-driven DEM contact /
crushing model first built in node ``IN32-N07`` (mission ``m5528``,
``backend/modules/ice_fatigue/ice_dem.py`` of the ship-performance platform) and
re-published here so that the combined ice+wave fatigue study of node
``P82W4-N06`` (mission ``m7361``) can regenerate the *identical* ice load series
without importing the platform repository.  Physics, event logic, random-number
seeding and default parameters are unchanged; only naming, typing and the
module boundary differ.  Criterion ``(e)`` of this node (single-load negative
control: waves off must reproduce the m5528 damage inside 2 %) is an end-to-end
check that the port is faithful.

Model (per contact zone, event driven)
--------------------------------------
Loading phase
    ``F = k_c (s - s_0) ** exp``, indentation ``s`` growing at ``v_indent``;
    the rise time follows from ``s_1 = (F_cr/k_c)**(1/exp)``.
Crushing phase
    ``F = F_cr [beta + (1 - beta) exp(-(t - t_1)/tau_dec)]`` with finite decay
    time ``tau_dec`` (a crushing event is not an instantaneous step, otherwise
    spurious high-frequency energy would be injected into the whipping band).
Crushing strength
    ``F_cr = sigma_c * A * Lambda``, ``Lambda ~ LogNormal(1, CoV)`` redrawn per
    event (spatial randomness of ice strength).
Slow modulation
    each event's indentation rate is sampled as
    ``v_indent (1 + mu sin(2 pi f_mod t + phi))`` (floe-encounter undulation).

Units: SI (m, s, N, Pa).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Dict, Tuple

import numpy as np
from scipy import signal

__all__ = [
    "IceContactParams",
    "CALIBRATED_M5528",
    "generate_ice_force",
    "load_statistics",
    "scaled_drift_rate",
]


@dataclass(frozen=True)
class IceContactParams:
    """Equivalent (statistically calibrated) DEM contact parameters.

    Defaults are the calibration result of node IN32-N07 (m5528): the grid
    search minimised the normalised deviation of the load statistics
    (mean/peak, rise-time median, quasi-static energy fraction) *and* of the
    whipping stress share against the target bands reported for full-scale
    ice-going ships.
    """

    n_zones: int = 3                 # number of simultaneous crushing contact zones
    v_indent: float = 0.12           # [m/s] indentation rate
    k_contact: float = 1.0e8         # [N/m^exp] contact stiffness
    exp_contact: float = 1.5         # [-] stiffness exponent (Hertz-like)
    sigma_crush: float = 2.0e6       # [Pa] crushing strength
    area_zone: float = 2.5           # [m^2] contact area per zone
    crush_depth: float = 0.8         # [m] indentation per event
    residual_ratio: float = 0.70     # [-] residual force ratio after crushing
    decay_time: float = 0.15         # [s] unloading characteristic time
    strength_cov: float = 0.30       # [-] lognormal CoV of crushing strength
    modulation: float = 0.25         # [-] floe-encounter slow modulation amplitude
    mod_freq: float = 0.03           # [Hz] slow modulation frequency
    seed: int = 20261106

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    def with_v_indent(self, v_indent: float) -> "IceContactParams":
        """Same ice field, different drift (indentation) rate.

        ``v_indent`` is the only knob that changes the *event rate* without
        touching the force statistics: ``f_pulse ~ n_zones * v_indent /
        crush_depth``.  Used for the slow-varying negative control
        (``v_indent -> 0.014 m/s`` gives ``f_pulse ~ 0.05 Hz``, the m5528
        "slow load" check).
        """
        return replace(self, v_indent=float(v_indent))

    def pulse_rate_hz(self) -> float:
        return self.n_zones * self.v_indent / self.crush_depth


#: Ice-contact parameter set that reproduces the m5528 (IN32-N07) calibration
#: optimum; see ``artifacts/m7361/results.json:provenance``.
CALIBRATED_M5528 = IceContactParams()


def scaled_drift_rate(pulse_rate_hz: float,
                      params: IceContactParams = CALIBRATED_M5528) -> IceContactParams:
    """Return the parameter set whose pulse rate equals ``pulse_rate_hz``."""
    if pulse_rate_hz <= 0.0:
        raise ValueError("pulse_rate_hz must be positive")
    v = pulse_rate_hz * params.crush_depth / params.n_zones
    return params.with_v_indent(v)


def generate_ice_force(
    params: IceContactParams = CALIBRATED_M5528,
    duration: float = 300.0,
    dt: float = 2.0e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate the total ice force history ``F(t)`` [N].

    Deterministic given ``params.seed``: the returned series does not depend on
    how it is sliced, which is what makes the end-to-end m5528 reproduction
    (criterion e) reproducible.
    """
    if duration <= 0.0 or dt <= 0.0:
        raise ValueError("duration and dt must be positive")
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
        t_cur = float(rng.uniform(0.0, 0.5))          # zones randomly phase-shifted
        while t_cur < duration:
            v = params.v_indent * (1.0 + params.modulation * math.sin(2 * math.pi * f_mod * t_cur + phase))
            v = max(v, 0.15 * params.v_indent)
            f_cr = params.sigma_crush * area * math.exp(ln_mu + ln_sig * rng.standard_normal())
            # loading phase: from residual level beta*F_cr up to F_cr
            s0 = (beta * f_cr / params.k_contact) ** (1.0 / expo)
            s1 = (f_cr / params.k_contact) ** (1.0 / expo)
            t_r = max((s1 - s0) / v, dt)
            i0 = int(round(t_cur / dt))
            i1 = min(n, i0 + max(2, int(round(t_r / dt))))
            if i0 < n:
                tt = t[i0:i1] - t[i0]
                fzone[i0:i1] = np.maximum(params.k_contact * (s0 + v * tt) ** expo, 0.0)[: i1 - i0]
                fzone[i0:i1] = np.minimum(fzone[i0:i1], f_cr)
            t_cur = t[i0] + t_r if i0 < n else duration
            # crushing phase: finite-time decay to the residual level
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


def load_statistics(F: np.ndarray, dt: float, f_cut: float) -> Dict[str, float]:
    """Full-scale load statistics used as calibration / audit quantities.

    mean_peak        mean / peak [-]          (continuous crushing ~0.3-0.5)
    rise_median      median rise time [s] between a local trough and the next peak
    qs_energy_frac   variance fraction below ``f_cut`` (quasi-static band)
    pulse_rate       rate [1/s] of local peaks above 0.3 * peak
    """
    F = np.asarray(F, dtype=float)
    n = F.size
    if n < 4:
        raise ValueError("need at least 4 samples")
    peak = float(np.max(F))
    mean = float(np.mean(F))
    out: Dict[str, float] = {
        "mean": mean,
        "peak": peak,
        "mean_peak": mean / peak if peak > 0 else 0.0,
        "std": float(np.std(F)),
    }
    d = np.diff(F)
    sign = np.sign(d)
    sign[sign == 0] = 1.0
    turns = np.where(np.diff(sign) != 0)[0] + 1
    rise = [
        (turns[j + 1] - turns[j]) * dt
        for j in range(len(turns) - 1)
        if F[turns[j + 1]] > F[turns[j]]
    ]
    if rise:
        out["rise_median"] = float(np.median(rise))
        out["rise_p10"] = float(np.percentile(rise, 10))
        out["rise_p90"] = float(np.percentile(rise, 90))
        out["rise_count"] = float(len(rise))
    else:
        out["rise_median"] = 0.0
        out["rise_p10"] = 0.0
        out["rise_p90"] = 0.0
        out["rise_count"] = 0.0
    if n > 16 and f_cut > 0:
        fs = 1.0 / dt
        wn = min(f_cut / (0.5 * fs), 0.99)
        b, a = signal.butter(4, wn, btype="low")
        low = signal.filtfilt(b, a, F - mean)
        tot_var = float(np.var(F))
        out["qs_energy_frac"] = float(np.var(low) / tot_var) if tot_var > 0 else 0.0
    else:
        out["qs_energy_frac"] = 0.0
    if peak > 0:
        pk_idx, _ = signal.find_peaks(F, height=0.3 * peak, distance=max(1, int(0.05 / dt)))
        out["pulse_rate"] = float(len(pk_idx) / (n * dt))
    else:
        out["pulse_rate"] = 0.0
    return out