"""Ice milling torque spectrum (N04 upstream) -> hull excitation force spectrum.

Upstream in the campaign DAG, node N04 supplies a **DEM milling torque
spectrum**: the propeller blade edge crushes ice, the shaft torque becomes a
random pulse train whose magnitudes are distributed between the milling level
and the ice-class envelope, and whose pulses are phase locked to blade passage.
Rather than re-inventing that model, this module *ingests* the existing
:mod:`tensorphysics.propeller_fatigue.ice_spectrum` implementation (crushing
area milling torque, log-normal impact magnitude spectrum, blade impact train)
and adds the two things the structural chain needs:

1. a one-sided **torque PSD** ``S_Q(f)`` [N^2 m^2 / Hz] obtained by Welch
   averaging of the synthesised pulse train;
2. the **hull excitation force** ``F(t) = eta_hull * (Q(t) - <Q>) / r_eff``
   where ``eta_hull`` is the declared shaft-line -> hull-girder transmission
   efficiency of the thrust/torque bearing path and ``r_eff`` the ice contact
   radius (the shaft torque is carried as a tangential force pair at that
   radius).  ``F`` is applied in the structural solver as a self-equilibrated
   couple (see :mod:`.hull_modal`).

The torque scale factor ``scale`` allows a controlled load-intensity sweep
(``F -> scale * F``) for the fatigue-life vs ice-load curve; the three ice-class
severities (light / design / heavy) change the ice strength, the advance speed
and the impact rate, i.e. the *shape* of the spectrum, not only its amplitude.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..propeller_fatigue.blade_stress import BladeGeometry
from ..propeller_fatigue.ice_spectrum import (
    ICE_CASES,
    IceEnvelope,
    blade_impact_train,
    ice_milling_torque,
    impact_spectrum,
)

__all__ = [
    "DESIGN_KQ",
    "IceTorqueExcitation",
    "RHO_WATER",
    "milling_torque_excitation",
    "welch_psd",
]

RHO_WATER = 1025.0
#: Open-water torque coefficient of the reference ice-class propeller at the
#: design advance ratio (high-thrust, heavily loaded blade section -- declared
#: parameter, used only to make the ice-class envelope ``k_torque_max * Q_design``
#: dimensional).
DESIGN_KQ = 0.023


def welch_psd(
    signal: np.ndarray,
    dt: float,
    nperseg: int = 4096,
    overlap: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """One-sided Welch PSD (Hann window, 50 % overlap) in units^2/Hz.

    ``S(f) = dt |X(f)|^2 / sum(w^2)`` (the standard Welch normalisation, so that
    ``integral S df`` equals the signal variance), implemented directly on
    ``numpy.fft`` so the node has no SciPy dependency.  The normalisation is
    verified against a time-domain simulation in the test suite.
    """
    x = np.asarray(signal, dtype=float)
    x = x - x.mean()
    n = x.size
    nseg = int(min(nperseg, n))
    if nseg < 16:
        raise ValueError("signal too short for a PSD estimate")
    step = max(1, int(round(nseg * (1.0 - overlap))))
    window = np.hanning(nseg)
    norm = float(np.sum(window**2))
    starts = range(0, n - nseg + 1, step)
    acc = np.zeros(nseg // 2 + 1)
    n_avg = 0
    for s in starts:
        seg = x[s : s + nseg] * window
        spec = np.fft.rfft(seg)
        acc += np.abs(spec) ** 2
        n_avg += 1
    if n_avg == 0:
        raise ValueError("no complete segment available for the PSD estimate")
    psd = acc * dt / (n_avg * norm)
    freqs = np.fft.rfftfreq(nseg, d=dt)
    # one-sided density: double all bins except DC (and Nyquist for even nseg)
    psd = psd.copy()
    if nseg % 2 == 0:
        psd[1:-1] *= 2.0
    else:
        psd[1:] *= 2.0
    psd[0] *= 0.5
    return freqs, psd


@dataclass
class IceTorqueExcitation:
    """Ice milling torque pulse train and its hull-force spectrum."""

    severity: str
    label: str
    scale: float
    n_rps: float
    duration_s: float
    dt_s: float
    seed: int
    eta_hull: float
    r_effective_m: float
    t_s: np.ndarray
    torque_nm: np.ndarray
    force_n: np.ndarray
    freqs_hz: np.ndarray
    s_torque_nm2_per_hz: np.ndarray
    s_force_n2_per_hz: np.ndarray
    q_mean_nm: float
    q_rms_nm: float
    q_p99_nm: float
    q_mill_nm: float
    q_design_nm: float
    q_envelope_nm: float
    n_impacts: int
    force_rms_n: float
    band_hz: tuple[float, float] = (0.0, 0.0)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ views
    @property
    def s_force_n2_per_hz_scaled(self) -> np.ndarray:
        return self.s_force_n2_per_hz * self.scale**2

    def force_spectrum_scaled(self) -> np.ndarray:
        """Force PSD including the load-intensity scale factor."""
        return self.s_force_n2_per_hz_scaled

    def energy_band(self, fraction: float = 0.90) -> tuple[float, float]:
        """Frequency band holding ``fraction`` of the force variance."""
        f = self.freqs_hz
        s = self.force_spectrum_scaled()
        total = float(np.trapezoid(s, f))
        if total <= 0.0:
            return (float(f[0]), float(f[-1]))
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (s[1:] + s[:-1]) * np.diff(f))]) / total
        tail = 0.5 * (1.0 - fraction)
        lo = float(np.interp(tail, cum, f))
        hi = float(np.interp(1.0 - tail, cum, f))
        return (lo, hi)

    def as_dict(self) -> dict[str, Any]:
        lo, hi = self.energy_band(0.90)
        peaks = self.freqs_hz[1:][np.argmax(self.force_spectrum_scaled()[1:])] if self.freqs_hz.size > 2 else 0.0
        return {
            "severity": self.severity,
            "label": self.label,
            "scale": float(self.scale),
            "n_rps": float(self.n_rps),
            "shaft_rpm": float(self.n_rps * 60.0),
            "blade_passing_hz": float(self.n_rps * self.metadata.get("n_blades", 4)),
            "duration_s": float(self.duration_s),
            "dt_s": float(self.dt_s),
            "seed": int(self.seed),
            "eta_hull": float(self.eta_hull),
            "r_effective_m": float(self.r_effective_m),
            "n_impacts": int(self.n_impacts),
            "impacts_per_hour": float(self.metadata.get("impacts_per_hour", float("nan"))),
            "Q_mill_Nm": float(self.q_mill_nm),
            "Q_design_Nm": float(self.q_design_nm),
            "Q_envelope_Nm": float(self.q_envelope_nm),
            "Q_mean_Nm": float(self.q_mean_nm),
            "Q_rms_Nm": float(self.q_rms_nm),
            "Q_p99_Nm": float(self.q_p99_nm),
            "Q_peak_Nm": float(np.max(np.abs(self.torque_nm))),
            "force_rms_N": float(self.force_rms_n),
            "force_psd_peak_hz": float(peaks),
            "force_energy_band_90pct_hz": [lo, hi],
            "metadata": dict(self.metadata),
        }


def milling_torque_excitation(
    severity: str = "design",
    n_rps: float = 2.5,
    duration_s: float = 180.0,
    dt_s: float = 1.0e-3,
    eta_hull: float = 0.30,
    scale: float = 1.0,
    seed: int = 20261110,
    geom: BladeGeometry | None = None,
    nperseg: int = 2048,
) -> IceTorqueExcitation:
    """Synthesise the N04 DEM milling torque train and its hull force spectrum.

    Parameters
    ----------
    severity : {'light', 'design', 'heavy'}
        Ice-class operating condition of :data:`~tensorphysics.propeller_fatigue
        .ice_spectrum.ICE_CASES`.
    n_rps : float
        Shaft revolutions per second at the ice operating point.
    eta_hull : float
        Declared transmission efficiency of the shaft-line -> hull-girder path
        (thrust/torque bearing compliance, engine mounts).  ``0.3`` is a
        deliberately conservative engineering value; it is a *declared*
        parameter and is stored in the payload.
    scale : float
        Load-intensity factor applied to the torque fluctuation ``(Q - <Q>)``.
    """
    if severity not in ICE_CASES:
        raise ValueError(f"unknown ice severity {severity!r}; available: {sorted(ICE_CASES)}")
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    env: IceEnvelope = ICE_CASES[severity]
    geom = BladeGeometry() if geom is None else geom

    mill = ice_milling_torque(geom, env, n_rps)
    q_mill = float(mill["Q_mill_Nm"])
    q_design = float(DESIGN_KQ * RHO_WATER * n_rps**2 * geom.D**5)
    spec = impact_spectrum(env, q_mill, q_design, n_samples=4096, seed=seed % 7919)
    train = blade_impact_train(
        geom,
        env,
        n_rps,
        duration_s,
        spec,
        dt=dt_s,
        seed=seed % 6151,
    )
    torque = np.asarray(train["torque"], dtype=float)
    r_eff = float(train["r_impact_m"])
    q_mean = float(torque.mean())
    force = float(eta_hull) / r_eff * (torque - q_mean) * float(scale)

    f, s_q = welch_psd(torque, dt_s, nperseg=nperseg)
    s_f = (float(eta_hull) / r_eff) ** 2 * s_q
    exc = IceTorqueExcitation(
        severity=severity,
        label=str(env.label),
        scale=float(scale),
        n_rps=float(n_rps),
        duration_s=float(duration_s),
        dt_s=float(dt_s),
        seed=int(seed),
        eta_hull=float(eta_hull),
        r_effective_m=r_eff,
        t_s=np.asarray(train["t"], dtype=float),
        torque_nm=torque,
        force_n=force,
        freqs_hz=f,
        s_torque_nm2_per_hz=s_q,
        s_force_n2_per_hz=s_f,
        q_mean_nm=q_mean,
        q_rms_nm=float(np.sqrt(np.mean((torque - q_mean) ** 2))),
        q_p99_nm=float(np.percentile(torque, 99.0)),
        q_mill_nm=q_mill,
        q_design_nm=q_design,
        q_envelope_nm=float(spec["Q_max_envelope_Nm"]),
        n_impacts=int(train["n_impacts"]),
        force_rms_n=float(np.sqrt(np.mean(force**2))),
        metadata={
            "n_blades": int(geom.Z),
            "propeller_D_m": float(geom.D),
            "impacts_per_hour": float(env.impacts_per_hour),
            "sigma_ice_Pa": float(env.sigma_ice),
            "advance_speed_m_s": float(env.advance_speed),
            "k_torque_max": float(env.k_torque_max),
            "impact_duration_s": float(env.impact_duration),
            "depth_of_cut_m": float(mill["depth_of_cut_m"]),
            "sigma_ln_impact_spectrum": float(spec["sigma_ln"]),
            "k_ice_envelope": float(spec["k_ice_envelope"]),
            "source": "tensorphysics.propeller_fatigue.ice_spectrum (N04 DEM milling torque)",
        },
    )
    exc.band_hz = exc.energy_band(0.90)
    return exc