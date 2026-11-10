"""Ice-excitation models feeding the spectral pipeline.

Two excitation representations are provided.

Time domain (the "DEM-style" input of the acceptance criterion)
    :func:`intermittent_crushing_series` builds the load history of a
    brittle-ductile crushing process against a compliant structure: the ice
    indents at the drift velocity ``U``, the load rises elastically at the
    contact stiffness until the crushing penetration ``s`` is reached, the
    contact then fails and the load drops, and the cycle repeats with a
    period ``T = s/U`` and a log-normal period jitter that widens the spectral
    peak.  This is the same load-generation mechanism used by discrete-element
    / crushing-event models at the engineering scale (Karna & Turunen 1990;
    Timco & Weeks 2010 for the ice-pressure scale), reduced to a declared
    one-dimensional parameterisation so the spectrum is reproducible.  A slow
    drift modulation and background noise are added to avoid a
    single-frequency comb.

Frequency domain
    :func:`narrowband_ice_psd` gives the analytic two-parameter narrow-band
    PSD of the crushing process (Gaussian peak at ``f_ice`` with half-power
    bandwidth ``b_ice``), used both as a closed-form anchor for the Welch
    estimate and as a smooth excitation model for parametric studies.

The Welch estimator itself is reused from
:mod:`tensorphysics.ice_vibration_transfer.excitation` (verified in that
module's tests) instead of being re-implemented here.

References
----------
* Karna, T., Turunen, R. (1990) A straightforward technique for analysing
  structural response to dynamic ice action, IAHR Ice Symposium, Espoo.
* Timco, G.W., Weeks, W.F. (2010) A review of the engineering properties of
  sea ice, Cold Regions Science and Technology 60, 107-129.
* Sodhi, D.S. (1995) An ice-structure interaction model, in Mechanics of
  Creep and Brittle Fracture of Ice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..ice_vibration_transfer.excitation import welch_psd

__all__ = [
    "IceCrushingSeries",
    "ICE_PRESSURE_RANGE_MPA",
    "crushing_frequency",
    "intermittent_crushing_series",
    "narrowband_ice_psd",
    "welch_psd",
]

#: Engineering range of the effective global ice pressure on a wide structure
#: (Timco & Weeks 2010, Fig. 8): 0.3 - 2.0 MPa for widths of a few metres.
ICE_PRESSURE_RANGE_MPA = (0.3, 2.0)


def crushing_frequency(drift_velocity_m_s: float, crushing_penetration_m: float) -> float:
    """Intermittent-crushing frequency ``f_ice = U / s`` [Hz].

    ``s`` is the penetration accumulated between two crushing failures
    (typically a fraction of the ice thickness).  This is the frequency at
    which the ice load repeats and which can lock onto a structural mode.
    """
    if crushing_penetration_m <= 0.0:
        raise ValueError("crushing penetration must be positive")
    if drift_velocity_m_s < 0.0:
        raise ValueError("drift velocity must be non-negative")
    return float(drift_velocity_m_s / crushing_penetration_m)


@dataclass
class IceCrushingSeries:
    """Intermittent ice-crushing load history and its one-sided PSD."""

    time_s: np.ndarray
    force_n: np.ndarray
    dt_s: float
    duration_s: float
    drift_velocity_m_s: float
    crushing_penetration_m: float
    contact_width_m: float
    effective_pressure_pa: float
    seed: int
    freqs_hz: np.ndarray
    psd_n2_per_hz: np.ndarray
    f_ice_hz: float
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ views
    @property
    def force_mean_n(self) -> float:
        return float(np.mean(self.force_n))

    @property
    def force_rms_n(self) -> float:
        return float(np.std(self.force_n))

    @property
    def force_peak_n(self) -> float:
        return float(np.max(self.force_n))

    @property
    def psd_peak_frequency_hz(self) -> float:
        return float(self.freqs_hz[int(np.argmax(self.psd_n2_per_hz))])

    def payload(self) -> dict[str, Any]:
        return {
            "dt_s": self.dt_s,
            "duration_s": self.duration_s,
            "n_samples": int(self.time_s.size),
            "drift_velocity_m_s": self.drift_velocity_m_s,
            "crushing_penetration_m": self.crushing_penetration_m,
            "contact_width_m": self.contact_width_m,
            "effective_pressure_mpa": self.effective_pressure_pa / 1.0e6,
            "seed": self.seed,
            "f_ice_hz": self.f_ice_hz,
            "force_mean_n": self.force_mean_n,
            "force_rms_n": self.force_rms_n,
            "force_peak_n": self.force_peak_n,
            "psd_peak_frequency_hz": self.psd_peak_frequency_hz,
            "psd_df_hz": float(self.freqs_hz[1] - self.freqs_hz[0]),
            "metadata": self.metadata,
        }


def intermittent_crushing_series(
    duration_s: float = 120.0,
    dt_s: float = 0.002,
    drift_velocity_m_s: float = 0.16,
    crushing_penetration_m: float = 0.15,
    contact_width_m: float = 2.0,
    effective_pressure_pa: float = 0.30e6,
    period_jitter_cv: float = 0.18,
    amplitude_jitter_cv: float = 0.25,
    rise_fraction: float = 0.65,
    background_fraction: float = 0.12,
    drift_modulation_fraction: float = 0.08,
    drift_modulation_hz: float = 0.11,
    seed: int = 20261110,
    nperseg: int = 4096,
) -> IceCrushingSeries:
    """Synthesise an intermittent ice-crushing load history and its Welch PSD.

    One crushing cycle of nominal period ``T = s/U`` consists of

    * a linear load ramp up to the peak ``F_pk = p_eff * width * s`` over
      ``rise_fraction * T`` (elastic loading of the ice-structure contact),
    * a sudden drop back to the residual (brittle failure of the contact),
    * random jitter of the period (log-normal, ``period_jitter_cv``) and of the
      peak amplitude (``amplitude_jitter_cv``),

    plus a slow amplitude modulation (drift of the ice sheet) and a fraction of
    broadband background noise, both declared parameters.

    Returns
    -------
    IceCrushingSeries
        with ``time_s``, ``force_n`` and the one-sided Welch PSD in N^2/Hz.
    """
    if dt_s <= 0.0 or duration_s <= 0.0:
        raise ValueError("dt_s and duration_s must be positive")
    rng = np.random.default_rng(seed)
    n = int(round(duration_s / dt_s))
    t = np.arange(n) * dt_s

    f_ice = crushing_frequency(drift_velocity_m_s, crushing_penetration_m)
    t_nom = 1.0 / f_ice if f_ice > 0.0 else duration_s
    f_peak = effective_pressure_pa * contact_width_m * crushing_penetration_m
    residual = 0.15 * f_peak

    force = np.zeros(n)
    t_event = 0.0
    while t_event < duration_s:
        period = t_nom * float(np.exp(rng.normal(0.0, period_jitter_cv)))
        peak = f_peak * float(max(0.2, 1.0 + rng.normal(0.0, amplitude_jitter_cv)))
        rise = rise_fraction * period
        i0 = int(round(t_event / dt_s))
        i_rise = int(round(rise / dt_s))
        i_end = min(n, int(round((t_event + period) / dt_s)))
        if i0 >= n:
            break
        i_rise_end = min(i0 + i_rise, i_end)
        if i_rise_end > i0:
            frac = np.linspace(0.0, 1.0, i_rise_end - i0, endpoint=False)
            force[i0:i_rise_end] = residual + (peak - residual) * frac
        if i_end > i_rise_end:
            force[i_rise_end:i_end] = residual * np.ones(i_end - i_rise_end)
        t_event += period

    # slow drift modulation + broadband background of the crushing process
    modulation = 1.0 + drift_modulation_fraction * np.sin(
        2.0 * np.pi * drift_modulation_hz * t + rng.uniform(0.0, 2.0 * np.pi)
    )
    force = force * modulation
    force += background_fraction * f_peak * rng.standard_normal(n)
    force = np.maximum(force, 0.0)

    freqs, psd = welch_psd(force, dt_s, nperseg=min(nperseg, n))
    return IceCrushingSeries(
        time_s=t,
        force_n=force,
        dt_s=dt_s,
        duration_s=duration_s,
        drift_velocity_m_s=drift_velocity_m_s,
        crushing_penetration_m=crushing_penetration_m,
        contact_width_m=contact_width_m,
        effective_pressure_pa=effective_pressure_pa,
        seed=seed,
        freqs_hz=freqs,
        psd_n2_per_hz=psd,
        f_ice_hz=f_ice,
        metadata={
            "model": "intermittent brittle-ductile crushing (1-D parameterisation)",
            "period_jitter_cv": period_jitter_cv,
            "amplitude_jitter_cv": amplitude_jitter_cv,
            "rise_fraction": rise_fraction,
            "background_fraction": background_fraction,
            "drift_modulation_fraction": drift_modulation_fraction,
            "drift_modulation_hz": drift_modulation_hz,
            "nperseg": int(min(nperseg, n)),
            "peak_force_model_n": f_peak,
        },
    )


def narrowband_ice_psd(
    frequency_hz: np.ndarray,
    f_ice_hz: float,
    variance_n2: float,
    half_power_bandwidth_hz: float,
) -> np.ndarray:
    """Analytic narrow-band ice-force PSD, one-sided [N^2/Hz].

    Gaussian-shaped spectral peak

        S_ff(f) = sqrt(4 ln2 / pi) * var / b * exp(-4 ln2 ((f - f_ice)/b)^2),

    normalised so that ``integral_0^inf S_ff df = variance_n2``.  The *full*
    half-power (3 dB) bandwidth of this shape is exactly ``b`` by construction,
    which is what makes it a clean anchor for the Welch estimate of a jittered
    crushing comb (and directly comparable with ``zeta f_n``-type structural
    bandwidths).
    """
    f = np.asarray(frequency_hz, dtype=float)
    if half_power_bandwidth_hz <= 0.0:
        raise ValueError("bandwidth must be positive")
    b = float(half_power_bandwidth_hz)
    shape = np.exp(-4.0 * np.log(2.0) * ((f - f_ice_hz) / b) ** 2)
    return np.sqrt(4.0 * np.log(2.0) / np.pi) * variance_n2 / b * shape