"""Broken-ice (brash / floe) contact field that becomes a *moving* load on the hull girder.

Physical picture
----------------
The ship advances at speed ``V`` through a broken-ice channel (brash ice, floe
fragments, small floes).  In the ship-fixed frame used here:

* the hull slides past the ice, so a floe that touches the stem at ``t_k``
  sweeps aft along the hull at speed ``V`` and separates after travelling its
  own contact extent ``s_k`` (the floe length along the track);
* floes arrive one after another, so the *time between contacts* is
  ``T_k = s_k / V`` and the associated *repetition frequency* is

      f_ice = V / s_mean          [Hz]

  which is the same relation the crushing-frequency modules of
  :mod:`tensorphysics.ice_vibration` use;
* during a contact the force is a brittle-ductile sawtooth between the peak
  ``F_max = sigma_c * h * w * beta`` (``h`` ice thickness, ``w`` contact width,
  ``sigma_c`` effective crushing pressure, ``beta`` a lognormal floe-strength
  factor) and a residual level after each crushing/spalling drop.

Resolution of the moving load
-----------------------------
The contact force is therefore *not* a point load at a fixed station: it is a
train of pulses whose application point travels aft along the hull.  Both the
pulse repetition (time modulation) and the sweep (spatial modulation) are
retained, which is what makes the load able to select hull modes.

Limits: the model is one-way coupled (ice force prescribed; the structural
velocity does not feed back on the crushing process), so it does not reproduce
velocity-weakening lock-in - that mechanism lives in
:mod:`tensorphysics.ice_vibration.lockin`.  Adding the feedback here would be
the natural next step.

Units: SI (m, N, s, Hz).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

__all__ = [
    "IceFloeField",
    "ContactForceHistory",
    "build_contact_history",
    "dominant_frequency_hz",
    "sawtooth_fraction",
]


@dataclass(frozen=True)
class IceFloeField:
    """Statistical description of the broken-ice field along the track.

    Parameters
    ----------
    ice_thickness_m:
        Level-ice / brash-ice thickness ``h``.
    crushing_pressure_pa:
        *Effective* crushing pressure ``sigma_c`` acting on the contact patch
        (IACS/ISO 19906 local pressure range at small areas is 1 - 3 MPa; brash
        ice is softer than intact level ice).
    contact_width_m:
        Width of the hull-ice contact band ``w`` (transverse).
    mean_floe_length_m:
        Mean floe extent along the track ``s_mean``, which fixes the contact
        repetition frequency ``f_ice = V / s_mean``.
    floe_length_cv / strength_cv:
        Coefficients of variation of the lognormal floe length and floe
        strength distributions.
    sawtooth_cycles:
        Number of brittle-ductile (crushing / spalling) cycles inside one floe
        contact.
    residual_fraction:
        Force level immediately after a spalling drop, as a fraction of the
        current crushing peak.
    seed:
        RNG seed for the realisation (reproducibility).
    """

    ice_thickness_m: float = 0.8
    crushing_pressure_pa: float = 2.0e6
    contact_width_m: float = 2.0
    mean_floe_length_m: float = 4.13
    floe_length_cv: float = 0.35
    strength_cv: float = 0.20
    sawtooth_cycles: int = 4
    residual_fraction: float = 0.55
    seed: int = 20261110

    def __post_init__(self) -> None:
        for name in (
            "ice_thickness_m",
            "crushing_pressure_pa",
            "contact_width_m",
            "mean_floe_length_m",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"IceFloeField.{name} must be positive and finite, got {value!r}")
        if self.sawtooth_cycles < 1:
            raise ValueError("sawtooth_cycles must be >= 1")
        if not 0.0 <= self.residual_fraction < 1.0:
            raise ValueError("residual_fraction must be in [0, 1)")
        if self.floe_length_cv < 0.0 or self.strength_cv < 0.0:
            raise ValueError("coefficients of variation must be non-negative")

    @property
    def peak_force_n(self) -> float:
        """Deterministic peak contact force ``sigma_c h w`` [N]."""
        return self.crushing_pressure_pa * self.ice_thickness_m * self.contact_width_m

    @property
    def nominal_contact_area_m2(self) -> float:
        return self.ice_thickness_m * self.contact_width_m

    def repetition_frequency_hz(self, speed_m_s: float) -> float:
        """``f_ice = V / s_mean`` - the frequency the pulse train presents [Hz]."""
        if speed_m_s <= 0.0:
            raise ValueError("speed must be positive")
        return speed_m_s / self.mean_floe_length_m

    def critical_speed_m_s(self, natural_frequency_hz: float) -> float:
        """Speed at which ``f_ice`` matches ``natural_frequency_hz`` [m/s].

        ``V_cr = s_mean * f_n``.  This is the *operational* moving-load
        resonance of a ship in broken ice, in contrast with the classical
        single-force critical speed ``omega_n L / (n pi)`` of a simply supported
        beam (of the order of hundreds of m/s at hull-girder scale).
        """
        if natural_frequency_hz <= 0.0:
            raise ValueError("natural frequency must be positive")
        return self.mean_floe_length_m * natural_frequency_hz

    # -- realisation --------------------------------------------------------
    def contact_train(self, speed_m_s: float, duration_s: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Arrival times, floe lengths and peak forces of the contact events.

        Returns ``(t_start, length, peak_force)`` arrays covering
        ``0 <= t_start <= duration_s``.
        """
        if speed_m_s <= 0.0 or duration_s <= 0.0:
            raise ValueError("speed and duration must be positive")
        rng = np.random.default_rng(self.seed)
        mu = math.log(self.mean_floe_length_m) - 0.5 * math.log(1.0 + self.floe_length_cv**2)
        sigma = math.sqrt(math.log(1.0 + self.floe_length_cv**2)) if self.floe_length_cv > 0 else 0.0
        s_mu = math.log(1.0) - 0.5 * math.log(1.0 + self.strength_cv**2)
        s_sigma = math.sqrt(math.log(1.0 + self.strength_cv**2)) if self.strength_cv > 0 else 0.0
        times, lengths, peaks = [], [], []
        t = 0.0
        while t <= duration_s:
            length = float(rng.lognormal(mu, sigma)) if sigma > 0 else self.mean_floe_length_m
            beta = float(rng.lognormal(s_mu, s_sigma)) if s_sigma > 0 else 1.0
            times.append(t)
            lengths.append(length)
            peaks.append(self.peak_force_n * beta)
            t += length / speed_m_s
        return np.array(times), np.array(lengths), np.array(peaks)


@dataclass(frozen=True)
class ContactForceHistory:
    """Sampled moving contact load.

    ``position_m`` is the along-hull coordinate of the *most recently activated*
    contact (0 = stern, ``L`` = stem).  ``force_density`` is the per-floe force so
    that the total force acting on the beam at time ``t`` is
    ``force_n = sum(force_density)``; the individual patches are exposed through
    :attr:`active_patches` for the modal force assembly.
    """

    time_s: np.ndarray
    force_n: np.ndarray
    position_m: np.ndarray
    patches_t0: np.ndarray
    patches_length: np.ndarray
    patches_peak: np.ndarray
    speed_m_s: float
    length_m: float
    sweep: bool

    @property
    def peak_force_n(self) -> float:
        return float(np.max(self.force_n))

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "class": "ContactForceHistory",
            "speed_m_s": self.speed_m_s,
            "sweep": self.sweep,
            "duration_s": float(self.time_s[-1]),
            "n_events": int(self.patches_t0.size),
            "peak_force_n": self.peak_force_n,
            "mean_force_n": float(np.mean(self.force_n)),
            "dominant_frequency_hz": dominant_frequency_hz(self.time_s, self.force_n),
        }


def sawtooth_fraction(u: np.ndarray, cycles: int, residual: float) -> np.ndarray:
    """Brittle-ductile sawtooth on the contact-interval phase ``u`` in [0, 1].

    The force ramps linearly from ``residual`` to 1.0 within each of the
    ``cycles`` sub-intervals (ductile crushing), then drops back to ``residual``
    (brittle spalling).  Mean value over a cycle is ``(1 + residual)/2``.
    """
    u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
    phase = (u * cycles) % 1.0
    return residual + (1.0 - residual) * phase


def build_contact_history(
    field: IceFloeField,
    speed_m_s: float,
    duration_s: float,
    dt_s: float,
    length_m: float,
    sweep: bool = True,
) -> ContactForceHistory:
    """Sample the moving contact force history.

    Parameters
    ----------
    sweep:
        ``True`` (default) moves each contact patch aft at ``speed_m_s`` so the
        load travels along the hull; ``False`` pins the patches at the stem,
        which reproduces a stationary pulse train used to verify the dynamics
        against a frequency-domain solution.
    """
    if dt_s <= 0.0:
        raise ValueError("dt must be positive")
    t0, lengths, peaks = field.contact_train(speed_m_s, duration_s)
    n = int(round(duration_s / dt_s)) + 1
    t = np.linspace(0.0, duration_s, n)
    force = np.zeros(n)
    position = np.zeros(n)
    for start, length, peak in zip(t0, lengths, peaks):
        span = length / speed_m_s
        mask = (t >= start) & (t <= start + span)
        if not np.any(mask):
            continue
        u = (t[mask] - start) / span
        force[mask] = peak * sawtooth_fraction(u, field.sawtooth_cycles, field.residual_fraction)
        if sweep:
            position[mask] = np.clip(length_m - speed_m_s * (t[mask] - start), 0.0, length_m)
        else:
            position[mask] = length_m
    return ContactForceHistory(
        time_s=t,
        force_n=force,
        position_m=position,
        patches_t0=t0,
        patches_length=lengths,
        patches_peak=peaks,
        speed_m_s=speed_m_s,
        length_m=length_m,
        sweep=sweep,
    )


def dominant_frequency_hz(time_s: np.ndarray, signal: np.ndarray, min_hz: float = 0.05) -> float:
    """Dominant frequency of a sampled signal by Hann-windowed FFT + parabolic peak fit.

    Returns the frequency of the largest spectral peak above ``min_hz``.  The
    DC / drift content is removed first (mean subtraction) so that a transient
    transient offset cannot masquerade as the dominant frequency.
    """
    t = np.asarray(time_s, dtype=float)
    y = np.asarray(signal, dtype=float)
    if t.size < 8 or t.size != y.size:
        raise ValueError("time and signal must be equally sized with at least 8 samples")
    dt = float(np.mean(np.diff(t)))
    y = y - np.mean(y)
    window = np.hanning(y.size)
    spec = np.abs(np.fft.rfft(y * window))
    freqs = np.fft.rfftfreq(y.size, dt)
    valid = freqs >= min_hz
    if not np.any(valid):
        raise ValueError("no spectral bin above min_hz")
    idx = int(np.argmax(np.where(valid, spec, -np.inf)))
    f_peak = float(freqs[idx])
    # parabolic interpolation on the log-magnitude peak
    if 0 < idx < spec.size - 1:
        a, b, c = np.log(np.maximum(spec[idx - 1 : idx + 2], 1e-300))
        denom = a - 2.0 * b + c
        if denom != 0.0:
            delta = 0.5 * (a - c) / denom
            if abs(delta) < 1.0:
                f_peak += delta * float(freqs[1] - freqs[0])
    return f_peak