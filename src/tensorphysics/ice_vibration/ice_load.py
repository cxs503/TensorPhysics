"""Ice load models used by the ice-induced vibration framework.

Three load representations cover the physics that drives frequency lock-in:

``harmonic``
    Regular crushing at a single frequency ``f_ice``.  This is the classical
    forced-vibration idealisation and is what the magnification factor
    ``D = 1/sqrt((1-r^2)^2 + (2 zeta r)^2)`` is defined against.
``quasi_periodic pulse train``
    A sequence of half-sine (or saw-tooth) crushing/impact pulses at a mean
    spacing set by the drift speed and the characteristic flaking length,
    ``f_ice = U / s``, with optional inter-arrival jitter.  Real first-year ice
    crushing is intermittent: this representation keeps the spectral peak at
    ``f_ice`` while adding realistic broadband content.
``poisson pulse train``
    Inter-arrival times exponentially distributed (memoryless impacts, e.g.
    broken-ice/flow impacts).  Broad-band; used as the "no lock-in" control.
``velocity_weakening ice law``
    ``F_ice = F_max / (1 + (v_rel/v_c)^p)`` with ``v_rel = U - x'``.  The
    negative slope ``dF/dv < 0`` gives *negative effective damping*
    ``c_eff = c + dF/dv`` at the structure, which is the Maeaettaenen-type
    mechanism by which ice-induced vibration self-locks onto the structural
    natural frequency regardless of drift speed.

Units: SI (m, s, N).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "IceLoadSeries",
    "half_sine_pulse",
    "sawtooth_pulse",
    "harmonic_load",
    "quasi_periodic_pulse_train",
    "poisson_pulse_train",
    "crushing_frequency",
    "velocity_weakening_ice_force",
    "velocity_weakening_slope",
    "self_excitation_velocity_window",
]


def _empty_array() -> np.ndarray:
    return np.zeros(0, dtype=float)


@dataclass(frozen=True)
class IceLoadSeries:
    """A sampled ice-force time history."""

    time_s: np.ndarray
    force_n: np.ndarray
    dt_s: float
    label: str = "ice-load"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        t = np.asarray(self.time_s, dtype=float)
        f = np.asarray(self.force_n, dtype=float)
        if t.ndim != 1 or f.ndim != 1:
            raise ValueError("time and force must be one-dimensional")
        if t.shape != f.shape:
            raise ValueError(f"time ({t.shape}) and force ({f.shape}) must have equal shape")
        if t.size < 2:
            raise ValueError("an ice load series needs at least two samples")
        if not np.all(np.isfinite(t)) or not np.all(np.isfinite(f)):
            raise ValueError("ice load series contains non-finite values")
        if np.any(np.diff(t) <= 0.0):
            raise ValueError("time must be strictly increasing")
        if not (self.dt_s > 0.0 and math.isfinite(self.dt_s)):
            raise ValueError("dt_s must be positive and finite")
        object.__setattr__(self, "time_s", t)
        object.__setattr__(self, "force_n", f)
        object.__setattr__(self, "metadata", dict(self.metadata))

    # -- basic statistics ---------------------------------------------------
    @property
    def duration_s(self) -> float:
        return float(self.time_s[-1] - self.time_s[0])

    @property
    def n_samples(self) -> int:
        return int(self.time_s.size)

    @property
    def peak_force_n(self) -> float:
        return float(np.max(np.abs(self.force_n)))

    @property
    def mean_force_n(self) -> float:
        return float(np.mean(self.force_n))

    @property
    def rms_force_n(self) -> float:
        return float(np.sqrt(np.mean(self.force_n**2)))

    def impulse_ns(self) -> float:
        """Integrated load (impulse) over the record [N*s]."""
        return float(np.trapezoid(self.force_n, self.time_s))

    # -- spectral tools -----------------------------------------------------
    def harmonic_amplitude(self, frequency_hz: float) -> float:
        """Least-squares amplitude of the spectral component at ``frequency_hz``.

        Uses the projection ``a = 2/T int F sin(wt) dt``, ``b = 2/T int F cos(wt) dt``
        and returns ``sqrt(a^2 + b^2)``.  Unlike a Fourier-bin value this is
        exact for an arbitrary (non-bin-centred) frequency, which matters when
        comparing a finite record against an analytic magnification factor.
        """
        if frequency_hz <= 0.0:
            raise ValueError("frequency_hz must be positive")
        w = 2.0 * math.pi * frequency_hz
        t = self.time_s
        f = self.force_n - self.mean_force_n
        span = t[-1] - t[0]
        a = 2.0 / span * float(np.trapezoid(f * np.sin(w * t), t))
        b = 2.0 / span * float(np.trapezoid(f * np.cos(w * t), t))
        return float(math.hypot(a, b))

    def dominant_frequency(self, f_min_hz: float = 0.0, f_max_hz: Optional[float] = None) -> Dict[str, float]:
        """Spectral peak frequency of the load, with parabolic refinement.

        The record is mean-removed, Hann-tapered and zero-padded; the magnitude
        peak inside ``[f_min_hz, f_max_hz]`` is refined by a parabolic fit on
        the log magnitude, giving sub-bin accuracy (needed for the 5 %
        frequency claims in the acceptance criteria).
        """
        t = self.time_s
        n = t.size
        dt = float(t[1] - t[0])
        x = self.force_n - self.mean_force_n
        window = np.hanning(n)
        n_fft = int(2 ** math.ceil(math.log2(max(4 * n, 8))))
        spectrum = np.abs(np.fft.rfft(x * window, n_fft))
        freqs = np.fft.rfftfreq(n_fft, d=dt)
        if f_max_hz is None:
            f_max_hz = 0.5 / dt
        mask = (freqs >= f_min_hz) & (freqs <= f_max_hz)
        if not np.any(mask):
            raise ValueError("no spectral bins inside the requested frequency window")
        idx_local = int(np.argmax(spectrum[mask]))
        idx = int(np.flatnonzero(mask)[idx_local])
        peak_freq = float(freqs[idx])
        if 0 < idx < len(spectrum) - 1:
            y0, y1, y2 = np.log(np.maximum(spectrum[idx - 1 : idx + 2], 1e-300))
            denom = y0 - 2.0 * y1 + y2
            if abs(denom) > 1e-12:
                delta = 0.5 * (y0 - y2) / denom
                delta = float(np.clip(delta, -0.5, 0.5))
                peak_freq = float(freqs[idx] + delta * (freqs[1] - freqs[0]))
        return {
            "frequency_hz": peak_freq,
            "bin_frequency_hz": float(freqs[idx]),
            "magnitude": float(spectrum[idx]),
            "bin_width_hz": float(freqs[1] - freqs[0]),
            "record_duration_s": self.duration_s,
            "rayleigh_resolution_hz": 1.0 / self.duration_s,
        }

    def spectral_concentration(
        self,
        f_min_hz: float = 0.0,
        f_max_hz: Optional[float] = None,
        relative_half_width: float = 0.05,
    ) -> Dict[str, float]:
        """Fraction of the record variance inside a narrow band around the peak.

        Discriminates *narrowband, coherent* excitation (intermittent crushing
        at ``f_ice``) from *broadband* excitation (memoryless impacts).  A
        narrow-band structure always rings at ``f_n``, so the response spectrum
        alone cannot tell lock-in from random impacts; the coherence of the
        *load* can.  ``concentration`` close to 1 means the load is essentially
        a single spectral line.
        """
        t = self.time_s
        n = t.size
        dt = float(t[1] - t[0])
        x = self.force_n - self.mean_force_n
        spectrum = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2
        freqs = np.fft.rfftfreq(n, d=dt)
        if f_max_hz is None:
            f_max_hz = 0.5 / dt
        mask = (freqs >= f_min_hz) & (freqs <= f_max_hz)
        if not np.any(mask):
            raise ValueError("no spectral bins inside the requested frequency window")
        peak_index = int(np.argmax(spectrum[mask]))
        peak_freq = float(freqs[np.flatnonzero(mask)[peak_index]])
        total = float(np.sum(spectrum))
        band = (freqs >= peak_freq * (1.0 - relative_half_width)) & (
            freqs <= peak_freq * (1.0 + relative_half_width)
        )
        band_power = float(np.sum(spectrum[band]))
        return {
            "peak_frequency_hz": peak_freq,
            "concentration": band_power / total if total > 0.0 else 0.0,
            "band_power": band_power,
            "total_power": total,
            "relative_half_width": relative_half_width,
        }

    def windowed(self, t_start_s: float, t_end_s: float) -> "IceLoadSeries":
        """Sub-record used to drop transients before spectral analysis."""
        if t_end_s <= t_start_s:
            raise ValueError("t_end_s must exceed t_start_s")
        mask = (self.time_s >= t_start_s) & (self.time_s <= t_end_s)
        if int(np.count_nonzero(mask)) < 2:
            raise ValueError("requested window contains fewer than two samples")
        return IceLoadSeries(
            time_s=self.time_s[mask],
            force_n=self.force_n[mask],
            dt_s=self.dt_s,
            label=f"{self.label}[{t_start_s:g},{t_end_s:g}]",
            metadata={**self.metadata, "window_start_s": t_start_s, "window_end_s": t_end_s},
        )

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "class": "IceLoadSeries",
            "label": self.label,
            "dt_s": self.dt_s,
            "n_samples": self.n_samples,
            "duration_s": self.duration_s,
            "peak_force_n": self.peak_force_n,
            "mean_force_n": self.mean_force_n,
            "rms_force_n": self.rms_force_n,
            "impulse_ns": self.impulse_ns(),
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Pulse shapes
# ---------------------------------------------------------------------------


def half_sine_pulse(t: np.ndarray, amplitude_n: float, duration_s: float) -> np.ndarray:
    """Half-sine pulse ``A sin(pi t / T)`` on ``[0, T]``, zero elsewhere."""
    t = np.asarray(t, dtype=float)
    if duration_s <= 0.0 or amplitude_n <= 0.0:
        raise ValueError("pulse amplitude and duration must be positive")
    out = np.zeros_like(t)
    mask = (t >= 0.0) & (t <= duration_s)
    out[mask] = amplitude_n * np.sin(math.pi * t[mask] / duration_s)
    return out


def sawtooth_pulse(t: np.ndarray, amplitude_n: float, duration_s: float, rise_fraction: float = 0.8) -> np.ndarray:
    """Saw-tooth crushing pulse: slow rise, fast drop (classic ductile crushing)."""
    t = np.asarray(t, dtype=float)
    if not 0.0 < rise_fraction < 1.0:
        raise ValueError("rise_fraction must be in (0, 1)")
    out = np.zeros_like(t)
    rise = rise_fraction * duration_s
    rise_mask = (t >= 0.0) & (t < rise)
    fall_mask = (t >= rise) & (t <= duration_s)
    out[rise_mask] = amplitude_n * t[rise_mask] / rise
    out[fall_mask] = amplitude_n * (1.0 - (t[fall_mask] - rise) / (duration_s - rise))
    return out


# ---------------------------------------------------------------------------
# Load generators
# ---------------------------------------------------------------------------


def harmonic_load(
    frequency_hz: float,
    amplitude_n: float,
    duration_s: float,
    dt_s: float,
    phase_rad: float = 0.0,
) -> IceLoadSeries:
    """Single-frequency crushing load ``F(t) = A sin(2 pi f t + phase)``."""
    if frequency_hz <= 0.0 or amplitude_n <= 0.0:
        raise ValueError("frequency and amplitude must be positive")
    n = int(round(duration_s / dt_s)) + 1
    t = np.arange(n) * dt_s
    f = amplitude_n * np.sin(2.0 * math.pi * frequency_hz * t + phase_rad)
    return IceLoadSeries(
        time_s=t,
        force_n=f,
        dt_s=dt_s,
        label=f"harmonic-{frequency_hz:g}Hz",
        metadata={
            "model": "harmonic",
            "frequency_hz": frequency_hz,
            "amplitude_n": amplitude_n,
            "phase_rad": phase_rad,
        },
    )


def quasi_periodic_pulse_train(
    pulse_frequency_hz: float,
    amplitude_n: float,
    duration_s: float,
    dt_s: float,
    pulse_duration_s: Optional[float] = None,
    shape: str = "half_sine",
    arrival_jitter: float = 0.05,
    amplitude_cv: float = 0.10,
    seed: int = 20260331,
) -> IceLoadSeries:
    """Intermittent crushing at a mean rate ``f_ice`` with realistic scatter.

    Arrival times follow a renewal process with mean period ``1/f_ice`` and
    jitter ``arrival_jitter`` (relative to the period, uniform); pulse
    amplitudes are log-normally distributed with coefficient of variation
    ``amplitude_cv``.  The spectrum therefore peaks at ``f_ice`` while carrying
    the broadband content of real intermittent ice crushing.
    """
    if pulse_frequency_hz <= 0.0 or amplitude_n <= 0.0:
        raise ValueError("pulse frequency and amplitude must be positive")
    if not 0.0 <= arrival_jitter < 1.0:
        raise ValueError("arrival_jitter must be in [0, 1)")
    if amplitude_cv < 0.0:
        raise ValueError("amplitude_cv must be non-negative")
    if pulse_duration_s is None:
        pulse_duration_s = 0.55 / pulse_frequency_hz
    if pulse_duration_s <= 0.0:
        raise ValueError("pulse_duration_s must be positive")
    rng = np.random.default_rng(seed)
    n = int(round(duration_s / dt_s)) + 1
    t = np.arange(n) * dt_s
    force = np.zeros(n)
    period = 1.0 / pulse_frequency_hz
    sigma = math.sqrt(math.log(1.0 + amplitude_cv**2)) if amplitude_cv > 0.0 else 0.0
    t0 = 0.0
    arrivals = []
    while t0 <= duration_s:
        arrivals.append(t0)
        step = period * (1.0 + arrival_jitter * (2.0 * rng.random() - 1.0))
        t0 += max(step, 0.05 * period)
    for arrival in arrivals:
        scale = math.exp(rng.normal(0.0, sigma) - 0.5 * sigma**2) if sigma > 0.0 else 1.0
        local = t - arrival
        if shape == "half_sine":
            force += half_sine_pulse(local, amplitude_n * scale, pulse_duration_s)
        elif shape == "sawtooth":
            force += sawtooth_pulse(local, amplitude_n * scale, pulse_duration_s)
        else:
            raise ValueError(f"unknown pulse shape {shape!r}")
    return IceLoadSeries(
        time_s=t,
        force_n=force,
        dt_s=dt_s,
        label=f"quasi-periodic-{pulse_frequency_hz:g}Hz",
        metadata={
            "model": "quasi_periodic_pulse_train",
            "pulse_frequency_hz": pulse_frequency_hz,
            "amplitude_n": amplitude_n,
            "pulse_duration_s": pulse_duration_s,
            "shape": shape,
            "arrival_jitter": arrival_jitter,
            "amplitude_cv": amplitude_cv,
            "n_pulses": len(arrivals),
            "seed": seed,
        },
    )


def poisson_pulse_train(
    mean_rate_hz: float,
    amplitude_n: float,
    duration_s: float,
    dt_s: float,
    pulse_duration_s: float,
    shape: str = "half_sine",
    amplitude_cv: float = 0.35,
    seed: int = 20260331,
) -> IceLoadSeries:
    """Memoryless impact train (exponentially distributed inter-arrival times).

    Used as the un-correlated control case: no preferred frequency, so no
    lock-in should be detected.
    """
    if mean_rate_hz <= 0.0 or amplitude_n <= 0.0 or pulse_duration_s <= 0.0:
        raise ValueError("rate, amplitude and pulse duration must be positive")
    rng = np.random.default_rng(seed)
    n = int(round(duration_s / dt_s)) + 1
    t = np.arange(n) * dt_s
    force = np.zeros(n)
    sigma = math.sqrt(math.log(1.0 + amplitude_cv**2)) if amplitude_cv > 0.0 else 0.0
    arrival = rng.exponential(1.0 / mean_rate_hz)
    count = 0
    while arrival <= duration_s:
        scale = math.exp(rng.normal(0.0, sigma) - 0.5 * sigma**2) if sigma > 0.0 else 1.0
        local = t - arrival
        if shape == "half_sine":
            force += half_sine_pulse(local, amplitude_n * scale, pulse_duration_s)
        elif shape == "sawtooth":
            force += sawtooth_pulse(local, amplitude_n * scale, pulse_duration_s)
        else:
            raise ValueError(f"unknown pulse shape {shape!r}")
        count += 1
        arrival += rng.exponential(1.0 / mean_rate_hz)
    return IceLoadSeries(
        time_s=t,
        force_n=force,
        dt_s=dt_s,
        label=f"poisson-{mean_rate_hz:g}Hz",
        metadata={
            "model": "poisson_pulse_train",
            "mean_rate_hz": mean_rate_hz,
            "amplitude_n": amplitude_n,
            "pulse_duration_s": pulse_duration_s,
            "shape": shape,
            "amplitude_cv": amplitude_cv,
            "n_pulses": count,
            "seed": seed,
        },
    )


def crushing_frequency(drift_speed_m_s: float, flaking_length_m: float) -> float:
    """Ice crushing / flaking frequency ``f_ice = U / s`` [Hz].

    ``U`` is the ice drift (or ship) speed and ``s`` the characteristic length
    crushed between two load events (flaking length).  This is the classical
    relation between speed and ice-induced excitation frequency: a ship that
    changes speed sweeps ``f_ice`` across the structural natural frequency,
    which is exactly how lock-in is entered and left.
    """
    if drift_speed_m_s <= 0.0 or flaking_length_m <= 0.0:
        raise ValueError("drift speed and flaking length must be positive")
    return drift_speed_m_s / flaking_length_m


# ---------------------------------------------------------------------------
# Velocity-weakening ice law (self-excited lock-in mechanism)
# ---------------------------------------------------------------------------


def velocity_weakening_ice_force(
    relative_velocity_m_s: float,
    max_force_n: float,
    critical_velocity_m_s: float,
    exponent: float = 2.0,
) -> float:
    """Ice force with velocity weakening: ``F = F_max / (1 + (v_rel/v_c)^p)``.

    ``v_rel = U - x'`` for a structure moving with velocity ``x'`` against ice
    drifting at ``U``.  For ``v_rel > 0`` the force decreases with increasing
    indentation rate (ductile-to-brittle transition / pressure-area softening),
    which is the ingredient that produces negative damping.
    ``v_rel <= 0`` (structure outrunning the ice) gives the full ``F_max``.
    """
    if max_force_n <= 0.0 or critical_velocity_m_s <= 0.0 or exponent <= 0.0:
        raise ValueError("F_max, v_c and exponent must be positive")
    v = max(float(relative_velocity_m_s), 0.0)
    return float(max_force_n / (1.0 + (v / critical_velocity_m_s) ** exponent))


def velocity_weakening_slope(
    velocity_m_s: float,
    max_force_n: float,
    critical_velocity_m_s: float,
    exponent: float = 2.0,
) -> float:
    """``dF/dv`` of the velocity-weakening law [N*s/m] (negative where weakening)."""
    if max_force_n <= 0.0 or critical_velocity_m_s <= 0.0 or exponent <= 0.0:
        raise ValueError("F_max, v_c and exponent must be positive")
    v = max(float(velocity_m_s), 0.0)
    if v == 0.0:
        return -0.0
    u = v / critical_velocity_m_s
    return float(
        -max_force_n * exponent * u ** (exponent - 1.0) / (critical_velocity_m_s * (1.0 + u**exponent) ** 2)
    )


def self_excitation_velocity_window(
    structural_damping_n_s_m: float,
    max_force_n: float,
    critical_velocity_m_s: float,
    exponent: float = 2.0,
) -> Dict[str, float]:
    """Velocity band in which the ice law overcomes structural damping.

    Self-excited vibration requires ``c_struct + dF/dv(U) < 0``.  The most
    negative slope of the ice law occurs at ``U = v_c * ((p-1)/(p+1))^(1/p)``
    (``U = v_c/sqrt(3)`` for ``p = 2``), where

        |dF/dv|_max = F_max * p * q^((p-1)/p) / (v_c (1+q)^2),  q = (p-1)/(p+1).

    Returned values are informative bounds for the sweep; the actual
    time-domain integration decides whether a limit cycle grows.
    """
    if structural_damping_n_s_m < 0.0:
        raise ValueError("structural damping must be non-negative")
    if max_force_n <= 0.0 or critical_velocity_m_s <= 0.0 or exponent <= 1.0:
        raise ValueError("F_max > 0, v_c > 0 and exponent > 1 are required")
    q = (exponent - 1.0) / (exponent + 1.0)
    u_star = q ** (1.0 / exponent)
    v_star = critical_velocity_m_s * u_star
    slope_max = abs(
        velocity_weakening_slope(v_star, max_force_n, critical_velocity_m_s, exponent)
    )
    return {
        "velocity_at_max_negative_slope_m_s": float(v_star),
        "max_negative_slope_n_s_m": float(-slope_max),
        "structural_damping_n_s_m": float(structural_damping_n_s_m),
        "self_excitation_possible": bool(slope_max > structural_damping_n_s_m),
        "c_eff_min_n_s_m": float(structural_damping_n_s_m - slope_max),
        "critical_velocity_m_s": float(critical_velocity_m_s),
        "exponent": float(exponent),
    }