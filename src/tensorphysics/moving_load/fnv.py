"""Frequency-normalised value (FNV) of a measured acceleration time history.

Definition used here (as specified by the ice-class habitability criterion):

    FNV = a_max / (g * f_dominant),   g = 9.81 m/s^2

with ``a_max = max_t |u''(t)|`` at the assessment station and ``f_dominant``
the frequency of the dominant spectral peak of the *same* acceleration signal
(FFT, Hann window, zero padding, parabolic peak refinement).

Classification thresholds:
    ``FNV < 0.5``                  habitable
    ``0.5 <= FNV <= 1.0``          tolerable
    ``FNV > 1.0``                  unacceptable

The metric is a single-number proxy for human comfort on board: it penalises
large accelerations and, for a given amplitude, low frequencies.  It is *not* a
structural strength criterion and says nothing about fatigue.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = [
    "GRAVITY_M_S2",
    "FNV_HABITABLE_MAX",
    "FNV_TOLERABLE_MAX",
    "FnvAssessment",
    "dominant_frequency",
    "assess_fnv",
    "classify_fnv",
]

#: Standard gravity (m/s^2).
GRAVITY_M_S2 = 9.81
#: FNV at or below which the environment is classified habitable.
FNV_HABITABLE_MAX = 0.5
#: FNV at or below which the environment is classified tolerable.
FNV_TOLERABLE_MAX = 1.0


@dataclass(frozen=True)
class FnvAssessment:
    """Result of an FNV evaluation."""

    peak_acceleration_m_s2: float
    dominant_frequency_hz: float
    fnv: float
    level: str
    n_samples: int
    sample_rate_hz: float

    def as_dict(self) -> dict:
        return {
            "peak_acceleration_m_s2": self.peak_acceleration_m_s2,
            "dominant_frequency_hz": self.dominant_frequency_hz,
            "fnv": self.fnv,
            "level": self.level,
            "n_samples": self.n_samples,
            "sample_rate_hz": self.sample_rate_hz,
        }


def dominant_frequency(
    time_s: Sequence[float],
    signal: Sequence[float],
    zero_pad_factor: int = 4,
    refine_peak: bool = True,
) -> float:
    """Dominant frequency (Hz) of ``signal`` by windowed FFT.

    The mean is removed, a Hann window is applied (reduces leakage from the
    transient envelope) and the peak bin is refined by a parabolic fit on the
    log-magnitude, which brings the frequency resolution well below one bin for
    a clean single-tone signal.
    """
    t = np.asarray(time_s, dtype=float)
    y = np.asarray(signal, dtype=float)
    if t.size != y.size or t.size < 8:
        raise ValueError("time and signal must have equal length >= 8")
    dt = float(np.mean(np.diff(t)))
    if dt <= 0.0:
        raise ValueError("time axis must be increasing")

    y = y - float(np.mean(y))
    window = np.hanning(y.size)
    n_fft = int(2 ** math.ceil(math.log2(y.size * max(1, int(zero_pad_factor)))))
    spectrum = np.abs(np.fft.rfft(y * window, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=dt)
    spectrum[0] = 0.0
    k = int(np.argmax(spectrum))
    if k == 0:
        return 0.0
    f_peak = freqs[k]
    if refine_peak and 0 < k < spectrum.size - 1:
        a, b, c = np.log(spectrum[k - 1] + 1e-300), np.log(spectrum[k] + 1e-300), np.log(spectrum[k + 1] + 1e-300)
        denom = a - 2.0 * b + c
        if abs(denom) > 1e-18:
            delta = 0.5 * (a - c) / denom
            delta = float(np.clip(delta, -0.5, 0.5))
            f_peak = freqs[k] + delta * (freqs[1] - freqs[0])
    return float(f_peak)


def classify_fnv(fnv: float) -> str:
    """Ice-class habitability class from the FNV value."""
    if fnv < 0.0 or not math.isfinite(fnv):
        raise ValueError("FNV must be finite and non-negative")
    if fnv < FNV_HABITABLE_MAX:
        return "habitable"
    if fnv <= FNV_TOLERABLE_MAX:
        return "tolerable"
    return "unacceptable"


def assess_fnv(
    time_s: Sequence[float],
    acceleration_m_s2: Sequence[float],
    gravity: float = GRAVITY_M_S2,
) -> FnvAssessment:
    """Evaluate ``FNV = a_max / (g f_dom)`` and classify the environment."""
    t = np.asarray(time_s, dtype=float)
    a = np.asarray(acceleration_m_s2, dtype=float)
    if t.size != a.size or t.size < 8:
        raise ValueError("time and acceleration must have equal length >= 8")
    a_max = float(np.max(np.abs(a)))
    f_dom = dominant_frequency(t, a)
    if f_dom <= 0.0:
        raise ValueError("dominant frequency is zero; signal has no spectral peak")
    fnv = a_max / (float(gravity) * f_dom)
    dt = float(np.mean(np.diff(t)))
    return FnvAssessment(
        peak_acceleration_m_s2=a_max,
        dominant_frequency_hz=f_dom,
        fnv=float(fnv),
        level=classify_fnv(fnv),
        n_samples=int(a.size),
        sample_rate_hz=1.0 / dt,
    )