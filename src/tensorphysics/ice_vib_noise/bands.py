"""One-third-octave band algebra (IEC 61260-1 / ANSI S1.11 exact base-10 series).

The band set is generated from the exact preferred-number series

    f_m(n) = 1000 * G**n,     G = 10**0.1 = 2**(1/3),   n integer
    f_lo   = f_m * G**(-1/2), f_hi = f_m * G**(+1/2)

(ten bands per decade, three bands per octave: the successive mid-band ratio
is exactly ``G = 10**0.1 = 2**(1/3)``.)

so that ``f_hi/f_lo = G**(1/3) = 2**(1/3)`` (the classic 1/3-octave ratio) and
the ratio of successive mid-band frequencies is exactly ``2**(1/3)``.

Everything is deliberately done in the *power* domain: a band mean-square is a
true integral of the one-sided PSD over the band edges, not a single-bin
sample.  That matters for the acceptance criterion that the band sum of the
spectrum reproduces the total mean square (Parseval closure) — see
``bands_sum_square``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Sequence, Tuple

import numpy as np

__all__ = [
    "G",
    "OneThirdOctaveBands",
    "one_third_octave_bands",
    "band_mean_square",
    "bands_sum_square",
    "rms_from_mean_square",
    "db_from_level",
]

#: Base of the preferred-number series used by IEC 61260-1.
G = 10.0 ** 0.1


@dataclass(frozen=True)
class OneThirdOctaveBands:
    """A set of contiguous 1/3-octave bands.

    Attributes
    ----------
    centre_hz:
        Mid-band frequencies (Hz) in ascending order.
    lower_hz, upper_hz:
        Exact band edges (Hz).
    """

    centre_hz: Tuple[float, ...]
    lower_hz: Tuple[float, ...]
    upper_hz: Tuple[float, ...]

    def __post_init__(self) -> None:
        n = len(self.centre_hz)
        if n == 0:
            raise ValueError("band set must contain at least one band")
        if not (len(self.lower_hz) == len(self.upper_hz) == n):
            raise ValueError("centre/lower/upper arrays must have equal length")
        for fc, lo, hi in zip(self.centre_hz, self.lower_hz, self.upper_hz):
            if not (0.0 < lo < fc < hi):
                raise ValueError(f"inconsistent band edges: {lo}, {fc}, {hi}")
        # Exactness audit: successive mid-band ratios must equal the IEC
        # exact base-10 ratio G = 10**0.1 (the "preferred" 2**(1/3) series is
        # also accepted, they differ by 8e-4).
        for i in range(n - 1):
            ratio = self.centre_hz[i + 1] / self.centre_hz[i]
            if min(abs(ratio / G - 1.0), abs(ratio / 2.0 ** (1.0 / 3.0) - 1.0)) > 1e-9:
                raise ValueError("centre frequencies are not a 1/3-octave series")

    @property
    def n_bands(self) -> int:
        return len(self.centre_hz)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "centre_hz": list(self.centre_hz),
            "lower_hz": list(self.lower_hz),
            "upper_hz": list(self.upper_hz),
            "ratio_centre_iec_base10": G,
            "ratio_centre_preferred": 2.0 ** (1.0 / 3.0),
            "ratio_band_edges": G,
            "octave_ratio_exact_base10": G ** 3,
            "nominal_standard": "IEC 61260-1 / ANSI S1.11 exact base-10 series",
        }


def one_third_octave_bands(f_min_hz: float = 1.0, f_max_hz: float = 1000.0) -> OneThirdOctaveBands:
    """Contiguous 1/3-octave bands with mid-band frequencies in ``[f_min, f_max]``."""
    if not (0.0 < f_min_hz < f_max_hz):
        raise ValueError("require 0 < f_min_hz < f_max_hz")
    n_lo = int(math.ceil(10.0 * math.log10(f_min_hz / 1000.0) - 1e-12))
    n_hi = int(math.floor(10.0 * math.log10(f_max_hz / 1000.0) + 1e-12))
    if n_hi < n_lo:
        raise ValueError("requested range contains no 1/3-octave mid-band frequency")
    centre = [1000.0 * G ** n for n in range(n_lo, n_hi + 1)]
    lower = [fc * G ** -0.5 for fc in centre]
    upper = [fc * G ** 0.5 for fc in centre]
    return OneThirdOctaveBands(tuple(centre), tuple(lower), tuple(upper))


def _cumulative_integral(frequencies: np.ndarray, psd: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Cumulative trapezoidal integral of ``psd`` over ``frequencies``."""
    f = np.asarray(frequencies, dtype=float)
    s = np.asarray(psd, dtype=float)
    if f.shape != s.shape:
        raise ValueError("frequencies and psd must have the same shape")
    if f.ndim != 1 or f.size < 2:
        raise ValueError("need a 1-D spectrum with at least two samples")
    inc = 0.5 * (s[1:] + s[:-1]) * np.diff(f)
    cum = np.concatenate(([0.0], np.cumsum(inc)))
    return f, cum


def band_mean_square(frequencies: np.ndarray, psd: np.ndarray, bands: OneThirdOctaveBands) -> np.ndarray:
    """Mean square (integral of the one-sided PSD) inside each band.

    Band edges that fall between spectral lines are handled by linear
    interpolation of the cumulative integral, so the returned values are exact
    trapezoidal integrals over the band interval clipped to the spectrum
    support.
    """
    f, cum = _cumulative_integral(frequencies, psd)
    out = np.zeros(bands.n_bands, dtype=float)
    for i, (lo, hi) in enumerate(zip(bands.lower_hz, bands.upper_hz)):
        lo_c = max(lo, float(f[0]))
        hi_c = min(hi, float(f[-1]))
        if hi_c <= lo_c:
            continue
        cum_lo = float(np.interp(lo_c, f, cum))
        cum_hi = float(np.interp(hi_c, f, cum))
        out[i] = max(cum_hi - cum_lo, 0.0)
    return out


def bands_sum_square(band_values: np.ndarray) -> float:
    """Sum of band mean squares (Parseval closure helper)."""
    return float(np.sum(np.asarray(band_values, dtype=float)))


def rms_from_mean_square(mean_square: float) -> float:
    """RMS from a mean-square value (negative input is a hard error)."""
    if mean_square < 0.0:
        raise ValueError("mean square must be non-negative")
    return math.sqrt(mean_square)


def db_from_level(level: float, reference: float = 1.0, floor: float = 1e-30) -> float:
    """``10 log10(level / reference)`` with a floor so that a zero band is finite."""
    if reference <= 0.0:
        raise ValueError("reference must be positive")
    return 10.0 * math.log10(max(float(level), floor) / reference)


def bands_over_range(frequencies: Sequence[float], bands: OneThirdOctaveBands) -> Iterable[int]:
    """Indices of bands that lie fully inside the spectral support (diagnostic)."""
    f0, f1 = float(frequencies[0]), float(frequencies[-1])
    for i, (lo, hi) in enumerate(zip(bands.lower_hz, bands.upper_hz)):
        if lo >= f0 and hi <= f1:
            yield i