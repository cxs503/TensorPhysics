"""Controlled stationary stress processes for the combination-law study.

Two ingredients are needed to decide whether ice- and wave-induced fatigue
damage are linearly superposable:

* a **controlled** two-band stress process -- each band a stationary Gaussian
  narrow-band process with independently prescribed centre frequency ``f``, RMS
  ``sigma`` and relative bandwidth ``bw = sigma_f / f`` -- so that only the
  frequency separation is varied between runs (a designed experiment, not an
  observational one);
* a **like-for-like** spectral basis: the very same discrete PSD that the
  random-phase synthesis realises is integrated for ``m0..m4``, so the spectral
  closures (Rayleigh / Dirlik / Tovo-Benasciutti) and the rainflow truth see
  the same band-limited spectrum (this is what
  :func:`tensorphysics.fatigue.spectral.discrete_psd_grid` is for).

The Gaussian shape is used because it has a closed-form irregularity factor
``alpha2 = (1 + 2 b^2)^(-1/2) * ... `` (b = bw) and lets the bandwidth be dialled
in continuously, including the narrow-band limit where the analytic combination
laws are exact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from ..spectral import (
    BimodalSpectrum,
    GaussianSpectrum,
    JONSWAP,
    SpectralMoments,
    discrete_psd_grid,
    moments_from_dict,
    spectral_moments,
    synthesize_gaussian_history,
)
from ..rainflow import rainflow_counting_stack

__all__ = [
    "TwoBandCase",
    "band_psd",
    "realised_process",
    "orthogonalise",
    "rainflow_damage_rate",
    "spectral_damage_rates",
    "combined_spectrum",
]


@dataclass
class TwoBandCase:
    """Two independent narrow-band stress bands (ice + wave)."""

    f_ice: float
    f_wave: float
    sigma_ice: float          # MPa
    sigma_wave: float         # MPa
    bw_ice: float = 0.05      # sigma_f / f
    bw_wave: float = 0.05
    duration: float = 10800.0  # s
    fs: float = 50.0           # Hz
    seed: int = 1

    @property
    def n_samples(self) -> int:
        return int(round(self.duration * self.fs))

    @property
    def ratio(self) -> float:
        """Frequency separation f_ice / f_wave (>1 means ice is the fast band)."""
        return self.f_ice / self.f_wave

    def as_dict(self) -> Dict[str, float]:
        return {
            "f_ice_Hz": self.f_ice,
            "f_wave_Hz": self.f_wave,
            "sigma_ice_MPa": self.sigma_ice,
            "sigma_wave_MPa": self.sigma_wave,
            "bw_ice": self.bw_ice,
            "bw_wave": self.bw_wave,
            "duration_s": self.duration,
            "fs_Hz": self.fs,
            "seed": self.seed,
            "ratio": self.ratio,
        }


def band_psd(f_centre: float, sigma: float, bw: float) -> GaussianSpectrum:
    """Gaussian-shaped one-sided stress PSD with total variance ``sigma**2``."""
    if f_centre <= 0.0 or sigma <= 0.0 or bw <= 0.0:
        raise ValueError("f_centre, sigma and bw must be positive")
    return GaussianSpectrum(f_mean=float(f_centre), sigma_f=float(bw) * float(f_centre),
                            m0=float(sigma) ** 2)


def realised_process(
    psd, n_samples: int, fs: float, seed: int, n_sigma_band: float = 6.0,
    f_centre: Optional[float] = None, sigma_f: Optional[float] = None,
) -> Dict[str, object]:
    """Band-limited realisation plus the matching spectral moments.

    Returns the history, the discrete PSD it was drawn from (``freqs``,
    ``s_band``, ``df``), the moments ``m0..m4`` of that band-limited PSD, and
    the realised variance ratio ``var(x)/m0`` (diagnostic, must be ~1).
    """
    if f_centre is None or sigma_f is None:
        band = psd.band()
        centre = 0.5 * (band[0] + band[1])
        half = 0.5 * (band[1] - band[0])
        f_lo = max(1.0 / (n_samples / fs), band[0])
        f_hi = min(0.98 * fs / 2.0, band[1])
    else:
        # explicit +/- n_sigma_band band around the spectral peak: keeps 1-2e-9 of
        # the variance out of the record and, more importantly, keeps the record
        # strictly inside the Nyquist band so that no power is aliased.
        centre = float(f_centre)
        half = float(sigma_f)
        f_lo = max(1.0 / (n_samples / fs), centre - n_sigma_band * half)
        f_hi = min(0.98 * fs / 2.0, centre + n_sigma_band * half)
    x = synthesize_gaussian_history(psd, n_samples, fs, seed=seed, f_lo=f_lo, f_hi=f_hi)
    freqs, s_band, df = discrete_psd_grid(psd, n_samples, fs, f_lo, f_hi)
    mom = moments_from_dict(spectral_moments(freqs, s_band), 0.0, freqs[-1])
    return {
        "history": x,
        "freqs": freqs,
        "s_band": s_band,
        "df": df,
        "moments": mom,
        "var_ratio": float(np.var(x) / mom.m0) if mom.m0 > 0 else float("nan"),
        "f_lo": float(f_lo),
        "f_hi": float(f_hi),
    }


def orthogonalise(y: np.ndarray, x: np.ndarray,
                  target_var: Optional[float] = None) -> np.ndarray:
    """Remove the projection of ``y`` on ``x`` and restore its variance.

    Random-phase synthesis gives every realisation *exactly* the target variance
    (Parseval), but the sample cross-covariance of two independent realisations of
    a finite record is only ~0 (std ~ 1/sqrt(N_eff)).  In a two-band experiment
    that nuisance covariance leaks straight into the variance of the sum and
    therefore into the combination ratio ``kappa`` with a ``(1+2rho)^(m/2)``
    amplification -- a confounder that has nothing to do with the physics (a
    measured +3.8 % covariance on one seed pair moved kappa by ~11 %).  Making the
    two realisations exactly orthogonal removes it: ``var(x+y) =
    var(x)+var(y)`` holds to machine precision and only the frequency/amplitude
    structure remains as an explanatory variable.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    tv = float(np.var(y)) if target_var is None else float(target_var)
    xc = x - x.mean()
    yc = y - y.mean()
    beta = float(np.dot(xc, yc) / np.dot(xc, xc)) if np.dot(xc, xc) > 0 else 0.0
    out = yc - beta * xc
    scale = math.sqrt(tv / float(np.var(out))) if np.var(out) > 0 else 1.0
    return out * scale


def rainflow_damage_rate(
    history: np.ndarray, dt: float, C: float = 1.0, m: float = 3.0
) -> Dict[str, float]:
    """Rainflow + Palmgren-Miner damage **rate** (per second) of a stress history.

    ``D = sum_i n_i S_i^m / (C T)`` with ``S`` in MPa and ``N = C S^-m``.
    """
    x = np.asarray(history, dtype=float)
    T = x.size * float(dt)
    rf = rainflow_counting_stack(x)
    if rf.ranges.size == 0:
        return {"damage_rate": 0.0, "n_cycles": 0.0, "n_cycles_per_s": 0.0,
                "range_max": 0.0, "duration_s": T}
    dmg = float(np.sum(rf.counts * rf.ranges ** m) / C)
    return {
        "damage_rate": dmg / T,
        "n_cycles": float(rf.counts.sum()),
        "n_cycles_per_s": float(rf.counts.sum() / T),
        "range_max": float(rf.ranges.max()),
        "duration_s": T,
    }


def spectral_damage_rates(
    mom: SpectralMoments, C: float = 1.0, m: float = 3.0
) -> Dict[str, float]:
    """Rayleigh (narrow band), Dirlik and Tovo-Benasciutti damage rates (1/s)."""
    from ..spectral import (
        benasciutti_tovo_damage_rate,
        dirlik_damage_rate,
        rayleigh_damage_rate,
    )

    return {
        "rayleigh": float(rayleigh_damage_rate(mom, C=C, m=m)),
        "dirlik": float(dirlik_damage_rate(mom, C=C, m=m)),
        "tovo_benasciutti": float(benasciutti_tovo_damage_rate(mom, C=C, m=m)),
        "alpha2": float(mom.alpha2),
        "alpha1": float(mom.alpha1),
        "nu0_Hz": float(mom.nu0),
        "nup_Hz": float(mom.nup),
        "sigma_MPa": float(mom.sigma),
    }


def combined_spectrum(
    case: TwoBandCase, n_samples: Optional[int] = None, fs: Optional[float] = None
) -> Tuple[GaussianSpectrum, GaussianSpectrum, BimodalSpectrum]:
    """(ice band, wave band, composite spectrum) of a :class:`TwoBandCase`."""
    ice = band_psd(case.f_ice, case.sigma_ice, case.bw_ice)
    wave = band_psd(case.f_wave, case.sigma_wave, case.bw_wave)
    low, high = (wave, ice) if case.f_wave < case.f_ice else (ice, wave)
    return ice, wave, BimodalSpectrum(low=low, high=high)