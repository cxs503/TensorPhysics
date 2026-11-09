"""Wave-induced hull-girder stress: spectra, structural transfer, synthesis.

Model chain (all APPROXIMATE-STRIP, documented so the error budget is explicit):

1. **Sea state** -- one-sided wave-elevation spectrum in Hz, either a JONSWAP
   with peak-enhancement ``gamma`` or a two-peak (swell + wind-sea) JONSWAP
   mixture whose peak separation ``r_p = T_p,swell / T_p,windsea`` controls the
   spectral width.  Both are normalised on ``H_s``.
2. **Load per unit length** (head seas, ``mu = 180 deg``)::

       w(x, omega) = rho g a B(x) exp(-k T/2) exp(i k (x - L/2)),   k = omega^2/g

   i.e. Froude-Krylov-like pressure integrated over the section breadth,
   attenuated with depth, with the wave phase along the hull retained (this is
   the term that produces hogging/sagging and its wavelength dependence).
   **Not modelled**: Smith correction (section emergence), diffraction /
   radiation added-mass in the wave bending moment, slamming.  Consequences:
   the absolute level of the wave bending moment is an approximation, so the
   study normalises the wave-stress variance where a controlled amplitude is
   needed (see ``combination.py``).
3. **Structural transfer** -- :meth:`BeamModes.stress_transfer_function` applies
   the free-free EB modal superposition of :mod:`beam` to ``w(x, omega)`` and
   returns ``T(f)`` = detail stress per unit wave amplitude [Pa/m].
4. **Stress PSD** ``S_sigma(f) = |T(f)|^2 S_eta(f)``; the time-domain leg is the
   random-phase synthesis of exactly this PSD, so the wave component carries
   the same dynamics as the ice component (whipping band included: a wave
   component near the hull-girder natural frequency is amplified too).

Units: SI.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Sequence, Tuple

import numpy as np

from .beam import BeamModes

__all__ = [
    "jonswap_omega",
    "jonswap_f",
    "bimodal_f",
    "waterline_breadth",
    "wave_load_shape",
    "wave_stress_psd",
    "spectral_width_measures",
    "spectral_moments",
    "normalised_psd",
]


# ---------------------------------------------------------------------------
# Wave elevation spectra
# ---------------------------------------------------------------------------
def jonswap_omega(omega, hs: float, tp: float, gamma: float = 3.3) -> np.ndarray:
    """JONSWAP one-sided spectrum ``S_eta(omega)`` [m^2 s/rad].

    ``alpha`` is obtained from the fetch-independent closed form
    ``alpha = 5.061 (H_s/T_p^2)^2 (1 - 0.287 ln gamma)`` (DNV-RP-C205 form), so
    the spectrum integrates to ``H_s^2/16`` up to the gamma-dependent tail.
    """
    om = np.asarray(omega, dtype=float)
    if hs <= 0 or tp <= 0:
        raise ValueError("hs and tp must be positive")
    if gamma < 1.0:
        raise ValueError("gamma must be >= 1")
    om_p = 2.0 * math.pi / tp
    sigma = np.where(om <= om_p, 0.07, 0.09)
    alpha = 5.061 * (hs / tp ** 2) ** 2 * (1.0 - 0.287 * math.log(gamma))
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        peak = gamma ** np.exp(-((om - om_p) ** 2) / (2.0 * sigma ** 2 * om_p ** 2))
        s = alpha * 9.81 ** 2 / om ** 5 * np.exp(-1.25 * (om_p / om) ** 4) * peak
    s = np.where(om > 0.0, s, 0.0)
    return s


def jonswap_f(freqs_hz, hs: float, tp: float, gamma: float = 3.3) -> np.ndarray:
    """JONSWAP in Hz: ``S_eta(f) = 2 pi S_eta(omega)`` [m^2/Hz]."""
    f = np.asarray(freqs_hz, dtype=float)
    return 2.0 * math.pi * jonswap_omega(2.0 * math.pi * np.maximum(f, 0.0), hs, tp, gamma)


def bimodal_f(
    freqs_hz,
    hs: float,
    tp: float,
    gamma: float = 3.3,
    *,
    swell_energy_fraction: float = 0.5,
    peak_separation: float = 2.0,
    swell_gamma: float = 6.0,
) -> np.ndarray:
    """Two-peak spectrum: swell (long period, sharp) + wind sea (short, broad).

    ``tp`` is the *wind-sea* peak period; the swell peak sits at
    ``tp * peak_separation``.  ``peak_separation = 1`` gives a nearly unimodal
    sea, large separation gives the widely separated bimodal case in which
    neither Dirlik nor the narrow-band formula is expected to hold.

    Energy is split by ``swell_energy_fraction`` and each component is
    normalised on its own share of ``H_s^2/16``.
    """
    if not (0.0 < swell_energy_fraction < 1.0):
        raise ValueError("swell_energy_fraction must be in (0, 1)")
    if peak_separation < 1.0:
        raise ValueError("peak_separation must be >= 1")
    hs_sw = hs * math.sqrt(swell_energy_fraction)
    hs_ws = hs * math.sqrt(1.0 - swell_energy_fraction)
    swell = jonswap_f(freqs_hz, hs_sw, tp * peak_separation, swell_gamma)
    wind = jonswap_f(freqs_hz, hs_ws, tp, gamma)
    return swell + wind


# ---------------------------------------------------------------------------
# Load distribution and transfer
# ---------------------------------------------------------------------------
def waterline_breadth(x, length: float, breadth: float, n_exp: float = 4.0, x0: float = 0.0) -> np.ndarray:
    """Super-elliptical waterline breadth distribution ``B(x)`` [m].

    ``B(x) = B (1 - |2 (x - x0)/L - 1|^n)^(1/n)`` -- ``n = 4`` gives a fairly
    full midbody with fine ends, representative of an ice-going tanker/bulker.
    """
    xi = np.asarray(x, dtype=float)
    u = np.abs(2.0 * (xi - x0) / length - 1.0)
    return breadth * np.clip(1.0 - u ** n_exp, 0.0, None) ** (1.0 / n_exp)


def wave_load_shape(
    length: float,
    breadth: float,
    draught: float,
    x0: float = 0.0,
    n_exp: float = 4.0,
    rho: float = 1025.0,
    g: float = 9.81,
) -> Callable:
    """Return ``w(x, omega)``: complex load per unit length, per unit amplitude.

    Depth attenuation ``exp(-k T/2)`` and the along-ship phase ``exp(i k (x-x0-L/2))``
    of a head sea are both retained; the section breadth modulates the pressure
    integral.  Smith correction and diffraction are not modelled (see module doc).
    """

    def w(x, omega: float):
        k = float(omega) ** 2 / g
        b = waterline_breadth(x, length, breadth, n_exp, x0)
        phase = np.exp(1j * k * (np.asarray(x, dtype=float) - x0 - 0.5 * length))
        return rho * g * b * math.exp(-k * draught / 2.0) * phase

    return w


def wave_stress_psd(
    freqs_hz,
    modes: BeamModes,
    elevation_spectrum,
    x_detail: float,
    z: float,
    E: float,
    length: float,
    breadth: float,
    draught: float,
    x0: float = 0.0,
    n_exp: float = 4.0,
    rho: float = 1025.0,
) -> Dict[str, np.ndarray]:
    """Wave-induced detail-stress PSD ``S_sigma(f)`` [Pa^2/Hz] and transfer ``T``.

    ``elevation_spectrum`` is a callable ``f -> S_eta(f)`` in m^2/Hz.  The
    transfer function is evaluated with 801 quadrature points along the hull and
    a 801-point frequency loop (vectorised over modes); the returned ``T`` is in
    Pa per unit wave amplitude, i.e. ``|T|^2 * S_eta`` is the stress PSD.
    """
    f = np.asarray(freqs_hz, dtype=float)
    shape = wave_load_shape(length, breadth, draught, x0, n_exp, rho)
    T = modes.stress_transfer_function(f, shape, x_detail, z, E)
    s_eta = np.asarray(elevation_spectrum(f), dtype=float)
    s_sigma = np.abs(T) ** 2 * s_eta
    return {"freqs_hz": f, "transfer_pa_per_m": T, "s_eta": s_eta, "s_sigma": s_sigma}


# ---------------------------------------------------------------------------
# Spectral moments of the *stress* spectrum
# ---------------------------------------------------------------------------
def spectral_moments(freqs_hz, s_vals, orders: Sequence[float] = (0, 1, 2, 3, 4)) -> Dict[float, float]:
    """``m_k = int f^k S(f) df`` (trapezoidal, zero-padded ends)."""
    f = np.asarray(freqs_hz, dtype=float)
    s = np.asarray(s_vals, dtype=float)
    if f.size != s.size:
        raise ValueError("freqs and PSD must have equal length")
    return {float(k): float(np.trapezoid(s * f ** k, f)) for k in orders}


def spectral_width_measures(freqs_hz, s_vals) -> Dict[str, float]:
    """Bandwidth diagnostics: ``sigma_rms``, ``eps``, ``alpha2``, ``nu0``, ``q``.

    ``eps = sqrt(1 - m1^2/(m0 m2))`` (0 = narrow band, ->1 = wide band),
    ``alpha2 = m2/sqrt(m0 m4)`` (1 = narrow band), ``nu0 = m1/m0`` [Hz] is the
    mean up-crossing rate and ``q = sqrt(1 - alpha2)``-like irregularity from
    ``eps``.
    """
    m = spectral_moments(freqs_hz, s_vals)
    m0, m1, m2, m4 = m[0.0], m[1.0], m[2.0], m[4.0]
    if m0 <= 0 or m2 <= 0:
        return {"m0": m0, "sigma_rms": 0.0, "eps": float("nan"), "alpha2": float("nan"),
                "nu0_hz": float("nan"), "q": float("nan")}
    eps = math.sqrt(max(0.0, 1.0 - m1 ** 2 / (m0 * m2)))
    alpha2 = m2 / math.sqrt(m0 * m4) if m4 > 0 else 0.0
    return {
        "m0": m0,
        "m1": m1,
        "m2": m2,
        "m4": m4,
        "sigma_rms": math.sqrt(m0),
        "eps": eps,
        "alpha2": alpha2,
        "nu0_hz": m1 / m0,
        "q": math.sqrt(max(0.0, 1.0 - alpha2)),
    }


def normalised_psd(freqs_hz, s_vals, target_rms: float) -> Tuple[np.ndarray, float]:
    """Scale a PSD so that ``sqrt(int S df) = target_rms``; returns gain and PSD.

    The gain is *reported* by the study: for the wave component it equals the
    square of the sea-state scale factor implied by the strip-transfer
    approximation (an honest diagnostic of how well the APPROXIMATE-STRIP model
    reproduces a realistic wave bending moment level).
    """
    if target_rms <= 0:
        raise ValueError("target_rms must be positive")
    f = np.asarray(freqs_hz, dtype=float)
    s = np.asarray(s_vals, dtype=float)
    m0 = float(np.trapezoid(s, f))
    if m0 <= 0:
        raise ValueError("PSD has zero variance; cannot normalise")
    gain = target_rms ** 2 / m0
    return s * gain, gain