"""Spectral (frequency-domain) fatigue assessment.

Chain implemented here::

    S(f) --> spectral moments m0..m4 --> rainflow range distribution
         --> damage rate  D = nu * E[S^m] / C   for the S-N law S^m N = C

Four independent closures of the same chain are provided so that they can be
cross-validated against each other and against the time domain:

1. **Narrow-band Rayleigh** (Rice 1944-45).  For a narrow-band Gaussian stress
   process the stress *range* (peak-to-trough) is Rayleigh distributed and

       D_R = (nu0 / C) * (2*sqrt(2)*sigma)^m * Gamma(1 + m/2),
       sigma = sqrt(m0),  nu0 = m1/m0.

2. **Dirlik (1985)**.  Four-parameter empirical range PDF built from
   (m0, m1, m2, m4); exact in the narrow-band limit and validated for
   wide-band processes.  The m-th range moment has a closed form, so the damage
   rate needs no numerical integration of the PDF.

3. **Tovo-Benasciutti (Benasciutti & Tovo 2005)**.  Narrow-band damage
   multiplied by a bandwidth correction

       lambda = b + (1 - b) * alpha2^(m-1),
       b = clip((alpha1 - alpha2) / (1 - alpha1), 0, 1)

4. **Bimodal combination/separation limit**.  For two well separated narrow-band
   components (f2/f1 >= 5) the rainflow ranges asymptotically split into the
   ranges of each component, hence D -> D1 + D2 (Torhaug 1990, Jiao & Moan 1990).
   This is used as a third anchor for strongly bimodal spectra.

The time-domain leg of the triangle is obtained by synthesising a stationary
Gaussian time history from the *same* (band limited) PSD with random phases at
fixed seeds, counting cycles with the ASTM E1049-85 rainflow rule and applying
Palmgren-Miner.

Units: frequencies in Hz, spectral density in unit^2/Hz, stress ranges in the
same unit as the PSD amplitude, S-N constant ``C`` in (stress)^m (single-slope
form N = C * S^-m).  Damage rates are per second; multiply by the design life in
seconds for the accumulated damage.

References
----------
* Rice, S.O. (1944-45) Mathematical analysis of random noise.
* Dirlik, T. (1985) Application of computers in fatigue analysis, PhD thesis,
  University of Warwick.
* Benasciutti, D., Tovo, R. (2005) Spectral methods for lifetime prediction under
  wide-band stationary random processes, Int. J. Fatigue 27, 867-877.
* Torhaug, R. (1990); Jiao, G., Moan, T. (1990) Probabilistic analysis of
  fatigue damage for a narrow band Gaussian process, FFEMS 13, 619-630.
* ASTM E1049-85 Standard practices for cycle counting in fatigue analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Sequence, Tuple

import numpy as np
from scipy import integrate, special

from .rainflow import rainflow_counting, rainflow_counting_stack

__all__ = [
    "SpectralMoments",
    "GaussianSpectrum",
    "PiersonMoskowitz",
    "JONSWAP",
    "BimodalSpectrum",
    "moments_from_dict",
    "spectral_moments",
    "spectral_moments_callable",
    "rayleigh_damage_rate",
    "DirlikParameters",
    "dirlik_parameters",
    "dirlik_range_pdf",
    "dirlik_expected_range_power",
    "dirlik_damage_rate",
    "rayleigh_expected_range_power",
    "tovo_benasciutti_weight",
    "benasciutti_tovo_damage_rate",
    "bimodal_analytic_damage_rate",
    "discrete_psd_grid",
    "synthesize_gaussian_history",
    "time_domain_damage_rate",
    "monte_carlo_damage_rate",
    "sensitivity_table",
    "spectral_fatigue_assessment",
]


# ---------------------------------------------------------------------------
# Spectral moments and bandwidth descriptors
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpectralMoments:
    """Spectral moments m0..m4 of a one-sided stress PSD plus descriptors.

    Attributes
    ----------
    m : tuple of 5 floats
        ``(m0, m1, m2, m3, m4)`` with ``mn = int f^n S(f) df``.
    f_lo, f_hi : float
        Frequency band used for the integration (Hz).  The band matters for m4
        whenever the PSD tail decays like f^-5 (PM / JONSWAP): there the
        untruncated m4 diverges, so m4 is always a band-limited quantity.
    """

    m: Tuple[float, float, float, float, float]
    f_lo: float = 0.0
    f_hi: float = float("inf")

    @property
    def m0(self) -> float:
        return float(self.m[0])

    @property
    def m1(self) -> float:
        return float(self.m[1])

    @property
    def m2(self) -> float:
        return float(self.m[2])

    @property
    def m3(self) -> float:
        return float(self.m[3])

    @property
    def m4(self) -> float:
        return float(self.m[4])

    @property
    def sigma(self) -> float:
        """RMS of the stress process, sqrt(m0)."""
        return float(np.sqrt(self.m0))

    @property
    def nu0(self) -> float:
        """Mean frequency ``m1/m0`` (Hz).

        Note: this is *not* the zero up-crossing rate of a Gaussian process -
        that is :attr:`nu_zero` = ``(1/2pi) sqrt(m2/m0)``.  The two coincide only
        in the narrow-band limit (``alpha1 -> 1``); for a wide-band spectrum
        ``m1/m0 = alpha1 * nu_zero`` and they differ by up to a factor of two.
        """
        return float(self.m1 / self.m0)

    @property
    def nu_zero(self) -> float:
        """Zero up-crossing rate ``(1/2pi) sqrt(m2/m0)`` (Hz), Rice (1944).

        This is the cycle rate of the Rayleigh narrow-band damage model: every
        upward zero crossing produces one stress cycle whose range is Rayleigh
        distributed with scale ``2 sqrt(2) sigma``.
        """
        return float(np.sqrt(self.m2 / self.m0))

    @property
    def nup(self) -> float:
        """Peak rate sqrt(m4/m2) (Hz) - cycle rate used by Dirlik."""
        return float(np.sqrt(self.m4 / self.m2))

    @property
    def alpha1(self) -> float:
        """Irregularity factor alpha1 = m1 / sqrt(m0 m2)."""
        return float(self.m1 / np.sqrt(self.m0 * self.m2))

    @property
    def alpha2(self) -> float:
        """Bandwidth parameter alpha2 = m2 / sqrt(m0 m4)."""
        return float(self.m2 / np.sqrt(self.m0 * self.m4))

    @property
    def xm(self) -> float:
        """Dirlik mean-frequency parameter xm = (m1/m0) sqrt(m2/m4)."""
        return float((self.m1 / self.m0) * np.sqrt(self.m2 / self.m4))

    @property
    def bandwidth_eps(self) -> float:
        """Cartwright-Longuet-Higgins bandwidth eps = sqrt(1 - alpha2^2)."""
        return float(np.sqrt(max(0.0, 1.0 - self.alpha2 ** 2)))

    def as_dict(self) -> Dict[str, float]:
        return {
            "m0": self.m0,
            "m1": self.m1,
            "m2": self.m2,
            "m3": self.m3,
            "m4": self.m4,
            "sigma": self.sigma,
            "nu0_hz": self.nu0,
            "nup_hz": self.nup,
            "alpha1": self.alpha1,
            "alpha2": self.alpha2,
            "xm": self.xm,
            "bandwidth_eps": self.bandwidth_eps,
            "f_lo_hz": self.f_lo,
            "f_hi_hz": self.f_hi,
        }


def moments_from_dict(m, f_lo: float = 0.0,
                      f_hi: float = float("inf")) -> SpectralMoments:
    """Build a :class:`SpectralMoments` from a ``{order: value}`` mapping."""
    return SpectralMoments(
        m=tuple(float(m[n]) for n in (0, 1, 2, 3, 4)), f_lo=f_lo, f_hi=f_hi
    )


def _make_breakpoints(f_lo: float, f_hi: float, split_points=(),
                      decade_step: float = 0.5):
    """Sorted breakpoints: geometric subdivision + user supplied split points.

    Adaptive quadrature is only reliable when each subinterval is narrower than
    the structure of the integrand (a spectral peak is often << 1 decade wide),
    so the band is subdivided geometrically before ``quad`` is called and any
    known peak locations are inserted explicitly.
    """
    lo = float(f_lo)
    hi = float(f_hi)
    pts = {lo, hi}
    start = lo if lo > 0 else hi * 1e-6
    if start > 0 and hi > start:
        n = max(1, int(np.ceil(np.log10(hi / start) / decade_step)))
        pts.update(np.geomspace(start, hi, n + 1).tolist())
    for p in split_points:
        if np.isfinite(p) and lo < float(p) < hi:
            pts.add(float(p))
    return sorted(pts)


def spectral_moments_callable(
    psd: Callable[[float], float],
    f_lo: float,
    f_hi: float,
    orders: Sequence[float] = (0, 1, 2, 3, 4),
    split_points=(),
) -> Dict[float, float]:
    """Adaptive quadrature of ``mn = int_{f_lo}^{f_hi} f^n S(f) df``.

    The band is subdivided geometrically (<= 0.5 decade per subinterval, plus any
    ``split_points`` such as spectral peak frequencies) and each subinterval is
    integrated with ``scipy.integrate.quad`` at relative tolerance 1e-13.  This
    keeps the rule accurate for peaks that are far narrower than the band, where
    a single ``quad`` call silently loses the peak.
    """
    if not (0.0 <= f_lo < f_hi):
        raise ValueError("require 0 <= f_lo < f_hi")
    breaks = _make_breakpoints(f_lo, f_hi, split_points)

    def integrand(f, n):
        return (f ** n) * float(psd(f))

    out: Dict[float, float] = {}
    with np.errstate(all="ignore"):
        for n in orders:
            total = 0.0
            for a, b in zip(breaks[:-1], breaks[1:]):
                if b <= a:
                    continue
                val = integrate.quad(integrand, a, b, args=(n,), epsabs=0.0,
                                     epsrel=1e-13, limit=200)[0]
                total += val
            out[float(n)] = float(total)
    return out


def spectral_moments(
    f: Sequence[float] | np.ndarray,
    s: Sequence[float] | np.ndarray,
    orders: Sequence[float] = (0, 1, 2, 3, 4),
) -> Dict[float, float]:
    """Spectral moments of a sampled one-sided PSD (trapezoidal rule).

    Parameters
    ----------
    f, s : array_like
        Frequency axis (Hz, ascending, non-negative) and one-sided PSD values.
    orders : sequence of float
        Moment orders to compute; non-integer orders are allowed.

    Returns
    -------
    dict
        ``{order: mn}``.

    Notes
    -----
    For tabulated PSFs the resolution of the high-frequency tail is what
    determines m4 - a coarse tail biases m4 low.  Where a closed form exists use
    :meth:`PiersonMoskowitz.analytic_moments` or
    :meth:`GaussianSpectrum.analytic_moments` instead.
    """
    fa = np.asarray(f, dtype=float)
    sa = np.asarray(s, dtype=float)
    if fa.shape != sa.shape:
        raise ValueError("f and s must have the same shape")
    if fa.size < 2:
        raise ValueError("need at least two samples")
    if np.any(fa < 0):
        raise ValueError("frequencies must be non-negative")
    out: Dict[float, float] = {}
    for n in orders:
        out[float(n)] = float(np.trapezoid(sa * fa ** float(n), fa))
    return out

# ---------------------------------------------------------------------------
# One-sided PSD models
# ---------------------------------------------------------------------------

class _Spectrum:
    """Common interface for one-sided stress PSD models."""

    name: str = "spectrum"

    def __call__(self, f):  # pragma: no cover - interface
        raise NotImplementedError

    def band(self, f_lo=None, f_hi=None) -> Tuple[float, float]:
        raise NotImplementedError

    def split_points(self) -> Tuple[float, ...]:
        """Frequencies where the integrand changes structure (peak locations)."""
        return ()

    def moments(self, orders: Sequence[float] = (0, 1, 2, 3, 4),
                f_lo=None, f_hi=None) -> Dict[float, float]:
        lo, hi = self.band(f_lo, f_hi)
        return spectral_moments_callable(self, lo, hi, orders,
                                         split_points=self.split_points())

    def moments_object(self, f_lo=None, f_hi=None) -> SpectralMoments:
        lo, hi = self.band(f_lo, f_hi)
        return moments_from_dict(self.moments((0, 1, 2, 3, 4), lo, hi), lo, hi)


@dataclass
class GaussianSpectrum(_Spectrum):
    """Single-peak Gaussian (normal-shaped) PSD.

    ``S(f) = m0 / (sqrt(2 pi) sigma_f) exp(-(f - f_mean)^2 / (2 sigma_f^2))``

    Analytic (untruncated) moments are exact::

        m_n = m0 * E[X^n],   X ~ N(f_mean, sigma_f^2)

    so ``m_n = m0 * sum_k C(n,k) f_mean^(n-k) (2k-1)!! sigma_f^(2k)`` over even
    subscripts ``k``.  The band is clipped at ``f > 0`` in the numerical
    integration; :meth:`truncation_ratio` reports how much of m_n survives that
    clipping (an explicit measure of how well the model is a "single-peak
    Gaussian family" rather than a folded distribution).
    """

    f_mean: float
    sigma_f: float
    m0: float = 1.0
    name: str = "gaussian"

    def __post_init__(self):
        if self.sigma_f <= 0:
            raise ValueError("sigma_f must be positive")
        if self.f_mean <= 0:
            raise ValueError("f_mean must be positive")
        if self.m0 <= 0:
            raise ValueError("m0 must be positive")

    def __call__(self, f):
        f = np.asarray(f, dtype=float)
        coef = self.m0 / (np.sqrt(2.0 * np.pi) * self.sigma_f)
        return coef * np.exp(-0.5 * ((f - self.f_mean) / self.sigma_f) ** 2)

    def band(self, f_lo=None, f_hi=None) -> Tuple[float, float]:
        lo = 0.0 if f_lo is None else float(f_lo)
        hi = self.f_mean + 40.0 * self.sigma_f if f_hi is None else float(f_hi)
        return lo, hi

    def split_points(self) -> Tuple[float, ...]:
        return (self.f_mean - 2.0 * self.sigma_f, self.f_mean - self.sigma_f,
                self.f_mean, self.f_mean + self.sigma_f,
                self.f_mean + 2.0 * self.sigma_f)

    def analytic_moments(self, orders: Sequence[float] = (0, 1, 2, 3, 4)
                         ) -> Dict[float, float]:
        mu, sd, m0 = self.f_mean, self.sigma_f, self.m0
        out: Dict[float, float] = {}
        for n in orders:
            if float(n) != int(n):
                raise ValueError("analytic Gaussian moments need integer orders")
            k_n = int(n)
            total = 0.0
            for k in range(0, k_n + 1, 2):
                coef = special.comb(k_n, k, exact=True)
                double_fact = float(special.factorial2(k - 1)) if k >= 2 else 1.0
                total += coef * mu ** (k_n - k) * sd ** k * double_fact
            out[float(n)] = float(m0 * total)
        return out

    def analytic_moments_object(self) -> SpectralMoments:
        return moments_from_dict(self.analytic_moments(), 0.0, float("inf"))

    def truncation_ratio(self, n: int = 0) -> float:
        """Fraction of m_n retained when the spectrum is clipped at f > 0."""
        lo, hi = self.band()
        full = self.analytic_moments((n,))[float(n)]
        trunc = spectral_moments_callable(self, lo, hi, (n,))[float(n)]
        return float(trunc / full)


@dataclass
class PiersonMoskowitz(_Spectrum):
    """Pierson-Moskowitz / ITTC two-parameter spectrum.

    ``S(f) = (5/16) Hs^2 fp^4 f^-5 exp(-(5/4)(fp/f)^4)``

    Band-limited moments in closed form (A = 5/16 Hs^2 fp^4, B = 5/4 fp^4,
    ``Gamma(.,.)`` the upper incomplete gamma function)::

        m_n = (A/4) B^((n-4)/4) [Gamma(1-n/4, B/f_hi^4) - Gamma(1-n/4, B/f_lo^4)]

    For ``n < 4`` the limit ``f_hi -> inf`` exists and, with ``f_lo -> 0``,
    ``m_n = (A/4) B^((n-4)/4) Gamma(1-n/4)``.  For ``n = 4`` the exponent
    ``1 - n/4`` vanishes and ``Gamma(0, u) = E1(u) -> -ln u - gamma_E``: m4
    *diverges* logarithmically.  This is a physical property of the f^-5 tail,
    not a numerical artefact, and is why every spectral fatigue result must quote
    the cut-off frequency that produced its m4.
    """

    hs: float
    fp: float
    name: str = "pierson_moskowitz"

    def __post_init__(self):
        if self.hs <= 0 or self.fp <= 0:
            raise ValueError("hs and fp must be positive")

    def __call__(self, f):
        f = np.asarray(f, dtype=float)
        out = np.zeros_like(f, dtype=float)
        mask = f > 0
        fm = f[mask]
        a = (5.0 / 16.0) * self.hs ** 2 * self.fp ** 4
        out[mask] = a * fm ** -5 * np.exp(-1.25 * (self.fp / fm) ** 4)
        return out

    def band(self, f_lo=None, f_hi=None) -> Tuple[float, float]:
        lo = 0.0 if f_lo is None else float(f_lo)
        # Default band = 20 fp: the tail beyond it carries (fp/f_hi)^4 = 6e-6 of
        # m0 for a PM spectrum, while m4 is a band-limited quantity anyway (its
        # untruncated value diverges) - always quote f_hi with an m4 based result.
        hi = 20.0 * self.fp if f_hi is None else float(f_hi)
        return lo, hi

    def split_points(self) -> Tuple[float, ...]:
        return (0.5 * self.fp, 0.75 * self.fp, self.fp, 1.25 * self.fp,
                1.5 * self.fp, 2.0 * self.fp, 3.0 * self.fp)

    @staticmethod
    def _upper_gamma(a: float, x: float) -> float:
        """Upper incomplete Gamma(a, x) for a >= 0 (E1 when a == 0)."""
        if x <= 0:
            return float("inf")
        if a == 0.0:
            return float(special.exp1(x))
        return float(special.gammaincc(a, x) * special.gamma(a))

    def analytic_moments(self, orders: Sequence[float] = (0, 1, 2, 3, 4),
                         f_lo: float = 0.0,
                         f_hi: float = float("inf")) -> Dict[float, float]:
        """Closed-form band-limited moments (see class docstring)."""
        a_coef = (5.0 / 16.0) * self.hs ** 2 * self.fp ** 4
        b_coef = 1.25 * self.fp ** 4
        u_lo = b_coef / float(f_lo) ** 4 if f_lo > 0 else float("inf")
        out: Dict[float, float] = {}
        for n in orders:
            n = float(n)
            a_arg = 1.0 - n / 4.0
            u_hi = b_coef / float(f_hi) ** 4 if np.isfinite(f_hi) else 0.0
            g_hi = self._upper_gamma(a_arg, u_hi) if u_hi > 0 else (
                float("inf") if a_arg <= 0 else float(special.gamma(a_arg)))
            g_lo = self._upper_gamma(a_arg, u_lo) if np.isfinite(u_lo) else 0.0
            if not np.isfinite(g_hi):
                raise ValueError(
                    f"moment order n={n} diverges for a PM spectrum (tail f^-5); "
                    "pass a finite f_hi"
                )
            out[float(n)] = float(
                (a_coef / 4.0) * b_coef ** ((n - 4.0) / 4.0) * (g_hi - g_lo)
            )
        return out


@dataclass
class JONSWAP(_Spectrum):
    """JONSWAP spectrum: PM base times a peak enhancement factor.

    ``S(f) = S_PM(f) * gamma^r``,  ``r = exp(-(f-fp)^2 / (2 sigma^2 fp^2))``
    with ``sigma = 0.07`` for ``f <= fp`` and ``0.09`` for ``f > fp``.

    No closed-form moments exist (the enhancement factor is not integrable in
    elementary / gamma functions), so moments are computed by adaptive
    quadrature.  This is a genuine limitation: for JONSWAP the moment check is a
    *self-consistency* check between two quadrature schemes, not an
    analytic-vs-numeric check.
    """

    hs: float
    fp: float
    gamma: float = 3.3
    sigma_low: float = 0.07
    sigma_high: float = 0.09
    name: str = "jonswap"

    def __post_init__(self):
        if self.hs <= 0 or self.fp <= 0:
            raise ValueError("hs and fp must be positive")
        if self.gamma < 1.0:
            raise ValueError("gamma must be >= 1")

    def peak_enhancement(self, f):
        f = np.asarray(f, dtype=float)
        sigma = np.where(f <= self.fp, self.sigma_low, self.sigma_high)
        return np.exp(-0.5 * ((f - self.fp) / (sigma * self.fp)) ** 2)

    def __call__(self, f):
        f = np.asarray(f, dtype=float)
        pm = PiersonMoskowitz(self.hs, self.fp)(f)
        return pm * self.gamma ** self.peak_enhancement(f)

    def band(self, f_lo=None, f_hi=None) -> Tuple[float, float]:
        lo = 0.0 if f_lo is None else float(f_lo)
        hi = 20.0 * self.fp if f_hi is None else float(f_hi)
        return lo, hi

    def split_points(self) -> Tuple[float, ...]:
        return (0.5 * self.fp, 0.75 * self.fp,
                self.fp * (1.0 - 3.0 * self.sigma_low), self.fp,
                self.fp * (1.0 + 3.0 * self.sigma_high),
                1.5 * self.fp, 2.0 * self.fp, 3.0 * self.fp)


@dataclass
class BimodalSpectrum(_Spectrum):
    """Sum of two one-sided PSD components (e.g. wind sea + swell)."""

    low: _Spectrum
    high: _Spectrum
    name: str = "bimodal"

    def __call__(self, f):
        return self.low(f) + self.high(f)

    def band(self, f_lo=None, f_hi=None) -> Tuple[float, float]:
        lo_l, _ = self.low.band(f_lo, None)
        lo_h, _ = self.high.band(f_lo, None)
        _, hi_l = self.low.band(None, f_hi)
        _, hi_h = self.high.band(None, f_hi)
        return max(0.0, min(lo_l, lo_h)), max(hi_l, hi_h)

    def split_points(self) -> Tuple[float, ...]:
        return tuple(self.low.split_points()) + tuple(self.high.split_points())

    def peak_frequencies(self) -> Tuple[float, float]:
        def _peak(spec):
            if isinstance(spec, GaussianSpectrum):
                return spec.f_mean
            if isinstance(spec, (PiersonMoskowitz, JONSWAP)):
                return spec.fp
            raise TypeError("peak frequency only defined for Gaussian/PM/JONSWAP")
        return _peak(self.low), _peak(self.high)

    def separation_ratio(self) -> float:
        f1, f2 = self.peak_frequencies()
        return float(max(f1, f2) / min(f1, f2))


# ---------------------------------------------------------------------------
# Frequency-domain damage rate models
# ---------------------------------------------------------------------------

def _check_sn(C: float, m: float) -> None:
    if C <= 0:
        raise ValueError("S-N constant C must be positive")
    if m <= 0:
        raise ValueError("S-N slope m must be positive")


def rayleigh_damage_rate(mom: SpectralMoments, C: float = 1.0,
                         m: float = 3.0) -> float:
    """Narrow-band (Rayleigh) damage rate, per second.

    D_R = (nu0 / C) * (2*sqrt(2)*sigma)^m * Gamma(1 + m/2)

    Derivation: for a narrow-band Gaussian process the range is
    ``S = 2 * amplitude`` with the amplitude Rayleigh distributed with scale
    ``sigma = sqrt(m0)``; hence ``E[S^m] = (2 sqrt(2) sigma)^m Gamma(1+m/2)`` and
    the cycle rate is the zero **up-crossing** rate ``nu0 = (1/2pi) sqrt(m2/m0)``
    (Rice 1944), *not* the mean frequency ``m1/m0``.

    .. note::
       Until the P27W3 integration audit this function used ``m1/m0`` (the mean
       frequency).  The two rates are equal in the narrow-band limit, so the
       narrow-band regression tests did not notice; on the wide-band contract
       spectrum of the ``stress-psd-v1`` payload (``alpha1 = 0.653``) the two
       differ by 34.7 %, which the cross-repo comparison against the ship
       platform surfaced.  ``tests/test_p27w3_integration_contracts.py`` pins the
       corrected convention.
    """
    _check_sn(C, m)
    return float(mom.nu_zero / C * rayleigh_expected_range_power(mom, m))


def rayleigh_expected_range_power(mom: SpectralMoments, m: float = 3.0) -> float:
    """``E[S^m] = (2 sqrt(2) sigma)^m Gamma(1 + m/2)`` for the narrow-band PDF."""
    return float((2.0 * np.sqrt(2.0) * mom.sigma) ** m * special.gamma(1.0 + 0.5 * m))


@dataclass(frozen=True)
class DirlikParameters:
    """Dirlik (1985) four-parameter range-PDF coefficients."""

    D1: float
    D2: float
    D3: float
    Q: float
    R: float
    xm: float
    alpha2: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "D1": self.D1, "D2": self.D2, "D3": self.D3,
            "Q": self.Q, "R": self.R, "xm": self.xm, "alpha2": self.alpha2,
        }


def dirlik_parameters(mom: SpectralMoments) -> DirlikParameters:
    """Dirlik coefficients from (m0, m1, m2, m4).

    ``xm = (m1/m0) sqrt(m2/m4)``, ``D1 = 2(xm - a2^2)/(1 + a2^2)``,
    ``R = (a2 - xm - D1^2)/(1 - a2 - D1 + D1^2)``,
    ``D2 = (1 - a2 - D1 + D1^2)/(1 - R)``, ``D3 = 1 - D1 - D2``,
    ``Q = 1.25 (a2 - D3 - D2 R)/D1`` with ``a2 = alpha2``.

    In the narrow-band limit ``a2 -> 1`` the denominators vanish; the PDF
    degenerates to Rayleigh and the coefficients are then not needed (see
    :func:`dirlik_damage_rate`, which switches to the Rayleigh branch).
    """
    a2 = mom.alpha2
    xm = mom.xm
    d1 = 2.0 * (xm - a2 ** 2) / (1.0 + a2 ** 2)
    den = 1.0 - a2 - d1 + d1 ** 2
    if abs(den) < 1e-300:
        raise ValueError("degenerate Dirlik coefficients (alpha2 == 1)")
    r = (a2 - xm - d1 ** 2) / den
    if abs(1.0 - r) < 1e-300:
        raise ValueError("degenerate Dirlik coefficients (R == 1)")
    d2 = (1.0 - a2 - d1 + d1 ** 2) / (1.0 - r)
    d3 = 1.0 - d1 - d2
    if abs(d1) < 1e-300:
        raise ValueError("degenerate Dirlik coefficients (D1 == 0)")
    q = 1.25 * (a2 - d3 - d2 * r) / d1
    return DirlikParameters(D1=d1, D2=d2, D3=d3, Q=q, R=r, xm=xm, alpha2=a2)


def dirlik_range_pdf(stress_range, mom: SpectralMoments,
                     params: DirlikParameters | None = None):
    """Dirlik range PDF ``p(S)`` in 1/(stress unit).

    ``Z = S/(2 sqrt(m0))``,
    ``p(S) = [ (D1/Q) e^{-Z/Q} + (D2 Z/R^2) e^{-Z^2/(2R^2)} + D3 Z e^{-Z^2/2} ]
             / (2 sqrt(m0))``
    """
    prm = dirlik_parameters(mom) if params is None else params
    s = np.asarray(stress_range, dtype=float)
    z = s / (2.0 * mom.sigma)
    val = (prm.D1 / prm.Q * np.exp(-z / prm.Q)
           + prm.D2 * z / prm.R ** 2 * np.exp(-z ** 2 / (2.0 * prm.R ** 2))
           + prm.D3 * z * np.exp(-z ** 2 / 2.0))
    return val / (2.0 * mom.sigma)


def dirlik_expected_range_power(mom: SpectralMoments, m: float = 3.0,
                                params: DirlikParameters | None = None) -> float:
    """E[S^m] under the Dirlik range PDF, in closed form.

    With ``Z = S/(2 sqrt(m0))``::

        E[S^m] = (2 sqrt(m0))^m [ D1 Q^m Gamma(m+1)
                                  + (D2 (sqrt(2) R)^m + D3 2^(m/2)) Gamma(1+m/2) ]

    both integrals being standard gamma integrals over ``Z in [0, inf)``.
    """
    prm = dirlik_parameters(mom) if params is None else params
    scale = 2.0 * mom.sigma
    term1 = prm.D1 * prm.Q ** m * special.gamma(m + 1.0)
    term2 = (prm.D2 * (np.sqrt(2.0) * prm.R) ** m
             + prm.D3 * 2.0 ** (m / 2.0)) * special.gamma(1.0 + 0.5 * m)
    return float(scale ** m * (term1 + term2))


def dirlik_damage_rate(mom: SpectralMoments, C: float = 1.0, m: float = 3.0,
                       alpha2_rayleigh_tol: float = 1e-9) -> float:
    """Dirlik damage rate per second: ``D = (nup / C) * E[S^m]``.

    ``nup = sqrt(m4/m2)`` is the peak rate (Dirlik's cycle rate).  For
    ``1 - alpha2 <= alpha2_rayleigh_tol`` the coefficients are singular and the
    Rayleigh closed form is returned instead - the exact limit of the model.
    """
    _check_sn(C, m)
    if 1.0 - mom.alpha2 <= alpha2_rayleigh_tol:
        return rayleigh_damage_rate(mom, C, m)
    return float(mom.nup / C * dirlik_expected_range_power(mom, m))


def tovo_benasciutti_weight(mom: SpectralMoments) -> float:
    """Tovo-Benasciutti weight ``b = clip((a1 - a2)/(1 - a1), 0, 1)``."""
    a1, a2 = mom.alpha1, mom.alpha2
    if 1.0 - a1 < 1e-12:
        return 0.0
    return float(np.clip((a1 - a2) / (1.0 - a1), 0.0, 1.0))


def benasciutti_tovo_damage_rate(mom: SpectralMoments, C: float = 1.0,
                                 m: float = 3.0, b: float | None = None) -> float:
    """Tovo-Benasciutti bandwidth-corrected damage rate per second.

    ``D_BT = lambda * D_R`` with ``lambda = b + (1-b) alpha2^(m-1)``.  The
    correction is constructed so that ``lambda -> 1`` as ``alpha2 -> 1``
    (narrow-band limit is reproduced exactly) and ``lambda -> b`` for a very
    wide-band process.  ``b`` defaults to :func:`tovo_benasciutti_weight`.
    """
    _check_sn(C, m)
    bb = tovo_benasciutti_weight(mom) if b is None else float(np.clip(b, 0.0, 1.0))
    lam = bb + (1.0 - bb) * mom.alpha2 ** (m - 1.0)
    return float(lam * rayleigh_damage_rate(mom, C, m))


def bimodal_analytic_damage_rate(low: _Spectrum, high: _Spectrum,
                                 C: float = 1.0, m: float = 3.0,
                                 f_lo_low: float | None = None,
                                 f_hi_high: float | None = None) -> Dict[str, float]:
    """Combination/separation limit for a strongly bimodal spectrum.

    When the two components are well separated (``f2/f1 >= 5``) each narrow-band
    component contributes its own rainflow cycles and the total damage tends to

        ``D -> D1 + D2``,  ``Di = nu0_i/C * (2 sqrt(2) sigma_i)^m Gamma(1+m/2)``

    with ``sigma_i^2 = m0_i`` and ``nu0_i = m1_i/m0_i`` of that component alone
    (Torhaug 1990; Jiao & Moan 1990).  Returns the two component rates plus the
    sum and the frequency separation ratio.
    """
    _check_sn(C, m)
    mom_low = low.moments_object(f_lo_low, f_hi_high)
    mom_high = high.moments_object(f_lo_low, f_hi_high)
    d1 = rayleigh_damage_rate(mom_low, C, m)
    d2 = rayleigh_damage_rate(mom_high, C, m)
    spec = BimodalSpectrum(low, high)
    return {
        "D_low": d1,
        "D_high": d2,
        "D_total": float(d1 + d2),
        "separation_ratio": spec.separation_ratio(),
        "sigma_low": mom_low.sigma,
        "sigma_high": mom_high.sigma,
        "nu0_low_hz": mom_low.nu0,
        "nu0_high_hz": mom_high.nu0,
        "low_m0": mom_low.m0,
        "high_m0": mom_high.m0,
    }


# ---------------------------------------------------------------------------
# Time-domain leg: PSD -> Gaussian history -> rainflow -> Miner
# ---------------------------------------------------------------------------

def discrete_psd_grid(psd, n_samples: int, fs: float, f_lo: float = 0.0,
                      f_hi: float | None = None):
    """PSD sampled on the rFFT grid of a record of ``n_samples`` at rate ``fs``.

    Returns ``(freqs, s_vals, df)`` with ``df = fs/n_samples`` and every bin
    outside ``[f_lo, f_hi]`` set to zero, i.e. exactly the band-limited spectrum
    that :func:`synthesize_gaussian_history` realises.  Keeping this array and
    integrating the *same* band-limited PSD for the frequency-domain models is
    what makes the spectral vs time-domain comparison a like-for-like test
    (the untruncated tail would otherwise inflate m4 and the peak rate).
    """
    if n_samples < 8 or fs <= 0:
        raise ValueError("need n_samples >= 8 and fs > 0")
    if np.isscalar(f_lo):
        pass
    df = fs / n_samples
    freqs = np.arange(n_samples // 2 + 1, dtype=float) * df
    if hasattr(psd, "band") and callable(psd):
        hi_default = f_hi if f_hi is not None else psd.band()[1]
        s_vals = np.asarray(psd(freqs), dtype=float)
    else:
        s_vals = np.asarray(psd, dtype=float)
        if s_vals.shape != freqs.shape:
            raise ValueError(
                f"PSD array must have {freqs.size} samples (= n_samples//2+1)")
        hi_default = f_hi if f_hi is not None else freqs[-1]
    hi = float(hi_default) if f_hi is None else float(f_hi)
    mask = (freqs >= float(f_lo)) & (freqs <= hi)
    s_band = np.where(mask, s_vals, 0.0)
    s_band[0] = 0.0
    if n_samples % 2 == 0:
        s_band[-1] = 0.0  # Nyquist bin: drop to avoid a cosine at fs/2
    return freqs, s_band, df


def synthesize_gaussian_history(psd, n_samples: int, fs: float, seed: int = 0,
                                f_lo: float = 0.0, f_hi: float | None = None):
    """Random-phase synthesis of a stationary Gaussian stress history.

    ``x(t) = sum_k sqrt(2 S(f_k) df) cos(2 pi f_k t + phi_k)`` with i.i.d.
    uniform phases generated from ``numpy.random.default_rng(seed)``; implemented
    with an inverse rFFT so that a 1e6-sample record costs O(N log N).

    The variance of the realisation tends to ``m0`` of the band-limited PSD
    (chi-square scatter of 1/sqrt(N_eff) around unity, N_eff being the number of
    effective independent cycles).

    Parameters
    ----------
    psd : callable or array_like
        One-sided PSD as a function of f, or already sampled on the rFFT grid
        (length ``n_samples//2 + 1``).
    n_samples, fs, seed, f_lo, f_hi
        Record length, sampling rate (Hz), RNG seed and band limits.  ``f_hi``
        defaults to the spectrum's own band; ``fs/2`` must be >= ``f_hi``.

    Returns
    -------
    x : ndarray, shape (n_samples,)
        Stress history (zero mean) sampled at ``dt = 1/fs``.
    """
    freqs, s_band, df = discrete_psd_grid(psd, n_samples, fs, f_lo, f_hi)
    if np.any(s_band > 0) and freqs[np.argmax(s_band > 0)] > fs / 2.0:
        raise ValueError("f_hi must not exceed the Nyquist frequency fs/2")
    rng = np.random.default_rng(seed)
    amp = np.sqrt(2.0 * s_band * df)
    phases = rng.uniform(0.0, 2.0 * np.pi, freqs.size)
    spec = 0.5 * n_samples * amp * np.exp(1j * phases)
    spec[0] = 0.0
    x = np.fft.irfft(spec, n=n_samples)
    return x


def time_domain_damage_rate(history, dt: float, C: float = 1.0, m: float = 3.0,
                            method: str = "stack") -> Dict[str, float]:
    """Rainflow + Palmgren-Miner damage rate of a stress history, per second.

    ``D = 1/(C T) sum_i n_i S_i^m`` with ``T = len(x) dt`` and rainflow cycles
    from the ASTM E1049-85 rule (``method='stack'`` uses the O(N) stack
    implementation, ``'reference'`` the list-based reference implementation;
    both give identical counts - see the test suite).
    """
    _check_sn(C, m)
    x = np.asarray(history, dtype=float)
    if method == "stack":
        rf = rainflow_counting_stack(x)
    elif method == "reference":
        rf = rainflow_counting(x)
    else:
        raise ValueError("method must be 'stack' or 'reference'")
    total_time = x.size * float(dt)
    if rf.ranges.size == 0:
        return {"damage_rate": 0.0, "expected_range_power": 0.0,
                "n_cycles": 0.0, "range_max": 0.0,
                "mean_upcrossing_hz": 0.0, "duration_s": total_time}
    ranges = np.asarray(rf.ranges, dtype=float)
    counts = np.asarray(rf.counts, dtype=float)
    n_cycles = float(counts.sum())
    damage = float(np.sum(counts * ranges ** m) / C)
    return {
        "damage_rate": damage / total_time,
        "expected_range_power": damage / n_cycles if n_cycles > 0 else 0.0,
        "n_cycles": n_cycles,
        "range_max": float(ranges.max()),
        "mean_upcrossing_hz": n_cycles / total_time,
        "duration_s": total_time,
    }


def monte_carlo_damage_rate(psd, n_samples: int, fs: float, seeds: Sequence[int],
                            C: float = 1.0, m: float = 3.0, f_lo: float = 0.0,
                            f_hi: float | None = None) -> Dict[str, object]:
    """Time-domain damage rate averaged over several random-phase realisations.

    Returns the per-seed damage rates, their mean / standard deviation, the
    number of rainflow cycles per realisation and the moments of the *band
    limited* PSD that the realisations were drawn from (so the spectral models
    can be evaluated on exactly the same m0..m4).
    """
    freqs, s_band, df = discrete_psd_grid(psd, n_samples, fs, f_lo, f_hi)
    mom = moments_from_dict(spectral_moments(freqs, s_band), 0.0, freqs[-1])
    rates = []
    cycles = []
    for seed in seeds:
        x = synthesize_gaussian_history(s_band, n_samples, fs, seed=seed)
        res = time_domain_damage_rate(x, 1.0 / fs, C=C, m=m)
        rates.append(res["damage_rate"])
        cycles.append(res["n_cycles"])
    rates_a = np.asarray(rates, dtype=float)
    n = rates_a.size
    return {
        "damage_rate_mean": float(rates_a.mean()),
        "damage_rate_std": float(rates_a.std(ddof=1)) if n > 1 else 0.0,
        "damage_rate_stderr": float(rates_a.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
        "damage_rates": rates_a.tolist(),
        "cycles_per_realisation": cycles,
        "n_samples": int(n_samples),
        "seeds": [int(s) for s in seeds],
        "sample_interval_s": 1.0 / fs,
        "fs_hz": float(fs),
        "f_lo_hz": float(f_lo),
        "f_hi_hz": float(freqs[-1]) if f_hi is None else float(f_hi),
        "moments": mom,
        "moments_dict": mom.as_dict(),
        "df_hz": float(df),
    }


# ---------------------------------------------------------------------------
# Sensitivity study
# ---------------------------------------------------------------------------

def sensitivity_table(mom: SpectralMoments, C: float = 1.0, m: float = 3.0,
                      m_span: float = 1.0, alpha2_step: float = 0.05,
                      f_hi_factors: Sequence[float] = (2.0, 4.0, 8.0, 16.0, 32.0),
                      f_hi_base: float | None = None,
                      f_hi_provider: Callable[[float], float] | None = None) -> list:
    """First-order sensitivity of the Dirlik damage rate to (m, alpha2, f_hi).

    Three axes are perturbed independently around a baseline moment set:

    * S-N slope ``m -> m +- m_span`` (central difference).
    * bandwidth ``alpha2``: m2 is rescaled so that ``alpha2 = m2/sqrt(m0 m4)``
      moves by ``alpha2_step`` at fixed m0, m1 and m4.  This is a *shape*
      perturbation ("what a broader peak with the same variance and the same
      tail would do"); it is not a full re-integration of a spectrum family.
    * PSD cut-off ``f_hi``: only m4 changes.  With ``f_hi_provider`` the m4 of
      the perturbed band is recomputed exactly (recommended: pass
      ``lambda fh: spectrum.moments((4,), 0.0, fh)[4.0]``); otherwise a linear
      tail model ``m4(f) = m4 + (m4/f_hi_ref)(f - f_hi_ref)`` is used, which is
      first-order accurate only.

    Each row reports the damage rate, its relative change against the baseline
    and the elasticity ``d ln D / d ln x``; rows are sorted by ``|elasticity|``
    (largest first), which is the sensitivity ranking.
    """
    base = dirlik_damage_rate(mom, C, m)
    rows = []

    def _elastic(x0, x1, d0, d1):
        if x0 <= 0 or d0 <= 0 or x1 <= 0 or d1 <= 0:
            return float("nan")
        denom = np.log(x1) - np.log(x0)
        if abs(denom) < 1e-15:
            return 0.0
        return float((np.log(d1) - np.log(d0)) / denom)

    # --- S-N slope ---------------------------------------------------------
    for mm in (m - m_span, m + m_span):
        d = dirlik_damage_rate(mom, C, mm)
        rows.append({
            "parameter": "sn_slope_m",
            "baseline": float(m), "varied": float(mm),
            "damage_rate": d,
            "rel_change": float(d / base - 1.0),
            "elasticity": _elastic(m, mm, base, d),
        })

    # --- bandwidth alpha2 (fixed m0, m1, m4) ------------------------------
    for da in (-alpha2_step, alpha2_step):
        a2_new = float(np.clip(mom.alpha2 + da, 1e-3, 1.0 - 1e-6))
        m2_new = a2_new * np.sqrt(mom.m0 * mom.m4)
        if m2_new <= 0:
            continue
        alt = SpectralMoments((mom.m0, mom.m1, m2_new, mom.m3, mom.m4),
                              mom.f_lo, mom.f_hi)
        try:
            d = dirlik_damage_rate(alt, C, m)
        except ValueError:
            continue
        rows.append({
            "parameter": "alpha2",
            "baseline": mom.alpha2, "varied": a2_new,
            "damage_rate": d,
            "alpha2_varied": alt.alpha2,
            "nup_varied": alt.nup,
            "m4_varied": alt.m4,
            "rel_change": float(d / base - 1.0),
            "elasticity": _elastic(mom.alpha2, a2_new, base, d),
        })

    # --- PSD cut-off (m4 only) --------------------------------------------
    f_hi_ref = float(f_hi_base if f_hi_base is not None else
                     (mom.f_hi if np.isfinite(mom.f_hi) else 20.0 * mom.nup))
    tail_per_hz = mom.m4 / f_hi_ref  # d m4 / d f_hi at the baseline cut-off
    for fac in f_hi_factors:
        f_new = fac * f_hi_ref
        if f_hi_provider is not None:
            m4_new = float(f_hi_provider(f_new))
        else:
            m4_new = mom.m4 + tail_per_hz * (f_new - f_hi_ref)
        alt = SpectralMoments((mom.m0, mom.m1, mom.m2, mom.m3, m4_new),
                              mom.f_lo, f_new)
        try:
            d = dirlik_damage_rate(alt, C, m)
        except ValueError:
            continue
        rows.append({
            "parameter": "cutoff_f_hi_hz",
            "baseline": f_hi_ref, "varied": f_new,
            "damage_rate": d,
            "alpha2_varied": alt.alpha2,
            "nup_varied": alt.nup,
            "m4_varied": m4_new,
            "rel_change": float(d / base - 1.0),
            "elasticity": _elastic(f_hi_ref, f_new, base, d),
        })

    rows.sort(key=lambda r: -abs(r["elasticity"]) if np.isfinite(r["elasticity"]) else 0.0)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


# ---------------------------------------------------------------------------
# Public assessment facade (JSON-ready, for the ship platform)
# ---------------------------------------------------------------------------

def spectral_fatigue_assessment(f, s, C: float = 1.0, m: float = 3.0,
                                bimodal_split_hz: float | None = None) -> Dict[str, object]:
    """Full spectral fatigue assessment of a one-sided stress PSD.

    Parameters
    ----------
    f, s : array_like
        Frequency axis (Hz, ascending) and one-sided PSD of the stress at the
        hotspot.  Frequencies outside the delivered table are assumed to carry
        no energy - quote the band with the result (m4 depends on it).
    C, m : float
        Single-slope S-N law ``N = C S^-m``.
    bimodal_split_hz : float, optional
        If given, the band is split at this frequency and the two sub-processes
        are each evaluated with the narrow-band Rayleigh model; their sum is the
        combination/separation limit used as the bimodal anchor.

    Returns
    -------
    dict
        Moments, bandwidth descriptors, cycle rates and the damage rates of all
        four models, plus ``flags`` describing applicability (narrow-band,
        bimodal, m4 tail sensitivity).
    """
    fa = np.asarray(f, dtype=float)
    sa = np.asarray(s, dtype=float)
    mom = moments_from_dict(spectral_moments(fa, sa), float(fa[0]), float(fa[-1]))
    out: Dict[str, object] = {
        "sn": {"C": float(C), "m": float(m)},
        "band": {"f_lo_hz": float(fa[0]), "f_hi_hz": float(fa[-1]),
                 "n_points": int(fa.size), "df_hz": float(np.mean(np.diff(fa)))},
        "moments": mom.as_dict(),
        "cycles": {"nu0_hz": mom.nu0, "nup_hz": mom.nup},
        "damage_rate_per_s": {},
        "expected_range_power": {},
    }
    rates = out["damage_rate_per_s"]
    esm = out["expected_range_power"]
    rates["rayleigh_narrowband"] = rayleigh_damage_rate(mom, C, m)
    esm["rayleigh_narrowband"] = rayleigh_expected_range_power(mom, m)
    try:
        prm = dirlik_parameters(mom)
        rates["dirlik"] = dirlik_damage_rate(mom, C, m)
        esm["dirlik"] = dirlik_expected_range_power(mom, m, prm)
        out["dirlik_parameters"] = prm.as_dict()
    except ValueError:
        rates["dirlik"] = rates["rayleigh_narrowband"]
        prm = None
    rates["tovo_benasciutti"] = benasciutti_tovo_damage_rate(mom, C, m)
    out["tovo_benasciutti_weight_b"] = tovo_benasciutti_weight(mom)

    flags = []
    if mom.alpha2 > 0.99:
        flags.append("narrow_band: all models coincide (alpha2 > 0.99)")
    if mom.alpha2 < 0.90:
        flags.append("wide_band: narrow-band Rayleigh is not applicable (alpha2 < 0.9)")
    if np.isfinite(mom.f_hi) and mom.f_hi > 0:
        flags.append(
            "m4 is band limited at f_hi = %.4g Hz; the f^-5 tail beyond it is not "
            "represented" % mom.f_hi)

    if bimodal_split_hz is not None:
        split = float(bimodal_split_hz)
        low_mask = fa <= split
        high_mask = fa > split
        if low_mask.sum() < 2 or high_mask.sum() < 2:
            raise ValueError("bimodal_split_hz must leave >= 2 points on each side")
        m_low = moments_from_dict(spectral_moments(fa[low_mask], sa[low_mask],
                                                   (0, 1, 2, 3, 4)))
        m_high = moments_from_dict(spectral_moments(fa[high_mask], sa[high_mask],
                                                    (0, 1, 2, 3, 4)))
        d_low = rayleigh_damage_rate(m_low, C, m)
        d_high = rayleigh_damage_rate(m_high, C, m)
        rates["bimodal_combination_limit"] = float(d_low + d_high)
        out["bimodal"] = {
            "split_hz": split,
            "D_low": d_low, "D_high": d_high, "D_total": float(d_low + d_high),
            "m0_low": m_low.m0, "m0_high": m_high.m0,
            "nu0_low_hz": m_low.nu0, "nu0_high_hz": m_high.nu0,
            "sigma_low": m_low.sigma, "sigma_high": m_high.sigma,
        }
        flags.append("bimodal split at %.4g Hz applied for the combination limit" % split)

    out["flags"] = flags
    return out
