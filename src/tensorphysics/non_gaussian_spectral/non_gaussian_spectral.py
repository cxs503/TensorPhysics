"""Non-Gaussian spectral fatigue damage methods.

Extends the Gaussian spectral methods (Dirlik, Benasciutti-Tovo, Rayleigh)
to non-Gaussian processes via the Winterstein moment-translation.

Key idea: For a non-Gaussian process with moments (γ3, γ4), the range
distribution is modified. The Winterstein transform maps the Gaussian
range distribution to the non-Gaussian one, and we compute the expected
range power E[S^m] under the transformed distribution.

Methods implemented:
1. Non-Gaussian Rayleigh: Rayleigh PDF modified by Winterstein transform
2. Non-Gaussian Dirlik: Dirlik's three-component PDF with NG correction
3. Non-Gaussian Benasciutti-Tovo: BT width correction with NG factor
4. Winterstein-corrected Rayleigh: Simple scaling of Gaussian Rayleigh
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy import integrate, special

from ..fatigue.spectral import (
    SpectralMoments,
    moments_from_dict,
    spectral_moments,
    rayleigh_damage_rate,
    dirlik_damage_rate,
    benasciutti_tovo_damage_rate,
)
from .winterstein import WintersteinTransform, winterstein_hermite_transform

__all__ = [
    "NonGaussianSpectralMoments",
    "non_gaussian_dirlik_damage_rate",
    "non_gaussian_rayleigh_damage_rate",
    "non_gaussian_bt_damage_rate",
    "winterstein_corrected_rayleigh",
    "spectral_damage_non_gaussian",
]


@dataclass(frozen=True)
class NonGaussianSpectralMoments:
    """Spectral moments plus non-Gaussian descriptors.

    Attributes
    ----------
    spectral : SpectralMoments
        Underlying Gaussian spectral moments.
    gamma3 : float
        Skewness of the non-Gaussian process.
    gamma4 : float
        Kurtosis of the non-Gaussian process (γ4=3 for Gaussian).
    """
    spectral: SpectralMoments
    gamma3: float
    gamma4: float

    @property
    def excess_kurtosis(self) -> float:
        return self.gamma4 - 3.0

    @property
    def alpha2(self) -> float:
        return self.spectral.alpha2

    @property
    def sigma(self) -> float:
        return self.spectral.sigma

    @property
    def nu0(self) -> float:
        return self.spectral.nu0

    @property
    def nup(self) -> float:
        return self.spectral.nup


def _winterstein_range_correction_factor(
    gamma3: float,
    gamma4: float,
    m: float,
    n_quad: int = 48,
) -> float:
    """Compute the correction factor for expected range power under Winterstein.

    For a Gaussian process, the peak distribution is Rayleigh and
    E[S^m] = (2*sqrt(2)*σ)^m * Γ(1 + m/2).

    For a non-Gaussian process with Winterstein transform h3(u),
    the peak distribution is modified. We compute:
        E[S_NG^m] / E[S_G^m]

    using Gauss-Hermite quadrature on the transformed variable.

    Parameters
    ----------
    gamma3, gamma4 : float
        Skewness and kurtosis of the non-Gaussian process.
    m : float
        S-N curve exponent.
    n_quad : int
        Number of quadrature points.

    Returns
    -------
    float
        Correction factor (1.0 for Gaussian).
    """
    if abs(gamma3) < 1e-10 and abs(gamma4 - 3.0) < 1e-10:
        return 1.0

    tfm = winterstein_hermite_transform(gamma3, gamma4)

    # Gauss-Hermite quadrature for E[|h3(u)|^m] where u ~ N(0,1)
    # The range is proportional to |peak|, so we compute E[|h3(u)|^m]
    nodes, weights = np.polynomial.hermite.hermgauss(n_quad)
    u = nodes * np.sqrt(2.0)
    w = weights / np.sqrt(np.pi)

    h3 = tfm.forward(u)
    # Range ~ |h3|, so E[|h3|^m]
    E_range_m_ng = np.sum(w * np.abs(h3)**m)

    # Gaussian reference: E[|u|^m] for u ~ N(0,1)
    # = 2^(m/2) * Γ((m+1)/2) / sqrt(π)
    E_range_m_g = 2.0**(m / 2.0) * special.gamma((m + 1.0) / 2.0) / np.sqrt(np.pi)

    if E_range_m_g < 1e-300:
        return 1.0

    return E_range_m_ng / E_range_m_g


def non_gaussian_rayleigh_damage_rate(
    ng_moments: NonGaussianSpectralMoments,
    C: float,
    m: float,
) -> float:
    """Non-Gaussian Rayleigh narrow-band damage rate.

    Modifies the Gaussian Rayleigh result by the Winterstein range correction.

    D_NG = D_G * correction_factor

    Parameters
    ----------
    ng_moments : NonGaussianSpectralMoments
        Spectral moments and non-Gaussian descriptors.
    C, m : float
        S-N law parameters: N = C * S^(-m).
    """
    D_gaussian = rayleigh_damage_rate(ng_moments.spectral, C, m)
    correction = _winterstein_range_correction_factor(
        ng_moments.gamma3, ng_moments.gamma4, m
    )
    return D_gaussian * correction


def non_gaussian_dirlik_damage_rate(
    ng_moments: NonGaussianSpectralMoments,
    C: float,
    m: float,
) -> float:
    """Non-Gaussian Dirlik damage rate.

    Dirlik's method assumes a specific range PDF (mixture of exponential and
    Rayleigh). For non-Gaussian processes, we apply the Winterstein correction
    to the expected range power.

    D_NG = D_Dirlik_G * correction_factor
    """
    D_gaussian = dirlik_damage_rate(ng_moments.spectral, C, m)
    correction = _winterstein_range_correction_factor(
        ng_moments.gamma3, ng_moments.gamma4, m
    )
    return D_gaussian * correction


def non_gaussian_bt_damage_rate(
    ng_moments: NonGaussianSpectralMoments,
    C: float,
    m: float,
) -> float:
    """Non-Gaussian Benasciutti-Tovo damage rate.

    BT method uses a bandwidth-dependent weighting between narrow-band and
    wide-band limits. For non-Gaussian, we apply the Winterstein correction.

    D_NG = D_BT_G * correction_factor
    """
    D_gaussian = benasciutti_tovo_damage_rate(ng_moments.spectral, C, m)
    correction = _winterstein_range_correction_factor(
        ng_moments.gamma3, ng_moments.gamma4, m
    )
    return D_gaussian * correction


def winterstein_corrected_rayleigh(
    spectral_mom: SpectralMoments,
    gamma3: float,
    gamma4: float,
    C: float,
    m: float,
) -> float:
    """Convenience function: Rayleigh damage with Winterstein non-Gaussian correction.

    Parameters
    ----------
    spectral_mom : SpectralMoments
        Gaussian spectral moments.
    gamma3, gamma4 : float
        Non-Gaussian skewness and kurtosis.
    C, m : float
        S-N law parameters.

    Returns
    -------
    float
        Damage rate (per second).
    """
    ng = NonGaussianSpectralMoments(spectral_mom, gamma3, gamma4)
    return non_gaussian_rayleigh_damage_rate(ng, C, m)


def spectral_damage_non_gaussian(
    f: np.ndarray,
    psd: np.ndarray,
    gamma3: float,
    gamma4: float,
    C: float = 1.0,
    m: float = 3.0,
    f_hi: Optional[float] = None,
) -> Dict[str, float]:
    """Full non-Gaussian spectral fatigue assessment.

    Parameters
    ----------
    f, psd : array_like
        Frequency (Hz) and one-sided PSD.
    gamma3, gamma4 : float
        Skewness and kurtosis of the non-Gaussian stress process.
    C, m : float
        S-N law: N = C * S^(-m).
    f_hi : float, optional
        Upper frequency bound for m4 integration.

    Returns
    -------
    dict
        Damage rates for each method (Gaussian and non-Gaussian).
    """
    fa = np.asarray(f, dtype=float)
    sa = np.asarray(psd, dtype=float)

    # Compute spectral moments
    mom_dict = spectral_moments(fa, sa)
    f_lo = float(fa[0])
    f_hi_val = float(f_hi if f_hi is not None else fa[-1])
    spec_mom = moments_from_dict(mom_dict, f_lo=f_lo, f_hi=f_hi_val)

    ng_mom = NonGaussianSpectralMoments(spec_mom, gamma3, gamma4)

    # Gaussian damage rates
    D_rayleigh_g = rayleigh_damage_rate(spec_mom, C, m)
    D_dirlik_g = dirlik_damage_rate(spec_mom, C, m)
    D_bt_g = benasciutti_tovo_damage_rate(spec_mom, C, m)

    # Non-Gaussian damage rates
    D_rayleigh_ng = non_gaussian_rayleigh_damage_rate(ng_mom, C, m)
    D_dirlik_ng = non_gaussian_dirlik_damage_rate(ng_mom, C, m)
    D_bt_ng = non_gaussian_bt_damage_rate(ng_mom, C, m)

    # Correction factor
    correction = _winterstein_range_correction_factor(gamma3, gamma4, m)

    return {
        "moments": {
            "m0": spec_mom.m0, "m1": spec_mom.m1, "m2": spec_mom.m2,
            "m3": spec_mom.m3, "m4": spec_mom.m4,
            "alpha2": spec_mom.alpha2, "nu0_hz": spec_mom.nu0, "nup_hz": spec_mom.nup,
        },
        "non_gaussian": {"gamma3": gamma3, "gamma4": gamma4},
        "sn": {"C": C, "m": m},
        "winterstein_correction_factor": correction,
        "damage_rate_per_s": {
            "rayleigh_gaussian": D_rayleigh_g,
            "dirlik_gaussian": D_dirlik_g,
            "bt_gaussian": D_bt_g,
            "rayleigh_non_gaussian": D_rayleigh_ng,
            "dirlik_non_gaussian": D_dirlik_ng,
            "bt_non_gaussian": D_bt_ng,
        },
    }
