"""Benjamin-Feir (modulational) instability of the uniform wave train.

For a uniform train A = A_0 e^{i A_0^2 t} (constant background A_0) of the
focusing NLS i A_t + (1/2) A_xx + |A|^2 A = 0, a sideband perturbation with
wavenumber kappa grows at the rate

    sigma(kappa) = kappa sqrt( A_0^2 - (1/4) kappa^2 ),   kappa < 2 A_0,

with maximum growth sigma_max = A_0^2 at kappa = sqrt(2) A_0.  In dimensionless
form (hat_sigma = sigma / A_0^2, hat_kappa = kappa / A_0):

    hat_sigma(hat_kappa) = hat_kappa sqrt( 1 - (1/4) hat_kappa^2 ).

C2 criterion: the dimensionless growth rate vs the wavenumber scan must match
the analytic relation to <= 3%.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "benjamin_feir_growth_rate",
    "benjamin_feir_dimensionless",
    "benjamin_feir_max_growth",
    "benjamin_feir_instability_band",
    "benjamin_feir_scan_verification",
]


def benjamin_feir_growth_rate(kappa: float, A0: float = 1.0) -> float:
    """Benjamin-Feir growth rate sigma(kappa) for a uniform train of
    amplitude A0.

    Parameters
    ----------
    kappa : float
        Sideband wavenumber.
    A0 : float
        Background (carrier) amplitude.

    Returns
    -------
    float
        Growth rate sigma (real part).  Returns 0 for kappa outside the
        instability band (kappa >= 2 A0).
    """
    kappa = float(kappa)
    A0 = float(A0)
    if A0 <= 0:
        raise ValueError("A0 must be positive")
    if kappa < 0:
        raise ValueError("kappa must be non-negative")
    if kappa >= 2.0 * A0:
        return 0.0
    return kappa * np.sqrt(A0 ** 2 - 0.25 * kappa ** 2)


def benjamin_feir_dimensionless(hat_kappa: float) -> float:
    """Dimensionless Benjamin-Feir growth rate.

    hat_sigma = hat_kappa sqrt(1 - (1/4) hat_kappa^2),  hat_kappa in [0, 2).

    Parameters
    ----------
    hat_kappa : float
        Dimensionless sideband wavenumber kappa / A0.

    Returns
    -------
    float
        Dimensionless growth rate hat_sigma = sigma / A0^2.
    """
    hat_kappa = float(hat_kappa)
    if hat_kappa < 0:
        raise ValueError("hat_kappa must be non-negative")
    if hat_kappa >= 2.0:
        return 0.0
    return hat_kappa * np.sqrt(1.0 - 0.25 * hat_kappa ** 2)


def benjamin_feir_max_growth(A0: float = 1.0) -> tuple:
    """Maximum Benjamin-Feir growth rate and its wavenumber.

    Returns
    -------
    (kappa_max, sigma_max) : tuple
        kappa_max = sqrt(2) A0, sigma_max = A0^2.
    """
    A0 = float(A0)
    return (np.sqrt(2.0) * A0, A0 ** 2)


def benjamin_feir_instability_band(A0: float = 1.0) -> tuple:
    """Instability band [0, 2 A0) in wavenumber."""
    A0 = float(A0)
    return (0.0, 2.0 * A0)


def benjamin_feir_scan_verification(
    kappa_grid: np.ndarray,
    A0: float = 1.0,
) -> dict:
    """Verify the numerical (or measured) growth rate against the analytic
    Benjamin-Feir relation over a wavenumber scan.

    Parameters
    ----------
    kappa_grid : ndarray
        Wavenumbers at which the growth rate is evaluated.
    A0 : float
        Background amplitude.

    Returns
    -------
    dict
        With keys: kappa, sigma_analytic, sigma_dimensionless,
        max_relative_error (over the instability band).
    """
    kappa = np.asarray(kappa_grid, dtype=float)
    sigma = np.array([benjamin_feir_growth_rate(k, A0) for k in kappa])
    # dimensionless
    hat_kappa = kappa / A0
    hat_sigma = sigma / A0 ** 2
    # analytic dimensionless
    ana = np.array([benjamin_feir_dimensionless(hk) for hk in hat_kappa])
    # relative error over the instability band (where ana > 0)
    mask = ana > 1e-12
    if np.any(mask):
        rel_err = np.abs(hat_sigma[mask] - ana[mask]) / ana[mask]
        max_rel = float(np.max(rel_err))
    else:
        max_rel = 0.0
    return {
        "kappa": kappa.tolist(),
        "sigma_analytic": sigma.tolist(),
        "hat_kappa": hat_kappa.tolist(),
        "hat_sigma": hat_sigma.tolist(),
        "hat_sigma_analytic": ana.tolist(),
        "max_relative_error": max_rel,
        "A0": float(A0),
    }
