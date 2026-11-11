"""Second-order spectral-synthesis crest vs the Peregrine breather peak crest
(C4).

For a given sea state (H_s, T_p), the *first-order* (linear) spectral synthesis
predicts a crest at the Rayleigh mode

    eta_1st = H_s / 2,

and the *second-order* (Stokes bound-harmonic) correction adds

    eta_2nd = eta_1st + (1/2) k_p eta_1st^2,

with k_p = (2 pi / T_p)^2 / g the peak wavenumber.

The Peregrine breather is a rogue wave that emerges from a background of
amplitude A_0; its peak crest is 3 A_0.  For a narrow-band sea state the
background amplitude is A_0 = H_s / (2 sqrt(2)) (since H_s = 4 sigma and
sigma = A_0 / sqrt(2) for a monochromatic carrier).

C4 criterion: the ratio of the breather peak crest to the second-order crest
is reported with an uncertainty band -- no one-sided verdict.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "linear_crest",
    "second_order_crest",
    "second_order_spectral_synthesis",
    "breather_vs_second_order",
    "crest_ratio_with_uncertainty",
]

GRAVITY = 9.81


def linear_crest(H_s: float) -> float:
    """First-order (Rayleigh-mode) crest: H_s / 2."""
    return float(H_s) / 2.0


def second_order_crest(H_s: float, T_p: float) -> float:
    """Second-order (Stokes bound-harmonic) crest.

    eta_2nd = H_s/2 + (1/2) k_p (H_s/2)^2,  k_p = (2 pi / T_p)^2 / g.
    """
    H_s = float(H_s)
    T_p = float(T_p)
    if T_p <= 0:
        raise ValueError("T_p must be positive")
    k_p = (2.0 * np.pi / T_p) ** 2 / GRAVITY
    eta_1 = H_s / 2.0
    return eta_1 + 0.5 * k_p * eta_1 ** 2


def second_order_spectral_synthesis(H_s: float, T_p: float) -> dict:
    """Full second-order spectral synthesis crest report for a sea state."""
    eta_1 = linear_crest(H_s)
    eta_2 = second_order_crest(H_s, T_p)
    k_p = (2.0 * np.pi / T_p) ** 2 / GRAVITY
    return {
        "H_s": float(H_s),
        "T_p": float(T_p),
        "k_p": float(k_p),
        "eta_first_order": float(eta_1),
        "eta_second_order": float(eta_2),
        "second_order_increment": float(eta_2 - eta_1),
    }


def _background_amplitude(H_s: float) -> float:
    """Background (carrier) amplitude A_0 for a narrow-band sea state.

    H_s = 4 sigma, sigma = A_0 / sqrt(2)  =>  A_0 = H_s / (2 sqrt(2)).
    """
    return float(H_s) / (2.0 * np.sqrt(2.0))


def breather_vs_second_order(H_s: float, T_p: float) -> dict:
    """Compare the Peregrine breather peak crest with the second-order crest
    for the same (H_s, T_p) sea state.

    Parameters
    ----------
    H_s : float
        Significant wave height (m).
    T_p : float
        Peak period (s).

    Returns
    -------
    dict
        With keys: H_s, T_p, A0 (background), eta_first_order,
        eta_second_order, eta_breather (3 A0), ratio, and the
        second-order-increment fraction.
    """
    A0 = _background_amplitude(H_s)
    eta_1 = linear_crest(H_s)
    eta_2 = second_order_crest(H_s, T_p)
    eta_b = 3.0 * A0
    ratio = eta_b / eta_2 if eta_2 > 0 else float("inf")
    return {
        "H_s": float(H_s),
        "T_p": float(T_p),
        "A0": float(A0),
        "eta_first_order": float(eta_1),
        "eta_second_order": float(eta_2),
        "eta_breather": float(eta_b),
        "ratio_breather_over_second_order": float(ratio),
        "second_order_increment_fraction": float((eta_2 - eta_1) / eta_1),
    }


def crest_ratio_with_uncertainty(H_s: float, T_p: float, n_ens: int = 200) -> dict:
    """Crest ratio with an uncertainty band from a Monte-Carlo ensemble of the
    second-order crest.

    The second-order crest depends on the *actual* peak wavenumber k_p, which
    for a real sea state is a random variable (the peak period T_p is a
    spectral estimate with finite-bandwidth scatter).  We draw T_p from a
    log-normal band around the nominal value and report the mean ratio and its
    +/- 1-sigma band.

    Parameters
    ----------
    H_s : float
        Significant wave height (m).
    T_p : float
        Nominal peak period (s).
    n_ens : int
        Ensemble size.

    Returns
    -------
    dict
        With keys: ratio_mean, ratio_std, ratio_band (lo, hi),
        eta_breather, and the ensemble.
    """
    rng = np.random.default_rng(12345)
    # log-normal scatter of T_p around the nominal (10% band)
    sigma_log = 0.10
    T_p_ens = T_p * np.exp(rng.normal(0.0, sigma_log, n_ens))
    A0 = _background_amplitude(H_s)
    eta_b = 3.0 * A0
    ratios = np.array([eta_b / second_order_crest(H_s, tp) for tp in T_p_ens])
    return {
        "ratio_mean": float(np.mean(ratios)),
        "ratio_std": float(np.std(ratios)),
        "ratio_band": (float(np.mean(ratios) - np.std(ratios)),
                       float(np.mean(ratios) + np.std(ratios))),
        "eta_breather": float(eta_b),
        "n_ens": int(n_ens),
        "H_s": float(H_s),
        "T_p": float(T_p),
    }
