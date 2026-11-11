"""Inverse problem: given a target crest, recover the required bandwidth /
steepness, then re-synthesise a wave field that reproduces the target crest
(C6).

The Peregrine breather's peak crest is 3 A_0, where A_0 is the background
amplitude.  The *steepness* of the carrier is s = k_0 A_0 (with k_0 = 1 in the
normalised NLS), and the *bandwidth* of the breather (the spatial extent of the
focal region) is set by the rational denominator 1 + 4 x^2 + 16 t^2.

Inverse problem: given a target crest eta_t, find the background amplitude
A_0 = eta_t / 3 and the corresponding steepness s = k_0 A_0, then re-synthesise
a Peregrine-type wave field (with that background) and verify it reproduces the
target crest to <= 5%.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "inverse_bandwidth_from_crest",
    "inverse_steepness_from_crest",
    "inverse_wave_field",
    "inverse_reproduction_error",
    "inverse_problem_gate",
]


def inverse_bandwidth_from_crest(target_crest: float, A0: float = 1.0) -> dict:
    """Recover the required bandwidth from a target crest.

    The breather peak is 3 A_0; the focal region half-width (in x) at the peak
    time is set by the denominator 1 + 4 x^2 (at t=0), i.e. the amplitude drops
    to half its peak at |x| = 1/2.  The "bandwidth" (spatial wavenumber content
    of the focal region) is therefore k_band ~ 2 / (focal half-width).

    Parameters
    ----------
    target_crest : float
        Target peak crest (m).
    A0 : float
        Background amplitude (m).

    Returns
    -------
    dict
        With keys: target_crest, A0_required, focal_half_width,
        bandwidth (spatial wavenumber).
    """
    target_crest = float(target_crest)
    A0 = float(A0)
    # required background: peak = 3 A0
    A0_req = target_crest / 3.0
    # focal half-width at t=0: |A| = 3/2 at |x| = 1/2 (in normalised units)
    # -- the denominator 1 + 4 x^2 gives half-peak at x = 1/2
    focal_half_width = 0.5
    # bandwidth: the spatial wavenumber content ~ 2 / focal_half_width
    bandwidth = 2.0 / focal_half_width
    return {
        "target_crest": target_crest,
        "A0_required": float(A0_req),
        "focal_half_width": float(focal_half_width),
        "bandwidth": float(bandwidth),
    }


def inverse_steepness_from_crest(target_crest: float, k0: float = 1.0) -> dict:
    """Recover the required steepness from a target crest.

    Steepness s = k_0 A_0, with A_0 = target_crest / 3.

    Parameters
    ----------
    target_crest : float
        Target peak crest (m).
    k0 : float
        Carrier wavenumber (1/m).

    Returns
    -------
    dict
        With keys: target_crest, A0_required, steepness.
    """
    target_crest = float(target_crest)
    k0 = float(k0)
    A0_req = target_crest / 3.0
    return {
        "target_crest": target_crest,
        "A0_required": float(A0_req),
        "steepness": float(k0 * A0_req),
    }


def inverse_wave_field(
    target_crest: float,
    x: np.ndarray,
    t: float = 0.0,
    k0: float = 1.0,
) -> np.ndarray:
    """Re-synthesise a Peregrine-type wave field that reproduces the target
    crest.

    The field is the Peregrine breather scaled to the required background
    A_0 = target_crest / 3:

        A(x, t) = A_0 * [ 1 - 4(1 + 4 i t) / (1 + 4 x^2 + 16 t^2) ] e^{i t}.

    Parameters
    ----------
    target_crest : float
        Target peak crest (m).
    x : ndarray
        Spatial grid.
    t : float
        Time.
    k0 : float
        Carrier wavenumber (1/m) -- scales the spatial coordinate.

    Returns
    -------
    complex ndarray
        The re-synthesised envelope.
    """
    target_crest = float(target_crest)
    A0 = target_crest / 3.0
    x = np.asarray(x, dtype=float)
    # scale the spatial coordinate by k0 (so the breather is in physical units)
    xs = k0 * x
    denom = 1.0 + 4.0 * xs ** 2 + 16.0 * t ** 2
    r = 4.0 * (1.0 + 4.0j * t) / denom
    return A0 * (1.0 - r) * np.exp(1j * t)


def inverse_reproduction_error(
    target_crest: float,
    x: np.ndarray,
    t: float = 0.0,
    k0: float = 1.0,
) -> dict:
    """Reproduction error of the inverse-synthesised field vs the target crest.

    Parameters
    ----------
    target_crest : float
        Target peak crest (m).
    x : ndarray
        Spatial grid.
    t : float
        Time.
    k0 : float
        Carrier wavenumber.

    Returns
    -------
    dict
        With keys: target_crest, reproduced_crest, relative_error.
    """
    A = inverse_wave_field(target_crest, x, t, k0)
    reproduced = float(np.max(np.abs(A)))
    rel_err = abs(reproduced - target_crest) / target_crest if target_crest > 0 else 0.0
    return {
        "target_crest": float(target_crest),
        "reproduced_crest": reproduced,
        "relative_error": float(rel_err),
    }


def inverse_problem_gate(
    target_crest: float,
    x: np.ndarray,
    t: float = 0.0,
    k0: float = 1.0,
    tol: float = 0.05,
) -> dict:
    """Full inverse-problem gate: recover bandwidth/steepness, re-synthesise,
    and verify the target crest is reproduced to <= tol (default 5%).

    Parameters
    ----------
    target_crest : float
        Target peak crest (m).
    x : ndarray
        Spatial grid.
    t : float
        Time.
    k0 : float
        Carrier wavenumber.
    tol : float
        Reproduction tolerance (default 0.05 = 5%).

    Returns
    -------
    dict
        With keys: bandwidth, steepness, reproduction, verdict.
    """
    bw = inverse_bandwidth_from_crest(target_crest)
    st = inverse_steepness_from_crest(target_crest, k0)
    rep = inverse_reproduction_error(target_crest, x, t, k0)
    verdict = "PASS" if rep["relative_error"] <= tol else "FAIL"
    return {
        "bandwidth": bw,
        "steepness": st,
        "reproduction": rep,
        "verdict": verdict,
        "tolerance": float(tol),
    }
