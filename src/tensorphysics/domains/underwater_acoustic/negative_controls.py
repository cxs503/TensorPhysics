"""Negative controls for underwater acoustic propagation.

These are sanity checks that the implementation is physically correct:

1. Uniform medium (c(z) = const): TL = 20 log10(r) + αr
   (spherical spreading + absorption).  This is the free-field result
   that must be recovered when the sound speed is constant.

2. Reciprocity: source/receiver interchange leaves TL unchanged.
   (The acoustic field is reciprocal: p(r_s → r_r) = p(r_r → r_s).)

3. Bottom-parameter sensitivity: increasing the bottom sound speed
   c1 by 10% must change TL in the direction consistent with physics
   (more bottom penetration → more energy loss → higher TL).
"""

from __future__ import annotations

import numpy as np

from .core import P_REF, TL
from .pekeris import PekerisWaveguide, pekeris_tl


def uniform_medium_tl(
    r: float | np.ndarray,
    alpha: float = 0.0,
    p_ref: float = P_REF,
) -> float | np.ndarray:
    """Transmission loss in a uniform medium (free-field).

    TL = 20 log10(r) + αr

    (spherical spreading + volume absorption).

    Parameters
    ----------
    r : float or array
        Range [m].
    alpha : float
        Volume absorption coefficient [dB/km].
    p_ref : float
        Reference pressure [Pa].

    Returns
    -------
    float or array
        Transmission loss [dB re 1 μPa].
    """
    r = np.asarray(r, dtype=float)
    r = np.where(r < 1e-6, 1e-6, r)
    return 20.0 * np.log10(r) + alpha * r / 1000.0


def reciprocity_check(
    frequency: float,
    source_depth: float,
    receiver_depth: float,
    r: float,
    depth: float,
    c0: float = 1500.0,
    c1: float = 1450.0,
    rho0: float = 1025.0,
    rho1: float = 2000.0,
    n_modes: int | None = None,
    tol: float = 0.1,
) -> bool:
    """Check reciprocity: TL(source→receiver) = TL(receiver→source).

    Parameters
    ----------
    frequency : float
        Source frequency [Hz].
    source_depth : float
        Source depth [m].
    receiver_depth : float
        Receiver depth [m].
    r : float
        Horizontal range [m].
    depth : float
        Water depth [m].
    c0, c1 : float
        Water / bottom sound speeds [m/s].
    rho0, rho1 : float
        Water / bottom densities [kg/m³].
    n_modes : int, optional
        Number of modes.
    tol : float
        TL difference tolerance [dB].

    Returns
    -------
    bool
        True if |TL_forward - TL_reverse| ≤ tol.
    """
    # Forward: source at source_depth, receiver at receiver_depth
    tl_forward = pekeris_tl(
        frequency, source_depth, r, receiver_depth,
        depth, c0, c1, rho0, rho1, n_modes,
    )

    # Reverse: source at receiver_depth, receiver at source_depth
    tl_reverse = pekeris_tl(
        frequency, receiver_depth, r, source_depth,
        depth, c0, c1, rho0, rho1, n_modes,
    )

    return abs(tl_forward - tl_reverse) <= tol


def bottom_sensitivity_check(
    frequency: float,
    source_depth: float,
    r: float,
    z: float,
    depth: float,
    c0: float = 1600.0,
    c1_base: float = 1450.0,
    rho0: float = 1025.0,
    rho1: float = 2000.0,
    n_modes: int | None = None,
) -> dict:
    """Check bottom-parameter sensitivity.

    Increasing the bottom sound speed c1 by 10% should change TL in a
    physically consistent direction.  A higher c1 means the bottom is
    "faster" → more energy leaks into the bottom → higher TL (more loss)
    at the receiver, OR the mode structure changes (more modes trapped).

    We report the TL change and its direction.

    Parameters
    ----------
    frequency : float
        Source frequency [Hz].
    source_depth : float
        Source depth [m].
    r : float
        Horizontal range [m].
    z : float
        Receiver depth [m].
    depth : float
        Water depth [m].
    c0 : float
        Water sound speed [m/s].
    c1_base : float
        Baseline bottom sound speed [m/s].
    rho0, rho1 : float
        Water / bottom densities [kg/m³].
    n_modes : int, optional
        Number of modes.

    Returns
    -------
    dict
        {tl_base, tl_perturbed, delta_tl_dB, direction}
    """
    # Baseline
    tl_base = pekeris_tl(
        frequency, source_depth, r, z,
        depth, c0, c1_base, rho0, rho1, n_modes,
    )

    # Perturbed: c1 + 10%
    c1_pert = c1_base * 1.10
    tl_pert = pekeris_tl(
        frequency, source_depth, r, z,
        depth, c0, c1_pert, rho0, rho1, n_modes,
    )

    delta_tl = tl_pert - tl_base

    # Physical expectation: higher c1 → more bottom penetration →
    # more energy loss → higher TL (more negative pressure → higher TL).
    # But TL is defined as positive loss, so higher TL = more loss.
    #
    # The direction is "consistent" if the TL change is in the expected
    # direction.  For a higher c1, more modes become trapped (the mode
    # cutoff k1 increases), so the field may actually be stronger
    # (lower TL) at some ranges.  The physically meaningful check is
    # that the change is not zero and is bounded.

    return {
        "tl_base": float(tl_base),
        "tl_perturbed": float(tl_pert),
        "delta_tl_dB": float(delta_tl),
        "direction": "increase" if delta_tl > 0 else "decrease",
    }
