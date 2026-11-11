"""Peregrine breather: the rational fundamental rogue-wave solution of the
focusing NLS, used as the exact-solution anchor for the third-order extreme
wave benchmark.

The Peregrine breather on a unit background is

    A(x, t) = [ 1 - 4(1 + 2 i t) / (1 + 4 x^2 + 4 t^2) ] e^{i t},

an exact solution of i A_t + (1/2) A_xx + |A|^2 A = 0 (the standard focusing
NLS with unit background).  Its peak |A| = 3 at (x, t) = (0, 0) is three times
the background -- the canonical rogue-wave amplification factor of 3.

C1 criterion: the NLS numerical solution must reproduce the analytic breather
to a phase-aligned L2 relative error <= 2%.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "PeregrineBreather",
    "peregrine_amplitude",
    "peregrine_envelope",
    "peregrine_peak_amplitude",
    "peregrine_phase_aligned_l2_error",
]


def peregrine_envelope(x: np.ndarray, t: float) -> np.ndarray:
    """Peregrine breather envelope A(x, t) on a unit background.

    Parameters
    ----------
    x : ndarray
        Spatial coordinate.
    t : float
        Time.

    Returns
    -------
    complex ndarray
        The complex envelope A(x, t).
    """
    x = np.asarray(x, dtype=float)
    denom = 1.0 + 4.0 * x ** 2 + 4.0 * t ** 2
    # rational term
    r = 4.0 * (1.0 + 2.0j * t) / denom
    return (1.0 - r) * np.exp(1j * t)


def peregrine_amplitude(x: np.ndarray, t: float) -> np.ndarray:
    """|A(x, t)| of the Peregrine breather."""
    return np.abs(peregrine_envelope(x, t))


def peregrine_peak_amplitude() -> float:
    """Peak |A| of the Peregrine breather (at x=0, t=0): 3.0."""
    return 3.0


@dataclass
class PeregrineBreather:
    """Peregrine breather analytic solution on a spatial grid.

    Parameters
    ----------
    x : ndarray
        Spatial grid.
    t : float
        Time at which the solution is evaluated.
    """

    x: np.ndarray
    t: float

    @property
    def envelope(self) -> np.ndarray:
        return peregrine_envelope(self.x, self.t)

    @property
    def amplitude(self) -> np.ndarray:
        return np.abs(self.envelope)

    @property
    def peak(self) -> float:
        return float(np.max(self.amplitude))


def _phase_align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Phase-align b to a by the global phase that minimises the L2 error.

    The NLS is phase-invariant under A -> A e^{i phi}; the numerical solution
    may carry a global phase offset relative to the analytic one.  We remove
    the best-fit global phase before computing the L2 error (this is the
    "phase-aligned" error of the C1 criterion).
    """
    # best-fit phase: phi = arg( <a, b> )
    inner = np.sum(np.conj(a) * b)
    phi = np.angle(inner)
    return b * np.exp(-1j * phi)


def peregrine_phase_aligned_l2_error(
    A_num: np.ndarray,
    x: np.ndarray,
    t: float,
) -> float:
    """Phase-aligned L2 relative error of the numerical solution vs the
    analytic Peregrine breather at time t.

    Parameters
    ----------
    A_num : complex ndarray
        Numerical envelope on the spatial grid.
    x : ndarray
        Spatial grid.
    t : float
        Time.

    Returns
    -------
    float
        L2 relative error (dimensionless, 0 = exact).
    """
    A_ana = peregrine_envelope(x, t)
    A_align = _phase_align(A_ana, A_num)
    num = np.sqrt(np.sum(np.abs(A_align - A_ana) ** 2))
    den = np.sqrt(np.sum(np.abs(A_ana) ** 2))
    if den < 1e-30:
        return 0.0
    return float(num / den)
