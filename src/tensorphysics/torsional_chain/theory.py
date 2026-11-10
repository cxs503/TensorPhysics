"""Closed-form references for the torsional chain.

Every function here is independent of the time integrator in
:mod:`tensorphysics.torsional_chain.drivetrain`, so a simulated number can be
compared against algebra instead of against another simulation.

References
----------
* Two-degree-of-freedom torsional systems: standard lumped-parameter result
  (e.g. Den Hartog, *Mechanical Vibrations*; Wright/Prabhu for shafting).
* Three-inertia free-free chain: characteristic polynomial obtained from
  ``det(K - lambda M) = 0`` (trace and second principal minor of ``M^-1 K``);
  both coefficients are derived in the docstring, so the reference is exact.
* SDOF frequency response and half-power bandwidth: standard.
* Mathieu growth exponent: two-timing expansion, see
  :func:`tensorphysics.torsional_chain.gear.mathieu_growth_rate`.
"""

from __future__ import annotations

import math
from typing import Tuple

__all__ = [
    "sdof_frf",
    "sdof_amplitude",
    "sdof_peak_frequency",
    "two_inertia_fixed_frequency",
    "two_inertia_free_frequency",
    "three_inertia_chain_frequencies",
    "reflected_inertia",
    "reflected_to_high_speed",
    "half_power_bandwidth",
]


def sdof_frf(omega: float, inertia: float, stiffness: float, damping: float = 0.0) -> complex:
    """Compliance ``theta / T`` [rad/(N*m)] of ``J theta'' + c theta' + k theta = T``."""
    return 1.0 / complex(stiffness - inertia * omega ** 2, damping * omega)


def sdof_amplitude(omega: float, inertia: float, stiffness: float, damping: float = 0.0) -> float:
    """Magnitude of :func:`sdof_frf`."""
    return abs(sdof_frf(omega, inertia, stiffness, damping))


def sdof_peak_frequency(inertia: float, stiffness: float, damping: float) -> float:
    """Frequency of maximum compliance ``omega_n sqrt(1 - 2 zeta**2)`` [rad/s]."""
    zeta = damping / (2.0 * math.sqrt(stiffness * inertia))
    wn = math.sqrt(stiffness / inertia)
    if zeta >= 1.0 / math.sqrt(2.0):
        return 0.0
    return wn * math.sqrt(1.0 - 2.0 * zeta ** 2)


def two_inertia_fixed_frequency(stiffness: float, inertia_a: float) -> float:
    """Two inertias, one end grounded: ``omega = sqrt(k / J_a)`` [rad/s]."""
    return math.sqrt(stiffness / inertia_a)


def two_inertia_free_frequency(stiffness: float, inertia_a: float, inertia_b: float) -> float:
    """Two inertias, free-free (non-zero mode): ``omega = sqrt(k (1/Ja + 1/Jb))``."""
    return math.sqrt(stiffness * (1.0 / inertia_a + 1.0 / inertia_b))


def three_inertia_chain_frequencies(k1: float, k2: float,
                                    j1: float, j2: float, j3: float) -> Tuple[float, float]:
    """Two non-zero natural frequencies of a free-free 3-inertia chain [rad/s].

    ``M^-1 K`` for ``K = [[k1,-k1,0],[-k1,k1+k2,-k2],[0,-k2,k2]]`` and
    ``M = diag(j1,j2,j3)`` has eigenvalues ``lambda = omega**2`` with
    ``lambda1 = 0`` (rigid-body mode) and ``lambda2, lambda3`` roots of

        ``lambda**2 - S lambda + P = 0``
        ``S = k1 (1/j1 + 1/j2) + k2 (1/j2 + 1/j3)``           (trace of M^-1 K)
        ``P = k1 k2 (j1 + j2 + j3) / (j1 j2 j3)``             (sum of 2x2 minors)

    Verified symbolically by expanding ``det(K - lambda M)``.
    """
    s = k1 * (1.0 / j1 + 1.0 / j2) + k2 * (1.0 / j2 + 1.0 / j3)
    p = k1 * k2 * (j1 + j2 + j3) / (j1 * j2 * j3)
    disc = max(s * s - 4.0 * p, 0.0)
    lam_hi = 0.5 * (s + math.sqrt(disc))
    lam_lo = 0.5 * (s - math.sqrt(disc))
    return math.sqrt(max(lam_lo, 0.0)), math.sqrt(max(lam_hi, 0.0))


def reflected_inertia(inertia_high_speed: float, ratio: float) -> float:
    """Reflect a high-speed-side inertia to the low-speed side: ``J * i**2``.

    ``ratio = Z_wheel / Z_pinion = omega_pinion / omega_wheel`` (> 1 for a
    reduction gear).
    """
    return inertia_high_speed * ratio ** 2


def reflected_to_high_speed(inertia_low_speed: float, ratio: float) -> float:
    """Reflect a low-speed-side inertia to the high-speed side: ``J / i**2``."""
    return inertia_low_speed / ratio ** 2


def half_power_bandwidth(omega_n: float, zeta: float) -> float:
    """``Delta omega`` between half-power points of a light-damped SDOF [rad/s]."""
    return 2.0 * zeta * omega_n