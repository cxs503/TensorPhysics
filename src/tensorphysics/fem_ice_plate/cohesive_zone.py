"""Cohesive zone models for ice-sheet bending fracture.

Two traction-separation laws are provided:

* :class:`BilinearTSL` -- linear elastic ramp, linear softening, total fracture
  energy ``G_f = sigma_max * delta_f / 2``.
* :class:`ExponentialTSL` -- Xu-Needleman exponential law
  ``sigma = sigma_max (delta/delta_0) exp(1 - delta/delta_0)`` with
  ``delta_0 = G_f / (e sigma_max)``.

Both are *irreversible* (damage grows monotonically, unloading to the origin
with the damaged secant stiffness) and report their dissipated energy
exactly, so they can be used for energy audits.

For **plate/beam bending** cracks the mode-I law is recast as a
moment-rotation *cohesive hinge* (:class:`CohesiveHinge`).  With a hinge of
relative rotation ``phi`` across a plate of thickness ``h``, the crack opening
at the tension face is ``delta = phi * h`` and the energy per unit created
crack area is

.. math::  G_f = \\frac{1}{h} \\int M \\, d\\varphi

so a bilinear hinge law with peak moment ``M_peak`` and failure rotation
``phi_f`` has ``G_f = M_peak * phi_f / (2 h)``.  This is the classical
Hillerborg fictitious-crack mapping for bending members and it lets the same
fracture energy drive a plate/beam model without refining the ligament.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

__all__ = [
    "BilinearTSL",
    "ExponentialTSL",
    "CohesiveHinge",
    "dissipated_energy_numerical",
]


def dissipated_energy_numerical(traction, delta_max: float, n: int = 20001) -> float:
    """``int_0^delta_max t(delta) d(delta)`` by Simpson integration.

    Used as an independent check of the closed-form energy expressions.
    """
    if delta_max <= 0.0:
        return 0.0
    d = np.linspace(0.0, delta_max, n)
    t = np.array([traction(float(x)) for x in d])
    return float(np.trapezoid(t, d))


@dataclass
class BilinearTSL:
    """Bilinear (triangular) traction-separation law.

    Parameters
    ----------
    sigma_max : float
        Peak traction (tensile strength) [Pa].
    delta_f : float
        Separation at complete failure [m].
    delta_0 : float, optional
        Separation at peak traction [m]; default ``G_f/(sigma_max)`` ... see
        notes -- the default is ``2 G_f / sigma_max`` recomputed from
        ``delta_f`` only when provided explicitly, otherwise
        ``delta_0 = 0.1 * delta_f``.
    """

    sigma_max: float
    delta_f: float
    delta_0: Optional[float] = None

    def __post_init__(self) -> None:
        if self.sigma_max <= 0.0 or self.delta_f <= 0.0:
            raise ValueError("sigma_max and delta_f must be positive")
        if self.delta_0 is None:
            # penalise the initial slope so the elastic ramp is stiff but not
            # singular: 10 % of the failure separation
            self.delta_0 = 0.1 * self.delta_f
        if not (0.0 < self.delta_0 <= self.delta_f):
            raise ValueError("need 0 < delta_0 <= delta_f")

    # -- law ---------------------------------------------------------------
    @property
    def k0(self) -> float:
        """Initial (penalty) stiffness [Pa/m]."""
        return self.sigma_max / self.delta_0

    @property
    def G_f(self) -> float:
        """Fracture energy ``int t ddelta`` [J/m^2]."""
        return 0.5 * self.sigma_max * self.delta_f

    def traction(self, delta: float, delta_history: Optional[float] = None) -> float:
        """Current traction, with irreversibility driven by ``delta_history``."""
        d = max(float(delta), 0.0)
        dmax = max(d, 0.0 if delta_history is None else float(delta_history))
        if dmax <= self.delta_0:
            return self.k0 * d
        if dmax >= self.delta_f:
            return 0.0
        # damaged secant: sigma = (1-d) k0 delta with d from the softening line
        damage = 1.0 - (self.delta_0 / dmax) * (self.delta_f - dmax) / (
            self.delta_f - self.delta_0
        )
        return (1.0 - damage) * self.k0 * d

    def damage(self, delta_history: float) -> float:
        dmax = float(delta_history)
        if dmax <= self.delta_0:
            return 0.0
        if dmax >= self.delta_f:
            return 1.0
        return 1.0 - (self.delta_0 / dmax) * (self.delta_f - dmax) / (
            self.delta_f - self.delta_0
        )

    def dissipate(self, delta_history: float) -> float:
        """Energy dissipated up to ``delta_history`` [J/m^2].

        For ``dmax <= delta_0`` nothing is dissipated; beyond that

        .. math:: D = G_f \\left[1 - \\left(\\frac{\\delta_f-\\delta_{max}}
                  {\\delta_f-\\delta_0}\\right)^2\\right]
        """
        dmax = float(delta_history)
        if dmax <= self.delta_0:
            return 0.0
        if dmax >= self.delta_f:
            return self.G_f
        r = (self.delta_f - dmax) / (self.delta_f - self.delta_0)
        return self.G_f * (1.0 - r**2)


@dataclass
class ExponentialTSL:
    """Xu-Needleman exponential traction-separation law.

    ``sigma(delta) = sigma_max (delta/delta_0) exp(1 - delta/delta_0)`` with
    ``delta_0 = G_f / (e sigma_max)`` so that ``int t ddelta = G_f`` and the
    peak ``sigma_max`` is reached at ``delta = delta_0``.
    """

    sigma_max: float
    G_f: float

    def __post_init__(self) -> None:
        if self.sigma_max <= 0.0 or self.G_f <= 0.0:
            raise ValueError("sigma_max and G_f must be positive")

    @property
    def delta_0(self) -> float:
        return self.G_f / (np.e * self.sigma_max)

    @property
    def delta_f(self) -> float:
        """Separation where the traction falls below 1 % of the peak."""
        return self.delta_0 * (1.0 - np.log(1e-2))

    def traction(self, delta: float, delta_history: Optional[float] = None) -> float:
        d = max(float(delta), 0.0)
        dmax = max(d, 0.0 if delta_history is None else float(delta_history))
        if dmax <= 0.0:
            return 0.0
        z = dmax / self.delta_0
        # Xu-Needleman peak traction at the current history point; unloading
        # follows the damaged secant to the origin for all d <= dmax
        t_peak = self.sigma_max * z * np.exp(1.0 - z)
        return t_peak * d / dmax

    def dissipate(self, delta_history: float) -> float:
        z = max(float(delta_history), 0.0) / self.delta_0
        if z <= 1.0:
            return 0.0
        # int_0^z sigma_max s e^{1-s} ds = G_f [1 - e^{1-z}(1+z)]
        return self.G_f * (1.0 - np.exp(1.0 - z) * (1.0 + z))


@dataclass
class CohesiveHinge:
    """Moment-rotation cohesive law for a bending crack (fictitious crack).

    Parameters
    ----------
    M_peak : float
        Peak moment per unit width [N] (= N m / m) at the crack.
    phi_f : float
        Rotation at complete failure [rad].
    h : float
        Plate/beam thickness [m] -- converts the dissipated hinge energy
        ``int M dphi`` into fracture energy per unit crack area
        ``G_f = (1/h) int M dphi``.
    phi_0 : float, optional
        Rotation at the peak [rad]; defaults to ``0.05 phi_f``.
    """

    M_peak: float
    phi_f: float
    h: float
    phi_0: Optional[float] = None

    def __post_init__(self) -> None:
        if self.M_peak <= 0.0 or self.phi_f <= 0.0 or self.h <= 0.0:
            raise ValueError("M_peak, phi_f and h must be positive")
        if self.phi_0 is None:
            self.phi_0 = 0.05 * self.phi_f
        if not (0.0 < self.phi_0 <= self.phi_f):
            raise ValueError("need 0 < phi_0 <= phi_f")

    # -- constructors ------------------------------------------------------
    @classmethod
    def from_fracture_energy(
        cls, M_peak: float, G_f: float, h: float, phi_0_frac: float = 0.05
    ) -> "CohesiveHinge":
        """Build a hinge whose dissipated energy per crack area equals ``G_f``."""
        phi_f = 2.0 * G_f * h / M_peak
        return cls(M_peak=M_peak, phi_f=phi_f, h=h, phi_0=phi_0_frac * phi_f)

    @property
    def G_f(self) -> float:
        """Fracture energy per unit created crack area [J/m^2]."""
        return 0.5 * self.M_peak * self.phi_f / self.h

    @property
    def k_phi(self) -> float:
        """Initial rotational stiffness [N/rad]."""
        return self.M_peak / self.phi_0

    # -- law ---------------------------------------------------------------
    def moment(self, phi: float, phi_history: Optional[float] = None) -> float:
        p = abs(float(phi))
        pmax = max(p, 0.0 if phi_history is None else abs(float(phi_history)))
        sign = 1.0 if phi >= 0.0 else -1.0
        if pmax <= self.phi_0:
            return sign * self.k_phi * p
        if pmax >= self.phi_f:
            return 0.0
        damage = 1.0 - (self.phi_0 / pmax) * (self.phi_f - pmax) / (
            self.phi_f - self.phi_0
        )
        return sign * (1.0 - damage) * self.k_phi * p

    def tangent(self, phi: float, phi_history: Optional[float] = None) -> float:
        """``dM/dphi`` (secant unloading, softening slope in the damage range)."""
        p = abs(float(phi))
        pmax = max(p, 0.0 if phi_history is None else abs(float(phi_history)))
        if pmax <= self.phi_0:
            return self.k_phi
        if pmax >= self.phi_f:
            return 0.0
        return -self.M_peak / (self.phi_f - self.phi_0)

    def dissipate(self, phi_history: float) -> float:
        """Hinge energy per unit width [J/m] (``int M dphi``)."""
        pmax = abs(float(phi_history))
        if pmax <= self.phi_0:
            return 0.0
        total = 0.5 * self.M_peak * self.phi_f  # int_0^phi_f M dphi
        if pmax >= self.phi_f:
            return total
        r = (self.phi_f - pmax) / (self.phi_f - self.phi_0)
        return total * (1.0 - r**2)

    def report(self) -> Dict[str, float]:
        return {
            "M_peak": self.M_peak,
            "phi_0": float(self.phi_0),
            "phi_f": self.phi_f,
            "h": self.h,
            "G_f": self.G_f,
            "k_phi": self.k_phi,
        }