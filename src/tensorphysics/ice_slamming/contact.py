"""Ice/structure contact law: elastic (Hertz-family) -> yield -> brittle fracture.

Contact idealisation
--------------------
The floe is a thin disk; against the panel it behaves as a **flat circular
punch** of radius ``a`` on an elastic foundation.  The punch law on a
half-space is linear in penetration (Johnson, *Contact Mechanics*, 1985,
ch. 5):

    F = k_punch * delta,   k_punch = 4 G a / (1 - nu)

which is the flat-contact limit of the Hertz family (the Hertz sphere law
``F ~ delta^{3/2}`` is recovered for a rounded edge; the flat limit is the
correct one for a floe whose thickness is small compared with its radius).
The *effective* ice-side stiffness is the series combination of the floe's
bending stiffness and the punch stiffness (the punch is very stiff, so the
floe bending dominates).

Ice constitutive path (linear-elastic -> yield -> fracture)
------------------------------------------------------------
* elastic:  ``sigma_c < sigma_y``            -> full stiffness;
* yield:    ``sigma_y <= sigma_c < sigma_ice`` -> stiffness degrades linearly
  (microcracking, quasi-brittle);
* fracture: ``sigma_c >= sigma_ice``         -> the ice can no longer carry
  load; the released elastic strain energy ``0.5 k delta^2`` drives a crack
  of area ``A_f = min(A_c, 0.5 k delta^2 / G_f)`` (energy-consistent), and
  the ice-side stiffness drops to zero (the load path shifts to the water).

The fracture area is capped at the contact area and the fracture energy
``G_f A_f`` is the sink in the floe energy balance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = ["ContactLaw", "IceFractureState", "contact_force", "fracture_area"]


@dataclass
class IceFractureState:
    """Mutable fracture / yield state for one impact."""

    yield_onset: bool = False
    fractured: bool = False
    fracture_area: float = 0.0
    fracture_energy: float = 0.0
    yield_time: float | None = None
    fracture_time: float | None = None


@dataclass(frozen=True)
class ContactLaw:
    """Flat-punch contact with the ice elastic->yield->fracture path."""

    a: float                 # contact radius [m]
    k_floe: float            # floe bending stiffness [N/m]
    k_panel: float           # panel FEM stiffness [N/m]
    sigma_y: float           # yield (microcrack onset) stress [Pa]
    sigma_ice: float         # brittle failure stress [Pa]
    G_f: float               # fracture energy [J/m^2]
    nu: float = 0.30
    G_ice: float = 3.42e9    # shear modulus of ice [Pa]

    @property
    def contact_area(self) -> float:
        return math.pi * self.a**2

    @property
    def k_punch(self) -> float:
        return 4.0 * self.G_ice * self.a / (1.0 - self.nu)

    @property
    def k_ice(self) -> float:
        """Series combination of floe bending and punch stiffness [N/m]."""
        return 1.0 / (1.0 / self.k_floe + 1.0 / self.k_punch)

    def stiffness(self, state: IceFractureState) -> float:
        """Current ice-side stiffness [N/m] (0 after fracture)."""
        if state.fractured:
            return 0.0
        return self.k_ice

    def force(self, delta: float, state: IceFractureState,
              t: float | None = None) -> float:
        """Contact force [N] for penetration ``delta`` [m]."""
        if delta <= 0.0:
            return 0.0
        k = self.stiffness(state)
        if k <= 0.0:
            return 0.0
        f = k * delta
        sigma = f / self.contact_area
        if not state.yield_onset and sigma >= self.sigma_y:
            state.yield_onset = True
            if t is not None:
                state.yield_time = t
        if not state.fractured and sigma >= self.sigma_ice:
            se = 0.5 * self.k_ice * delta**2
            a_f = min(self.contact_area, se / self.G_f)
            state.fractured = True
            state.fracture_area = a_f
            state.fracture_energy = self.G_f * a_f
            if t is not None:
                state.fracture_time = t
            return f  # force at the instant of fracture
        # yield-stage stiffness degradation
        if state.yield_onset and not state.fractured:
            deg = (sigma - self.sigma_y) / max(1e-30, self.sigma_ice - self.sigma_y)
            k *= (1.0 - 0.5 * min(1.0, max(0.0, deg)))
            f = k * delta
        return f


def contact_force(law: ContactLaw, delta: float, state: IceFractureState,
                  t: float | None = None) -> float:
    """Convenience wrapper around :meth:`ContactLaw.force`."""
    return law.force(delta, state, t)


def fracture_area(strain_energy: float, G_f: float, a_max: float) -> float:
    """Fractured area from the released strain energy (energy-consistent)."""
    if G_f <= 0:
        return 0.0
    return min(a_max, max(0.0, strain_energy) / G_f)
