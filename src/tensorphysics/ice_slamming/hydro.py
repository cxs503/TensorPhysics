"""Water side of the triad: Wagner-type slamming pressure + wave kinematics.

Slamming pressure
-----------------
A body entering water at speed ``V`` with deadrise ``beta`` generates a
pressure field peaked at the spray root; the classical (singular) Wagner
(1971) scaling is

    p ~ C_p * 0.5 * rho * V^2,   C_p = pi / tan(beta)

which is the same bounded idealisation used by
:mod:`tensorphysics.chain_ledger.slamming` (kept consistent with the
platform's existing slamming source).  Here the "body" is the wave-carried
floe pressing the water against the flare panel; ``V`` is the floe speed
relative to the panel and ``beta`` is the panel deadrise (angle from the
horizontal).  The pressure acts over the wetted contact area and *opposes*
the floe motion (the compressed water pushes back) -- it is therefore a
load path that scales as ``V^2`` and a sink in the floe energy balance.

Wave orbital velocity
---------------------
Linear deep-water theory: the surface orbital velocity amplitude is
``u_max = pi H_s / T`` (Faltinsen, *Sea Loads on Ships*, 1990).  The wave
height enters the model through (i) the orbital velocity that adds to the
floe's approach speed and (ii) the wetted fraction of the floe face.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["SlammingHydro", "slamming_pressure", "wave_orbital_velocity"]


def wave_orbital_velocity(h_s: float, period_s: float) -> float:
    """Surface orbital velocity amplitude ``pi H_s / T`` [m/s]."""
    if h_s < 0:
        raise ValueError("h_s must be >= 0")
    if period_s <= 0:
        raise ValueError("period_s must be > 0")
    return math.pi * h_s / period_s


def slamming_pressure(v_rel: float, rho: float, deadrise_deg: float,
                      c_p_override: float | None = None) -> float:
    """Wagner-type slamming pressure [Pa] for deadrise ``beta`` [deg]."""
    if v_rel < 0:
        raise ValueError("v_rel must be >= 0")
    beta = math.radians(deadrise_deg)
    if beta <= 1e-6 or beta >= math.pi / 2 - 1e-6:
        raise ValueError("deadrise must lie in (0, 90) deg")
    c_p = c_p_override if c_p_override is not None else math.pi / math.tan(beta)
    return c_p * 0.5 * rho * v_rel**2


@dataclass(frozen=True)
class SlammingHydro:
    """Water-side model for one impact case."""

    rho_water: float
    wave_period_s: float
    added_mass_coeff: float
    damping_coeff: float      # N s / m^2 (per unit area)

    def deadrise(self, impact_angle_deg: float) -> float:
        """Panel deadrise from the impact angle.

        ``impact_angle_deg`` is the angle between the floe velocity and the
        panel surface; the panel inclination from vertical equals it, so the
        deadrise (from horizontal) is ``90 - theta``.
        """
        return 90.0 - impact_angle_deg

    def c_p(self, impact_angle_deg: float) -> float:
        beta = math.radians(self.deadrise(impact_angle_deg))
        return math.pi / math.tan(beta)

    def pressure(self, v_rel: float, impact_angle_deg: float) -> float:
        return slamming_pressure(v_rel, self.rho_water,
                                 self.deadrise(impact_angle_deg))

    def wetted_fraction(self, h_s: float, floe_thickness: float) -> float:
        """Fraction of the floe face wetted by a wave of height ``h_s``.

        A wave of height ``H_s`` submerges a fraction of the floe thickness;
        the wetted face fraction is ``min(1, H_s / (2 h_f))`` (the floe
        floats with roughly half its thickness submerged at rest, so a wave
        of height ``H_s`` adds ``H_s/2`` of submergence on the crest side).
        """
        if floe_thickness <= 0:
            raise ValueError("floe_thickness must be > 0")
        return min(1.0, 0.5 + h_s / (2.0 * floe_thickness))

    def added_mass(self, area_m2: float) -> float:
        return self.added_mass_coeff * self.rho_water * area_m2

    def damping_force(self, v_rel: float, area_m2: float) -> float:
        """Hydrodynamic drag on the floe [N], opposing the motion."""
        return self.damping_coeff * area_m2 * v_rel
