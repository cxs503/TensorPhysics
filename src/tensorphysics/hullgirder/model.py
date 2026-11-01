"""Hull girder idealisation used by the ice / wave extreme bending workflow.

A ``ShipGirder`` is deliberately minimal: length, breadth, draught, block
coefficient, depth, midship section modulus, yield stress and an equivalent
plate slenderness.  Longitudinal distributions are generated analytically:

* waterplane width  b(x)/B = (1 - (2x/L)^2)^p, with p calibrated to the
  standard waterplane-area estimate  C_wp = (1 + 2 C_b) / 3  (Schneekluth /
  Parsons-type correlation for merchant hulls).  This keeps the hull form a
  deterministic function of ``(L, B, Cb)`` instead of a tuned curve;
* mass per unit length is uniform (homogeneous full-load condition), which is
  the standard assumption when no loading manual is available.

The module also exposes the beam grid and helpers shared by the ice and wave
load models.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

RHO_WATER = 1025.0  # kg/m^3, sea water
GRAVITY = 9.81  # m/s^2

__all__ = ["ShipGirder", "RHO_WATER", "GRAVITY"]


@dataclass(frozen=True)
class ShipGirder:
    """Minimal long-ship idealisation (SI units, moments in N*m)."""

    length: float = 200.0
    breadth: float = 32.0
    draught: float = 12.0
    block_coefficient: float = 0.8
    depth: float = 18.0
    section_modulus: float = 19.0  # m^3, midship elastic section modulus
    yield_stress: float = 235.0e6  # Pa
    plate_slenderness: float = 1.7  # beta = (b/t) sqrt(sigma_y / E)
    n_nodes: int = 401
    loading_factor: float = 1.0  # multiplies the uniform mass distribution
    waterplane_exponent: float | None = None
    _p_cache: float = field(default=float("nan"), init=False, repr=False, compare=False)

    # ---------------------------------------------------------------- geometry
    @property
    def waterplane_coefficient(self) -> float:
        """Standard estimate C_wp = (1 + 2 C_b)/3 (clipped to a physical range)."""
        return float(min(0.92, max(0.55, (1.0 + 2.0 * self.block_coefficient) / 3.0)))

    def _waterplane_exponent(self) -> float:
        if self.waterplane_exponent is not None:
            return float(self.waterplane_exponent)
        x = self.grid()

        def cwp(p: float) -> float:
            b = self.breadth * np.clip(1.0 - (2.0 * x / self.length) ** 2, 0.0, None) ** p
            return float(np.trapezoid(b, x)) / (self.length * self.breadth)

        target = self.waterplane_coefficient
        lo, hi = 1e-3, 6.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if cwp(mid) > target:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-10:
                break
        return float(0.5 * (lo + hi))

    def grid(self) -> np.ndarray:
        return np.linspace(-0.5 * self.length, 0.5 * self.length, int(self.n_nodes))

    def waterplane_width(self, x: np.ndarray | None = None) -> np.ndarray:
        x = self.grid() if x is None else np.asarray(x, dtype=float)
        p = self._waterplane_exponent()
        return self.breadth * np.clip(1.0 - (2.0 * x / self.length) ** 2, 0.0, None) ** p

    # ---------------------------------------------------------------- mass
    @property
    def displacement_volume(self) -> float:
        return self.length * self.breadth * self.draught * self.block_coefficient

    @property
    def displacement_mass(self) -> float:
        return RHO_WATER * self.displacement_volume

    @property
    def weight(self) -> float:
        return self.displacement_mass * GRAVITY

    def mass_per_length(self, x: np.ndarray | None = None) -> np.ndarray:
        x = self.grid() if x is None else np.asarray(x, dtype=float)
        return np.full(x.shape, self.loading_factor * self.displacement_mass / self.length)

    # ---------------------------------------------------------------- summary
    def as_dict(self) -> dict:
        return {
            "length_m": self.length,
            "breadth_m": self.breadth,
            "draught_m": self.draught,
            "depth_m": self.depth,
            "block_coefficient": self.block_coefficient,
            "waterplane_coefficient": self.waterplane_coefficient,
            "displacement_t": self.displacement_mass / 1000.0,
            "section_modulus_m3": self.section_modulus,
            "yield_stress_MPa": self.yield_stress / 1e6,
            "plate_slenderness_beta": self.plate_slenderness,
            "n_nodes": int(self.n_nodes),
        }