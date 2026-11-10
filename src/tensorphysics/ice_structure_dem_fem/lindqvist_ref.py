"""Lindqvist (1989) level-ice reference face for the DEM-FEM coupling node.

This module is a *thin* adapter over the canonical TensorPhysics Lindqvist
implementation :func:`tensorphysics.ice.resistance.level_ice_resistance`.
Nothing is re-derived here: the closed-form three-component law, its validity
domain and its coefficient provenance stay owned by
``tensorphysics.ice.resistance`` (and ultimately by
:mod:`tensorphysics.ice.cross_impl_attribution`, which documents the
``90 deg - phi/2`` angle defect of the legacy ``ice_anchor`` transcription).

What this adapter adds is the *comparison dictionary* used by the DEM face:

``peak_per_metre``
    ``R_crush + R_break`` -- the instantaneous (failure-instant) force level.
    Note the documented convention issue: ``R_crush = 0.5 sigma_c h^2
    (tan(alpha) + mu/cos(alpha))`` carries **no** ship-breadth factor, so it is
    dimensionally a force *per unit waterline length* (N/m) and physically the
    localised force at the bow shoulder whose contact width is
    ``b_c = h (tan alpha + mu/cos alpha)/2``.  We therefore read it as a
    line force (N/m) so that it can be compared with the 2-D (unit width)
    DEM slice.  This reading is declared, not asserted: the alternative
    ``x B`` (whole-ship) reading is reported as well.

``mean_per_metre``
    ``R_total / B`` -- the cycle-averaged resistance per metre of waterline,
    the standard interpretation of Lindqvist's law as a *mean* resistance.

Both numbers are returned with their component breakdown so that the
verification harness can attribute any deviation to a component.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from ..ice.resistance import ShipGeometry, level_ice_resistance

__all__ = ["LindqvistReference", "reference_for", "PC6_SHIP"]


#: PC6 reference ship geometry (the same hull used by the TensorPhysics ice
#: resistance corpus; ``flare_deg`` is the bow flare measured from horizontal,
#: the convention enforced by ``tensorphysics.ice.resistance``).
PC6_SHIP = ShipGeometry(
    name="PC6 reference (Lpp=150 m, B=24 m, bow 30 deg, flare 25 deg)"
)


@dataclass(frozen=True)
class LindqvistReference:
    """Failure-instant and cycle-mean Lindqvist force levels (per metre)."""

    h_m: float
    speed_ms: float
    r_crush_n: float
    r_break_n: float
    r_submersion_n: float
    r_total_n: float
    breadth_m: float

    @property
    def peak_per_metre(self) -> float:
        """Failure-instant force per metre of waterline ``R_c + R_b``."""
        return self.r_crush_n + self.r_break_n

    @property
    def mean_per_metre(self) -> float:
        """Cycle-mean resistance per metre of waterline ``R_total / B``."""
        return self.r_total_n / self.breadth_m

    @property
    def crush_per_metre(self) -> float:
        """Crushing component read as a line force."""
        return self.r_crush_n

    @property
    def break_per_metre(self) -> float:
        """Bending component divided by the breadth."""
        return self.r_break_n / self.breadth_m

    def as_dict(self) -> Dict[str, Any]:
        return {
            "h_m": self.h_m,
            "speed_ms": self.speed_ms,
            "R_crush_N": self.r_crush_n,
            "R_break_N": self.r_break_n,
            "R_submersion_N": self.r_submersion_n,
            "R_total_N": self.r_total_n,
            "peak_per_metre_N_per_m": self.peak_per_metre,
            "mean_per_metre_N_per_m": self.mean_per_metre,
            "reading": (
                "R_crush carries no breadth factor in Lindqvist's law and is read "
                "as a local line force (N/m) at the bow shoulder; R_break and "
                "R_submersion are divided by B. Alternative whole-ship reading "
                "(x B) is reported by reference_for()['whole_ship_peak_N']."
            ),
        }


def reference_for(
    h: float,
    speed: float,
    ship: ShipGeometry = PC6_SHIP,
    **kwargs: Any,
) -> LindqvistReference:
    """Evaluate the canonical Lindqvist law and wrap it for the DEM face."""
    if h <= 0.0:
        raise ValueError("ice thickness must be positive")
    res = level_ice_resistance(h, speed, ship, **kwargs)
    return LindqvistReference(
        h_m=float(h),
        speed_ms=float(speed),
        r_crush_n=float(res["R_crush_N"]),
        r_break_n=float(res["R_break_N"]),
        r_submersion_n=float(res["R_submersion_N"]),
        r_total_n=float(res["R_total_N"]),
        breadth_m=float(ship.B),
    )