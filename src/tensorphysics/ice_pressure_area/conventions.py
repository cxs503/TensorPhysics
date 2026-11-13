"""Area-convention algebra for the ice pressure-area relation ``p = p0 * A**-ex``.

Motivation
==========
The pressure-area exponent ``ex`` is quoted all over the ice-loads literature
without a statement of *which* area ``A`` is meant.  Candidates in current use:

* the projected (normal) contact area of an indenter or load patch on the
  structure -- ``nominal_projected_contact`` (the reference convention here);
* the same area widened by the crushing / damaged process zone --
  ``damaged_zone_area``;
* a per-metre (line-load) area -- ``line_load_area``;
* the nominal structural panel area over which a *global* load is averaged --
  ``panel_nominal_area``.

This module answers the question the gap ``GAP-PP-2-ICE-STRUCT-PRESSURE-AREA-
CALIB`` actually poses: *can a change of area convention move ``ex``, and if so
by how much?*  The answer is analytic and two-part:

1. **Ratio-invariance theorem.**  If the source-convention area relates to the
   reference-convention area by a *constant* factor (``A_src = k * A_ref``,
   ``k`` unitless), then ``ex`` is invariant: ``ex_ref = ex_src``.  Pure
   bookkeeping between "panel area", "per-metre area" and "contact area" at a
   fixed footprint therefore *cannot* explain scatter in ``ex``.

2. **Log-derivative transfer.**  In general, with ``A_src = g(A_ref)`` and
   ``r = d ln g / d ln A_ref``,

       ex_ref = 1 + (ex_src - 1) * r                                     (T)

   which follows from ``p_ref = F/A_ref``, ``p_src = F/A_src``: the *force* is
   convention-free but the *pressure* is not, and a constant offset ``p_src =
   p_ref / k`` only rescales ``p0``.  Only an area dependence of the ratio
   (``r != 1``) shifts the exponent.

   The physically motivated case is the damage-zone convention with a
   *material* process-zone width ``delta`` (independent of footprint size,
   ``lambda = P/sqrt(A)`` = perimeter factor):

       A_d(A_c) = A_c + lambda * delta * sqrt(A_c)
                = A_c * (1 + t),              t = lambda*delta/sqrt(A_c)
       r(A_c)   = (1 + t/2) / (1 + t)   <  1

   so a damaged-zone exponent maps to a *larger* nominal-contact exponent
   (``ex_c = 1 + (ex_d - 1) * r``).  The shift is bounded by
   ``|ex_c - ex_d| <= (1 - ex) * t/2`` and vanishes for large footprints --
   i.e. the area definition matters only at the *small* end of the area range.

Consequences used by the calibration
====================================
* Every library entry must declare an ``area_definition``; entries are mapped
  to the reference convention with (T) before pooling.
* The mapping needs the process-zone parameter ``t``.  It is *not* measured in
  this repository, so it is carried as an explicit, swept assumption
  (``DEFAULT_DAMAGE_RATIO_SWEEP``) and the pooled result is reported as a band
  over that sweep.  No number in this module is a measured quantity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Mapping, Sequence, Tuple

#: Reference convention of the library: projected (normal) contact area of the
#: indenter / load patch on the structure.
REFERENCE_CONVENTION = "nominal_projected_contact"

#: Controlled vocabulary of area definitions.
AREA_CONVENTIONS: Dict[str, str] = {
    "nominal_projected_contact": (
        "projected (normal) contact area of the indenter or load patch; "
        "reference convention of this library"
    ),
    "damaged_zone_area": (
        "nominal contact area widened by the crushing/damaged process zone, "
        "A_d = A_c + lambda*delta*sqrt(A_c)"
    ),
    "line_load_area": (
        "area per metre of structural width (A = b * 1 m): a constant rescaling "
        "of the contact area, hence exponent-invariant"
    ),
    "panel_nominal_area": (
        "nominal structural panel area over which a global load is averaged; "
        "constant factor wrt the contact area at fixed footprint"
    ),
    "wetted_surface_area": (
        "surface-following (wetted) contact area embracing roughness/ridges; "
        "grows faster than the projected area with footprint size"
    ),
}

#: Conventions that are a *constant* rescaling of the reference area, i.e. for
#: which (T) is the identity.
CONSTANT_RATIO_CONVENTIONS: Tuple[str, ...] = (
    "line_load_area",
    "panel_nominal_area",
)

#: Default sweep of the dimensionless process-zone width ``delta/sqrt(A_c)``
#: (adopted assumption, not a measurement).  ``t = lambda * delta/sqrt(A_c)``.
DEFAULT_DAMAGE_RATIO_SWEEP: Tuple[float, ...] = (0.0, 0.05, 0.10, 0.20, 0.30)


class ConventionError(ValueError):
    """Raised on an unknown area convention or an ill-posed transfer."""


def check_convention(name: str) -> str:
    if name not in AREA_CONVENTIONS:
        raise ConventionError(
            f"unknown area convention {name!r}; known: {sorted(AREA_CONVENTIONS)}"
        )
    return name


# ---------------------------------------------------------------------------
# log-derivative ratio r(A_ref) = d ln A_src / d ln A_ref
# ---------------------------------------------------------------------------


def damage_ratio_factor(area_m2: float, *, lambda_perimeter: float, delta_over_sqrtA: float) -> float:
    """``r = d ln A_d / d ln A_c`` for the damage-zone convention.

    ``t = lambda * delta/sqrt(A_c)`` with ``delta = delta_over_sqrtA * sqrt(A_c)``
    is the dimensionless ring width; ``r = (1 + t/2)/(1 + t) < 1``.
    """
    if area_m2 <= 0:
        raise ConventionError(f"area must be > 0, got {area_m2}")
    if lambda_perimeter <= 0:
        raise ConventionError(f"lambda_perimeter must be > 0, got {lambda_perimeter}")
    if delta_over_sqrtA < 0:
        raise ConventionError(f"delta_over_sqrtA must be >= 0, got {delta_over_sqrtA}")
    t = lambda_perimeter * delta_over_sqrtA  # == lambda*delta/sqrt(A_c), A-independent
    return (1.0 + 0.5 * t) / (1.0 + t)


def wetted_ratio_factor(area_m2: float, *, roughness_exponent: float) -> float:
    """``r`` for the wetted-surface convention: ``A_w = A_c * (A_c/A0)**beta``.

    ``r = 1 + beta``.  The exponent shift is then ``(ex - 1) * beta``.
    """
    if area_m2 <= 0:
        raise ConventionError(f"area must be > 0, got {area_m2}")
    if roughness_exponent < 0:
        raise ConventionError(
            f"roughness_exponent must be >= 0, got {roughness_exponent}"
        )
    return 1.0 + roughness_exponent


def ratio_factor(
    convention: str,
    area_m2: float,
    *,
    lambda_perimeter: float = 4.0,
    delta_over_sqrtA: float = 0.0,
    roughness_exponent: float = 0.0,
) -> float:
    """``r = d ln A_src / d ln A_ref`` for ``convention`` at a contact area."""
    check_convention(convention)
    if convention in CONSTANT_RATIO_CONVENTIONS:
        return 1.0
    if convention == "damaged_zone_area":
        return damage_ratio_factor(
            area_m2,
            lambda_perimeter=lambda_perimeter,
            delta_over_sqrtA=delta_over_sqrtA,
        )
    if convention == "wetted_surface_area":
        return wetted_ratio_factor(area_m2, roughness_exponent=roughness_exponent)
    # REFERENCE_CONVENTION
    return 1.0


# ---------------------------------------------------------------------------
# transfer law (T)
# ---------------------------------------------------------------------------


def to_reference(
    ex_src: float, convention: str, area_m2: float, **geometry: float
) -> float:
    """Map ``ex_src`` measured under ``convention`` to the reference convention.

    ``ex_ref = 1 + (ex_src - 1) * r``   (law T)

    ``geometry`` are the parameters of :func:`ratio_factor`.
    """
    r = ratio_factor(convention, area_m2, **geometry)
    return 1.0 + (ex_src - 1.0) * r


def from_reference(
    ex_ref: float, convention: str, area_m2: float, **geometry: float
) -> float:
    """Inverse of :func:`to_reference`."""
    r = ratio_factor(convention, area_m2, **geometry)
    return 1.0 + (ex_ref - 1.0) / r


def transfer_sensitivity(ex_src: float, convention: str, area_m2: float, **geometry: float) -> float:
    """``d ex_ref / d ex_src`` = r, i.e. the amplification of an ex uncertainty."""
    return ratio_factor(convention, area_m2, **geometry)


@dataclass(frozen=True)
class PressureAreaLaw:
    """Consumer view of the calibrated relation ``p = p0 * A**-ex``.

    ``convention`` is *mandatory*: an exponent quoted without its area
    definition is not consumable by this library.
    """

    p0_MPa: float
    ex: float
    convention: str = REFERENCE_CONVENTION
    p0_band_MPa: Tuple[float, float] | None = None
    ex_band: Tuple[float, float] | None = None
    source: str = "ice_pressure_area.library"

    def __post_init__(self) -> None:
        check_convention(self.convention)
        if self.p0_MPa <= 0:
            raise ConventionError(f"p0_MPa must be > 0, got {self.p0_MPa}")
        if not 0.0 <= self.ex <= 2.0:
            raise ConventionError(f"ex out of physical range [0,2]: {self.ex}")
        for band, name, nominal in (
            (self.p0_band_MPa, "p0_band_MPa", self.p0_MPa),
            (self.ex_band, "ex_band", self.ex),
        ):
            if band is not None:
                lo, hi = band
                if not lo <= nominal <= hi:
                    raise ConventionError(
                        f"{name}={band} must bracket the nominal value {nominal}"
                    )

    def pressure_MPa(self, area_m2: float) -> float:
        if area_m2 <= 0:
            raise ConventionError(f"area must be > 0, got {area_m2}")
        return self.p0_MPa * area_m2 ** (-self.ex)

    def pressure_band_MPa(self, area_m2: float) -> Tuple[float, float]:
        """Pressure band implied by the ``p0`` and ``ex`` bands (corner values)."""
        if self.p0_band_MPa is None or self.ex_band is None:
            raise ConventionError("both p0_band_MPa and ex_band are required")
        vals = [
            p0 * area_m2 ** (-ex)
            for p0 in self.p0_band_MPa
            for ex in self.ex_band
        ]
        return (min(vals), max(vals))

    def demand_bandwidth(self, area_ratio: float) -> float:
        """Ratio of peak to minimum pressure over an area range of ``area_ratio``.

        ``p(A_lo)/p(A_hi) = area_ratio**ex`` -- the quantity the ship-structure
        demand band (1.32x-1.74x in the gap) is made of.
        """
        if area_ratio < 1.0:
            raise ConventionError(f"area_ratio must be >= 1, got {area_ratio}")
        return area_ratio ** self.ex

    def demand_bandwidth_band(self, area_ratio: float) -> Tuple[float, float]:
        if self.ex_band is None:
            raise ConventionError("ex_band is required")
        return (
            area_ratio ** self.ex_band[0],
            area_ratio ** self.ex_band[1],
        )


def required_area_ratio(
    bandwidth_pair: Sequence[float], ex_pair: Sequence[float]
) -> float:
    """Area ratio that best reproduces a measured demand-bandwidth pair.

    Given ``bandwidth_pair = (b_lo, b_hi)`` measured with an exponent band
    ``ex_pair = (ex_lo, ex_hi)``, each end is inverted with ``r = b**(1/ex)`` and
    the area ratio is reported as the geometric mean of the two solutions.  A
    small spread between the two solutions is the consistency check that the
    declared pinch of bandwidth really is a single ``(A_hi/A_lo)**ex`` relation.
    """
    if len(bandwidth_pair) != 2 or len(ex_pair) != 2:
        raise ConventionError(
            f"expected 2-tuples, got {bandwidth_pair!r} and {ex_pair!r}"
        )
    b_lo, b_hi = bandwidth_pair
    ex_lo, ex_hi = ex_pair
    if min(b_lo, b_hi) <= 0 or ex_lo <= 0 or ex_hi <= 0:
        raise ConventionError("bandwidths and exponents must be positive")
    r_lo = b_lo ** (1.0 / ex_lo)
    r_hi = b_hi ** (1.0 / ex_hi)
    return float(math.sqrt(r_lo * r_hi))


def required_area_ratio_solutions(
    bandwidth_pair: Sequence[float], ex_pair: Sequence[float]
) -> Tuple[float, float]:
    """The two per-end solutions ``r = b**(1/ex)`` (spread = inconsistency)."""
    b_lo, b_hi = bandwidth_pair
    ex_lo, ex_hi = ex_pair
    return (float(b_lo ** (1.0 / ex_lo)), float(b_hi ** (1.0 / ex_hi)))
