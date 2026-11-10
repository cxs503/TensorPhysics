"""Ice-underside roughness-length (z0) closure from keel morphology.

Physical model
--------------
Sea-ice drift is resisted by the shear stress at the ice/water interface.  The
magnitude of that stress is set by the *under-ice roughness length* ``z0`` which
is produced by the geometry of the ice underside: pressure-ridge keels, rubble
fields and small-scale bottom relief.  This module builds ``z0`` from keel
statistics (mean keel depth and mean keel spacing) instead of prescribing a
constant.

The surface is treated as a two-class mosaic (flux-matching homogenisation, the
standard way of assigning an *effective* roughness length to a heterogeneous
surface):

1. **Undeformed ice class** -- smooth first-year ice underside, roughness length
   ``z0_skin`` (default 5 mm, the value used in under-ice boundary-layer work;
   see McPhee 2008, ch. 2).
2. **Ridge/keel class** -- a roughness element whose height is the keel relief
   below the surrounding ice underside, ``h_e = d_k - h``, converted to a
   roughness length with the Nikuradse fully-rough result ``z0 = k_s/30``, i.e.
   ``z0_ridge = h_e / 30`` (capped at ``z0_ceiling``).

The keel class covers a fractional area

    A_r = min(1, lambda_w * w_k / L_s),      w_k = 0.82 * d_k                (1)

with ``w_k`` the keel width.  The width-to-depth ratio 0.82 follows from the
Timco & Burden (1997) first-year-ridge regressions ``d_k = 4.4 h_s`` and
``w_k = 3.6 h_s``.  ``lambda_w`` is the *wake/form-drag amplification*: the
separated-flow footprint that a keel imposes on the boundary layer is wider than
the ice contact width because the wake recovers over a finite downstream
distance.  ``lambda_w`` is the single calibrated parameter of the module
(calibrated in ``scripts/ice_drift_roughness_validation.py`` against the
Beaufort Sea under-ice drag-coefficient anchor set).

The effective drag coefficient of the ice underside is the area-weighted
(averaged momentum-flux) mixture of the two classes,

    C_d_ice = A_r * C_d( z0_ridge ) + (1 - A_r) * C_d( z0_skin )              (2)

with the logarithmic law-of-the-wall drag coefficient

    C_d(z0) = [ kappa / ln(z_ref / z0) ] ** 2                                 (3)

and the reported roughness length is the log-law-consistent inverse of (2)-(3),

    z0_ice = z_ref * exp( -kappa / sqrt(C_d_ice) ).                           (4)

Equation (2) is expressed in *drag* (stress) space rather than in log-z0 space
because on a mosaic surface the momentum flux -- not the logarithm of z0 -- is
what is area-averaged; inverting (3) then yields the roughness length the
boundary layer "sees".  Both formulations coincide in the single-class limits.

Mixing with open water (leads) is handled by
:func:`surface_interface_roughness`: the total-area momentum flux is
``C * C_d_ice + (1 - C) * C_d_water``, which recovers the open-water value
exactly as ``C -> 0``.

References
----------
* McPhee, M.G. (2008) *Air-Ice-Ocean Interaction: Turbulent Ocean Boundary
  Layer Exchange*, Springer. (under-ice log layer, z0 of 5 mm for smooth FY ice)
* Timco, G.W. & Burden, R.P. (1997) "An analysis of the shapes of sea ice
  ridges", Cold Regions Science and Technology 25, 65-77.
* Nikuradse, J. (1933) "Stroemungsgesetze in rauhen Rohren", VDI-Forschungsheft
  361. (fully rough wall: z0 = k_s / 30)
* Schlichting, H. & Gersten, K. (2000) *Boundary-Layer Theory*, 8th ed.
* Lu, P., Li, Z., Cheng, B. & Lepparanta, M. (2011) "A parameterization of the
  ice-ocean drag coefficient", J. Geophys. Res. 116, C07019. (keel-geometry
  control of the under-ice drag coefficient)

Units: SI throughout (m, s, kg, Pa).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Constants (all documented in the module docstring / PARAMETER_PROVENANCE)
# ---------------------------------------------------------------------------
KAPPA = 0.40                    # von Karman constant, -
Z_REF = 10.0                    # reference depth below the ice underside, m
NU_SEAWATER = 1.83e-6           # kinematic viscosity of seawater at 0 C, m^2/s
Z0_SKIN_DEFAULT = 0.005         # smooth FY ice underside, m (McPhee 2008)
Z0_CEILING_DEFAULT = 0.25       # observed under-ice z0 rarely exceeds ~0.25 m
Z0_WATER_SURFACE = 1.0e-4       # open-water free surface, m (Schlichting)
KEEL_WIDTH_RATIO = 0.82         # w_k / d_k, Timco & Burden (1997)
ELEMENT_ROUGHNESS_RATIO = 30.0  # k_s / z0 for a fully rough wall (Nikuradse)
WAKE_AMPLIFICATION_DEFAULT = 2.0  # calibrated, see module docstring and report
CD_OPEN_WATER_REFERENCE = 1.3e-3  # Large & Pond (1981) / COARE neutral 10 m value

PARAMETER_PROVENANCE: Dict[str, str] = {
    "KAPPA": "von Karman constant 0.40 (standard, McPhee 2008)",
    "Z_REF": "under-ice drag reference depth 10 m (McPhee 2008 convention)",
    "NU_SEAWATER": "seawater at 0 C and 34 psu, 1.83e-6 m^2/s",
    "Z0_SKIN_DEFAULT": "smooth first-year ice underside 5 mm (McPhee 2008)",
    "Z0_CEILING_DEFAULT": "upper bound of reported under-ice z0, 0.25 m",
    "Z0_WATER_SURFACE": "open-water free surface roughness 0.1 mm",
    "KEEL_WIDTH_RATIO": "w_k/d_k = 3.6/4.4 from Timco & Burden (1997)",
    "ELEMENT_ROUGHNESS_RATIO": "Nikuradse fully rough wall, k_s = 30 z0",
    "WAKE_AMPLIFICATION_DEFAULT": "calibrated form-drag footprint amplification",
}


class IceDriftRoughnessError(ValueError):
    """Raised for physically inadmissible morphology / forcing inputs."""


# ---------------------------------------------------------------------------
# Log-law primitives
# ---------------------------------------------------------------------------
def log_law_drag_coefficient(
    z0_m: float, z_ref_m: float = Z_REF, kappa: float = KAPPA
) -> float:
    """``C_d = [kappa / ln(z_ref/z0)]**2`` -- logarithmic law of the wall.

    >>> round(log_law_drag_coefficient(1.0e-4), 6)
    0.001207
    """
    if not (0.0 < z0_m < z_ref_m):
        raise IceDriftRoughnessError(
            f"roughness length must satisfy 0 < z0 < z_ref, got z0={z0_m}"
        )
    return (kappa / math.log(z_ref_m / z0_m)) ** 2


def roughness_length_from_drag_coefficient(
    cd: float, z_ref_m: float = Z_REF, kappa: float = KAPPA
) -> float:
    """Inverse of :func:`log_law_drag_coefficient`."""
    if not (0.0 < cd < 1.0):
        raise IceDriftRoughnessError(f"drag coefficient out of range: {cd}")
    return z_ref_m * math.exp(-kappa / math.sqrt(cd))


def open_water_drag_coefficient(
    z_ref_m: float = Z_REF, kappa: float = KAPPA, z0_water_m: float = Z0_WATER_SURFACE
) -> float:
    """Open-water (lead/free-surface) drag coefficient, log-law value.

    With ``z0 = 0.1 mm`` and ``z_ref = 10 m`` this gives 1.207e-3, i.e. within
    ~7 % of the classical neutral 10 m bulk value of 1.3e-3 (Large & Pond 1981).
    """
    return log_law_drag_coefficient(z0_water_m, z_ref_m, kappa)


# ---------------------------------------------------------------------------
# Keel morphology
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class KeelMorphology:
    """Statistics of the ice-underside keel field.

    Parameters
    ----------
    keel_depth_m : mean (or representative) keel depth below the waterline, m.
    keel_spacing_m : mean spacing between keel crest lines, m.
    z0_skin_m : roughness length of undeformed (smooth) ice underside, m.
    wake_amplification : form-drag footprint amplification ``lambda_w``.
    element_roughness_ratio : ``k_s / z0`` for the keel elements (30).
    z0_ceiling_m : physical upper bound of the keel-class roughness length, m.
    """

    keel_depth_m: float
    keel_spacing_m: float
    z0_skin_m: float = Z0_SKIN_DEFAULT
    wake_amplification: float = WAKE_AMPLIFICATION_DEFAULT
    element_roughness_ratio: float = ELEMENT_ROUGHNESS_RATIO
    z0_ceiling_m: float = Z0_CEILING_DEFAULT

    def __post_init__(self) -> None:
        if not (math.isfinite(self.keel_depth_m) and self.keel_depth_m > 0.0):
            raise IceDriftRoughnessError(f"keel_depth_m must be > 0, got {self.keel_depth_m}")
        if not (math.isfinite(self.keel_spacing_m) and self.keel_spacing_m > 0.0):
            raise IceDriftRoughnessError(
                f"keel_spacing_m must be > 0, got {self.keel_spacing_m}"
            )
        if not (math.isfinite(self.z0_skin_m) and self.z0_skin_m > 0.0):
            raise IceDriftRoughnessError(f"z0_skin_m must be > 0, got {self.z0_skin_m}")
        if not (self.wake_amplification > 0.0):
            raise IceDriftRoughnessError("wake_amplification must be > 0")
        if not (self.element_roughness_ratio > 1.0):
            raise IceDriftRoughnessError("element_roughness_ratio must be > 1")

    @property
    def keel_width_m(self) -> float:
        """Keel width ``w_k = 0.82 d_k`` (Timco & Burden 1997)."""
        return KEEL_WIDTH_RATIO * self.keel_depth_m

    def relief_m(self, h_ice_m: float) -> float:
        """Keel relief below the surrounding ice underside, ``d_k - h``."""
        return max(0.0, self.keel_depth_m - h_ice_m)

    def area_fraction(self, h_ice_m: float) -> float:
        """Keel-class area fraction ``A_r`` of equation (1)."""
        if self.relief_m(h_ice_m) <= 0.0:
            return 0.0
        raw = self.wake_amplification * self.keel_width_m / self.keel_spacing_m
        return min(1.0, raw)

    def element_roughness_length(self, h_ice_m: float) -> float:
        """Keel-class roughness length ``min(z0_ceiling, relief/30)``."""
        relief = self.relief_m(h_ice_m)
        if relief <= 0.0:
            return self.z0_skin_m
        return min(self.z0_ceiling_m, relief / self.element_roughness_ratio)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "keel_depth_m": self.keel_depth_m,
            "keel_spacing_m": self.keel_spacing_m,
            "keel_width_m": self.keel_width_m,
            "z0_skin_m": self.z0_skin_m,
            "wake_amplification": self.wake_amplification,
            "element_roughness_ratio": self.element_roughness_ratio,
            "z0_ceiling_m": self.z0_ceiling_m,
        }


@dataclass(frozen=True)
class IceUndersideRoughness:
    """Roughness state of the ice underside (per unit ice area)."""

    area_fraction: float     # A_r, keel-class area fraction
    z0_skin_m: float         # undeformed-class roughness length
    z0_ridge_m: float        # keel-class roughness length
    drag_coefficient: float  # C_d_ice, equation (2)
    z0_m: float              # z0_ice, equation (4)
    z_ceiling_active: bool   # True if the keel-class z0 was capped

    def as_dict(self) -> Dict[str, Any]:
        return {
            "area_fraction": self.area_fraction,
            "z0_skin_m": self.z0_skin_m,
            "z0_ridge_m": self.z0_ridge_m,
            "drag_coefficient": self.drag_coefficient,
            "z0_ice_m": self.z0_m,
            "z0_ceiling_active": self.z_ceiling_active,
        }


def ice_underside_roughness(
    morphology: KeelMorphology,
    h_ice_m: float,
    z_ref_m: float = Z_REF,
    kappa: float = KAPPA,
) -> IceUndersideRoughness:
    """Assemble the two-class mosaic roughness of the ice underside.

    Raises :class:`IceDriftRoughnessError` for non-positive ice thickness.
    """
    if not (math.isfinite(h_ice_m) and h_ice_m > 0.0):
        raise IceDriftRoughnessError(f"h_ice_m must be > 0, got {h_ice_m}")

    z0_ridge = morphology.element_roughness_length(h_ice_m)
    area_fraction = morphology.area_fraction(h_ice_m)
    z0_skin = min(morphology.z0_skin_m, z0_ridge)

    cd_ridge = log_law_drag_coefficient(z0_ridge, z_ref_m, kappa)
    cd_skin = log_law_drag_coefficient(z0_skin, z_ref_m, kappa)
    cd_ice = area_fraction * cd_ridge + (1.0 - area_fraction) * cd_skin
    z0_ice = roughness_length_from_drag_coefficient(cd_ice, z_ref_m, kappa)

    ceiling_active = (
        morphology.relief_m(h_ice_m) > 0.0
        and morphology.relief_m(h_ice_m) / morphology.element_roughness_ratio
        > morphology.z0_ceiling_m
    )
    return IceUndersideRoughness(
        area_fraction=area_fraction,
        z0_skin_m=z0_skin,
        z0_ridge_m=z0_ridge,
        drag_coefficient=cd_ice,
        z0_m=z0_ice,
        z_ceiling_active=ceiling_active,
    )


def surface_interface_roughness(
    ice_roughness: IceUndersideRoughness,
    concentration: float,
    z_ref_m: float = Z_REF,
    kappa: float = KAPPA,
    z0_water_m: float = Z0_WATER_SURFACE,
) -> Dict[str, float]:
    """Area-averaged drag coefficient / roughness length of the ice+lead mosaic.

    ``C_d_surface = C * C_d_ice + (1 - C) * C_d_water``; the open-water limit
    ``C -> 0`` is therefore reproduced exactly (negative-control criterion C5).
    """
    if not (0.0 <= concentration <= 1.0):
        raise IceDriftRoughnessError(
            f"ice concentration must lie in [0, 1], got {concentration}"
        )
    cd_water = open_water_drag_coefficient(z_ref_m, kappa, z0_water_m)
    cd_surface = concentration * ice_roughness.drag_coefficient + (1.0 - concentration) * cd_water
    z0_surface = roughness_length_from_drag_coefficient(cd_surface, z_ref_m, kappa)
    return {
        "concentration": float(concentration),
        "cd_water": cd_water,
        "cd_surface": cd_surface,
        "z0_surface_m": z0_surface,
    }