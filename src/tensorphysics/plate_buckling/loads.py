"""Ice patch load (IACS UR I geometry) and lateral water pressure load models.

Ice load idealisation
---------------------
IACS Unified Requirements for Polar Class ships (UR I2/I3, mirrored by the IMO
Polar Code) describe the design ice load on shell plating as a rectangular
load patch whose **height is 0.6 times the design ice thickness** and whose
aspect ratio is close to 4:1 (a line-like patch on the ice belt).  The patch is
applied on the plate panel at its most unfavourable position (panel centre).

The *magnitude* of the design ice pressure is a class- and location-dependent
input (typical ice-belt values for PC4/PC5 forebodies are of the order of
4-6 MPa); it is therefore an explicit parameter of
:func:`ice_patch_pressure` rather than a hard-coded number, so that the ULS
utilisation reported by :mod:`tensorphysics.plate_buckling.uls_check` can be
recomputed for any Polar Class.

The water pressure is a quasi-static uniform lateral pressure representing the
design draught head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .core import PlateGeometry

__all__ = ["IcePatchLoad", "ice_patch_pressure", "combined_lateral_pressure"]

#: IACS UR I load-patch height factor (patch height = 0.6 h_ice).
PATCH_HEIGHT_FACTOR = 0.6
#: IACS UR I load-patch aspect ratio (width / height).
PATCH_ASPECT_RATIO = 4.0


@dataclass(frozen=True)
class IcePatchLoad:
    """Rectangular ice load patch on the plate panel.

    Parameters
    ----------
    h_ice:
        Design ice thickness (m).
    p_ice:
        Nominal ice pressure over the patch (Pa).
    patch_height_factor, patch_aspect:
        IACS UR I patch geometry factors.
    """

    h_ice: float
    p_ice: float
    patch_height_factor: float = PATCH_HEIGHT_FACTOR
    patch_aspect: float = PATCH_ASPECT_RATIO

    @property
    def height(self) -> float:
        return self.patch_height_factor * self.h_ice

    @property
    def width(self) -> float:
        return self.patch_aspect * self.height

    def box(self, geom: PlateGeometry) -> Tuple[float, float, float, float]:
        """Patch rectangle ``(x0, x1, y0, y1)`` centred on the panel."""
        hw, hh = 0.5 * min(self.width, geom.a), 0.5 * min(self.height, geom.b)
        xc, yc = 0.5 * geom.a, 0.5 * geom.b
        return (xc - hw, xc + hw, yc - hh, yc + hh)

    def nodal_load(self, model) -> np.ndarray:
        """Consistent transverse nodal load vector of the patch."""
        return model.pressure_load(self.p_ice, boxes=[self.box(model.geom)])

    def as_dict(self) -> Dict[str, float]:
        return {
            "h_ice": float(self.h_ice),
            "p_ice": float(self.p_ice),
            "patch_height": float(self.height),
            "patch_width": float(self.width),
            "patch_aspect": float(self.patch_aspect),
        }


def ice_patch_pressure(h_ice: float, p_ref: float = 5.0e6,
                       h_ref: float = 1.0) -> float:
    """Nominal design ice pressure on the patch.

    A bilinear thickness scaling ``p = p_ref * min(1, h_ref / h_ice)`` is used:
    the patch pressure is capped for thick ice and increases for thin ice, which
    reproduces the qualitative behaviour of the UR I pressure-thickness relation
    with an explicitly stated reference point ``(h_ref, p_ref)``.
    """
    if h_ice <= 0.0:
        raise ValueError("h_ice must be positive")
    return float(p_ref * min(1.0, h_ref / h_ice))


def combined_lateral_pressure(model, h_ice: float, p_ice: float,
                              p_water: float, scale_ice: float = 1.0
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(constant_part, ice_part)`` nodal loads.

    Total lateral load is ``constant_part + scale_ice * ice_part``: the water
    pressure is held constant while the ice patch load is scaled, which is the
    appropriate continuation for the ULS check (the ice load is the variable
    action, the still-water pressure the permanent one).
    """
    ice = IcePatchLoad(h_ice=h_ice, p_ice=p_ice)
    f_water = model.pressure_load(p_water) if p_water else np.zeros(model.ndof)
    f_ice = ice.nodal_load(model)
    return f_water, f_ice