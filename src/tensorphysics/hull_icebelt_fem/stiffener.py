"""Ice-belt longitudinal stiffener check (beam theory, closed form).

The plate panel (:mod:`tensorphysics.hull_icebelt_fem.panel_fem`) carries the
local ice patch; the longitudinal stiffener carries the same ice pressure over
its tributary breadth ``s`` as a line load ``q = p s`` [N/m] over the span
between web frames.  The stiffener is a simple beam, so the standard
Euler-Bernoulli results apply

.. math::

    M_{max} = \\frac{q L^2}{8},\\quad
    \\delta_{max} = \\frac{5 q L^4}{384 E I},\\quad
    \\sigma = \\frac{M}{Z}

with the section built from web + flange + the attached plate strip of
effective breadth equal to the stiffener spacing.  This is deliberately a
*closed form* (not an FE result): it is the second, independent leg of the
stiffener check, and the report states it as such.

The attached plate is in tension when the ice pressure acts outboard->inboard
on the shell; the sign convention of the reported stresses is:
``sigma_plate`` positive in tension, ``sigma_flange`` positive in compression.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

__all__ = ["StiffenerProfile", "stiffener_response"]


@dataclass(frozen=True)
class StiffenerProfile:
    """T-bar / flat-bar longitudinal with attached plating.

    Parameters
    ----------
    web_h, web_t:
        Web height above the plate and web thickness [m].
    flange_b, flange_t:
        Flange breadth and thickness [m]; ``flange_b = 0`` gives a flat bar.
    spacing:
        Stiffener spacing ``s`` (tributary plate breadth) [m].
    t_plate:
        Plate thickness [m].
    span:
        Stiffener span between web frames ``L`` [m].
    E:
        Steel modulus [Pa].
    """

    web_h: float
    web_t: float
    flange_b: float
    flange_t: float
    spacing: float
    t_plate: float
    span: float
    E: float = 2.06e11

    def __post_init__(self) -> None:
        if self.web_h <= 0.0 or self.web_t <= 0.0:
            raise ValueError("web height and thickness must be positive")
        if self.flange_b < 0.0 or self.flange_t < 0.0:
            raise ValueError("flange dimensions must be non-negative")
        if (self.flange_b == 0.0) != (self.flange_t == 0.0):
            raise ValueError("flange breadth and thickness must be given together")
        for name in ("spacing", "t_plate", "span", "E"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")

    # ------------------------------------------------------------- section
    def section(self) -> Dict[str, float]:
        """Area, centroid, second moment and section moduli (about y-y).

        The origin is the plate mid-thickness; positive ``z`` points towards
        the flange (inboard, i.e. away from the ice).
        """
        # (area, centroid z, own I about own centroid)
        parts = [(self.spacing * self.t_plate, 0.0,
                  self.spacing * self.t_plate ** 3 / 12.0),
                 (self.web_h * self.web_t, self.t_plate / 2.0 + self.web_h / 2.0,
                  self.web_t * self.web_h ** 3 / 12.0)]
        if self.flange_b > 0.0:
            parts.append((self.flange_b * self.flange_t,
                          self.t_plate / 2.0 + self.web_h + self.flange_t / 2.0,
                          self.flange_b * self.flange_t ** 3 / 12.0))
        area = sum(p[0] for p in parts)
        zbar = sum(p[0] * p[1] for p in parts) / area
        inertia = sum(p[2] + p[0] * (p[1] - zbar) ** 2 for p in parts)
        z_top = (self.t_plate / 2.0 + self.web_h
                 + (self.flange_t if self.flange_b > 0.0 else 0.0)) - zbar
        z_bot = zbar + self.t_plate / 2.0
        return {
            "area_m2": float(area),
            "centroid_m": float(zbar),
            "I_m4": float(inertia),
            "Z_plate_m3": float(inertia / z_bot),
            "Z_flange_m3": float(inertia / z_top),
            "EI_Nm2": float(self.E * inertia),
        }

    # ------------------------------------------------------------ response
    def response(self, line_load: float) -> Dict[str, float]:
        """Bending moment, surface stresses and deflection under ``q`` [N/m].

        Raises
        ------
        ValueError
            For a negative line load.
        """
        q = float(line_load)
        if q < 0.0:
            raise ValueError(f"line load must be non-negative, got {line_load!r}")
        sec = self.section()
        M = q * self.span ** 2 / 8.0
        delta = 5.0 * q * self.span ** 4 / (384.0 * sec["EI_Nm2"])
        return {
            "line_load_N_per_m": q,
            "M_max_Nm": float(M),
            "delta_max_m": float(delta),
            "delta_max_mm": float(delta) * 1e3,
            "delta_over_span": float(delta / self.span),
            "sigma_plate_Pa": float(M / sec["Z_plate_m3"]),
            "sigma_flange_Pa": float(M / sec["Z_flange_m3"]),
            "sigma_plate_MPa": float(M / sec["Z_plate_m3"]) / 1e6,
            "sigma_flange_MPa": float(M / sec["Z_flange_m3"]) / 1e6,
            **sec,
        }


def stiffener_response(profile: StiffenerProfile, pressure: float,
                       spacing: float) -> Dict[str, float]:
    """Convenience wrapper: pressure [Pa] on tributary breadth -> response."""
    if pressure < 0.0:
        raise ValueError(f"pressure must be non-negative, got {pressure!r}")
    if spacing <= 0.0:
        raise ValueError(f"spacing must be positive, got {spacing!r}")
    out = profile.response(pressure * spacing)
    out["pressure_Pa"] = float(pressure)
    out["spacing_m"] = float(spacing)
    return out