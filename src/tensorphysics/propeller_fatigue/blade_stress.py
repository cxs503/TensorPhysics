"""Blade geometry, section properties and root stress components.

Section idealisation: aerofoil-like section of chord ``c`` and maximum thickness
``t`` treated as an ellipse for the second moments of area, with a section area
coefficient of 0.70 for mass and centrifugal terms::

    I_chord  = pi/64 c t^3      (flatwise bending, driven by axial/thrust force)
    I_thick  = pi/64 c^3 t      (edgewise bending, driven by tangential/torque)
    W_chord  = pi/32 c t^2 ,  W_thick = pi/32 c^2 t

The three stress components at the blade root are superposed on the critical
outer fibre: axial bending (flatwise), tangential bending (edgewise) and
centrifugal tension.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .hydro_unsteady import BladeStations

SECTION_AREA_COEFF = 0.70


@dataclass
class BladeGeometry:
    """Reference ice-class propeller (documented in docs/reports/m1135)."""

    D: float = 4.50                    # diameter [m]
    r_hub: float = 0.50                # hub radius [m]
    Z: int = 4
    pitch_ratio: float = 1.00          # P/D
    c_root: float = 1.05               # chord at r/R = 0.25 [m]
    chord_taper: float = 0.50          # chord reduction from root to tip
    tc_root: float = 0.24              # thickness ratio at r/R = 0.25
    tc_tip: float = 0.05
    r_hat_root: float = 0.25
    r_hat_tip: float = 0.98
    n_stations: int = 24

    @property
    def R(self) -> float:
        return self.D / 2.0

    @property
    def span(self) -> float:
        return self.R - self.r_hub

    # ------------------------------------------------------------ radial law
    def chord(self, r):
        r = np.asarray(r, dtype=float)
        x = (r / self.R - self.r_hat_root) / (self.r_hat_tip - self.r_hat_root)
        return self.c_root * (1.0 - self.chord_taper * np.clip(x, 0.0, 1.2))

    def thickness(self, r):
        r = np.asarray(r, dtype=float)
        x = (r / self.R - self.r_hat_root) / (self.r_hat_tip - self.r_hat_root)
        tc = self.tc_root + (self.tc_tip - self.tc_root) * np.clip(x, 0.0, 1.2)
        return tc * self.chord(r)

    def pitch_angle(self, r):
        r = np.asarray(r, dtype=float)
        return np.arctan(self.pitch_ratio * self.D / (2.0 * np.pi * r))

    def area(self, r):
        return SECTION_AREA_COEFF * self.chord(r) * self.thickness(r)

    def w_flat(self, r):
        return np.pi / 32.0 * self.chord(r) * self.thickness(r) ** 2

    def w_edge(self, r):
        return np.pi / 32.0 * self.chord(r) ** 2 * self.thickness(r)

    def i_flat(self, r):
        return np.pi / 64.0 * self.chord(r) * self.thickness(r) ** 3

    def i_edge(self, r):
        return np.pi / 64.0 * self.chord(r) ** 3 * self.thickness(r)

    def planform_area(self, n: int = 240) -> float:
        """Developed (planform) blade area ``int c dr`` between root and tip."""
        r = np.linspace(self.r_hat_root * self.R, self.r_hat_tip * self.R, n)
        return float(np.trapezoid(self.chord(r), r))

    def expanded_area_ratio(self) -> float:
        """Expanded area ratio ``Z A_p / (pi R^2)``."""
        return self.planform_area() * self.Z / (np.pi * self.R ** 2)

    def stations(self, r_hat_min: float = 0.26, r_hat_max: float = 0.97) -> BladeStations:
        r = np.linspace(r_hat_min * self.R, r_hat_max * self.R, self.n_stations)
        return BladeStations(r=r, c=self.chord(r), beta=self.pitch_angle(r),
                             r_root=self.r_hub, z_blades=self.Z)

    # --------------------------------------------------------- root values
    def root_radius(self) -> float:
        return self.r_hat_root * self.R

    def root_properties(self) -> Dict[str, float]:
        rr = self.root_radius()
        return {
            "r_root_m": rr,
            "chord_m": float(self.chord(rr)),
            "thickness_m": float(self.thickness(rr)),
            "W_flat_m3": float(self.w_flat(rr)),
            "W_edge_m3": float(self.w_edge(rr)),
            "area_m2": float(self.area(rr)),
            "I_flat_m4": float(self.i_flat(rr)),
            "I_edge_m4": float(self.i_edge(rr)),
            "planform_area_m2": self.planform_area(),
            "expanded_area_ratio": self.expanded_area_ratio(),
        }

    def equivalent_beam_section(self, r_hat: float = 0.5, rho: float = 7640.0) -> Dict[str, float]:
        """Equivalent uniform beam properties (flatwise) at ``r_hat``."""
        r = r_hat * self.R
        area = SECTION_AREA_COEFF * self.chord(r) * self.thickness(r)
        return {
            "I": float(self.i_flat(r)),
            "c_half": float(0.5 * self.thickness(r)),
            "area_m2": float(area),
            "mass_per_length": float(area * rho),
        }


def root_stress_components(geom: BladeGeometry, material, n_rps: float,
                           M_axial: float = 0.0, M_tangential: float = 0.0
                           ) -> Dict[str, float]:
    """Root stress components [Pa] for given axial/tangential bending moments.

    ``M_axial`` = bending moment produced by the axial (thrust / ice axial)
    force, ``M_tangential`` = bending moment produced by the tangential
    (torque / ice tangential) force.
    """
    props = geom.root_properties()
    sigma_ax = abs(M_axial) / props["W_flat_m3"]
    sigma_tan = abs(M_tangential) / props["W_edge_m3"]
    r = geom.stations().r
    A = geom.area(r)
    omega = 2 * np.pi * n_rps
    cf_integral = np.trapezoid(material.rho * omega ** 2 * A * r, r)
    sigma_cf = cf_integral / props["area_m2"]
    return {
        "sigma_axial_bending": sigma_ax,
        "sigma_tangential_bending": sigma_tan,
        "sigma_centrifugal": sigma_cf,
        "sigma_combined_outer_fibre": sigma_ax + sigma_tan + sigma_cf,
    }