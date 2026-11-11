"""Fracture energy computation for ice plate bending fracture.

Fracture criterion
------------------
When the maximum bending stress σ_max exceeds the flexural strength σ_flex,
the plate fractures.  The fracture energy is:

    G_f = 2 · γ_ice × A_fracture

where γ_ice is the surface energy of ice and A_fracture is the created
fracture surface area.

For ice, the surface energy γ_ice ≈ 0.1 J/m² (pure ice at -10°C), but the
effective fracture energy G_f (including microcracking and other dissipative
processes) is typically 10–100 J/m² for first-year sea ice.

The fracture surface area for a through-thickness crack in a plate of
thickness h and crack length L is:

    A_fracture = h × L

For a radial crack pattern from a concentrated load, the total fracture
length is the sum of all crack segments.

References
----------
* Schulson, E.M. & Duval, P. (2009). *Creep and Fracture of Ice*,
  Cambridge University Press, Ch. 7.
* Dempsey, J.P. et al. (1999). "Scale effects in ice fracture",
  Cold Regions Science and Technology, 30(1-3), 111-123.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


#: Surface energy of pure ice at -10°C [J/m²]
GAMMA_ICE = 0.1

#: Typical effective fracture energy for first-year sea ice [J/m²]
G_F_ICE_TYPICAL = 50.0


@dataclass
class CrackSegment:
    """A single crack segment in the fracture pattern."""
    x0: float  # start x [m]
    y0: float  # start y [m]
    x1: float  # end x [m]
    y1: float  # end y [m]

    @property
    def length(self) -> float:
        return math.sqrt((self.x1 - self.x0) ** 2 + (self.y1 - self.y0) ** 2)


@dataclass
class FractureEnergy:
    """Fracture energy computation result."""
    gamma_ice: float          # surface energy [J/m²]
    crack_length_total: float # total crack length [m]
    plate_thickness: float    # plate thickness [m]
    fracture_area: float      # A_fracture = h × L [m²]
    G_f: float                # fracture energy per unit area [J/m²]
    E_fracture: float         # total fracture energy [J]
    sigma_max: float          # maximum bending stress [Pa]
    sigma_flex: float         # flexural strength [Pa]
    fractured: bool           # whether fracture occurred

    @property
    def G_f_theoretical(self) -> float:
        """Theoretical G_f = 2 · γ_ice [J/m²]."""
        return 2.0 * self.gamma_ice

    def as_dict(self) -> dict:
        return {
            "gamma_ice_J_m2": self.gamma_ice,
            "crack_length_total_m": self.crack_length_total,
            "plate_thickness_m": self.plate_thickness,
            "fracture_area_m2": self.fracture_area,
            "G_f_J_m2": self.G_f,
            "E_fracture_J": self.E_fracture,
            "sigma_max_Pa": self.sigma_max,
            "sigma_flex_Pa": self.sigma_flex,
            "fractured": self.fractured,
            "G_f_theoretical_J_m2": self.G_f_theoretical,
        }


def compute_fracture_energy(
    cracks: List[CrackSegment],
    h: float,
    sigma_max: float,
    sigma_flex: float,
    gamma_ice: float = GAMMA_ICE,
    G_f_effective: Optional[float] = None,
) -> FractureEnergy:
    """Compute the fracture energy for a given crack pattern.

    Parameters
    ----------
    cracks : list of CrackSegment
        The crack segments forming the fracture pattern.
    h : float
        Plate thickness [m].
    sigma_max : float
        Maximum bending stress in the plate [Pa].
    sigma_flex : float
        Flexural strength of ice [Pa].
    gamma_ice : float
        Surface energy of ice [J/m²].
    G_f_effective : float, optional
        Effective fracture energy (including microcracking).
        If None, uses 2·γ_ice (pure Griffith).

    Returns
    -------
    FractureEnergy
    """
    crack_length = sum(c.length for c in cracks)
    fracture_area = h * crack_length

    # Use effective G_f if provided, otherwise use Griffith value
    if G_f_effective is not None:
        G_f = G_f_effective
    else:
        G_f = 2.0 * gamma_ice

    E_fracture = G_f * fracture_area
    fractured = sigma_max >= sigma_flex

    return FractureEnergy(
        gamma_ice=gamma_ice,
        crack_length_total=crack_length,
        plate_thickness=h,
        fracture_area=fracture_area,
        G_f=G_f,
        E_fracture=E_fracture,
        sigma_max=sigma_max,
        sigma_flex=sigma_flex,
        fractured=fractured,
    )


def radial_crack_pattern(
    n_cracks: int = 4,
    crack_length: float = 1.0,
    centre: tuple = (0.0, 0.0),
) -> List[CrackSegment]:
    """Generate a radial crack pattern from a concentrated load.

    Parameters
    ----------
    n_cracks : int
        Number of radial cracks.
    crack_length : float
        Length of each crack [m].
    centre : tuple
        Centre of the crack pattern (x, y) [m].

    Returns
    -------
    list of CrackSegment
    """
    cracks = []
    for i in range(n_cracks):
        angle = 2.0 * math.pi * i / n_cracks
        x1 = centre[0] + crack_length * math.cos(angle)
        y1 = centre[1] + crack_length * math.sin(angle)
        cracks.append(CrackSegment(centre[0], centre[1], x1, y1))
    return cracks


def compute_max_bending_stress(
    P: float,
    a: float,
    b: float,
    h: float,
    E: float,
    nu: float,
    load_radius: Optional[float] = None,
) -> float:
    """Compute maximum bending stress for a simply supported rectangular plate.

    For a concentrated load P at the centre of a simply supported rectangular
    plate, the maximum bending stress occurs at the centre (bottom fibre).

    For a point load, the stress has a logarithmic singularity.  In practice,
    the load is distributed over a small circular area of radius c.  The
    maximum stress (Timoshenko & Woinowsky-Krieger, §43) is:

        σ_max = (3(1+ν)P) / (2πh²) · [ln(a/(2c)) + 0.5 + ...]

    For a square plate (a = b) with load radius c ≈ 0.05a:

        σ_max ≈ 0.75 · P / h²  (approximate)

    Parameters
    ----------
    P : float
        Concentrated load [N].
    a, b : float
        Plate dimensions [m].
    h : float
        Plate thickness [m].
    E : float
        Young's modulus [Pa].
    nu : float
        Poisson's ratio.
    load_radius : float, optional
        Radius of the load application area [m].
        Default: 0.05 × min(a, b).

    Returns
    -------
    float
        Maximum bending stress [Pa].
    """
    if load_radius is None:
        load_radius = 0.05 * min(a, b)

    # Timoshenko formula for maximum stress under a circular load patch
    # on a simply supported plate (approximate, valid for c/h > 0.5)
    # σ_max = (3(1+ν)P) / (2πh²) · [ln(a/(2c)) + 0.5 + (1-ν)/4]
    # This is for a square plate; for rectangular plates the coefficient varies

    aspect_ratio = max(a, b) / min(a, b)

    # Base coefficient for square plate
    base_coeff = 3.0 * (1.0 + nu) / (2.0 * math.pi)

    # Logarithmic term (stress concentration due to small load area)
    log_term = math.log(min(a, b) / (2.0 * load_radius))

    # Additional terms from plate theory
    additional = 0.5 + (1.0 - nu) / 4.0

    # Aspect ratio correction (plates with a/b > 1 have higher stress)
    ar_correction = 1.0 + 0.1 * (aspect_ratio - 1.0)

    sigma_max = base_coeff * (P / h ** 2) * (log_term + additional) * ar_correction

    return sigma_max
