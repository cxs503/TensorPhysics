"""Plate-beam combination structure: geometry and section properties.

Physical model
--------------
A stiffened plate panel consisting of:
- Plate: rectangular panel between stiffeners, thickness t_p
- T-stiffener: web + flange attached to plate, forming a T-section with effective plate width

The effective width of plate attached to the stiffener is computed per
classification society rules (typically min(s, L/6) where s = stiffener spacing,
L = span).

Section properties
------------------
For the T-stiffener with attached plating:
- Neutral axis position
- Second moment of area I
- Elastic section modulus Z_el (top and bottom)
- Plastic section modulus Z_pl (top and bottom)
- Plastic moment capacity M_p = sigma_y * Z_pl
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class TStiffenerGeometry:
    """T-stiffener cross-section geometry.

    Parameters
    ----------
    h_w : float
        Web height [m]
    t_w : float
        Web thickness [m]
    b_f : float
        Flange width [m]
    t_f : float
        Flange thickness [m]
    """
    h_w: float  # web height [m]
    t_w: float  # web thickness [m]
    b_f: float  # flange width [m]
    t_f: float  # flange thickness [m]

    @property
    def A_web(self) -> float:
        return self.h_w * self.t_w

    @property
    def A_flange(self) -> float:
        return self.b_f * self.t_f

    @property
    def A_total(self) -> float:
        return self.A_web + self.A_flange


@dataclass(frozen=True)
class PlateGeometry:
    """Plate panel geometry between stiffeners.

    Parameters
    ----------
    a : float
        Panel length (along stiffener) [m]
    b : float
        Panel breadth (stiffener spacing) [m]
    t_p : float
        Plate thickness [m]
    """
    a: float  # length [m]
    b: float  # breadth / stiffener spacing [m]
    t_p: float  # thickness [m]

    @property
    def aspect_ratio(self) -> float:
        return self.a / self.b

    @property
    def A_plate(self) -> float:
        return self.a * self.b


@dataclass(frozen=True)
class PlateBeamSection:
    """Plate-beam combination section (T-stiffener with attached plating).

    The effective plate width b_eff is the lesser of:
    - Stiffener spacing s
    - Span / 6 (classification rule)
    - Actual plate breadth

    Parameters
    ----------
    plate : PlateGeometry
        Plate panel geometry
    stiffener : TStiffenerGeometry
        T-stiffener geometry
    E : float
        Young's modulus [Pa]
    sigma_y : float
        Yield stress [Pa]
    nu : float
        Poisson's ratio
    boundary_condition : str
        'simply_supported' or 'clamped'
    """
    plate: PlateGeometry
    stiffener: TStiffenerGeometry
    E: float = 2.06e11  # Pa
    sigma_y: float = 355e6  # Pa
    nu: float = 0.3
    boundary_condition: str = "simply_supported"

    @property
    def b_eff(self) -> float:
        """Effective plate width attached to stiffener [m].

        Per classification rules: b_eff = min(s, L/6, actual_breadth)
        where s = stiffener spacing, L = span.
        """
        s = self.plate.b  # stiffener spacing
        L = self.plate.a  # span
        return min(s, L / 6.0, self.plate.b)

    @property
    def A_eff_plate(self) -> float:
        """Effective plate area [m²]."""
        return self.b_eff * self.plate.t_p

    @property
    def A_total(self) -> float:
        """Total cross-sectional area [m²]."""
        return self.A_eff_plate + self.stiffener.A_total

    def _compute_section_properties(self) -> dict:
        """Compute neutral axis, I, Z_el, Z_pl for the combined section.

        Coordinate system: y=0 at bottom of web, y positive upward.
        Plate is at the top (attached to flange).
        """
        t_p = self.plate.t_p
        h_w = self.stiffener.h_w
        t_w = self.stiffener.t_w
        b_f = self.stiffener.b_f
        t_f = self.stiffener.t_f
        b_eff = self.b_eff

        # Component areas and centroids (from bottom of web)
        # Web
        A_web = h_w * t_w
        y_web = h_w / 2.0

        # Flange
        A_flange = b_f * t_f
        y_flange = h_w + t_f / 2.0

        # Effective plate
        A_plate = b_eff * t_p
        y_plate = h_w + t_f + t_p / 2.0

        # Total area
        A_total = A_web + A_flange + A_plate

        # Neutral axis (from bottom)
        y_NA = (A_web * y_web + A_flange * y_flange + A_plate * y_plate) / A_total

        # Second moment of area about NA
        # Web
        I_web = t_w * h_w**3 / 12.0 + A_web * (y_web - y_NA)**2
        # Flange
        I_flange = b_f * t_f**3 / 12.0 + A_flange * (y_flange - y_NA)**2
        # Plate
        I_plate = b_eff * t_p**3 / 12.0 + A_plate * (y_plate - y_NA)**2

        I_total = I_web + I_flange + I_plate

        # Elastic section moduli
        y_top = h_w + t_f + t_p - y_NA  # distance from NA to top
        y_bot = y_NA  # distance from NA to bottom

        Z_el_top = I_total / y_top if y_top > 0 else float('inf')
        Z_el_bot = I_total / y_bot if y_bot > 0 else float('inf')

        # Plastic section modulus (equal area axis)
        # Find the axis that divides the area in half
        A_half = A_total / 2.0

        # Start from bottom and accumulate area
        # Web contribution
        if A_web >= A_half:
            # Plastic neutral axis is in the web
            y_PNA = A_half / t_w
            S_web = t_w * (y_PNA**2 + (h_w - y_PNA)**2) / 2.0
            S_flange = A_flange * (y_flange - y_PNA)
            S_plate = A_plate * (y_plate - y_PNA)
        else:
            # PNA is above the web
            A_remaining = A_half - A_web
            if A_web + A_flange >= A_half:
                # PNA is in the flange
                y_PNA = h_w + (A_remaining / b_f)
                S_web = A_web * (y_PNA - y_web)
                h_flange_below = y_PNA - h_w
                S_flange_bot = b_f * h_flange_below * (h_flange_below / 2.0)
                h_flange_above = t_f - h_flange_below
                S_flange_top = b_f * h_flange_above * (h_flange_above / 2.0 + h_flange_below)
                S_flange = S_flange_bot + S_flange_top
                S_plate = A_plate * (y_plate - y_PNA)
            else:
                # PNA is in the plate
                A_remaining2 = A_half - A_web - A_flange
                y_PNA = h_w + t_f + (A_remaining2 / b_eff)
                S_web = A_web * (y_PNA - y_web)
                S_flange = A_flange * (y_PNA - y_flange)
                h_plate_below = y_PNA - (h_w + t_f)
                S_plate_bot = b_eff * h_plate_below * (h_plate_below / 2.0)
                h_plate_above = t_p - h_plate_below
                S_plate_top = b_eff * h_plate_above * (h_plate_above / 2.0 + h_plate_below)
                S_plate = S_plate_bot + S_plate_top

        Z_pl = S_web + S_flange + S_plate

        return {
            "y_NA": y_NA,
            "y_PNA": y_PNA,
            "I": I_total,
            "Z_el_top": Z_el_top,
            "Z_el_bot": Z_el_bot,
            "Z_pl": Z_pl,
            "A_total": A_total,
            "y_top": y_top,
            "y_bot": y_bot,
        }

    @property
    def section_props(self) -> dict:
        """Cached section properties."""
        return self._compute_section_properties()

    @property
    def I(self) -> float:
        return self.section_props["I"]

    @property
    def Z_el(self) -> float:
        """Elastic section modulus (governing, minimum of top/bottom)."""
        return min(self.section_props["Z_el_top"], self.section_props["Z_el_bot"])

    @property
    def Z_pl(self) -> float:
        """Plastic section modulus."""
        return self.section_props["Z_pl"]

    @property
    def M_el(self) -> float:
        """Elastic moment capacity (first yield) [N·m]."""
        return self.sigma_y * self.Z_el

    @property
    def M_pl(self) -> float:
        """Plastic moment capacity [N·m]."""
        return self.sigma_y * self.Z_pl

    @property
    def shape_factor(self) -> float:
        """Shape factor = M_pl / M_el = Z_pl / Z_el."""
        return self.Z_pl / self.Z_el

    @property
    def EI(self) -> float:
        """Flexural rigidity [N·m²]."""
        return self.E * self.I

    # Plate-only properties (for plate-first-yield check)
    @property
    def plate_I(self) -> float:
        """Plate moment of inertia per unit width [m⁴/m]."""
        return self.plate.t_p**3 / 12.0

    @property
    def plate_Z_el(self) -> float:
        """Plate elastic section modulus per unit width [m³/m]."""
        return self.plate.t_p**2 / 6.0

    @property
    def plate_Z_pl(self) -> float:
        """Plate plastic section modulus per unit width [m³/m]."""
        return self.plate.t_p**2 / 4.0

    @property
    def plate_M_el(self) -> float:
        """Plate elastic moment capacity per unit width [N·m/m]."""
        return self.sigma_y * self.plate_Z_el

    @property
    def plate_M_pl(self) -> float:
        """Plate plastic moment capacity per unit width [N·m/m]."""
        return self.sigma_y * self.plate_Z_pl

    @property
    def plate_D(self) -> float:
        """Plate flexural rigidity per unit width [N·m]."""
        return self.E * self.plate.t_p**3 / (12.0 * (1.0 - self.nu**2))


def default_plate_beam_section(pc: int = 5) -> PlateBeamSection:
    """Create a default plate-beam section for given polar class.

    Typical dimensions for a PC5 ice-strengthened vessel.
    Plate is designed to yield before the stiffener (plate t_p = 8mm).
    """
    plate = PlateGeometry(
        a=2.4,    # frame spacing [m]
        b=0.6,    # stiffener spacing [m]
        t_p=0.008  # plate thickness [m] - thin plate yields first
    )
    stiffener = TStiffenerGeometry(
        h_w=0.300,  # web height [m] - robust stiffener
        t_w=0.012,  # web thickness [m]
        b_f=0.150,  # flange width [m]
        t_f=0.016   # flange thickness [m]
    )
    return PlateBeamSection(
        plate=plate,
        stiffener=stiffener,
        E=2.06e11,
        sigma_y=355e6,
        nu=0.3,
        boundary_condition="simply_supported"
    )
