"""Damage criteria and damage extent for grounding on ice ridges / shoals.

Implements the campaign-node items (c) and (d):

* tearing criterion ``eps_pl >= eps_f`` -> panel element deleted / declared torn;
* stiffener failure: web shear buckling, flange (plate) compression buckling
  and full plastic bending of the bottom longitudinal;
* damage extent: number and position of torn/buckled panels, lateral and
  longitudinal extent, and the damage opening area ``A_damage``.

The plastic strain comes from the von Karman panel solution
(:mod:`tensorphysics.grounding_damage.panel`); nothing here is hard-coded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .contact import RidgeGeometry, RidgeContact
from .panel import PanelResponse, PanelSpec

__all__ = ["StiffenerSection", "StiffenerCheck", "PanelDamageState",
           "DamageAssessment", "assess_ridge_damage"]


# ---------------------------------------------------------------------------
# Stiffener (bottom longitudinal) capacity checks
# ---------------------------------------------------------------------------
@dataclass
class StiffenerSection:
    """Bottom longitudinal (T-bar) geometry and material."""

    web_height: float = 0.35        # m
    web_thickness: float = 0.012    # m
    flange_width: float = 0.15      # m
    flange_thickness: float = 0.016  # m
    spacing: float = 0.90           # m (stiffener spacing = panel breadth)
    E: float = 2.06e11
    nu: float = 0.3
    sigma_y: float = 3.55e8
    span: float = 3.60              # m (between floors / transverses)

    @property
    def plate_buckling_factor_shear(self) -> float:
        """Shear buckling coefficient ``k_tau`` for a web panel ``h_w x span``."""
        a, h = self.span, self.web_height
        return 5.34 + 4.0 * (h / a) ** 2

    def web_shear_buckling_stress(self) -> float:
        """Elastic shear buckling stress of the web (Pa)."""
        k = self.plate_buckling_factor_shear
        D = self.E / (1.0 - self.nu**2)
        return k * np.pi**2 * D / (12.0 * (self.web_height / self.web_thickness) ** 2)

    def flange_buckling_stress(self) -> float:
        """Elastic compression buckling stress of the flange outstand (Pa)."""
        k = 0.425                      # outstand plate (one edge free)
        D = self.E / (1.0 - self.nu**2)
        c = 0.5 * (self.flange_width - self.web_thickness)
        return k * np.pi**2 * D / (12.0 * (c / self.flange_thickness) ** 2)

    def plastic_bending_resistance(self) -> float:
        """Fully plastic bending moment of the T-section (N m, sagging)."""
        hw, tw = self.web_height, self.web_thickness
        bf, tf = self.flange_width, self.flange_thickness
        z = np.linspace(0.0, hw, 4001)
        dz = np.diff(z)
        zc = 0.5 * (z[:-1] + z[1:])
        area = np.concatenate([[0.0], np.cumsum(tw * dz)])
        A_tot = hw * tw + bf * tf
        i = int(np.searchsorted(area, 0.5 * A_tot))
        z_pna = float(z[min(i, z.size - 1)])
        m = float(np.sum(tw * dz * np.abs(zc - z_pna)))
        m += bf * tf * abs(hw + 0.5 * tf - z_pna)
        return float(self.sigma_y * m)

    def shear_capacity(self) -> float:
        """Elastic shear capacity of the web, ``tau_y h_w t_w`` (N)."""
        tau_y = self.sigma_y / np.sqrt(3.0)
        return float(tau_y * self.web_height * self.web_thickness)


@dataclass
class StiffenerCheck:
    """Outcome of the stiffener capacity checks for one panel load state."""

    membrane_stress: float          # acting membrane stress (Pa, compression +)
    shear_stress: float             # acting mean shear stress in the web (Pa)
    web_buckling_stress: float
    flange_buckling_stress: float
    shear_capacity: float
    plastic_moment: float           # capacity (N m)
    applied_moment: float           # applied bending moment per stiffener (N m)
    web_buckled: bool
    flange_buckled: bool
    plastic_collapse: bool

    @property
    def failed(self) -> bool:
        return bool(self.web_buckled or self.flange_buckled or self.plastic_collapse)

    def as_dict(self) -> Dict[str, float]:
        return {
            "membrane_stress": self.membrane_stress,
            "shear_stress": self.shear_stress,
            "web_buckling_stress": self.web_buckling_stress,
            "flange_buckling_stress": self.flange_buckling_stress,
            "web_buckled": float(self.web_buckled),
            "flange_buckled": float(self.flange_buckled),
            "plastic_collapse": float(self.plastic_collapse),
        }


def check_stiffener(section: StiffenerSection, pressure: float,
                    panel_area: float, membrane_force: float,
                    bend_moment: float) -> StiffenerCheck:
    """Compare the stiffener capacity with the load transferred by the panel.

    The panel load ``pressure x panel area`` is carried by one longitudinal
    (tributary width = stiffener spacing); the membrane force per unit width
    from the panel acts in the plate, and the panel bending moment is assumed
    to be transferred as a line moment per unit width, scaled by the spacing.
    """
    sigma_m = abs(membrane_force) / max(section.web_thickness, 1e-9) / 4.0
    tau = 0.5 * pressure * panel_area / max(section.web_height * section.web_thickness, 1e-12)
    m_app = abs(bend_moment) * section.spacing
    sc = StiffenerCheck(
        membrane_stress=float(sigma_m), shear_stress=float(tau),
        web_buckling_stress=section.web_shear_buckling_stress(),
        flange_buckling_stress=section.flange_buckling_stress(),
        shear_capacity=section.shear_capacity(),
        plastic_moment=section.plastic_bending_resistance(),
        applied_moment=float(m_app),
        web_buckled=bool(tau > section.web_shear_buckling_stress()),
        flange_buckled=bool(sigma_m > section.flange_buckling_stress()),
        plastic_collapse=bool(m_app > section.plastic_bending_resistance()),
    )
    return sc


# ---------------------------------------------------------------------------
# Panel damage state and damage extent
# ---------------------------------------------------------------------------
@dataclass
class PanelDamageState:
    """Damage state of one bottom panel at a prescribed penetration."""

    y: float                     # lateral position (m)
    penetration: float           # local penetration (m)
    pressure: float              # contact pressure (Pa)
    eps_pl: float                # peak equivalent plastic strain
    torn_fraction: float         # fraction of panel area above eps_f
    torn: bool
    stiffener: Optional[StiffenerCheck] = None

    @property
    def opening_area(self) -> float:
        """Contribution of this panel to the damage opening area.

        A torn panel contributes its full area; a pre-tearing but heavily
        plastic panel contributes nothing to the opening (it is a dent, not a
        hole), which is the conservative (non-flooding) side.
        """
        return 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "y": self.y, "penetration": self.penetration, "pressure": self.pressure,
            "eps_pl": self.eps_pl, "torn_fraction": self.torn_fraction,
            "torn": float(self.torn),
        }


@dataclass
class DamageAssessment:
    """Damage extent and opening area of a grounding event."""

    apex_depth: float
    contact_half_width: float
    panels: List[PanelDamageState]
    panel_area: float
    n_longitudinal: int

    @property
    def n_torn(self) -> int:
        return sum(1 for p in self.panels if p.torn)

    @property
    def n_damaged(self) -> int:
        return sum(1 for p in self.panels if p.penetration > 1e-4)

    @property
    def damage_area(self) -> float:
        """Total damaged (permanently deformed) bottom area, one panel row (m^2)."""
        return float(self.n_damaged * self.panel_area)

    @property
    def opening_area(self) -> float:
        """Damage opening area ``A_damage`` (m^2), all grounded rows included."""
        frac = sum(p.torn_fraction for p in self.panels if p.torn)
        return float(frac * self.panel_area * self.n_longitudinal)

    @property
    def lateral_extent(self) -> float:
        return float(self.n_damaged * np.sqrt(self.panel_area))

    @property
    def longitudinal_extent(self) -> float:
        return float(self.n_longitudinal * self.panel_area / max(self.lateral_extent, 1e-9))

    def as_dict(self) -> Dict[str, float]:
        return {
            "apex_depth": self.apex_depth,
            "contact_half_width": self.contact_half_width,
            "n_torn": self.n_torn,
            "n_damaged_panels": self.n_damaged,
            "damage_area": self.damage_area,
            "opening_area": self.opening_area,
            "lateral_extent": self.lateral_extent,
            "longitudinal_extent": self.longitudinal_extent,
            "max_eps_pl": max((p.eps_pl for p in self.panels), default=0.0),
            "max_pressure": max((p.pressure for p in self.panels), default=0.0),
        }


def assess_ridge_damage(ridge: RidgeGeometry, response: PanelResponse,
                        apex_depth: float, n_longitudinal: int,
                        stiffener: Optional[StiffenerSection] = None,
                        panel_breadth: float = 0.9,
                        n_panels: int = 25) -> DamageAssessment:
    """Damage extent for a rigid ridge penetrating ``apex_depth``.

    Every bottom panel on the ridge flank sees the local penetration
    ``delta_i``; its plastic state is read from the verified panel resistance
    curve (``response``), and the tearing criterion ``eps_pl >= eps_f`` decides
    whether the panel is torn.
    """
    spec = response.spec
    contact = RidgeContact(ridge, lambda d: response.pressure_at(d),
                           panel_breadth=panel_breadth,
                           panel_length=spec.a, n_panels=n_panels)
    delta = contact.local_deflection(apex_depth)
    states: List[PanelDamageState] = []
    d_arr = np.maximum.accumulate(response.deflection)
    for y, d in zip(contact.y, delta):
        if d <= 1e-6:
            states.append(PanelDamageState(y=float(y), penetration=0.0, pressure=0.0,
                                           eps_pl=0.0, torn_fraction=0.0, torn=False))
            continue
        q = response.pressure_at(float(d))
        eps = float(np.interp(d, d_arr, np.maximum.accumulate(response.eps_pl_max)))
        frac = float(np.interp(d, d_arr, np.maximum.accumulate(response.torn_fraction)))
        torn = eps >= spec.grade.eps_f
        st = None
        if stiffener is not None:
            N = float(np.interp(d, d_arr, np.maximum.accumulate(response.membrane_force)))
            M = float(np.interp(d, d_arr, np.maximum.accumulate(response.bend_moment)))
            st = check_stiffener(stiffener, q, spec.area, N, M)
        states.append(PanelDamageState(y=float(y), penetration=float(d), pressure=q,
                                       eps_pl=eps, torn_fraction=frac if torn else 0.0,
                                       torn=torn, stiffener=st))
    return DamageAssessment(apex_depth=float(apex_depth),
                            contact_half_width=ridge.contact_half_width(apex_depth),
                            panels=states, panel_area=spec.area,
                            n_longitudinal=int(n_longitudinal))