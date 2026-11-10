"""End-to-end grounding damage assessment for a polar 15 000 DWT vessel.

Chain (campaign-node items (a)-(f)):

1. **Kinetic energy** of the ship at the grounding speed (added mass included).
2. **Contact / energy balance**: the rigid ice-ridge keel penetrates the bottom
   until the energy absorbed by the bottom panels equals the kinetic energy
   (``RidgeContact.apex_depth_for_energy``).
3. **Plate response**: the bottom panel resistance curve ``q(delta)`` from the
   verified von Karman elastoplastic panel solver (membrane dominated).
4. **Damage extent**: panels with ``eps_pl >= eps_f`` are torn; the damage
   opening area ``A_damage`` and the longitudinal/lateral extent follow.
5. **Flooding**: Bernoulli orifice inflow into the breached compartment with
   added-weight sinkage until the internal level meets the outside surface.
6. **Residual stability**: ``GZ`` curve of the flooded ship with the
   free-surface correction and the damage-stability criteria.

The ship particulars are the polar 15 000 DWT reference vessel used by the
campaign (L = 148 m, B = 23 m, T = 8.6 m, Cb = 0.76, KG = 7.6 m).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence

import numpy as np

from .contact import RidgeGeometry, RidgeContact, ShoalContact
from .damage import DamageAssessment, StiffenerSection, assess_ridge_damage
from .flooding import (BottomOpening, Compartment, RHO_SEA, quasi_static_flood,
                       transient_flood)
from .panel import PanelResponse, PanelSpec, VonKarmanPanel
from .stability import HullForm, check_imo_criteria, gz_curve

__all__ = ["SHIP_15000DWT", "GroundingCase", "design_panel", "run_grounding_case",
           "parameter_sweep"]

#: Polar 15 000 DWT reference vessel (campaign node C7).
SHIP_15000DWT = dict(
    name="Polar 15 000 DWT",
    length=148.0,
    breadth=23.0,
    depth=12.5,
    draft=8.6,
    cb=0.76,
    kg=7.6,                 # loaded departure KG (m)
    lcg=-1.0,               # longitudinal centre of gravity (m from midship)
    double_bottom_height=1.8,
    hold_height=9.5,
    added_mass_factor=1.10,  # sliding/added-mass factor for grounding
    cargo_hold_length=96.0,
)

#: Bottom panel between longitudinals / floors of the reference vessel.
DEFAULT_PANEL = PanelSpec(a=3.60, b=0.90, t=0.022)

#: Bottom longitudinal (T-bar) of the reference vessel.
DEFAULT_STIFFENER = StiffenerSection(web_height=0.35, web_thickness=0.012,
                                     flange_width=0.15, flange_thickness=0.016,
                                     spacing=0.90, span=3.60)


def design_panel(spec: PanelSpec = DEFAULT_PANEL, delta_max: float = 0.60,
                 steps: int = 14, n_x: int = 21, n_y: int = 21,
                 w_modes: int = 3, uv_modes: int = 2, n_layers: int = 3) -> PanelResponse:
    """Compute the design-panel resistance curve (displacement controlled)."""
    sp = replace(spec, n_x=n_x, n_y=n_y, w_modes=w_modes, uv_modes=uv_modes,
                 n_layers=n_layers)
    solver = VonKarmanPanel(sp, load_steps=steps)
    return solver.resistance_curve(delta_max, steps=steps)


def _ship_mass(ship: Dict[str, float]) -> float:
    vol = ship["length"] * ship["breadth"] * ship["draft"] * ship["cb"]
    return RHO_SEA * vol


@dataclass
class GroundingCase:
    """Input description of one grounding event."""

    velocity_kn: float = 3.0
    ridge_height: float = 4.0
    ridge_slope_deg: float = 12.0
    ridge_length: float = 30.0
    ship: Dict[str, float] = field(default_factory=lambda: dict(SHIP_15000DWT))
    stiffener: StiffenerSection = field(default_factory=lambda: DEFAULT_STIFFENER)
    n_panels: int = 25
    hold_flooded: Optional[bool] = None   # None: decided by the penetration depth
    damage_stability_angles: Sequence[float] = tuple(np.arange(0.0, 61.0, 5.0))


def run_grounding_case(case: GroundingCase, response: PanelResponse,
                       hull: Optional[HullForm] = None,
                       verbose: bool = False) -> Dict[str, object]:
    """Run the complete chain of a single grounding event."""
    ship = case.ship
    if hull is None:
        hull = HullForm(length=ship["length"], breadth=ship["breadth"],
                        depth=ship["depth"], draft=ship["draft"], cb=ship["cb"])
        hull.solve_fullness()
    gate = hull.gate()

    # ---- 1. kinetic energy -------------------------------------------------
    mass = _ship_mass(ship)
    v = case.velocity_kn * 1852.0 / 3600.0
    e_kin = 0.5 * mass * ship["added_mass_factor"] * v**2

    # ---- 2./3. contact and energy balance ----------------------------------
    ridge = RidgeGeometry(h_ridge=case.ridge_height, slope_deg=case.ridge_slope_deg,
                          length=case.ridge_length)
    contact = RidgeContact(ridge, lambda d: response.pressure_at(d),
                           panel_breadth=response.spec.b,
                           panel_length=response.spec.a,
                           n_panels=case.n_panels)
    apex = contact.apex_depth_for_energy(e_kin)
    energy_absorbed = contact.energy(apex)
    resistance_max = contact.resistance(apex)
    crush_energy = 0.0
    if ridge.crushing_pressure is not None:
        crush_energy = ridge.crushing_pressure * 2.0 * ridge.contact_half_width(apex) \
            * ridge.length * apex

    # ---- 4. damage extent --------------------------------------------------
    n_long = max(int(round(case.ridge_length / response.spec.a)), 1)
    damage: DamageAssessment = assess_ridge_damage(
        ridge, response, apex, n_longitudinal=n_long, stiffener=case.stiffener,
        panel_breadth=response.spec.b, n_panels=case.n_panels)

    # ---- 5. flooding ------------------------------------------------------
    a_damage = damage.opening_area
    # the inner bottom (top of the double bottom) is breached only if the ridge
    # penetrates deeper than the double-bottom height
    hold_flooded = (apex > ship["double_bottom_height"]) if case.hold_flooded is None \
        else bool(case.hold_flooded)
    if hold_flooded:
        comp = Compartment("cargo-hold", -0.5 * ship["cargo_hold_length"],
                           0.5 * ship["cargo_hold_length"],
                           -0.5 * ship["breadth"], 0.5 * ship["breadth"],
                           ship["double_bottom_height"], ship["depth"],
                           permeability=0.95)
    else:
        comp = Compartment("double-bottom", -0.5 * ship["cargo_hold_length"],
                           0.5 * ship["cargo_hold_length"],
                           -0.5 * ship["breadth"], 0.5 * ship["breadth"],
                           0.0, ship["double_bottom_height"], permeability=0.95)
    opening = BottomOpening(area=a_damage, discharge_coefficient=0.6, compartment=comp)
    awp = hull.waterplane_area()
    flood = quasi_static_flood(opening, comp, draft0=ship["draft"], waterplane_area=awp)
    transient = None
    if a_damage > 0.0:
        transient = transient_flood(opening, comp, draft0=ship["draft"],
                                    waterplane_area=awp, t_end=1800.0, dt=1.0)

    # ---- 6. residual stability --------------------------------------------
    vol_ship = hull.displacement_volume()
    m_flood = RHO_SEA * flood["volume"]
    m_ship = RHO_SEA * vol_ship
    kg_comb = (m_ship * ship["kg"] + m_flood * comp.centroid[2]) / (m_ship + m_flood)
    lcg_comb = (m_ship * ship["lcg"] + m_flood * comp.centroid[0]) / (m_ship + m_flood)
    # free-surface effect: only a *partially* filled compartment has a free
    # surface; a completely filled compartment has none
    fsm = 0.0
    if flood["volume"] > 0.0 and flood.get("full", 0.0) < 0.5:
        fsm = RHO_SEA * comp.waterplane_inertia()
    vol_total = vol_ship + flood["volume"]
    gz = gz_curve(hull, kg=kg_comb, volume=vol_total, lcg=lcg_comb,
                  angles=case.damage_stability_angles, free_surface_moment=fsm)
    kb = hull.properties(0.0, flood["draft"])["zb"]
    km = kb + hull.waterplane_inertia_transverse() / vol_total
    gm = km - (kg_comb + fsm / (RHO_SEA * vol_total))
    imo = check_imo_criteria(gz["angle_deg"], gz["gz"], gm=gm)

    result = {
        "inner_bottom_breached": bool(hold_flooded),
        "case": {
            "velocity_kn": case.velocity_kn,
            "ridge_height": case.ridge_height,
            "ridge_slope_deg": case.ridge_slope_deg,
            "ridge_length": case.ridge_length,
            "ship": {k: ship[k] for k in ("length", "breadth", "draft", "cb", "kg")},
            "panel": {"a": response.spec.a, "b": response.spec.b,
                      "t": response.spec.t, "eps_f": response.spec.grade.eps_f},
        },
        "geometry_gate": gate,
        "energy": {"kinetic": e_kin, "absorbed": energy_absorbed,
                   "residual": energy_absorbed - e_kin,
                   "relative_residual": (energy_absorbed - e_kin) / e_kin if e_kin else 0.0,
                   "ice_crushing": crush_energy},
        "contact": {"apex_depth": apex, "resistance_max": resistance_max,
                    "contact_half_width": ridge.contact_half_width(apex),
                    "penetration_profile": {
                        "y": contact.y.tolist(),
                        "delta": contact.local_deflection(apex).tolist(),
                        "pressure": contact.local_pressure(apex).tolist()}},
        "damage": damage.as_dict(),
        "flooding": {"opening_area": a_damage, **{k: float(v) for k, v in flood.items()},
                     "capacity": comp.capacity,
                     "time_to_equilibrium": float(transient["t"][-1]) if transient else 0.0},
        "stability": {"gm": gm, "km": km, "kg_effective": gz["kg_effective"],
                      "gz": {"angle_deg": gz["angle_deg"].tolist(),
                             "gz": gz["gz"].tolist()},
                      "imo": {k: v for k, v in imo.items() if k != "criteria"}},
    }
    if verbose:
        print(f"[grounding] V={case.velocity_kn} kn h={case.ridge_height} m "
              f"alpha={case.ridge_slope_deg} deg -> apex={apex:.3f} m, "
              f"A_damage={a_damage:.3f} m2, V_flood={flood['volume']:.1f} m3, "
              f"GM={gm:.2f} m")
    return result


def parameter_sweep(response: PanelResponse,
                    velocities: Sequence[float] = (1.0, 3.0, 5.0),
                    ridge_heights: Sequence[float] = (2.0, 4.0, 6.0),
                    slopes: Sequence[float] = (5.0, 12.0, 20.0),
                    hull: Optional[HullForm] = None) -> List[Dict[str, object]]:
    """Campaign-node parameter scan ``V in [1,5] kn, h in [2,6] m, alpha in [5,20] deg``."""
    rows: List[Dict[str, object]] = []
    for v in velocities:
        for h in ridge_heights:
            for al in slopes:
                case = GroundingCase(velocity_kn=float(v), ridge_height=float(h),
                                     ridge_slope_deg=float(al),
                                     damage_stability_angles=tuple(np.arange(0.0, 61.0, 10.0)))
                res = run_grounding_case(case, response, hull=hull)
                rows.append({
                    "velocity_kn": float(v), "ridge_height": float(h),
                    "slope_deg": float(al),
                    "apex_depth": res["contact"]["apex_depth"],
                    "contact_half_width": res["contact"]["contact_half_width"],
                    "n_torn": res["damage"]["n_torn"],
                    "opening_area": res["damage"]["opening_area"],
                    "damage_area": res["damage"]["damage_area"],
                    "max_eps_pl": res["damage"]["max_eps_pl"],
                    "flood_volume": res["flooding"]["volume"],
                    "sinkage": res["flooding"]["sinkage"],
                    "gm": res["stability"]["gm"],
                    "gz_max": res["stability"]["imo"]["gz_max"],
                    "area_to_30": res["stability"]["imo"]["area_to_30"],
                    "imo_ok": res["stability"]["imo"]["all_ok"],
                    "energy_residual": res["energy"]["relative_residual"],
                })
    return rows