"""Superstructure geometry model and icing mass distribution.

The vessel superstructure is represented as a set of *exposed components*
(masts, pipes, crane booms, bulkheads, deck areas) each carrying

* an iced area ``A`` [m2],
* the height of its centroid above the keel ``z`` [m] (needed for KG),
* the representative two-dimensional section used to compute the local
  collection efficiency (cylinder / blunt plate / horizontal deck).

Ice thickness is *not* prescribed: it comes either from the coupled
impingement + Messinger chain (:func:`component_ice_thickness`) or from a
declared design icing load in kg/m2 (ISO 12494 icing classes), which is the
standard way stability assessments are performed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import air
from .droplets import ImpingementResult, impingement
from .flowfield import CylinderFlow, EllipseFlow
from .messinger import heat_transfer_coefficient, ice_growth_1d


@dataclass
class Component:
    """One exposed superstructure element."""

    name: str
    kind: str  # 'cylinder' | 'plate' | 'deck'
    area: float  # iced (exposed) area [m2]
    z: float  # centroid height above keel [m]
    diameter: float = 0.0  # characteristic size for the section [m]
    catch_fraction: float = 1.0  # fraction of A exposed to the windward flow
    note: str = ""

    def __post_init__(self):
        self.area = float(self.area)
        self.z = float(self.z)


@dataclass
class Superstructure:
    """A collection of components plus the deck reference height."""

    name: str
    deck_height: float
    components: list[Component] = field(default_factory=list)

    @property
    def area(self) -> float:
        return float(sum(c.area for c in self.components))

    def moments(self, thickness: dict[str, float], rho_ice: float = air.RHO_ICE_GLAZE):
        """Return ``(mass, mass_moment, area_weighted_z)`` for a thickness map."""
        m = 0.0
        mz = 0.0
        for c in self.components:
            h = thickness.get(c.name, 0.0)
            mi = rho_ice * h * c.area * c.catch_fraction
            m += mi
            mz += mi * c.z
        return m, mz, (mz / m if m > 0 else 0.0)


# ---------------------------------------------------------------------------
# reference superstructures (declared synthetic parametric models)
# ---------------------------------------------------------------------------
def reference_superstructure(name: str = "coaster") -> Superstructure:
    """Declared parametric superstructure of the reference hull.

    Values are representative of a small ice-classed coaster / offshore vessel
    (``coaster``) and of a 20 m fishing vessel (``fishing``); they are *inputs*,
    not the particulars of a real ship.
    """
    if name == "coaster":
        deck = 7.0
        comps = [
            Component("foremast", "cylinder", np.pi * 0.30 * 9.0 / 2, deck + 4.5, 0.30),
            Component("radar_mast", "cylinder", np.pi * 0.20 * 4.0 / 2, deck + 6.0, 0.20),
            Component("derrick", "cylinder", np.pi * 0.25 * 12.0 / 2, deck + 3.0, 0.25),
            Component("piping", "cylinder", np.pi * 0.08 * 90.0 / 2, deck + 0.6, 0.08),
            Component("windlass_rails", "cylinder", np.pi * 0.05 * 120.0 / 2, deck + 0.6, 0.05),
            Component("superstructure_front", "plate", 12.0 * 3.2, deck + 1.6, 0.5),
            Component("superstructure_sides", "plate", 2 * 8.0 * 3.2, deck + 1.6, 0.5, 0.4),
            Component("container_stack_faces", "plate", 2 * 6.24 * 4.0, deck + 4.5, 0.5),
            Component("container_tops", "deck", 24 * 12.2 * 2.4, deck + 5.6, 1.0),
            Component("main_deck", "deck", 20.0 * 14.0, deck + 0.1, 1.0),
        ]
        return Superstructure("coaster_reference", deck, comps)
    if name == "fishing":
        deck = 1.8
        comps = [
            Component("mast", "cylinder", np.pi * 0.15 * 7.0 / 2, deck + 3.5, 0.15),
            Component("derrick", "cylinder", np.pi * 0.16 * 6.0 / 2, deck + 2.0, 0.16),
            Component("rigging", "cylinder", np.pi * 0.03 * 60.0 / 2, deck + 2.3, 0.03),
            Component("wheelhouse_front", "plate", 3.0 * 2.4, deck + 1.2, 0.3),
            Component("wheelhouse_sides", "plate", 2 * 2.6 * 2.4, deck + 1.2, 0.3, 0.4),
            Component("bulwark", "plate", 2 * 12.0 * 0.9, deck + 0.45, 0.3, 0.5),
            Component("deck", "deck", 12.0 * 4.0, deck + 0.05, 1.0),
        ]
        return Superstructure("fishing_reference", deck, comps)
    raise ValueError(f"unknown reference superstructure: {name!r}")


# ---------------------------------------------------------------------------
# ice thickness from the coupled impingement / Messinger chain
# ---------------------------------------------------------------------------
def component_section(component: Component, U: float = 15.0):
    """Build the 2-D flow section representative of a component."""
    if component.kind == "cylinder":
        return CylinderFlow(R=max(component.diameter / 2.0, 1e-3), U=U)
    if component.kind == "plate":
        # blunt leading edge of a flat panel: strongly elongated ellipse
        d = max(component.diameter, 0.1)
        return EllipseFlow(A=2.5 * d, B=0.25 * d, U=U)
    if component.kind == "deck":
        # horizontal surface: no blunt-body acceleration, catch is the settling
        # of drops plus a small sweep inlet -> modelled as a very flat ellipse
        d = 1.0
        return EllipseFlow(A=10.0 * d, B=0.05 * d, U=U)
    raise ValueError(f"unknown component kind: {component.kind!r}")


def component_impingement(
    component: Component,
    U: float,
    T_air: float,
    d_m: float,
    n_traj: int = 121,
) -> ImpingementResult:
    """Impingement characteristics of the component section."""
    section = component_section(component, U=U)
    return impingement(section, d_m, U=U, t_air=T_air, n_traj=n_traj)


def component_ice_thickness(
    component: Component,
    U: float,
    T_air: float,
    lwc: float,
    d_m: float,
    t_end: float,
    dt: float = 10.0,
    n_traj: int = 121,
    n_cell: int = 33,
) -> tuple[float, float, np.ndarray]:
    """Area-averaged iced thickness on a component [m].

    Returns ``(h_area_mean, h_stagnation, s_grid_thickness)``.  The area average
    uses the local thickness weighted by the cell arc length of the strip that
    represents the component surface.
    """
    res = component_impingement(component, U, T_air, d_m, n_traj=n_traj)
    s = np.linspace(0.0, res.s_limit if res.s_limit > 0 else 1e-3, n_cell)
    beta = res.beta_at(s)
    L_char = max(component.diameter, 0.1)
    h_c = heat_transfer_coefficient(U, L_char, T_air, kind="cylinder" if component.kind == "cylinder" else "plate")
    growth = ice_growth_1d(
        s,
        beta,
        U,
        T_air,
        lwc,
        t_end,
        dt=dt,
        h_c=h_c,
        L_char=L_char,
        kind="cylinder" if component.kind == "cylinder" else "plate",
    )
    h = growth.h_final
    ds = np.gradient(s) if s.size > 1 else np.array([1.0])
    h_mean = float(np.sum(h * ds) / max(np.sum(ds), 1e-30))
    return h_mean, float(h[0]), h


def design_ice_thickness(load_kg_m2: float, rho_ice: float = air.RHO_ICE_GLAZE) -> float:
    """Equivalent ice thickness of a declared design icing load [m]."""
    return float(load_kg_m2) / rho_ice