"""Stability consequences of topside icing: KG rise, GM loss and roll period.

The added ice mass is treated as a set of point masses (per component) placed at
the component centroid; the standard mass-moment bookkeeping gives

.. math::

    KG' = \\frac{\\Delta\\,KG + \\sum_i m_i z_i}{\\Delta + \\sum_i m_i},
    \\qquad GM' = GM - (KG' - KG)

(the vertical position of the centre of buoyancy and the waterplane inertia are
unchanged because icing does not alter the underwater hull).  The natural roll
period follows from the roll equation with an added-inertia coefficient:

.. math::

    T_\\phi = 2\\pi \\sqrt{\\frac{I_{xx} + J_{add}}{(\\Delta + \\sum m_i)\\,g\\,GM'}},
    \\qquad I_{xx} = (\\Delta + \\sum m_i) k_x^2 + \\sum_i m_i (z_i - KG')^2

with ``k_x = k_x\\_factor * B`` the bare-hull roll radius of gyration and
``J_add = added_inertia * I_xx`` the hydrodynamic added inertia.  Both effects of
icing (higher KG -> smaller GM, and mass added high up -> larger inertia) are
retained, so ``dT_roll`` is not simply ``sqrt(GM/GM')``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import air
from .geometry import Component, Superstructure

G = 9.80665


@dataclass
class ShipParticulars:
    """Declared parametric particulars of the reference hull."""

    name: str
    L: float
    B: float
    T: float
    depth: float
    displacement_t: float
    KG: float
    GM: float
    k_x_factor: float = 0.35
    added_inertia: float = 0.20

    @property
    def I_xx(self) -> float:
        """Bare-hull roll inertia (t m2)."""
        k = self.k_x_factor * self.B
        return self.displacement_t * k * k


def reference_hull(name: str = "coaster") -> ShipParticulars:
    """Declared parametric reference hulls (inputs, not real ships)."""
    if name == "coaster":
        return ShipParticulars("coaster_reference", L=88.0, B=14.5, T=5.2, depth=7.0,
                               displacement_t=3000.0, KG=6.4, GM=0.80)
    if name == "fishing":
        return ShipParticulars("fishing_reference", L=20.0, B=6.2, T=2.4, depth=1.8,
                               displacement_t=170.0, KG=2.05, GM=0.45, k_x_factor=0.38)
    raise ValueError(f"unknown reference hull: {name!r}")


@dataclass
class StabilityResult:
    """Result of the icing stability assessment."""

    ship: ShipParticulars
    ice_mass_t: float
    ice_moment_tm: float
    z_ice: float
    KG_new: float
    dKG: float
    GM_new: float
    dGM: float
    dGM_pct: float
    I_xx_new: float
    I_xx_rel_increase: float
    T_roll0: float
    T_roll_new: float
    dT_roll_pct: float
    meets_dKG: bool
    meets_dT_roll: bool

    def as_dict(self) -> dict:
        return {
            "ship": self.ship.name,
            "ice_mass_t": self.ice_mass_t,
            "ice_moment_tm": self.ice_moment_tm,
            "z_ice_m": self.z_ice,
            "KG_new_m": self.KG_new,
            "dKG_m": self.dKG,
            "GM_new_m": self.GM_new,
            "dGM_m": self.dGM,
            "dGM_pct": self.dGM_pct,
            "I_xx_new_tm2": self.I_xx_new,
            "I_xx_rel_increase": self.I_xx_rel_increase,
            "T_roll_0_s": self.T_roll0,
            "T_roll_new_s": self.T_roll_new,
            "dT_roll_pct": self.dT_roll_pct,
            "meets_dKG_0p1": bool(self.meets_dKG),
            "meets_dT_roll_10pct": bool(self.meets_dT_roll),
        }


def roll_period(displacement_t: float, I_xx: float, GM: float, added_inertia: float = 0.20) -> float:
    """Natural roll period [s] from the linear roll equation."""
    return float(2.0 * np.pi * np.sqrt((1.0 + added_inertia) * I_xx / (displacement_t * G * GM)))


def icing_stability(
    ship: ShipParticulars,
    superstructure: Superstructure,
    thickness: dict[str, float],
    rho_ice: float = air.RHO_ICE_GLAZE,
    dKG_threshold: float = 0.10,
    dT_roll_threshold: float = 10.0,
) -> StabilityResult:
    """Assess KG / GM / roll-period change for a given icing thickness map."""
    m_ice, mz_ice, z_ice = superstructure.moments(thickness, rho_ice=rho_ice)
    m_ice_t = m_ice / 1000.0  # kg -> t
    mz_ice_t = mz_ice / 1000.0

    delta_new = ship.displacement_t + m_ice_t
    KG_new = (ship.displacement_t * ship.KG + mz_ice_t) / delta_new
    dKG = KG_new - ship.KG
    GM_new = ship.GM - dKG

    # roll inertia: bare hull + ice masses about the new KG
    I_new = ship.displacement_t * (ship.k_x_factor * ship.B) ** 2
    for c in superstructure.components:
        h = thickness.get(c.name, 0.0)
        mi_t = rho_ice * h * c.area * c.catch_fraction / 1000.0
        I_new += mi_t * (c.z - KG_new) ** 2
    I_new += delta_new * 0.0  # ice mass far from KG already accounted for above

    # the displacement increase itself raises the hull inertia proportionally
    I_hull_new = delta_new * (ship.k_x_factor * ship.B) ** 2
    I_ice = I_new - ship.I_xx
    I_total = I_hull_new + max(I_ice, 0.0)

    T0 = roll_period(ship.displacement_t, ship.I_xx, ship.GM, ship.added_inertia)
    T1 = roll_period(delta_new, I_total, max(GM_new, 1e-3), ship.added_inertia)
    return StabilityResult(
        ship=ship,
        ice_mass_t=m_ice_t,
        ice_moment_tm=mz_ice_t,
        z_ice=z_ice,
        KG_new=KG_new,
        dKG=dKG,
        GM_new=GM_new,
        dGM=-dKG,
        dGM_pct=100.0 * (GM_new - ship.GM) / ship.GM,
        I_xx_new=I_total,
        I_xx_rel_increase=100.0 * (I_total - ship.I_xx) / ship.I_xx,
        T_roll0=T0,
        T_roll_new=T1,
        dT_roll_pct=100.0 * (T1 - T0) / T0,
        meets_dKG=bool(dKG >= dKG_threshold),
        meets_dT_roll=bool((T1 - T0) / T0 * 100.0 >= dT_roll_threshold),
    )


def uniform_load_thickness(superstructure: Superstructure, load_kg_m2: float) -> dict[str, float]:
    """Thickness map for a uniform design icing load [kg/m2] on every component."""
    h = load_kg_m2 / air.RHO_ICE_GLAZE
    return {c.name: h for c in superstructure.components}


def icing_load_sweep(
    ship: ShipParticulars,
    superstructure: Superstructure,
    loads_kg_m2,
    rho_ice: float = air.RHO_ICE_GLAZE,
) -> list[StabilityResult]:
    """Stability metrics for a sweep of uniform design icing loads."""
    return [
        icing_stability(ship, superstructure, uniform_load_thickness(superstructure, L0), rho_ice)
        for L0 in np.atleast_1d(loads_kg_m2)
    ]


def threshold_load(
    ship: ShipParticulars,
    superstructure: Superstructure,
    metric: str = "dKG",
    target: float = 0.10,
    lo: float = 0.0,
    hi: float = 400.0,
    n_iter: int = 60,
) -> float:
    """Design icing load [kg/m2] at which ``metric`` first reaches ``target``.

    ``metric`` is ``'dKG'`` or ``'dT_roll'`` (percent).  Returns ``inf`` when the
    target is not reachable within ``hi`` (honest, no silent clamping).
    """
    def value(L0):
        r = icing_stability(
            ship, superstructure, uniform_load_thickness(superstructure, L0)
        )
        return r.dKG if metric == "dKG" else r.dT_roll_pct

    if value(hi) < target:
        return float("inf")
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        if value(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)