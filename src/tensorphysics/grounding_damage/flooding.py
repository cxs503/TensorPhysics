"""Quasi-static flooding of a breached bottom compartment.

Campaign-node item (e): water enters through the damage opening ``A_damage``
with discharge coefficient ``C_d`` under the instantaneous head between the
outside water surface and the internal level; the flooded mass increases the
displacement, which increases the draft, which increases the head
(added-weight method).  The module provides

* :func:`inflow_rate`      - Bernoulli orifice law ``Q = C_d A sqrt(2 g h)``;
* :func:`transient_flood`  - explicit time integration of the coupled
  inflow / sinkage problem;
* :func:`analytic_filling_time` - closed-form solution of the filling ODE
  used to verify the integrator (acceptance criterion C4, target < 10 %);
* :func:`quasi_static_flood` - the equilibrium end state (flooded volume and
  final level) used by the residual-stability analysis.

All quantities SI (m, m^3, s, Pa).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["Compartment", "BottomOpening", "inflow_rate", "transient_flood",
           "analytic_filling_time", "quasi_static_flood", "G", "RHO_SEA"]

G = 9.80665
RHO_SEA = 1025.0


@dataclass
class Compartment:
    """Box-shaped compartment with permeability ``mu``.

    ``z0``/``z1`` are measured from the ship's baseline (keel) upwards.
    """

    name: str
    x0: float
    x1: float
    y0: float
    y1: float
    z0: float
    z1: float
    permeability: float = 0.95

    @property
    def plan_area(self) -> float:
        return (self.x1 - self.x0) * (self.y1 - self.y0)

    @property
    def capacity(self) -> float:
        """Maximum water volume the compartment can hold (m^3)."""
        return self.plan_area * (self.z1 - self.z0) * self.permeability

    @property
    def effective_plan_area(self) -> float:
        return self.plan_area * self.permeability

    def level(self, volume: float) -> float:
        """Free-surface level (m above baseline) for a given water volume."""
        v = min(max(volume, 0.0), self.capacity)
        return self.z0 + v / self.effective_plan_area

    def volume_at_level(self, level: float) -> float:
        lv = min(max(level, self.z0), self.z1)
        return (lv - self.z0) * self.effective_plan_area

    def waterplane_inertia(self) -> float:
        """Free-surface second moment of area of the compartment (m^4)."""
        lx = self.x1 - self.x0
        ly = self.y1 - self.y0
        return lx * ly**3 / 12.0 * self.permeability

    @property
    def centroid(self) -> Tuple[float, float, float]:
        return (0.5 * (self.x0 + self.x1), 0.5 * (self.y0 + self.y1),
                0.5 * (self.z0 + self.z1))


@dataclass
class BottomOpening:
    """Damage opening in the bottom shell."""

    area: float                       # m^2
    discharge_coefficient: float = 0.6
    compartment: Optional[Compartment] = None

    def flow(self, head: float) -> float:
        """Volume flow rate (m^3/s) for the instantaneous head ``head`` (m)."""
        return inflow_rate(self.area, head, self.discharge_coefficient)


def inflow_rate(area: float, head: float, cd: float = 0.6) -> float:
    """Bernoulli orifice law ``Q = C_d A sqrt(2 g h)`` (m^3/s)."""
    h = max(float(head), 0.0)
    return float(cd * area * np.sqrt(2.0 * G * h))


def analytic_filling_time(area: float, cd: float, tank_plan_area: float,
                          head_initial: float, level_range: float) -> float:
    """Closed-form filling time for a box compartment at constant outside level.

    With ``A_t dh/dt = C_d A sqrt(2 g (H - h))`` and ``h(0) = H - h0``:

    ``t = (2 A_t / (C_d A sqrt(2 g))) (sqrt(h0) - sqrt(H - h_final))``
    """
    h_final = max(head_initial - level_range, 0.0)
    return float(2.0 * tank_plan_area / (cd * area * np.sqrt(2.0 * G))
                 * (np.sqrt(head_initial) - np.sqrt(h_final)))


def transient_flood(opening: BottomOpening, compartment: Compartment,
                    draft0: float, waterplane_area: float,
                    t_end: float = 1800.0, dt: float = 0.05,
                    allow_sinkage: bool = True,
                    sea_water_density: float = RHO_SEA) -> Dict[str, np.ndarray]:
    """Integrate the coupled inflow / sinkage problem (explicit Euler, substeps).

    ``head = (draft + z_baseline) - level`` where the outside water surface is
    at the instantaneous draft (added-weight sinkage
    ``delta_T = V / A_waterplane``).  Inflow stops when the internal level
    reaches the outside surface or the compartment is full.
    """
    n = int(np.ceil(t_end / dt)) + 1
    t = np.zeros(n)
    vol = np.zeros(n)
    head = np.zeros(n)
    rate = np.zeros(n)
    draft = np.zeros(n)
    V = 0.0
    sub = max(int(0.5 * min(dt, 1.0) / max(dt, 1e-9)), 1)
    del sub
    for i in range(1, n):
        lvl = compartment.level(V)
        T = draft0 + (V / waterplane_area if allow_sinkage else 0.0)
        h = T - lvl
        Q = opening.flow(h)
        if V >= compartment.capacity - 1e-12:
            Q = 0.0
        V = min(V + Q * dt, compartment.capacity)
        t[i], vol[i], head[i], rate[i], draft[i] = t[i - 1] + dt, V, h, Q, T
        if Q <= 0.0 and i > 1:
            # equilibrium reached: fill the rest of the record with the end state
            t[i:] = t[i - 1] + dt * np.arange(1, n - i + 1)
            vol[i:], head[i:], rate[i:], draft[i:] = V, h, 0.0, T
            break
    del sea_water_density
    return {"t": t, "volume": vol, "head": head, "rate": rate, "draft": draft}


def quasi_static_flood(opening: BottomOpening, compartment: Compartment,
                       draft0: float, waterplane_area: float,
                       tol: float = 1e-6, max_iter: int = 500) -> Dict[str, float]:
    """Equilibrium flooded volume (added-weight method, fixed point iteration).

    Solve ``level(V) = draft0 + V / A_wp`` (internal level meets the outside
    surface) subject to ``0 <= V <= capacity``.
    """
    if opening.area <= 0.0:
        return {"volume": 0.0, "level": compartment.z0, "draft": draft0,
                "sinkage": 0.0, "full": 0.0, "head": 0.0}
    lo, hi = 0.0, compartment.capacity
    def f(V):
        T = draft0 + V / waterplane_area
        return compartment.level(V) - T
    f_lo, f_hi = f(lo), f(hi)
    if f_hi < 0.0:                     # compartment fills completely
        V = hi
        full = 1.0
    else:
        full = 0.0
        for _ in range(int(max_iter)):
            mid = 0.5 * (lo + hi)
            if f(mid) < 0.0:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol:
                break
        V = 0.5 * (lo + hi)
    T = draft0 + V / waterplane_area
    return {"volume": float(V), "level": float(compartment.level(V)),
            "draft": float(T), "sinkage": float(T - draft0), "full": full,
            "head": float(T - compartment.level(V)), "f_lo": float(f_lo)}