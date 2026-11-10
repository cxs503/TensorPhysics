"""Refinement map and Richardson order estimate for the momentum audit.

Refinement keeps the *physics* fixed and the *lattice* healthy, which is what
makes a convergence order meaningful:

============================  =====================================================
box length                    ``L = 0.1 m``            (fixed)
reference speed               ``U = 0.01 m/s``         (fixed)
kinematic viscosity           ``nu = 1e-5 m^2/s``       (fixed -> fixed Re = 100)
lattice Mach                  ``U dt / dx = 0.02``      (fixed, incompressible)
physical window               ``T = 0.5 s``             (fixed)
============================  =====================================================

Consequences: ``dx = L/n``, ``dt = u_lat dx / U``, ``nu_lat = nu dt / dx^2``,
``tau = 1/2 + 3 nu_lat`` and ``steps = T / dt``.  Time and space are therefore
refined together (convective scaling) and the temporal error is ``O(dx)``, so an
order of at least one is what a *first-order* scheme delivers -- the criterion
``p >= 1`` is a lower bound, not a target.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["RefinementLevel", "refinement_levels", "richardson_order", "richardson_residuals"]


@dataclass(frozen=True)
class RefinementLevel:
    name: str
    n: int
    dx: float
    dt: float
    tau: float
    nu_lat: float
    steps: int
    u_lat: float
    length: float
    speed: float
    viscosity: float
    window: float

    @property
    def cells(self) -> int:
        return self.n * self.n

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "n": self.n,
            "dx_m": self.dx,
            "dt_s": self.dt,
            "tau": self.tau,
            "nu_lattice": self.nu_lat,
            "steps": self.steps,
            "u_lattice": self.u_lat,
            "physical": {
                "length_m": self.length,
                "speed_m_s": self.speed,
                "viscosity_m2_s": self.viscosity,
                "window_s": self.window,
            },
        }


def refinement_levels(
    ns=(64, 96, 128),
    *,
    length: float = 0.1,
    speed: float = 0.01,
    viscosity: float = 1e-5,
    u_lat: float = 0.02,
    window: float = 0.5,
) -> list[RefinementLevel]:
    """Build the three audit levels (their ratio is 1.5, used by Richardson)."""
    levels = []
    for n in ns:
        dx = length / n
        dt = u_lat * dx / speed
        nu_lat = viscosity * dt / dx ** 2
        tau = 0.5 + 3.0 * nu_lat
        steps = int(round(window / dt))
        levels.append(
            RefinementLevel(
                name=f"n{n}", n=int(n), dx=dx, dt=dt, tau=tau, nu_lat=nu_lat,
                steps=steps, u_lat=u_lat, length=length, speed=speed,
                viscosity=viscosity, window=window,
            )
        )
    return levels


def richardson_order(values, ratio: float) -> float:
    """Observed order from three solutions of the same functional.

    With ``e_i = C h_i**p`` the ratio of successive differences is
    ``(q3 - q2) / (q2 - q1) = r**(-p)``, hence ``p = -ln(ratio)/ln(r)`` (the sign
    matters: the ``+`` form silently reports ``-p``, which turns a diverging
    sequence into a converging one).  ``q1`` is the coarsest and ``q3`` the
    finest grid.  Returns ``nan`` when the sequence is not monotone (the honest
    answer: the three levels do not lie in the asymptotic range) and a negative
    order when the differences *grow* under refinement, i.e. when the study has
    not reached its asymptotic range at all.
    """
    q = [float(v) for v in values]
    if len(q) != 3:
        raise ValueError("richardson_order needs exactly three grid levels")
    d1, d2 = q[1] - q[0], q[2] - q[1]
    if d1 == 0.0 or d2 == 0.0:
        return float("nan")
    r = d2 / d1
    if r <= 0.0:
        return float("nan")
    return float(-math.log(r) / math.log(ratio))


def richardson_residuals(values, ratio: float):
    """Return ``(order, extrapolated, residuals)``.

    ``residuals[i] = |q_i - q_ref| / |q_ref|`` with ``q_ref`` the Richardson
    extrapolation of the three values; by construction they decay like
    ``dx**p``, which is exactly the *error bound* claim of the audit and is
    reported next to the structural (round-off) momentum residual.
    """
    q = [float(v) for v in values]
    p = richardson_order(q, ratio)
    if not math.isfinite(p):
        return float("nan"), float("nan"), [float("nan")] * 3
    q_ref = q[2] + (q[2] - q[1]) / (ratio ** p - 1.0)
    scale = abs(q_ref) if q_ref != 0.0 else 1.0
    res = [abs(v - q_ref) / scale for v in q]
    return p, q_ref, res