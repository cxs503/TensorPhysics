"""Bow-impact contact model: geometry, crushing pressure-area law, ramming dynamics.

Geometry
--------
Two idealised icebreaker bows are supported:

* :class:`WedgeBow` -- a plan-view wedge of half angle ``alpha`` penetrating the
  ice edge; the crushed contact width grows as ``b = 2 delta tan(alpha)`` up to a
  cap ``b_max`` (waterline fullness).
* :class:`SphericalBow` -- a bow of radius ``R``; chord ``b = 2 sqrt(2 R delta)``.

Crushing
--------
The projected crushed area is ``A = b(delta) h_i`` (contact width times ice
thickness) and the crushing pressure follows the empirical pressure-area law

    p(A) = C A^(-n),      n = 0.5,  C = 2.6 MPa m^0.5 (first-year ice),

truncated to ``[p_min, p_max]``.  The law with ``n = 0.5`` is the classical
pressure-area envelope fitted to full-scale and medium-scale ice load data
(Sanderson 1988; Masterson & Frederking 1998); the coefficient is exposed as a
parameter so that sensitivity to the empirical envelope can be quantified
instead of hidden.

Ramming dynamics
----------------
During a ram the vessel decelerates against the ice:

    M dV/dt = -F,   d(delta)/dt = V,   F = p(A(delta)) A(delta).

With an unbounded wedge, an exact ``n = 0.5`` law and no other sink, the
kinetic-energy balance ``M V^2 / 2 = int_0^delta K s^{1/2} ds`` with
``K = C sqrt(2 tan(alpha) h_i)`` gives the closed-form peak force

    F_peak = K^{2/3} ( (3/4) M V^2 )^{1/3},

which is used as an independent analytic anchor for the time integrator
(:func:`ram_rigid_closed_form`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "CrushingLaw",
    "WedgeBow",
    "SphericalBow",
    "RammingSolution",
    "pressure_area",
    "crushing_force",
    "contact_area",
    "ram_rigid_closed_form",
    "simulate_ram_rigid",
]


@dataclass(frozen=True)
class CrushingLaw:
    """Empirical pressure-area crushing law ``p = C A^(-n)`` (SI units)."""

    coefficient: float = 2.6e6      # C [Pa m^(2n)]
    exponent: float = 0.5
    pressure_min: float = 1.0e5     # [Pa]
    pressure_max: float = 1.2e7     # [Pa]

    def __post_init__(self) -> None:
        if self.coefficient <= 0.0:
            raise ValueError("coefficient must be positive")
        if not 0.0 <= self.exponent < 1.0:
            raise ValueError("exponent must lie in [0, 1)")
        if not 0.0 < self.pressure_min < self.pressure_max:
            raise ValueError("need 0 < pressure_min < pressure_max")

    def pressure(self, area) -> np.ndarray:
        A = np.asarray(area, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            p = self.coefficient * np.maximum(A, 1e-12) ** (-self.exponent)
        p = np.clip(p, self.pressure_min, self.pressure_max)
        return p if p.ndim else float(p)

    def force(self, area) -> np.ndarray:
        A = np.asarray(area, dtype=float)
        f = self.pressure(A) * A
        return f if np.ndim(f) else float(f)


@dataclass(frozen=True)
class WedgeBow:
    """Plan-view wedge bow (the classic icebreaker stem)."""

    half_angle_deg: float = 20.0
    max_contact_width: float = 12.0   # [m]
    label: str = "wedge"

    @property
    def half_angle_rad(self) -> float:
        return float(np.deg2rad(self.half_angle_deg))

    def contact_width(self, penetration) -> np.ndarray:
        d = np.maximum(np.asarray(penetration, dtype=float), 0.0)
        b = 2.0 * d * np.tan(self.half_angle_rad)
        b = np.minimum(b, self.max_contact_width)
        return b if np.ndim(b) else float(b)

    def contact_area(self, penetration, ice_thickness: float) -> np.ndarray:
        return self.contact_width(penetration) * ice_thickness


@dataclass(frozen=True)
class SphericalBow:
    """Spherical / rounded bow of radius ``R``."""

    radius: float = 4.0               # [m]
    max_contact_width: float = 12.0   # [m]
    label: str = "sphere"

    def contact_width(self, penetration) -> np.ndarray:
        d = np.maximum(np.asarray(penetration, dtype=float), 0.0)
        b = 2.0 * np.sqrt(2.0 * self.radius * d)
        b = np.minimum(b, self.max_contact_width)
        return b if np.ndim(b) else float(b)

    def contact_area(self, penetration, ice_thickness: float) -> np.ndarray:
        return self.contact_width(penetration) * ice_thickness


def contact_area(bow, penetration, ice_thickness: float) -> np.ndarray:
    """Projected crushed area ``A(delta)`` [m^2]."""
    return bow.contact_area(penetration, ice_thickness)


def pressure_area(area, law: CrushingLaw | None = None) -> np.ndarray:
    """Crushing pressure [Pa] for a projected contact area [m^2]."""
    return (law or CrushingLaw()).pressure(area)


def crushing_force(area, law: CrushingLaw | None = None) -> np.ndarray:
    """Crushing contact force [N] for a projected contact area [m^2]."""
    return (law or CrushingLaw()).force(area)


def ram_rigid_closed_form(
    ship_mass: float,
    impact_speed: float,
    bow: WedgeBow,
    ice_thickness: float,
    law: CrushingLaw | None = None,
) -> dict:
    """Analytic peak force of an unbounded rigid ram (see module docstring)."""
    law = law or CrushingLaw()
    if law.exponent != 0.5:
        raise ValueError("closed form requires exponent == 0.5")
    if not isinstance(bow, WedgeBow):
        raise ValueError("closed form implemented for the wedge bow")
    if bow.contact_width(1e9) <= bow.max_contact_width and False:  # pragma: no cover
        pass
    K = law.coefficient * np.sqrt(2.0 * np.tan(bow.half_angle_rad) * ice_thickness)
    energy = 0.5 * ship_mass * impact_speed**2
    delta_max = (1.5 * energy / K) ** (2.0 / 3.0)
    f_peak = K * np.sqrt(delta_max)
    area_peak = 2.0 * delta_max * np.tan(bow.half_angle_rad) * ice_thickness
    return {
        "K_N_m0.5": float(K),
        "kinetic_energy_J": float(energy),
        "penetration_max_m": float(delta_max),
        "peak_force_N": float(f_peak),
        "peak_area_m2": float(area_peak),
        "peak_pressure_Pa": float(law.pressure(area_peak)),
    }


@dataclass
class RammingSolution:
    """Time history of a rigid-hull ram (no plate compliance)."""

    time_s: np.ndarray
    speed_m_s: np.ndarray
    penetration_m: np.ndarray
    force_N: np.ndarray
    area_m2: np.ndarray
    pressure_Pa: np.ndarray
    label: str = "rigid_ram"

    @property
    def peak_force(self) -> float:
        return float(np.max(self.force_N))

    @property
    def peak_pressure(self) -> float:
        return float(np.max(self.pressure_Pa))

    @property
    def peak_area(self) -> float:
        return float(np.max(self.area_m2))

    @property
    def time_to_stop(self) -> float:
        i = int(np.argmax(self.speed_m_s <= 1e-6 * max(self.speed_m_s[0], 1e-9)))
        return float(self.time_s[i]) if self.speed_m_s[i] <= 1e-6 * self.speed_m_s[0] else float(self.time_s[-1])

    def crushing_energy(self) -> float:
        """``int F d(delta)`` [J] by the trapezoidal rule on the history."""
        return float(np.trapezoid(self.force_N, self.penetration_m))


def simulate_ram_rigid(
    ship_mass: float,
    impact_speed: float,
    bow,
    ice_thickness: float,
    law: CrushingLaw | None = None,
    dt: float = 2.0e-4,
    max_time: float = 30.0,
    stop_speed_ratio: float = 1e-4,
) -> RammingSolution:
    """Integrate the rigid ram with classical RK4 (independent of the closed form).

    The state is ``(penetration, speed)``; the crushing force follows from the
    pressure-area law.  Integration stops when the speed drops below
    ``stop_speed_ratio * V0`` or ``max_time`` is reached.
    """
    law = law or CrushingLaw()
    if ship_mass <= 0.0 or impact_speed <= 0.0:
        raise ValueError("ship_mass and impact_speed must be positive")

    def rhs(state: np.ndarray) -> np.ndarray:
        delta, v = state
        delta = max(delta, 0.0)
        v = max(v, 0.0)
        A = bow.contact_area(delta, ice_thickness)
        F = law.force(A)
        return np.array([v, -F / ship_mass])

    state = np.array([1e-6, float(impact_speed)])
    t = 0.0
    ts, vs, ds, fs, As, ps = [t], [state[1]], [state[0]], [0.0], [0.0], [0.0]
    stop_speed = stop_speed_ratio * impact_speed
    while t < max_time and state[1] > stop_speed:
        k1 = rhs(state)
        k2 = rhs(state + 0.5 * dt * k1)
        k3 = rhs(state + 0.5 * dt * k2)
        k4 = rhs(state + dt * k3)
        state = state + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        state[0] = max(state[0], 0.0)
        state[1] = max(state[1], 0.0)
        t += dt
        A = float(bow.contact_area(state[0], ice_thickness))
        ts.append(t)
        vs.append(state[1])
        ds.append(state[0])
        fs.append(float(law.force(A)))
        As.append(A)
        ps.append(float(law.pressure(A)))
    return RammingSolution(
        time_s=np.asarray(ts),
        speed_m_s=np.asarray(vs),
        penetration_m=np.asarray(ds),
        force_N=np.asarray(fs),
        area_m2=np.asarray(As),
        pressure_Pa=np.asarray(ps),
    )