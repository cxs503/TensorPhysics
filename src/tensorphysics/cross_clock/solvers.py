"""Reduced solvers with *their own* clocks and interval-complete ledgers.

Three solvers take part in the clock study, each integrated with an exact or
closed-form step so that a measured error can be attributed to the **exchange**
and not to the integrator:

``FemStructure``
    Linear oscillator ``m x'' + c x' + k x = F`` advanced with the exact matrix
    exponential of the companion matrix under a constant force over the step.
    Its mechanical energy ``E = 0.5 m v^2 + 0.5 k x^2`` is the conserved quantity
    it exposes, and the impulse it receives is the conserved quantity it
    consumes.

``LbmFluid1D``
    D1Q3 lattice-BGK column (``c_s^2 = 1/3``) with a moving wall (the structure)
    at the left end and a rigid wall at the right end, in dimensionless lattice
    units.  Mass and momentum are booked from the populations themselves: the
    stage-wise momentum balance (collision conserves momentum exactly, streaming
    conserves it exactly, only the two wall links exchange it) gives the impulse
    on each wall without any post-hoc formula.

``DemParticle``
    1-D particle under Stokes drag plus gravity, advanced exactly for a
    piecewise-constant fluid velocity (closed-form exponential relaxation).  It
    exposes its momentum and mechanical-plus-potential energy, and receives the
    drag impulse.

The lattice/particle units are dimensionless on purpose: the node's claims are
about the *exchange*, and a dimensionless instrument cannot hide a unit error
inside a "physical" number.  Physical anchors are attached where they exist:
the acoustic-structure eigenfrequency of :func:`acoustic_structure_frequency`
and the relaxation time of the drag law.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq

__all__ = [
    "FemStructure",
    "LbmFluid1D",
    "DemParticle",
    "acoustic_structure_frequency",
    "lattice_sound_speed",
]

W_REST = 2.0 / 3.0
W_SIDE = 1.0 / 6.0
CS2 = 1.0 / 3.0


def lattice_sound_speed() -> float:
    """D1Q3 sound speed in lattice units (``c_s = 1/sqrt(3)``)."""
    return float(np.sqrt(CS2))


# --------------------------------------------------------------------------- #
# FEM structure
# --------------------------------------------------------------------------- #
@dataclass
class FemStructure:
    """1-D linear structure with exact stepping (no time-integration error).

    Units are self-consistent; the coupling only ever touches ``advance`` (an
    impulse over a window) and ``displacement`` (what the fluid needs).
    """

    mass: float
    stiffness: float
    damping: float = 0.0
    x: float = 0.0
    v: float = 0.0
    name: str = "fem"

    def __post_init__(self) -> None:
        if self.mass <= 0.0 or self.stiffness <= 0.0:
            raise ValueError("mass and stiffness must be positive")

    # -- exact propagator ------------------------------------------------ #
    def advance(self, h: float, force: float) -> tuple[float, float]:
        """Advance ``h`` under a constant force; returns the new ``(x, v)``."""
        if h <= 0.0:
            raise ValueError("h must be positive")
        m, k, c = self.mass, self.stiffness, self.damping
        A = np.array([[0.0, 1.0], [-k / m, -c / m]])
        xp = force / k
        y0 = np.array([self.x - xp, self.v])
        y = expm(A * h) @ y0
        self.x = float(y[0] + xp)
        self.v = float(y[1])
        return self.x, self.v

    # -- state ------------------------------------------------------------ #
    @property
    def displacement(self) -> float:
        return self.x

    def energy(self) -> float:
        return 0.5 * self.mass * self.v ** 2 + 0.5 * self.stiffness * self.x ** 2

    def momentum(self) -> float:
        return self.mass * self.v

    def snapshot(self) -> dict:
        return {"x": self.x, "v": self.v, "E": self.energy(), "p": self.momentum()}


# --------------------------------------------------------------------------- #
# LBM fluid (D1Q3)
# --------------------------------------------------------------------------- #
@dataclass
class LbmFluid1D:
    """D1Q3 BGK column, moving wall at node 0, rigid wall at node n-1.

    Directions: ``0`` rest, ``1`` left-going (``c = -1``), ``2`` right-going
    (``c = +1``).  Streaming is done explicitly (no wrap-around), and the two
    wall links are the only places where total lattice momentum can change --
    which is what makes the impulse booking exact.
    """

    cells: int = 64
    tau: float = 0.8
    rho0: float = 1.0
    periodic: bool = False
    name: str = "lbm"
    f: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.cells < 2:
            raise ValueError("need at least two cells")
        if self.tau <= 0.5:
            raise ValueError("tau must exceed 0.5 for positive viscosity")
        if self.f is None:
            self.f = np.zeros((self.cells, 3), dtype=float)
            self.f[:, 0] = W_REST * self.rho0
            self.f[:, 1] = W_SIDE * self.rho0
            self.f[:, 2] = W_SIDE * self.rho0
        self.steps = 0
        self.impulse_left = 0.0
        self.impulse_right = 0.0
        self.mass_left = 0.0
        self.mass_right = 0.0
        self.peak_mach = 0.0

    # -- diagnostics ------------------------------------------------------ #
    def density(self) -> np.ndarray:
        return self.f.sum(axis=1)

    def velocity(self) -> np.ndarray:
        mom = self.f[:, 2] - self.f[:, 1]
        return mom / self.density()

    def total_mass(self) -> float:
        return float(self.f.sum())

    def total_momentum(self) -> float:
        return float(np.sum(self.f[:, 2] - self.f[:, 1]))

    def pressure(self) -> np.ndarray:
        return CS2 * self.density()

    # -- one lattice step ------------------------------------------------- #
    def step(self, u_wall: float = 0.0) -> dict:
        """Collide + stream + wall links; returns the stage momentum balance."""
        f = self.f
        rho = f.sum(axis=1)
        mom = f[:, 2] - f[:, 1]
        u = mom / rho
        self.peak_mach = max(self.peak_mach, float(np.max(np.abs(u))))

        # equilibrium (at-rest weights + linear velocity term)
        feq = np.empty_like(f)
        feq[:, 0] = W_REST * rho
        feq[:, 1] = W_SIDE * rho * (1.0 - 3.0 * u)
        feq[:, 2] = W_SIDE * rho * (1.0 + 3.0 * u)
        fp = f - (f - feq) / self.tau  # post-collision (momentum preserved exactly)

        if self.periodic:
            fnew = np.empty_like(fp)
            fnew[:, 0] = fp[:, 0]
            fnew[:, 1] = np.roll(fp[:, 1], -1)   # c=-1 arrives at x from x+1
            fnew[:, 2] = np.roll(fp[:, 2], 1)    # c=+1 arrives at x from x-1
            dp_left = dp_right = dm_left = dm_right = 0.0
        else:
            fnew = np.zeros_like(fp)
            fnew[:, 0] = fp[:, 0]
            fnew[:-1, 1] = fp[1:, 1]      # c=-1 arrives at x from x+1
            fnew[1:, 2] = fp[:-1, 2]      # c=+1 arrives at x from x-1
            # left (moving) wall: reflect the outgoing population, add wall term
            wall_term = float(rho[0]) * float(u_wall)
            fnew[0, 2] = fp[0, 1] + wall_term
            # right (rigid) wall: reflect
            fnew[-1, 1] = fp[-1, 2]
            # global momentum is conserved by collision and streaming, so the two
            # wall links carry the whole balance: at the left wall the outgoing
            # population fp[0,1] (c=-1) becomes fp[0,1]+wall (c=+1); at the right
            # wall fp[-1,2] (c=+1) becomes fp[-1,2] (c=-1).
            dp_left = 2.0 * float(fp[0, 1]) + wall_term
            dp_right = -2.0 * float(fp[-1, 2])
            dm_left = wall_term
            dm_right = 0.0
            # subtract the static-pressure (gauge) part: a wall in a fluid at rest
            # exchanges +p0 per step, which is not a coupling quantity but the
            # ambient pressure the structure is in equilibrium with.
            p0 = CS2 * self.rho0
            dp_left -= p0
            dp_right += p0
        # independent check: the population-level global balance must agree
        delta_global = float(np.sum(fnew[:, 2] - fnew[:, 1])) - float(np.sum(f[:, 2] - f[:, 1]))
        if abs(delta_global - (dp_left + dp_right)) > 1e-12 * max(1.0, abs(delta_global)):
            raise RuntimeError(
                f"wall impulse booking inconsistent with global momentum balance: "
                f"{delta_global!r} vs {dp_left + dp_right!r}"
            )
        rho_new = fnew.sum(axis=1)
        self.impulse_left += dp_left      # gauge impulse received by the fluid at x=0
        self.impulse_right += dp_right
        self.mass_left += dm_left
        self.mass_right += dm_right
        self.f = fnew
        self.steps += 1
        return {
            "dp_left": dp_left,
            "dp_right": dp_right,
            "dm_left": dm_left,
            "dm_right": dm_right,
        }

    def drain(self) -> dict:
        """Take and reset the accumulated wall exchange (interval-complete)."""
        out = {
            "impulse_left": self.impulse_left,
            "impulse_right": self.impulse_right,
            "mass_left": self.mass_left,
            "mass_right": self.mass_right,
            "steps": self.steps,
        }
        self.impulse_left = 0.0
        self.impulse_right = 0.0
        self.mass_left = 0.0
        self.mass_right = 0.0
        self.steps = 0
        return out

    def set_uniform(self, rho: float, u: float = 0.0) -> None:
        self.f[:, 0] = W_REST * rho
        self.f[:, 1] = W_SIDE * rho * (1.0 - 3.0 * u)
        self.f[:, 2] = W_SIDE * rho * (1.0 + 3.0 * u)

    def set_density_profile(self, rho: np.ndarray) -> None:
        """Initialise from a density field at rest (acoustic initial condition)."""
        rho = np.asarray(rho, dtype=float)
        if rho.size != self.cells:
            raise ValueError(f"expected {self.cells} densities, got {rho.size}")
        self.f[:, 0] = W_REST * rho
        self.f[:, 1] = W_SIDE * rho
        self.f[:, 2] = W_SIDE * rho

    def add_body_force(self, force_per_cell: float) -> None:
        """Apply a momentum source uniformly: +J/2 right-going, -J/2 left-going.

        Density is untouched by construction (mass change J/2 - J/2 = 0), so a
        uniform source only moves the momentum mode.
        """
        self.f[:, 2] += 0.5 * force_per_cell
        self.f[:, 1] -= 0.5 * force_per_cell

    def add_force_profile(self, force_per_cell: np.ndarray) -> None:
        """Apply a per-cell momentum source (force density), mass-neutral."""
        j = np.asarray(force_per_cell, dtype=float).ravel()
        if j.size != self.cells:
            raise ValueError(f"expected {self.cells} forces, got {j.size}")
        self.f[:, 2] += 0.5 * j
        self.f[:, 1] -= 0.5 * j


# --------------------------------------------------------------------------- #
# DEM particle
# --------------------------------------------------------------------------- #
@dataclass
class DemParticle:
    """1-D particle with Stokes drag and gravity, exact for piecewise-constant ``u``."""

    mass: float
    drag: float          # gamma [force / velocity]
    gravity: float = 0.0  # effective gravity (buoyancy included)
    x: float = 0.0
    v: float = 0.0
    name: str = "dem"
    fluid_velocity: float = 0.0

    def __post_init__(self) -> None:
        if self.mass <= 0.0 or self.drag <= 0.0:
            raise ValueError("mass and drag must be positive")

    @property
    def relaxation_time(self) -> float:
        return self.mass / self.drag

    @property
    def terminal_velocity(self) -> float:
        return self.fluid_velocity + self.mass * self.gravity / self.drag

    def advance(self, h: float, fluid_velocity: float | None = None) -> dict:
        """Exact step for a constant fluid velocity over ``h``."""
        if h <= 0.0:
            raise ValueError("h must be positive")
        u = self.fluid_velocity if fluid_velocity is None else float(fluid_velocity)
        vt = u + self.mass * self.gravity / self.drag
        decay = float(np.exp(-h / self.relaxation_time))
        v0, x0 = self.v, self.x
        self.v = vt + (v0 - vt) * decay
        # ∫ v dt with v(t) = vt + (v0-vt) e^{-t/tau}
        self.x = x0 + vt * h + (v0 - vt) * self.relaxation_time * (1.0 - decay)
        return {
            "impulse_on_particle": self.mass * (self.v - v0),
            "displacement": self.x - x0,
            "drag_force_initial": self.drag * (v0 - u),
        }

    def energy(self) -> float:
        """Mechanical + potential energy relative to ``x = 0``."""
        return 0.5 * self.mass * self.v ** 2 - self.mass * self.gravity * self.x

    def momentum(self) -> float:
        return self.mass * self.v

    def snapshot(self) -> dict:
        return {"x": self.x, "v": self.v, "E": self.energy(), "p": self.momentum()}


# --------------------------------------------------------------------------- #
# analytic anchor: coupled acoustic-structure eigenfrequency
# --------------------------------------------------------------------------- #
def acoustic_structure_frequency(
    mass: float,
    stiffness: float,
    area: float,
    rho: float,
    c: float,
    length: float,
) -> dict:
    """Lowest eigenfrequency of a piston-structure coupled to a closed acoustic column.

    Linear acoustics in ``0 < x < L`` closed at ``L``, piston of mass ``m`` and
    spring ``k`` at ``x = 0``::

        m w^2 sin(wL/c) + rho c A w cos(wL/c) - k sin(wL/c) = 0

    which reduces in the long-wave limit to ``(m + rho A L / 3) w^2 = k + rho c^2 A / L``
    -- the classical added mass ``rho A L / 3`` plus the acoustic (bulk) stiffness.
    Both branches are returned so the lumped approximation can be *seen* to be an
    approximation: the difference is the point of the analytic anchor.
    """
    def G(w: float) -> float:
        # (m w^2 - k) sin(wL/c) - rho c A w cos(wL/c) = 0   (w > 0)
        kk = w / c
        return (mass * w * w - stiffness) * np.sin(kk * length) - rho * c * area * w * np.cos(kk * length)

    upper = np.pi * c / length * 0.999999
    lo = 1e-8 * c / length
    grid = np.linspace(lo, upper, 40001)
    vals = np.array([G(w) for w in grid])
    idx = np.where(np.sign(vals[:-1]) != np.sign(vals[1:]))[0]
    if idx.size == 0:
        raise RuntimeError("no eigenfrequency bracket found below the first acoustic mode")
    i = int(idx[0])
    root = brentq(G, grid[i], grid[i + 1], xtol=1e-14, rtol=1e-15)
    k_gas = rho * c * c * area / length
    m_add = rho * area * length / 3.0
    w_lumped = float(np.sqrt((stiffness + k_gas) / (mass + m_add)))
    return {
        "omega_exact": float(root),
        "f_exact": float(root / (2.0 * np.pi)),
        "omega_lumped": w_lumped,
        "f_lumped": w_lumped / (2.0 * np.pi),
        "lumped_relative_error": abs(w_lumped - root) / root,
        "acoustic_stiffness": k_gas,
        "added_mass": m_add,
        "first_acoustic_mode": float(np.pi * c / length),
    }