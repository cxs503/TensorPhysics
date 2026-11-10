"""Rigid circular floes (2-D DEM) used as the ice side of the coupling.

The DEM layer is deliberately minimal, but it must not fake anything that the
audit relies on:

* **One force per pair.**  A contact force is computed once per pair and then
  applied with opposite signs to the two bodies, so the internal contact
  impulse cancels exactly (bit-identical negation).  Newton's third law is
  therefore a property of the code that the test suite measures, not an
  assumption.
* **Conservative contacts by default.**  A linear spring with no dashpot
  stores ``1/2 k delta^2`` of elastic energy, which is tracked in the energy
  budget as contact potential energy; contact damping would add a second sink
  and is switched off so that the energy audit has exactly one dissipative
  channel (the viscous one).
* **Prescribed ship body.**  The ship of case 3 is not a DEM floe: its motion
  is imposed, it can therefore exchange momentum with the rest of the world
  without a bookkeeping partner, and the coupling books its exchanges to the
  ``ship`` account instead of to a floe momentum.  That is why it is passed in
  as a :class:`Floe` whose ``mass`` is ``inf`` and whose velocity is overwritten
  every step.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Floe", "DEMConfig", "DEMField"]


@dataclass
class Floe:
    """One rigid circular body (ice floe, or the prescribed ship body)."""

    x: np.ndarray
    v: np.ndarray
    radius: float
    mass: float
    fixed: bool = False

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=np.float64).reshape(2).copy()
        self.v = np.asarray(self.v, dtype=np.float64).reshape(2).copy()
        self.radius = float(self.radius)
        self.mass = float(self.mass)


@dataclass(frozen=True)
class DEMConfig:
    """Contact law and integration settings (lattice units)."""

    stiffness: float = 2.0
    restitution: float = 1.0
    substeps: int = 4


class DEMField:
    """A set of :class:`Floe` bodies with pairwise linear-spring contacts."""

    def __init__(self, floes, config: DEMConfig = DEMConfig()):
        self.floes = list(floes)
        self.config = config
        self.contact_potential = 0.0

    # ------------------------------------------------------------------ state
    def positions(self) -> np.ndarray:
        return np.array([f.x for f in self.floes]) if self.floes else np.zeros((0, 2))

    def velocities(self) -> np.ndarray:
        return np.array([f.v for f in self.floes]) if self.floes else np.zeros((0, 2))

    def radii(self) -> np.ndarray:
        return np.array([f.radius for f in self.floes])

    def masses(self) -> np.ndarray:
        return np.array([f.mass for f in self.floes])

    def momentum(self) -> np.ndarray:
        p = np.zeros(2)
        for f in self.floes:
            if not f.fixed and np.isfinite(f.mass):
                p += f.mass * f.v
        return p

    def kinetic_energy(self) -> float:
        e = 0.0
        for f in self.floes:
            if not f.fixed and np.isfinite(f.mass):
                e += 0.5 * f.mass * float(f.v @ f.v)
        return e

    def n_dynamic(self) -> int:
        return sum(1 for f in self.floes if not f.fixed and np.isfinite(f.mass))

    # --------------------------------------------------------------- contacts
    def contacts(self):
        """Return ``(forces, potential)`` for the pairwise contacts.

        Only pairs of *dynamic* floes are treated here; contacts with the
        prescribed ship body are handled by :meth:`wall_forces` because the ship
        is an external actuator for the momentum ledger.
        """
        n = len(self.floes)
        forces = np.zeros((n, 2))
        potential = 0.0
        k = self.config.stiffness
        for i in range(n):
            for j in range(i + 1, n):
                fi, fj = self.floes[i], self.floes[j]
                if fi.fixed or fj.fixed:
                    continue
                d = fi.x - fj.x
                dist = float(np.hypot(d[0], d[1]))
                overlap = fi.radius + fj.radius - dist
                if overlap <= 0.0:
                    continue
                if dist <= 1e-12:
                    normal = np.array([1.0, 0.0])
                else:
                    normal = d / dist
                fmag = k * overlap
                fvec = fmag * normal
                forces[i] += fvec
                forces[j] -= fvec          # bit-identical negation -> Newton 3
                potential += 0.5 * fmag * overlap
        self.contact_potential = potential
        return forces, potential

    def wall_forces(self, box, wall_velocity=None):
        """Contacts with the domain walls and with prescribed bodies.

        ``box = (nx, ny)``; ``wall_velocity`` is a ``{"west"/"east"/...: (2,)}``
        map.  Returns ``(forces, potential, impulses)`` where the per-floe
        impulses will be booked as external by the caller.
        """
        forces = np.zeros((len(self.floes), 2))
        potential = 0.0
        k = self.config.stiffness
        nx, ny = box
        wall_velocity = wall_velocity or {}
        for i, f in enumerate(self.floes):
            if f.fixed or not np.isfinite(f.mass):
                continue
            # prescribed bodies act as moving walls with an oriented boundary
            for j, body in enumerate(self.floes):
                if body is f or not (body.fixed or not np.isfinite(body.mass)):
                    continue
                d = f.x - body.x
                dist = float(np.hypot(d[0], d[1]))
                overlap = f.radius + body.radius - dist
                if overlap <= 0.0:
                    continue
                normal = d / dist if dist > 1e-12 else np.array([1.0, 0.0])
                fmag = k * overlap
                forces[i] += fmag * normal
                potential += 0.5 * fmag * overlap
            # axis-aligned walls: the floe centre plus radius must stay inside
            for side, slack, normal in (
                ("west", f.x[0] - f.radius, np.array([1.0, 0.0])),
                ("east", nx - (f.x[0] + f.radius), np.array([-1.0, 0.0])),
                ("south", f.x[1] - f.radius, np.array([0.0, 1.0])),
                ("north", ny - (f.x[1] + f.radius), np.array([0.0, -1.0])),
            ):
                if slack < 0.0:
                    fmag = k * (-slack)
                    forces[i] += fmag * normal
                    potential += 0.5 * fmag * (-slack)
        return forces, potential

    # ------------------------------------------------------------ integration
    def advance(self, dt: float, force: np.ndarray) -> None:
        """Symplectic Euler update with sub-stepping for contact stiffness."""
        sub = max(1, int(self.config.substeps))
        h = dt / sub
        for _ in range(sub):
            self._substep(h, force)

    def _substep(self, h: float, force: np.ndarray) -> None:
        contacts, _ = self.contacts()
        total = force + contacts
        for i, f in enumerate(self.floes):
            if f.fixed or not np.isfinite(f.mass):
                continue
            f.v = f.v + h * total[i] / f.mass
            f.x = f.x + h * f.v

    def apply_impulse(self, impulses) -> None:
        """Apply per-body impulses (lattice units: ``dt = 1``)."""
        for i, f in enumerate(self.floes):
            if f.fixed or not np.isfinite(f.mass):
                continue
            if i in impulses:
                f.v = f.v + np.asarray(impulses[i], dtype=np.float64) / f.mass

    # ------------------------------------------------------------- rasterising
    def rasterize(self, nx: int, ny: int, solid=None, node_id=None,
                  first_id: int = 0):
        """Mark lattice nodes inside a body (staircase boundary).

        ``solid`` / ``node_id`` are updated in place when provided, otherwise
        fresh arrays are returned as ``(solid, node_id)``.
        """
        if solid is None:
            solid = np.zeros((nx, ny), dtype=bool)
            node_id = np.full((nx, ny), -1, dtype=np.int64)
        gx = np.arange(nx, dtype=np.float64)[:, None] + 0.5
        gy = np.arange(ny, dtype=np.float64)[None, :] + 0.5
        for k, f in enumerate(self.floes):
            inside = (gx - f.x[0]) ** 2 + (gy - f.x[1]) ** 2 < f.radius * f.radius
            solid |= inside
            node_id = np.where(inside, first_id + k, node_id)
        return solid, node_id