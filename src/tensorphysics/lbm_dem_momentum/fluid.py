"""D2Q9 BGK lattice Boltzmann solver with exact momentum-exchange boundaries.

Design constraints (they are what makes the audit possible):

* **Full-way bounce back** on every fluid/solid link, with the moving-wall
  correction ``f_opp(i) = f_post(i) - 2 w_i rho (c_i . u_w) / cs^2``.  The
  momentum handed to the wall by one link is

      ``dP = 2 c_i f_post(i) - 2 w_i rho (c_i . u_w) c_i / cs^2``

  i.e. *exactly* the negative of the momentum the fluid node lost relative to
  free streaming.  The exchange is therefore **adjoint by construction**: the
  particle receives the bit-identical impulse the fluid gave up, so the
  interface cannot inject momentum.  This is the property measured by the
  ``<= 1e-10`` momentum criterion (and the reason its residual is round-off).

* **Exact wall tallies.** Every bounce-back link is grouped (outer wall or
  particle ``k``) and its impulse is accumulated, so the system momentum budget
  is closed without an approximate force model.

* **Node conversion accounting.** When a particle moves, lattice nodes switch
  between fluid and solid.  Destroyed fluid nodes hand their populations'
  momentum to the particle; created fluid nodes are initialised at
  ``(rho0, u_particle)`` and the same momentum is *subtracted* from the
  particle.  Without this bookkeeping a moving staircase boundary would leak
  momentum of order ``rho * u`` per conversion (``~1e-2`` relative for the
  configurations here) and no ``1e-10`` claim could be made.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lattice import C, CS2, E, OPP, Q, W, equilibrium, shift, strain_rate_from_noneq, viscosity

__all__ = ["StepDiagnostics", "LBMFluid2D", "OUTSIDE", "FLUID"]

OUTSIDE = -2
FLUID = -1


@dataclass
class StepDiagnostics:
    """Per-step bookkeeping returned by :meth:`LBMFluid2D.step`."""

    wall_impulse: np.ndarray          # (2,) impulse handed to the outer walls
    particle_impulse: dict            # {particle index: (2,) impulse}
    wall_power: float                 # lattice energy handed to the outer walls
    wall_work: float                  # mechanical work done by the outer walls on the fluid
    particle_power: dict              # {particle index: lattice energy handed to it}
    collision_energy_loss: float      # exact BGK lattice energy removed by collision
    kinetic_energy: float             # sum 0.5 rho |u|^2 over fluid nodes
    dissipation: float                # sum 2 mu S:S (independent estimate)
    mass: float
    momentum: np.ndarray              # (2,) fluid momentum sum_i f_i c_i


class LBMFluid2D:
    """Closed-box D2Q9 BGK solver with moving bounce-back boundaries."""

    def __init__(self, nx: int, ny: int, tau: float = 0.6, rho0: float = 1.0):
        self.nx = int(nx)
        self.ny = int(ny)
        self.tau = float(tau)
        self.rho0 = float(rho0)
        self.nu = viscosity(self.tau)
        self.mu = self.rho0 * self.nu
        shape = (Q, self.nx, self.ny)
        self.f = equilibrium(self.rho0, 0.0, 0.0)[:, None, None] * np.ones(shape)
        self.f_post = np.empty(shape)
        self.f_new = np.empty(shape)
        self.solid = np.zeros((self.nx, self.ny), dtype=bool)
        self.node_id = np.full((self.nx, self.ny), FLUID, dtype=np.int64)
        self.particle_velocity = np.zeros((0, 2))
        self.wall_velocity = {name: np.zeros(2) for name in ("west", "east", "south", "north")}
        self.body_force = np.zeros((self.nx, self.ny, 2))
        self.rho = np.full((self.nx, self.ny), self.rho0)
        self.ux = np.zeros((self.nx, self.ny))
        self.uy = np.zeros((self.nx, self.ny))
        self.time = 0
        self.conversion_energy = 0.0
        self._grid_x = np.arange(self.nx)[:, None] * np.ones((1, self.ny), dtype=np.int64)
        self._grid_y = np.ones((self.nx, 1), dtype=np.int64) * np.arange(self.ny)[None, :]

    # ------------------------------------------------------------------ state
    @property
    def fluid(self) -> np.ndarray:
        return ~self.solid

    def set_state(self, rho, ux, uy) -> None:
        """Overwrite the populations with a local equilibrium state."""
        rho = np.broadcast_to(np.asarray(rho, dtype=float), (self.nx, self.ny)).copy()
        ux = np.broadcast_to(np.asarray(ux, dtype=float), (self.nx, self.ny)).copy()
        uy = np.broadcast_to(np.asarray(uy, dtype=float), (self.nx, self.ny)).copy()
        ux[self.solid] = 0.0
        uy[self.solid] = 0.0
        self.f[:] = equilibrium(rho, ux, uy)
        self.f[:, self.solid] = 0.0
        self._macroscopic()

    def macroscopic(self):
        rho = self.f.sum(axis=0)
        ux = np.tensordot(C[:, 0], self.f, axes=(0, 0)) / rho
        uy = np.tensordot(C[:, 1], self.f, axes=(0, 0)) / rho
        return rho, ux, uy

    def _macroscopic(self) -> None:
        self.rho, self.ux, self.uy = self.macroscopic()

    def momentum(self) -> np.ndarray:
        px = np.tensordot(C[:, 0], self.f, axes=(0, 0))
        py = np.tensordot(C[:, 1], self.f, axes=(0, 0))
        fl = ~self.solid
        return np.array([float(px[fl].sum()), float(py[fl].sum())])

    def mass(self) -> float:
        return float(self.f[:, ~self.solid].sum())

    def kinetic_energy(self) -> float:
        ke = 0.5 * self.rho * (self.ux * self.ux + self.uy * self.uy)
        return float(ke[~self.solid].sum())

    def lattice_energy(self) -> float:
        return float(np.tensordot(E, self.f[:, ~self.solid], axes=(0, 0)).sum())

    def dissipation(self) -> float:
        """``sum 2 mu S:S`` -- the *independent* (continuum) dissipation rate."""
        feq = equilibrium(self.rho, self.ux, self.uy)
        sxx, syy, sxy = strain_rate_from_noneq(self.f, feq, self.rho, self.tau)
        rate = 2.0 * self.mu * (sxx * sxx + 2.0 * sxy * sxy + syy * syy)
        return float(rate[~self.solid].sum())

    def restore_mass(self, target: float) -> float:
        """Restore lattice mass to ``target`` on the rest population only.

        Moving bounce back exchanges mass at O(Ma^2) whenever the boundary link
        set is not mirror symmetric (a staircase always is not).  The deficit is
        spread uniformly over the rest population, which carries **no momentum
        and no lattice kinetic energy**, so this repair is momentum exact and
        removes a known artefact instead of hiding it.  The returned value is the
        added mass.
        """
        m = self.mass()
        dm = float(target) - m
        if dm == 0.0:
            return 0.0
        n_fluid = int((~self.solid).sum())
        if n_fluid <= 0:
            return 0.0
        self.f[0][~self.solid] += dm / n_fluid
        self._macroscopic()
        return dm

    # ------------------------------------------------------------- geometry
    def apply_mask(self, solid: np.ndarray, node_id: np.ndarray, particle_velocity: np.ndarray):
        """Install a new solid mask, booking node conversions exactly.

        Returns ``(fluid_impulse, per_particle_impulse)``.  ``fluid_impulse`` is
        the momentum the fluid gained through node creation/destruction;
        ``per_particle_impulse[k]`` is what particle ``k`` gained, so the two
        sums cancel to round-off.
        """
        solid = np.asarray(solid, dtype=bool)
        node_id = np.asarray(node_id, dtype=np.int64)
        particle_velocity = np.asarray(particle_velocity, dtype=np.float64).reshape(-1, 2)
        if solid.shape != self.solid.shape:
            raise ValueError("mask shape mismatch")
        fluid_impulse = np.zeros(2)
        per_particle = {k: np.zeros(2) for k in range(particle_velocity.shape[0])}
        e_removed = 0.0
        e_added = 0.0

        created = self.solid & ~solid          # solid -> fluid
        destroyed = ~self.solid & solid        # fluid -> solid

        if destroyed.any():
            xs, ys = np.nonzero(destroyed)
            px = np.tensordot(C[:, 0], self.f[:, xs, ys], axes=(0, 0))
            py = np.tensordot(C[:, 1], self.f[:, xs, ys], axes=(0, 0))
            ids = node_id[xs, ys]
            taken = np.zeros(2)
            for k in range(particle_velocity.shape[0]):
                sel = ids == k
                if sel.any():
                    per_particle[k] = np.array([float(px[sel].sum()), float(py[sel].sum())])
                    taken += per_particle[k]
            # links with no particle id (only possible for wall nodes) stay unclaimed
            fluid_impulse -= taken
            rho_d = self.f.sum(axis=0)[xs, ys]
            ux_d = px / np.where(rho_d > 0.0, rho_d, 1.0)
            uy_d = py / np.where(rho_d > 0.0, rho_d, 1.0)
            e_removed = float(np.sum(0.5 * rho_d * (ux_d * ux_d + uy_d * uy_d)))
            self.f[:, xs, ys] = 0.0

        if created.any():
            xs, ys = np.nonzero(created)
            ids = node_id[xs, ys]
            uv = np.zeros((xs.size, 2))
            for k in range(particle_velocity.shape[0]):
                sel = ids == k
                if sel.any():
                    uv[sel] = particle_velocity[k]
            self.f[:, xs, ys] = equilibrium(self.rho0, uv[:, 0], uv[:, 1])
            px = np.tensordot(C[:, 0], self.f[:, xs, ys], axes=(0, 0))
            py = np.tensordot(C[:, 1], self.f[:, xs, ys], axes=(0, 0))
            given = np.zeros(2)
            for k in range(particle_velocity.shape[0]):
                sel = ids == k
                if sel.any():
                    pk = np.array([float(px[sel].sum()), float(py[sel].sum())])
                    per_particle[k] += -pk
                    given += pk
            fluid_impulse += given
            e_added = float(np.sum(0.5 * self.rho0 * np.sum(uv * uv, axis=1)))

        self.solid = solid
        self.node_id = np.where(solid, node_id, FLUID)
        self.particle_velocity = particle_velocity
        self._macroscopic()
        #: kinetic energy created / destroyed by node conversion.  This is a
        #: numerical source/sink of the moving-staircase method (momentum is
        #: exact, energy is not) and is reported as its own ledger line so that
        #: it cannot hide inside the energy residual.
        self.conversion_energy = e_added - e_removed
        return fluid_impulse, per_particle

    # ------------------------------------------------------------------ step
    def _collide(self):
        rho = self.f.sum(axis=0)
        ux = np.tensordot(C[:, 0], self.f, axes=(0, 0)) / rho
        uy = np.tensordot(C[:, 1], self.f, axes=(0, 0)) / rho
        fx = self.body_force[:, :, 0]
        fy = self.body_force[:, :, 1]
        # Guo forcing: half-explicit velocity shift
        ux = ux + 0.5 * fx / rho
        uy = uy + 0.5 * fy / rho
        self.rho, self.ux, self.uy = rho, ux, uy
        feq = equilibrium(rho, ux, uy)
        omega = 1.0 / self.tau
        fpost = self.f_post
        np.subtract(self.f, feq, out=fpost)
        fpost *= -omega
        fpost += self.f
        if fx.any() or fy.any():
            pref = 1.0 - 0.5 * omega
            for i in range(Q):
                ci_f = C[i, 0] * fx + C[i, 1] * fy
                cu = C[i, 0] * ux + C[i, 1] * uy
                src = pref * W[i] * (((C[i, 0] - ux) * fx + (C[i, 1] - uy) * fy) / CS2
                                     + cu * ci_f / (CS2 * CS2))
                fpost[i] += src
        # exact BGK energy removed by collision (lattice measure)
        loss = -float(np.tensordot(E, fpost - self.f, axes=(0, 0))[~self.solid].sum())
        return loss

    def _wall_velocity_at(self, i, outside_mask):
        """Wall velocity for links in direction ``i`` that leave the domain."""
        cx, cy = C[i]
        ux = np.zeros((self.nx, self.ny))
        uy = np.zeros((self.nx, self.ny))
        gx, gy = self._grid_x, self._grid_y
        east = outside_mask & (gx + cx >= self.nx)
        west = outside_mask & (gx + cx < 0)
        north = outside_mask & ~east & ~west & (gy + cy >= self.ny)
        south = outside_mask & ~east & ~west & (gy + cy < 0)
        for name, mask in (("east", east), ("west", west), ("north", north), ("south", south)):
            if mask.any():
                v = self.wall_velocity[name]
                ux[mask] = v[0]
                uy[mask] = v[1]
        return ux, uy

    def step(self) -> StepDiagnostics:
        """Advance one lattice step and return the exact exchange bookkeeping."""
        loss = self._collide()
        fpost = self.f_post
        fnew = self.f_new
        fnew[:] = 0.0
        fluid = ~self.solid
        rho = self.rho
        n_part = self.particle_velocity.shape[0]
        wall_imp = np.zeros(2)
        wall_pow = 0.0
        wall_work = 0.0
        part_imp = {k: np.zeros(2) for k in range(n_part)}
        part_pow = {k: 0.0 for k in range(n_part)}

        # the rest population does not stream
        fnew[0] = np.where(fluid, fpost[0], 0.0)

        for j in range(1, Q):
            cx, cy = int(C[j, 0]), int(C[j, 1])
            tx0, tx1 = max(0, cx), min(self.nx, self.nx + cx)
            ty0, ty1 = max(0, cy), min(self.ny, self.ny + cy)
            if tx1 <= tx0 or ty1 <= ty0:
                continue
            src = fpost[j, tx0 - cx: tx1 - cx, ty0 - cy: ty1 - cy]
            ok = fluid[tx0 - cx: tx1 - cx, ty0 - cy: ty1 - cy] & fluid[tx0:tx1, ty0:ty1]
            fnew[j, tx0:tx1, ty0:ty1] = np.where(ok, src, 0.0)

        for i in range(1, Q):
            blocked = shift(self.solid, C[i], fill=True)        # neighbour in dir i is solid / outside
            link = fluid & blocked
            if not link.any():
                continue
            ids = shift(self.node_id, C[i], fill=OUTSIDE)
            outside = link & (ids == OUTSIDE)
            uwx = np.zeros((self.nx, self.ny))
            uwy = np.zeros((self.nx, self.ny))
            if outside.any():
                uwx, uwy = self._wall_velocity_at(i, outside)
            for k in range(n_part):
                sel = link & (ids == k)
                if sel.any():
                    uwx[sel] = self.particle_velocity[k, 0]
                    uwy[sel] = self.particle_velocity[k, 1]
            cuw = C[i, 0] * uwx + C[i, 1] * uwy
            corr = 2.0 * W[i] * rho * cuw / CS2
            fnew[OPP[i]][link] = fpost[i][link] - corr[link]
            # impulse handed to the wall (negative of what the fluid lost)
            mx = 2.0 * C[i, 0] * fpost[i] - corr * C[i, 0]
            my = 2.0 * C[i, 1] * fpost[i] - corr * C[i, 1]
            pw = E[i] * corr
            if outside.any():
                wall_imp += np.array([mx[outside].sum(), my[outside].sum()])
                wall_pow += float(pw[outside].sum())
                # mechanical work of the wall on the fluid: -(momentum taken by
                # the wall) . u_wall ; identically zero for a stationary wall
                wall_work -= float(np.sum((mx[outside] * uwx[outside]
                                           + my[outside] * uwy[outside])))
            for k in range(n_part):
                sel = link & (ids == k)
                if sel.any():
                    part_imp[k] += np.array([mx[sel].sum(), my[sel].sum()])
                    part_pow[k] += float(pw[sel].sum())

        self.f, self.f_new = fnew, self.f
        self._macroscopic()
        self.time += 1
        return StepDiagnostics(
            wall_impulse=wall_imp,
            particle_impulse=part_imp,
            wall_power=wall_pow,
            wall_work=wall_work,
            particle_power=part_pow,
            collision_energy_loss=loss,
            kinetic_energy=self.kinetic_energy(),
            dissipation=self.dissipation(),
            mass=self.mass(),
            momentum=self.momentum(),
        )