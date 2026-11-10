"""D3Q19 / D2Q9 lattice Boltzmann fluid kernel with free surface.

Why this file exists
--------------------
The three-field coupler needs a fluid field whose *lateral boundary condition*
can be switched between "plane flow" (2-D, the plane-strain baseline) and
"3-D open water" without touching anything else.  That controlled switch is the
whole point of the campaign node: it isolates the 3-D lateral-relief effect.

Numerics
--------
* D3Q19 (``dim=3``) or D2Q9 embedded in the (x, z) plane (``dim=2``, ``ny=1``).
* BGK collision with Guo body forcing (gravity for the free-surface column).
* Half-way bounce-back for solid cells, with the Ladd momentum-exchange force
  and the moving-wall velocity correction, so wall forces are *measured* from
  the populations instead of being post-processed from a pressure integral.
* Free surface: Körner--Thürey mass tracking (liquid / interface / gas cells,
  fill fraction phi = m / rho, gas reconstruction at atmospheric pressure,
  explicit mass advection between full and interface cells).

Units are lattice units throughout.  The mapping to SI is declared by the
caller; nothing here assumes a Reynolds number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Lattice velocity sets.  Axis order of every array in this module is (z, y, x)
# and the velocity tuples are therefore (cx, cy, cz) listed as (cx, cy, cz) with
# components addressed as c[..., 0] -> x, c[..., 1] -> y, c[..., 2] -> z.
# ---------------------------------------------------------------------------

D3Q19 = np.array(
    [
        [0, 0, 0],
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1],
        [1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0],
        [1, 0, 1], [-1, 0, -1], [1, 0, -1], [-1, 0, 1],
        [0, 1, 1], [0, -1, -1], [0, 1, -1], [0, -1, 1],
    ],
    dtype=int,
)
W3Q19 = np.array(
    [1.0 / 3.0]
    + [1.0 / 18.0] * 6
    + [1.0 / 36.0] * 12,
    dtype=float,
)

#: D2Q9 traced in the (x, z) plane; the y component is identically zero.
D2Q9 = np.array(
    [
        [0, 0, 0],
        [1, 0, 0], [-1, 0, 0],
        [0, 0, 1], [0, 0, -1],
        [1, 0, 1], [-1, 0, -1], [1, 0, -1], [-1, 0, 1],
    ],
    dtype=int,
)
W2Q9 = np.array(
    [4.0 / 9.0] + [1.0 / 9.0] * 4 + [1.0 / 36.0] * 4, dtype=float
)

CS2 = 1.0 / 3.0


def opposite_table(c: np.ndarray) -> np.ndarray:
    lookup = {tuple(v): i for i, v in enumerate(c)}
    opp = np.zeros(len(c), dtype=int)
    for i, v in enumerate(c):
        opp[i] = lookup[tuple(-v)]
    return opp


def _shift(a: np.ndarray, c: Sequence[int], periodic: Sequence[bool] = (False, False, False)) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(b, valid)`` with ``b[z,y,x] = a[z-cz, y-cy, x-cx]``.

    ``valid`` is False wherever the source index fell outside the domain, which
    lets the caller apply bounce-back there instead of silently wrapping.
    Axes flagged in ``periodic`` wrap instead of returning invalid sources.
    """
    b = a
    valid = np.ones(a.shape, dtype=bool)
    # c is given as (cx, cy, cz); array axes are (z, y, x) -> reverse the order
    for ax, ci in enumerate(reversed(tuple(c))):  # ax 0 -> z, 1 -> y, 2 -> x
        if ci == 0:
            continue
        if periodic[ax]:
            b = np.roll(b, ci, axis=ax)
            continue
        b = np.roll(b, ci, axis=ax)
        valid = np.roll(valid, ci, axis=ax)
        if ci > 0:
            sl = [slice(None)] * a.ndim
            sl[ax] = slice(0, ci)
            b[tuple(sl)] = 0.0
            valid[tuple(sl)] = False
        else:
            sl = [slice(None)] * a.ndim
            sl[ax] = slice(ci, None)
            b[tuple(sl)] = 0.0
            valid[tuple(sl)] = False
    return b, valid


@dataclass
class FreeSurfaceConfig:
    """Parameters of the mass-tracking free surface."""

    enabled: bool = False
    #: Initial water depth in lattice cells measured from the bottom wall.
    initial_depth: int = 0
    #: Conversion thresholds for interface -> liquid / gas.
    liquid_threshold: float = 1.0 + 1e-6
    gas_threshold: float = 1e-6
    #: Below this many interface cells nothing is done (dry domain).
    min_interface: int = 1


@dataclass
class LatticeSpec:
    """Grid + collision parameters in lattice units."""

    nx: int
    ny: int
    nz: int
    dim: int = 3
    tau: float = 0.8
    rho0: float = 1.0
    gravity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    #: Boundary treatment on the y faces (dim=3): "wall" or "periodic".
    y_boundary: str = "wall"
    #: Velocity (lattice units) prescribed on the x = 0 inlet plane.
    inlet_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    #: Top boundary when the free surface is disabled: "slip" or "wall".
    top_boundary: str = "slip"
    #: Periodic streamwise boundary (used by the analytic Poiseuille check).
    x_periodic: bool = False
    #: Initial velocity field; defaults to the inlet velocity.
    initial_velocity: Optional[Tuple[float, float, float]] = None

    def __post_init__(self) -> None:
        if self.nx < 2 or self.ny < 1 or self.nz < 1:
            raise ValueError("grid must be at least 2 x 1 x 1")
        if self.dim not in (2, 3):
            raise ValueError("dim must be 2 or 3")
        if self.dim == 2 and self.ny != 1:
            raise ValueError("2-D mode requires ny == 1")
        if not (0.5 < self.tau < 50.0):
            raise ValueError("BGK relaxation time tau must be in (0.5, 50)")
        if self.y_boundary not in ("wall", "periodic"):
            raise ValueError("y_boundary must be 'wall' or 'periodic'")
        if self.top_boundary not in ("slip", "wall"):
            raise ValueError("top_boundary must be 'slip' or 'wall'")

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self.nz, self.ny, self.nx)

    @property
    def viscosity(self) -> float:
        return (self.tau - 0.5) / 3.0


class LatticeBoltzmannSolver:
    """Reduced D3Q19/D2Q9 solver driving the coupled benchmark."""

    def __init__(self, spec: LatticeSpec, free_surface: Optional[FreeSurfaceConfig] = None):
        self.spec = spec
        self.fs_cfg = free_surface or FreeSurfaceConfig(enabled=False)
        if spec.dim == 3:
            self.c = D3Q19
            self.w = W3Q19
        else:
            self.c = D2Q9
            self.w = W2Q9
        self.opp = opposite_table(self.c)
        self.q = len(self.c)
        shp = spec.shape
        self.rho = np.full(shp, spec.rho0)
        self.u = np.zeros(shp + (3,))
        self.solid = np.zeros(shp, dtype=bool)
        self.wall_velocity = np.zeros(shp + (3,))
        self._periodic = (False, False, bool(spec.x_periodic))
        init = spec.initial_velocity if spec.initial_velocity is not None else spec.inlet_velocity
        self.u = np.zeros(shp + (3,))
        self.u[..., 0], self.u[..., 1], self.u[..., 2] = init
        self.f = self._equilibrium(self.rho, self.u)
        self.force_accumulator = np.zeros(3)
        # free-surface state
        if self.fs_cfg.enabled:
            self.state = np.full(shp, 2, dtype=np.int8)  # 2 liquid
            self.mass = np.ones(shp)
            # gas occupies the rows *above* the water column
            gas_rows = slice(max(1, self.fs_cfg.initial_depth), spec.nz)
            self.state[gas_rows, :, :] = 0
            self.mass[gas_rows, :, :] = 0.0
            self._mark_interface()
        else:
            self.state = np.full(shp, 2, dtype=np.int8)
            self.mass = np.ones(shp)

    # -- equilibrium / macroscopic -----------------------------------------
    def _equilibrium(self, rho: np.ndarray, u: np.ndarray) -> np.ndarray:
        cu = np.zeros(rho.shape + (self.q,))
        usq = (u * u).sum(axis=-1)
        for i in range(self.q):
            ci = self.c[i].astype(float)
            cu[..., i] = ci[0] * u[..., 0] + ci[1] * u[..., 1] + ci[2] * u[..., 2]
        feq = np.empty_like(cu)
        for i in range(self.q):
            ci = self.c[i].astype(float)
            feq[..., i] = self.w[i] * rho * (
                1.0 + cu[..., i] / CS2 + 0.5 * (cu[..., i] / CS2) ** 2 - 0.5 * usq / CS2
            )
        return feq

    def _macroscopic(self) -> None:
        f = self.f
        self.rho = f.sum(axis=-1)
        if self.solid.any():
            # solid cells carry no populations; keep the reference state there
            # so that boundary conditions never divide by zero
            self.rho = np.where(self.solid, self.spec.rho0, self.rho)
        for a in range(3):
            self.u[..., a] = sum(f[..., i] * self.c[i, a] for i in range(self.q)) / self.rho
        if self.solid.any():
            self.u[self.solid] = 0.0
        g = np.asarray(self.spec.gravity, dtype=float)
        if np.any(g != 0.0):
            self.u += 0.5 * g  # half force kick (Guo)

    # -- boundary helpers ---------------------------------------------------
    def _apply_gas_reconstruction(self) -> None:
        """Set missing populations at interface cells to atmospheric pressure."""
        if not self.fs_cfg.enabled:
            return
        # ghost gas cells carry no momentum: reset them every step so that they
        # cannot feed spurious mass back into the interface
        gas_cells = self.state == 0
        if gas_cells.any():
            rho_g = np.ones_like(self.rho)
            u_g = np.zeros_like(self.u)
            feq_g = self._equilibrium(rho_g, u_g)
            self.f = np.where(gas_cells[..., None], feq_g, self.f)
        iface = self.state == 1
        if not iface.any():
            return
        gas = self.state == 0
        feq = self._equilibrium(np.ones_like(self.rho), self.u)
        for i in range(1, self.q):
            nb_gas, valid = _shift(gas.astype(float), tuple(self.c[i]), self._periodic)
            mask = iface & (nb_gas > 0.5) & valid
            if not mask.any():
                continue
            o = self.opp[i]
            self.f[..., i] = np.where(mask, feq[..., i] + feq[..., o] - self.f[..., o], self.f[..., i])

    def _mark_interface(self) -> None:
        """Promote liquid cells that touch gas to interface cells."""
        gas = self.state == 0
        if not gas.any():
            return
        adjacent = np.zeros_like(gas)
        for i in range(1, self.q):
            nb, valid = _shift(gas.astype(float), tuple(self.c[i]), self._periodic)
            adjacent |= (nb > 0.5) & valid
        promote = adjacent & (self.state == 2)
        self.state[promote] = 1

    # -- one time step ------------------------------------------------------
    def step(self) -> np.ndarray:
        """Advance one lattice time step; returns the force on the body."""
        self._apply_gas_reconstruction()
        self._macroscopic()
        feq = self._equilibrium(self.rho, self.u)

        f_post = self.f + (feq - self.f) / self.spec.tau

        g = np.asarray(self.spec.gravity, dtype=float)
        if np.any(g != 0.0):
            for i in range(self.q):
                ci = self.c[i].astype(float)
                cu = ci[0] * self.u[..., 0] + ci[1] * self.u[..., 1] + ci[2] * self.u[..., 2]
                gu = ci[0] * g[0] + ci[1] * g[1] + ci[2] * g[2]
                # Guo forcing: (1 - 1/(2 tau)) * w_i * [ (c_i - u)/cs^2 + (c_i.u) c_i / cs^4 ] . F
                fi = 3.0 * (
                    (ci[0] - self.u[..., 0]) * g[0]
                    + (ci[1] - self.u[..., 1]) * g[1]
                    + (ci[2] - self.u[..., 2]) * g[2]
                ) + 9.0 * cu * (ci[0] * g[0] + ci[1] * g[1] + ci[2] * g[2])
                f_post[..., i] += self.w[i] * (1.0 - 0.5 / self.spec.tau) * fi * self.rho

        force = np.zeros(3)
        f_new = np.empty_like(f_post)
        for i in range(self.q):
            nb, valid = _shift(f_post[..., i].copy(), tuple(self.c[i]), self._periodic)
            src_solid = np.zeros(self.spec.shape, dtype=bool)
            if self.solid.any():
                s, v2 = _shift(self.solid.astype(float), tuple(self.c[i]), self._periodic)
                src_solid = (s > 0.5) & v2
            bounce = (~valid) | src_solid
            o = self.opp[i]
            f_new[..., i] = np.where(bounce, f_post[..., o], nb)
            if np.any(self.wall_velocity != 0.0):
                uw = self.wall_velocity
                ci = self.c[i].astype(float)
                cu = ci[0] * uw[..., 0] + ci[1] * uw[..., 1] + ci[2] * uw[..., 2]
                f_new[..., i] = np.where(
                    bounce & self.solid,
                    f_new[..., i] - 2.0 * self.w[i] * self.rho * 3.0 * cu,
                    f_new[..., i],
                )
            # momentum exchange counts only fluid/solid links, never the
            # domain-edge bounce-back (that is a boundary condition, not a body)
            solid_link = src_solid & valid & (~self.solid)
            if solid_link.any():
                ci = self.c[i].astype(float)
                # subtract the static (u=0) momentum flux so that a fluid at
                # rest exerts no spurious force on open surfaces; what remains
                # is the dynamic (resistance) force
                link_force = (
                    f_post[..., i] + f_new[..., i] - 2.0 * self.w[i] * self.spec.rho0
                ) * solid_link
                for a in range(3):
                    force[a] += float(link_force.sum() * ci[a])

        self.f = f_new
        if self.solid.any():
            self.f[self.solid] = 0.0  # solid populations are not part of the state
        self.force_accumulator = force

        # domain boundaries that are not solid cells
        self._apply_domain_boundaries()
        if self.fs_cfg.enabled:
            self._free_surface_update()
        return force

    def _apply_domain_boundaries(self) -> None:
        spec = self.spec
        # inlet: equilibrium with the prescribed velocity, density from neighbour
        u_in = np.zeros(self.spec.shape + (3,))
        u_in[..., 0] = spec.inlet_velocity[0]
        u_in[..., 1] = spec.inlet_velocity[1]
        u_in[..., 2] = spec.inlet_velocity[2]
        nb_rho = self.f[:, :, 1, :].sum(axis=-1)  # (nz, ny)
        rho_in = np.repeat(nb_rho[:, :, None], self.spec.nx, axis=2)
        if not spec.x_periodic:
            feq = self._equilibrium(rho_in, u_in)
            if self.fs_cfg.enabled:
                # prescribed water level at the inlet: refresh only wetted cells
                wet = self.state[:, :, 0] > 0
                f0 = np.where(wet[..., None], feq[:, :, 0, :], self.f[:, :, 0, :])
                self.f[:, :, 0, :] = f0
                iface_in = (self.state[:, :, 0] == 1) & (
                    np.arange(spec.nz)[:, None] < self.fs_cfg.initial_depth
                )
                self.mass[:, :, 0] = np.where(iface_in, 1.0, self.mass[:, :, 0])
            else:
                self.f[:, :, 0, :] = feq[:, :, 0, :]
            # outlet: zero-gradient copy
            self.f[:, :, -1, :] = self.f[:, :, -2, :]
        # top boundary: specular reflection for the rigid-lid approximation
        # (streaming already renders every other domain face no-slip)
        if not self.fs_cfg.enabled and spec.top_boundary == "slip":
            self._slip_plane(axis=0, index=-1)
        # y faces (dim = 3)
        if spec.dim == 3 and spec.y_boundary == "periodic":
            f_top = self.f[:, 0, :, :].copy()
            f_bot = self.f[:, -1, :, :].copy()
            self.f[:, 0, :, :] = f_bot
            self.f[:, -1, :, :] = f_top

    def _bounce_back_plane(self, axis: int, index: int) -> None:
        """Simple no-slip treatment: mirror the populations crossing the plane."""
        for i in range(1, self.q):
            ci = self.c[i]
            comp = ci[axis]
            # populations that would leave through the plane are reflected
            leaves = (comp < 0 and index == 0) or (comp > 0 and index == -1)
            if not leaves:
                continue
            o = self.opp[i]
            sl = [slice(None)] * 3
            sl[axis] = index
            self.f[tuple(sl) + (i,)] = self.f[tuple(sl) + (o,)]

    def _slip_plane(self, axis: int, index: int) -> None:
        """Free-slip (specular) reflection of the tangential population."""
        for i in range(1, self.q):
            ci = self.c[i]
            comp = ci[axis]
            leaves = (comp > 0 and index == -1) or (comp < 0 and index == 0)
            if not leaves:
                continue
            ci_mirror = ci.copy()
            ci_mirror[axis] = -ci_mirror[axis]
            j = int(np.where((self.c == ci_mirror).all(axis=1))[0][0])
            sl = [slice(None)] * 3
            sl[axis] = index
            self.f[tuple(sl) + (i,)] = self.f[tuple(sl) + (j,)]

    # -- free surface -------------------------------------------------------
    def _free_surface_update(self) -> None:
        cfg = self.fs_cfg
        state = self.state
        iface = state == 1
        if iface.sum() < cfg.min_interface:
            return
        phi = np.where(state == 2, 1.0, np.where(state == 1, np.clip(self.mass, 0.0, 1.0), 0.0))
        dm = np.zeros(self.spec.shape)
        for i in range(1, self.q):
            nb_phi, valid = _shift(phi, tuple(self.c[i]), self._periodic)
            nb_f, _ = _shift(self.f[..., i].copy(), tuple(self.c[i]), self._periodic)
            nb_state, _ = _shift(state.astype(float), tuple(self.c[i]), self._periodic)
            exchangeable = (nb_state > 0.5) & valid
            dm += np.where(iface & exchangeable, nb_phi * nb_f - phi * self.f[..., self.opp[i]], 0.0)

        self.mass = np.where(iface, self.mass + dm, self.mass)
        self.mass = np.where(state == 2, 1.0, self.mass)
        self.mass = np.where(state == 0, 0.0, self.mass)

        # interface -> liquid, with excess mass shared with interface neighbours
        excess = np.where(iface, self.mass - 1.0, 0.0)
        convert = iface & (self.mass > cfg.liquid_threshold)
        if convert.any():
            self.state[convert] = 2
            self.mass[convert] = 1.0
            self._redistribute(excess * convert)
        # interface -> gas
        deficit = np.where(iface, self.mass, 0.0)
        dry = iface & (self.mass < cfg.gas_threshold)
        if dry.any():
            self.state[dry] = 0
            self.mass[dry] = 0.0
            self._redistribute(-deficit * dry)
        self._mark_interface()

    def _redistribute(self, amount: np.ndarray) -> None:
        """Give the residual mass of a converted cell to interface neighbours."""
        if not np.any(amount):
            return
        total = np.zeros(self.spec.shape)
        count = np.zeros(self.spec.shape)
        for i in range(1, self.q):
            a, valid = _shift(amount, tuple(self.c[i]), self._periodic)
            total += np.where(valid, a, 0.0)
            count += valid.astype(float)
        target = (self.state == 1) & (count > 0)
        share = np.where(target, total / np.maximum(count, 1.0), 0.0)
        self.mass = np.where(target, self.mass + share, self.mass)

    # -- geometry / sampling ------------------------------------------------
    def set_solid(self, mask: np.ndarray, wall_velocity: Optional[np.ndarray] = None) -> None:
        if mask.shape != self.spec.shape:
            raise ValueError("solid mask shape mismatch")
        self.solid = np.asarray(mask, dtype=bool)
        self.wall_velocity = np.zeros(self.spec.shape + (3,)) if wall_velocity is None else wall_velocity
        if self.fs_cfg.enabled:
            self.state[self.solid] = 0
            self.mass[self.solid] = 0.0

    def sample_velocity(self, points: np.ndarray) -> np.ndarray:
        """Trilinear-free (nearest cell) velocity sampling at (n, 3) points."""
        pts = np.asarray(points, dtype=float)
        idx = np.rint(pts).astype(int)
        idx[:, 0] = np.clip(idx[:, 0], 0, self.spec.nx - 1)
        idx[:, 1] = np.clip(idx[:, 1], 0, self.spec.ny - 1)
        idx[:, 2] = np.clip(idx[:, 2], 0, self.spec.nz - 1)
        return self.u[idx[:, 2], idx[:, 1], idx[:, 0], :]

    def sample_pressure(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float)
        idx = np.rint(pts).astype(int)
        idx[:, 0] = np.clip(idx[:, 0], 0, self.spec.nx - 1)
        idx[:, 1] = np.clip(idx[:, 1], 0, self.spec.ny - 1)
        idx[:, 2] = np.clip(idx[:, 2], 0, self.spec.nz - 1)
        return CS2 * self.rho[idx[:, 2], idx[:, 1], idx[:, 0]]

    @property
    def total_mass(self) -> float:
        if self.fs_cfg.enabled:
            return float(self.mass.sum())
        return float(self.rho.sum())

    @property
    def kinetic_energy(self) -> float:
        fluid = self.state > 0
        return float(0.5 * (self.rho * np.einsum("z...,z...->...", self.u, self.u))[fluid].sum())