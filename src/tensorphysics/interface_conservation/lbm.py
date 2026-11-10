"""D2Q9 lattice-Boltzmann solver with flux diagnostics for coupling gates.

The class is deliberately small (one lattice, BGK collision, straight
axis-aligned boundaries) but it exposes the two objects the conservation gate
needs and that most "LBM wrapper" codes hide:

* :meth:`D2Q9.interface_flux` -- the mass / normal-momentum / energy flux
  carried by the populations across a boundary column (lattice units).  These
  are measurements *of the distribution function*, not of the velocity field,
  so a boundary closure that fails to conserve shows up here.
* :meth:`D2Q9.bounce_back_exchange` -- the momentum exchanged with internal
  solids (particles) accumulated per link during streaming.  The exchange is the
  momentum carried into the wall minus the momentum carried back into the fluid,
  which is exactly what an immersed-body / momentum-exchange force model must
  reproduce, measured instead of formula-derived.

Boundary closures
-----------------
``velocity`` (Zou & He 1997) prescribes both velocity components; the
x-moment constraint is enforced by construction, so ``sum_i f_i c_ix = rho u_x``
holds to round-off.  That is what makes a mass-flux-consistent inlet possible
without a special closure: the caller derives ``u_x`` from the mass flux it
wants to transfer and from the local density.

``pressure`` (Zou & He 1997) prescribes density instead of the normal velocity.

``wall`` is half-way bounce-back, either against solid nodes outside the domain
(channel walls, closed cavity) or against the moving surface of an internal
particle, in which case Ladd's correction ``2 w_i rho (c_i . u_w) / c_s^2`` is
added with ``u_w`` the surface velocity at the link centre.

Structural limitation (documented, not papered over)
----------------------------------------------------
A straight boundary owns three unknown incoming populations -- three scalar
degrees of freedom.  They can carry mass flux, tangential momentum and one
first-moment combination, but *not* the second moment (normal stress)
independently: the three moment equations are rank deficient with respect to
``(mass flux, u_t, Pi_nn)``.  A pointwise normal-momentum-flux-imposing D2Q9
closure therefore does not exist; :meth:`D2Q9.interface_flux` reports what a
closure actually delivers, and the gate distinguishes the *integral*
conservation statement (which a conservative fix-up can enforce exactly) from
the *local* consistency statement (which is a discretisation error).
"""
from __future__ import annotations

import numpy as np

__all__ = ["D2Q9", "EX", "EY", "W", "OPP", "CS2"]

W = np.array([4.0 / 9.0] + [1.0 / 9.0] * 4 + [1.0 / 36.0] * 4)
EX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
EY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
OPP = np.array([0, 3, 4, 1, 2, 6, 5, 8, 7])
CS2 = 1.0 / 3.0

#: incoming directions per straight side (pointing into the domain)
_INCOMING = {
    "west": (1, 5, 8),
    "east": (3, 6, 7),
    "south": (2, 5, 6),
    "north": (4, 7, 8),
}


class D2Q9:
    """D2Q9 BGK lattice Boltzmann solver on an ``ny x nx`` lattice.

    Parameters
    ----------
    nx, ny:
        lattice size (nodes, all of them live in the domain).
    tau:
        BGK relaxation time; the lattice viscosity is ``(tau - 0.5) / 3``.
    solid:
        optional boolean mask ``(ny, nx)`` of solid *nodes* (particles).  Links
        to those nodes are closed with bounce-back.
    """

    def __init__(self, nx: int, ny: int, tau: float = 0.6,
                 solid: np.ndarray | None = None) -> None:
        if nx < 3 or ny < 3:
            raise ValueError("lattice must be at least 3x3")
        if tau <= 0.5:
            raise ValueError("tau must exceed 0.5 (positive viscosity)")
        self.nx, self.ny = int(nx), int(ny)
        self.tau = float(tau)
        self.omega = 1.0 / self.tau
        self.solid = (np.zeros((self.ny, self.nx), dtype=bool)
                      if solid is None else np.asarray(solid, dtype=bool).copy())
        self.f = np.repeat(W[:, None, None], self.ny, axis=1).repeat(self.nx, axis=2)
        # per-node solid velocity used by the Ladd correction (particles)
        self.wall_u = np.zeros((2, self.ny, self.nx))
        # flux accounting of the last step (momentum handed to solids)
        self.last_exchange_momentum = np.zeros(2)
        self.last_exchange_count = 0
        self._map = None

    # -- state helpers ---------------------------------------------------------
    @property
    def rho(self) -> np.ndarray:
        return self.f.sum(axis=0)

    def velocity(self) -> np.ndarray:
        r = self.rho
        ux = np.tensordot(EX, self.f, axes=(0, 0)) / r
        uy = np.tensordot(EY, self.f, axes=(0, 0)) / r
        return np.stack([ux, uy])

    def set_equilibrium(self, u=(0.0, 0.0), rho: float = 1.0) -> None:
        ux, uy = float(u[0]), float(u[1])
        cu = EX[:, None, None] * ux + EY[:, None, None] * uy
        usq = 3.0 * (ux * ux + uy * uy) / 2.0
        self.f = (rho * W[:, None, None] * (1.0 + 3.0 * cu + 4.5 * cu * cu - usq)
                  * np.ones((self.ny, self.nx)))

    def momentum(self) -> np.ndarray:
        """Total fluid momentum ``sum f_i c_i`` over fluid nodes."""
        mask = ~self.solid
        p = np.array([0.0, 0.0])
        for i in range(9):
            fi = self.f[i][mask].sum()
            p[0] += fi * EX[i]
            p[1] += fi * EY[i]
        return p

    # -- collision + streaming -------------------------------------------------
    def step(self, bcs: dict[str, tuple] | None = None,
             track_exchange: bool = False) -> None:
        """One collide-and-stream step.

        ``bcs`` maps a side (``west``/``east``/``south``/``north``) to a tuple
        ``("velocity", ux, uy)``, ``("pressure", rho)`` or ``("wall", ux, uy)``
        where the velocity arguments may be scalars or 1-D arrays of the
        boundary line length.  ``("wall", ...)`` uses half-way bounce-back with
        a (possibly moving) wall velocity.
        """
        bcs = bcs or {}
        rho = self.rho
        ux = np.tensordot(EX, self.f, axes=(0, 0)) / rho
        uy = np.tensordot(EY, self.f, axes=(0, 0)) / rho
        # BGK collision
        cu = EX[:, None, None] * ux[None] + EY[:, None, None] * uy[None]
        usq = 1.5 * (ux * ux + uy * uy)
        feq = rho[None] * W[:, None, None] * (1.0 + 3.0 * cu + 4.5 * cu * cu - usq[None])
        post = self.f - self.omega * (self.f - feq)

        new = np.empty_like(self.f)
        exchange = np.zeros(2)
        wlinks = 0
        maps = self._streaming_maps()
        # links whose bounce-back surface is a *wall* side: only those reflect;
        # on a velocity/pressure side nothing arrives from outside, so the local
        # post-collision value is kept and the closure sets the incoming ones
        wall_out = [np.zeros((self.ny, self.nx), bool) for _ in range(9)]
        for side, spec in (bcs or {}).items():
            if spec[0] != "wall":
                continue
            sm = self._side_mask(side)
            for i in range(9):
                wall_out[i] |= maps[i][0] & sm
        # pull streaming with bounce-back on closed links (outside the domain or
        # solid nodes); Ladd's moving-wall correction is added on solid links
        for i in range(9):
            outside, fluid_idx, blocked, by, bx = maps[i]
            back = post[OPP[i]]
            val = np.where(wall_out[i], back, np.where(maps[i][0], post[i], back))
            if len(fluid_idx[0]):
                val.ravel()[fluid_idx[0]] = post[i].ravel()[fluid_idx[1]]
            if blocked.any():
                wu = self.wall_u[:, by, bx]
                ciw = EX[i] * wu[0] + EY[i] * wu[1]
                corr = 2.0 * W[i] * rho[by, bx] * ciw / CS2
                val[blocked] = post[OPP[i]][blocked] + corr
                if track_exchange:
                    exchange[0] += float(np.sum((post[i][by, bx] - val[blocked]) * EX[i]))
                    exchange[1] += float(np.sum((post[i][by, bx] - val[blocked]) * EY[i]))
                    wlinks += by.size
            new[i] = val
        self.f = new
        if bcs:
            self._apply_moving_walls(bcs, rho)
        self.last_exchange_momentum = exchange
        self.last_exchange_count = wlinks
        if bcs:
            self._apply_bcs(bcs)

    def _apply_moving_walls(self, bcs: dict[str, tuple], rho: np.ndarray) -> None:
        """Ladd correction for a moving wall on a *domain edge*.

        Applied only to the links whose bounce-back surface is the moving wall:
        the source node of the link must lie outside the domain across that wall
        *and* inside along the wall, otherwise a diagonal link at the corner (which
        belongs to the perpendicular wall) would receive the tangential momentum
        of the moving wall.  That shortcut is what makes naive implementations
        inject momentum in the wrong direction and blow up.
        """
        yy = np.arange(self.ny)[:, None]
        xx = np.arange(self.nx)[None, :]
        for side, spec in bcs.items():
            if spec[0] != "wall":
                continue
            ux = float(spec[1]) if len(spec) > 1 else 0.0
            uy = float(spec[2]) if len(spec) > 2 else 0.0
            if ux == 0.0 and uy == 0.0:
                continue
            for i in range(9):
                ciw = EX[i] * ux + EY[i] * uy
                if ciw == 0.0:
                    continue
                sy = yy - EY[i]
                sx = xx - EX[i]
                inside_y = (sy >= 0) & (sy < self.ny)
                inside_x = (sx >= 0) & (sx < self.nx)
                if side in ("north", "south"):
                    mask = (~inside_y) & inside_x
                else:
                    mask = (~inside_x) & inside_y
                mask = np.broadcast_to(mask, (self.ny, self.nx)).copy()
                if side == "north":
                    mask[:-1, :] = False
                elif side == "south":
                    mask[1:, :] = False
                elif side == "west":
                    mask[:, 1:] = False
                else:
                    mask[:, :-1] = False
                if not mask.any():
                    continue
                self.f[i][mask] += 2.0 * W[i] * rho[mask] * ciw / CS2

    def _side_mask(self, side: str) -> np.ndarray:
        m = np.zeros((self.ny, self.nx), bool)
        if side == "north":
            m[-1, :] = True
        elif side == "south":
            m[0, :] = True
        elif side == "west":
            m[:, 0] = True
        elif side == "east":
            m[:, -1] = True
        else:
            raise ValueError(f"unknown side {side!r}")
        return m

    def _streaming_maps(self):
        """Static streaming bookkeeping (built once, reused every step)."""
        if self._map is None:
            yy = np.arange(self.ny)[:, None]
            xx = np.arange(self.nx)[None, :]
            maps = []
            for i in range(9):
                sy = yy - EY[i]
                sx = xx - EX[i]
                inside = (sy >= 0) & (sy < self.ny) & (sx >= 0) & (sx < self.nx)
                syc = np.clip(sy, 0, self.ny - 1)
                sxc = np.clip(sx, 0, self.nx - 1)
                solid_src = self.solid[syc, sxc]
                outside = ~inside
                fluid = inside & ~solid_src
                blocked = inside & solid_src
                dest_grid = np.arange(self.ny)[:, None] * self.nx + np.arange(self.nx)[None, :]
                fidx = (dest_grid[fluid], (syc * self.nx + sxc)[fluid])
                by, bx = np.nonzero(blocked) if blocked.any() else (np.zeros(0, int), np.zeros(0, int))
                maps.append((outside, fidx, blocked, by, bx))
            self._map = maps
        return self._map

    # -- Zou & He closures -----------------------------------------------------
    def _apply_bcs(self, bcs: dict[str, tuple]) -> None:
        """Zou & He (1997) closures for the four straight boundaries.

        Formulas are the standard ones; they are re-derived from the moment
        constraints in the report, because a sign slip in the tangential
        correction is invisible in the velocity field but destroys the
        momentum-flux diagnostic this package is built on.
        """
        f = self.f
        for side, spec in bcs.items():
            kind = spec[0]
            if kind == "wall":
                continue                      # handled by bounce-back
            if side == "west":
                f0, f2, f4 = f[0, :, 0], f[2, :, 0], f[4, :, 0]
                f3, f6, f7 = f[3, :, 0], f[6, :, 0], f[7, :, 0]
                if kind == "velocity":
                    uxb, uyb = _line(spec[1], self.ny), _line(spec[2], self.ny)
                    rho_b = (f0 + f2 + f4 + 2.0 * (f3 + f6 + f7)) / (1.0 - uxb)
                elif kind == "pressure":
                    rho_b = np.full(self.ny, float(spec[1]))
                    uyb = _line(spec[2], self.ny) if len(spec) > 2 else np.zeros(self.ny)
                    uxb = -1.0 + (f0 + f2 + f4 + 2.0 * (f3 + f6 + f7)) / rho_b
                else:
                    raise ValueError(f"unknown closure {kind!r}")
                f[1, :, 0] = f3 + (2.0 / 3.0) * rho_b * uxb
                f[5, :, 0] = (f7 - 0.5 * (f2 - f4) + (1.0 / 6.0) * rho_b * uxb
                              + 0.5 * rho_b * uyb)
                f[8, :, 0] = (f6 + 0.5 * (f2 - f4) + (1.0 / 6.0) * rho_b * uxb
                              - 0.5 * rho_b * uyb)
            elif side == "east":
                f0, f2, f4 = f[0, :, -1], f[2, :, -1], f[4, :, -1]
                f1, f5, f8 = f[1, :, -1], f[5, :, -1], f[8, :, -1]
                if kind == "velocity":
                    uxb, uyb = _line(spec[1], self.ny), _line(spec[2], self.ny)
                    rho_b = (f0 + f2 + f4 + 2.0 * (f1 + f5 + f8)) / (1.0 + uxb)
                elif kind == "pressure":
                    rho_b = np.full(self.ny, float(spec[1]))
                    uyb = _line(spec[2], self.ny) if len(spec) > 2 else np.zeros(self.ny)
                    uxb = -1.0 + (f0 + f2 + f4 + 2.0 * (f1 + f5 + f8)) / rho_b
                else:
                    raise ValueError(f"unknown closure {kind!r}")
                f[3, :, -1] = f1 - (2.0 / 3.0) * rho_b * uxb
                f[6, :, -1] = (f8 - 0.5 * (f2 - f4) + 0.5 * rho_b * uyb
                               - (1.0 / 6.0) * rho_b * uxb)
                f[7, :, -1] = (f5 + 0.5 * (f2 - f4) - 0.5 * rho_b * uyb
                               - (1.0 / 6.0) * rho_b * uxb)
            elif side == "south":
                f0, f1, f3 = f[0, 0, :], f[1, 0, :], f[3, 0, :]
                f4, f7, f8 = f[4, 0, :], f[7, 0, :], f[8, 0, :]
                if kind == "velocity":
                    uxb, uyb = _line(spec[1], self.nx), _line(spec[2], self.nx)
                    rho_b = (f0 + f1 + f3 + 2.0 * (f4 + f7 + f8)) / (1.0 - uyb)
                elif kind == "pressure":
                    rho_b = np.full(self.nx, float(spec[1]))
                    uxb = _line(spec[2], self.nx) if len(spec) > 2 else np.zeros(self.nx)
                    uyb = -1.0 + (f0 + f1 + f3 + 2.0 * (f4 + f7 + f8)) / rho_b
                else:
                    raise ValueError(f"unknown closure {kind!r}")
                f[2, 0, :] = f4 + (2.0 / 3.0) * rho_b * uyb
                f[5, 0, :] = (f7 - 0.5 * (f1 - f3) + (1.0 / 6.0) * rho_b * uyb
                              + 0.5 * rho_b * uxb)
                f[6, 0, :] = (f8 + 0.5 * (f1 - f3) + (1.0 / 6.0) * rho_b * uyb
                              - 0.5 * rho_b * uxb)
            elif side == "north":
                f0, f1, f3 = f[0, -1, :], f[1, -1, :], f[3, -1, :]
                f2, f5, f6 = f[2, -1, :], f[5, -1, :], f[6, -1, :]
                if kind == "velocity":
                    uxb, uyb = _line(spec[1], self.nx), _line(spec[2], self.nx)
                    rho_b = (f0 + f1 + f3 + 2.0 * (f2 + f5 + f6)) / (1.0 + uyb)
                elif kind == "pressure":
                    rho_b = np.full(self.nx, float(spec[1]))
                    uxb = _line(spec[2], self.nx) if len(spec) > 2 else np.zeros(self.nx)
                    uyb = -1.0 + (f0 + f1 + f3 + 2.0 * (f2 + f5 + f6)) / rho_b
                else:
                    raise ValueError(f"unknown closure {kind!r}")
                f[4, -1, :] = f2 - (2.0 / 3.0) * rho_b * uyb
                f[7, -1, :] = (f5 + 0.5 * (f1 - f3) - 0.5 * rho_b * uxb
                               - (1.0 / 6.0) * rho_b * uyb)
                f[8, -1, :] = (f6 - 0.5 * (f1 - f3) + 0.5 * rho_b * uxb
                               - (1.0 / 6.0) * rho_b * uyb)
            else:
                raise ValueError(f"unknown side {side!r}")

    # -- diagnostics -----------------------------------------------------------
    def boundary_line(self, side: str) -> slice:
        return {"west": (slice(None), 0), "east": (slice(None), -1),
                "south": (0, slice(None)), "north": (-1, slice(None))}[side]

    def interface_flux(self, side: str) -> dict[str, np.ndarray]:
        """Per-node conserved fluxes across a boundary column (lattice units).

        Returns a dict with ``mass`` (``sum f_i c_in`` where ``n`` is the outward
        normal), ``momentum`` (the ``n``-component of the momentum flux,
        ``sum f_i c_in c_i . n`` measured *relative* to the rest-state
        contribution ``rho c_s^2``, i.e. in the dynamic-pressure gauge) and
        ``energy`` (``sum f_i c_in |c_i|^2 / 2``; the discrete kinetic-energy
        flux, equal to the mechanical energy flux up to ``O(Ma^3)``).
        """
        idx = self.boundary_line(side)
        f = self.f[(slice(None),) + idx]
        rho = f.sum(axis=0)
        n = {"west": (-1, 0), "east": (1, 0), "south": (0, -1),
             "north": (0, 1)}[side]
        cn = EX * n[0] + EY * n[1]
        mass = np.tensordot(cn, f, axes=(0, 0))
        mom = np.tensordot(cn * cn, f, axes=(0, 0)) - rho * CS2
        csq = EX ** 2 + EY ** 2
        energy = 0.5 * np.tensordot(cn * csq, f, axes=(0, 0))
        return {"mass": mass, "momentum": mom, "energy": energy, "rho": rho}

    def boundary_velocity(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        """Velocity reconstructed on a boundary column (lattice units)."""
        idx = self.boundary_line(side)
        f = self.f[(slice(None),) + idx]
        rho = f.sum(axis=0)
        ux = np.tensordot(EX, f, axes=(0, 0)) / rho
        uy = np.tensordot(EY, f, axes=(0, 0)) / rho
        return ux, uy


def _line(value, n: int) -> np.ndarray:
    if np.isscalar(value):
        return np.full(n, float(value))
    arr = np.asarray(value, dtype=float)
    if arr.shape != (n,):
        raise ValueError(f"boundary profile must have length {n}, got {arr.shape}")
    return arr