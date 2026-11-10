"""Staggered finite-volume incompressible Navier-Stokes solver (MAC + projection).

Why this solver exists in TensorPhysics: the conservation gate of campaign node
``P189-N01`` must compare an *independent* discretisation against the lattice
side.  A harness that compares a solver with a re-labelled copy of itself proves
nothing, so a second, structurally different discretisation is part of the
deliverable:

* :class:`StaggeredFVM` -- finite volume on a MAC (marker-and-cell) grid:
  ``u`` on vertical faces, ``v`` on horizontal faces, ``p`` at cell centres.
  Face velocities *are* mass fluxes, so discrete mass conservation holds per
  cell by construction and the interface flux of a sub-domain is read directly
  off the face that lies on the interface.  Advection is a blend of central and
  upwind face reconstruction (``kappa``), diffusion is a compact second-order
  Laplacian, and the pressure correction solves the discrete Neumann Poisson
  problem exactly (sparse LU of the constant 5-point matrix, mean divergence
  removed for compatibility, one pressure cell pinned).

Conventions (``h`` = cell size, ``rho = 1``, velocity unit ``U``, length unit
``L``)::

    u[i, j]  at (i h, (j + 1/2) h)        shape (nx + 1, ny)
    v[i, j]  at ((i + 1/2) h, j h)        shape (nx, ny + 1)
    p[i, j]  at ((i + 1/2) h, (j + 1/2) h)  shape (nx, ny)

``p`` is the dynamic pressure in units of ``rho U^2`` with a zero hydrostatic
reference, matching the dynamic-pressure gauge in which the lattice momentum
flux is reported (:mod:`tensorphysics.interface_conservation.lbm`).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = ["StaggeredFVM"]


class StaggeredFVM:
    """MAC finite-volume solver for 2D incompressible flow on a rectangle.

    Parameters
    ----------
    nx, ny:
        cell counts; ``lx/nx`` must equal ``ly/ny`` (square cells) so that the
        exchange with the lattice is one-to-one.
    re:
        Reynolds number based on ``l_ref = 1`` and ``u_ref = 1``.
    kappa:
        advection reconstruction blend (``0`` first-order upwind, ``1`` central).
    """

    def __init__(self, nx: int, ny: int, lx: float = 1.0, ly: float = 1.0,
                 re: float = 100.0, rho: float = 1.0, kappa: float = 0.8) -> None:
        if nx < 4 or ny < 4:
            raise ValueError("need at least 4x4 cells")
        hx, hy = lx / nx, ly / ny
        if abs(hx - hy) > 1e-12 * max(hx, hy):
            raise ValueError("square cells required: lx/nx must equal ly/ny")
        self.nx, self.ny = int(nx), int(ny)
        self.lx, self.ly = float(lx), float(ly)
        self.h = hx
        self.re = float(re)
        self.rho = float(rho)
        self.nu = float(lx) / float(re)
        self.kappa = float(kappa)
        self.u = np.zeros((nx + 1, ny))
        self.v = np.zeros((nx, ny + 1))
        self.p = np.zeros((nx, ny))
        self.lid = 0.0
        self.bc_u_west = np.zeros(ny)
        self.bc_u_east = np.zeros(ny)
        self.bc_v_south = np.zeros(nx)
        self.bc_v_north = np.zeros(nx)
        self._lu = None
        self.pin = (0, 0)          # pressure datum cell (any cell is equivalent)
        self.p_east = None         # optional Dirichlet pressure on the east column

    # -- boundary conditions ---------------------------------------------------
    def set_closed_cavity(self, lid: float = 1.0) -> None:
        """Still walls on three sides, moving lid (``+x``) on the north side."""
        self.lid = float(lid)
        self.bc_u_west = np.zeros(self.ny)
        self.bc_u_east = np.zeros(self.ny)
        self.bc_v_south = np.zeros(self.nx)
        self.bc_v_north = np.zeros(self.nx)

    def set_interface_pressure(self, values) -> None:
        """Impose a Dirichlet pressure (dynamic, ``rho U^2``) on the east column.

        Used by the conservative exchange mode: the lattice hands its interface
        pressure to the finite-volume sub-domain, which then solves the Poisson
        problem with that datum instead of a zero-gradient condition.  The
        matrix structure changes only on the first call, so the sparse
        factorisation is invalidated exactly once.
        """
        arr = np.asarray(values, dtype=float)
        if arr.shape != (self.ny,):
            raise ValueError(f"east pressure profile must have length {self.ny}")
        if self.p_east is None:
            self._lu = None
        self.p_east = arr.copy()

    def set_open_east(self, pressure=None) -> None:
        """Open east boundary: zero-gradient velocity + Dirichlet pressure.

        An incompressible sub-domain whose boundary normal velocity is prescribed
        everywhere has no divergence-free solution unless the prescribed profile
        integrates to zero over the whole boundary.  The textbook remedy -- and
        the one used by the conservative exchange mode -- is to let the boundary
        that couples the two solvers be an *outflow*: the normal velocity is
        extrapolated, the pressure is imposed, and the mass flux the sub-domain
        actually delivers is the quantity that gets transferred.
        """
        self.east_open = True
        if pressure is not None:
            self.set_interface_pressure(pressure)

    def apply_bcs(self) -> None:
        self.u[0, :] = self.bc_u_west
        if not getattr(self, "east_open", False):
            self.u[-1, :] = self.bc_u_east
        self.v[:, 0] = self.bc_v_south
        self.v[:, -1] = self.bc_v_north

    # -- discretisation --------------------------------------------------------
    def _blend(self, face, lo, hi):
        return self.kappa * face + (1.0 - self.kappa) * np.where(face > 0.0, lo, hi)

    def _u_ghost_y(self):
        """``u`` extended to the wall rows ``j = -1`` and ``j = ny``."""
        g = np.empty((self.nx + 1, self.ny + 2))
        g[:, 1:-1] = self.u
        g[:, 0] = -self.u[:, 0]            # u = 0 on the south wall
        g[:, -1] = 2.0 * self.lid - self.u[:, -1]   # lid on the north wall
        return g

    def _u_at_vnode(self):
        """``u`` on the ``v``-cell x-faces, shape ``(nx + 1, ny + 1)``.

        ``u`` lives at ``y = (j + 1/2) h`` and the ``v`` node at ``y = j h``, so
        the x-face values of a ``v`` cell are the ``u`` values averaged in ``y``
        only -- averaging in ``x`` as well would smear the face velocity that the
        conservative difference needs.
        """
        g = self._u_ghost_y()
        return 0.5 * (g[:, :-1] + g[:, 1:])

    def step(self, dt: float) -> None:
        """Advance one explicit step of size ``dt`` and project."""
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.apply_bcs()
        h, nu = self.h, self.nu
        u, v = self.u, self.v

        # ---------------- u momentum (inner x-faces, all j) -------------------
        ue = 0.5 * (u[1:-1, :] + u[2:, :])
        uw = 0.5 * (u[:-2, :] + u[1:-1, :])
        ue_a = self._blend(ue, u[1:-1, :], u[2:, :])
        uw_a = self._blend(uw, u[1:-1, :], u[:-2, :])
        i = np.arange(self.nx + 1)
        im = np.clip(i - 1, 0, self.nx - 1)
        ic = np.clip(i, 0, self.nx - 1)
        v_face = 0.5 * (v[im, :] + v[ic, :])                # v at u-node, y-faces
        vtop = v_face[1:-1, 1:]
        vbot = v_face[1:-1, :-1]
        gg = self._u_ghost_y()                              # rows j = -1..ny
        # u control volume of node (i, j) spans y in [j h, (j+1) h]
        ubot_c, ubot_u = gg[1:-1, 0:self.ny], gg[1:-1, 1:self.ny + 1]
        utop_c, utop_u = gg[1:-1, 1:self.ny + 1], gg[1:-1, 2:self.ny + 2]
        ubot_avg = 0.5 * (ubot_c + ubot_u)
        utop_avg = 0.5 * (utop_c + utop_u)
        ubot_a = self._blend(vbot, ubot_c, ubot_u)
        utop_a = self._blend(vtop, utop_c, utop_u)
        div_u = ((ue * ue_a - uw * uw_a) + (vtop * utop_a - vbot * ubot_a)) / h
        d2x = (u[2:, :] - 2.0 * u[1:-1, :] + u[:-2, :]) / h ** 2
        d2y = (gg[:, 2:] - 2.0 * gg[:, 1:-1] + gg[:, :-2]) / h ** 2
        d2y = d2y[1:-1, :]
        u_new = u.copy()
        u_new[1:-1, :] += dt * (-div_u + nu * (d2x + d2y))

        # ---------------- v momentum (inner y-faces, all i) -------------------
        vn = 0.5 * (v[:, 1:-1] + v[:, 2:])
        vs = 0.5 * (v[:, :-2] + v[:, 1:-1])
        vn_a = self._blend(vn, v[:, 1:-1], v[:, 2:])
        vs_a = self._blend(vs, v[:, :-2], v[:, 1:-1])
        uvn = self._u_at_vnode()                    # u on v-cell x-faces
        uR = uvn[1:, 1:-1]
        uL = uvn[:-1, 1:-1]
        vpad = np.empty((self.nx + 2, self.ny + 1))
        vpad[1:-1, :] = v
        vpad[0, :] = v[0, :]
        vpad[-1, :] = v[-1, :]
        v_c = vpad[1:-1, 1:-1]
        v_r = vpad[2:, 1:-1]
        v_l = vpad[:-2, 1:-1]
        uR_a = self._blend(uR, v_c, v_r)
        uL_a = self._blend(uL, v_l, v_c)
        div_v = ((uR * uR_a - uL * uL_a) + (vn * vn_a - vs * vs_a)) / h
        d2x_v = (vpad[2:, 1:-1] - 2.0 * v_c + vpad[:-2, 1:-1]) / h ** 2
        d2y_v = (v[:, 2:] - 2.0 * v[:, 1:-1] + v[:, :-2]) / h ** 2
        v_new = v.copy()
        v_new[:, 1:-1] += dt * (-div_v + nu * (d2x_v + d2y_v))

        self.u, self.v = u_new, v_new
        self.apply_bcs()
        self.project(dt)

    # -- pressure projection ---------------------------------------------------
    def _poisson(self):
        if self._lu is None:
            nx, ny, h = self.nx, self.ny, self.h
            n = nx * ny
            rows, cols, vals = [], [], []
            for j in range(ny):
                for i in range(nx):
                    k = i + j * nx
                    diag = 0.0
                    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ii, jj = i + di, j + dj
                        if 0 <= ii < nx and 0 <= jj < ny:
                            rows.append(k); cols.append(ii + jj * nx)
                            vals.append(1.0 / h ** 2)
                            diag -= 1.0 / h ** 2
                    rows.append(k); cols.append(k); vals.append(diag)
            A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n)).tolil()
            k0 = self.pin[0] + self.pin[1] * nx
            A[k0, :] = 0.0
            A[k0, k0] = 1.0                     # pressure datum
            if self.p_east is not None:
                for j in range(ny):
                    kk = (nx - 1) + j * nx      # Dirichlet column on the interface
                    A[kk, :] = 0.0
                    A[kk, kk] = 1.0
            self._lu = spla.splu(A.tocsc())
        return self._lu

    def divergence(self) -> np.ndarray:
        return ((self.u[1:, :] - self.u[:-1, :]) / self.h
                + (self.v[:, 1:] - self.v[:, :-1]) / self.h)

    def project(self, dt: float) -> None:
        div = self.divergence()
        # solve  lap(phi) = div(u*) / dt   then  u <- u* - dt grad(phi),  p <- p + phi
        rhs = ((div - div.mean()) / dt).ravel()
        if self.p_east is not None:
            rhs[(self.nx - 1)::self.nx] = self.p_east
        phi = self._poisson().solve(rhs).reshape(self.nx, self.ny)
        self.p += phi
        self.u[1:-1, :] -= dt * (phi[1:, :] - phi[:-1, :]) / self.h
        self.v[:, 1:-1] -= dt * (phi[:, 1:] - phi[:, :-1]) / self.h
        if getattr(self, "east_open", False):
            # mass-consistent outflow: the sub-domain cannot store mass, so the
            # column integral of the outflow is projected onto the value that the
            # (solid) remainder of the boundary allows, i.e. zero net flux for a
            # closed cavity.  Without this the Poisson problem is incompatible and
            # the residual divergence equals mean(div) for ever.
            self.u[-1, :] = self.u[-1, :] - self.u[-1, :].mean()
        self.apply_bcs()

    # -- diagnostics -----------------------------------------------------------
    def cell_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        return 0.5 * (self.u[:-1, :] + self.u[1:, :]), 0.5 * (self.v[:, :-1] + self.v[:, 1:])

    def velocity_at(self, x: float, y: float) -> tuple[float, float]:
        uc, vc = self.cell_velocity()
        h = self.h
        fx = np.clip(x / h - 0.5, 0.0, self.nx - 1 - 1e-12)
        fy = np.clip(y / h - 0.5, 0.0, self.ny - 1 - 1e-12)
        i0, j0 = int(fx), int(fy)
        tx, ty = fx - i0, fy - j0
        i1, j1 = min(i0 + 1, self.nx - 1), min(j0 + 1, self.ny - 1)

        def bil(arr):
            return ((1 - tx) * (1 - ty) * arr[i0, j0] + tx * (1 - ty) * arr[i1, j0]
                    + (1 - tx) * ty * arr[i0, j1] + tx * ty * arr[i1, j1])

        return float(bil(uc)), float(bil(vc))

    def face_flux(self, side: str) -> dict[str, np.ndarray]:
        """Pointwise conserved fluxes through a boundary face (per unit length).

        Returns ``mass`` (``u_n``), ``momentum`` (``u_n^2 + p - 2 nu du_n/dn``,
        the normal component of the momentum flux in the dynamic-pressure gauge)
        and ``energy`` (``(p + |u|^2/2) u_n - u_n tau_nn - u_t tau_nt``) plus the
        primitives used to build them.  Multiply by ``h`` per cell to integrate.
        """
        h, nu = self.h, self.nu
        if side in ("east", "west"):
            icol = self.nx - 1 if side == "east" else 0
            sign = 1.0 if side == "east" else -1.0
            un = sign * self.u[-1, :] if side == "east" else sign * self.u[0, :]
            ut = 0.5 * (self.v[icol, :-1] + self.v[icol, 1:])
            p_face = self.p[icol, :]
            if side == "east":
                du = (3.0 * self.u[-1, :] - 4.0 * self.u[-2, :] + self.u[-3, :]) / (2 * h)
                if self.nx >= 4:
                    dut = (3.0 * ut - 4.0 * 0.5 * (self.v[self.nx - 2, :-1] + self.v[self.nx - 2, 1:])
                           + 0.5 * (self.v[self.nx - 3, :-1] + self.v[self.nx - 3, 1:])) / (2 * h)
                else:
                    dut = (ut - 0.5 * (self.v[self.nx - 2, :-1] + self.v[self.nx - 2, 1:])) / h
            else:
                du = (3.0 * self.u[0, :] - 4.0 * self.u[1, :] + self.u[2, :]) / (2 * h)
                if self.nx >= 4:
                    dut = (3.0 * ut - 4.0 * 0.5 * (self.v[1, :-1] + self.v[1, 1:])
                           + 0.5 * (self.v[2, :-1] + self.v[2, 1:])) / (2 * h)
                else:
                    dut = (ut - 0.5 * (self.v[1, :-1] + self.v[1, 1:])) / h
        elif side in ("north", "south"):
            jcol = self.ny - 1 if side == "north" else 0
            sign = 1.0 if side == "north" else -1.0
            un = sign * self.v[:, -1] if side == "north" else sign * self.v[:, 0]
            ut = 0.5 * (self.u[:-1, jcol] + self.u[1:, jcol])
            p_face = self.p[:, jcol]
            if side == "north":
                du = (3.0 * self.v[:, -1] - 4.0 * self.v[:, -2] + self.v[:, -3]) / (2 * h)
                dut = (3.0 * ut - 4.0 * 0.5 * (self.u[:-1, self.ny - 2] + self.u[1:, self.ny - 2])
                       + 0.5 * (self.u[:-1, self.ny - 3] + self.u[1:, self.ny - 3])) / (2 * h)
            else:
                du = (3.0 * self.v[:, 0] - 4.0 * self.v[:, 1] + self.v[:, 2]) / (2 * h)
                dut = (3.0 * ut - 4.0 * 0.5 * (self.u[:-1, 1] + self.u[1:, 1])
                       + 0.5 * (self.u[:-1, 2] + self.u[1:, 2])) / (2 * h)
        else:
            raise ValueError(f"unknown side {side!r}")
        tau_nn = 2.0 * nu * du
        tau_nt = nu * dut
        kinetic = 0.5 * self.rho * (un * un + ut * ut)
        energy = (p_face + kinetic) * un - un * tau_nn - ut * tau_nt
        return {"mass": self.rho * un, "momentum": self.rho * un * un + p_face - tau_nn,
                "energy": energy, "u_n": un, "u_t": ut, "p": p_face,
                "tau_nn": tau_nn}

    def max_divergence(self) -> float:
        return float(np.abs(self.divergence()).max())