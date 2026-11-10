"""FEM targets for the exchange verification (all accept arbitrary nodal loads).

Three minimal structural targets are provided, because the point of a
conservative *exchange* is that the structural code must be able to receive an
arbitrary load vector and nothing else:

* :class:`PlaneStressQ4` -- 2-D plane-stress membrane (in-plane shear traction,
  e.g. Couette wall shear acting on the plate surface).
* :class:`PlateStripBending` -- 1-D Kirchhoff plate strip in cylindrical bending
  with plate rigidity ``D = E t^3 / (12 (1 - nu^2))`` (exact closed-form
  references exist for uniform and sinusoidal pressure).
* :class:`PlateBFS` -- 2-D Kirchhoff plate with the 16-DOF Bogner-Fox-Schmit
  bicubic Hermite element (4 DOF/node: ``w, w_x, w_y, w_xy``), simply supported
  by suppressing ``w`` on the boundary only, which is the hard simple support of
  the Navier solution family.

All solvers are linear elastic, so superposition is available for multi-contact
checks; each exposes ``strain_energy``/``external_work`` for the energy audit.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = ["PlaneStressQ4", "PlateStripBending", "PlateBFS"]


# ---------------------------------------------------------------------------
# 2-D plane stress membrane
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PlaneStressQ4:
    """Uniform 4-node bilinear plane-stress membrane on ``[0,L] x [0,W]``."""

    L: float
    W: float
    t: float
    E: float
    nu: float
    nx: int = 20
    ny: int = 4

    @property
    def n_nodes(self) -> int:
        return (self.nx + 1) * (self.ny + 1)

    def coords(self) -> np.ndarray:
        xs = np.linspace(0.0, self.L, self.nx + 1)
        ys = np.linspace(0.0, self.W, self.ny + 1)
        xx, yy = np.meshgrid(xs, ys, indexing="xy")
        return np.column_stack([xx.ravel(), yy.ravel()])

    def node_index(self, i: int, j: int) -> int:
        return j * (self.nx + 1) + i

    def _D(self) -> np.ndarray:
        c = self.E / (1.0 - self.nu ** 2)
        return c * np.array([[1.0, self.nu, 0.0], [self.nu, 1.0, 0.0],
                             [0.0, 0.0, 0.5 * (1.0 - self.nu)]])

    def element_stiffness(self) -> np.ndarray:
        D = self._D()
        hx = self.L / self.nx
        hy = self.W / self.ny
        K = np.zeros((8, 8))
        g = 1.0 / np.sqrt(3.0)
        for s in (-g, g):
            for r in (-g, g):
                dN = 0.25 * np.array([
                    [-(1.0 - s), -(1.0 - r)],
                    [(1.0 - s), -(1.0 + r)],
                    [(1.0 + s), (1.0 + r)],
                    [-(1.0 + s), (1.0 - r)],
                ])
                dN[:, 0] *= 2.0 / hx
                dN[:, 1] *= 2.0 / hy
                Bm = np.zeros((3, 8))
                for a in range(4):
                    Bm[0, 2 * a] = dN[a, 0]
                    Bm[1, 2 * a + 1] = dN[a, 1]
                    Bm[2, 2 * a] = dN[a, 1]
                    Bm[2, 2 * a + 1] = dN[a, 0]
                K += Bm.T @ D @ Bm * (hx * hy / 4.0) * self.t
        return K

    def solve(self, f: np.ndarray, fixed: str = "left") -> np.ndarray:
        """Solve ``K u = f`` with a clamped ``left``/``right`` edge or free-free.

        ``f`` is a ``(n_nodes, 2)`` nodal force array (arbitrary, from the exchange).
        """
        Ke = self.element_stiffness()
        nd = 2 * self.n_nodes
        rows, cols, vals = [], [], []
        for j in range(self.ny):
            for i in range(self.nx):
                dofs = []
                for (di, dj) in ((0, 0), (1, 0), (1, 1), (0, 1)):
                    nid = self.node_index(i + di, j + dj)
                    dofs += [2 * nid, 2 * nid + 1]
                for a in range(8):
                    for b in range(8):
                        rows.append(dofs[a])
                        cols.append(dofs[b])
                        vals.append(Ke[a, b])
        K = sp.csr_matrix((vals, (rows, cols)), shape=(nd, nd))
        fac = np.asarray(f, dtype=float)
        if fac.ndim == 2 and fac.shape[1] == 3:
            fac = fac[:, :2]      # drop any out-of-plane column from a 3-D exchange
        fvec = fac.reshape(-1)
        fixed_dofs = []
        if fixed == "left":
            for j in range(self.ny + 1):
                nid = self.node_index(0, j)
                fixed_dofs += [2 * nid, 2 * nid + 1]
        elif fixed == "right":
            for j in range(self.ny + 1):
                nid = self.node_index(self.nx, j)
                fixed_dofs += [2 * nid, 2 * nid + 1]
        elif fixed == "none":
            fixed_dofs = []
        else:
            raise ValueError("fixed must be 'left', 'right' or 'none'")
        fixed_dofs = np.array(sorted(set(fixed_dofs)), dtype=int)
        free = np.setdiff1d(np.arange(nd), fixed_dofs)
        u = np.zeros(nd)
        if len(free):
            Kff = K[np.ix_(free, free)].tocsc()
            u[free] = spla.spsolve(Kff, fvec[free])
        return u.reshape(-1, 2)

    def tip_mean_ux(self, u: np.ndarray) -> float:
        return float(np.mean(u[[self.node_index(self.nx, j) for j in range(self.ny + 1)], 0]))

    @staticmethod
    def external_work(f: np.ndarray, u: np.ndarray) -> float:
        return float(0.5 * np.sum(np.asarray(f) * np.asarray(u)))


# ---------------------------------------------------------------------------
# 1-D plate strip in cylindrical bending
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PlateStripBending:
    """Euler-Bernoulli strip with plate rigidity ``D`` (per unit width).

    ``bc`` in ``{"simple-simple", "clamped-clamped", "clamped-free"}``.
    Node DOFs are ``(w, theta)``; ``solve`` accepts an arbitrary nodal load
    vector, which is exactly what the exchange operator produces.
    """

    L: float
    D: float
    n_elem: int = 40
    bc: str = "simple-simple"

    @property
    def le(self) -> float:
        return self.L / self.n_elem

    @property
    def n_nodes(self) -> int:
        return self.n_elem + 1

    def stiffness(self) -> sp.csr_matrix:
        L = self.le
        Ke = self.D / L ** 3 * np.array([
            [12, 6 * L, -12, 6 * L],
            [6 * L, 4 * L ** 2, -6 * L, 2 * L ** 2],
            [-12, -6 * L, 12, -6 * L],
            [6 * L, 2 * L ** 2, -6 * L, 4 * L ** 2],
        ])
        rows, cols, vals = [], [], []
        for e in range(self.n_elem):
            dofs = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
            for a in range(4):
                for b in range(4):
                    rows.append(dofs[a])
                    cols.append(dofs[b])
                    vals.append(Ke[a, b])
        return sp.csr_matrix((vals, (rows, cols)), shape=(2 * self.n_nodes, 2 * self.n_nodes))

    def solve(self, load: np.ndarray) -> np.ndarray:
        K = self.stiffness()
        nd = 2 * self.n_nodes
        if self.bc == "simple-simple":
            fixed = [0, nd - 2]
        elif self.bc == "clamped-clamped":
            fixed = [0, 1, nd - 2, nd - 1]
        elif self.bc == "clamped-free":
            fixed = [0, 1]
        else:
            raise ValueError("unknown bc")
        free = np.setdiff1d(np.arange(nd), fixed)
        u = np.zeros(nd)
        f = np.asarray(load, dtype=float).reshape(-1)
        Kff = K[np.ix_(free, free)].tocsc()
        u[free] = spla.spsolve(Kff, f[free])
        return u

    def mid_deflection(self, u: np.ndarray) -> float:
        n = self.n_elem
        if n % 2 == 0:
            return float(u[2 * (n // 2)])
        return float(0.5 * (u[2 * (n // 2)] + u[2 * (n // 2 + 1)]))

    def consistent_load(self, q_of_x) -> np.ndarray:
        """Consistent nodal loads for a distributed load ``q(x)`` [N/m] (3-pt Gauss)."""
        f = np.zeros(2 * self.n_nodes)
        L = self.le
        gx, gw = np.polynomial.legendre.leggauss(3)
        for e in range(self.n_elem):
            x0 = e * L
            for s, w in zip(gx, gw):
                x = x0 + 0.5 * L * (1.0 + s)
                q = q_of_x(x)
                H = self._hermite((s + 1.0) / 2.0, L)
                for a in range(4):
                    f[2 * e + a] += q * H[a] * w * (0.5 * L)
        return f

    @staticmethod
    def _hermite(s: float, L: float) -> np.ndarray:
        return np.array([
            1 - 3 * s ** 2 + 2 * s ** 3,
            L * (s - 2 * s ** 2 + s ** 3),
            3 * s ** 2 - 2 * s ** 3,
            L * (-s ** 2 + s ** 3),
        ])

    def external_work(self, load: np.ndarray, u: np.ndarray) -> float:
        return float(0.5 * np.dot(load, u))

    def strain_energy(self, u: np.ndarray) -> float:
        return float(0.5 * u @ (self.stiffness() @ u))


# ---------------------------------------------------------------------------
# 2-D Kirchhoff plate: Bogner-Fox-Schmit bicubic Hermite element
# ---------------------------------------------------------------------------
def _hermite_1d(s: float, h: float) -> list[np.ndarray]:
    """Value, 1st, 2nd and 3rd x-derivatives of the four 1-D Hermite functions."""
    val = np.array([
        1 - 3 * s ** 2 + 2 * s ** 3,
        h * (s - 2 * s ** 2 + s ** 3),
        3 * s ** 2 - 2 * s ** 3,
        h * (-s ** 2 + s ** 3),
    ])
    d1 = np.array([(-6 * s + 6 * s ** 2) / h, (1 - 4 * s + 3 * s ** 2),
                   (6 * s - 6 * s ** 2) / h, (-2 * s + 3 * s ** 2)])
    d2 = np.array([(-6 + 12 * s) / h ** 2, (-4 + 6 * s) / h,
                   (6 - 12 * s) / h ** 2, (-2 + 6 * s) / h])
    d3 = np.array([12 / h ** 3, 6 / h ** 2, -12 / h ** 3, 6 / h ** 2])
    return [val, d1, d2, d3]


# 1-D function index for a corner (0/1) and derivative order (0=value, 1=slope)
_FUNC_IDX = {(0, 0): 0, (1, 0): 2, (0, 1): 1, (1, 1): 3}


@dataclass(frozen=True)
class PlateBFS:
    """Simply-supported Kirchhoff plate on ``[0,a] x [0,b]``, BFS element."""

    a: float
    b: float
    t: float
    E: float
    nu: float
    nx: int = 12
    ny: int = 12

    @property
    def hx(self) -> float:
        return self.a / self.nx

    @property
    def hy(self) -> float:
        return self.b / self.ny

    @property
    def D(self) -> float:
        return self.E * self.t ** 3 / (12.0 * (1.0 - self.nu ** 2))

    @property
    def n_nodes(self) -> int:
        return (self.nx + 1) * (self.ny + 1)

    @property
    def ndof(self) -> int:
        return 4 * self.n_nodes

    def node_index(self, i: int, j: int) -> int:
        return j * (self.nx + 1) + i

    def coords(self) -> np.ndarray:
        xs = np.linspace(0.0, self.a, self.nx + 1)
        ys = np.linspace(0.0, self.b, self.ny + 1)
        xx, yy = np.meshgrid(xs, ys, indexing="xy")
        return np.column_stack([xx.ravel(), yy.ravel()])

    def _basis(self, s: float, r: float, hx: float, hy: float):
        """Local 16 basis values and their curvatures at ``(s, r)`` in [0,1]^2."""
        Hx = _hermite_1d(s, hx)
        Hy = _hermite_1d(r, hy)
        n = 16
        val = np.zeros(n)
        dxx = np.zeros(n)
        dyy = np.zeros(n)
        dxy = np.zeros(n)
        for corner in range(4):
            ci = 1 if corner in (1, 2) else 0
            cj = 1 if corner in (2, 3) else 0
            for dx in (0, 1):
                for dy in (0, 1):
                    k = 4 * corner + dx + 2 * dy
                    kx = _FUNC_IDX[(ci, dx)]
                    ky = _FUNC_IDX[(cj, dy)]
                    # the DOF type (value vs slope) is encoded in ``kx``/``ky``;
                    # the derivative order below is always the *physical* one
                    val[k] = Hx[0][kx] * Hy[0][ky]
                    dxx[k] = Hx[2][kx] * Hy[0][ky]
                    dyy[k] = Hx[0][kx] * Hy[2][ky]
                    dxy[k] = Hx[1][kx] * Hy[1][ky]
        return val, dxx, dyy, dxy

    def element_stiffness(self) -> np.ndarray:
        K = np.zeros((16, 16))
        D = self.D
        nu = self.nu
        g = 1.0 / np.sqrt(3.0)
        for s in (0.5 - g / 2, 0.5 + g / 2):
            for r in (0.5 - g / 2, 0.5 + g / 2):
                val, dxx, dyy, dxy = self._basis(s, r, self.hx, self.hy)
                B = D * (
                    np.outer(dxx, dxx) + np.outer(dyy, dyy)
                    + nu * (np.outer(dxx, dyy) + np.outer(dyy, dxx))
                    + 2 * (1 - nu) * np.outer(dxy, dxy)
                )
                K += B * (self.hx * self.hy / 4.0)
        return K

    def consistent_load(self, pressure_of_xy) -> np.ndarray:
        """Consistent nodal load vector for a pressure field ``p(x, y)``."""
        f = np.zeros(self.ndof)
        Ke_ref = None  # not needed
        del Ke_ref
        gx, gw = np.polynomial.legendre.leggauss(3)
        for j in range(self.ny):
            for i in range(self.nx):
                x0 = i * self.hx
                y0 = j * self.hy
                nodes = [
                    self.node_index(i, j), self.node_index(i + 1, j),
                    self.node_index(i + 1, j + 1), self.node_index(i, j + 1),
                ]
                for sx, wx in zip(gx, gw):
                    for sy, wy in zip(gx, gw):
                        s = 0.5 * (1 + sx)
                        r = 0.5 * (1 + sy)
                        x = x0 + s * self.hx
                        y = y0 + r * self.hy
                        p = pressure_of_xy(x, y)
                        val, _, _, _ = self._basis(s, r, self.hx, self.hy)
                        w = wx * wy * (self.hx * self.hy / 4.0)
                        for a, nid in enumerate(nodes):
                            f[4 * nid:4 * nid + 4] += p * val[4 * a:4 * a + 4] * w
        return f

    def nodal_force_vector(self, grid_forces: np.ndarray) -> np.ndarray:
        """Embed vertical nodal forces (``(n_nodes, 3)``) into the 4-DOF/node layout."""
        f = np.zeros(self.ndof)
        fz = np.asarray(grid_forces, dtype=float)[:, 2]
        f[0::4] = fz
        return f

    def _assemble(self) -> sp.csr_matrix:
        Ke = self.element_stiffness()
        rows, cols, vals = [], [], []
        for j in range(self.ny):
            for i in range(self.nx):
                nodes = [
                    self.node_index(i, j), self.node_index(i + 1, j),
                    self.node_index(i + 1, j + 1), self.node_index(i, j + 1),
                ]
                dofs = []
                for nid in nodes:
                    dofs += [4 * nid, 4 * nid + 1, 4 * nid + 2, 4 * nid + 3]
                for a in range(16):
                    for bidx in range(16):
                        rows.append(dofs[a])
                        cols.append(dofs[bidx])
                        vals.append(Ke[a, bidx])
        return sp.csr_matrix((vals, (rows, cols)), shape=(self.ndof, self.ndof))

    def solve(self, load: np.ndarray) -> np.ndarray:
        K = self._assemble()
        fixed = []
        for j in range(self.ny + 1):
            for i in range(self.nx + 1):
                if i in (0, self.nx) or j in (0, self.ny):
                    fixed.append(4 * self.node_index(i, j))  # w = 0 (hard simple support)
        free = np.setdiff1d(np.arange(self.ndof), np.array(fixed))
        u = np.zeros(self.ndof)
        Kff = K[np.ix_(free, free)].tocsc()
        u[free] = spla.spsolve(Kff, np.asarray(load, dtype=float)[free])
        return u

    def center_deflection(self, u: np.ndarray) -> float:
        if self.nx % 2 == 0 and self.ny % 2 == 0:
            nid = self.node_index(self.nx // 2, self.ny // 2)
            return float(u[4 * nid])
        raise ValueError("even nx/ny required for a center node")

    def strain_energy(self, u: np.ndarray) -> float:
        K = self._assemble()
        return float(0.5 * u @ (K @ u))

    @staticmethod
    def external_work(load: np.ndarray, u: np.ndarray) -> float:
        return float(0.5 * np.dot(load, u))