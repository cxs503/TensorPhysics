"""Hexahedral linear-elasticity solver for the hull structural response.

The hull is represented by the wetted bow panel: a regular box meshed with
8-node trilinear hexahedra, integrated with 2x2x2 Gauss quadrature.  Time
integration is Newmark-beta (beta = 1/4, gamma = 1/2, unconditionally stable)
with lumped mass and stiffness-proportional damping, which is the minimum
machinery needed by the coupler: receive nodal contact loads, return nodal
displacements and velocities.

The load transfer from DEM contacts to the mesh is *conservative*: each contact
force is distributed to the eight nodes of the element that contains the
contact point with trilinear weights, so the nodal resultant equals the applied
contact resultant to machine precision (verified in the test suite).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Local node ordering of the 8-node hex (values of xi, eta, zeta)
_HEX = np.array(
    [
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ],
    dtype=float,
)
_GP = 1.0 / np.sqrt(3.0)


@dataclass
class HullFEMConfig:
    """Mesh and material parameters (lattice units)."""

    nx: int = 6
    ny: int = 8
    nz: int = 4
    size: Tuple[float, float, float] = (6.0, 8.0, 4.0)
    youngs_modulus: float = 50.0
    poisson: float = 0.3
    density: float = 1.0
    damping_ratio: float = 0.02
    #: Face that is clamped: "xmin" (ship-fixed, panel root) or "zmin".
    clamp_face: str = "xmin"

    def __post_init__(self) -> None:
        if min(self.nx, self.ny, self.nz) < 1:
            raise ValueError("mesh must have at least one element per axis")
        if any(s <= 0 for s in self.size):
            raise ValueError("element block size must be positive")
        if not (0.0 <= self.poisson < 0.5):
            raise ValueError("poisson ratio must be in [0, 0.5)")
        if self.clamp_face not in ("xmin", "xmax", "zmin", "zmax", "xmin_axial"):
            raise ValueError("unsupported clamp face")


class Hex8Mesh:
    """Regular box mesh of 8-node hexahedra."""

    def __init__(self, config: HullFEMConfig):
        self.cfg = config
        self.nx, self.ny, self.nz = config.nx, config.ny, config.nz
        self.lx, self.ly, self.lz = (s / n for s, n in zip(config.size, (self.nx, self.ny, self.nz)))
        self.node_grid = (self.nx + 1, self.ny + 1, self.nz + 1)
        gx, gy, gz = np.meshgrid(
            np.linspace(0.0, config.size[0], self.nx + 1),
            np.linspace(0.0, config.size[1], self.ny + 1),
            np.linspace(0.0, config.size[2], self.nz + 1),
            indexing="ij",
        )
        # node index order (ix, iy, iz)
        self.nodes = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
        self.n_nodes = len(self.nodes)
        self.elements = self._connectivity()

    def _connectivity(self) -> np.ndarray:
        nx, ny, nz = self.nx, self.ny, self.nz
        elems = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    n = []
                    for dx, dy, dz in _HEX.astype(int):
                        ix = i + (1 if dx > 0 else 0)
                        iy = j + (1 if dy > 0 else 0)
                        iz = k + (1 if dz > 0 else 0)
                        n.append(self.node_id(ix, iy, iz))
                    elems.append(n)
        return np.array(elems, dtype=int)

    def node_id(self, ix: int, iy: int, iz: int) -> int:
        ngy, ngz = self.node_grid[1], self.node_grid[2]
        return (ix * ngy + iy) * ngz + iz

    def volume(self, e: int) -> float:
        return self.lx * self.ly * self.lz

    def shape(self, xi: float, eta: float, zeta: float):
        N = 0.125 * (1 + _HEX[:, 0] * xi) * (1 + _HEX[:, 1] * eta) * (1 + _HEX[:, 2] * zeta)
        dN = np.stack(
            [
                0.125 * _HEX[:, 0] * (1 + _HEX[:, 1] * eta) * (1 + _HEX[:, 2] * zeta),
                0.125 * (1 + _HEX[:, 0] * xi) * _HEX[:, 1] * (1 + _HEX[:, 2] * zeta),
                0.125 * (1 + _HEX[:, 0] * xi) * (1 + _HEX[:, 1] * eta) * _HEX[:, 2],
            ],
            axis=1,
        )
        # Jacobian of a regular box is diagonal and constant
        dNdx = dN * np.array([2.0 / self.lx, 2.0 / self.ly, 2.0 / self.lz])
        return N, dNdx

    def strain_matrix(self, dNdx: np.ndarray) -> np.ndarray:
        B = np.zeros((6, 24))
        for a in range(8):
            dx, dy, dz = dNdx[a]
            B[0, 3 * a] = dx
            B[1, 3 * a + 1] = dy
            B[2, 3 * a + 2] = dz
            B[3, 3 * a] = dy
            B[3, 3 * a + 1] = dx
            B[4, 3 * a + 1] = dz
            B[4, 3 * a + 2] = dy
            B[5, 3 * a] = dz
            B[5, 3 * a + 2] = dx
        return B


def _elastic_matrix(E: float, nu: float) -> np.ndarray:
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[0, 0] = D[1, 1] = D[2, 2] = lam + 2 * mu
    D[3, 3] = D[4, 4] = D[5, 5] = mu
    return D


class ElastostaticHull:
    """Newmark-integrated hull panel responding to ice contact loads."""

    def __init__(self, config: HullFEMConfig):
        self.cfg = config
        self.mesh = Hex8Mesh(config)
        self.D = _elastic_matrix(config.youngs_modulus, config.poisson)
        self.ndof = 3 * self.mesh.n_nodes
        self.K = self._assemble_stiffness()
        self.M = self._assemble_mass()
        self.fixed = self._clamped_dofs()
        self.free = np.setdiff1d(np.arange(self.ndof), self.fixed)
        self.u = np.zeros(self.ndof)
        self.v = np.zeros(self.ndof)
        self.a = np.zeros(self.ndof)
        self.dt = 1.0
        self.damping_work = 0.0
        self._factored = None

    # -- assembly -----------------------------------------------------------
    def _assemble_stiffness(self) -> sp.csr_matrix:
        rows, cols, vals = [], [], []
        B_store = []
        gps = (-_GP, _GP)
        for xi in gps:
            for eta in gps:
                for zeta in gps:
                    B_store.append(self.mesh.strain_matrix(self.mesh.shape(xi, eta, zeta)[1]))
        for e, conn in enumerate(self.mesh.elements):
            dofs = np.array([[3 * n, 3 * n + 1, 3 * n + 2] for n in conn]).ravel()
            ke = np.zeros((24, 24))
            for B in B_store:
                ke += B.T @ self.D @ B * (self.mesh.volume(e) / 8.0)
            for a in range(24):
                rows.extend([dofs[a]] * 24)
                cols.extend(dofs)
            vals.append(ke.ravel())
        vals = np.concatenate(vals)
        K = sp.coo_matrix((vals, (rows, cols)), shape=(self.ndof, self.ndof)).tocsr()
        return (K + K.T) * 0.5

    def _assemble_mass(self) -> sp.csr_matrix:
        diag = np.zeros(self.ndof)
        for e, conn in enumerate(self.mesh.elements):
            m = self.cfg.density * self.mesh.volume(e) / 8.0
            for n in conn:
                diag[3 * n: 3 * n + 3] += m
        return sp.diags(diag).tocsr()

    def _clamped_dofs(self) -> np.ndarray:
        nodes = self.mesh.nodes
        face = self.cfg.clamp_face
        tol = 1e-9
        if face == "xmin_axial":
            # minimal constraints: only the axial component on the xmin face
            # plus two transverse pins, so Poisson contraction stays free
            sel = nodes[:, 0] <= tol
            idx = np.where(sel)[0]
            dofs = [3 * n for n in idx]
            dofs.append(3 * 0 + 1)
            dofs.append(3 * 0 + 2)
            return np.array(dofs, dtype=int)
        if face == "xmin":
            sel = nodes[:, 0] <= tol
        elif face == "xmax":
            sel = nodes[:, 0] >= self.cfg.size[0] - tol
        elif face == "zmin":
            sel = nodes[:, 2] <= tol
        else:
            sel = nodes[:, 2] >= self.cfg.size[2] - tol
        idx = np.where(sel)[0]
        return np.array([[3 * n, 3 * n + 1, 3 * n + 2] for n in idx]).ravel()

    # -- load mapping -------------------------------------------------------
    def map_contact_loads(self, points: np.ndarray, forces: np.ndarray) -> np.ndarray:
        """Distribute point forces to nodes with trilinear weights (conservative)."""
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        frc = np.atleast_2d(np.asarray(forces, dtype=float))
        if pts.shape != frc.shape or pts.shape[1] != 3:
            raise ValueError("points and forces must both be (n, 3)")
        f = np.zeros(self.ndof)
        if len(pts) == 0:
            return f
        nx, ny, nz = self.mesh.nx, self.mesh.ny, self.mesh.nz
        lx, ly, lz = self.mesh.lx, self.mesh.ly, self.mesh.lz
        # element index and local coordinates, clipped into the block
        gx = np.clip(pts[:, 0] / lx, 0.0, nx - 1e-12)
        gy = np.clip(pts[:, 1] / ly, 0.0, ny - 1e-12)
        gz = np.clip(pts[:, 2] / lz, 0.0, nz - 1e-12)
        ex = gx.astype(int)
        ey = gy.astype(int)
        ez = gz.astype(int)
        xi = 2.0 * (gx - ex) - 1.0
        eta = 2.0 * (gy - ey) - 1.0
        zeta = 2.0 * (gz - ez) - 1.0
        for c in range(len(pts)):
            N, _ = self.mesh.shape(xi[c], eta[c], zeta[c])
            for a, (dx, dy, dz) in enumerate(_HEX.astype(int)):
                nid = self.mesh.node_id(
                    ex[c] + (1 if dx > 0 else 0),
                    ey[c] + (1 if dy > 0 else 0),
                    ez[c] + (1 if dz > 0 else 0),
                )
                f[3 * nid: 3 * nid + 3] += N[a] * frc[c]
        return f

    def face_loads(self, axis: int, at_max: bool, traction: Sequence[float]) -> np.ndarray:
        """Consistent nodal loads for a uniform traction on one boundary face.

        Integrates ``N_i * t`` over the face with a 2x2 Gauss rule so the
        resultant equals ``t * A`` and the element is loaded without spurious
        local stress concentrations (a patch test in the strict sense).
        """
        t = np.asarray(traction, dtype=float)
        if t.shape != (3,):
            raise ValueError("traction must be a 3-vector")
        f = np.zeros(self.ndof)
        nx, ny, nz = self.mesh.nx, self.mesh.ny, self.mesh.nz
        area = 1.0
        if axis == 0:
            area = self.mesh.ly * self.mesh.lz
            i_range = [nx - 1] if at_max else [0]
            jk = [(j, k) for j in range(ny) for k in range(nz)]
        elif axis == 1:
            area = self.mesh.lx * self.mesh.lz
            i_range = range(nx)
            jk = [(ny - 1 if at_max else 0, k) for k in range(nz)]
        elif axis == 2:
            area = self.mesh.lx * self.mesh.ly
            i_range = range(nx)
            jk = [(j, nz - 1 if at_max else 0) for j in range(ny)]
        else:
            raise ValueError("axis must be 0, 1 or 2")
        gps = (-_GP, _GP)
        sgn = 1.0 if at_max else -1.0
        for i in i_range:
            for (j, k) in jk:
                di = 1 if (axis == 0 and at_max) else 0
                dj = 1 if (axis == 1 and at_max) else 0
                dk = 1 if (axis == 2 and at_max) else 0
                base = (i + di, j + dj, k + dk)
                if axis == 0:
                    ijk = [base, (i + di, j + 1, k + dk), (i + di, j + 1, k + 1), (i + di, j, k + 1)]
                elif axis == 1:
                    ijk = [base, (i + 1, j + dj, k + dk), (i + 1, j + dj, k + 1), (i, j + dj, k + 1)]
                else:
                    ijk = [base, (i + 1, j + dj, k + dk), (i + 1, j + 1, k + dk), (i, j + 1, k + dk)]
                for a in gps:
                    for b in gps:
                        if axis == 0:
                            N, _ = self.mesh.shape(sgn, a, b)
                            pick = [1, 2, 6, 5] if at_max else [0, 3, 7, 4]
                        elif axis == 1:
                            N, _ = self.mesh.shape(a, sgn, b)
                            pick = [3, 2, 6, 7] if at_max else [0, 1, 5, 4]
                        else:
                            N, _ = self.mesh.shape(a, b, sgn)
                            pick = [4, 5, 6, 7] if at_max else [0, 1, 2, 3]
                        for local, nijk in zip(pick, ijk):
                            nid = self.mesh.node_id(*nijk)
                            f[3 * nid: 3 * nid + 3] += N[local] * t * area / 4.0
        return f

    # -- solves -------------------------------------------------------------
    def _reduced_solve(self, A: sp.csr_matrix, rhs: np.ndarray) -> np.ndarray:
        u = np.zeros(self.ndof)
        if len(self.free) == 0:
            return u
        A_ff = A[self.free][:, self.free]
        u[self.free] = spla.spsolve(A_ff.tocsc(), rhs[self.free])
        return u

    def static_solve(self, nodal_load: np.ndarray) -> np.ndarray:
        """Solve K u = f with the clamped face fixed."""
        return self._reduced_solve(self.K, np.asarray(nodal_load, dtype=float))

    def initialise_newmark(self, dt: float) -> None:
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.dt = float(dt)
        beta, gamma = 0.25, 0.5
        omega = self.first_omega()
        a0 = self.cfg.damping_ratio * omega
        a1 = self.cfg.damping_ratio / max(omega, 1e-12)
        self.C = (a0 * self.M + a1 * self.K).tocsr()
        self._c1 = a0 / (beta * self.dt)
        self._c2 = a1 + a0 * gamma / (beta * self.dt)
        self._c3 = a0 * self.dt * gamma / beta
        self._c4 = a1 * self.dt * (1.0 - gamma / beta)
        K_eff = self.K + self._c1 * self.M + self._c2 * self.K
        K_ff = K_eff[self.free][:, self.free].tocsc()
        self._factored = spla.factorized(K_ff)

    def first_omega(self) -> float:
        """Fundamental frequency from a generalised eigenproblem (small meshes)."""
        if len(self.free) > 900:
            return float(np.sqrt(self.cfg.youngs_modulus / max(self.cfg.density, 1e-12)) / max(self.cfg.size))
        K = self.K[self.free][:, self.free].toarray()
        M = self.M[self.free][:, self.free].toarray()
        try:
            vals = np.linalg.eigvalsh(np.linalg.solve(M, K + 1e-9 * M))
            vals = vals[vals > 0]
            return float(np.sqrt(vals.min())) if len(vals) else 1.0
        except np.linalg.LinAlgError:
            return 1.0

    def step_newmark(self, nodal_load: np.ndarray) -> np.ndarray:
        """One Newmark step with the given external nodal load."""
        if self._factored is None:
            self.initialise_newmark(self.dt)
        f = np.asarray(nodal_load, dtype=float)
        beta = 0.25
        gamma = 0.5
        dt = self.dt
        rhs = f.copy()
        rhs[self.free] += (
            self.M[self.free][:, self.free] @ (self.u[self.free] / (beta * dt**2) + self.v[self.free] / (beta * dt)
                                               + (1.0 / (2 * beta) - 1.0) * self.a[self.free])
        )
        rhs[self.free] += (
            self.C[self.free][:, self.free] @ (gamma * self.u[self.free] / (beta * dt)
                                               + (gamma / beta - 1.0) * self.v[self.free]
                                               + dt * (gamma / (2 * beta) - 1.0) * self.a[self.free])
        )
        u_new = np.zeros(self.ndof)
        u_new[self.free] = self._factored(rhs[self.free])
        a_new = (u_new - self.u) / (beta * dt**2) - self.v / (beta * dt) - (1.0 / (2 * beta) - 1.0) * self.a
        v_new = self.v + dt * ((1 - gamma) * self.a + gamma * a_new)
        self.u, self.v, self.a = u_new, v_new, a_new
        # Rayleigh damping dissipation over the step (always >= 0)
        self.damping_work += float(v_new @ (self.C @ v_new)) * dt
        return self.u.copy()

    def strain_energy(self) -> float:
        return float(0.5 * self.u @ (self.K @ self.u))

    def kinetic_energy(self) -> float:
        return float(0.5 * self.v @ (self.M @ self.v))

    def displacement_at(self, points: np.ndarray) -> np.ndarray:
        """Sample the displacement field by trilinear interpolation."""
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        nx, ny, nz = self.mesh.nx, self.mesh.ny, self.mesh.nz
        lx, ly, lz = self.mesh.lx, self.mesh.ly, self.mesh.lz
        gx = np.clip(pts[:, 0] / lx, 0.0, nx - 1e-12)
        gy = np.clip(pts[:, 1] / ly, 0.0, ny - 1e-12)
        gz = np.clip(pts[:, 2] / lz, 0.0, nz - 1e-12)
        ex, ey, ez = gx.astype(int), gy.astype(int), gz.astype(int)
        xi = 2.0 * (gx - ex) - 1.0
        eta = 2.0 * (gy - ey) - 1.0
        zeta = 2.0 * (gz - ez) - 1.0
        out = np.zeros((len(pts), 3))
        u = self.u.reshape(-1, 3)
        for c in range(len(pts)):
            N, _ = self.mesh.shape(xi[c], eta[c], zeta[c])
            for a, (dx, dy, dz) in enumerate(_HEX.astype(int)):
                nid = self.mesh.node_id(
                    ex[c] + (1 if dx > 0 else 0),
                    ey[c] + (1 if dy > 0 else 0),
                    ez[c] + (1 if dz > 0 else 0),
                )
                out[c] += N[a] * u[nid]
        return out

    def probe_node(self, which: str = "max_x_max_z") -> int:
        nodes = self.mesh.nodes
        if which == "max_x_max_z":
            score = nodes[:, 0] + nodes[:, 2]
        elif which == "max_x":
            score = nodes[:, 0]
        else:
            score = -nodes[:, 2]
        return int(np.argmax(score))


def newmark_integrate(
    hull: ElastostaticHull,
    loads: Sequence[np.ndarray],
    dt: float,
) -> np.ndarray:
    """Convenience wrapper: integrate a prescribed load history."""
    hull.initialise_newmark(dt)
    probe = hull.probe_node()
    hist = []
    for f in loads:
        hull.step_newmark(np.asarray(f, dtype=float))
        hist.append(hull.u[3 * probe: 3 * probe + 3].copy())
    return np.array(hist)