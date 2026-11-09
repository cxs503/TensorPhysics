"""Compact reference field solvers for the coupling-energy contract harness.

TensorPhysics owns cross-solver coordination and verification, not solver
kernels; these are the *minimum* executables needed to produce a real interface
flux/force with a real discretisation, so that conservation statements are made
about actual numerical objects instead of about bookkeeping identities:

* :class:`ThermalLBM2D` — D2Q9 advection-diffusion LBM for a passive scalar
  (constant velocity field, u = 0 by default).  Collision is BGK; walls are
  conservative full-way bounce-back except the coupled (south) face, which is a
  Dirichlet wall implemented as anti-bounce-back.  The scalar injected by that
  wall is returned step by step, which makes the interface flux an *exactly*
  measured quantity rather than a modelling assumption.
* :class:`Q4ThermalFEM` — 4-node bilinear quadrilateral transient heat
  conduction with a lumped capacity matrix.  Lumping is what makes the discrete
  energy balance coordinate-wise exact: ``E = sum_i M_i T_i`` changes only by
  the boundary flux terms.
* :class:`Q4PlaneStressFEM` — the same mesh, plane-stress elasticity, explicit
  central-difference (velocity-Verlet) integration with a lumped mass matrix.
* :class:`DEM2D` — rigid circular particles with linear spring-dashpot contact
  against a deforming boundary; the contact force is a single scalar per contact
  which the caller books into both sides (shared force).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

__all__ = [
    "ThermalLBM2D",
    "Q4Mesh",
    "Q4ThermalFEM",
    "Q4PlaneStressFEM",
    "DEM2D",
    "ContactLaw",
]

# ---------------------------------------------------------------------------
# D2Q9 lattice
# ---------------------------------------------------------------------------
_W = np.array([4.0 / 9.0] + [1.0 / 9.0] * 4 + [1.0 / 36.0] * 4)
_EX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
_EY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
#: opposite direction of each D2Q9 velocity; must be consistent with
#: ``_EX``/``_EY`` (opp(5)=(1,1) -> 7, opp(6)=(-1,1) -> 8, ...).
_OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
#: lattice sound speed squared for D2Q9.
CS2 = 1.0 / 3.0


class ThermalLBM2D:
    """D2Q9 passive-scalar (temperature) LBM on an ``ny x nx`` lattice.

    Parameters
    ----------
    tau:
        BGK relaxation time; the lattice diffusivity is ``(tau - 0.5) / 3``.
    u:
        Optional constant advection velocity (lattice units).  ``u=0`` keeps the
        total scalar exactly conserved by collision and streaming, which is the
        regime used for the strict energy-conservation gates.
    """

    def __init__(self, nx: int, ny: int, tau: float = 1.0, T_init: float = 0.0,
                 u: tuple[float, float] = (0.0, 0.0)) -> None:
        if nx < 3 or ny < 3:
            raise ValueError("lattice must be at least 3x3")
        if tau <= 0.5:
            raise ValueError("tau must exceed 0.5 for a positive diffusivity")
        self.nx, self.ny = int(nx), int(ny)
        self.tau = float(tau)
        self.u = (float(u[0]), float(u[1]))
        self.f = np.zeros((9, self.ny, self.nx))
        self.f[:] = _W[:, None, None] * float(T_init)
        #: per-column scalar injected by the coupled (south) wall at the last step
        self.wall_injection = np.zeros(self.nx)
        #: running total of the scalar injected by the coupled wall (diagnostic)
        self.injected_total = 0.0
        self._iy, self._ix = np.meshgrid(np.arange(self.ny), np.arange(self.nx),
                                         indexing="ij")

    # -- observables -----------------------------------------------------------
    @property
    def diffusivity(self) -> float:
        """Lattice thermal diffusivity of the BGK collision model."""
        return CS2 * (self.tau - 0.5)

    @property
    def T(self) -> np.ndarray:
        return self.f.sum(axis=0)

    def total(self) -> float:
        return float(self.f.sum())

    def south_wall_temperature(self) -> np.ndarray:
        return self.T[0, :].copy()

    def nonequilibrium_wall_flux(self) -> np.ndarray:
        """Per-column interface flux from the non-equilibrium populations.

        For the advection-diffusion LBM the wall-normal flux is
        ``q_y = -sum_i e_iy (f_i - f_i^eq) / tau`` in lattice units.  This is the
        *independently estimated* flux (the 'order 0' side of the consistency
        check): it is not the value handed to the solid.
        """
        T = self.T
        feq = _W[:, None, None] * T[None, :, :]
        fneq = self.f - feq
        return -np.tensordot(_EY, fneq[:, 0, :], axes=(0, 0)) / self.tau

    # -- time step -------------------------------------------------------------
    def step(self, wall_T: Sequence[float] | None = None) -> np.ndarray:
        """Advance one lattice step, returning the per-column wall injection.

        The return value is an array of ``nx`` scalars, one per lattice column:
        the amount of passive scalar that the anti-bounce-back Dirichlet wall
        added to (positive) or removed from (negative) the fluid in that column
        during this step.  ``sum()`` over the array is the *exact* interface
        exchange of the step, which is what the energy ledger books -- it is
        measured, not modelled.  The same vector is stored in
        :attr:`wall_injection`.
        """
        f = self.f
        T = f.sum(axis=0)
        feq = _W[:, None, None] * T[None, :, :]
        if self.u[0] != 0.0 or self.u[1] != 0.0:
            ux, uy = self.u
            cu = _EX[:, None, None] * ux + _EY[:, None, None] * uy
            feq = feq * (1.0 + 3.0 * cu)
        fpost = f + (feq - f) / self.tau

        # --- streaming (pull) with full-way bounce-back on straight walls ------
        # ``g[i][y, x]`` is the population of direction ``i`` *at* node (x, y)
        # after streaming.  Its source is (x - ex_i, y - ey_i); where that source
        # lies outside the lattice the population is the mirror image of the
        # outgoing one, which makes the sweep an exact permutation of the
        # populations (total scalar conserved to round-off).
        g = np.zeros_like(f)
        g[0] = fpost[0]
        for i in range(1, 9):
            ex, ey = int(_EX[i]), int(_EY[i])
            dst_x = slice(max(0, ex), self.nx + min(0, ex))
            src_x = slice(max(0, -ex), self.nx + min(0, -ex))
            dst_y = slice(max(0, ey), self.ny + min(0, ey))
            src_y = slice(max(0, -ey), self.ny + min(0, -ey))
            g[i, dst_y, dst_x] = fpost[i, src_y, src_x]
            mask = np.zeros((self.ny, self.nx), dtype=bool)
            if ex > 0:
                mask[:, 0] = True
            elif ex < 0:
                mask[:, -1] = True
            if ey > 0:
                mask[0, :] = True
            elif ey < 0:
                mask[-1, :] = True
            g[i][mask] = fpost[_OPP[i]][mask]

        # --- coupled south wall: anti-bounce-back Dirichlet -------------------
        injected_col = np.zeros(self.nx)
        if wall_T is not None:
            Tw = np.asarray(wall_T, dtype=float).reshape(-1)
            if Tw.size != self.nx:
                raise ValueError("wall_T must provide one value per lattice column")
            for i in (4, 7, 8):  # directions pointing into the south wall
                old = fpost[i][0, :]
                new = -old + 2.0 * _W[i] * Tw
                g[_OPP[i]][0, :] = new
                injected_col += new - old
        self.f = g
        self.wall_injection = injected_col
        self.injected_total += float(injected_col.sum())
        return injected_col


# ---------------------------------------------------------------------------
# 2D quadrilateral finite elements
# ---------------------------------------------------------------------------
_G = np.array([-1.0, 1.0]) / math.sqrt(3.0)


def _q4_shape(xi: float, eta: float) -> tuple[np.ndarray, np.ndarray]:
    N = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                         (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
    dN = 0.25 * np.array([[-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
                          [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]])
    return N, dN


class Q4Mesh:
    """Structured bilinear quad mesh on ``[0, lx] x [0, ly]``."""

    def __init__(self, nx: int, ny: int, lx: float, ly: float) -> None:
        if nx < 1 or ny < 1:
            raise ValueError("mesh needs at least one cell per direction")
        self.nx, self.ny = int(nx), int(ny)
        self.lx, self.ly = float(lx), float(ly)
        xs = np.linspace(0.0, self.lx, self.nx + 1)
        ys = np.linspace(0.0, self.ly, self.ny + 1)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        self.nodes = np.column_stack([X.ravel(order="C"), Y.ravel(order="C")])
        quads = []
        for i in range(self.nx):
            for j in range(self.ny):
                quads.append([self.node(i, j), self.node(i + 1, j),
                              self.node(i + 1, j + 1), self.node(i, j + 1)])
        self.quads = np.array(quads, dtype=int)

    def node(self, i: int, j: int) -> int:
        return i * (self.ny + 1) + j

    def row(self, j: int) -> np.ndarray:
        return np.array([self.node(i, j) for i in range(self.nx + 1)], dtype=int)

    def column(self, i: int) -> np.ndarray:
        return np.array([self.node(i, j) for j in range(self.ny + 1)], dtype=int)

    @property
    def n_nodes(self) -> int:
        return self.nodes.shape[0]

    def element_area(self) -> float:
        return (self.lx / self.nx) * (self.ly / self.ny)

    def _gradients(self, coords: np.ndarray):
        grads, dets = [], []
        for xi in _G:
            for eta in _G:
                _, dN = _q4_shape(xi, eta)
                J = dN @ coords
                detJ = float(J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0])
                if detJ <= 0.0:
                    raise ValueError("inverted quadrilateral element")
                # analytic 2x2 inverse: J^{-1} = 1/det [[b, -b01], [-b10, a]]
                Jinv = np.array([[J[1, 1], -J[0, 1]],
                                 [-J[1, 0], J[0, 0]]]) / detJ
                grads.append(Jinv @ dN)
                dets.append(detJ)
        return grads, dets

    def gradients(self):
        """Yield ``(element_nodes, [dNdx_gp], [detJ_gp])`` for every element."""
        for quad in self.quads:
            grads, dets = self._gradients(self.nodes[quad])
            yield quad, grads, dets


class Q4ThermalFEM:
    """Transient heat conduction with lumped capacity (backward Euler).

    Discrete energy ``E = sum_i M_i T_i`` [J] changes exactly by ``dt * sum b``
    per step for insulated boundaries, which is the property the ledger relies
    on.
    """

    def __init__(self, mesh: Q4Mesh, conductivity: float, volumetric_heat_capacity: float,
                 thickness: float = 1.0, T_init: float = 0.0) -> None:
        self.mesh = mesh
        self.k = float(conductivity)
        self.rhoc = float(volumetric_heat_capacity)
        self.thickness = float(thickness)
        n = mesh.n_nodes
        K = np.zeros((n, n))
        for quad, grads, dets in mesh.gradients():
            ke = np.zeros((4, 4))
            for dNdx, detJ in zip(grads, dets):
                ke += self.k * self.thickness * (dNdx.T @ dNdx) * detJ
            for a in range(4):
                for b in range(4):
                    K[quad[a], quad[b]] += ke[a, b]
        area4 = mesh.element_area() / 4.0
        M = np.zeros(n)
        for quad in mesh.quads:
            M[quad] += self.rhoc * self.thickness * area4
        self.K = K
        self.M = M
        self.T = np.full(n, float(T_init))

    def energy(self) -> float:
        return float(np.dot(self.M, self.T))

    def step(self, dt: float, flux: np.ndarray | None = None,
             matrix_inverse: np.ndarray | None = None) -> None:
        """One backward-Euler step with an optional nodal Neumann heat rate [W]."""
        rhs = (self.M / dt) * self.T
        if flux is not None:
            rhs = rhs + np.asarray(flux, dtype=float)
        if matrix_inverse is None:
            self.T = np.linalg.solve(np.diag(self.M / dt) + self.K, rhs)
        else:
            self.T = matrix_inverse @ rhs

    def cached_inverse(self, dt: float) -> np.ndarray:
        return np.linalg.inv(np.diag(self.M / dt) + self.K)

    def interface_flux_estimate(self, row: np.ndarray, spacing: float,
                                band: float,
                                below_row: np.ndarray | None = None) -> np.ndarray:
        """One-sided Fourier flux ``-k dT/dy`` across the interface row [W].

        ``below_row`` must be the node row one element *inside* the solid; it
        used to be hardcoded to ``mesh.row(1)``, which silently measured the
        wrong gradient for every mesh with more than two element rows (found by
        the interface-flux consistency gate of P40W3C-N09, 1260 % error).
        """
        below = self.mesh.row(1) if below_row is None else below_row
        grad = (self.T[row] - self.T[below]) / band
        return -self.k * grad * spacing * self.thickness


class Q4PlaneStressFEM:
    """Plane-stress elasticity with explicit velocity-Verlet integration."""

    def __init__(self, mesh: Q4Mesh, young: float, poisson: float, density: float,
                 thickness: float = 1.0) -> None:
        self.mesh = mesh
        self.E = float(young)
        self.nu = float(poisson)
        self.rho = float(density)
        self.thickness = float(thickness)
        n = mesh.n_nodes
        K = np.zeros((2 * n, 2 * n))
        D = (self.E / (1.0 - self.nu ** 2)) * np.array(
            [[1.0, self.nu, 0.0], [self.nu, 1.0, 0.0],
             [0.0, 0.0, (1.0 - self.nu) / 2.0]])
        for quad, grads, dets in mesh.gradients():
            ke = np.zeros((8, 8))
            for dNdx, detJ in zip(grads, dets):
                B = np.zeros((3, 8))
                for a in range(4):
                    B[0, 2 * a] = dNdx[0, a]
                    B[1, 2 * a + 1] = dNdx[1, a]
                    B[2, 2 * a] = dNdx[1, a]
                    B[2, 2 * a + 1] = dNdx[0, a]
                ke += self.thickness * detJ * (B.T @ D @ B)
            dofs = np.array([[2 * q, 2 * q + 1] for q in quad]).ravel()
            K[np.ix_(dofs, dofs)] += ke
        area4 = mesh.element_area() / 4.0
        m = np.zeros(n)
        for quad in mesh.quads:
            m[quad] += self.rho * self.thickness * area4
        self.K = K
        self.mass = np.repeat(m, 2)
        self.u = np.zeros(2 * n)
        self.v = np.zeros(2 * n)
        self.fixed = np.zeros(2 * n, dtype=bool)

    def fix(self, nodes: Sequence[int]) -> None:
        for node in nodes:
            self.fixed[2 * node] = True
            self.fixed[2 * node + 1] = True

    def critical_dt(self, safety: float = 0.5) -> float:
        omega2 = np.linalg.eigvalsh(np.linalg.pinv(np.diag(self.mass)) @ self.K).max()
        return safety * 2.0 / math.sqrt(max(float(omega2), 1e-300))

    def strain_energy(self) -> float:
        return 0.5 * float(self.u @ (self.K @ self.u))

    def kinetic_energy(self) -> float:
        return 0.5 * float(np.dot(self.mass, self.v ** 2))

    def energy(self) -> float:
        return self.strain_energy() + self.kinetic_energy()

    def accelerations(self, f_ext: np.ndarray | None = None) -> np.ndarray:
        f = -(self.K @ self.u)
        if f_ext is not None:
            f = f + np.asarray(f_ext, dtype=float)
        f[self.fixed] = 0.0
        return f / self.mass

    def step(self, dt: float, f_ext: np.ndarray | None = None) -> None:
        """Velocity-Verlet (symplectic) step with an external nodal force [N]."""
        a0 = self.accelerations(f_ext)
        self.u = self.u + dt * self.v + 0.5 * dt * dt * a0
        a1 = self.accelerations(f_ext)
        self.v = self.v + 0.5 * dt * (a0 + a1)
        self.u[self.fixed] = 0.0
        self.v[self.fixed] = 0.0


# ---------------------------------------------------------------------------
# DEM
# ---------------------------------------------------------------------------
@dataclass
class ContactLaw:
    """Linear spring-dashpot contact law (Hertzian reduction for small overlap)."""

    stiffness: float = 2.0e7
    damping: float = 0.0

    def force(self, overlap: float, overlap_rate: float) -> float:
        if overlap <= 0.0:
            return 0.0
        return self.stiffness * overlap + self.damping * max(overlap_rate, 0.0)

    def potential(self, overlap: float) -> float:
        if overlap <= 0.0:
            return 0.0
        return 0.5 * self.stiffness * overlap * overlap


class DEM2D:
    """Rigid circular particles interacting with a moving wall (``x = x_wall(y)``).

    The particles are integrated with velocity-Verlet.  Contact against the
    plate's right edge uses the *nearest plate node* for the wall velocity and
    position, which keeps the force sharing exactly one scalar per contact.
    """

    def __init__(self, positions: np.ndarray, velocities: np.ndarray, radii: np.ndarray,
                 density: float, thickness: float = 1.0,
                 law: ContactLaw | None = None) -> None:
        self.pos = np.asarray(positions, dtype=float).copy()
        self.vel = np.asarray(velocities, dtype=float).copy()
        self.radius = np.asarray(radii, dtype=float).copy()
        if not (self.pos.shape[0] == self.vel.shape[0] == self.radius.shape[0]):
            raise ValueError("positions, velocities and radii must agree")
        self.area = math.pi * self.radius ** 2
        self.mass = density * thickness * self.area
        self.law = law or ContactLaw()

    @property
    def kinetic_energy(self) -> float:
        return float(0.5 * np.dot(self.mass, np.sum(self.vel ** 2, axis=1)))

    def momentum(self) -> np.ndarray:
        return np.sum(self.mass[:, None] * self.vel, axis=0)

    def contact_forces(self, wall_x: np.ndarray, wall_vel: np.ndarray,
                       wall_y: np.ndarray, node_of: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(force_on_particles, force_per_wall_node, overlaps, node_ids)``.

        Contact is one-sided along ``-x``: the particle approaches the plate's
        right edge located at ``wall_x[node_of]``.  Overlap is
        ``r - (x_p - x_wall)``; the overlap rate is ``-(v_px - v_wall_x)``.
        """
        gaps = self.pos[:, 0] - wall_x[node_of] - self.radius
        overlaps = np.maximum(-gaps, 0.0)
        rates = -(self.vel[:, 0] - wall_vel[node_of])
        mags = np.array([self.law.force(o, r) for o, r in zip(overlaps, rates)])
        # The particles sit on the +x side of the wall face, so a repulsive
        # contact pushes them in +x and the reaction on the plate is -x.  One
        # and the same magnitude array is used for both sides, which is what
        # makes the exchange Newton-consistent; evaluating the two sides
        # independently is the ``independent`` negative control of scenarios.
        f_particles = np.column_stack([mags, np.zeros_like(mags)])
        f_nodes = np.zeros((wall_x.size, 2))
        np.add.at(f_nodes, node_of, np.column_stack([-mags, np.zeros_like(mags)]))
        return f_particles, f_nodes, overlaps, node_of

    def step(self, dt: float, force: np.ndarray,
             force_fn: "Callable[[np.ndarray, np.ndarray], np.ndarray] | None" = None
             ) -> None:
        """Advance the discs by ``dt``.

        With ``force_fn=None`` the supplied ``force`` is held constant over the
        step, for which the update below is *exact* (position and velocity of a
        constant-acceleration motion) -- that is the mode the conservative
        shared-force coupling uses, so the coupling error is purely the stale
        force of the exchange interval and not an integration error.

        With ``force_fn(positions, velocities) -> force`` a kick-drift-kick
        (velocity-Verlet) step is taken instead, which is what a single-solver
        impact study with a stiff contact needs: the plain constant-force update
        pumps energy into the contact spring (measured: +37 % kinetic energy at
        ``omega dt = 0.2``, see the module report).
        """
        acc = force / self.mass[:, None]
        if force_fn is None:
            self.pos = self.pos + dt * self.vel + 0.5 * dt * dt * acc
            self.vel = self.vel + dt * acc
            return
        self.vel = self.vel + 0.5 * dt * acc
        self.pos = self.pos + dt * self.vel
        acc1 = force_fn(self.pos, self.vel) / self.mass[:, None]
        self.vel = self.vel + 0.5 * dt * acc1