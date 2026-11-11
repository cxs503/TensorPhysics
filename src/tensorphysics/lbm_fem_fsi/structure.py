"""Plane-stress Q4 finite-element elastic tether carrying a rigid cylinder.

Campaign node ``p362w2-n09`` -- the *structural* half of the LBM -> FEM exchange.

The elastically mounted cylinder of the VIV benchmark is realised as a real FE
model rather than a one-degree-of-freedom spring: a vertical cantilever tether
(clamped at the top, outside the test section) built from bilinear plane-stress
Q4 elements, whose free end carries the cylinder as a lumped tip mass.  The
cylinder itself is rigid, its centre-locked streamwise motion is constrained
(the standard 1-DOF transverse VIV arrangement), so the structural dynamics that
the fluid actually feels are those of the FE model: many modes, a consistent
stiffness matrix, and an eigenfrequency computed from the assembly -- not
imposed.

Plane-stress Q4, thickness ``t``, material ``(E, nu, rho)``::

    K_e = t * integral B^T D B dA,        M_e = rho t * integral N^T N dA
    D   = E/(1-nu^2) [[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]]

Rayleigh damping ``C = alpha M + beta K`` is calibrated from a target modal
damping ratio at the first eigenfrequency (:meth:`PlaneStressTether.set_rayleigh`),
and the implicit Newmark-beta scheme (``beta = 1/4, gamma = 1/2``,
unconditionally stable, no numerical damping) advances the structure.  Every
quantity the conservation ledger needs is produced incrementally:

``kinetic``    1/2 v^T M v
``strain``     1/2 u^T K u
``damping``    int c v^2 dt  (trapezoidal, accumulated)
``external``   int F_tip . v_tip dt (accumulated by the coupling loop)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["PlaneStressTether", "q4_stiffness", "q4_mass", "PlaneStressState"]


def _shape(xi: float, eta: float):
    N = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                         (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
    dxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
    deta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
    return N, dxi, deta


def q4_stiffness(xy: np.ndarray, E: float, nu: float, t: float) -> np.ndarray:
    """8x8 plane-stress stiffness of a Q4 element with node coordinates ``xy``."""
    D = E / (1.0 - nu * nu) * np.array([[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, 0.5 * (1 - nu)]])
    Ke = np.zeros((8, 8))
    g = 1.0 / np.sqrt(3.0)
    for xi, eta in ((-g, -g), (g, -g), (g, g), (-g, g)):
        N, dxi, deta = _shape(xi, eta)
        J = np.array([[dxi @ xy[:, 0], dxi @ xy[:, 1]],
                      [deta @ xy[:, 0], deta @ xy[:, 1]]])
        detJ = np.linalg.det(J)
        dN = np.linalg.inv(J) @ np.vstack([dxi, deta])
        B = np.zeros((3, 8))
        B[0, 0::2] = dN[0]
        B[1, 1::2] = dN[1]
        B[2, 0::2] = dN[1]
        B[2, 1::2] = dN[0]
        Ke += B.T @ D @ B * detJ * t
    return Ke


def q4_mass(xy: np.ndarray, rho: float, t: float) -> np.ndarray:
    """8x8 *consistent* plane-stress mass matrix of a Q4 element."""
    Me = np.zeros((8, 8))
    g = 1.0 / np.sqrt(3.0)
    for xi, eta in ((-g, -g), (g, -g), (g, g), (-g, g)):
        N, dxi, deta = _shape(xi, eta)
        J = np.array([[dxi @ xy[:, 0], dxi @ xy[:, 1]],
                      [deta @ xy[:, 0], deta @ xy[:, 1]]])
        detJ = np.linalg.det(J)
        Ni = np.zeros((2, 8))
        Ni[0, 0::2] = N
        Ni[1, 1::2] = N
        Me += rho * t * (Ni.T @ Ni) * detJ
    return Me


@dataclass
class PlaneStressState:
    u: np.ndarray
    v: np.ndarray
    a: np.ndarray


class PlaneStressTether:
    """Vertical cantilever tether, clamped at ``s = 0``, free at ``s = L``.

    Coordinates are ``(s, w)``: ``s`` along the tether, ``w`` the transverse
    (flow-normal) direction, which is the only direction the cylinder moves in.
    """

    def __init__(self, length: float = 64.0, height: float = 8.0, n_s: int = 24,
                 n_w: int = 3, E: float = 324.0, nu: float = 0.3, rho: float = 1.0,
                 thickness: float = 1.0):
        self.L = float(length)
        self.H = float(height)
        self.n_s = int(n_s)
        self.n_w = int(n_w)
        self.E = float(E)
        self.nu = float(nu)
        self.rho = float(rho)
        self.t = float(thickness)
        self._mesh()
        self._assemble()
        self.cyl_mass = 0.0
        self.dt_fem = 1.0
        self.alpha = 0.0
        self.beta = 0.0
        self.state = PlaneStressState(np.zeros(2 * self.n_nodes),
                                      np.zeros(2 * self.n_nodes),
                                      np.zeros(2 * self.n_nodes))
        self.damping_dissipation = 0.0
        self.external_work = 0.0

    # ------------------------------------------------------------------ mesh
    def _mesh(self):
        ns, nw = self.n_s, self.n_w
        s = np.linspace(0.0, self.L, ns + 1)
        w = np.linspace(-0.5 * self.H, 0.5 * self.H, nw + 1)
        S, Wg = np.meshgrid(s, w, indexing="ij")
        self.coords = np.column_stack([S.ravel(), Wg.ravel()])
        self.n_nodes = self.coords.shape[0]
        conn = []
        for i in range(ns):
            for j in range(nw):
                n0 = i * (nw + 1) + j
                conn.append([n0, n0 + (nw + 1), n0 + (nw + 1) + 1, n0 + 1])
        self.conn = np.array(conn, dtype=int)
        # clamped edge s = 0 (both displacement components)
        fixed = np.where(self.coords[:, 0] <= 0.0)[0]
        self.fixed_dofs = np.sort(np.concatenate([2 * fixed, 2 * fixed + 1]))
        # free edge nodes = attachment face of the cylinder
        self.tip_nodes = np.where(np.isclose(self.coords[:, 0], self.L))[0]
        self.tip_w = self.coords[self.tip_nodes, 1]

    def _assemble(self):
        ndof = 2 * self.n_nodes
        K = np.zeros((ndof, ndof))
        M = np.zeros((ndof, ndof))
        for el in self.conn:
            xy = self.coords[el]
            Ke = q4_stiffness(xy, self.E, self.nu, self.t)
            Me = q4_mass(xy, self.rho, self.t)
            idx = np.zeros(8, dtype=int)
            idx[0::2] = 2 * el
            idx[1::2] = 2 * el + 1
            K[np.ix_(idx, idx)] += Ke
            M[np.ix_(idx, idx)] += Me
        self.K = K
        self.M = M
        self.free = np.setdiff1d(np.arange(ndof), self.fixed_dofs)

    # ---------------------------------------------------------------- properties
    @property
    def transverse_dofs(self) -> np.ndarray:
        """Transverse (``w``) DOFs of the free edge -- the cylinder attachment."""
        return 2 * self.tip_nodes + 1

    @property
    def tip_axis(self) -> float:
        """Transverse coordinate of the attachment centroid."""
        return float(self.tip_w.mean())

    def add_tip_mass(self, mass: float) -> None:
        """Lump the rigid cylinder mass on the attachment face (sum = ``mass``)."""
        idx = self.transverse_dofs
        self.M[np.ix_(idx, idx)] += np.eye(idx.size) * (mass / idx.size)
        self.cyl_mass = float(mass)

    def set_rayleigh(self, zeta: float, f1: float | None = None) -> float:
        """Calibrate ``C = alpha M + beta K`` for damping ratio ``zeta`` at mode 1."""
        if f1 is None:
            f1 = self.first_frequency()
        w1 = 2.0 * np.pi * f1
        self.alpha = 2.0 * zeta * w1 * 0.25     # mass-proportional part (25 %)
        self.beta = 2.0 * zeta / w1 * 0.75      # stiffness-proportional part
        self.damping_dissipation = 0.0
        return f1

    def first_frequency(self, n_modes: int = 2) -> float:
        """Lowest eigenfrequency [cycles/time] from the assembled FE matrices."""
        fr = self.free
        Kr = self.K[np.ix_(fr, fr)]
        Mr = self.M[np.ix_(fr, fr)]
        # dense, small: the generalised eigenproblem M^-1 K is well conditioned
        # here (lumped tip mass keeps M positive definite).  LAPACK's
        # subset-by-index driver is avoided on purpose: it is not portable
        # across the BLAS builds used by the cluster.
        vals = np.linalg.eigvals(np.linalg.solve(Mr, Kr))
        vals = np.sort(np.abs(np.asarray(vals).real))
        vals = vals[vals > 0.0]
        if vals.size < n_modes:                      # pragma: no cover
            raise RuntimeError("degenerate FE spectrum")
        return float(np.sqrt(vals[0]) / (2.0 * np.pi))

    # ------------------------------------------------------------------ dynamics
    def set_attachment(self, nodes: np.ndarray) -> None:
        """Rigid-link patch: the nodes that carry the cylinder (default: tip row)."""
        self.attachment_nodes = np.asarray(nodes, dtype=int)
        self._attachment_dofs = 2 * self.attachment_nodes + 1

    @property
    def attachment_dofs(self) -> np.ndarray:
        return getattr(self, "_attachment_dofs", self.transverse_dofs)

    def tip_displacement(self) -> float:
        return float(np.mean(self.state.u[self.attachment_dofs]))

    def tip_velocity(self) -> float:
        return float(np.mean(self.state.v[self.attachment_dofs]))

    def kinetic(self) -> float:
        v = self.state.v
        return 0.5 * float(v @ self.M @ v)

    def strain(self) -> float:
        u = self.state.u
        return 0.5 * float(u @ self.K @ u)

    def damping_force(self) -> np.ndarray:
        return self.alpha * (self.M @ self.state.v) + self.beta * (self.K @ self.state.v)

    def advance(self, load: np.ndarray, dt: float) -> dict:
        """One implicit Newmark-beta step with a constant nodal load vector.

        ``load`` is a full-length DOF vector (zeros on the constrained DOFs).
        The load is held over the step, which is the standard partition for the
        sub-cycled coupling used in :mod:`tensorphysics.lbm_fem_fsi.coupling`.
        """
        self.dt_fem = float(dt)
        fr = self.free
        M = self.M[np.ix_(fr, fr)]
        K = self.K[np.ix_(fr, fr)]
        Cr = self.alpha * M + self.beta * K
        beta, gamma = 0.25, 0.5
        a0 = 1.0 / (beta * dt * dt)
        a1 = gamma / (beta * dt)
        a2 = 1.0 / (beta * dt)
        a3 = 1.0 / (2.0 * beta) - 1.0
        a4 = gamma / beta - 1.0
        a5 = dt * 0.5 * (gamma / beta - 2.0)
        u = self.state.u[fr].copy()
        v = self.state.v[fr].copy()
        u_old_full = self.state.u.copy()
        a = self.state.a[fr].copy()
        F = load[fr]
        Keff = K + a0 * M + a1 * Cr
        rhs = (F + M @ (a0 * u + a2 * v + a3 * a)
               + Cr @ (a1 * u + a4 * v + a5 * a))
        u_new = np.linalg.solve(Keff, rhs)
        a_new = a0 * (u_new - u) - a2 * v - a3 * a
        v_new = v + dt * ((1.0 - gamma) * a + gamma * a_new)
        # damping dissipation over the step: trapz of c v^2
        c = lambda vv: float(vv @ Cr @ vv)  # noqa: E731
        self.damping_dissipation += 0.5 * (c(v) + c(v_new)) * dt
        self.state.u[fr] = u_new
        self.state.v[fr] = v_new
        self.state.a[fr] = a_new
        # discrete work of a load held constant over the step: F . (u_new - u_old).
        # Using the trapezoidal power instead leaves an O(dt^3) per-step defect
        # that accumulates over 10^4 steps and shows up as a spurious few-percent
        # energy residual in a lightly damped resonant VIV run.
        self.external_work += float(load @ (self.state.u - u_old_full))
        return {
            "u": u_new,
            "v": v_new,
            "kinetic": self.kinetic(),
            "strain": self.strain(),
            "damping": self.damping_dissipation,
            "external_work": self.external_work,
        }

    def modal_amplitude(self, u_hist: np.ndarray) -> float:
        """Peak-to-peak tip displacement of a recorded history."""
        return float(np.max(u_hist) - np.min(u_hist))