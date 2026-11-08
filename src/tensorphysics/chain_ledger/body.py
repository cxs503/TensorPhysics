"""Six-degree-of-freedom floating box: geometry, panels, hydrostatics, dynamics.

The body is a rectangular box (barge section extruded across the flume width).
It is the *middle* stage of the chain: it receives the fluid force through
interface A, returns its motion/velocity through interface B, and its relative
velocity drives the impact pressure handed to interface C.

Evidence levels
---------------
REAL      panel geometry, rigid-body kinematics, hydrostatic restoring from
          waterplane integrals (cross-checked against a numerical pressure
          integration), RK4 time marching, the linear frequency-domain
          steady state (cross-checked against the RK4 march).
MODEL-INPUT  added mass ``A`` and radiation damping ``B`` are *inputs* to the
          chain.  ``A`` is taken from the repository's 2-D boundary-value
          sectional solver (``hull_hydroelastic.sections``), i.e. a real
          infinite-frequency added mass; ``B`` is a documented viscous/radiation
          surrogate because the radiating free-surface Green function is not
          solved in this repository (see ``hull_hydroelastic`` scope notes).
          No claim is made about the absolute accuracy of ``B``; the chain gates
          only require that *whatever is handed over is conserved*.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["BoxBody", "RigidPose", "Panels", "box_panels", "rotation_matrix"]

G_ACC = 9.80665


def rotation_matrix(theta: np.ndarray) -> np.ndarray:
    """Rodrigues rotation matrix from a rotation vector ``theta`` (rad)."""
    th = np.asarray(theta, dtype=float).reshape(3)
    angle = float(np.linalg.norm(th))
    if angle < 1.0e-15:
        return np.eye(3)
    k = th / angle
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


@dataclass(frozen=True)
class RigidPose:
    """Rigid-body pose: translation of the origin + rotation vector."""

    r: np.ndarray = field(default_factory=lambda: np.zeros(3))
    theta: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @property
    def R(self) -> np.ndarray:
        return rotation_matrix(self.theta)

    def to_world(self, x_body: np.ndarray) -> np.ndarray:
        return np.asarray(x_body, dtype=float) @ self.R.T + np.asarray(self.r, dtype=float).reshape(1, 3)


@dataclass(frozen=True)
class Panels:
    """Wetted panel set: centroids, outward unit normals, areas."""

    centroid: np.ndarray   # (n, 3)
    normal: np.ndarray     # (n, 3)
    area: np.ndarray       # (n,)
    tag: np.ndarray        # (n,) int: 0 bottom, 1 side +x, 2 side -x, 3 end +y, 4 end -y, 5 top
    tangent1: np.ndarray   # (n, 3) half-extent vector of the panel, first in-plane axis
    tangent2: np.ndarray   # (n, 3) half-extent vector of the panel, second in-plane axis

    def __len__(self) -> int:
        return int(self.area.size)

    @property
    def total_area(self) -> float:
        return float(np.sum(self.area))


@dataclass(frozen=True)
class BoxBody:
    """Rectangular floating box (half-beam ``b``, draught ``d``, width ``W``)."""

    half_beam: float = 0.5
    draught: float = 0.3
    width: float = 1.0
    mass: float = 600.0
    depth: float = 0.6
    rho: float = 1025.0
    g: float = G_ACC

    @property
    def displacement(self) -> float:
        """Displaced water mass at the design draught [kg]."""
        return self.rho * self.volume

    @property
    def volume(self) -> float:
        """Immersed volume at the design draught [m^3]."""
        return 2.0 * self.half_beam * self.draught * self.width

    @property
    def mass_ratio(self) -> float:
        """``m / (rho nabla)``; < 1 means the box floats light and is held down."""
        return float(self.mass / self.displacement)

    # ---------------------------------------------------------- inertia
    @property
    def inertia(self) -> np.ndarray:
        """Principal inertia of the (solid) box about its centroid, (3, 3)."""
        lx, ly, lz = 2.0 * self.half_beam, self.width, self.depth
        m = self.mass
        ixx = m * (ly * ly + lz * lz) / 12.0
        iyy = m * (lx * lx + lz * lz) / 12.0
        izz = m * (lx * lx + ly * ly) / 12.0
        return np.diag([ixx, iyy, izz])

    @property
    def mass_matrix(self) -> np.ndarray:
        """6x6 rigid-body mass matrix (DOF order ``u, v, w, p, q, r``)."""
        M = np.zeros((6, 6))
        M[0:3, 0:3] = np.eye(3) * self.mass
        M[3:6, 3:6] = self.inertia
        return M

    # -------------------------------------------------------- hydrostatics
    @property
    def waterplane_area(self) -> float:
        return 2.0 * self.half_beam * self.width

    @property
    def waterplane_inertia_pitch(self) -> float:
        """Second moment of the waterplane about the ``y`` axis [m^4]."""
        return self.width * (2.0 * self.half_beam) ** 3 / 12.0

    @property
    def waterplane_inertia_roll(self) -> float:
        """Second moment of the waterplane about the ``x`` axis [m^4]."""
        return (2.0 * self.half_beam) * self.width ** 3 / 12.0

    @property
    def restoring_matrix(self) -> np.ndarray:
        """Linear hydrostatic restoring matrix, DOF order ``u, v, w, p, q, r``."""
        K = np.zeros((6, 6))
        K[2, 2] = self.rho * self.g * self.waterplane_area
        K[3, 3] = self.rho * self.g * self.waterplane_inertia_roll
        K[4, 4] = self.rho * self.g * self.waterplane_inertia_pitch
        return K

    def hydrostatic_pressure(self, z: np.ndarray) -> np.ndarray:
        """Gauge hydrostatic pressure [Pa] at world height ``z`` (``z`` up)."""
        return -self.rho * self.g * np.asarray(z, dtype=float)

    # ------------------------------------------------------------ panels
    def panels(self, pose: RigidPose | None = None, n_edge: int = 8, wetted: bool = True) -> Panels:
        """Panels at ``pose`` (``None`` = design pose, waterplane at z = 0)."""
        pose = RigidPose() if pose is None else pose
        return box_panels(self, pose, n_edge=n_edge, wetted=wetted)


def box_panels(
    body: BoxBody,
    pose: RigidPose,
    n_edge: int = 8,
    wetted: bool = True,
) -> Panels:
    """Panels of the box.

    ``wetted=True`` returns the wetted surface only (bottom + four sides, panels
    whose centroid rises above the still-water plane dropped).  ``wetted=False``
    returns the *closed* six-face surface, which is what the rigid-body
    zero-flux identity ``oint u.n dA = 0`` needs.

    Each panel also carries two half-extent tangent vectors so that a genuine
    Gauss quadrature (not only the piecewise-constant boundary-node hold) can be
    evaluated on it.
    """
    if n_edge < 1:
        raise ValueError("n_edge must be >= 1")
    b, d, w = body.half_beam, body.draught, body.width
    top = body.depth - body.draught          # deck height above the SWL
    R = pose.R
    r = np.asarray(pose.r, dtype=float).reshape(3)

    cent, nrm, area, tag, t1s, t2s = [], [], [], [], [], []

    def grid(u0, u1, n):
        e = np.linspace(u0, u1, int(n) + 1)
        return 0.5 * (e[:-1] + e[1:]), float(e[1] - e[0])

    def append_face(pts_body, normal_body, a_panel, t, t1_body, t2_body):
        p_w = R @ np.asarray(pts_body, dtype=float) + r
        if wetted and p_w[2] > 1.0e-12:
            return
        cent.append(p_w)
        nrm.append(R @ np.asarray(normal_body, dtype=float))
        area.append(float(a_panel))
        tag.append(t)
        t1s.append(R @ np.asarray(t1_body, dtype=float))
        t2s.append(R @ np.asarray(t2_body, dtype=float))

    xc, dx = grid(-b, b, n_edge)
    yc, dy = grid(-w / 2.0, w / 2.0, n_edge)
    # the vertical grid is built from *two* blocks so that the still-water plane
    # z = 0 always falls on a panel edge: the faceted waterline then contributes
    # no error and the quadrature-order study is clean.
    z_blocks = [(grid(-d, 0.0, n_edge)), (grid(0.0, top, n_edge))] if top > 0.0 else [grid(-d, 0.0, n_edge)]
    hx, hy = 0.5 * dx, 0.5 * dy

    # bottom  z = -d  (inward-positive z is up, so the outward normal is -z)
    for x0 in xc:
        for y0 in yc:
            append_face([x0, y0, -d], (0.0, 0.0, -1.0), dx * dy, 0, (hx, 0.0, 0.0), (0.0, hy, 0.0))
    # sides  x = +-b   (spanning the full box height, waterline on a panel edge)
    for sign, t in ((1.0, 1), (-1.0, 2)):
        for y0 in yc:
            for zc, dz in z_blocks:
                for z0 in zc:
                    append_face([sign * b, y0, z0], (sign, 0.0, 0.0), dy * dz, t,
                                (0.0, hy, 0.0), (0.0, 0.0, 0.5 * dz))
    # ends  y = +-w/2
    for sign, t in ((1.0, 3), (-1.0, 4)):
        for x0 in xc:
            for zc, dz in z_blocks:
                for z0 in zc:
                    append_face([x0, sign * w / 2.0, z0], (0.0, sign, 0.0), dx * dz, t,
                                (hx, 0.0, 0.0), (0.0, 0.0, 0.5 * dz))
    # deck  z = +top (only for the closed surface)
    if not wetted:
        for x0 in xc:
            for y0 in yc:
                append_face([x0, y0, top], (0.0, 0.0, 1.0), dx * dy, 5, (hx, 0.0, 0.0), (0.0, hy, 0.0))

    cents = np.asarray(cent, dtype=float)
    norms = np.asarray(nrm, dtype=float)
    areas = np.asarray(area, dtype=float)
    if cents.shape[0] != areas.size:  # pragma: no cover - internal consistency
        raise RuntimeError("panel bookkeeping mismatch")
    return Panels(
        centroid=cents,
        normal=norms,
        area=areas,
        tag=np.asarray(tag, dtype=int),
        tangent1=np.asarray(t1s, dtype=float),
        tangent2=np.asarray(t2s, dtype=float),
    )


# --------------------------------------------------------------------------
# Rigid-body kinematics helpers
# --------------------------------------------------------------------------
def rigid_velocity(pose: RigidPose, v: np.ndarray, omega: np.ndarray, x_world: np.ndarray) -> np.ndarray:
    """Velocity of material points: ``U + omega x (x - r)``."""
    v = np.asarray(v, dtype=float).reshape(3)
    omega = np.asarray(omega, dtype=float).reshape(3)
    rel = np.asarray(x_world, dtype=float) - np.asarray(pose.r, dtype=float).reshape(1, 3)
    return v[None, :] + np.cross(omega[None, :], rel)


def force_moment(panels: Panels, pressure: np.ndarray, origin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Integrate ``-p n dA`` over the wetted surface into a resultant force/moment.

    Sign convention: ``n`` is the outward normal of the *solid*, so the force of
    the fluid on the solid is ``-p n`` (pressure pushes the solid inward).
    """
    p = np.asarray(pressure, dtype=float)
    dA = panels.area
    f_panel = -(p * dA)[:, None] * panels.normal
    F = f_panel.sum(axis=0)
    arm = panels.centroid - np.asarray(origin, dtype=float).reshape(1, 3)
    M = np.cross(arm, f_panel).sum(axis=0)
    return F, M


# --------------------------------------------------------------------------
# Time marching of the 6-DOF body in a prescribed wave field
# --------------------------------------------------------------------------
@dataclass
class BodyDynamics:
    """Linear 6-DOF dynamics with added mass and damping supplied as inputs.

    ``M + A`` multiplies the acceleration, the fluid force (interface A) drives
    the right-hand side, ``-B v`` is the radiation/viscous sink and ``-K x`` the
    hydrostatic restoring.  The radiation-reaction form is what makes the
    energy identity ``Re{F conj(v)} = v^T B v`` hold exactly in the periodic
    steady state, which is the physical content of the chain energy ledger.
    """

    body: BoxBody
    added_mass: np.ndarray
    damping: np.ndarray
    gravity_restoring: bool = True
    #: DOF held by the carriage/mooring (default: horizontal-plane motions are
    #: locked, which is what a flume model mounted on a carriage sees and what
    #: keeps the drift-free surge response out of the energy ledger).
    locked_dofs: tuple = (0, 1, 3, 5)

    def __post_init__(self) -> None:
        self.added_mass = np.asarray(self.added_mass, dtype=float).reshape(6, 6)
        self.damping = np.asarray(self.damping, dtype=float).reshape(6, 6)
        self.locked_dofs = tuple(int(i) for i in self.locked_dofs)

    @property
    def effective_mass(self) -> np.ndarray:
        return self.body.mass_matrix + self.added_mass

    @property
    def restoring(self) -> np.ndarray:
        return self.body.restoring_matrix if self.gravity_restoring else np.zeros((6, 6))

    def acceleration(self, state: np.ndarray, force: np.ndarray) -> np.ndarray:
        """Solve ``(M+A) a = F - B v - K x`` for the 6-DOF acceleration."""
        x, v = state[0:6], state[6:12]
        rhs = np.asarray(force, dtype=float) - self.damping @ v - self.restoring @ x
        acc = np.linalg.solve(self.effective_mass, rhs)
        if self.locked_dofs:
            acc[list(self.locked_dofs)] = 0.0
        return acc

    def march(self, state: np.ndarray, dt: float, force_fn) -> np.ndarray:
        """One RK4 step with ``force_fn(t, state) -> F`` (6-vector)."""
        def f(t, s):
            F = np.asarray(force_fn(t, s), dtype=float)
            return np.concatenate([s[6:12], self.acceleration(s, F)])

        k1 = f(0.0, state)
        y2 = state + 0.5 * dt * k1
        k2 = f(0.5 * dt, y2)
        y3 = state + 0.5 * dt * k2
        k3 = f(0.5 * dt, y3)
        y4 = state + dt * k3
        k4 = f(dt, y4)
        return state + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # -------------------------------------------------------- steady state
    def harmonic_steady_state(self, force_amplitude: complex, omega: float) -> tuple[np.ndarray, np.ndarray]:
        """Complex 6-DOF response ``x_hat`` to ``F = Re{F_hat e^{i omega t}}``.

        Returns ``(x_hat, v_hat)`` with ``v_hat = i omega x_hat``.  Used both to
        initialise the march at the periodic steady state and as a benchmark for
        the time-domain integration.
        """
        F = np.asarray(force_amplitude, dtype=complex).reshape(6).astype(complex)
        K = self.restoring
        Z = (K - omega ** 2 * self.effective_mass + 1j * omega * self.damping).astype(complex)
        for dof in self.locked_dofs:
            Z[dof, :] = 0.0
            Z[:, dof] = 0.0
            Z[dof, dof] = 1.0
            F[dof] = 0.0
        x_hat = np.linalg.solve(Z, F)
        return x_hat, 1j * omega * x_hat

    def energy(self, state: np.ndarray) -> dict:
        """Kinetic (incl. added mass), hydrostatic and total energy of the body."""
        x, v = state[0:6], state[6:12]
        ke = 0.5 * float(v @ self.effective_mass @ v)
        pe = 0.5 * float(x @ self.restoring @ x)
        return {"kinetic_J": ke, "potential_J": pe, "total_J": ke + pe}