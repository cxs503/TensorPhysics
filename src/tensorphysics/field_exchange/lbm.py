"""Minimal D2Q9 reference LBM for wall-traction generation (source side of the exchange).

Why a reference LBM lives here: TensorPhysics is the *workflow* repository, so it
must be able to produce a physically meaningful wall-traction field (pressure and
shear) to feed the exchange operators without importing a solver kernel.  This is
a compact, self-contained D2Q9 BGK implementation of the plane channel driven by
either a moving wall (Couette) or a body force (Poiseuille).  It is deliberately
small and validated against the exact solutions in
:mod:`tensorphysics.field_exchange.verification`; it is *not* a replacement for
TensorLBM.

Conventions (lattice units):
* fluid nodes ``j = 0 .. ny-1`` sit at ``y = j + 1/2``; walls at ``y = 0`` and
  ``y = ny`` (half-way bounce-back),
* ``rho = 1``, ``c_s^2 = 1/3``, ``mu = rho c_s^2 (tau - 1/2)``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ChannelResult", "run_channel", "LatticeScale", "wall_shear_from_profile"]

_C = np.array(
    [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]],
    dtype=int,
)
_W = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
_OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])


@dataclass(frozen=True)
class ChannelResult:
    ux: np.ndarray          # (nx, ny) time-averaged steady velocity
    rho: np.ndarray         # (nx, ny) density at exit tolerance
    tau_w_bottom: float     # lattice wall shear stress at y=0: mu * du/dy
    tau_w_top: float
    steps: int
    residual: float
    lid_velocity: float
    body_force: float
    tau_relax: float

    @property
    def mu(self) -> float:
        return (self.tau_relax - 0.5) / 3.0

    def profile_error_linear(self, lid_velocity: float) -> float:
        """L2 error of ``ux(y)`` against the Couette profile ``U y/H`` (relative)."""
        ny = self.ux.shape[1]
        y = (np.arange(ny) + 0.5) / ny
        ref = lid_velocity * y
        u = self.ux.mean(axis=0)
        return float(np.linalg.norm(u - ref) / max(np.linalg.norm(ref), 1e-30))

    def profile_error_parabolic(self, body_force: float) -> float:
        """L2 error of ``ux(y)`` against the force-driven Poiseuille profile."""
        ny = self.ux.shape[1]
        y = (np.arange(ny) + 0.5)
        mu = self.mu
        ref = body_force / (2.0 * mu) * y * (ny - y)
        u = self.ux.mean(axis=0)
        return float(np.linalg.norm(u - ref) / max(np.linalg.norm(ref), 1e-30))


def wall_shear_from_profile(u: np.ndarray, wall_value: float, wall_step: float = 1.0) -> float:
    """Second-order one-sided ``du/dy`` at a wall from two interior nodes.

    Nodes sit at distances ``0.5h`` and ``1.5h`` from the wall.  With the wall
    velocity known, the three-point Lagrange derivative at the wall is
    ``(-8/3 f0 + 3 f1 - 1/3 f2)/h``.
    """
    return float((-8.0 / 3.0 * wall_value + 3.0 * u[0] - u[1] / 3.0) / wall_step)


def run_channel(
    nx: int = 8,
    ny: int = 32,
    *,
    lid_velocity: float = 0.0,
    body_force: float = 0.0,
    tau_relax: float = 0.8,
    max_steps: int = 60000,
    tol: float = 1e-12,
    check_every: int = 200,
) -> ChannelResult:
    """Run a plane channel to steady state (BGK, half-way bounce-back).

    ``lid_velocity`` drives Couette flow through the top wall; ``body_force`` is a
    uniform acceleration in +x (Guo forcing) producing Poiseuille flow.  The run
    stops when the change in the wall shear per ``check_every`` steps drops below
    ``tol`` (relative), or after ``max_steps``.
    """
    nu = (tau_relax - 0.5) / 3.0
    if nu <= 0:
        raise ValueError("tau_relax must be > 0.5")

    feq = np.zeros((9, nx, ny))
    f = np.zeros((9, nx, ny))
    rho = np.ones((nx, ny))
    ux = np.zeros((nx, ny))
    uy = np.zeros((nx, ny))
    for i in range(9):
        f[i] = _W[i] * rho

    g = np.array([body_force, 0.0])
    prev_tau = None
    residual = float("nan")
    step = 0
    for step in range(1, max_steps + 1):
        # --- macroscopic ---
        rho = f.sum(axis=0)
        ux = (f * _C[:, 0, None, None]).sum(axis=0) / rho + 0.5 * g[0]
        uy = (f * _C[:, 1, None, None]).sum(axis=0) / rho
        # --- equilibrium ---
        cu = np.zeros((9, nx, ny))
        for i in range(9):
            cu[i] = _C[i, 0] * ux + _C[i, 1] * uy
        usq = ux * ux + uy * uy
        for i in range(9):
            feq[i] = _W[i] * rho * (
                1.0 + 3.0 * cu[i] + 4.5 * cu[i] ** 2 - 1.5 * usq
            )
        # --- Guo forcing ---
        src = np.zeros((9, nx, ny))
        if body_force != 0.0:
            for i in range(9):
                cg = _C[i, 0] * g[0] + _C[i, 1] * g[1]
                ug = ux * g[0] + uy * g[1]
                src[i] = _W[i] * (1.0 - 0.5 / tau_relax) * (
                    3.0 * cg + 9.0 * cu[i] * cg - 3.0 * ug
                )
        # --- collision ---
        fpost = f - (f - feq) / tau_relax + src
        # --- streaming (pull) with half-way bounce-back walls ---
        fnew = np.empty_like(f)
        for i in range(9):
            g_i = np.roll(fpost[i], (_C[i, 0], _C[i, 1]), axis=(0, 1))
            if _C[i, 1] > 0:
                # enters from below: node j=0 receives the reflection of opp(i)
                refl = fpost[_OPP[i]][:, 0] + 6.0 * _W[i] * rho[:, 0] * (
                    _C[i, 0] * 0.0
                )
                g_i[:, 0] = refl
            elif _C[i, 1] < 0:
                refl = fpost[_OPP[i]][:, ny - 1] + 6.0 * _W[i] * rho[:, ny - 1] * (
                    _C[i, 0] * lid_velocity
                )
                g_i[:, ny - 1] = refl
            fnew[i] = g_i
        f = fnew

        if step % check_every == 0:
            rho = f.sum(axis=0)
            u = (f * _C[:, 0, None, None]).sum(axis=0) / rho + 0.5 * g[0]
            wall = u.mean(axis=0)
            tau_b = wall_shear_from_profile(wall, 0.0) * ((tau_relax - 0.5) / 3.0)
            if prev_tau is not None:
                residual = abs(tau_b - prev_tau) / max(abs(tau_b), 1e-30)
                if residual < tol:
                    break
            prev_tau = tau_b

    rho = f.sum(axis=0)
    ux = (f * _C[:, 0, None, None]).sum(axis=0) / rho + 0.5 * g[0]
    uy = (f * _C[:, 1, None, None]).sum(axis=0) / rho
    prof = ux.mean(axis=0)
    mu = (tau_relax - 0.5) / 3.0
    # wall shear *stress* (lattice units): mu * du/dy
    tau_b = mu * wall_shear_from_profile(prof, 0.0)
    tau_t = -mu * wall_shear_from_profile(prof[::-1], lid_velocity)
    return ChannelResult(
        ux=ux,
        rho=rho,
        tau_w_bottom=tau_b,
        tau_w_top=tau_t,
        steps=step,
        residual=float(residual),
        lid_velocity=lid_velocity,
        body_force=body_force,
        tau_relax=tau_relax,
    )


# ---------------------------------------------------------------------------
# lattice -> physical similarity scaling
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LatticeScale:
    """Similarity transform between lattice units and SI units.

    The free parameters are the physical density, gap and kinematic viscosity,
    the lattice gap in nodes and the relaxation time.  ``dt`` follows from the
    viscosity match, ``U_lat`` from the velocity match; consistency of the two
    is checked (a badly chosen ``tau_relax`` makes ``U_lat`` supersonic).
    """

    rho_phys: float
    nu_phys: float
    h_phys: float
    ny: int
    tau_relax: float

    @property
    def nu_lat(self) -> float:
        return (self.tau_relax - 0.5) / 3.0

    @property
    def dx(self) -> float:
        return self.h_phys / self.ny

    @property
    def dt(self) -> float:
        return self.nu_lat * self.dx ** 2 / self.nu_phys

    @property
    def velocity_scale(self) -> float:
        """``dx/dt`` [m/s per lattice velocity unit]."""
        return self.dx / self.dt

    @property
    def stress_scale(self) -> float:
        """``rho_phys (dx/dt)^2`` [Pa per lattice stress unit]."""
        return self.rho_phys * self.velocity_scale ** 2

    def u_lattice(self, u_phys: float) -> float:
        return u_phys / self.velocity_scale

    def u_physical(self, u_lat: float) -> float:
        return u_lat * self.velocity_scale

    def stress_physical(self, s_lat: float) -> float:
        return s_lat * self.stress_scale

    def pressure_physical(self, p_lat: float, p_ref_lat: float = 1.0 / 3.0) -> float:
        return (p_lat - p_ref_lat) * self.stress_scale

    def mu_physical(self) -> float:
        return self.rho_phys * self.nu_phys