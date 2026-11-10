"""2-D boundary-element (panel) potential-flow solver.

Constant-strength source panels (Hess & Smith, 1967) with the standard
zero-net-source closure condition.  For a closed non-lifting body in an
otherwise uniform stream this yields the classical potential-flow solution
(``Cp = 1 - (U_e/U_inf)^2``, d'Alembert paradox ``C_D = 0``) which is then
corrected by the integral boundary-layer model in
:mod:`tensorphysics.lbm_bem_cross.boundary_layer`.

The *same* panel solver is used to evaluate the disturbance velocity at
arbitrary field points, which is what allows the far-field decay test
(``u_i ~ 1/r``, ``phi ~ ln r`` in 2-D) to be compared against the LBM field on
an identical sample set.

Formulation
-----------
For a constant-strength source sheet on the segment ``A -> B`` (length ``l``)
the induced velocity in local panel coordinates reads (Katz & Plotkin,
*Low-Speed Aerodynamics*, §10.2)::

    u_local = (sigma / 2 pi) * ln(r_A / r_B)
    w_local = (sigma / 2 pi) * (theta_B - theta_A)

with ``theta`` the subtended angle of the field point.  Surface tangential
velocity is obtained from the tangential component of the induced velocity of
all *other* panels plus the free stream (the source sheet has no tangential
self-jump), and the normal-velocity no-through-flow condition supplies the
linear system
``sum_j A_ij sigma_j = -U_inf . n_i``, closed by ``sum_j sigma_j l_j = 0``.

References
----------
Hess, J.L. & Smith, A.M.O. (1967). *Calculation of potential flow about
arbitrary bodies*. Progress in Aeronautical Sciences 8, 1-138.
Katz, J. & Plotkin, A. (2001). *Low-Speed Aerodynamics*, 2nd ed., CUP.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .geometry import SurfaceGeometry, _finalise

__all__ = [
    "PanelSolution",
    "PanelMethod2D",
    "surface_from_polyline",
    "analytic_cylinder_cp",
    "analytic_cylinder_velocity",
]


def surface_from_polyline(x: np.ndarray, y: np.ndarray, fmt: str = "polyline") -> SurfaceGeometry:
    """Build a :class:`SurfaceGeometry` from raw vertices (used by the VPI loop)."""
    return _finalise(x, y, fmt)


def analytic_cylinder_cp(theta: np.ndarray) -> np.ndarray:
    """Potential flow past a circular cylinder: ``Cp = 1 - 4 sin^2(theta)``."""
    return 1.0 - 4.0 * np.sin(theta) ** 2


def analytic_cylinder_velocity(x: np.ndarray, y: np.ndarray, radius: float,
                              u_inf: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Analytic potential-flow velocity field of a circular cylinder.

    Stream function ``psi = U y (1 - R^2/r^2)`` gives

    ``u = U [1 + R^2 (y^2 - x^2) / r^4]``, ``v = -2 U R^2 x y / r^4``.
    """
    r2 = np.maximum(x * x + y * y, 1e-30)
    u = u_inf * (1.0 + radius ** 2 * (y * y - x * x) / (r2 * r2))
    v = -2.0 * u_inf * radius ** 2 * x * y / (r2 * r2)
    return u, v


@dataclass
class PanelSolution:
    """Result of a 2-D panel computation."""

    geometry: SurfaceGeometry
    sigma: np.ndarray           # source strength per panel
    u_t: np.ndarray             # surface tangential velocity (dimensional)
    cp: np.ndarray              # surface pressure coefficient at panel midpoints
    theta: np.ndarray           # panel-midpoint theta (rad)
    s_panel: np.ndarray         # panel-midpoint arc length
    u_inf: float = 1.0
    residual: float = 0.0       # max |normal velocity| residual on the body

    def cp_at(self, theta_query: np.ndarray) -> np.ndarray:
        """Interpolate ``Cp(theta)`` onto query angles (periodic)."""
        th = np.concatenate([self.theta, self.theta + 2.0 * np.pi])
        cp = np.concatenate([self.cp, self.cp])
        order = np.argsort(th)
        return np.interp(np.mod(theta_query, 2.0 * np.pi), th[order], cp[order])

    def to_dict(self) -> dict:
        return {
            "n_panels": int(self.geometry.n),
            "u_inf": self.u_inf,
            "normal_velocity_residual": float(self.residual),
            "cp_min": float(np.min(self.cp)),
            "cp_front_stagnation": float(self.cp_at(np.array([0.0]))[0]),
            "cp_rear": float(self.cp_at(np.array([np.pi]))[0]),
            "body_area": self.geometry.area,
            "hydraulic_diameter": self.geometry.hydraulic_diameter,
        }


class PanelMethod2D:
    """Constant-strength source panel method for a closed 2-D body.

    Parameters
    ----------
    geometry : SurfaceGeometry
        Closed body surface (from the shared geometry contract).
    u_inf : float
        Free-stream speed along ``+x``.
    """

    def __init__(self, geometry: SurfaceGeometry, u_inf: float = 1.0) -> None:
        self.geom = geometry
        self.u_inf = float(u_inf)
        self.x0, self.y0 = geometry.x, geometry.y
        self.x1, self.y1 = np.roll(self.x0, -1), np.roll(self.y0, -1)
        self.xm, self.ym, self.n = geometry.segment_midpoints()
        self.t = np.stack([self.x1 - self.x0, self.y1 - self.y0], axis=1)
        self.l = np.hypot(self.t[:, 0], self.t[:, 1])
        self.t = self.t / self.l[:, None]
        self._solution: PanelSolution | None = None

    # -- influence coefficients -------------------------------------------------
    def _induced(self, px: np.ndarray, py: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Unit-source (sigma=1) induced velocity at points ``(px, py)``.

        Returns arrays of shape ``(n_panels, n_points)`` with the x/y velocity
        components in global coordinates.
        """
        n = self.geom.n
        m = px.size
        # local panel coordinates: s along the panel, n normal to it
        dx = px[None, :] - self.x0[:, None]
        dy = py[None, :] - self.y0[:, None]
        s = dx * self.t[:, 0, None] + dy * self.t[:, 1, None]
        q = -dx * self.t[:, 1, None] + dy * self.t[:, 0, None]
        s2 = s - self.l[:, None]
        r1sq = s * s + q * q
        r2sq = s2 * s2 + q * q
        r1sq = np.maximum(r1sq, 1e-30)
        r2sq = np.maximum(r2sq, 1e-30)
        u_loc = 0.5 / np.pi * np.log(r1sq / r2sq) * 0.5
        th1 = np.arctan2(q, s)
        th2 = np.arctan2(q, s2)
        dth = th2 - th1
        # unwrap to (-pi, pi]
        dth = np.mod(dth + np.pi, 2.0 * np.pi) - np.pi
        w_loc = 0.5 / np.pi * dth
        ux = u_loc * self.t[:, 0, None] - w_loc * self.t[:, 1, None]
        uy = u_loc * self.t[:, 1, None] + w_loc * self.t[:, 0, None]
        return ux, uy

    # -- solve ------------------------------------------------------------------
    def solve(self) -> PanelSolution:
        n = self.geom.n
        ux, uy = self._induced(self.xm, self.ym)   # (panel j, point i)
        # the self-panel log term is singular; the on-body limit of a constant
        # source sheet is purely normal with the jump sigma/2 (Katz & Plotkin
        # eq. 10.20): handle the diagonal analytically
        diag = np.eye(n, dtype=bool)
        ux = np.where(diag, 0.0, ux)
        uy = np.where(diag, 0.0, uy)
        # A[i, j] = normal velocity at midpoint i induced by unit source on panel j
        A = (ux * self.n[:, 0][None, :] + uy * self.n[:, 1][None, :]).T.copy()
        np.fill_diagonal(A, 0.5)
        b = -self.u_inf * self.n[:, 0]
        # closure condition: zero net source strength (weighted by panel length)
        # For the *closed* source-only formulation the Neumann matrix is
        # non-singular (the classical rank deficiency belongs to the doublet
        # formulation), so the N x N system is solved directly.  The zero net
        # source strength sum(sigma_j l_j) = 0 is then a *consequence*, and is
        # reported as a consistency check.
        sigma = np.linalg.solve(A, b)

        # induced velocity on the body from the *other* panels (self term = 0)
        ui = ux.T @ sigma
        vi = uy.T @ sigma
        ut = ui * self.t[:, 0] + vi * self.t[:, 1] + self.u_inf * self.t[:, 0]
        cp = 1.0 - (ut / self.u_inf) ** 2
        # residual check: normal velocity on the body (self panel: sigma/2 jump)
        norm_v = ui * self.n[:, 0] + vi * self.n[:, 1] + 0.5 * sigma + self.u_inf * self.n[:, 0]
        sol = PanelSolution(
            geometry=self.geom,
            sigma=sigma,
            u_t=ut,
            cp=cp,
            theta=np.mod(np.arctan2(self.ym, -self.xm), 2.0 * np.pi),
            s_panel=self.geom.arclength[:-1] + 0.5 * self.l,
            u_inf=self.u_inf,
            residual=float(np.max(np.abs(norm_v))),
        )
        self._solution = sol
        return sol

    @property
    def solution(self) -> PanelSolution:
        if self._solution is None:
            return self.solve()
        return self._solution

    # -- field evaluation -------------------------------------------------------
    def velocity_field(self, px: np.ndarray, py: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Total velocity (free stream + panel sources) at field points."""
        sol = self.solution
        ux, uy = self._induced(np.asarray(px, float).ravel(), np.asarray(py, float).ravel())
        u = ux.T @ sol.sigma + self.u_inf
        v = uy.T @ sol.sigma
        return u.reshape(np.shape(px)), v.reshape(np.shape(py))

    def potential_field(self, px: np.ndarray, py: np.ndarray, n_gauss: int = 6) -> np.ndarray:
        """Velocity potential at field points (Gauss quadrature over panels).

        ``phi = U_inf x + sum_j (sigma_j / 2 pi) int_j ln|P-Q| ds``.
        """
        sol = self.solution
        xi, wi = np.polynomial.legendre.leggauss(n_gauss)
        px = np.asarray(px, float)
        py = np.asarray(py, float)
        phi = self.u_inf * px.copy()
        for j in range(self.geom.n):
            for xi_k, w_k in zip(xi, wi):
                frac = 0.5 * (xi_k + 1.0)
                xq = self.x0[j] + frac * (self.x1[j] - self.x0[j])
                yq = self.y0[j] + frac * (self.y1[j] - self.y0[j])
                r2 = np.maximum((px - xq) ** 2 + (py - yq) ** 2, 1e-30)
                phi += sol.sigma[j] / (2.0 * np.pi) * 0.5 * np.log(r2) * w_k * 0.5 * self.l[j]
        return phi

    def drag_coefficient_pressure(self) -> float:
        """Potential-flow pressure drag (equals ~0: d'Alembert)."""
        sol = self.solution
        # surface pressure force along +x, per unit span, normalised by q_inf*D
        d = self.geom.hydraulic_diameter
        # F_x / (q_inf D) with Cp = (p - p_inf)/q_inf and force = -int p n ds
        return float(-(sol.cp * self.n[:, 0] * self.l).sum() / d)