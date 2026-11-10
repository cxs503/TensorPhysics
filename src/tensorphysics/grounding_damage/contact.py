"""Axisymmetric elastic half-space contact solver (Boussinesq / Love kernel).

The vertical surface displacement produced by a uniform pressure ``p`` on an
annulus ``[a_in, a_out]`` is obtained by integrating the Boussinesq point-load
kernel ``1 / (pi E* rho)`` over the annulus.  With the standard angular
reduction

.. math:: \\int_0^{2\\pi} \\frac{d\\varphi}{|\\mathbf{r}-\\mathbf{r}'|}
        = \\frac{4 K(m)}{r + r'}, \\qquad
        m = \\frac{4 r r'}{(r + r')^2}

(:math:`K` = complete elliptic integral of the first kind) the annular
integration reduces to a one-dimensional Gauss quadrature in the radial
direction, which is accurate to a few per cent even for the self term (the
kernel has only a logarithmic singularity).

The solver is used to verify the contact algorithm against the closed-form
Hertz solution (acceptance criterion C1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.special import ellipk

__all__ = ["HalfSpaceCompliance", "HertzContact", "hertz_sphere_on_plane",
           "HertzSolution"]


@dataclass(frozen=True)
class HertzSolution:
    """Closed-form Hertz contact quantities (sphere on a plane half-space)."""

    load: float
    radius: float
    peak_pressure: float
    mean_pressure: float
    approach: float
    e_star: float


def hertz_sphere_on_plane(load: float, R: float, E1: float, nu1: float,
                          E2: Optional[float] = None,
                          nu2: float = 0.3) -> HertzSolution:
    """Classical Hertz solution for a sphere of radius ``R`` on a half-space."""
    e_inv = (1.0 - nu1**2) / E1
    if E2 is not None:
        e_inv += (1.0 - nu2**2) / E2
    e_star = 1.0 / e_inv
    a = (3.0 * load * R / (4.0 * e_star)) ** (1.0 / 3.0)
    return HertzSolution(
        load=load, radius=a,
        peak_pressure=3.0 * load / (2.0 * np.pi * a**2),
        mean_pressure=load / (np.pi * a**2),
        approach=a**2 / R, e_star=e_star,
    )


class HalfSpaceCompliance:
    """Ring-averaged elastic half-space compliance on a radial grid."""

    def __init__(self, r_edges: np.ndarray, n_sub: int = 6, n_gauss: int = 10) -> None:
        self.r = np.asarray(r_edges, dtype=float)
        self.r_mid = 0.5 * (self.r[1:] + self.r[:-1])
        self.n_sub = int(n_sub)
        self.n_gauss = int(n_gauss)

    def matrix(self) -> np.ndarray:
        """Compliance matrix ``C[i, j]`` in m/Pa (unit pressure on ring ``j``)."""
        n = self.r_mid.size
        xg, wg = np.polynomial.legendre.leggauss(self.n_gauss)
        C = np.zeros((n, n))
        for j in range(n):
            edges = np.linspace(self.r[j], self.r[j + 1], self.n_sub + 1)
            for s in range(self.n_sub):
                lo, hi = edges[s], edges[s + 1]
                half = 0.5 * (hi - lo)
                rc = 0.5 * (hi + lo) + half * xg
                wr = half * wg
                denom = self.r_mid[:, None] + rc[None, :]
                m = np.clip(4.0 * self.r_mid[:, None] * rc[None, :] / denom**2,
                            0.0, 1.0 - 1e-12)
                integrand = 4.0 * ellipk(m) / denom * rc[None, :]
                C[:, j] += (integrand * wr[None, :]).sum(axis=1)
        return C / np.pi


class HertzContact:
    """Discrete axisymmetric sphere-on-half-space contact solver.

    The contact problem is the linear complementarity problem

    ``C p + gap >= delta``,  ``p >= 0``,  ``p * (C p + gap - delta) = 0``

    with the paraboloid gap ``gap = r^2 / (2R)``; it is solved by an active-set
    iteration.  ``C`` is the ring compliance of :class:`HalfSpaceCompliance`
    divided by the effective modulus ``E*``.
    """

    def __init__(self, R: float, E: float, nu: float, n_rings: int = 120,
                 n_sub: int = 6, n_gauss: int = 10, grid_scale: float = 2.0) -> None:
        self.R = float(R)
        self.E = float(E)
        self.nu = float(nu)
        self.e_star = self.E / (1.0 - self.nu**2)
        self.n_rings = int(n_rings)
        self.n_sub = int(n_sub)
        self.n_gauss = int(n_gauss)
        self.grid_scale = float(grid_scale)

    def _grid(self, delta: float):
        a_max = self.grid_scale * np.sqrt(2.0 * self.R * max(delta, 1e-12))
        r = np.linspace(0.0, a_max, self.n_rings + 1)
        return r, HalfSpaceCompliance(r, self.n_sub, self.n_gauss).matrix() / self.e_star

    def solve(self, delta: float) -> Dict[str, np.ndarray | float]:
        """Solve the normal contact problem for the rigid approach ``delta``."""
        r, C = self._grid(delta)
        r_mid = 0.5 * (r[1:] + r[:-1])
        area = 2.0 * np.pi * r_mid * np.diff(r)
        rhs = delta - r_mid**2 / (2.0 * self.R)
        active = rhs > 0.0
        p = np.zeros_like(r_mid)
        tol = 1e-6 * max(delta, 1e-12)
        for _ in range(400):
            p = np.zeros_like(r_mid)
            if np.any(active):
                sub = np.ix_(active, active)
                try:
                    p[active] = np.linalg.solve(C[sub], rhs[active])
                except np.linalg.LinAlgError:
                    p[active] = np.linalg.lstsq(C[sub], rhs[active], rcond=None)[0]
            neg = active & (p <= 0.0)
            p = np.where(p > 0.0, p, 0.0)
            pen = rhs - C @ p
            pen_new = (~active) & (pen > tol)
            if not np.any(neg) and not np.any(pen_new):
                break
            active = (active & ~neg) | pen_new
        load = float(np.sum(p * area))
        if np.any(p > 0.0):
            j_last = int(np.flatnonzero(p > 0.0)[-1])
            a_c = float(r[j_last + 1])            # outer edge of the loaded ring
        else:
            a_c = 0.0
        return {
            "r": r_mid, "pressure": p, "load": load,
            "contact_radius": a_c,
            "peak_pressure": float(p.max()) if p.size else 0.0,
            "delta": float(delta),
            "contact_area": float(np.sum(area[p > 0.0])),
        }

    def load_curve(self, deltas: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        d = np.asarray(deltas, dtype=float)
        return d, np.asarray([self.solve(x)["load"] for x in d], dtype=float)

# ---------------------------------------------------------------------------
# Ice ridge (rigid wedge) and shoal (elastic sand bed) contact models
# ---------------------------------------------------------------------------
@dataclass
class RidgeGeometry:
    """Rigid ice-ridge keel idealised as a symmetric wedge.

    Parameters
    ----------
    h_ridge:
        Keel depth of the ridge measured from the *undamaged* ship bottom plane
        (m), i.e. the vertical height that must be penetrated.
    slope_deg:
        Flank slope (angle between ridge flank and the bottom plane).
    length:
        Longitudinal length of the ridge in contact with the bottom (m).
    crushing_pressure:
        Optional ridge crushing pressure (Pa).  ``None`` = rigid ridge (as
        specified by the campaign node); a finite value splits the absorbed
        energy between hull deformation and ice crushing (sensitivity study).
    """

    h_ridge: float
    slope_deg: float
    length: float = 30.0
    crushing_pressure: Optional[float] = None

    @property
    def slope_rad(self) -> float:
        return np.deg2rad(self.slope_deg)

    def penetration(self, y: np.ndarray | float, apex_depth: float,
                    y_apex: float = 0.0) -> np.ndarray:
        """Local bottom penetration ``max(0, apex - |y - y_apex| tan(alpha))``."""
        y = np.asarray(y, dtype=float)
        raw = apex_depth - np.abs(y - y_apex) * np.tan(self.slope_rad)
        return np.clip(np.minimum(raw, self.h_ridge), 0.0, None)

    def contact_half_width(self, apex_depth: float) -> float:
        d = min(float(apex_depth), self.h_ridge)
        return 0.0 if d <= 0.0 else float(d / np.tan(self.slope_rad))


@dataclass
class ShoalContact:
    """Elastic sand bed (shoal): ``p = min(k_s delta, q_ult)``."""

    sand_stiffness: float = 2.0e8     # Pa/m (medium dense sand)
    bearing_capacity: float = 1.0e6   # Pa (Terzaghi-type ultimate bearing)
    slope_deg: float = 10.0
    length: float = 30.0

    def pressure(self, delta: np.ndarray | float) -> np.ndarray:
        d = np.asarray(delta, dtype=float)
        return np.minimum(self.sand_stiffness * np.clip(d, 0.0, None), self.bearing_capacity)

    def penetration(self, x: np.ndarray | float, drag: float,
                    alpha: Optional[float] = None) -> np.ndarray:
        """Penetration of a sloping seabed along the ship length (m)."""
        a = np.deg2rad(self.slope_deg if alpha is None else alpha)
        x = np.asarray(x, dtype=float)
        return np.clip(drag - x * np.tan(a), 0.0, None)


class RidgeContact:
    """Ridge/panel contact on a non-linear Winkler foundation.

    The bottom is represented by ``n_panels`` identical plate panels of breadth
    ``panel_breadth``, each with the pressure-deflection relation
    ``q(delta)`` supplied by the verified von Karman panel model
    (:class:`tensorphysics.grounding_damage.panel.VonKarmanPanel`).  The
    contact pressure along the ridge flank then follows from the geometry:

    ``R(apex) = length * sum_i b_i * q(delta_i)``
    """

    def __init__(self, ridge: RidgeGeometry, q_of_delta: Callable[[float], float],
                 panel_breadth: float = 0.9, panel_length: float = 3.6,
                 n_panels: int = 25) -> None:
        self.ridge = ridge
        self.q_of_delta = q_of_delta
        self.b = float(panel_breadth)
        self.a = float(panel_length)
        self.n_panels = int(n_panels)
        self.y = (np.arange(self.n_panels) - (self.n_panels - 1) / 2.0) * self.b

    def local_pressure(self, apex_depth: float) -> np.ndarray:
        delta = self.ridge.penetration(self.y, apex_depth)
        return np.array([self.q_of_delta(float(d)) if d > 0.0 else 0.0 for d in delta])

    def local_deflection(self, apex_depth: float) -> np.ndarray:
        return self.ridge.penetration(self.y, apex_depth)

    def resistance(self, apex_depth: float) -> float:
        """Total grounding resistance (N) at the given apex penetration."""
        p = self.local_pressure(apex_depth)
        return float(p.sum() * self.b * self.ridge.length)

    def energy(self, apex_depth: float, n_steps: int = 24) -> float:
        """Absorbed energy (J) up to ``apex_depth`` (trapezoidal rule)."""
        d = np.linspace(0.0, float(apex_depth), int(n_steps) + 1)
        R = np.array([self.resistance(x) for x in d])
        return float(np.trapezoid(R, d))

    def apex_depth_for_energy(self, energy_target: float,
                              d_max: Optional[float] = None, n_steps: int = 40) -> float:
        """Invert the absorbed-energy curve ``E(apex)`` by bisection."""
        d_max = min(self.ridge.h_ridge, d_max if d_max is not None else self.ridge.h_ridge)
        lo, hi = 0.0, float(d_max)
        if energy_target <= 0.0:
            return 0.0
        if self.energy(hi, n_steps) < energy_target:
            return hi
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if self.energy(mid, n_steps) < energy_target:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-6:
                break
        return 0.5 * (lo + hi)
