"""Blade-element / momentum theory (BEMT) open-water model of a pod propeller.

The model is deliberately first-principles: the blade geometry (radius, chord,
pitch, blade number) is an input and K_T / K_Q are obtained by integrating
blade-element forces over the span with momentum-theory induction factors.
No measured open-water table is embedded, so every number is reproducible from
the geometry alone (see docs/POD_ICE_INTERACTION.md for the accuracy statement).

Formulation
-----------
Per annulus at radius ``r`` with chord ``c`` and pitch angle ``theta``::

    tan(phi) = V (1 + a) / (omega r (1 - a'))

    dT_blade = 0.5 rho W^2 Z c (Cl cos phi - Cd sin phi) dr
    dQ_blade = 0.5 rho W^2 Z c r (Cl sin phi + Cd cos phi) dr
    dT_mom   = 4 pi rho r V^2 a (1 + a) F dr
    dQ_mom   = 4 pi rho r^3 V omega (1 + a) a' F dr

with the Prandtl tip/hub loss factor ``F`` and the Wilson-Walker empirical
high-load correction for ``a > 0.4`` (axial momentum theory is invalid in the
turbulent wake state). Equating blade and momentum expressions gives the
fixed-point update for ``(a, a')``.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import acos, atan, cos, exp, pi, radians, sin
from typing import Callable

import numpy as np

RHO_SEAWATER = 1025.0  # kg/m^3, brackish/Arctic seawater used for ice-class work


def _prandtl_loss(r_over_r: float, phi: float, n_blades: int, hub_ratio: float) -> float:
    """Prandtl tip + hub loss factor ``F``."""
    s = max(sin(phi), 1e-6)
    f_tip = 0.5 * n_blades * (1.0 - r_over_r) / s
    f_hub = 0.5 * n_blades * (r_over_r - hub_ratio) / max(r_over_r * s, 1e-9)
    f = 1.0
    for fi in (f_tip, f_hub):
        if fi <= 1e-9:
            return 1e-6
        arg = min(exp(-fi), 1.0)
        f *= (2.0 / pi) * acos(max(-1.0, min(1.0, arg)))
    return max(f, 1e-6)


@dataclass(frozen=True)
class BladeGeometry:
    """Wide-blade (Kaplan type) pod propeller geometry."""

    diameter: float = 6.0
    hub_ratio: float = 0.35
    n_blades: int = 4
    pitch_ratio: float = 0.95  # P/D of the constant-pitch helix
    # (r/R, c/R) control points of the chord distribution
    chord_points: tuple[tuple[float, float], ...] = (
        (0.35, 0.78),
        (0.50, 0.68),
        (0.70, 0.50),
        (0.85, 0.30),
        (1.00, 0.10),
    )
    alpha_stall_deg: float = 13.0
    cd0: float = 0.010
    k_cd: float = 0.015
    alpha_l0_deg: float = -2.0

    def __post_init__(self) -> None:
        if self.diameter <= 0.0:
            raise ValueError("diameter must be positive")
        if not 0.0 < self.hub_ratio < 1.0:
            raise ValueError("hub_ratio must lie in (0, 1)")
        if self.n_blades < 2:
            raise ValueError("a propeller needs at least two blades")
        if self.pitch_ratio <= 0.0:
            raise ValueError("pitch_ratio must be positive")

    # ---------------------------------------------------------------- geometry
    @property
    def radius(self) -> float:
        return 0.5 * self.diameter

    @property
    def pitch(self) -> float:
        return self.pitch_ratio * self.diameter

    def chord_ratio_at(self, r_over_r: float) -> float:
        xs = np.array([p[0] for p in self.chord_points])
        ys = np.array([p[1] for p in self.chord_points])
        return float(np.interp(r_over_r, xs, ys))

    def chord_at(self, r_over_r: float) -> float:
        return self.chord_ratio_at(r_over_r) * self.radius

    def twist_at(self, r_over_r: float) -> float:
        """Blade pitch angle [rad] at ``r/R`` for a constant-pitch helix."""
        r = max(r_over_r, 1e-6) * self.radius
        return atan(self.pitch / (2.0 * pi * r))

    @property
    def expanded_area_ratio(self) -> float:
        """EAR = (Z/pi) * integral (c/R) d(r/R) over the blade span."""
        r = np.linspace(self.hub_ratio, 1.0, 401)
        cr = np.array([self.chord_ratio_at(x) for x in r])
        return float(self.n_blades / pi * np.trapezoid(cr, r))


# --------------------------------------------------------------------- polars
def _airfoil(alpha: float, geom: BladeGeometry) -> tuple[float, float]:
    """Lift and drag coefficients of the cambered blade section."""
    a0 = radians(geom.alpha_l0_deg)
    a_stall = radians(geom.alpha_stall_deg)
    cl_attached = 2.0 * pi * (alpha - a0)
    cl_max = 2.0 * pi * (a_stall - a0)
    if alpha <= a_stall:
        cl = cl_attached
    else:  # simple post-stall plateau (Viterna-like, lift frozen at Cl_max)
        cl = cl_max * max(0.0, cos(min(alpha - a_stall, pi / 2.0)) ** 0.5)
    cl = max(cl, -cl_max)
    cd = geom.cd0 + geom.k_cd * cl * cl
    return cl, cd


# ---------------------------------------------------------------- annulus BEMT
def solve_annulus(
    r_over_r: float,
    advance_ratio: float,
    geom: BladeGeometry,
    *,
    n_iter: int = 80,
    relax: float = 0.35,
) -> dict[str, float]:
    """Fixed-point BEMT solution of one annulus, nondimensionalised with R=1, V=1.

    ``advance_ratio`` is J = V/(n D). The returned ``dct``/``dcq`` are the
    annulus contributions to K_T / K_Q when multiplied by d(r/R).
    """
    if advance_ratio <= 0.0:
        raise ValueError("advance_ratio must be positive")
    # R = 1, V = 1  =>  J = 1/(2n)  =>  n = 1/(2J), omega = 2 pi n
    omega = pi / advance_ratio
    r = r_over_r
    c = geom.chord_ratio_at(r_over_r)  # = c/R (R = 1)
    theta = geom.twist_at(r_over_r)
    sigma = geom.n_blades * c / (2.0 * pi * r)
    a = 0.02
    a_s = 0.01
    phi = theta
    for _ in range(n_iter):
        phi = atan((1.0 + a) / (omega * r * (1.0 - a_s)))
        alpha = theta - phi
        cl, cd = _airfoil(alpha, geom)
        f_loss = _prandtl_loss(r_over_r, phi, geom.n_blades, geom.hub_ratio)
        s2 = max(sin(phi) ** 2, 1e-9)
        ct_blade = sigma * (cl * cos(phi) - cd * sin(phi)) / s2
        # axial induction from momentum theory (Wilson-Walker high-load branch)
        a_new = _axial_induction(ct_blade, f_loss)
        cq_blade = sigma * (cl * sin(phi) + cd * cos(phi)) / s2
        a_s_new = cq_blade / (4.0 * f_loss * (1.0 + a_new))
        a = a + relax * (a_new - a)
        a_s = a_s + relax * (a_s_new - a_s)
        a = min(max(a, -0.5), 0.95)
        a_s = min(max(a_s, -0.5), 0.95)
    phi = atan((1.0 + a) / (omega * r * (1.0 - a_s)))
    alpha = theta - phi
    cl, cd = _airfoil(alpha, geom)
    f_loss = _prandtl_loss(r_over_r, phi, geom.n_blades, geom.hub_ratio)
    s2 = max(sin(phi) ** 2, 1e-9)
    w2 = (1.0 + a) ** 2 / s2  # W^2 / V^2
    c_ratio = geom.chord_ratio_at(r_over_r)
    j2 = advance_ratio ** 2
    # dK_T/d(r/R) and dK_Q/d(r/R); see module docstring for the derivation
    dkt = geom.n_blades * c_ratio * w2 * (cl * cos(phi) - cd * sin(phi)) * j2 / 8.0
    dkq = geom.n_blades * c_ratio * r_over_r * w2 * (cl * sin(phi) + cd * cos(phi)) * j2 / 16.0
    return {
        "a": a,
        "a_prime": a_s,
        "phi": phi,
        "alpha": alpha,
        "cl": cl,
        "cd": cd,
        "loss": f_loss,
        "dkt": dkt,
        "dkq": dkq,
        "solidity": sigma,
    }


def _axial_induction(ct: float, f_loss: float) -> float:
    """Axial induction factor from the blade thrust loading ``ct``.

    Momentum theory for ``a <= 0.4``; Wilson-Walker empirical polynomial in the
    turbulent wake state (``a > 0.4``) where ``CT = 8/9 + (4F - 40/9) a
    + (50/9 - 4F) a^2``.
    """
    ct = max(ct, 0.0)
    f = max(f_loss, 1e-6)
    a_mom = ct / (4.0 * f * (1.0 + 0.25 * ct))
    if a_mom <= 0.4:
        return a_mom
    qa = 50.0 / 9.0 - 4.0 * f
    qb = 4.0 * f - 40.0 / 9.0
    qc = 8.0 / 9.0 - ct
    disc = qb * qb - 4.0 * qa * qc
    if disc < 0.0 or abs(qa) < 1e-12:
        return 0.5
    # physical root (a in [0.4, 1]); the other root is negative
    return float((-qb + disc ** 0.5) / (2.0 * qa))


@dataclass(frozen=True)
class OpenWaterCoefficients:
    """K_T, K_Q and efficiency at one advance ratio."""

    j: float
    kt: float
    kq: float
    eta: float


def open_water(
    geom: BladeGeometry,
    advance_ratio: float,
    *,
    n_annuli: int = 40,
) -> OpenWaterCoefficients:
    """Thrust/torque coefficients from spanwise BEMT integration."""
    r = np.linspace(geom.hub_ratio, 1.0, n_annuli)
    dkt = np.empty_like(r)
    dkq = np.empty_like(r)
    for i, rr in enumerate(r):
        sol = solve_annulus(float(rr), advance_ratio, geom)
        dkt[i] = sol["dkt"]
        dkq[i] = sol["dkq"]
    kt = float(np.trapezoid(dkt, r))
    kq = float(np.trapezoid(dkq, r))
    eta = 0.0 if kq <= 0.0 else advance_ratio * kt / (2.0 * pi * kq)
    return OpenWaterCoefficients(j=advance_ratio, kt=kt, kq=kq, eta=eta)


class OpenWaterCurve:
    """Tabulated BEMT open-water curve with spline interpolation.

    The table is generated once from the geometry and then reused by the
    rotor-dynamics and thrust-envelope solvers (thousands of K_T/K_Q lookups).
    """

    def __init__(self, geom: BladeGeometry, *, j_min: float = 0.02,
                 j_max: float = 1.30, n_points: int = 27, n_annuli: int = 40) -> None:
        js = np.linspace(j_min, j_max, n_points)
        kts, kqs = [], []
        for j in js:
            ow = open_water(geom, float(j), n_annuli=n_annuli)
            kts.append(ow.kt)
            kqs.append(ow.kq)
        self.geometry = geom
        self.j = js
        self.kt = np.array(kts)
        self.kq = np.array(kqs)
        self.rho = RHO_SEAWATER

    # ------------------------------------------------------------- properties
    def kt_at(self, j: float) -> float:
        return float(np.interp(j, self.j, self.kt))

    def kq_at(self, j: float) -> float:
        return float(np.interp(j, self.j, self.kq))

    def eta_at(self, j: float) -> float:
        kq = self.kq_at(j)
        return 0.0 if kq <= 0.0 else j * self.kt_at(j) / (2.0 * pi * kq)

    # ------------------------------------------------------------- dimensional
    def thrust(self, n_rps: float, v_ship: float) -> float:
        d = self.geometry.diameter
        if n_rps <= 1e-9:
            return 0.0
        j = v_ship / (n_rps * d)
        return self.kt_at(j) * self.rho * n_rps ** 2 * d ** 4

    def torque(self, n_rps: float, v_ship: float) -> float:
        d = self.geometry.diameter
        if n_rps <= 1e-9:
            return 0.0
        j = v_ship / (n_rps * d)
        return self.kq_at(j) * self.rho * n_rps ** 2 * d ** 5

    def delivered_power(self, n_rps: float, v_ship: float) -> float:
        return 2.0 * pi * n_rps * self.torque(n_rps, v_ship)

    # ----------------------------------------------------------- derivatives
    def torque_gradient(self, n_rps: float, v_ship: float, d_n: float = 1e-4) -> float:
        """dQ/dn at fixed ship speed [N m s] (central difference)."""
        qp = self.torque(n_rps + d_n * max(n_rps, 1.0), v_ship)
        qm = self.torque(max(n_rps - d_n * max(n_rps, 1.0), 1e-6), v_ship)
        return (qp - qm) / (2.0 * d_n * max(n_rps, 1.0))

    def thrust_gradient(self, n_rps: float, v_ship: float, d_n: float = 1e-4) -> float:
        """dT/dn at fixed ship speed [N s]."""
        tp = self.thrust(n_rps + d_n * max(n_rps, 1.0), v_ship)
        tm = self.thrust(max(n_rps - d_n * max(n_rps, 1.0), 1e-6), v_ship)
        return (tp - tm) / (2.0 * d_n * max(n_rps, 1.0))

    def solve_rpm_for_torque(self, torque: float, v_ship: float,
                            n_bounds: tuple[float, float] = (0.05, 8.0)) -> float:
        """Rotation rate [rps] at which the propeller absorbs ``torque``."""
        lo, hi = n_bounds
        if self.torque(hi, v_ship) < torque:
            return hi
        for _ in range(120):
            mid = 0.5 * (lo + hi)
            if self.torque(mid, v_ship) < torque:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-10:
                break
        return 0.5 * (lo + hi)

    def solve_rpm_for_power(self, power: float, v_ship: float,
                            n_bounds: tuple[float, float] = (0.05, 8.0)) -> float:
        lo, hi = n_bounds
        for _ in range(120):
            mid = 0.5 * (lo + hi)
            if self.delivered_power(mid, v_ship) < power:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-10:
                break
        return 0.5 * (lo + hi)