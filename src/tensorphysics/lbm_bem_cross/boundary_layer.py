"""Integral boundary-layer correction of the BEM potential-flow solution.

The BEM solves the *inviscid* problem; a boundary-layer correction is what makes
it comparable with a Navier-Stokes solution at all.  This module implements the
classical viscous-inviscid interaction (VPI) chain:

1. **Thwaites' (1949) laminar integral method** on the BEM edge velocity
   ``U_e(s)`` gives momentum thickness, shape factor, skin friction and the
   laminar separation point::

       theta^2 = 0.45 nu / U_e^6 * int_0^s U_e^5 ds'
       lambda  = theta^2 / nu * dU_e/ds          (separation at lambda = -0.09)
       Cf/2    = l(lambda) nu / (U_e theta)
       H       = 2.61 - 3.75 lambda + 5.24 lambda^2
       delta*  = H theta

2. **Displacement-thickness coupling**: the body is inflated by ``delta*(s)``
   (the *displacement body*) and the panel method is re-solved iteratively until
   edge velocity and ``delta*`` are mutually consistent.  The pressure therefore
   *responds* to the boundary layer, which is the whole point of a VPI
   correction — plain BL post-processing cannot shift the mid-body suction peak
   (for the potential cylinder ``Cp_min = -3``, whereas even at ``Re = 1e4`` the
   measured value is about ``-1.2``).

3. **Separated wake.** Downstream of separation the integral method is invalid.
   The classical constant-pressure wake assumption is the *baseline* closure
   (``Cp = Cp(s_sep)`` over the separated region, Kirchhoff/Helmholtz
   free-streamline view).  The wake under-pressure is the dominant model
   uncertainty, so :func:`drag_band` sweeps it (``wake_factor`` from 0 to -0.6)
   and reports a band instead of pretending to a single number.

Drag convention
---------------
Non-dimensional (``U_inf = 1``, ``D = 1``, ``nu = 1/Re``)::

    Cd_p = 1/2 * int_0^{2pi} Cp(theta) cos(theta) dtheta
    Cd_f = 2 * int_0^{s_sep} Cf(s)/2 * sin(theta) ds

with ``theta`` measured from the upstream stagnation point, so that the outward
normal is ``n_x = -cos(theta)`` and the surface tangent ``t_x = sin(theta)``.

Validity range
--------------
Thwaites is a *thin* boundary-layer theory: reliable for ``Re >~ 1e4`` and
progressively worse at ``Re = 100..1000``, where ``delta*/D`` reaches 0.1-0.5
and strong interaction invalidates the marching assumptions.  This is a known
limitation of the potential+BEM route (documented in
``docs/LBM_BEM_CROSS_VALIDATION.md``), reported rather than hidden.

References
----------
Thwaites, B. (1949). Approximate calculation of the laminar boundary layer.
*Aeronautical Quarterly* 1, 245-280.
Schlichting, H. & Gersten, K. (2017). *Boundary-Layer Theory*, 9th ed., ch. 7.
Cebeci, T. & Bradshaw, P. (1977). *Momentum Transfer in Boundary Layers*.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .bem import PanelMethod2D, surface_from_polyline
from .geometry import SurfaceGeometry

__all__ = [
    "BoundaryLayerSolution",
    "thwaites_march",
    "VPIResult",
    "viscous_inviscid_iteration",
    "drag_band",
    "thwaites_flat_plate_check",
]


@dataclass
class BoundaryLayerSolution:
    """Laminar integral boundary layer along one side of the body."""

    s: np.ndarray            # arc length from the front stagnation point (/D)
    ue: np.ndarray           # edge velocity U_e/U_inf
    theta: np.ndarray        # momentum thickness (/D)
    delta_star: np.ndarray   # displacement thickness (/D)
    H: np.ndarray            # shape factor
    cf: np.ndarray           # skin-friction coefficient Cf
    lam: np.ndarray          # Thwaites pressure-gradient parameter
    sep_index: int = -1      # index of separation (-1: attached)
    sep_s: float = float("nan")

    @property
    def separated(self) -> bool:
        return self.sep_index >= 0

    def cd_friction(self, theta_surface: np.ndarray | None = None, n_sides: int = 2) -> float:
        """Friction drag ``2 int Cf/2 sin(theta) ds`` (see module docstring)."""
        n = self.sep_index if self.sep_index > 0 else self.s.size
        s = self.s[:n]
        cf = np.clip(self.cf[:n], 0.0, None)
        if s.size < 2:
            return 0.0
        if theta_surface is not None:
            w = np.sin(np.asarray(theta_surface)[:n])
        else:
            w = np.ones_like(s)
        return float(n_sides * 0.5 * np.trapezoid(cf * w, s))


def thwaites_march(s: np.ndarray, ue: np.ndarray, re: float,
                   sep_lambda: float = -0.09, ue_start: float = 0.10,
                   cf_cap: float = 0.02) -> BoundaryLayerSolution:
    """Thwaites laminar integral boundary layer (see module docstring).

    ``ue_start`` excludes the stagnation neighbourhood, where ``theta -> 0``
    makes ``Cf`` singular; ``cf_cap`` bounds the (physically meaningless) very
    high friction of that region.
    """
    s = np.asarray(s, dtype=np.float64)
    ue = np.asarray(ue, dtype=np.float64)
    nu = 1.0 / re
    n = s.size

    theta = np.zeros(n)
    lam = np.zeros(n)
    cf = np.zeros(n)
    H = np.full(n, 2.61)
    delta_star = np.zeros(n)
    sep_index = -1

    above = np.where(ue > ue_start)[0]
    if above.size < 3:
        return BoundaryLayerSolution(s=s, ue=ue, theta=theta, delta_star=delta_star, H=H,
                                     cf=cf, lam=lam, sep_index=-1, sep_s=float("nan"))
    i0 = int(above[0])
    sl = s[i0:]
    ul = np.maximum(ue[i0:], 1e-6)
    integ = ul ** 5
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (integ[1:] + integ[:-1]) * np.diff(sl))])
    theta2 = 0.45 * nu * cum / ul ** 6
    th_l = np.sqrt(np.maximum(theta2, 1e-30))
    due = np.gradient(ul, sl)
    lam_l = theta2 / nu * due
    cf_l = np.empty_like(th_l)
    for i, l in enumerate(lam_l):
        if l <= 0.0:
            ell = 0.22 + 1.575 * l - 1.8 * l * l
        else:
            ell = 0.22 + 1.402 * l + 0.018 * l / (0.107 + l)
        cf_l[i] = 2.0 * ell * nu / (ul[i] * th_l[i])
    cf_l = np.clip(cf_l, 0.0, cf_cap)
    H_l = 2.61 - 3.75 * lam_l + 5.24 * lam_l ** 2
    ds_l = H_l * th_l

    # separation: first point beyond the suction peak with lambda < sep_lambda,
    # while the edge flow is still energetic (the rear stagnation neighbourhood
    # is singular for Thwaites and excluded)
    i_peak = int(np.argmax(ul))
    for i in range(max(i_peak, 2), ul.size):
        if ul[i] < 0.05:
            break
        if lam_l[i] < sep_lambda:
            sep_index = i0 + i
            break
    if sep_index > 0:
        k = sep_index - i0
        for arr in (th_l, cf_l, H_l, ds_l, lam_l):
            arr[k:] = arr[k]

    theta[i0:] = th_l
    lam[i0:] = lam_l
    cf[i0:] = cf_l
    H[i0:] = H_l
    delta_star[i0:] = ds_l
    return BoundaryLayerSolution(
        s=s, ue=ue, theta=theta, delta_star=delta_star, H=H, cf=cf, lam=lam,
        sep_index=sep_index, sep_s=float(s[sep_index]) if sep_index > 0 else float("nan"))


@dataclass
class VPIResult:
    """Result of the viscous-inviscid interaction."""

    theta_grid: np.ndarray        # surface angle grid (0..2pi)
    cp: np.ndarray                # corrected Cp (separated region = wake pressure)
    cp_attached: np.ndarray       # Cp of the displaced-body inviscid solution
    ue: np.ndarray                # edge velocity on the upper surface
    theta_upper: np.ndarray       # theta of the upper-surface BL stations
    bl: BoundaryLayerSolution
    delta_star_max: float
    sep_angle: float              # rad
    cp_wake: float
    n_iterations: int
    convergence: float
    cd_pressure: float
    cd_friction: float
    body_area_ratio: float = 1.0

    @property
    def cd(self) -> float:
        return self.cd_pressure + self.cd_friction

    def to_dict(self) -> dict:
        return {
            "sep_angle_deg": float(math.degrees(self.sep_angle)),
            "cp_wake": float(self.cp_wake),
            "cp_min": float(np.min(self.cp)),
            "delta_star_max_over_D": float(self.delta_star_max),
            "n_iterations": int(self.n_iterations),
            "convergence": float(self.convergence),
            "cd_pressure": float(self.cd_pressure),
            "cd_friction": float(self.cd_friction),
            "cd": float(self.cd),
            "cf_max": float(np.nanmax(self.bl.cf)),
            "H_at_separation": float(self.bl.H[max(self.bl.sep_index - 1, 0)]),
            "body_area_ratio": float(self.body_area_ratio),
        }


def _displaced_body(geom: SurfaceGeometry, theta_up: np.ndarray,
                    delta_up: np.ndarray) -> SurfaceGeometry:
    """Inflate the body along its outward normal by ``delta*`` (symmetrised)."""
    th_up_mod = np.mod(geom.theta(), np.pi)
    order = np.argsort(theta_up)
    ds = np.interp(th_up_mod, theta_up[order], delta_up[order])
    x = geom.x + ds * geom.nx
    y = geom.y + ds * geom.ny
    return surface_from_polyline(x, y, fmt="vpi")


def _wake_taper(n: int, i_sep: int) -> np.ndarray:
    """Displacement-thickness shape in the separated region (constant + taper)."""
    r = np.zeros(n)
    if i_sep < 0 or i_sep >= n - 1:
        return np.ones(n)
    r[:i_sep + 1] = 1.0
    tail = n - i_sep
    ramp = np.clip(1.0 - np.arange(1, tail + 1) / max(tail * 1.25, 1.0), 0.0, 1.0)
    r[i_sep:] = ramp
    return r


def viscous_inviscid_iteration(geom: SurfaceGeometry, re: float, u_inf: float = 1.0,
                               n_iter: int = 12, relax: float = 0.3,
                               wake_factor: float = 0.0, delta_cap: float = 0.02,
                               n_theta: int = 145) -> VPIResult:
    """Couple the panel method with the Thwaites boundary layer.

    Parameters
    ----------
    geom : rigid body surface (shared geometry contract)
    re : Reynolds number based on the diameter
    wake_factor : under-pressure of the wake relative to ``Cp(s_sep)``
        (``Cp_wake = Cp_sep + wake_factor``, ``wake_factor <= 0``); the baseline
        constant-pressure wake is ``wake_factor = 0``.
    """
    theta_grid = np.linspace(0.0, 2.0 * np.pi, n_theta)
    idx_up = np.where(geom.theta() <= np.pi)[0]
    th_up = np.sort(np.mod(geom.theta()[idx_up], np.pi))

    delta_prev = np.zeros_like(th_up)
    conv = float("inf")
    bl = thwaites_march(np.linspace(0.0, 1.0, th_up.size), np.ones_like(th_up) * 1e-3, re)
    body = geom
    for it in range(1, n_iter + 1):
        body = _displaced_body(geom, th_up, delta_prev) if np.any(delta_prev) else geom
        pm = PanelMethod2D(body, u_inf=u_inf)
        sol = pm.solve()
        cp_at = sol.cp_at(th_up)
        ue = np.sqrt(np.clip(1.0 - cp_at, 0.0, None))
        # arc length along the *displaced* body, from the front stagnation point
        seg = np.hypot(np.diff(body.x), np.diff(body.y))
        s_all = np.concatenate([[0.0], np.cumsum(seg)])
        s = np.sort(s_all[idx_up])
        s = s - s[0]
        bl = thwaites_march(s, ue[np.argsort(s_all[idx_up])], re)
        # weak-interaction limit: the direct (simultaneous) VPI is only
        # well-posed while delta*/D << 1; beyond that the iteration is the known
        # strong-interaction divergence case (reported through ``convergence``)
        ds_capped = np.minimum(bl.delta_star, delta_cap)
        ds_new = ds_capped * _wake_taper(ds_new_size := ds_capped.size, bl.sep_index)
        delta_new = (1.0 - relax) * delta_prev + relax * ds_new
        conv = float(np.max(np.abs(delta_new - delta_prev)))
        delta_prev = delta_new
        if conv < 2e-5 and it >= 3:
            break

    body = _displaced_body(geom, th_up, delta_prev)
    pm = PanelMethod2D(body, u_inf=u_inf)
    sol = pm.solve()
    cp_att = sol.cp_at(theta_grid)
    sep = float(th_up[min(bl.sep_index, th_up.size - 1)]) if bl.sep_index > 0 else float(np.pi)
    cp_sep = float(np.interp(sep, theta_grid, cp_att))
    cp_wake = cp_sep + wake_factor
    cp = cp_att.copy()
    if bl.sep_index > 0:
        sep_mask = (theta_grid >= sep) & (theta_grid <= 2.0 * np.pi - sep)
        cp[sep_mask] = cp_wake
    cd_p = drag_from_cp(theta_grid, cp)
    cd_f = bl.cd_friction(theta_surface=np.mod(np.interp(bl.s, np.sort(s), np.sort(np.mod(body.theta()[idx_up], np.pi))), np.pi))
    return VPIResult(
        theta_grid=theta_grid, cp=cp, cp_attached=cp_att, ue=bl.ue, theta_upper=th_up,
        bl=bl, delta_star_max=float(np.max(delta_prev)), sep_angle=sep,
        cp_wake=float(cp_wake), n_iterations=it, convergence=conv,
        cd_pressure=cd_p, cd_friction=cd_f, body_area_ratio=float(body.area / geom.area))


def drag_from_cp(theta: np.ndarray, cp: np.ndarray) -> float:
    """Pressure drag ``Cd_p = 1/2 * int Cp cos(theta) dtheta``."""
    return float(0.5 * np.trapezoid(np.asarray(cp) * np.cos(np.asarray(theta)), np.asarray(theta)))


def drag_band(geom: SurfaceGeometry, re: float,
              wake_offsets=(0.0, 0.4, 0.8, 1.2, 1.6), **kwargs) -> dict:
    """Cd of the BEM+BL model versus the wake-pressure closure.

    ``wake_offsets`` adds to ``Cp(s_sep)`` inside the separated region
    (``Cp_wake = Cp_sep + offset``).  ``offset = 0`` is the classical
    constant-pressure wake; positive offsets represent the pressure *recovery*
    towards the real base pressure that the potential solution cannot produce.
    """
    out = {"wake_offset": [], "cd": [], "cd_pressure": [], "cd_friction": [],
           "sep_angle_deg": [], "cp_min": [], "cp_wake": []}
    base = None
    for du in wake_offsets:
        res = viscous_inviscid_iteration(geom, re, wake_factor=du, **kwargs)
        out["wake_offset"].append(float(du))
        out["cp_wake"].append(float(res.cp_wake))
        out["cd"].append(float(res.cd))
        out["cd_pressure"].append(float(res.cd_pressure))
        out["cd_friction"].append(float(res.cd_friction))
        out["sep_angle_deg"].append(float(math.degrees(res.sep_angle)))
        out["cp_min"].append(float(np.min(res.cp)))
        if base is None:
            base = res
    out["baseline"] = base.to_dict()
    return out


def thwaites_flat_plate_check(re: float = 1e5, n: int = 400, cf_cap: float = 1.0) -> dict:
    """Verification of :func:`thwaites_march` against the Blasius flat plate.

    With ``U_e = const`` Thwaites reduces to ``theta = sqrt(0.45 nu x)`` and
    ``Cf = 0.664 / sqrt(Re_x)`` (Blasius 1908).
    """
    re_x = np.logspace(2, 6, n)
    s = re_x / re                     # x/D with nu = 1/Re
    ue = np.ones_like(s)
    bl = thwaites_march(s, ue, re, cf_cap=cf_cap)
    cf_blasius = 0.664 / np.sqrt(re_x)
    theta_exact = np.sqrt(0.45 * (1.0 / re) * s)
    # the first stations carry the (inevitable) start-up discretisation error of
    # the integral; the model is used over Re_x >= 1e4, where it is compared here
    k = np.where(re_x >= 1e4)[0][0]
    return {
        "window_re_x": [float(re_x[k]), float(re_x[-1])],
        "max_rel_err_cf_vs_blasius": float(np.max(np.abs(bl.cf[k:] - cf_blasius[k:]) / cf_blasius[k:])),
        "max_rel_err_theta_vs_thwaites_exact": float(
            np.max(np.abs(bl.theta[k:] - theta_exact[k:]) / theta_exact[k:])),
        "blasius_cf_at_re_1e5": float(bl.cf[np.argmin(np.abs(re_x - 1e5))]),
        "reference_cf_at_re_1e5": float(0.664 / np.sqrt(1e5)),
    }