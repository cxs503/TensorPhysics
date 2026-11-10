"""Lagrangian supercooled-droplet tracking and impingement collection efficiency.

The classical Langmuir-Blodgett trajectory method is implemented: a cloud of
monodisperse droplets (MVD ``d``) is released far upstream, each droplet is
integrated through the analytic potential-flow field of the body with the
particle momentum equation

.. math::

    \\frac{\\mathrm{d}\\mathbf{v}}{\\mathrm{d}t} =
        \\frac{f_D}{\\tau_p}\\left(\\mathbf{u} - \\mathbf{v}\\right) + \\mathbf{g},
    \\qquad
    \\tau_p = \\frac{\\rho_w d^2}{18 \\mu_a},

with the Schiller-Naumann drag correction
``f_D = 1 + 0.15 Re_p^0.687`` and ``Re_p = |u-v| d / nu_a``.

For a launch ordinate ``y0`` (measured from the stagnation streamline) that
impacts at surface arc length ``s``, the local collection efficiency is

.. math::

    \\beta(s) = -\\frac{\\mathrm{d} y_0}{\\mathrm{d} s}
              = \\frac{\\dot m''_{imp}}{LWC \\cdot U},

i.e. the ratio of the local water catch to the freestream flux.  The *total*
collection efficiency is ``E = y0_max / h`` with ``h`` the half-height of the
section normal to the flow, so that ``E -> 1`` in the ballistic limit and
``E -> 0`` in the inertia-free limit.

Both limits are available in closed form (:func:`beta_ballistic`) and are used
as acceptance benchmarks in ``tests/test_superstructure_icing.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .air import mu_air, nu_air, rho_air, rho_ice  # noqa: F401  (re-export)
from .flowfield import BluffBody

RHO_W = 1000.0  # water density [kg/m3]


def particle_relaxation_time(d_m: float, t_air: float = -15.0) -> float:
    """Stokes relaxation time of a droplet [s]."""
    return RHO_W * d_m**2 / (18.0 * mu_air(t_air))


def stokes_number(d_m: float, U: float, L: float, t_air: float = -15.0) -> float:
    """Dimensionless inertia parameter ``K = rho_w d^2 U / (18 mu L)``."""
    return particle_relaxation_time(d_m, t_air) * U / L


def droplet_reynolds(d_m: float, U: float, t_air: float = -15.0) -> float:
    """Droplet Reynolds number based on the freestream slip."""
    return U * d_m / nu_air(t_air)


@dataclass
class ImpingementResult:
    """Local and integral impingement characteristics on a section."""

    s: np.ndarray  # surface arc length from stagnation [m]
    beta: np.ndarray  # local collection efficiency [-]
    y0: np.ndarray  # launch ordinates that produced the impacts [m]
    E_total: float  # total collection efficiency [-]
    s_limit: float  # impingement limit (arc length) [m]
    K: float  # inertia parameter [-]
    Re_d: float  # droplet Reynolds number [-]
    d_m: float  # MVD [m]
    U: float  # airspeed [m/s]
    body_height: float  # half-height normal to flow [m]
    n_traj: int  # number of trajectories used

    def beta_at(self, s_query, off_surface: bool = False) -> np.ndarray:
        """Interpolate ``beta`` at arbitrary arc lengths.

        Outside the impingement limits ``beta = 0`` (shadow zone / no catch);
        with ``off_surface=True`` the value at the impingement limit is returned
        instead (used by the Messinger runback model).
        """
        from scipy.interpolate import PchipInterpolator

        s_query = np.atleast_1d(np.asarray(s_query, dtype=float))
        if self.s.size < 3:
            return np.zeros_like(s_query)
        f = PchipInterpolator(self.s, self.beta, extrapolate=False)
        out = f(np.clip(s_query, 0.0, self.s_limit))
        out = np.where(np.isnan(out), 0.0, out)
        out = np.where(s_query > self.s_limit, self.beta[-1] if off_surface else 0.0, out)
        return np.maximum(out, 0.0)

    def water_flux(self, s_query, lwc: float) -> np.ndarray:
        """Impingement mass flux ``LWC*U*beta`` [kg/(m2 s)]."""
        return lwc * self.U * self.beta_at(s_query)


def beta_ballistic(s, body: BluffBody) -> np.ndarray:
    """Ballistic (infinite Stokes number) limit of the collection efficiency.

    Straight-line trajectories hit the section normally, so on a cylinder
    ``beta = cos(theta) = sqrt(1 - (s/R)^2)``; for the general section the
    same projection is evaluated from the local surface tangent.
    """
    s = np.asarray(s, dtype=float)
    if hasattr(body, "R"):  # cylinder
        x = np.clip(s / body.R, -1.0, 1.0)
        return np.sqrt(np.maximum(1.0 - x * x, 0.0))
    xs, ys, ss = body.surface(20001)
    # dy/ds is the projection of the surface normal on the flow direction
    dy = np.gradient(ys, ss)
    return np.sqrt(np.maximum(1.0 - dy**2, 0.0)).clip(0.0, 1.0) * np.interp(
        s, ss, np.ones_like(ss)
    )


def beta_area_weighted_cos(s, body: BluffBody) -> np.ndarray:
    """Ballistic limit as ``|n_x|`` evaluated from the surface polyline."""
    xs, ys, ss = body.surface(20001)
    dx = np.gradient(xs, ss)
    # tangent (dx,dy) normalised; normal = (dy, -dx); n_x = dy
    t = np.hypot(dx, np.gradient(ys, ss))
    nx = np.gradient(ys, ss) / np.maximum(t, 1e-12)
    return np.abs(np.interp(s, ss, nx))


# ---------------------------------------------------------------------------
# trajectory integration
# ---------------------------------------------------------------------------
# The particle equation is stiff for small droplets (tau_p << U/L), so the drag
# term is integrated *exactly* (exponential integrator) while the air velocity
# is frozen at the midpoint of the step:
#
#   dv/dt = (u - v)/tau_eff + g  =>  w = u + tau_eff g
#   v(t+dt) = w + (v - w) exp(-dt/tau_eff)
#   x(t+dt) = x + w dt + (v - w) tau_eff (1 - exp(-dt/tau_eff))
#
# This is unconditionally stable for any d (the inertia-free limit v -> u is
# recovered exactly) and second order in the position for a smooth field.


def _exp_step(body: BluffBody, x, y, vx, vy, dt, tau, nu, g, d_m):
    """One exponential-integrator step of the particle momentum equation.

    ``d_m`` is the *droplet* diameter: the Schiller-Naumann correction uses the
    particle Reynolds number ``Re_p = |u-v| d_m / nu``.  (Using the body
    diameter here is a silent unit error that turns every droplet into a
    tracer.)
    """
    xp = x + vx * dt
    yp = y + vy * dt
    u, v = body.velocity(0.5 * (x + xp), 0.5 * (y + yp))
    sx = vx - u
    sy = vy - v
    slip = np.hypot(sx, sy)
    re = slip * d_m / nu
    fd = 1.0 + 0.15 * np.maximum(re, 1e-12) ** 0.687
    te = tau / fd  # effective relaxation time
    ex = np.exp(-dt / te)
    wx = u
    wy = v + te * g
    vxn = wx + (vx - wx) * ex
    vyn = wy + (vy - wy) * ex
    xn = x + wx * dt + (vx - wx) * te * (1.0 - ex)
    yn = y + wy * dt + (vy - wy) * te * (1.0 - ex)
    return xn, yn, vxn, vyn


def _bisect_hit(body, x, y, vx, vy, dt, tau, nu, g, d_m, g_tol=2e-4, n_bisect=26):
    """Refine the last step to locate the first wall contact.

    Contact is detected on the normalised gap ``body.gap`` rather than on
    "inside the body": near the stagnation point a low-inertia droplet
    asymptotically approaches the wall and never enters the body in finite
    time, so an inside-test would silently drop stagnation-region impacts.
    """
    lo = np.zeros_like(x)
    hi = np.full_like(x, dt)
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        xm, ym, _, _ = _exp_step(body, x, y, vx, vy, mid, tau, nu, g, d_m)
        contact = body.gap(xm, ym) <= g_tol
        hi = np.where(contact, mid, hi)
        lo = np.where(contact, lo, mid)
    xe, ye, _, _ = _exp_step(body, x, y, vx, vy, hi, tau, nu, g, d_m)
    return xe, ye


def track_droplets(
    body: BluffBody,
    d_m: float,
    y0: np.ndarray,
    U: float | None = None,
    t_air: float = -15.0,
    gravity: bool = False,
    x_start_factor: float = 2.5,
    conv_factor: float = 0.01,
    g_tol: float = 2e-4,
    max_steps: int = 400000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate a bundle of droplets and return ``(y0_hit, s_hit, hit_mask)``.

    Parameters
    ----------
    body
        Analytic flow section (see :mod:`.flowfield`).
    d_m
        Droplet diameter (MVD) [m].
    y0
        Launch ordinates upstream [m].
    U
        Airspeed [m/s]; defaults to ``body.U``.
    conv_factor
        Convective Courant factor: ``dt = conv_factor * L / U``.
    """
    U = float(body.U if U is None else U)
    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    n = y0.size
    L_char = max(float(body.A) if hasattr(body, "A") else float(body.R), body.diameter)
    tau = particle_relaxation_time(d_m, t_air)
    x_start = -x_start_factor * L_char
    nu = nu_air(t_air)
    g = -9.81 if gravity else 0.0

    x = np.full(n, x_start)
    y = y0.copy()
    vx = np.full(n, U)
    vy = np.zeros(n)
    active = np.ones(n, dtype=bool)
    hit = np.zeros(n, dtype=bool)
    s_hit = np.full(n, np.nan)

    dt = conv_factor * L_char / U
    steps = 0
    while active.any() and steps < max_steps:
        xn, yn, vxn, vyn = _exp_step(body, x, y, vx, vy, dt, tau, nu, g, d_m)
        newly_in = active & (body.gap(xn, yn) <= g_tol)
        if newly_in.any():
            idx = np.nonzero(newly_in)[0]
            xe, ye = _bisect_hit(
                body, x[idx], y[idx], vx[idx], vy[idx], dt, tau, nu, g, d_m, g_tol=g_tol
            )
            ss, _ = body.project(xe, ye)
            s_hit[idx] = ss
            hit[idx] = True
            active[idx] = False
        # deactivate droplets that passed the body (wake) or went non-finite
        active &= np.isfinite(xn) & (xn < 1.5 * L_char)
        x, y, vx, vy = xn, yn, vxn, vyn
        steps += 1

    return y0[hit], s_hit[hit], hit


def trajectory_paths(
    body: BluffBody,
    d_m: float,
    y0_values,
    U: float | None = None,
    t_air: float = -15.0,
    gravity: bool = False,
    x_start_factor: float = 2.5,
    conv_factor: float = 0.01,
    max_steps: int = 40000,
) -> list[np.ndarray]:
    """Return the recorded paths (``N x 2`` arrays) of a set of droplets.

    Used for the trajectory figure of the report; the integrator is identical to
    :func:`track_droplets`.
    """
    U = float(body.U if U is None else U)
    y0_values = np.atleast_1d(np.asarray(y0_values, dtype=float))
    L_char = max(float(body.A) if hasattr(body, "A") else float(body.R), body.diameter)
    tau = particle_relaxation_time(d_m, t_air)
    nu = nu_air(t_air)
    g = -9.81 if gravity else 0.0
    dt = conv_factor * L_char / U
    x = np.full(y0_values.size, -x_start_factor * L_char)
    y = y0_values.copy()
    vx = np.full(y0_values.size, U)
    vy = np.zeros(y0_values.size)
    active = np.ones(y0_values.size, dtype=bool)
    paths = [[(xx, yy)] for xx, yy in zip(x, y)]
    steps = 0
    while active.any() and steps < max_steps:
        xn, yn, vxn, vyn = _exp_step(body, x, y, vx, vy, dt, tau, nu, g, d_m)
        newly = active & (body.gap(xn, yn) <= 1e-3)
        for i in np.nonzero(active)[0]:
            paths[i].append((float(xn[i]), float(yn[i])))
        active &= np.isfinite(xn) & (xn < 1.5 * L_char)
        active &= ~newly
        x, y, vx, vy = xn, yn, vxn, vyn
        steps += 1
    return [np.asarray(pth) for pth in paths]


def impingement(
    body: BluffBody,
    d_m: float,
    U: float | None = None,
    t_air: float = -15.0,
    n_traj: int = 241,
    y0_max_factor: float = 1.5,
    gravity: bool = False,
    conv_factor: float = 0.01,
    g_tol: float = 2e-4,
    n_table: int = 161,
) -> ImpingementResult:
    """Full impingement analysis: ``beta(s)``, ``E`` and the impingement limits.

    ``n_traj`` trajectories are launched uniformly over
    ``y0 in [0, y0_max_factor * height]``; the largest impacting ordinate is then
    refined by bisection so that ``E = y0_max / height`` is grid independent.
    """
    U = float(body.U if U is None else U)
    h = body.height
    # launch ordinates: half linearly spaced (resolves the impingement limit,
    # where beta -> 0) and half geometrically spaced near the stagnation
    # streamline (resolves the low-inertia cases, where the catch width is a
    # tiny fraction of the section height)
    n_lin = max(4, n_traj // 2)
    n_geo = max(4, n_traj - n_lin)
    y_lin = np.linspace(1e-4 * h, y0_max_factor * h, n_lin)
    y_geo = 1e-5 * h * (y0_max_factor / 1e-5) ** (np.arange(n_geo) / (n_geo - 1))
    y0 = np.unique(np.concatenate([y_lin, y_geo]))
    y0h, sh, _ = track_droplets(
        body,
        d_m,
        y0,
        U=U,
        t_air=t_air,
        gravity=gravity,
        conv_factor=conv_factor,
        g_tol=g_tol,
    )
    K = stokes_number(d_m, U, body.diameter, t_air)
    Re_d = droplet_reynolds(d_m, U, t_air)
    if y0h.size < 3:
        # inertia-free limit: no impingement at all
        s_tab = np.linspace(0.0, 0.02 * h, n_table)
        return ImpingementResult(
            s=s_tab,
            beta=np.zeros_like(s_tab),
            y0=np.zeros(1),
            E_total=0.0,
            s_limit=0.0,
            K=K,
            Re_d=Re_d,
            d_m=d_m,
            U=U,
            body_height=h,
            n_traj=n_traj,
        )

    # refine the maximum impacting launch ordinate (defines E = y0_max / h)
    lo = float(np.max(y0h))
    above = y0[y0 > lo]
    hi = float(above.min()) if above.size else float(y0_max_factor * h)
    for _ in range(32):
        mid = 0.5 * (lo + hi)
        _, _, hm = track_droplets(
            body,
            d_m,
            np.array([mid]),
            U=U,
            t_air=t_air,
            gravity=gravity,
            conv_factor=conv_factor,
            g_tol=g_tol,
        )
        if hm[0]:
            lo = mid
        else:
            hi = mid
    y0_max = lo

    order = np.argsort(sh)
    s_sorted = sh[order]
    y0_sorted = y0h[order]
    s_u, inv = np.unique(np.round(s_sorted, 12), return_inverse=True)
    y0_u = np.bincount(inv, weights=y0_sorted) / np.bincount(inv)

    # beta(s) = -dy0/ds: differentiate the (s, y0) mapping on a common grid
    s_grid = np.linspace(float(s_u.min()), float(s_u.max()), n_table)
    y0_grid = np.interp(s_grid, s_u, y0_u)
    # beta = dy0/ds (mass conservation between adjacent launch streamtubes);
    # enforce monotone y0(s) before differentiating so round-off cannot produce
    # a negative collection efficiency
    y0_grid = np.maximum.accumulate(y0_grid)
    beta = np.clip(np.gradient(y0_grid, s_grid), 0.0, None)
    if beta.size > 7:  # light smoothing keeps beta differentiable
        k = np.ones(5) / 5.0
        beta = np.convolve(np.pad(beta, 2, mode="edge"), k, mode="valid")

    return ImpingementResult(
        s=s_grid,
        beta=beta,
        y0=y0_grid,
        E_total=float(y0_max / h),
        s_limit=float(s_u.max()),
        K=K,
        Re_d=Re_d,
        d_m=d_m,
        U=U,
        body_height=h,
        n_traj=n_traj,
    )


def stagnation_streamline_standoff(
    d_m: float, U: float, R: float, t_air: float = -15.0, n: int = 2001
) -> float:
    """Analytic (Stokes, linearised) stagnation-streamline stand-off distance.

    Near the front stagnation point the radial cylinder flow is
    ``u_r = 2 U xi / R``; with Stokes drag the droplet equation is linear and the
    motion of a droplet released far upstream is analytic.  We integrate the
    *analytic* solution and report the largest initial stand-off from the wall
    for which the droplet still reaches the surface.  This is an independent
    (closed-form, non-RK4) benchmark for the trajectory kernel.
    """
    tau = particle_relaxation_time(d_m, t_air)
    L = 8.0 * R
    k = 2.0 * U / R  # velocity gradient at the stagnation point
    # xi'' + xi'/tau - k xi/tau = 0
    disc = np.sqrt(1.0 + 4.0 * k * tau)
    lam_p = (-1.0 + disc) / (2.0 * tau)
    lam_m = (-1.0 - disc) / (2.0 * tau)
    # droplet starts at xi0 = L with v = U (initial condition at the wall? no :
    # parameterise by starting stand-off, integrate the analytic solution and
    # test whether xi(t) reaches 0 for t > 0)
    best = 0.0
    tgrid = np.linspace(0.0, 8.0 * tau, n)
    for xi0 in L * np.linspace(1e-3, 1.0, n):
        # xi(0) = xi0, v(0) = -U  (approaching the wall)
        c_p = (xi0 * lam_m + U) / (lam_m - lam_p)
        c_m = xi0 - c_p
        xi = c_p * np.exp(lam_p * tgrid) + c_m * np.exp(lam_m * tgrid)
        if np.any(xi <= 0.0):
            best = max(best, xi0)
    return best