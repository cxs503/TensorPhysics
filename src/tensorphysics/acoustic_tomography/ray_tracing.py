"""Two-dimensional acoustic ray tracing (Snell's law) under an ice canopy.

The tracer integrates the eikonal ray equations for a stationary sound-speed
field ``c(r, z)`` in a vertical plane, parametrised by travel time ``t``

    dr/dt = c cos(theta)
    dz/dt = c sin(theta)
    dtheta/dt = sin(theta) * dc/dr - cos(theta) * dc/dz

(the classical form for an isotropic medium, obtained from the eikonal
Hamiltonian ``H = c (p_r^2 + p_z^2)^{1/2}``, with ``theta`` measured from the
horizontal).  ``r`` is horizontal range and ``z`` is depth, positive downwards,
so the ice underside sits at ``z = 0``.

In the Arctic the sound speed increases with depth (cold, fresh surface water
over a warmer, saltier halocline), so a ray launched downward turns back
upward within tens of metres and *no* purely refracted ray connects two points
with a large depth offset at long range.  What connects the array are
ice-underside reflections: the classic Arctic surface duct.  The tracer
therefore supports specular reflection at the ice boundary (``z_top``): the
position is mirrored, ``z -> 2 z_top - z``, and the angle is flipped,
``theta -> -theta``, which leaves the range coordinate and the travel time
untouched.  Multipath arrivals are then found by
:func:`find_eigenrays_multipath`.

Contents
--------
``SoundSpeedField``            callable field + central-difference gradient
``grid_sound_speed_field``     bilinear field from a coarse node grid
``trace_rays``                 vectorised RK4 ray bundle, endpoint at r = r_goal
``find_eigenray``              shooting/bisection eigenray between two points
``linear_profile_arc``         closed-form range/time for a linear c(z) profile
``straight_leg_time``          travel time along a straight segment (Fermat use)
``specular_reflection_time``   Fermat-minimised ice-underside reflection time

Verification of the tracer is done against ``linear_profile_arc`` (closed form
for ``c(z) = c0 + g z``) in ``tests/test_acoustic_tomography.py``; the tracer
reproduces the analytic range and time to better than 1e-6 relative.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import numpy as np


# ---------------------------------------------------------------------------
# Fields
# ---------------------------------------------------------------------------
class SoundSpeedField:
    """Sound-speed field ``c(r, z) [m/s]`` with a central-difference gradient."""

    def __init__(self, speed, dr: float = 25.0, dz: float = 5.0, gradient=None, evaluate=None):
        if not callable(speed):
            raise TypeError("speed must be callable: speed(r, z) -> c [m/s]")
        if gradient is not None and not callable(gradient):
            raise TypeError("gradient must be callable: gradient(r, z) -> (dc/dr, dc/dz)")
        if evaluate is not None and not callable(evaluate):
            raise TypeError("evaluate must be callable: evaluate(r, z) -> (c, dc/dr, dc/dz)")
        self._speed = speed
        self._gradient = gradient
        self._evaluate = evaluate
        self.dr = float(dr)
        self.dz = float(dz)

    def __call__(self, r, z):
        return np.asarray(self._speed(r, z), dtype=float)

    def evaluate(self, r, z):
        """Return ``(c, dc/dr, dc/dz)`` in one shot (avoids duplicated work)."""
        if self._evaluate is not None:
            c, g_r, g_z = self._evaluate(r, z)
            return np.asarray(c, dtype=float), np.asarray(g_r, dtype=float), np.asarray(g_z, dtype=float)
        g_r, g_z = self.gradient(r, z)
        return self(r, z), g_r, g_z

    def gradient(self, r, z):
        """Return ``(dc/dr, dc/dz)``: analytic if supplied, else central differences."""
        if self._gradient is not None:
            g_r, g_z = self._gradient(r, z)
            return np.asarray(g_r, dtype=float), np.asarray(g_z, dtype=float)
        g_r = (self(r + self.dr, z) - self(r - self.dr, z)) / (2.0 * self.dr)
        g_z = (self(r, z + self.dz) - self(r, z - self.dz)) / (2.0 * self.dz)
        return g_r, g_z


def bilinear(values, r_nodes, z_nodes, r, z):
    """Bilinear interpolation of ``values[nr, nz]`` at ``(r, z)``."""
    r_nodes = np.asarray(r_nodes, dtype=float)
    z_nodes = np.asarray(z_nodes, dtype=float)
    values = np.asarray(values, dtype=float)
    r = np.clip(np.asarray(r, dtype=float), r_nodes[0], r_nodes[-1])
    z = np.clip(np.asarray(z, dtype=float), z_nodes[0], z_nodes[-1])
    ir = np.clip(np.searchsorted(r_nodes, r, side="right") - 1, 0, r_nodes.size - 2)
    iz = np.clip(np.searchsorted(z_nodes, z, side="right") - 1, 0, z_nodes.size - 2)
    tr = (r - r_nodes[ir]) / (r_nodes[ir + 1] - r_nodes[ir])
    tz = (z - z_nodes[iz]) / (z_nodes[iz + 1] - z_nodes[iz])
    v00 = values[ir, iz]
    v10 = values[ir + 1, iz]
    v01 = values[ir, iz + 1]
    v11 = values[ir + 1, iz + 1]
    return (
        v00 * (1 - tr) * (1 - tz)
        + v10 * tr * (1 - tz)
        + v01 * (1 - tr) * tz
        + v11 * tr * tz
    )


def bilinear_gradient(values, r_nodes, z_nodes, r, z):
    """Analytic ``(dc/dr, dc/dz)`` of the bilinear interpolant on a node grid."""
    r_nodes = np.asarray(r_nodes, dtype=float)
    z_nodes = np.asarray(z_nodes, dtype=float)
    values = np.asarray(values, dtype=float)
    r = np.clip(np.asarray(r, dtype=float), r_nodes[0], r_nodes[-1])
    z = np.clip(np.asarray(z, dtype=float), z_nodes[0], z_nodes[-1])
    ir = np.clip(np.searchsorted(r_nodes, r, side="right") - 1, 0, r_nodes.size - 2)
    iz = np.clip(np.searchsorted(z_nodes, z, side="right") - 1, 0, z_nodes.size - 2)
    tr = (r - r_nodes[ir]) / (r_nodes[ir + 1] - r_nodes[ir])
    tz = (z - z_nodes[iz]) / (z_nodes[iz + 1] - z_nodes[iz])
    v00 = values[ir, iz]
    v10 = values[ir + 1, iz]
    v01 = values[ir, iz + 1]
    v11 = values[ir + 1, iz + 1]
    dz_nodes = z_nodes[iz + 1] - z_nodes[iz]
    dr_nodes = r_nodes[ir + 1] - r_nodes[ir]
    g_r = ((v10 - v00) * (1 - tz) + (v11 - v01) * tz) / dr_nodes
    g_z = ((v01 - v00) * (1 - tr) + (v11 - v10) * tr) / dz_nodes
    return g_r, g_z


def grid_sound_speed_field(node_values, r_nodes, z_nodes, dr=25.0, dz=5.0):
    """Build a :class:`SoundSpeedField` by bilinear interpolation of a node grid."""
    r_nodes = np.asarray(r_nodes, dtype=float)
    z_nodes = np.asarray(z_nodes, dtype=float)
    values = np.asarray(node_values, dtype=float)

    def speed(r, z):
        return bilinear(values, r_nodes, z_nodes, r, z)

    def gradient(r, z):
        return bilinear_gradient(values, r_nodes, z_nodes, r, z)

    def evaluate(r, z):
        return (
            bilinear(values, r_nodes, z_nodes, r, z),
            *bilinear_gradient(values, r_nodes, z_nodes, r, z),
        )

    return SoundSpeedField(speed, dr=dr, dz=dz, gradient=gradient, evaluate=evaluate)


# ---------------------------------------------------------------------------
# Ray integration
# ---------------------------------------------------------------------------
@dataclass
class RayBundle:
    """Result of a vectorised ray integration."""

    r: np.ndarray
    z: np.ndarray
    theta: np.ndarray
    time: np.ndarray
    valid: np.ndarray
    converged: np.ndarray = dc_field(default=None)  # placeholder, filled below

    def __post_init__(self):
        if self.converged is None:
            self.converged = np.zeros_like(self.valid)


def _rk4_step(fld: SoundSpeedField, r, z, th, dt):
    """One classical RK4 step of the ray equations (arrays in, arrays out)."""

    def rhs(rr, zz, tt):
        c, g_r, g_z = fld.evaluate(rr, zz)
        return c * np.cos(tt), c * np.sin(tt), np.sin(tt) * g_r - np.cos(tt) * g_z

    k1 = rhs(r, z, th)
    dt2 = 0.5 * dt
    k2 = rhs(r + dt2 * k1[0], z + dt2 * k1[1], th + dt2 * k1[2])
    k3 = rhs(r + dt2 * k2[0], z + dt2 * k2[1], th + dt2 * k2[2])
    k4 = rhs(r + dt * k3[0], z + dt * k3[1], th + dt * k3[2])
    dr = dt / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
    dz = dt / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    dth = dt / 6.0 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
    return r + dr, z + dz, th + dth


def trace_rays(
    fld: SoundSpeedField,
    r0: float,
    z0: float,
    theta0,
    r_goal: float,
    step_len: float = 25.0,
    n_steps: int = 800,
    z_min: float = 1.0,
    z_max: float = 1.0e4,
    keep_path: bool = False,
    z_top: float | None = None,
    max_reflections: int = 4,
):
    """Integrate a bundle of rays and report the state at ``r = r_goal``.

    Parameters
    ----------
    fld : SoundSpeedField
    r0, z0 : float or array_like
        Source position(s) [m], broadcast against ``theta0`` (an array of
        sources lets a whole source/receiver ensemble be traced at once).
    theta0 : array_like
        Launch angles [rad] measured from the horizontal (positive downwards).
    r_goal : float or array_like
        Range at which the ray state is sampled (broadcast against ``theta0``).
    step_len : float
        Arc-length step [m].  RK4 is fourth order: the dominant error scales
        as ``(step_len * dc/dz / c)^4`` and is verified against the closed form
        for a linear profile in the test suite.
    z_min, z_max : float
        Vertical domain [m].  Rays leaving the domain are marked invalid.  When
        ``z_top`` is given the lower limit is set by the ice lid (rays reflect
        there), so ``z_min`` is ignored.

    Returns
    -------
    dict with ``theta``, ``z`` (at ``r_goal``), ``time``, ``valid``,
    ``converged`` and, if ``keep_path``, the sampled ``r_path``/``z_path``.
    """
    theta0 = np.atleast_1d(np.asarray(theta0, dtype=float))
    shape = theta0.shape
    n = theta0.size
    th = theta0.reshape(-1).copy()

    def _flat(a):
        a = np.asarray(a, dtype=float)
        if a.size == n:
            return a.reshape(-1).astype(float)
        return np.broadcast_to(a, shape).reshape(-1).astype(float)

    r = _flat(r0)
    z = _flat(z0)
    r_start = r.copy()
    r_goal = _flat(r_goal)
    t = np.zeros(n)
    alive = np.ones(n, dtype=bool)
    hit = np.zeros(n, dtype=bool)
    z_end = np.full(n, np.nan)
    t_end = np.full(n, np.nan)
    th_end = np.full(n, np.nan)

    if keep_path:
        r_path = np.full((n, n_steps + 1), np.nan)
        z_path = np.full((n, n_steps + 1), np.nan)
        r_path[:, 0] = r
        z_path[:, 0] = z
        t_path = np.zeros((n, n_steps + 1))
        polylines = [[(float(r[i]), float(z[i]))] for i in range(n)]

    # with an ice lid the domain floor is set by the lid, not by z_min
    z_floor = z_min if z_top is None else -np.inf
    n_reflect = np.zeros(n, dtype=int)
    reflect = np.zeros(n, dtype=bool)
    direction = np.where(r_goal >= r_start, 1.0, -1.0)
    u_goal = direction * r_goal
    u = direction * r
    for k in range(n_steps):
        if not np.any(alive):
            break
        c_now = fld(r, z)
        dt = step_len / np.maximum(c_now, 1.0)
        r_new, z_new, th_new = _rk4_step(fld, r, z, th, dt)
        u_new = direction * r_new
        t_new = t + dt
        if z_top is not None:
            # specular reflection at the ice underside (mirror in z, flip theta)
            hit_ice = alive & (z_new < z_top)
            if np.any(hit_ice):
                if keep_path:
                    z_raw = z_new
                    f_cross = np.where(hit_ice, (z_top - z) / (z_raw - z), 0.0)
                    r_cross = r + np.clip(f_cross, 0.0, 1.0) * (r_new - r)
                    for i in np.flatnonzero(hit_ice):
                        polylines[i].append((float(r_cross[i]), float(z_top)))
                z_new = np.where(hit_ice, 2.0 * z_top - z_new, z_new)
                th_new = np.where(hit_ice, -th_new, th_new)
                n_reflect = n_reflect + hit_ice.astype(int)
                reflect = reflect | hit_ice
                too_deep = hit_ice & ((z_new < z_top) | (n_reflect > max_reflections))
                alive = alive & ~too_deep
        crossed = alive & (u_new >= u_goal)
        if np.any(crossed):
            # Secant refinement of the fractional step so the endpoint is
            # reached at fourth order (a plain chord interpolation would be
            # first order and dominate the travel-time error).
            with np.errstate(divide="ignore", invalid="ignore"):
                frac = np.where(crossed, np.clip((u_goal - u) / (u_new - u), 0.0, 1.0), 0.0)
            r_ref = r.copy()
            z_ref = z.copy()
            th_ref = th.copy()
            for _ in range(3):
                r_ref, z_ref, th_ref = _rk4_step(fld, r, z, th, frac * dt)
                u_ref = direction * r_ref
                with np.errstate(divide="ignore", invalid="ignore"):
                    frac = np.where(
                        crossed,
                        np.clip(frac * (u_goal - u) / np.where(np.abs(u_ref - u) > 0.0, u_ref - u, np.nan), 0.0, 1.0),
                        frac,
                    )
            frac = np.where(np.isfinite(frac), frac, 0.0)
            r_ref, z_ref, th_ref = _rk4_step(fld, r, z, th, frac * dt)
            if z_top is not None:
                below = z_ref < z_top
                z_ref = np.where(below, 2.0 * z_top - z_ref, z_ref)
                th_ref = np.where(below, -th_ref, th_ref)
            z_end = np.where(crossed, z_ref, z_end)
            t_end = np.where(crossed, t + frac * dt, t_end)
            th_end = np.where(crossed, th_ref, th_end)
            hit = hit | crossed
            if keep_path:
                for i in np.flatnonzero(crossed & np.isfinite(z_ref)):
                    polylines[i].append((float(r_goal[i]), float(z_ref[i])))
        alive = alive & ~crossed
        out_of_box = (z_new < z_floor) | (z_new > z_max) | (u_new > u_goal) | ~np.isfinite(z_new)
        alive = alive & ~out_of_box
        r = np.where(alive, r_new, r)
        z = np.where(alive, z_new, z)
        th = np.where(alive, th_new, th)
        t = np.where(alive, t_new, t)
        u = direction * r
        if keep_path:
            r_path[:, k + 1] = np.where(alive, r, np.nan)
            z_path[:, k + 1] = np.where(alive, z, np.nan)
            t_path[:, k + 1] = np.where(alive, t, np.nan)
            for i in np.flatnonzero(alive):
                polylines[i].append((float(r[i]), float(z[i])))

    out = {
        "theta": theta0,
        "n_reflections": n_reflect.reshape(shape),
        "reflected": reflect.reshape(shape),
        "z": z_end.reshape(shape),
        "time": t_end.reshape(shape),
        "theta_end": th_end.reshape(shape),
        "valid": hit.reshape(shape),
        "converged": hit.reshape(shape),
    }
    if keep_path:
        out["r_path"] = r_path.reshape(shape + (n_steps + 1,))
        out["z_path"] = z_path.reshape(shape + (n_steps + 1,))
        out["t_path"] = t_path.reshape(shape + (n_steps + 1,))
        out["polylines"] = [np.asarray(pl, dtype=float) for pl in polylines]
    return out


@dataclass
class EigenRay:
    """A single converged ray connecting two points."""

    r: np.ndarray
    z: np.ndarray
    time: float
    theta0: float
    residual_z: float
    n_steps: int
    n_reflections: int = 0

    @property
    def path(self):
        return np.column_stack([self.r, self.z])


def find_eigenray(
    fld: SoundSpeedField,
    r0: float,
    z0: float,
    r1: float,
    z1: float,
    step_len: float = 25.0,
    n_scan: int = 121,
    half_window_deg: float = 20.0,
    n_bisect: int = 60,
    z_min: float = 1.0,
    z_max: float = 1.0e4,
    n_steps: int = 900,
):
    """Shooting / bisection search for the direct (refracted) eigenray.

    Scans launch angles on a window around the straight-line direction, then
    bisects the bracketing interval of ``f(theta0) = z(r1; theta0) - z1``.  The
    solution closest to the straight-line angle (the direct arrival) is kept.
    Returns ``None`` when no bracket is found, e.g. when the pair is shadowed.
    """
    straight = np.arctan2(z1 - z0, r1 - r0)
    window = np.deg2rad(half_window_deg)
    thetas = np.linspace(straight - window, straight + window, n_scan)
    scan = trace_rays(
        fld, r0, z0, thetas, r1, step_len=step_len, n_steps=n_steps, z_min=z_min, z_max=z_max
    )
    f = scan["z"] - z1
    ok = scan["valid"] & np.isfinite(f)
    brackets = []
    for i in range(n_scan - 1):
        if ok[i] and ok[i + 1] and f[i] * f[i + 1] <= 0.0:
            brackets.append(i)
    if not brackets:
        return None
    i = min(brackets, key=lambda k: abs(0.5 * (thetas[k] + thetas[k + 1]) - straight))
    a, b = thetas[i], thetas[i + 1]
    fa = f[i]
    lo, hi, flo = a, b, fa
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        res = trace_rays(
            fld, r0, z0, np.array([mid]), r1, step_len=step_len, n_steps=n_steps,
            z_min=z_min, z_max=z_max,
        )
        if not res["valid"][0]:
            # lost the ray: shrink towards the known good side
            hi = mid
            continue
        fm = res["z"][0] - z1
        if abs(fm) < 1e-9:
            lo = hi = mid
            flo = 0.0
            break
        if flo * fm <= 0.0:
            hi = mid
        else:
            lo = mid
            flo = fm
    th = 0.5 * (lo + hi)
    res = trace_rays(
        fld, r0, z0, np.array([th]), r1, step_len=step_len, n_steps=n_steps,
        z_min=z_min, z_max=z_max, keep_path=True,
    )
    if not res["valid"][0]:
        return None
    if "polylines" in res:
        pl = res["polylines"][0]
        rp, zp = pl[:, 0], pl[:, 1]
        good = np.ones(rp.size, dtype=bool)
    else:
        rp = res["r_path"][0]
        zp = res["z_path"][0]
        good = np.isfinite(rp) & np.isfinite(zp)
    return EigenRay(
        r=rp[good],
        z=zp[good],
        time=float(res["time"][0]),
        theta0=float(th),
        residual_z=float(res["z"][0] - z1),
        n_steps=int(good.sum()),
    )


# ---------------------------------------------------------------------------
# Closed-form reference for a linear profile (tracer verification)
# ---------------------------------------------------------------------------
def linear_profile_arc(c0: float, grad: float, p: float, z_a: float, z_b: float):
    """Exact horizontal range and travel time for ``c(z) = c0 + grad * z``.

    ``p = cos(theta)/c`` is the conserved horizontal slowness.  With
    ``u = p c(z)`` the ray integrals are

        t = (1/g) * [-ln((1 + sqrt(1 - u^2)) / u)]      from u(z_a) to u(z_b)
        x = (1/(p g)) * [-sqrt(1 - u^2)]                from u(z_a) to u(z_b)

    Returns ``(range_m, time_s)``.
    """
    def integral(u_a, u_b):
        def g(u):
            return -np.log((1.0 + np.sqrt(1.0 - u * u)) / u), -np.sqrt(1.0 - u * u)

        ta, xa = g(u_a)
        tb, xb = g(u_b)
        return tb - ta, xb - xa

    u_a = p * (c0 + grad * z_a)
    u_b = p * (c0 + grad * z_b)
    t, x = integral(u_a, u_b)
    return x / (p * grad), t / grad


# ---------------------------------------------------------------------------
# Straight-path travel times (Fermat-stationary constructions)
# ---------------------------------------------------------------------------
def straight_leg_time(fld: SoundSpeedField, r_a: float, z_a: float, r_b: float, z_b: float, n_seg: int = 400):
    """Travel time along a straight segment, midpoint quadrature of ``int ds/c``."""
    rr = np.linspace(r_a, r_b, n_seg + 1)
    zz = np.linspace(z_a, z_b, n_seg + 1)
    ds = np.hypot(np.diff(rr), np.diff(zz))
    c_mid = fld(0.5 * (rr[:-1] + rr[1:]), 0.5 * (zz[:-1] + zz[1:]))
    return float(np.sum(ds / c_mid))


def specular_reflection_time(
    fld: SoundSpeedField,
    r_src: float,
    z_src: float,
    r_rcv: float,
    z_rcv: float,
    z_interface: float = 0.0,
    n_golden: int = 80,
):
    """Fermat-minimised travel time of the ice-underside specular reflection.

    The reflection point ``r_b`` on the interface is found by golden-section
    minimisation of ``T(r_b) = t(src -> (r_b, z_interface)) + t((r_b,
    z_interface) -> rcv)`` with each leg's time evaluated along the straight
    segment.  Travel time is stationary with respect to path perturbations
    (Fermat), so replacing the refracted legs by straight ones perturbs the
    total time only at second order in the refraction; this is quantified
    against the exact closed form in the test suite (<1e-4 relative for a
    realistic 0.05 1/s vertical gradient).

    Returns ``(time_s, r_bounce_m)``.
    """
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    lo, hi = float(r_src), float(r_rcv)
    a = hi - phi * (hi - lo)
    b = lo + phi * (hi - lo)
    fa = _reflection_cost(fld, r_src, z_src, r_rcv, z_rcv, z_interface, a)
    fb = _reflection_cost(fld, r_src, z_src, r_rcv, z_rcv, z_interface, b)
    for _ in range(n_golden):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - phi * (hi - lo)
            fa = _reflection_cost(fld, r_src, z_src, r_rcv, z_rcv, z_interface, a)
        else:
            lo, a, fa = a, b, fb
            b = lo + phi * (hi - lo)
            fb = _reflection_cost(fld, r_src, z_src, r_rcv, z_rcv, z_interface, b)
    r_bounce = 0.5 * (lo + hi)
    return _reflection_cost(fld, r_src, z_src, r_rcv, z_rcv, z_interface, r_bounce), r_bounce


def _reflection_cost(fld, r_src, z_src, r_rcv, z_rcv, z_interface, r_b):
    return straight_leg_time(fld, r_src, z_src, r_b, z_interface) + straight_leg_time(
        fld, r_b, z_interface, r_rcv, z_rcv
    )

def find_eigenrays(
    fld: SoundSpeedField,
    r0,
    z0,
    r1,
    z1,
    step_len: float = 10.0,
    n_scan: int = 51,
    half_window_deg: float = 8.0,
    n_refine: int = 10,
    z_min: float = 1.0,
    z_max: float = 1.0e4,
    n_steps: int | None = None,
    z_tol: float = 1.0,
):
    """Vectorised eigenray search for an array of source/receiver pairs.

    The whole ensemble is traced as one bundle (the tracer accepts per-ray
    sources), which turns an O(N_pairs) Python loop over shooting iterations
    into ``O(n_refine)`` vectorised calls (Illinois regula falsi on the bracket).

    ``z_tol`` is the endpoint tolerance [m] of the arrival; pairs that do not
    converge to it are reported as ``None`` instead of being used.

    Returns
    -------
    dict with ``eigenrays`` (list of :class:`EigenRay` or ``None``),
    ``time`` [s], ``valid`` [bool], ``theta0`` [rad], ``n_steps_total``.
    """
    r0 = np.atleast_1d(np.asarray(r0, dtype=float))
    z0 = np.atleast_1d(np.asarray(z0, dtype=float))
    r1 = np.atleast_1d(np.asarray(r1, dtype=float))
    z1 = np.atleast_1d(np.asarray(z1, dtype=float))
    n_pairs = r0.size
    if n_steps is None:
        span = float(np.max(np.hypot(r1 - r0, z1 - z0)))
        n_steps = int(1.2 * span / step_len) + 50
    straight = np.arctan2(z1 - z0, r1 - r0)
    window = np.deg2rad(half_window_deg)
    offsets = np.linspace(-window, window, n_scan)
    thetas = straight[:, None] + offsets[None, :]

    scan = trace_rays(
        fld,
        np.repeat(r0, n_scan),
        np.repeat(z0, n_scan),
        thetas,
        np.repeat(r1, n_scan),
        step_len=step_len,
        n_steps=n_steps,
        z_min=z_min,
        z_max=z_max,
    )
    zz = scan["z"].reshape(n_pairs, n_scan)
    ok = scan["valid"].reshape(n_pairs, n_scan) & np.isfinite(zz)
    f = zz - z1[:, None]

    lo = np.full(n_pairs, np.nan)
    hi = np.full(n_pairs, np.nan)
    flo = np.full(n_pairs, np.nan)
    fhi = np.full(n_pairs, np.nan)
    for i in range(n_pairs):
        brackets = [
            j
            for j in range(n_scan - 1)
            if ok[i, j] and ok[i, j + 1] and f[i, j] * f[i, j + 1] <= 0.0
        ]
        if not brackets:
            continue
        j = min(brackets, key=lambda k: abs(0.5 * (thetas[i, k] + thetas[i, k + 1]) - straight[i]))
        lo[i] = thetas[i, j]
        hi[i] = thetas[i, j + 1]
        flo[i] = f[i, j]
        fhi[i] = f[i, j + 1]

    active = np.isfinite(lo)
    for _ in range(n_refine):
        if not np.any(active):
            break
        idx = np.flatnonzero(active)
        with np.errstate(divide="ignore", invalid="ignore"):
            secant = lo[idx] - flo[idx] * (hi[idx] - lo[idx]) / (fhi[idx] - flo[idx])
        bad = ~np.isfinite(secant) | (secant <= lo[idx]) | (secant >= hi[idx])
        mid = np.where(bad, 0.5 * (lo[idx] + hi[idx]), secant)
        res = trace_rays(
            fld,
            r0[idx],
            z0[idx],
            mid,
            r1[idx],
            step_len=step_len,
            n_steps=n_steps,
            z_min=z_min,
            z_max=z_max,
        )
        good = res["valid"] & np.isfinite(res["z"])
        fm = res["z"] - z1[idx]
        same_side = flo[idx] * fm <= 0.0
        hi[idx] = np.where(good & same_side, mid, hi[idx])
        fhi[idx] = np.where(good & same_side, fm, fhi[idx])
        lo[idx] = np.where(good & ~same_side, mid, lo[idx])
        flo[idx] = np.where(good & ~same_side, fm, flo[idx])
        # pairs whose root already converged
        active = active & (np.abs(hi - lo) > 1e-13)
    theta_sol = 0.5 * (lo + hi)
    valid = np.isfinite(theta_sol)
    final = trace_rays(
        fld,
        np.where(valid, r0, r0),
        z0,
        np.where(valid, theta_sol, straight),
        r1,
        step_len=step_len,
        n_steps=n_steps,
        z_min=z_min,
        z_max=z_max,
        keep_path=True,
    )
    eigenrays = []
    for i in range(n_pairs):
        if not (valid[i] and final["valid"][i]):
            eigenrays.append(None)
            continue
        if abs(final["z"][i] - z1[i]) > z_tol:
            # bracketed but not converged (e.g. a ray grazing the ice boundary):
            # rejected rather than silently used -- no fabricated arrivals.
            valid[i] = False
            eigenrays.append(None)
            continue
        rp = final["r_path"][i]
        zp = final["z_path"][i]
        good = np.isfinite(rp) & np.isfinite(zp)
        eigenrays.append(
            EigenRay(
                r=rp[good],
                z=zp[good],
                time=float(final["time"][i]),
                theta0=float(theta_sol[i]),
                residual_z=float(final["z"][i] - z1[i]),
                n_steps=int(good.sum()),
            )
        )
    return {
        "eigenrays": eigenrays,
        "time": final["time"],
        "valid": np.array([e is not None for e in eigenrays]),
        "theta0": theta_sol,
        "straight_angle": straight,
        "n_steps": n_steps,
    }


def eigenray_paths(result):
    """Extract ``[r, z]`` polylines from a :func:`find_eigenrays` result."""
    return [er.path for er in result["eigenrays"] if er is not None]


def find_eigenrays_multipath(
    fld: SoundSpeedField,
    r0,
    z0,
    r1,
    z1,
    step_len: float = 30.0,
    n_scan: int = 241,
    theta_max_deg: float = 18.0,
    n_refine: int = 16,
    z_top: float = 0.0,
    z_max: float = 200.0,
    max_reflections: int = 8,
    z_tol: float = 1.0,
    max_arrivals: int = 6,
    n_steps: int | None = None,
    time_tol: float = 1e-3,
):
    """All first-arrival-ordered eigenrays (multipath) for each pair.

    Under an ice canopy the sound speed increases with depth, so rays are
    trapped between the ice underside and their turning depth: several
    specular bounce orders reach a receiver at different times (the classic
    Arctic surface duct).  This routine scans launch angles over the full
    duct cone, brackets *every* sign change of ``z(r1; theta0) - z1``,
    refines each bracket by Illinois regula falsi, keeps the converged
    arrivals (``|dz| <= z_tol``) and returns them sorted by travel time.

    Returns
    -------
    dict with
        ``times``     list of arrays, one per pair, sorted ascending [s]
        ``eigenrays`` list of lists of :class:`EigenRay`
        ``valid``     bool array, True when at least one arrival was found
        ``n_roots``   array with the number of bracketed roots per pair
    """
    r0 = np.atleast_1d(np.asarray(r0, dtype=float))
    z0 = np.atleast_1d(np.asarray(z0, dtype=float))
    r1 = np.atleast_1d(np.asarray(r1, dtype=float))
    z1 = np.atleast_1d(np.asarray(z1, dtype=float))
    n_pairs = r0.size
    if n_steps is None:
        span = float(np.max(np.hypot(r1 - r0, z1 - z0)))
        n_steps = int(1.6 * span / step_len) + 80
    theta_max = np.deg2rad(theta_max_deg)
    thetas = np.linspace(-theta_max, theta_max, n_scan)

    scan = trace_rays(
        fld,
        np.repeat(r0, n_scan),
        np.repeat(z0, n_scan),
        np.tile(thetas, n_pairs),
        np.repeat(r1, n_scan),
        step_len=step_len,
        n_steps=n_steps,
        z_min=-1e9,
        z_max=z_max,
        z_top=z_top,
        max_reflections=max_reflections,
    )
    zz = scan["z"].reshape(n_pairs, n_scan)
    ok = scan["valid"].reshape(n_pairs, n_scan) & np.isfinite(zz)
    f = zz - z1[:, None]

    br_pair, br_lo, br_hi, br_flo, br_fhi = [], [], [], [], []
    for i in range(n_pairs):
        for j in range(n_scan - 1):
            if ok[i, j] and ok[i, j + 1] and f[i, j] * f[i, j + 1] <= 0.0:
                br_pair.append(i)
                br_lo.append(thetas[j])
                br_hi.append(thetas[j + 1])
                br_flo.append(f[i, j])
                br_fhi.append(f[i, j + 1])
    n_roots = np.bincount(np.asarray(br_pair, dtype=int), minlength=n_pairs) if br_pair else np.zeros(n_pairs, dtype=int)
    if not br_pair:
        return {
            "times": [np.array([]) for _ in range(n_pairs)],
            "eigenrays": [[] for _ in range(n_pairs)],
            "valid": np.zeros(n_pairs, dtype=bool),
            "n_roots": n_roots,
            "n_steps": n_steps,
        }

    br_pair = np.asarray(br_pair, dtype=int)
    lo = np.asarray(br_lo, dtype=float)
    hi = np.asarray(br_hi, dtype=float)
    flo = np.asarray(br_flo, dtype=float)
    fhi = np.asarray(br_fhi, dtype=float)
    active = np.ones(lo.size, dtype=bool)
    for _ in range(n_refine):
        if not np.any(active):
            break
        idx = np.flatnonzero(active)
        p = br_pair[idx]
        with np.errstate(divide="ignore", invalid="ignore"):
            secant = lo[idx] - flo[idx] * (hi[idx] - lo[idx]) / (fhi[idx] - flo[idx])
        bad = ~np.isfinite(secant) | (secant <= lo[idx]) | (secant >= hi[idx])
        mid = np.where(bad, 0.5 * (lo[idx] + hi[idx]), secant)
        res = trace_rays(
            fld,
            r0[p],
            z0[p],
            mid,
            r1[p],
            step_len=step_len,
            n_steps=n_steps,
            z_min=-1e9,
            z_max=z_max,
            z_top=z_top,
            max_reflections=max_reflections,
        )
        good = res["valid"] & np.isfinite(res["z"])
        fm = res["z"] - z1[p]
        same_side = flo[idx] * fm <= 0.0
        hi[idx] = np.where(good & same_side, mid, hi[idx])
        fhi[idx] = np.where(good & same_side, fm, fhi[idx])
        lo[idx] = np.where(good & ~same_side, mid, lo[idx])
        flo[idx] = np.where(good & ~same_side, fm, flo[idx])
        active = active & (np.abs(hi - lo) > 1e-13)

    theta_sol = 0.5 * (lo + hi)
    p = br_pair
    final = trace_rays(
        fld,
        r0[p],
        z0[p],
        theta_sol,
        r1[p],
        step_len=step_len,
        n_steps=n_steps,
        z_min=-1e9,
        z_max=z_max,
        z_top=z_top,
        max_reflections=max_reflections,
        keep_path=True,
    )
    times = [[] for _ in range(n_pairs)]
    rays = [[] for _ in range(n_pairs)]
    for k in range(theta_sol.size):
        i = int(p[k])
        if not final["valid"][k]:
            continue
        if abs(final["z"][k] - z1[i]) > z_tol:
            continue
        if "polylines" in final:
            pl = final["polylines"][k]
            rp, zp = pl[:, 0], pl[:, 1]
            good = np.ones(rp.size, dtype=bool)
        else:
            rp = final["r_path"][k]
            zp = final["z_path"][k]
            good = np.isfinite(rp) & np.isfinite(zp)
        er = EigenRay(
            r=rp[good],
            z=zp[good],
            time=float(final["time"][k]),
            theta0=float(theta_sol[k]),
            residual_z=float(final["z"][k] - z1[i]),
            n_steps=int(good.sum()),
            n_reflections=int(final["n_reflections"][k]),
        )
        # deduplicate: neighbouring brackets can converge to the same root
        if any(abs(er.time - t_prev) < time_tol for t_prev in times[i]):
            continue
        times[i].append(er.time)
        rays[i].append(er)

    out_times, out_rays = [], []
    for i in range(n_pairs):
        order = np.argsort(times[i]) if times[i] else np.array([], dtype=int)
        picked = order[:max_arrivals]
        out_times.append(np.asarray(times[i], dtype=float)[picked])
        out_rays.append([rays[i][k] for k in picked])
    return {
        "times": out_times,
        "eigenrays": out_rays,
        "valid": np.array([t.size > 0 for t in out_times]),
        "n_roots": n_roots,
        "n_steps": n_steps,
    }
