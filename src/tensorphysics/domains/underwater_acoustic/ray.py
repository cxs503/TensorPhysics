"""Ray tracing for underwater acoustic propagation.

Implements classical ray theory for range-independent environments:

1. Snell's law:  p = cos(θ(z)) / c(z)  = const (horizontal slowness)
2. Turning-point detection: a ray turns where c(z) = c_turn
   (the local sound speed equals the ray's turning speed)
3. Eigenray computation: for a given source/receiver geometry, find
   the launch angles that connect them
4. Arrival time and amplitude: geometric spreading + absorption

The ray-theoretic TL is:

    TL_ray(r) = 20 log10(r) + α r   (spherical spreading + absorption)

for a single eigenray, or the coherent sum over multiple eigenrays
for the full field.

References
----------
- Jensen, Kuperman, Porter, Schmidt (2011). "Computational Ocean
  Acoustics", 2nd ed., Springer. Ch. 3.
- Brekhovskikh & Lysanov (2003). "Fundamentals of Ocean Acoustics",
  3rd ed., Springer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .core import RHO_WATER, C_WATER, P_REF, TL


# ---------------------------------------------------------------------------
# Snell's law and ray geometry
# ---------------------------------------------------------------------------
def snell_launch_angle(
    c_source: float,
    c_turn: float,
) -> float:
    """Launch angle (grazing) for a ray with turning speed c_turn.

    From Snell's law: cos(θ) / c = const.  At the source (c = c_source),
    the ray has grazing angle θ_s.  At the turning point (c = c_turn),
    the ray is horizontal (θ = 0).  So:

        cos(θ_s) / c_source = 1 / c_turn
        θ_s = arccos(c_source / c_turn)

    Parameters
    ----------
    c_source : float
        Sound speed at the source [m/s].
    c_turn : float
        Turning-point sound speed [m/s] (must be ≥ c_source for a
        downward-turning ray, or ≤ c_source for an upward-turning ray).

    Returns
    -------
    float
        Launch grazing angle [rad].
    """
    ratio = c_source / c_turn
    ratio = np.clip(ratio, -1.0, 1.0)
    return float(np.arccos(ratio))


def turning_point_depth(
    c_profile,
    c_turn: float,
    z_min: float = 0.0,
    z_max: float = 6000.0,
    n_z: int = 2000,
) -> float:
    """Find the turning-point depth for a ray with turning speed c_turn.

    The ray turns where c(z) = c_turn.  For a general profile, this is
    found by locating the depth(s) where the profile crosses c_turn.

    Parameters
    ----------
    c_profile : callable
        Sound-speed profile function c(z) → float.
    c_turn : float
        Turning-point sound speed [m/s].
    z_min, z_max : float
        Depth search bounds [m].
    n_z : int
        Number of depth grid points.

    Returns
    -------
    float
        Turning-point depth [m].
    """
    z = np.linspace(z_min, z_max, n_z)
    c = np.asarray([c_profile(zz) for zz in z], dtype=float)

    # Find where c crosses c_turn
    crossings = np.where(np.diff(np.sign(c - c_turn)) != 0)[0]
    if len(crossings) == 0:
        # No turning point in range → ray is surface/bottom reflected
        # Return the boundary where c is closest to c_turn
        idx = np.argmin(np.abs(c - c_turn))
        return float(z[idx])

    # Return the first (shallowest) turning point
    idx = crossings[0]
    # Linear interpolation for accuracy
    z1, z2 = z[idx], z[idx + 1]
    c1, c2 = c[idx], c[idx + 1]
    return float(z1 + (c_turn - c1) * (z2 - z1) / (c2 - c1))


def ray_arrival_time(
    c_profile,
    source_depth: float,
    receiver_depth: float,
    receiver_range: float,
    launch_angle: float,
    n_steps: int = 2000,
    dz: float = 1.0,
) -> float:
    """Compute the arrival time of a ray from source to receiver.

    Uses the ray equations: the travel time is

        t = ∫ ds / c(s)

    along the ray path, where ds is the arc-length element.

    Parameters
    ----------
    c_profile : callable
        Sound-speed profile function c(z) → float.
    source_depth : float
        Source depth [m].
    receiver_depth : float
        Receiver depth [m].
    receiver_range : float
        Receiver horizontal range [m].
    launch_angle : float
        Launch grazing angle [rad].
    n_steps : int
        Maximum number of integration steps.
    dz : float
        Vertical step size [m].

    Returns
    -------
    float
        Arrival time [s].
    """
    tracer = RayTracer(c_profile=c_profile, depth=6000.0)
    traj = tracer.trace_ray(launch_angle, source_depth, n_steps, dz)

    # Find the point on the trajectory closest to the receiver
    r_traj = traj["r"]
    z_traj = traj["z"]
    t_traj = traj["t"]

    dist = np.sqrt((r_traj - receiver_range)**2 + (z_traj - receiver_depth)**2)
    idx = np.argmin(dist)
    return float(t_traj[idx])


# ---------------------------------------------------------------------------
# Ray tracing
# ---------------------------------------------------------------------------
@dataclass
class Eigenray:
    """An eigenray connecting source to receiver.

    Parameters
    ----------
    launch_angle : float
        Launch grazing angle [rad].
    arrival_time : float
        Travel time [s].
    range : float
        Horizontal range [m].
    amplitude : float
        Pressure amplitude factor (relative to unit source).
    n_turns : int
        Number of turning points (cycles).
    """

    launch_angle: float
    arrival_time: float
    range: float
    amplitude: float
    n_turns: int


@dataclass
class RayTracer:
    """Ray tracer for range-independent underwater acoustics.

    Parameters
    ----------
    c_profile : callable
        Sound-speed profile function c(z) → float.
    depth : float
        Water depth [m].
    """

    c_profile: callable
    depth: float = 5000.0

    def __post_init__(self):
        if self.depth <= 0:
            raise ValueError(f"Water depth must be positive, got {self.depth}")

    def trace_ray(
        self,
        launch_angle: float,
        source_depth: float,
        n_steps: int = 2000,
        dz: float = 1.0,
    ) -> dict:
        """Trace a single ray from the source.

        Uses the ray equations in the (r, z) plane:

            dr/ds = c(z) · cos(θ)   (horizontal)
            dz/ds = c(z) · sin(θ)   (vertical)

        with Snell's law: cos(θ)/c = const = p.

        Parameters
        ----------
        launch_angle : float
            Launch grazing angle [rad] (measured from horizontal).
        source_depth : float
            Source depth [m].
        n_steps : int
            Maximum number of integration steps.
        dz : float
            Vertical step size [m].

        Returns
        -------
        dict
            Ray trajectory: {r: array, z: array, t: array, c: array}.
        """
        # Horizontal slowness (Snell invariant)
        c_s = self.c_profile(source_depth)
        p = np.cos(launch_angle) / c_s

        r = [0.0]
        z = [source_depth]
        t = [0.0]
        c_vals = [c_s]

        z_cur = source_depth
        r_cur = 0.0
        t_cur = 0.0

        for _ in range(n_steps):
            c_cur = self.c_profile(z_cur)
            # Vertical component of slowness
            q = np.sqrt(max(1.0 / c_cur**2 - p**2, 0.0))
            # Ray direction: dz/ds = c·sin(θ) = c·q·c = c²·q
            # (using the ray equations in arc-length form)
            # Actually: sin(θ) = q·c, so dz/ds = c·sin(θ) = c·q·c = c²·q
            # and dr/ds = c·cos(θ) = c·p·c = c²·p
            #
            # With step ds = dz / (c·sin(θ)):
            dr = c_cur * p * c_cur * (dz / (c_cur * q * c_cur)) if q > 0 else 0.0
            # Simplify: dr = p/q · dz
            if q > 0:
                dr = (p / q) * dz
            else:
                dr = 0.0

            # Step in depth
            z_new = z_cur + np.sign(np.sin(launch_angle)) * dz
            if z_new < 0:
                # Surface reflection
                z_new = -z_new
                # Flip vertical direction
                launch_angle = -launch_angle
            if z_new > self.depth:
                # Bottom reflection
                z_new = 2.0 * self.depth - z_new
                launch_angle = -launch_angle

            r_cur += dr
            t_cur += dz / (c_cur * q) if q > 0 else 0.0

            r.append(r_cur)
            z.append(z_new)
            t.append(t_cur)
            c_vals.append(self.c_profile(z_new))

            z_cur = z_new

            # Stop if we've gone far enough
            if r_cur > 100000.0:  # 100 km
                break

        return {
            "r": np.array(r),
            "z": np.array(z),
            "t": np.array(t),
            "c": np.array(c_vals),
        }

    def find_eigenrays(
        self,
        source_depth: float,
        receiver_depth: float,
        receiver_range: float,
        n_angles: int = 200,
        tol: float = 0.5,
    ) -> list[Eigenray]:
        """Find all eigenrays connecting source to receiver.

        An eigenray is a ray that, after propagation, reaches the
        receiver at (r, z) = (receiver_range, receiver_depth).

        We search over launch angles and find those whose trajectory
        passes through the receiver position.

        Parameters
        ----------
        source_depth : float
            Source depth [m].
        receiver_depth : float
            Receiver depth [m].
        receiver_range : float
            Receiver horizontal range [m].
        n_angles : int
            Number of launch angles to search.
        tol : float
            Range tolerance for eigenray detection [m].

        Returns
        -------
        list[Eigenray]
            List of eigenrays.
        """
        c_s = self.c_profile(source_depth)
        c_r = self.c_profile(receiver_depth)

        # Launch angle range: from -90° to +90° (grazing)
        # For a source in the water column, the relevant angles are
        # those that can reach the receiver.
        angles = np.linspace(-np.pi / 2 + 0.01, np.pi / 2 - 0.01, n_angles)

        eigenrays = []
        for theta in angles:
            traj = self.trace_ray(theta, source_depth)
            r_traj = traj["r"]
            z_traj = traj["z"]

            # Check if the ray passes near the receiver
            # Find the closest approach
            dist = np.sqrt((r_traj - receiver_range)**2 + (z_traj - receiver_depth)**2)
            min_dist = np.min(dist)
            if min_dist < tol:
                idx = np.argmin(dist)
                eigenrays.append(Eigenray(
                    launch_angle=float(theta),
                    arrival_time=float(traj["t"][idx]),
                    range=float(r_traj[idx]),
                    amplitude=1.0 / float(r_traj[idx]) if r_traj[idx] > 0 else 1.0,
                    n_turns=0,
                ))

        return eigenrays


# ---------------------------------------------------------------------------
# Ray-theoretic TL
# ---------------------------------------------------------------------------
def eigenray_tl(
    r: float | np.ndarray,
    alpha: float = 0.0,
    p_ref: float = P_REF,
) -> float | np.ndarray:
    """Ray-theoretic transmission loss (single eigenray).

    TL = 20 log10(r) + α r

    (spherical spreading + volume absorption).

    Parameters
    ----------
    r : float or array
        Horizontal range [m].
    alpha : float
        Volume absorption coefficient [dB/km].
    p_ref : float
        Reference pressure [Pa].

    Returns
    -------
    float or array
        Transmission loss [dB re 1 μPa].
    """
    r = np.asarray(r, dtype=float)
    r = np.where(r < 1e-6, 1e-6, r)
    # Spherical spreading: TL = 20 log10(r) (r in m, relative to 1 m)
    # + absorption: α [dB/km] · r [km]
    return 20.0 * np.log10(r) + alpha * r / 1000.0


def ray_trace(
    c_profile,
    source_depth: float,
    receiver_depth: float,
    receiver_range: float,
    n_angles: int = 200,
    tol: float = 0.5,
    alpha: float = 0.0,
) -> list[Eigenray]:
    """Convenience wrapper: trace eigenrays and compute TL.

    Parameters
    ----------
    c_profile : callable
        Sound-speed profile function c(z) → float.
    source_depth : float
        Source depth [m].
    receiver_depth : float
        Receiver depth [m].
    receiver_range : float
        Receiver horizontal range [m].
    n_angles : int
        Number of launch angles to search.
    tol : float
        Range tolerance [m].
    alpha : float
        Volume absorption [dB/km].

    Returns
    -------
    list[Eigenray]
        List of eigenrays with TL computed.
    """
    tracer = RayTracer(c_profile=c_profile, depth=6000.0)
    eigenrays = tracer.find_eigenrays(
        source_depth, receiver_depth, receiver_range, n_angles, tol
    )
    for er in eigenrays:
        er.amplitude = 1.0 / er.range if er.range > 0 else 1.0
    return eigenrays
