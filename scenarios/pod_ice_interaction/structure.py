"""Strut / hull-root structural model of a podded propulsor.

The azimuthing pod is idealised as a vertical faired strut cantilevered at the
hull root, carrying the propeller thrust and the ice contact load at the pod
body (depth ``H`` below the root). Two independent structural levels are used:

1. a three-dimensional frame finite element model (axial bar + bi-axial
   Euler-Bernoulli bending, 10 elements) that delivers the root section forces
   from the applied pod loads -- this is the *FEM* load path;
2. closed-form hollow-box section stress recovery at the root (axial + bi-axial
   bending + transverse shear) giving the von Mises stress used as the strength
   constraint of the thrust attainable domain.

The FEM root moment is reconciled against the analytic cantilever value
(``M = F H + M_applied``) inside :func:`beam_root_forces` and in the test suite,
so the section stress is never taken from an unverified load path. The lateral
tip stiffness ``3EI/H^3`` is exported for the pod vibration model used by the
propeller-ice interaction (structural natural frequency).
"""
from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, degrees, pi, radians, sin, sqrt

import numpy as np


@dataclass(frozen=True)
class StrutSection:
    """Hollow faired-box strut section (chord in the thrust direction)."""

    width: float = 1.40        # b [m], along the thrust (x) direction
    thickness: float = 0.60    # t [m], transverse (y) direction
    wall: float = 0.08         # s [m]
    height: float = 5.00       # H [m], hull root to pod centre
    young_modulus: float = 2.06e11   # E [Pa], ice-class hull steel
    poisson: float = 0.30
    yield_stress: float = 355.0e6    # sigma_y [Pa]
    allowable_factor: float = 0.60   # DNV limit-state factor for ice extremes
    stress_concentration: float = 1.15  # fillet SCF at the strut root

    def __post_init__(self) -> None:
        if not (self.width > self.thickness > 2.0 * self.wall > 0.0):
            raise ValueError("require width > thickness > 2*wall > 0")
        if self.height <= 0.0:
            raise ValueError("height must be positive")

    # ------------------------------------------------------------ properties
    @property
    def allowable_stress(self) -> float:
        return self.allowable_factor * self.yield_stress

    @property
    def area(self) -> float:
        b, t, s = self.width, self.thickness, self.wall
        return b * t - (b - 2.0 * s) * (t - 2.0 * s)

    @property
    def second_moment_x(self) -> float:
        """I about the x axis (resists lateral y bending)."""
        b, t, s = self.width, self.thickness, self.wall
        return (b * t ** 3 - (b - 2.0 * s) * (t - 2.0 * s) ** 3) / 12.0

    @property
    def second_moment_y(self) -> float:
        """I about the y axis (resists thrust-direction x bending)."""
        b, t, s = self.width, self.thickness, self.wall
        return (t * b ** 3 - (t - 2.0 * s) * (b - 2.0 * s) ** 3) / 12.0

    @property
    def lateral_stiffness(self) -> float:
        """Cantilever tip stiffness [N/m] in the thrust direction."""
        return 3.0 * self.young_modulus * self.second_moment_y / self.height ** 3

    def section_modulus(self, axis: str) -> float:
        if axis == "x":
            return self.second_moment_x / (0.5 * self.thickness)
        if axis == "y":
            return self.second_moment_y / (0.5 * self.width)
        raise ValueError("axis must be 'x' or 'y'")


def beam_root_forces(
    section: StrutSection,
    *,
    fx: float,
    fy: float,
    fz: float = 0.0,
    mx: float = 0.0,
    my: float = 0.0,
    n_elements: int = 10,
) -> dict[str, float]:
    """Root section forces from a 3-D frame FEM of the cantilevered strut.

    ``fx``/``fy`` are the lateral pod loads at the pod centre, ``fz`` the axial
    (vertical) load, ``mx``/``my`` concentrated moments. Returns the FEM root
    bending moments together with the analytic cantilever values and the
    relative reconciliation error (a load-path audit quantity).
    """
    if n_elements < 2:
        raise ValueError("at least two elements are needed")
    n_nodes = n_elements + 1
    ndof = 5 * n_nodes  # u, v, w, theta_x, theta_y
    h = section.height
    length = h / n_elements
    ei_x = section.young_modulus * section.second_moment_x
    ei_y = section.young_modulus * section.second_moment_y
    ea = section.young_modulus * section.area

    kb = np.array(
        [
            [12.0, 6.0 * length, -12.0, 6.0 * length],
            [6.0 * length, 4.0 * length ** 2, -6.0 * length, 2.0 * length ** 2],
            [-12.0, -6.0 * length, 12.0, -6.0 * length],
            [6.0 * length, 2.0 * length ** 2, -6.0 * length, 4.0 * length ** 2],
        ]
    ) / length ** 3
    ka = ea / length * np.array([[1.0, -1.0], [-1.0, 1.0]])

    def element_stiffness() -> np.ndarray:
        """Element matrix in node-major ordering (u, v, w, tx, ty) x 2 nodes."""
        ke = np.zeros((10, 10))
        # x-z plane bending: displacements (u) with rotations (theta_y), E I_y
        dofs = [0, 4, 5, 9]
        ke[np.ix_(dofs, dofs)] += ei_y * kb
        # y-z plane bending: displacements (v) with rotations (theta_x), E I_x
        dofs = [1, 3, 6, 8]
        ke[np.ix_(dofs, dofs)] += ei_x * kb
        # axial bar on w
        dofs = [2, 7]
        ke[np.ix_(dofs, dofs)] += ka
        return ke

    ke0 = element_stiffness()
    k = np.zeros((ndof, ndof))
    for e in range(n_elements):
        dofs = list(range(5 * e, 5 * e + 10))
        k[np.ix_(dofs, dofs)] += ke0

    f = np.zeros(ndof)
    tip = n_nodes - 1
    f[5 * tip + 0] = fx
    f[5 * tip + 1] = fy
    f[5 * tip + 2] = fz
    f[5 * tip + 3] = mx
    f[5 * tip + 4] = my

    # clamped root: all five DOF of node 0 fixed
    fixed = np.arange(5)
    free = np.setdiff1d(np.arange(ndof), fixed)
    u = np.zeros(ndof)
    u[free] = np.linalg.solve(k[np.ix_(free, free)], f[free])

    fm = ke0 @ u[0:10]
    my_root = abs(fm[4])   # moment about y at the root node (x-z plane)
    mx_root = abs(fm[3])   # moment about x at the root node (y-z plane)
    n_root = abs(fm[2])    # axial force at the root node
    vy_root = abs(fy)
    vx_root = abs(fx)

    my_analytic = abs(fx * h + my)
    mx_analytic = abs(fy * h + mx)
    err = max(
        abs(my_root - my_analytic) / max(my_analytic, 1e-9),
        abs(mx_root - mx_analytic) / max(mx_analytic, 1e-9),
    )
    return {
        "mx": mx_root,
        "my": my_root,
        "n": n_root,
        "vx": vx_root,
        "vy": vy_root,
        "mx_analytic": mx_analytic,
        "my_analytic": my_analytic,
        "reconciliation_error": float(err),
        "tip_deflection_x": float(u[5 * tip + 0]),
        "tip_deflection_y": float(u[5 * tip + 1]),
    }


def section_stress(
    section: StrutSection,
    *,
    axial: float,
    mx: float,
    my: float,
    vx: float,
    vy: float,
) -> dict[str, float]:
    """von Mises stress at the four root-section corners.

    Bending about ``x`` produces stress varying with ``y`` (extreme fibre
    ``t/2``); bending about ``y`` produces stress varying with ``x`` (extreme
    fibre ``b/2``). The maximum combines both at a corner and is amplified by
    the fillet stress concentration factor.
    """
    sigma_axial = axial / section.area
    sigma_mx = mx / section.section_modulus("x")
    sigma_my = my / section.section_modulus("y")
    # peak bending corner stress (both bending actions additive at a corner)
    sigma_bend = abs(sigma_mx) + abs(sigma_my)
    sigma = abs(sigma_axial) + sigma_bend
    shear = 1.5 * sqrt(vx * vx + vy * vy) / section.area
    vm = section.stress_concentration * sqrt(sigma * sigma + 3.0 * shear * shear)
    return {
        "sigma_axial": sigma_axial,
        "sigma_bending_x": sigma_mx,
        "sigma_bending_y": sigma_my,
        "tau_shear": shear,
        "sigma_vm": vm,
        "utilization": vm / section.allowable_stress,
    }


def pod_root_response(
    section: StrutSection,
    *,
    thrust: float,
    azimuth_deg: float,
    ice_force: float = 0.0,
    ice_direction_deg: float | None = None,
    n_elements: int = 10,
) -> dict[str, float]:
    """Root stress of the pod for a thrust magnitude and azimuth angle.

    ``thrust`` acts at the pod centre along the thrust vector; the transverse
    (side-force) component ``thrust * sin(azimuth)`` plus the ice load acts
    laterally. ``ice_direction_deg`` defaults to the direction perpendicular to
    the thrust vector, which is the worst combining direction for the strut
    root (axial bending and side bending then peak at the same corner).
    """
    beta = radians(azimuth_deg)
    fx = thrust * cos(beta)
    fy = thrust * sin(beta)
    if ice_force:
        if ice_direction_deg is None:
            gamma = beta + 0.5 * pi
        else:
            gamma = radians(ice_direction_deg)
        fx += ice_force * cos(gamma)
        fy += ice_force * sin(gamma)
    root = beam_root_forces(section, fx=fx, fy=fy, n_elements=n_elements)
    stress = section_stress(
        section,
        axial=root["n"],
        mx=root["mx"],
        my=root["my"],
        vx=root["vx"],
        vy=root["vy"],
    )
    out = {f"root_{k}": v for k, v in root.items()}
    out.update(stress)
    out["thrust_component_x"] = fx - (0.0 if not ice_force else ice_force * cos(beta + 0.5 * pi))
    out["side_force_y"] = thrust * sin(beta)
    out["azimuth_deg"] = azimuth_deg
    return out


def max_thrust_for_strength(
    section: StrutSection,
    *,
    azimuth_deg: float,
    ice_force: float,
    thrust_bounds: tuple[float, float] = (0.0, 8.0e6),
    tol: float = 1.0,
) -> float:
    """Largest thrust magnitude whose root ``sigma_vm`` stays within the allowable.

    Monotone bisection: the root stress grows monotonically with the thrust
    magnitude for a fixed ice load, so the boundary is unique.
    """
    lo, hi = thrust_bounds
    util_hi = pod_root_response(
        section, thrust=hi, azimuth_deg=azimuth_deg, ice_force=ice_force
    )["utilization"]
    if util_hi < 1.0:
        return hi
    if pod_root_response(
        section, thrust=lo, azimuth_deg=azimuth_deg, ice_force=ice_force
    )["utilization"] > 1.0:
        return 0.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if pod_root_response(
            section, thrust=mid, azimuth_deg=azimuth_deg, ice_force=ice_force
        )["utilization"] < 1.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)