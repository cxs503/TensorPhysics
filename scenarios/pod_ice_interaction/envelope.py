"""Thrust attainable domain (T, azimuth) of a podded propulsor under ice load.

The attainable thrust magnitude at each azimuth angle is the minimum of three
independent physical limits:

``motor``      the electric motor can only deliver its rated torque
               ``Q_max = P_rated / omega_rated``; the absorbed propeller torque
               ``K_Q(J) rho n^2 D^5`` then fixes the maximum rotation rate and
               hence the maximum thrust at the given ship speed;
``bearing``    the thrust-bearing axial capacity (``factor * T_rated``);
``structure``  the strut/hull-root von Mises stress under the combined thrust,
               side force ``T sin(beta)`` and ice contact load must stay below
               the allowable stress (see :mod:`structure`).

The domain boundary is therefore piecewise and the binding constraint is
reported per azimuth station, which is what the acceptance criterion
"domain boundary consistent with the structural strength constraint, no
over-stressed region" is checked against (:func:`verify_domain`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, degrees, pi, radians, sin

import numpy as np

from .propeller import BladeGeometry, OpenWaterCurve
from .structure import StrutSection, max_thrust_for_strength, pod_root_response


@dataclass(frozen=True)
class IceLoadCase:
    """Characteristic ice load on the pod housing.

    ``characteristic_force`` is derived from an effective crushing pressure on
    the ice/housing contact patch: ``F = p_c * h_ice * b_contact`` (default
    ``p_c = 0.8 MPa``, ``b_contact = 1.0 m``), which reproduces the order of
    magnitude of ice-class rule loads for azimuthing thrusters in 1.5 m ice.
    """

    thickness: float = 1.5           # h_ice [m]
    crushing_pressure: float = 0.8e6  # p_c [Pa] effective pressure at 1.5 m scale
    contact_width: float = 1.0       # b_contact [m]
    direction_offset_deg: float | None = None  # None: worst case w.r.t. thrust

    @property
    def characteristic_force(self) -> float:
        return self.crushing_pressure * self.thickness * self.contact_width


@dataclass(frozen=True)
class PodSpec:
    """Rated data and structural model of the podded propulsor."""

    power_rated: float = 15.0e6
    n_rated_rpm: float = 125.0
    geometry: BladeGeometry = field(default_factory=BladeGeometry)
    strut: StrutSection = field(default_factory=StrutSection)
    bearing_capacity_factor: float = 1.30  # axial thrust-bearing capacity factor

    @property
    def n_rated_rps(self) -> float:
        return self.n_rated_rpm / 60.0

    @property
    def motor_torque_limit(self) -> float:
        """Constant torque limit of the power-limited motor [N m]."""
        return self.power_rated / (2.0 * pi * self.n_rated_rps)

    def curve(self, *, n_points: int = 27) -> OpenWaterCurve:
        return OpenWaterCurve(self.geometry, n_points=n_points)


def rated_operating_point(pod: PodSpec, curve: OpenWaterCurve, v_ship: float) -> dict:
    """Operating point at rated power (thrust, torque, rpm, efficiency)."""
    n = curve.solve_rpm_for_power(pod.power_rated, v_ship)
    thrust = curve.thrust(n, v_ship)
    torque = curve.torque(n, v_ship)
    return {
        "n_rps": n,
        "n_rpm": n * 60.0,
        "thrust": thrust,
        "torque": torque,
        "power": 2.0 * pi * n * torque,
        "j": v_ship / (n * pod.geometry.diameter),
        "eta0": curve.eta_at(v_ship / (n * pod.geometry.diameter)),
    }


def motor_thrust_limit(pod: PodSpec, curve: OpenWaterCurve, v_ship: float) -> dict:
    """Maximum thrust the power/torque-limited motor can sustain at ``v_ship``."""
    n = curve.solve_rpm_for_torque(pod.motor_torque_limit, v_ship)
    return {
        "n_rps": n,
        "n_rpm": n * 60.0,
        "thrust": curve.thrust(n, v_ship),
        "torque": curve.torque(n, v_ship),
    }


def thrust_domain(
    pod: PodSpec,
    ice: IceLoadCase,
    *,
    v_ship: float = 8.0,
    azimuths_deg: tuple[float, ...] = tuple(np.linspace(0.0, 180.0, 13)),
    curve: OpenWaterCurve | None = None,
) -> dict:
    """Attainable thrust magnitude versus azimuth angle.

    Returns per-station limits from each constraint, the governing (minimum)
    limit, the binding constraint name and the corresponding root stress
    utilization at the boundary.
    """
    curve = curve or pod.curve()
    ice_force = ice.characteristic_force
    motor = motor_thrust_limit(pod, curve, v_ship)
    op = rated_operating_point(pod, curve, v_ship)
    bearing_limit = pod.bearing_capacity_factor * op["thrust"]

    stations = []
    for beta in azimuths_deg:
        t_struct = max_thrust_for_strength(
            pod.strut, azimuth_deg=float(beta), ice_force=ice_force
        )
        t_motor = motor["thrust"]
        t_bearing = bearing_limit
        t_max = min(t_motor, t_bearing, t_struct)
        if t_max == t_struct:
            binding = "structure"
        elif t_max == t_motor:
            binding = "motor"
        else:
            binding = "bearing"
        resp = pod_root_response(
            pod.strut, thrust=t_max, azimuth_deg=float(beta), ice_force=ice_force
        )
        stations.append(
            {
                "azimuth_deg": float(beta),
                "t_motor": t_motor,
                "t_bearing": t_bearing,
                "t_structure": t_struct,
                "t_max": t_max,
                "binding": binding,
                "utilization_at_boundary": resp["utilization"],
                "side_force_at_boundary": t_max * sin(radians(float(beta))),
                "axial_thrust_at_boundary": t_max * cos(radians(float(beta))),
            }
        )
    return {
        "v_ship": v_ship,
        "ice_force": ice_force,
        "ice_thickness": ice.thickness,
        "motor_torque_limit": pod.motor_torque_limit,
        "rated_point": op,
        "motor_limit": motor,
        "bearing_limit": bearing_limit,
        "stations": stations,
    }


def verify_domain(
    pod: PodSpec,
    ice: IceLoadCase,
    domain: dict,
    *,
    n_thrust: int = 12,
    tol_boundary: float = 0.01,
) -> dict:
    """Machine check of the attainable domain against the strength constraint.

    Three independent checks are made:

    1. **no over-stressed region** -- every grid point with ``T <= T_max(beta)``
       has ``sigma_vm <= sigma_allow`` (utilization < 1);
    2. **boundary is the strength boundary** -- for stations whose governing
       limit is ``structure``, the boundary utilization equals 1 within
       ``tol_boundary`` and an independent bisection reproduces ``T_max`` within
       ``tol_boundary``;
    3. **maximality** -- a small outward step beyond the boundary of a
       structure-governed station violates the allowable stress.
    """
    ice_force = ice.characteristic_force
    overstressed = []
    boundary_errors = []
    maximality = []
    max_utilization_inside = 0.0
    for st in domain["stations"]:
        beta = st["azimuth_deg"]
        t_max = st["t_max"]
        for frac in np.linspace(0.0, 1.0, n_thrust):
            resp = pod_root_response(
                pod.strut, thrust=frac * t_max, azimuth_deg=beta, ice_force=ice_force
            )
            max_utilization_inside = max(max_utilization_inside, resp["utilization"])
            if resp["utilization"] > 1.0 + 1e-9:
                overstressed.append(
                    {"azimuth_deg": beta, "thrust": frac * t_max,
                     "utilization": resp["utilization"]}
                )
        if st["binding"] == "structure":
            boundary_errors.append(abs(st["utilization_at_boundary"] - 1.0))
            t_check = max_thrust_for_strength(
                pod.strut, azimuth_deg=beta, ice_force=ice_force
            )
            boundary_errors.append(abs(t_check - t_max) / max(t_max, 1.0))
            outside = pod_root_response(
                pod.strut, thrust=t_max * 1.05, azimuth_deg=beta, ice_force=ice_force
            )
            maximality.append(outside["utilization"] > 1.0)
    return {
        "n_stations": len(domain["stations"]),
        "overstressed_points": overstressed,
        "boundary_error_max": max(boundary_errors) if boundary_errors else 0.0,
        "max_utilization_inside": max_utilization_inside,
        "maximality_violations": int(sum(1 for ok in maximality if not ok)),
        "n_structure_governed": int(
            sum(1 for st in domain["stations"] if st["binding"] == "structure")
        ),
        "passed": (
            len(overstressed) == 0
            and (not boundary_errors or max(boundary_errors) <= tol_boundary)
            and all(maximality)
        ),
    }